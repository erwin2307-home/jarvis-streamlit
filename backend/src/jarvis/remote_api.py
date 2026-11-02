"""Minimal standalone text-chat API, meant to be tunneled to the public
internet (ngrok/cloudflared) so a Streamlit Cloud frontend can reach this
machine's LLM.

Deliberately separate from app.py's voice-pipeline FastAPI app: this one
exposes ONLY plain chat (HybridLlm, same routing/time-injection as the
voice path). No coding_handler (auto-runs generated scripts — would be
remote code execution if exposed to the internet), no qa_matcher, no
WebSocket, no mic/speaker/wake/VAD/STT/TTS.

Run: uv run uvicorn jarvis.remote_api:app --host 0.0.0.0 --port 8800
"""
from __future__ import annotations

import base64
import io
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from jarvis import __version__
from jarvis.config import get_settings
from jarvis.llm import (
    HybridLlm,
    Llm,
    OpenAiCompatLlm,
    RealLlm,
    load_deepseek_key,
    load_groq_key,
    select_cloud_provider,
)
from jarvis.stt import RealStt, Stt
from jarvis.tts import EdgeTts, RealTts, Tts

settings = get_settings()

if not settings.remote_api_key:
    raise RuntimeError(
        "JARVIS_REMOTE_API_KEY is not set. This API is meant to be exposed "
        "to the public internet via a tunnel — refusing to start without "
        "an API key rather than starting wide open."
    )

app = FastAPI(title="Jarvis Remote Chat API", version=__version__)

# The Streamlit Cloud frontend (its own origin) calls this API's tunnel
# URL directly from the browser — a cross-origin request, so the browser
# sends an OPTIONS preflight before the real POST. Auth is via the
# X-API-Key header (not cookies), so a wildcard origin without credentials
# is fine here; the API key is still the real gate on every request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_llm() -> Llm:
    local: Llm = RealLlm(
        host=settings.llm_host,
        model=settings.llm_model,
        system_prompt=settings.llm_system_prompt,
        timeout_s=settings.llm_timeout_s,
        max_tokens=settings.llm_max_tokens,
    )
    if not settings.use_cloud:
        return local

    groq_key = load_groq_key()
    deepseek_key = load_deepseek_key()
    provider = select_cloud_provider(groq_key=groq_key, deepseek_key=deepseek_key)

    cloud: Llm | None = None
    if provider == "groq":
        cloud = OpenAiCompatLlm(
            api_key=groq_key,
            model=settings.groq_model,
            system_prompt=settings.llm_system_prompt,
            base_url=settings.groq_base_url,
            timeout_s=settings.llm_timeout_s,
            max_tokens=settings.llm_max_tokens,
        )
    elif provider == "deepseek":
        cloud = OpenAiCompatLlm(
            api_key=deepseek_key,
            model=settings.deepseek_model,
            system_prompt=settings.llm_system_prompt,
            base_url=settings.deepseek_base_url,
            timeout_s=settings.llm_timeout_s,
            max_tokens=settings.llm_max_tokens,
        )

    return HybridLlm(
        local=local,
        cloud=cloud,
        threshold=settings.complexity_threshold,
        strategy=settings.route_strategy,  # type: ignore[arg-type]
    )


def _build_stt() -> Stt:
    return RealStt(
        model_name=settings.whisper_model,
        compute_type=settings.whisper_compute,
        language=settings.whisper_lang,
        initial_prompt=settings.whisper_initial_prompt,
    )


def _build_tts() -> Tts:
    if settings.tts_backend == "piper":
        return RealTts(
            model_path=settings.tts_model_path, length_scale=settings.tts_length_scale
        )
    return EdgeTts(voice=settings.tts_edge_voice)


app.state.llm = _build_llm()
app.state.stt = _build_stt()
app.state.tts = _build_tts()


def _decode_wav_to_pcm16(raw: bytes, target_sr: int) -> npt.NDArray[np.int16]:
    """Decode uploaded WAV bytes to mono int16 PCM at target_sr (Whisper's
    expected rate). Browsers/Streamlit's audio widget may hand us a
    different rate than configured, so resample defensively."""
    raw_data, sr = sf.read(io.BytesIO(raw), dtype="int16")
    data: npt.NDArray[np.int16] = np.asarray(raw_data, dtype=np.int16)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)
    if sr != target_sr:
        frac = Fraction(target_sr, sr).limit_denominator(1000)
        from scipy.signal import resample_poly  # type: ignore[import-untyped]

        resampled = resample_poly(data.astype(np.float32), frac.numerator, frac.denominator)
        data = np.clip(resampled, -32768, 32767).astype(np.int16)
    return data


async def _synthesize_wav(tts: Tts, text: str) -> bytes:
    chunks: list[npt.NDArray[np.float32]] = []
    async for chunk in tts.synthesize(text):
        chunks.append(chunk)
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, settings.tts_sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []  # [{"role": "user"|"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    reply: str


def _check_key(x_api_key: str | None) -> None:
    if x_api_key != settings.remote_api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/health")
async def health() -> dict[str, object]:
    return {"ok": True, "version": __version__}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest, x_api_key: str | None = Header(default=None)
) -> ChatResponse:
    _check_key(x_api_key)
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    messages = [*req.history, {"role": "user", "content": req.message}]
    llm: Llm = app.state.llm

    full = ""
    try:
        async for ev in llm.chat(messages):
            if ev.kind == "chunk":
                full += ev.text
            elif ev.kind == "done":
                full = ev.text or full
    except Exception as exc:
        logger.error(f"remote_api: chat failed: {exc!r}")
        raise HTTPException(status_code=502, detail="LLM backend failed") from exc

    return ChatResponse(reply=full)


class VoiceResponse(BaseModel):
    transcript: str
    reply: str
    audio_base64: str
    sample_rate: int


@app.post("/voice", response_model=VoiceResponse)
async def voice(
    audio: UploadFile = File(...), x_api_key: str | None = Header(default=None)
) -> VoiceResponse:
    _check_key(x_api_key)

    raw = await audio.read()
    try:
        pcm = _decode_wav_to_pcm16(raw, settings.sample_rate)
    except Exception as exc:
        logger.warning(f"remote_api: failed to decode uploaded audio: {exc!r}")
        raise HTTPException(status_code=400, detail="could not decode audio") from exc

    stt: Stt = app.state.stt
    transcript = (await stt.transcribe(pcm)).strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="no speech recognized")

    llm: Llm = app.state.llm
    full = ""
    try:
        async for ev in llm.chat([{"role": "user", "content": transcript}]):
            if ev.kind == "chunk":
                full += ev.text
            elif ev.kind == "done":
                full = ev.text or full
    except Exception as exc:
        logger.error(f"remote_api: chat failed: {exc!r}")
        raise HTTPException(status_code=502, detail="LLM backend failed") from exc

    tts: Tts = app.state.tts
    try:
        wav_bytes = await _synthesize_wav(tts, full)
    except Exception as exc:
        logger.warning(f"remote_api: TTS failed (replying text-only): {exc!r}")
        wav_bytes = b""

    return VoiceResponse(
        transcript=transcript,
        reply=full,
        audio_base64=base64.b64encode(wav_bytes).decode("ascii"),
        sample_rate=settings.tts_sample_rate,
    )
