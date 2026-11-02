"""Minimal standalone text-chat API, meant to be tunneled to the public
internet (ngrok/cloudflared) so a Streamlit Cloud frontend can reach this
machine's LLM.

Deliberately separate from app.py's voice-pipeline FastAPI app: this one
exposes plain chat (HybridLlm, same routing/time-injection as the voice
path) plus the BeadChip QA fast-path (qa_beadchip.py — exact-match
answers + reagent lookup + broad topic-context injection for the LLM).
No coding_handler (auto-runs generated scripts — would be remote code
execution if exposed to the internet), no WebSocket, no mic/speaker/
wake/VAD/STT/TTS.

Run: uv run uvicorn jarvis.remote_api:app --host 0.0.0.0 --port 8800
"""
from __future__ import annotations

import base64
import io
import re
from datetime import datetime
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
from jarvis.qa_beadchip import BEADCHIP_QA, beadchip_topic_context, reagent_answer
from jarvis.qa_lookup import best_match
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


_DE_WEEKDAY = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]

# A tiny local model (llama3.2:1b) can hallucinate on simple factual asks
# despite the injected time context (observed live: asked "wie spät ist
# es", got told the weather in London). Time questions are cheap to answer
# deterministically, so they bypass the LLM entirely instead of trusting
# it to read the system message correctly.
_TIME_QUERY_RE = re.compile(
    r"\b(wie spät|wieviel uhr|wie viel uhr|uhrzeit|what time|time is it)\b",
    re.IGNORECASE,
)


def is_time_query(text: str) -> bool:
    return bool(_TIME_QUERY_RE.search(text.strip()))


def time_answer(now: datetime | None = None) -> str:
    n = now or datetime.now().astimezone()
    weekday = _DE_WEEKDAY[n.weekday()]
    return f"Es ist {n.strftime('%H:%M')} Uhr ({weekday}, {n.day}.{n.month}.{n.year})."


# Voice-set timers. Actually running the countdown + playing the alarm
# happens client-side in the Streamlit widget (this API has no persistent
# per-user state and the browser tab is what's actually open) — this just
# detects the request and parses the duration deterministically, for the
# same hallucination-avoidance reason as time_answer() above.
_TIMER_TRIGGER_RE = re.compile(r"\btimer\b", re.IGNORECASE)

_DE_NUMBER_WORDS = {
    "ein": 1, "eine": 1, "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10, "elf": 11,
    "zwölf": 12, "fünfzehn": 15, "zwanzig": 20, "dreißig": 30, "vierzig": 40,
    "fünfundvierzig": 45, "fünfzig": 50, "sechzig": 60, "neunzig": 90,
}

_TIMER_DURATION_RE = re.compile(
    r"(\d+|" + "|".join(_DE_NUMBER_WORDS) + r")\s*"
    r"(stunden?|std\.?|minuten?|min\.?|sekunden?|sek\.?)",
    re.IGNORECASE,
)

def _unit_to_seconds(unit: str) -> int:
    u = unit.lower()
    if u.startswith("std") or u.startswith("stunde"):
        return 3600
    if u.startswith("min"):
        return 60
    return 1  # sek/sekunde(n)


def is_timer_query(text: str) -> bool:
    return bool(_TIMER_TRIGGER_RE.search(text.strip()))


def parse_timer_seconds(text: str) -> int | None:
    """Sum every "<number> <unit>" match found (e.g. "1 Stunde 30 Minuten"
    -> 5400). Returns None if no duration could be found at all."""
    total = 0
    found = False
    for amount_str, unit in _TIMER_DURATION_RE.findall(text):
        amount = _DE_NUMBER_WORDS.get(amount_str.lower()) or int(amount_str)
        total += amount * _unit_to_seconds(unit)
        found = True
    return total if found else None


# "nenn ihn X" / "benenn ihn X" / "bennen ihn X" (common typo/mishearing
# of "benennen") — whatever follows the naming verb (and an optional
# "ihn"/"ihr"/"es"/"den timer") becomes the timer's label. Only this
# explicit-naming-verb order is supported; a name given before the verb
# ("timer chip washing auf 5 minuten") has no reliable boundary to parse.
_TIMER_NAME_RE = re.compile(
    r"\b(?:benenn\w*|bennen\w*|nenn\w*)\b\s*"
    r"(?:ihn|ihr|es|den\s+timer|diesen\s+timer)?\s*(.+)$",
    re.IGNORECASE,
)


def parse_timer_name(text: str) -> str | None:
    m = _TIMER_NAME_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip().strip(".!?,;:")
    return name or None


def timer_reply(seconds: int | None, name: str | None = None) -> str:
    if seconds is None:
        return "Wie lange soll der Timer laufen?"
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} Stunde{'n' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} Minute{'n' if minutes != 1 else ''}")
    if secs:
        parts.append(f"{secs} Sekunde{'n' if secs != 1 else ''}")
    label = f" '{name}'" if name else ""
    return f"Timer{label} auf {' '.join(parts)} gestellt."


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []  # [{"role": "user"|"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    reply: str
    timer_seconds: int | None = None
    timer_name: str | None = None


def _check_key(x_api_key: str | None) -> None:
    if x_api_key != settings.remote_api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def beadchip_fastpath_answer(text: str) -> str | None:
    """Exact-match (curated question) or bare-reagent-abbreviation
    BeadChip answer — checked before the LLM; a hit skips model
    inference entirely, same reasoning as is_time_query/is_timer_query."""
    return best_match(text, BEADCHIP_QA) or reagent_answer(text)


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

    if is_time_query(req.message):
        return ChatResponse(reply=time_answer())

    if is_timer_query(req.message):
        secs = parse_timer_seconds(req.message)
        name = parse_timer_name(req.message)
        return ChatResponse(reply=timer_reply(secs, name), timer_seconds=secs, timer_name=name)

    beadchip_answer = beadchip_fastpath_answer(req.message)
    if beadchip_answer is not None:
        return ChatResponse(reply=beadchip_answer)

    messages = [*req.history, {"role": "user", "content": req.message}]
    topic_context = beadchip_topic_context(req.message)
    if topic_context:
        messages = [{"role": "system", "content": topic_context}, *messages]

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
    timer_seconds: int | None = None
    timer_name: str | None = None


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

    timer_seconds: int | None = None
    timer_name: str | None = None
    beadchip_answer = beadchip_fastpath_answer(transcript)
    if is_time_query(transcript):
        full = time_answer()
    elif is_timer_query(transcript):
        timer_seconds = parse_timer_seconds(transcript)
        timer_name = parse_timer_name(transcript)
        full = timer_reply(timer_seconds, timer_name)
    elif beadchip_answer is not None:
        full = beadchip_answer
    else:
        messages = [{"role": "user", "content": transcript}]
        topic_context = beadchip_topic_context(transcript)
        if topic_context:
            messages = [{"role": "system", "content": topic_context}, *messages]

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
        timer_seconds=timer_seconds,
        timer_name=timer_name,
    )
