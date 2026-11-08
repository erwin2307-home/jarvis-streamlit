"""Minimal standalone text-chat API, meant to be tunneled to the public
internet (ngrok/cloudflared) so a Streamlit Cloud frontend can reach this
machine's LLM.

Deliberately separate from app.py's voice-pipeline FastAPI app: this one
exposes plain chat (HybridLlm, same routing/time-injection as the voice
path) plus the combined QA fast-path (qa_topics.py — BeadChip, NGS,
Blutanalyse and Pharmakogenomik: exact-match answers + BeadChip's
reagent lookup + broad topic-context injection for the LLM).
No coding_handler (auto-runs generated scripts — would be remote code
execution if exposed to the internet), no WebSocket, no mic/speaker/
wake/VAD/STT/TTS.

Run: uv run uvicorn jarvis.remote_api:app --host 0.0.0.0 --port 8800
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel

from jarvis import __version__
from jarvis.coding import progress_path_for, write_progress
from jarvis.config import get_settings
from jarvis.games import GAME_IDS, GAMES_SYSTEM_PROMPT, extract_html_game, game_prompt
from jarvis.llm import (
    HybridLlm,
    Llm,
    OpenAiCompatLlm,
    RealLlm,
    load_deepseek_key,
    load_groq_key,
    select_cloud_provider,
)
from jarvis.qa_topics import fastpath_answer, topic_context as qa_topic_context
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
# Dedicated model load for games.py's game-generation endpoints below —
# always the "large" coding tier (settings.coding_model), independent of
# app.state.llm's own routing. A full game is a much bigger ask than a
# normal chat/voice answer, hence game_max_tokens instead of
# llm_max_tokens.
app.state.game_llm = RealLlm(
    host=settings.llm_host,
    model=settings.coding_model,
    system_prompt=GAMES_SYSTEM_PROMPT,
    timeout_s=settings.llm_timeout_s,
    max_tokens=settings.game_max_tokens,
)

# Short-lived tokens so the browser never has to carry the real master
# key. The Streamlit frontend's voice widget runs entirely client-side
# (see streamlit_app.py — it has to, to keep its countdown timers alive
# across Streamlit reruns) and talks to this API directly from the
# browser, which means whatever credential it holds is visible to anyone
# who opens page source on the public Streamlit app. Streamlit mints a
# token here (server-side call, real key never leaves Streamlit's own
# Python process) and embeds only the token in the page; a leaked token
# is still usable, same as the leaked real key would be, but it expires
# and can be revoked without rotating JARVIS_REMOTE_API_KEY everywhere
# else. In-memory only — restarting this process invalidates all of them.
SESSION_TOKEN_TTL = timedelta(hours=12)
_session_tokens: dict[str, datetime] = {}
app.state.session_tokens = _session_tokens

# Single-use, short-TTL tickets for serve_game() below — a game page is
# loaded via a top-level browser navigation (window.open), which can't
# attach an X-API-Key header, so it needs *some* credential in the URL.
# The real master key or even the 12h session token would sit in browser
# history/referrers/proxy logs for far longer than needed; minted only
# from an already-authenticated request, scoped to one game, consumed on
# first use. In-memory only, same as session_tokens.
GAME_TICKET_TTL = timedelta(minutes=5)
_game_tickets: dict[str, tuple[str, datetime]] = {}
app.state.game_tickets = _game_tickets

# In-process, in-memory dispatch guard for generate_game() below — NOT
# the on-disk progress file. A file-based "is it already generating?"
# check is TOCTOU-racy (two near-simultaneous requests can both read
# "not generating" before either writes "generating") and, worse, a
# crashed background task would leave the file stuck on "generating"
# forever with nothing to ever clear it. A plain set with no `await`
# between the membership check and the insert is already atomic under
# asyncio's cooperative scheduling (no other coroutine can run in that
# gap), and it's cleared in generate's `finally`, so a crash just drops
# it — same reasoning as coding.py's coding_state dict needing no lock.
_game_in_flight: set[str] = set()
app.state.game_in_flight = _game_in_flight


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
    """Accepts the real master key OR a currently-valid session token —
    see the session_tokens comment above for why both exist."""
    if x_api_key == settings.remote_api_key:
        return
    expiry = app.state.session_tokens.get(x_api_key or "")
    if expiry is not None and expiry > datetime.now(timezone.utc):
        return
    raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


# "Cool down" lab-protocol macro (30 min cooldown -> named wash/bath timers
# -> instrument choice, each step voice-gated on "weiter") lives entirely
# client-side in the Streamlit voice widget's JS — it has to, since that's
# the only thing that stays alive across the whole multi-step sequence
# (see the session_tokens comment above). The server's only job for it is
# to not waste an LLM call (and a spoken-aloud irrelevant answer) on these
# two trigger phrases; the client independently re-checks the transcript
# text for the same patterns to actually drive the workflow.
_COOLDOWN_TRIGGER_RE = re.compile(r"\bco+l\s*down\b", re.IGNORECASE)
_CONTINUE_RE = re.compile(r"^\s*weiter\.?\s*$", re.IGNORECASE)


def is_cooldown_trigger(text: str) -> bool:
    return bool(_COOLDOWN_TRIGGER_RE.search(text.strip()))


def is_continue_query(text: str) -> bool:
    return bool(_CONTINUE_RE.search(text.strip()))


@app.get("/health")
async def health() -> dict[str, object]:
    return {"ok": True, "version": __version__}


class SessionTokenResponse(BaseModel):
    token: str
    expires_at: str


@app.post("/session-token", response_model=SessionTokenResponse)
async def session_token(x_api_key: str | None = Header(default=None)) -> SessionTokenResponse:
    """Mints a short-lived token for the browser to use instead of the real
    key — must be called with the real master key, never with another
    token (minting from a token would let a leaked token mint itself new
    tokens forever, defeating the expiry)."""
    if x_api_key != settings.remote_api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    token = secrets.token_urlsafe(24)
    expires_at = datetime.now(timezone.utc) + SESSION_TOKEN_TTL
    app.state.session_tokens[token] = expires_at
    return SessionTokenResponse(token=token, expires_at=expires_at.isoformat())


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

    if is_cooldown_trigger(req.message):
        return ChatResponse(reply="Cool-down gestartet. 30 Minuten.")

    if is_continue_query(req.message):
        return ChatResponse(reply="Weiter.")

    qa_answer = fastpath_answer(req.message)
    if qa_answer is not None:
        return ChatResponse(reply=qa_answer)

    messages = [*req.history, {"role": "user", "content": req.message}]
    context = qa_topic_context(req.message)
    if context:
        messages = [{"role": "system", "content": context}, *messages]

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
    qa_answer = fastpath_answer(transcript)
    if is_time_query(transcript):
        full = time_answer()
    elif is_timer_query(transcript):
        timer_seconds = parse_timer_seconds(transcript)
        timer_name = parse_timer_name(transcript)
        full = timer_reply(timer_seconds, timer_name)
    elif is_cooldown_trigger(transcript):
        full = "Cool-down gestartet. 30 Minuten."
    elif is_continue_query(transcript):
        full = "Weiter."
    elif qa_answer is not None:
        full = qa_answer
    else:
        messages = [{"role": "user", "content": transcript}]
        context = qa_topic_context(transcript)
        if context:
            messages = [{"role": "system", "content": context}, *messages]

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


# --- Browser-playable games (games.py) -------------------------------
# See games.py's module docstring for why this is a different, safe-to-
# expose-remotely trust model than the Python coding fast-path (which
# app.py deliberately keeps out of this remote API entirely).


def _game_html_path(game: str) -> Path:
    return settings.games_dir / f"{game}.html"


def _game_progress_path(game: str) -> Path:
    return progress_path_for(game, settings.games_dir)


def _read_progress(game: str) -> dict[str, object]:
    path = _game_progress_path(game)
    try:
        data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        return data
    except (OSError, ValueError):
        return {}


async def _generate_game(game: str) -> None:
    progress_path = _game_progress_path(game)
    write_progress(
        progress_path,
        status="generating",
        tokens=0,
        max_tokens=settings.game_max_tokens,
        model=settings.coding_model,
    )
    try:
        game_llm: Llm = app.state.game_llm
        full = ""
        async for ev in game_llm.chat([{"role": "user", "content": game_prompt(game)}]):
            if ev.kind == "chunk":
                full += ev.text
                write_progress(
                    progress_path,
                    status="generating",
                    tokens=len(full) // 4,
                    max_tokens=settings.game_max_tokens,
                    model=settings.coding_model,
                )
            elif ev.kind == "done":
                full = ev.text or full
        html = extract_html_game(full)
        if html is None:
            logger.warning(f"games: model produced no valid game for {game!r}")
            write_progress(progress_path, status="error", error="no_valid_game_in_response")
            return
        settings.games_dir.mkdir(parents=True, exist_ok=True)
        # Temp-file-then-replace, same reasoning as write_progress's own
        # atomic write — serve_game() should never be able to read a
        # half-written file.
        final_path = _game_html_path(game)
        tmp_path = final_path.with_suffix(".html.tmp")
        tmp_path.write_text(html, encoding="utf-8")
        tmp_path.replace(final_path)
        write_progress(progress_path, status="done")
        logger.info(f"games: generated {game}")
    except Exception as exc:
        # Full detail stays server-side in the log only — the progress
        # file is readable by anyone holding a (possibly leaked) session
        # token via /games/{game}/status, so it gets a fixed, generic
        # code instead of str(exc), which could otherwise reflect
        # internal paths/config/stack details back to that caller.
        logger.warning(f"games: generation failed for {game!r} (ignored): {exc!r}")
        write_progress(progress_path, status="error", error="generation_failed")
    finally:
        app.state.game_in_flight.discard(game)


@app.post("/games/{game}/generate")
async def generate_game(game: str, x_api_key: str | None = Header(default=None)) -> dict[str, str]:
    _check_key(x_api_key)
    if game not in GAME_IDS:
        raise HTTPException(status_code=404, detail=f"unknown game {game!r}")
    if game in app.state.game_in_flight:
        return {"status": "generating"}  # already in flight — don't double-dispatch
    app.state.game_in_flight.add(game)  # no `await` since the check above — atomic
    asyncio.create_task(_generate_game(game))
    return {"status": "generating"}


@app.get("/games/{game}/status")
async def game_status(game: str, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    _check_key(x_api_key)
    if game not in GAME_IDS:
        raise HTTPException(status_code=404, detail=f"unknown game {game!r}")
    progress = _read_progress(game)
    if progress.get("status") == "done" or (not progress and _game_html_path(game).exists()):
        return {"status": "done"}
    if progress:
        return progress
    return {"status": "idle"}


@app.post("/games/{game}/ticket")
async def mint_game_ticket(
    game: str, x_api_key: str | None = Header(default=None)
) -> dict[str, str]:
    """Authenticated (X-API-Key header) — mints the single-use ticket
    serve_game() below accepts instead of the real key/session token."""
    _check_key(x_api_key)
    if game not in GAME_IDS:
        raise HTTPException(status_code=404, detail=f"unknown game {game!r}")
    if not _game_html_path(game).exists():
        raise HTTPException(status_code=404, detail="game not generated yet")
    ticket = secrets.token_urlsafe(32)
    app.state.game_tickets[ticket] = (game, datetime.now(timezone.utc) + GAME_TICKET_TTL)
    return {"ticket": ticket}


@app.get("/games/{game}")
async def serve_game(game: str, ticket: str | None = None) -> HTMLResponse:
    # Served via a top-level browser navigation (window.open), which
    # can't attach an X-API-Key header — a scoped single-use ticket
    # (minted above) instead of the real key/session token, so nothing
    # long-lived or broadly-valid ends up sitting in browser history,
    # Referer headers, or proxy logs.
    entry = app.state.game_tickets.pop(ticket or "", None)
    if entry is None or entry[0] != game or entry[1] <= datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="invalid or expired ticket")
    if game not in GAME_IDS:
        raise HTTPException(status_code=404, detail=f"unknown game {game!r}")
    path = _game_html_path(game)
    if not path.exists():
        raise HTTPException(status_code=404, detail="game not generated yet")
    # Model-generated content serving itself as HTML — the extra
    # lockdown costs nothing since a canvas game needs no network/frame
    # access anyway, and caps the blast radius of anything the model
    # slipped in (sandbox still allows the script the game itself needs).
    return HTMLResponse(
        content=path.read_text(encoding="utf-8"),
        headers={
            "Content-Security-Policy": "sandbox allow-scripts",
            "Referrer-Policy": "no-referrer",
        },
    )
