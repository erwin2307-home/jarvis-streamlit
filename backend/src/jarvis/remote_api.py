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

from fastapi import FastAPI, Header, HTTPException
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

settings = get_settings()

if not settings.remote_api_key:
    raise RuntimeError(
        "JARVIS_REMOTE_API_KEY is not set. This API is meant to be exposed "
        "to the public internet via a tunnel — refusing to start without "
        "an API key rather than starting wide open."
    )

app = FastAPI(title="Jarvis Remote Chat API", version=__version__)


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


app.state.llm = _build_llm()


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
