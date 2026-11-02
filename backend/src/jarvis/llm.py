"""LLM client — Ollama (local) + OpenAI-compatible cloud (Groq/DeepSeek) + Hybrid router + Fake.

`HybridLlm` scores each user query and routes:
- short / small-talk / function-call style → local ollama (fast, private, free)
- complex / reasoning / long-answer → cloud (Groq if configured, else DeepSeek — smarter)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal, Protocol

from loguru import logger


@dataclass
class LlmEvent:
    kind: str  # "chunk" | "done" | "tool_call"
    text: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] | None = field(default=None)
    call_id: str = ""


class Llm(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> AsyncIterator[LlmEvent]: ...


# ── Complexity scoring ────────────────────────────────────────────────

# Keywords that strongly suggest a reasoning / explanation / coding task.
# Matched as whole words, case-insensitive. German + English.
_COMPLEX_KEYWORDS = {
    "erkläre", "erklär", "erklaere", "erklären",
    "warum", "wieso", "weshalb",
    "vergleiche", "vergleich",
    "analysiere", "analyse",
    "schreibe", "code", "programmiere", "programmier", "implementiere",
    "schritt für schritt", "schritt-für-schritt",
    "beweise", "herleitung", "beweis",
    "diskutiere", "diskussion",
    "explain", "why", "because", "compare", "analyze", "write code",
    "step by step", "prove", "discuss",
    "summarize", "fasse zusammen", "zusammenfassung",
    "übersetze", "translate",
    "definiere", "definition", "bedeutung",
}

# Keywords that strongly suggest a small-talk / fast-fact query.
# These pull the score DOWN even if length is moderate.
_SIMPLE_KEYWORDS = {
    "hallo", "hi", "hey",
    "wie spät", "uhrzeit", "zeit",
    "wetter", "temperatur",
    "wie geht's", "wie geht es dir", "alles klar",
    "danke", "tschüss", "bye",
    "what time", "time is it", "weather",
    "how are you", "thanks", "bye",
    "ja", "nein", "ok", "okay",
}


def complexity_score(query: str) -> float:
    """Return a complexity score in [0, ∞). > threshold → route to cloud LLM.

    Heuristics (sum of):
    - length contribution:  len(query) / 80      (≈ 1 word ≈ 0.06)
    - +1.0 per complex keyword hit
    - -0.5 per simple keyword hit (caps at 0)
    """
    q = query.strip().lower()
    if not q:
        return 0.0

    length_score = len(q) / 80.0

    complex_hits = 0
    for kw in _COMPLEX_KEYWORDS:
        # Whole-word match for single-word keywords; substring for multi-word phrases.
        if " " in kw:
            if kw in q:
                complex_hits += 1
        else:
            if re.search(rf"\b{re.escape(kw)}\b", q):
                complex_hits += 1

    simple_hits = 0
    for kw in _SIMPLE_KEYWORDS:
        if " " in kw:
            if kw in q:
                simple_hits += 1
        else:
            if re.search(rf"\b{re.escape(kw)}\b", q):
                simple_hits += 1

    score = length_score + complex_hits - 0.5 * simple_hits
    return max(0.0, score)


# ── Key loading helper ────────────────────────────────────────────────

def _load_key(env_var: str, filename: str) -> str:
    """env var first, then `filename` in the repo root. Empty if not found."""
    import os

    val = os.getenv(env_var, "").strip()
    if val:
        return val

    # backend/src/jarvis/llm.py → parents[3] = repo root (jarvis/)
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / filename
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8").strip()
    return ""


def load_deepseek_key(env_var: str = "JARVIS_DEEPSEEK_API_KEY") -> str:
    """Look up DeepSeek key. env var first, then Deepseek_api_key.txt in
    the repo root. Returns empty string if not found."""
    return _load_key(env_var, "Deepseek_api_key.txt")


def load_groq_key(env_var: str = "JARVIS_GROQ_API_KEY") -> str:
    """Look up Groq key. env var first, then Groq_api_key.txt in
    the repo root. Returns empty string if not found."""
    return _load_key(env_var, "Groq_api_key.txt")


def select_cloud_provider(*, groq_key: str, deepseek_key: str) -> Literal["groq", "deepseek"] | None:
    """Pick which cloud backend to use: Groq preferred (much lower latency),
    DeepSeek as fallback, None if neither key is present."""
    if groq_key:
        return "groq"
    if deepseek_key:
        return "deepseek"
    return None


# ── Ollama ────────────────────────────────────────────────────────────

class RealLlm:
    """Ollama-backed LLM."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        system_prompt: str = "",
        timeout_s: float = 300.0,
        max_tokens: int = 200,
    ) -> None:
        import ollama

        # Long timeout: first-call model load on CPU takes 20-60s before the
        # first token streams. Ollama python client default (~5 s connect /
        # no overall cap) disconnects mid-load. Pass an explicit httpx timeout.
        self._client = ollama.AsyncClient(host=host, timeout=timeout_s)
        self._model = model
        self._system = system_prompt
        self._max_tokens = max_tokens

    async def chat(self, messages: list[dict[str, str]]) -> AsyncIterator[LlmEvent]:
        full_messages: list[dict[str, str]] = []
        if self._system:
            full_messages.append({"role": "system", "content": self._system})
        full_messages.extend(messages)

        full_text = ""
        async for part in await self._client.chat(
            model=self._model,
            messages=full_messages,
            stream=True,
            options={"num_predict": self._max_tokens},
        ):
            content: str = part.message.content or ""
            if content:
                full_text += content
                yield LlmEvent(kind="chunk", text=content)
        yield LlmEvent(kind="done", text=full_text)


# ── OpenAI-compatible cloud client (DeepSeek, Groq, ...) ───────────────

class OpenAiCompatLlm:
    """Client for any OpenAI-compatible /v1/chat/completions stream —
    DeepSeek, Groq, or similar."""

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str = "",
        base_url: str = "https://api.deepseek.com/v1",
        timeout_s: float = 300.0,
        max_tokens: int = 200,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAiCompatLlm requires a non-empty api_key")
        import httpx

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        self._model = model
        self._system = system_prompt
        self._max_tokens = max_tokens

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, messages: list[dict[str, str]]) -> AsyncIterator[LlmEvent]:
        full_messages: list[dict[str, str]] = []
        if self._system:
            full_messages.append({"role": "system", "content": self._system})
        full_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "stream": True,
            "max_tokens": self._max_tokens,
        }

        full_text = ""
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    full_text += content
                    yield LlmEvent(kind="chunk", text=content)
        yield LlmEvent(kind="done", text=full_text)


# ── Realtime context ──────────────────────────────────────────────────

def realtime_context(now: datetime | None = None) -> str:
    """Build a German-localized 'current time/date' system context line.

    Injected into every chat() call so the LLM can answer time/date
    queries correctly without a separate tool. Wochentags-Name auf Deutsch.
    """
    DE_WEEKDAY = ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
                  "Freitag", "Samstag", "Sonntag"]
    DE_MONTH = ["", "Januar", "Februar", "März", "April", "Mai", "Juni",
                "Juli", "August", "September", "Oktober", "November", "Dezember"]
    n = now or datetime.now().astimezone()
    weekday = DE_WEEKDAY[n.weekday()]
    month = DE_MONTH[n.month]
    return (
        f"Aktuelle lokale Zeit: {n.strftime('%H:%M')} ({weekday}, "
        f"{n.day}. {month} {n.year}). "
        # %Z (locale-named timezone) has been observed to return a
        # mis-encoded string depending on process-wide C locale state on
        # Windows, which crashes downstream UTF-8 encoding with a lone
        # surrogate. %z (numeric offset) is always plain ASCII.
        f"Zeitzone: UTC{n.strftime('%z')}."
    )


# ── Hybrid router ─────────────────────────────────────────────────────

RouteStrategy = Literal["cloud_first", "complexity"]


class HybridLlm:
    """Routes each `chat()` call to local or cloud LLM.

    Strategies:
      - "cloud_first": every turn goes to the configured cloud backend (Groq
        if present, else DeepSeek). If it fails BEFORE delivering its first
        chunk, fall back to local Ollama for that turn. If it fails after
        streaming has started, surface the error.
      - "complexity": score the prompt; cloud above threshold, local below.

    Both modes also inject a current-time system message before every
    backend call so time/date queries don't need a separate tool.
    """

    def __init__(
        self,
        local: Llm,
        cloud: Llm | None,
        threshold: float = 1.0,
        now_fn: Callable[[], datetime] | None = None,
        strategy: RouteStrategy = "cloud_first",
    ) -> None:
        self._local = local
        self._cloud = cloud
        self._threshold = threshold
        self._strategy: RouteStrategy = strategy
        # Injection point for deterministic time in tests.
        self._now: Callable[[], datetime] = now_fn or (lambda: datetime.now().astimezone())

    def _pick(self, messages: list[dict[str, str]]) -> tuple[Llm, str]:
        if self._cloud is None:
            return self._local, "local"
        if self._strategy == "cloud_first":
            return self._cloud, "cloud"
        # complexity-based
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        score = complexity_score(last_user)
        if score >= self._threshold:
            return self._cloud, "cloud"
        return self._local, "local"

    def _inject_time(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Prepend a system message with the current time/date."""
        time_msg = {"role": "system", "content": realtime_context(self._now())}
        return [time_msg, *messages]

    async def chat(self, messages: list[dict[str, str]]) -> AsyncIterator[LlmEvent]:
        msgs = self._inject_time(messages)
        backend, route = self._pick(messages)

        if backend is self._cloud:
            # Try cloud, fall back to local ONLY if it fails before any chunk
            # was yielded — otherwise the user has already seen partial output
            # and a retry would duplicate it.
            first_chunk_yielded = False
            try:
                async for ev in backend.chat(msgs):
                    if ev.kind == "chunk":
                        first_chunk_yielded = True
                    yield ev
                return
            except Exception as exc:
                if first_chunk_yielded:
                    logger.error(
                        f"cloud LLM failed mid-stream ({exc!r}); cannot fall back"
                    )
                    raise
                logger.warning(
                    f"cloud LLM failed before first chunk ({exc!r}); falling back to local"
                )

        async for ev in self._local.chat(msgs):
            yield ev


# ── Fake (tests) ──────────────────────────────────────────────────────

class FakeLlm:
    """Emits scripted chunks then a done event."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def chat(self, messages: list[dict[str, str]]) -> AsyncIterator[LlmEvent]:
        full = ""
        for c in self._chunks:
            full += c
            yield LlmEvent(kind="chunk", text=c)
        yield LlmEvent(kind="done", text=full)
