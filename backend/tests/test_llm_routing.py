"""Tests for HybridLlm routing + complexity scoring."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator

import pytest

from jarvis.llm import (
    FakeLlm,
    HybridLlm,
    LlmEvent,
    complexity_score,
    load_deepseek_key,
    load_groq_key,
    realtime_context,
)


# ── Scoring ───────────────────────────────────────────────────────────

def test_score_empty_zero() -> None:
    assert complexity_score("") == 0.0
    assert complexity_score("   ") == 0.0


def test_score_short_simple_below_threshold() -> None:
    # "wie spät ist es" is short small-talk → ollama
    assert complexity_score("wie spät ist es") < 1.0
    assert complexity_score("hallo Jarvis") < 1.0
    assert complexity_score("danke") < 1.0


def test_score_complex_keyword_pushes_above_threshold() -> None:
    # "erkläre" is a strong reasoning indicator
    s = complexity_score("erkläre mir relativität")
    assert s >= 1.0


def test_score_long_query_above_threshold() -> None:
    # 100+ char query alone should exceed threshold via length contribution
    long_q = "Was sind die wichtigsten Unterschiede zwischen Python und Rust für ein Voice Assistant Projekt"
    assert complexity_score(long_q) >= 1.0


def test_score_simple_keyword_pulls_down() -> None:
    # A medium-length query with "danke" should score lower than the same
    # query without it
    base = "kannst du mir das mal kurz darlegen vielleicht"
    with_simple = base + " danke"
    assert complexity_score(with_simple) < complexity_score(base)


def test_score_code_keyword_complex() -> None:
    assert complexity_score("schreibe eine Python-Funktion") >= 1.0
    assert complexity_score("write code for FizzBuzz") >= 1.0


# ── HybridLlm routing ─────────────────────────────────────────────────

class _RecordingLlm:
    """Test-only LLM that records every chat() invocation."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict[str, str]]) -> AsyncIterator[LlmEvent]:
        self.calls.append(messages)
        yield LlmEvent(kind="chunk", text=self.label)
        yield LlmEvent(kind="done", text=self.label)


async def test_hybrid_routes_simple_to_local() -> None:
    local = _RecordingLlm("LOCAL")
    cloud = _RecordingLlm("CLOUD")
    h = HybridLlm(local=local, cloud=cloud, threshold=1.0, strategy="complexity")

    events = []
    async for ev in h.chat([{"role": "user", "content": "wie spät"}]):
        events.append(ev)

    assert len(local.calls) == 1
    assert len(cloud.calls) == 0
    assert events[-1].text == "LOCAL"


async def test_hybrid_routes_complex_to_cloud() -> None:
    local = _RecordingLlm("LOCAL")
    cloud = _RecordingLlm("CLOUD")
    h = HybridLlm(local=local, cloud=cloud, threshold=1.0, strategy="complexity")

    events = []
    async for ev in h.chat([{"role": "user", "content": "erkläre mir Quantenmechanik im Detail"}]):
        events.append(ev)

    assert len(cloud.calls) == 1
    assert len(local.calls) == 0
    assert events[-1].text == "CLOUD"


async def test_hybrid_no_cloud_falls_back_to_local() -> None:
    local = _RecordingLlm("LOCAL")
    h = HybridLlm(local=local, cloud=None, threshold=1.0, strategy="complexity")

    events = []
    async for ev in h.chat([{"role": "user", "content": "erkläre quantenmechanik schritt für schritt"}]):
        events.append(ev)

    assert len(local.calls) == 1
    assert events[-1].text == "LOCAL"


async def test_cloud_first_always_uses_cloud() -> None:
    local = _RecordingLlm("LOCAL")
    cloud = _RecordingLlm("CLOUD")
    h = HybridLlm(local=local, cloud=cloud, strategy="cloud_first")

    async for _ev in h.chat([{"role": "user", "content": "hi"}]):
        pass
    async for _ev in h.chat([{"role": "user", "content": "erkläre relativitätstheorie schritt für schritt"}]):
        pass

    # Both routed to cloud regardless of complexity score.
    assert len(cloud.calls) == 2
    assert len(local.calls) == 0


class _FailingLlm:
    """LLM that raises before yielding any chunk."""
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def chat(self, messages: list[dict[str, str]]) -> AsyncIterator[LlmEvent]:
        raise self._exc
        yield LlmEvent(kind="done", text="")  # unreachable; satisfies generator protocol


async def test_cloud_first_falls_back_when_cloud_fails_pre_first_chunk() -> None:
    local = _RecordingLlm("LOCAL")
    cloud = _FailingLlm(RuntimeError("network down"))
    h = HybridLlm(local=local, cloud=cloud, strategy="cloud_first")

    events = []
    async for ev in h.chat([{"role": "user", "content": "hi"}]):
        events.append(ev)

    assert len(local.calls) == 1
    assert events[-1].text == "LOCAL"


class _PartialFailLlm:
    """Yields one chunk then raises."""
    async def chat(self, messages: list[dict[str, str]]) -> AsyncIterator[LlmEvent]:
        yield LlmEvent(kind="chunk", text="partial ")
        raise RuntimeError("network died after first chunk")


async def test_cloud_first_propagates_error_after_first_chunk() -> None:
    local = _RecordingLlm("LOCAL")
    cloud = _PartialFailLlm()
    h = HybridLlm(local=local, cloud=cloud, strategy="cloud_first")

    seen: list[LlmEvent] = []
    with pytest.raises(RuntimeError, match="after first chunk"):
        async for ev in h.chat([{"role": "user", "content": "hi"}]):
            seen.append(ev)

    # We DID get the first chunk before the failure.
    assert any(ev.kind == "chunk" and ev.text == "partial " for ev in seen)
    # And local was NOT invoked (partial output already delivered).
    assert len(local.calls) == 0


async def test_hybrid_scores_last_user_message() -> None:
    """Routing key is the most recent user message, not the system prompt."""
    local = _RecordingLlm("LOCAL")
    cloud = _RecordingLlm("CLOUD")
    h = HybridLlm(local=local, cloud=cloud, threshold=1.0, strategy="complexity")

    messages = [
        {"role": "system", "content": "Du bist ein Experte für theoretische Physik."},
        {"role": "user", "content": "danke"},
    ]
    async for _ev in h.chat(messages):
        pass
    # System prompt is dense + complex but routing should use the short "danke"
    assert len(local.calls) == 1
    assert len(cloud.calls) == 0


# ── Key loading ───────────────────────────────────────────────────────

def test_load_key_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_DEEPSEEK_API_KEY", "env-key-xyz")
    assert load_deepseek_key() == "env-key-xyz"


def test_realtime_context_contains_time_and_weekday() -> None:
    # Wednesday 2026-05-14 at 18:30 local (some timezone)
    fixed = datetime(2026, 5, 14, 18, 30, tzinfo=timezone.utc)
    s = realtime_context(fixed)
    assert "18:30" in s
    assert "Donnerstag" in s
    assert "Mai" in s
    assert "2026" in s


def test_realtime_context_timezone_is_always_utf8_safe() -> None:
    # strftime('%Z') (the locale-named timezone) has been observed to return
    # a mis-encoded string depending on process-wide C locale state on
    # Windows, producing lone surrogates that crash downstream UTF-8
    # encoding. The numeric offset (%z) is always plain ASCII regardless of
    # locale, so it must be what's actually used.
    fixed = datetime(2026, 5, 14, 18, 30, tzinfo=timezone.utc)
    s = realtime_context(fixed)
    s.encode("utf-8")  # must not raise
    assert "+0000" in s


async def test_hybrid_injects_time_context() -> None:
    """Every chat() prepends a system message with the current time."""
    fixed = datetime(2026, 5, 14, 14, 25, tzinfo=timezone.utc)
    local = _RecordingLlm("LOCAL")
    h = HybridLlm(
        local=local, cloud=None, threshold=1.0,
        now_fn=lambda: fixed, strategy="complexity",
    )

    async for _ in h.chat([{"role": "user", "content": "wie spät"}]):
        pass

    assert len(local.calls) == 1
    forwarded = local.calls[0]
    assert forwarded[0]["role"] == "system"
    assert "14:25" in forwarded[0]["content"]
    # Original user message preserved as the last entry.
    assert forwarded[-1] == {"role": "user", "content": "wie spät"}


def test_load_key_empty_when_nothing_set(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("JARVIS_DEEPSEEK_API_KEY", raising=False)
    # If the parent of the repo root happens to contain Deepseek_api_key.txt
    # on this dev machine, this test would pick it up. That's a known
    # local-environment leak we accept; document it in the assertion message.
    val = load_deepseek_key()
    if val:
        pytest.skip(f"local key file present on this machine ({len(val)} chars)")
    assert val == ""


def test_load_groq_key_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_GROQ_API_KEY", "groq-env-key-xyz")
    assert load_groq_key() == "groq-env-key-xyz"


def test_load_groq_key_empty_when_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JARVIS_GROQ_API_KEY", raising=False)
    val = load_groq_key()
    if val:
        pytest.skip(f"local Groq key file present on this machine ({len(val)} chars)")
    assert val == ""
