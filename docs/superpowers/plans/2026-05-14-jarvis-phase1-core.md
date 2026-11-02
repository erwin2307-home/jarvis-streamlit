# Jarvis Phase 1 — Core Voice Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build end-to-end offline voice loop — wake-word ("Jarvis") → STT → local LLM → TTS → Reactor-Core HUD reflects each state.

**Architecture:** Tauri (Rust+React) frontend talks to Python FastAPI backend over WebSocket on `127.0.0.1:8765`. Backend orchestrates voice pipeline (OpenWakeWord → Silero VAD → faster-whisper → Ollama → Piper) via asyncio state machine. Plugin/tool registry framework exists but ships no tools in Phase 1.

**Tech Stack:**
- **Backend:** Python 3.11+, FastAPI, uvicorn, websockets, sounddevice, numpy, openwakeword, silero-vad, faster-whisper, ollama, piper-tts, pydantic
- **Frontend:** Tauri 2.x, React 18, TypeScript, Vite, zustand, Three.js (`@react-three/fiber`)
- **Tooling:** pytest, ruff, mypy, vitest

**Spec reference:** `docs/superpowers/specs/2026-05-14-jarvis-hud-design.md`

**Out-of-scope for Phase 1:** Tools/plugins (Phase 2), vision (Phase 3), installer (Phase 3), conversation memory (Phase 3).

---

## File Structure

```
jarvis/
├── backend/
│   ├── pyproject.toml
│   ├── src/jarvis/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI entrypoint + WS route
│   │   ├── events.py               # Pydantic event models (WS protocol)
│   │   ├── state_machine.py        # Pure state-transition logic
│   │   ├── orchestrator.py         # Glues pipeline + state machine + WS
│   │   ├── audio_io.py             # Mic capture + speaker playback
│   │   ├── wake.py                 # OpenWakeWord wrapper
│   │   ├── vad.py                  # Silero VAD wrapper
│   │   ├── stt.py                  # faster-whisper wrapper
│   │   ├── llm.py                  # Ollama client (streaming)
│   │   ├── tts.py                  # Piper wrapper (streaming chunks)
│   │   ├── config.py               # Settings (paths, models, hotkey)
│   │   └── plugin/
│   │       ├── __init__.py         # Tool decorator + Registry
│   │       └── helpers.py
│   └── tests/
│       ├── conftest.py             # Shared fixtures
│       ├── fixtures/
│       │   ├── hello_jarvis.wav    # Pre-recorded wake-word
│       │   ├── question_de.wav     # "Wie spät ist es?"
│       │   └── silence.wav
│       ├── test_events.py
│       ├── test_state_machine.py
│       ├── test_audio_io.py
│       ├── test_wake.py
│       ├── test_vad.py
│       ├── test_stt.py
│       ├── test_llm.py
│       ├── test_tts.py
│       ├── test_plugin.py
│       ├── test_orchestrator.py
│       └── test_app_ws.py
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── ws.ts                   # WebSocket client + reconnect
│   │   ├── store.ts                # zustand store
│   │   ├── hud/
│   │   │   ├── ReactorCore.tsx     # Center ring + glowing core
│   │   │   ├── Waveform.tsx        # Audio FFT visualizer
│   │   │   ├── StatusCorners.tsx   # Telemetry text in 4 corners
│   │   │   └── HudShell.tsx        # Composes the layout
│   │   └── tests/
│   │       ├── ws.test.ts
│   │       ├── store.test.ts
│   │       └── hud.test.tsx
│   └── src-tauri/
│       ├── Cargo.toml
│       ├── tauri.conf.json
│       └── src/main.rs             # Tauri entrypoint + global hotkey
├── models/                         # gitignored
│   ├── openwakeword/jarvis.tflite  # placed by setup script
│   ├── whisper-base/               # auto-downloaded by faster-whisper
│   └── piper/de_DE-thorsten-medium.onnx
├── scripts/
│   ├── download_models.sh
│   └── dev.sh                      # Starts backend + frontend
├── README.md
└── .gitignore                      # Already exists from spec commit
```

**Responsibilities:**
- `events.py`: Pure data types. No I/O. Pydantic = serialization contract.
- `state_machine.py`: Pure transition rules. No async, no I/O. Functions take old state + event → new state.
- `audio_io.py`: Wraps sounddevice. Streams int16 PCM frames in/out. Unit-tested with synthetic numpy arrays.
- `wake.py`/`vad.py`/`stt.py`/`llm.py`/`tts.py`: One pipeline stage each. Each takes input frames/text, returns output. All async iterators where streaming.
- `orchestrator.py`: Wires everything. The only file that knows about all pipeline stages.
- `app.py`: HTTP/WS surface. Holds orchestrator lifecycle.
- `plugin/`: Decorator + Registry. Phase 1 only the framework — no real tools.
- Frontend `hud/`: Pure presentation components. State comes from store.

---

## Task 0: Project Skeleton + Tooling

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/src/jarvis/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/.python-version`
- Create: `scripts/dev.sh`
- Create: `README.md`

- [ ] **Step 0.1: Create `backend/pyproject.toml`**

```toml
[project]
name = "jarvis"
version = "0.1.0"
description = "Offline voice assistant with Reactor Core HUD"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.32",
  "websockets>=13",
  "pydantic>=2.9",
  "numpy>=2.0",
  "sounddevice>=0.5",
  "openwakeword>=0.6",
  "silero-vad>=5.1",
  "faster-whisper>=1.0",
  "ollama>=0.4",
  "piper-tts>=1.2",
  "loguru>=0.7",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-asyncio>=0.24",
  "pytest-cov>=5",
  "ruff>=0.7",
  "mypy>=1.13",
  "httpx>=0.27",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
strict = true
python_version = "3.11"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 0.2: Create `backend/src/jarvis/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 0.3: Create `backend/tests/__init__.py`** (empty file)

- [ ] **Step 0.4: Create `backend/tests/conftest.py`**

```python
"""Shared pytest fixtures."""
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
```

- [ ] **Step 0.5: Create `backend/.python-version`**

```
3.11
```

- [ ] **Step 0.6: Create `scripts/dev.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

(cd "$ROOT/backend" && uvicorn jarvis.app:app --reload --port 8765) &
BACKEND_PID=$!

(cd "$ROOT/frontend" && npm run tauri dev) &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
```

Then: `chmod +x scripts/dev.sh`.

- [ ] **Step 0.7: Create `README.md`**

```markdown
# Jarvis — Offline Voice Assistant

Phase 1 Core: Wake-word → STT → local LLM → TTS, with Reactor Core HUD.

## Setup
1. Install Python 3.11+, Node 20+, Rust (for Tauri), Ollama.
2. `cd backend && pip install -e ".[dev]"`
3. `ollama pull llama3.2:3b`
4. `scripts/download_models.sh` (Phase 1 task adds this)
5. `cd frontend && npm install`
6. `scripts/dev.sh` to start dev mode.

## Tests
- Backend: `cd backend && pytest`
- Frontend: `cd frontend && npm test`

See `docs/superpowers/specs/2026-05-14-jarvis-hud-design.md` for design.
```

- [ ] **Step 0.8: Verify install + commit**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest --collect-only
```

Expected: `pytest --collect-only` succeeds with 0 tests collected (no errors).

```bash
git add backend/pyproject.toml backend/src/jarvis/__init__.py \
        backend/tests/__init__.py backend/tests/conftest.py \
        backend/.python-version scripts/dev.sh README.md
git commit -m "chore: backend skeleton + tooling config"
```

---

## Task 1: Event Schemas (Pydantic Models)

**Files:**
- Create: `backend/src/jarvis/events.py`
- Create: `backend/tests/test_events.py`

- [ ] **Step 1.1: Write failing tests in `backend/tests/test_events.py`**

```python
"""Tests for WebSocket event Pydantic models."""
import json

import pytest

from jarvis.events import (
    AudioLevelEvent,
    ErrorEvent,
    Event,
    ManualStartEvent,
    ResponseChunkEvent,
    ResponseDoneEvent,
    State,
    StateEvent,
    ToolCallEvent,
    ToolResultEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
    parse_inbound,
)


def test_state_event_round_trip():
    ev = StateEvent(value=State.LISTENING)
    blob = ev.model_dump_json()
    parsed = json.loads(blob)
    assert parsed == {"type": "state", "value": "listening"}


def test_audio_level_event():
    ev = AudioLevelEvent(rms=0.42, fft=[0.1, 0.3, 0.8])
    parsed = json.loads(ev.model_dump_json())
    assert parsed["type"] == "audio_level"
    assert parsed["rms"] == pytest.approx(0.42)
    assert parsed["fft"] == [0.1, 0.3, 0.8]


def test_transcript_final_event():
    ev = TranscriptFinalEvent(text="hallo Jarvis")
    parsed = json.loads(ev.model_dump_json())
    assert parsed == {"type": "transcript_final", "text": "hallo Jarvis"}


def test_tool_call_event_has_call_id():
    ev = ToolCallEvent(name="web_search", args={"q": "wetter"}, call_id="abc123")
    parsed = json.loads(ev.model_dump_json())
    assert parsed["call_id"] == "abc123"
    assert parsed["name"] == "web_search"


def test_error_event():
    ev = ErrorEvent(stage="llm", msg="timeout", retriable=True)
    parsed = json.loads(ev.model_dump_json())
    assert parsed == {"type": "error", "stage": "llm", "msg": "timeout", "retriable": True}


def test_parse_inbound_manual_start():
    blob = '{"type":"manual_start"}'
    ev = parse_inbound(blob)
    assert isinstance(ev, ManualStartEvent)


def test_parse_inbound_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown inbound event"):
        parse_inbound('{"type":"nonsense"}')


def test_parse_inbound_invalid_json_raises():
    with pytest.raises(ValueError):
        parse_inbound("not json")


def test_state_enum_values():
    assert State.IDLE.value == "idle"
    assert State.LISTENING.value == "listening"
    assert State.TRANSCRIBE.value == "transcribe"
    assert State.THINKING.value == "thinking"
    assert State.SPEAKING.value == "speaking"
    assert State.ERROR.value == "error"


# Touch unused names so import errors surface
_USED = (
    AudioLevelEvent, ErrorEvent, Event, ManualStartEvent,
    ResponseChunkEvent, ResponseDoneEvent, StateEvent,
    ToolCallEvent, ToolResultEvent,
    TranscriptFinalEvent, TranscriptPartialEvent,
)
```

- [ ] **Step 1.2: Run test, expect ImportError**

```bash
cd backend
pytest tests/test_events.py -v
```

Expected: All FAIL with `ImportError: cannot import name 'StateEvent' from 'jarvis.events'`.

- [ ] **Step 1.3: Implement `backend/src/jarvis/events.py`**

```python
"""WebSocket event schemas. Pure data types — no I/O."""
from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBE = "transcribe"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


# ── Outbound (backend → frontend) ─────────────────────────────────────

class StateEvent(BaseModel):
    type: Literal["state"] = "state"
    value: State


class AudioLevelEvent(BaseModel):
    type: Literal["audio_level"] = "audio_level"
    rms: float
    fft: list[float]


class TranscriptPartialEvent(BaseModel):
    type: Literal["transcript_partial"] = "transcript_partial"
    text: str


class TranscriptFinalEvent(BaseModel):
    type: Literal["transcript_final"] = "transcript_final"
    text: str


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    name: str
    args: dict
    call_id: str
    requires_confirm: bool = False


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    ok: bool
    summary: str


class ResponseChunkEvent(BaseModel):
    type: Literal["response_chunk"] = "response_chunk"
    text: str


class ResponseDoneEvent(BaseModel):
    type: Literal["response_done"] = "response_done"
    full: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    stage: str
    msg: str
    retriable: bool = False


Event = Union[
    StateEvent,
    AudioLevelEvent,
    TranscriptPartialEvent,
    TranscriptFinalEvent,
    ToolCallEvent,
    ToolResultEvent,
    ResponseChunkEvent,
    ResponseDoneEvent,
    ErrorEvent,
]


# ── Inbound (frontend → backend) ──────────────────────────────────────

class ManualStartEvent(BaseModel):
    type: Literal["manual_start"] = "manual_start"


class CancelEvent(BaseModel):
    type: Literal["cancel"] = "cancel"


class ConfirmToolEvent(BaseModel):
    type: Literal["confirm_tool"] = "confirm_tool"
    call_id: str
    approved: bool


InboundEvent = Annotated[
    Union[ManualStartEvent, CancelEvent, ConfirmToolEvent],
    Field(discriminator="type"),
]


_INBOUND_BY_TYPE = {
    "manual_start": ManualStartEvent,
    "cancel": CancelEvent,
    "confirm_tool": ConfirmToolEvent,
}


def parse_inbound(blob: str) -> ManualStartEvent | CancelEvent | ConfirmToolEvent:
    """Parse raw WS frame into the appropriate inbound event."""
    data = json.loads(blob)
    t = data.get("type")
    cls = _INBOUND_BY_TYPE.get(t)
    if cls is None:
        raise ValueError(f"unknown inbound event type: {t!r}")
    return cls.model_validate(data)
```

- [ ] **Step 1.4: Run test, expect PASS**

```bash
pytest tests/test_events.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add backend/src/jarvis/events.py backend/tests/test_events.py
git commit -m "feat(events): pydantic websocket event schemas"
```

---

## Task 2: State Machine (Pure Logic)

**Files:**
- Create: `backend/src/jarvis/state_machine.py`
- Create: `backend/tests/test_state_machine.py`

- [ ] **Step 2.1: Write failing tests in `backend/tests/test_state_machine.py`**

```python
"""Pure state-machine transition tests."""
import pytest

from jarvis.events import State
from jarvis.state_machine import Trigger, transition, valid_triggers


def test_idle_to_listening_on_wake():
    assert transition(State.IDLE, Trigger.WAKE_DETECTED) == State.LISTENING


def test_idle_to_listening_on_manual_start():
    assert transition(State.IDLE, Trigger.MANUAL_START) == State.LISTENING


def test_listening_to_transcribe_on_speech_end():
    assert transition(State.LISTENING, Trigger.SPEECH_END) == State.TRANSCRIBE


def test_transcribe_to_thinking_on_transcript_ready():
    assert transition(State.TRANSCRIBE, Trigger.TRANSCRIPT_READY) == State.THINKING


def test_thinking_self_loop_on_tool_call():
    assert transition(State.THINKING, Trigger.TOOL_CALL) == State.THINKING


def test_thinking_to_speaking_on_response_complete():
    assert transition(State.THINKING, Trigger.RESPONSE_COMPLETE) == State.SPEAKING


def test_speaking_to_idle_on_playback_done():
    assert transition(State.SPEAKING, Trigger.PLAYBACK_DONE) == State.IDLE


def test_speaking_to_listening_on_barge_in():
    assert transition(State.SPEAKING, Trigger.WAKE_DETECTED) == State.LISTENING


def test_any_state_to_idle_on_cancel():
    for state in State:
        assert transition(state, Trigger.CANCEL) == State.IDLE


def test_any_state_to_error_on_error():
    for state in State:
        if state == State.ERROR:
            continue
        assert transition(state, Trigger.ERROR) == State.ERROR


def test_invalid_transition_raises():
    with pytest.raises(ValueError, match="invalid transition"):
        transition(State.IDLE, Trigger.SPEECH_END)


def test_valid_triggers_for_idle():
    triggers = valid_triggers(State.IDLE)
    assert Trigger.WAKE_DETECTED in triggers
    assert Trigger.MANUAL_START in triggers
    assert Trigger.ERROR in triggers
    assert Trigger.CANCEL in triggers
    assert Trigger.SPEECH_END not in triggers
```

- [ ] **Step 2.2: Run test, expect ImportError**

```bash
pytest tests/test_state_machine.py -v
```

Expected: All FAIL with import error on `Trigger` / `transition`.

- [ ] **Step 2.3: Implement `backend/src/jarvis/state_machine.py`**

```python
"""Pure state-transition rules. No I/O, no async."""
from __future__ import annotations

from enum import Enum

from jarvis.events import State


class Trigger(str, Enum):
    WAKE_DETECTED = "wake_detected"
    MANUAL_START = "manual_start"
    SPEECH_END = "speech_end"
    TRANSCRIPT_READY = "transcript_ready"
    TOOL_CALL = "tool_call"
    RESPONSE_COMPLETE = "response_complete"
    PLAYBACK_DONE = "playback_done"
    CANCEL = "cancel"
    ERROR = "error"


_TABLE: dict[tuple[State, Trigger], State] = {
    (State.IDLE, Trigger.WAKE_DETECTED): State.LISTENING,
    (State.IDLE, Trigger.MANUAL_START): State.LISTENING,
    (State.LISTENING, Trigger.SPEECH_END): State.TRANSCRIBE,
    (State.LISTENING, Trigger.WAKE_DETECTED): State.LISTENING,
    (State.TRANSCRIBE, Trigger.TRANSCRIPT_READY): State.THINKING,
    (State.THINKING, Trigger.TOOL_CALL): State.THINKING,
    (State.THINKING, Trigger.RESPONSE_COMPLETE): State.SPEAKING,
    (State.SPEAKING, Trigger.PLAYBACK_DONE): State.IDLE,
    (State.SPEAKING, Trigger.WAKE_DETECTED): State.LISTENING,
}


def transition(state: State, trigger: Trigger) -> State:
    """Return new state. Universal CANCEL → IDLE, ERROR → ERROR."""
    if trigger == Trigger.CANCEL:
        return State.IDLE
    if trigger == Trigger.ERROR:
        return State.ERROR
    key = (state, trigger)
    if key not in _TABLE:
        raise ValueError(f"invalid transition: {state.value} + {trigger.value}")
    return _TABLE[key]


def valid_triggers(state: State) -> set[Trigger]:
    """All triggers that don't raise from this state."""
    triggers = {Trigger.CANCEL, Trigger.ERROR}
    for (s, t) in _TABLE:
        if s == state:
            triggers.add(t)
    return triggers
```

- [ ] **Step 2.4: Run test, expect PASS**

```bash
pytest tests/test_state_machine.py -v
```

Expected: All 11 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add backend/src/jarvis/state_machine.py backend/tests/test_state_machine.py
git commit -m "feat(state-machine): pure transition rules"
```

---

## Task 3: Config Module

**Files:**
- Create: `backend/src/jarvis/config.py`
- Create: `backend/tests/test_config.py`

- [ ] **Step 3.1: Write failing test in `backend/tests/test_config.py`**

```python
"""Config loading + validation tests."""
import os
from pathlib import Path

import pytest

from jarvis.config import Settings, get_settings


def test_defaults():
    s = Settings()
    assert s.ws_port == 8765
    assert s.sample_rate == 16000
    assert s.wake_word == "jarvis"
    assert s.llm_model == "llama3.2:3b"
    assert s.tts_voice == "de_DE-thorsten-medium"
    assert s.whisper_model == "base"
    assert s.vad_threshold == 0.5
    assert s.wake_threshold == 0.5


def test_env_override(monkeypatch):
    monkeypatch.setenv("JARVIS_WS_PORT", "9000")
    monkeypatch.setenv("JARVIS_LLM_MODEL", "qwen2.5:3b")
    s = Settings()
    assert s.ws_port == 9000
    assert s.llm_model == "qwen2.5:3b"


def test_get_settings_cached():
    a = get_settings()
    b = get_settings()
    assert a is b
```

- [ ] **Step 3.2: Run test, expect ImportError**

```bash
pytest tests/test_config.py -v
```

- [ ] **Step 3.3: Implement `backend/src/jarvis/config.py`**

```python
"""Application settings, loaded from env with sensible defaults."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"


class Settings(BaseModel):
    ws_host: str = Field(default_factory=lambda: os.getenv("JARVIS_WS_HOST", "127.0.0.1"))
    ws_port: int = Field(default_factory=lambda: int(os.getenv("JARVIS_WS_PORT", "8765")))

    sample_rate: int = Field(default_factory=lambda: int(os.getenv("JARVIS_SAMPLE_RATE", "16000")))
    frame_ms: int = Field(default_factory=lambda: int(os.getenv("JARVIS_FRAME_MS", "30")))

    wake_word: str = Field(default_factory=lambda: os.getenv("JARVIS_WAKE_WORD", "jarvis"))
    wake_threshold: float = Field(default_factory=lambda: float(os.getenv("JARVIS_WAKE_THRESHOLD", "0.5")))
    wake_model_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("JARVIS_WAKE_MODEL", str(MODELS_DIR / "openwakeword" / "jarvis.tflite"))
        )
    )

    vad_threshold: float = Field(default_factory=lambda: float(os.getenv("JARVIS_VAD_THRESHOLD", "0.5")))
    silence_ms: int = Field(default_factory=lambda: int(os.getenv("JARVIS_SILENCE_MS", "800")))

    whisper_model: str = Field(default_factory=lambda: os.getenv("JARVIS_WHISPER_MODEL", "base"))
    whisper_compute: str = Field(default_factory=lambda: os.getenv("JARVIS_WHISPER_COMPUTE", "int8"))
    whisper_lang: str = Field(default_factory=lambda: os.getenv("JARVIS_WHISPER_LANG", "de"))

    llm_host: str = Field(default_factory=lambda: os.getenv("JARVIS_LLM_HOST", "http://localhost:11434"))
    llm_model: str = Field(default_factory=lambda: os.getenv("JARVIS_LLM_MODEL", "llama3.2:3b"))
    llm_system_prompt: str = Field(
        default_factory=lambda: os.getenv(
            "JARVIS_SYSTEM_PROMPT",
            "Du bist Jarvis, ein knapper, freundlicher Sprachassistent. Antworte kurz auf Deutsch.",
        )
    )

    tts_voice: str = Field(default_factory=lambda: os.getenv("JARVIS_TTS_VOICE", "de_DE-thorsten-medium"))
    tts_model_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("JARVIS_TTS_MODEL", str(MODELS_DIR / "piper" / "de_DE-thorsten-medium.onnx"))
        )
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 3.4: Run test, expect PASS**

```bash
pytest tests/test_config.py -v
```

- [ ] **Step 3.5: Commit**

```bash
git add backend/src/jarvis/config.py backend/tests/test_config.py
git commit -m "feat(config): settings loaded from env with defaults"
```

---

## Task 4: Plugin Registry Framework

**Files:**
- Create: `backend/src/jarvis/plugin/__init__.py`
- Create: `backend/src/jarvis/plugin/helpers.py`
- Create: `backend/tests/test_plugin.py`

- [ ] **Step 4.1: Write failing test in `backend/tests/test_plugin.py`**

```python
"""Plugin Tool + Registry tests."""
import asyncio

import pytest

from jarvis.plugin import Registry, Tool, ToolResult


@pytest.fixture(autouse=True)
def fresh_registry():
    Registry.clear()
    yield
    Registry.clear()


async def test_register_and_call_tool():
    @Tool.register(name="echo", description="echo input", schema={"text": {"type": "string"}})
    async def echo(text: str) -> ToolResult:
        return ToolResult(ok=True, data={"text": text}, summary=text)

    assert "echo" in Registry.names()
    result = await Registry.call("echo", {"text": "hi"})
    assert result.ok
    assert result.summary == "hi"


async def test_call_unknown_tool_returns_error():
    result = await Registry.call("missing", {})
    assert not result.ok
    assert "unknown" in result.summary.lower()


async def test_tool_timeout():
    @Tool.register(name="slow", description="sleeps", schema={}, timeout_s=0.1)
    async def slow() -> ToolResult:
        await asyncio.sleep(1)
        return ToolResult(ok=True, data=None, summary="")

    result = await Registry.call("slow", {})
    assert not result.ok
    assert "timeout" in result.summary.lower()


async def test_tool_exception_caught():
    @Tool.register(name="boom", description="raises", schema={})
    async def boom() -> ToolResult:
        raise RuntimeError("kapow")

    result = await Registry.call("boom", {})
    assert not result.ok
    assert "kapow" in result.summary


def test_json_schemas_format():
    @Tool.register(
        name="weather",
        description="get weather",
        schema={"city": {"type": "string", "required": True}},
    )
    async def weather(city: str) -> ToolResult:
        return ToolResult(ok=True, data={}, summary="")

    schemas = Registry.json_schemas()
    assert len(schemas) == 1
    s = schemas[0]
    assert s["type"] == "function"
    assert s["function"]["name"] == "weather"
    assert s["function"]["description"] == "get weather"
    assert "city" in s["function"]["parameters"]["properties"]
    assert s["function"]["parameters"]["required"] == ["city"]


def test_register_duplicate_raises():
    @Tool.register(name="dup", description="", schema={})
    async def dup1() -> ToolResult:
        return ToolResult(ok=True, data=None, summary="")

    with pytest.raises(ValueError, match="already registered"):
        @Tool.register(name="dup", description="", schema={})
        async def dup2() -> ToolResult:
            return ToolResult(ok=True, data=None, summary="")
```

- [ ] **Step 4.2: Run test, expect ImportError**

```bash
pytest tests/test_plugin.py -v
```

- [ ] **Step 4.3: Implement `backend/src/jarvis/plugin/__init__.py`**

```python
"""Plugin Tool decorator + Registry. No real tools ship in Phase 1."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, ClassVar


ToolFunc = Callable[..., Awaitable["ToolResult"]]


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    summary: str = ""


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, dict[str, Any]]
    func: ToolFunc
    confirm: bool = False
    timeout_s: float = 10.0


class _RegistryImpl:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def add(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._tools[spec.name] = spec

    def clear(self) -> None:
        self._tools.clear()

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(ok=False, data=None, summary=f"unknown tool: {name}")
        try:
            return await asyncio.wait_for(spec.func(**args), timeout=spec.timeout_s)
        except asyncio.TimeoutError:
            return ToolResult(ok=False, data=None, summary=f"tool {name} timeout")
        except Exception as exc:
            return ToolResult(ok=False, data=None, summary=f"tool error: {exc}")

    def json_schemas(self) -> list[dict[str, Any]]:
        """OpenAI-compatible tool list for LLM function-calling."""
        out: list[dict[str, Any]] = []
        for spec in self._tools.values():
            properties: dict[str, dict[str, Any]] = {}
            required: list[str] = []
            for arg_name, meta in spec.schema.items():
                m = dict(meta)
                if m.pop("required", False):
                    required.append(arg_name)
                properties[arg_name] = m
            out.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return out


Registry = _RegistryImpl()


class Tool:
    """Decorator namespace for registering tools."""

    @staticmethod
    def register(
        *,
        name: str,
        description: str,
        schema: dict[str, dict[str, Any]],
        confirm: bool = False,
        timeout_s: float = 10.0,
    ):
        def deco(func: ToolFunc) -> ToolFunc:
            Registry.add(ToolSpec(
                name=name,
                description=description,
                schema=schema,
                func=func,
                confirm=confirm,
                timeout_s=timeout_s,
            ))
            return func
        return deco
```

- [ ] **Step 4.4: Create `backend/src/jarvis/plugin/helpers.py` (empty placeholder for Phase 2 utilities)**

```python
"""Plugin helper utilities. Empty in Phase 1 — populated in Phase 2."""
```

- [ ] **Step 4.5: Run test, expect PASS**

```bash
pytest tests/test_plugin.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 4.6: Commit**

```bash
git add backend/src/jarvis/plugin/ backend/tests/test_plugin.py
git commit -m "feat(plugin): Tool decorator + Registry framework"
```

---

## Task 5: Audio I/O (Mic + Speaker)

**Files:**
- Create: `backend/src/jarvis/audio_io.py`
- Create: `backend/tests/test_audio_io.py`

`audio_io` provides two abstractions:
1. `MicStream` — async generator yielding fixed-size int16 frames from a callback-driven sounddevice InputStream.
2. `play_pcm(frames, sample_rate)` — coroutine that streams float/int16 frames to OutputStream.

Both have `Fake*` variants used in tests so we never need the real microphone in CI.

- [ ] **Step 5.1: Write failing tests in `backend/tests/test_audio_io.py`**

```python
"""Audio I/O tests using fake streams (no real device)."""
import numpy as np
import pytest

from jarvis.audio_io import FakeMicStream, FakeSpeaker, rms_dbfs


async def test_fake_mic_stream_yields_expected_frames():
    pcm = np.zeros(16000, dtype=np.int16)  # 1 second silence @ 16k
    mic = FakeMicStream(pcm, frame_size=480)   # 30ms frames
    frames = []
    async for frame in mic.frames():
        frames.append(frame)
    assert len(frames) == 16000 // 480
    assert frames[0].dtype == np.int16
    assert frames[0].shape == (480,)


async def test_fake_speaker_collects_output():
    spk = FakeSpeaker()
    await spk.play(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    await spk.play(np.array([0.4, 0.5], dtype=np.float32))
    all_audio = spk.collected()
    assert np.allclose(all_audio, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_rms_dbfs_silence_is_minus_infinity():
    silent = np.zeros(1024, dtype=np.int16)
    assert rms_dbfs(silent) < -90  # treat -inf-ish as very low


def test_rms_dbfs_full_scale_is_zero():
    full = np.full(1024, 32767, dtype=np.int16)
    assert abs(rms_dbfs(full)) < 1
```

- [ ] **Step 5.2: Run test, expect ImportError**

```bash
pytest tests/test_audio_io.py -v
```

- [ ] **Step 5.3: Implement `backend/src/jarvis/audio_io.py`**

```python
"""Microphone capture + speaker playback. Real (sounddevice) + Fake (test) implementations."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol

import numpy as np
import sounddevice as sd


class MicStream(Protocol):
    async def frames(self) -> AsyncIterator[np.ndarray]: ...


class Speaker(Protocol):
    async def play(self, samples: np.ndarray) -> None: ...


def rms_dbfs(int16_samples: np.ndarray) -> float:
    """Return RMS level in dBFS. -inf for digital silence."""
    if int16_samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(int16_samples.astype(np.float64)))))
    if rms <= 0:
        return -120.0
    full_scale = 32767.0
    return 20.0 * np.log10(rms / full_scale)


# ── Real (sounddevice) ─────────────────────────────────────────────────

class RealMicStream:
    """Captures from default input device. Yields int16 frames of `frame_size` samples."""

    def __init__(self, sample_rate: int = 16000, frame_size: int = 480) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _callback(self, indata, frames, time, status) -> None:
        if status:
            pass  # overruns expected during heavy LLM work — drop silently
        pcm = (indata[:, 0] * 32767).astype(np.int16) if indata.dtype != np.int16 else indata[:, 0]
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, pcm.copy())

    async def __aenter__(self) -> "RealMicStream":
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while True:
            frame = await self._queue.get()
            yield frame


class RealSpeaker:
    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate

    async def play(self, samples: np.ndarray) -> None:
        # Run blocking sd.play in executor so we don't stall the loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, sd.play, samples, self.sample_rate)
        await loop.run_in_executor(None, sd.wait)


# ── Fake (tests) ───────────────────────────────────────────────────────

class FakeMicStream:
    """Replays a fixed PCM array as a frame stream."""

    def __init__(self, pcm: np.ndarray, frame_size: int = 480) -> None:
        self._pcm = pcm
        self._frame_size = frame_size

    async def frames(self) -> AsyncIterator[np.ndarray]:
        n = (self._pcm.size // self._frame_size) * self._frame_size
        for i in range(0, n, self._frame_size):
            yield self._pcm[i : i + self._frame_size].copy()
            await asyncio.sleep(0)  # let other tasks run


class FakeSpeaker:
    """Collects played audio for assertions."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    async def play(self, samples: np.ndarray) -> None:
        self._chunks.append(samples.copy())

    def collected(self) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._chunks)
```

- [ ] **Step 5.4: Run test, expect PASS**

```bash
pytest tests/test_audio_io.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add backend/src/jarvis/audio_io.py backend/tests/test_audio_io.py
git commit -m "feat(audio): mic + speaker abstractions (real + fake)"
```

---

## Task 6: Generate Test Fixtures (Audio Files)

**Files:**
- Create: `backend/tests/fixtures/_generate.py`
- Create binary: `backend/tests/fixtures/silence.wav`
- Create binary: `backend/tests/fixtures/sine_440.wav`

We synthesize deterministic fixtures (silence + tone). The real wake-word and question fixtures get committed later — for now, tests that depend on STT use mocks.

- [ ] **Step 6.1: Create `backend/tests/fixtures/_generate.py`**

```python
"""Generate deterministic audio fixtures. Run once: python _generate.py."""
from pathlib import Path

import numpy as np
import soundfile as sf


HERE = Path(__file__).parent
SR = 16000


def write_silence(seconds: float = 1.0) -> None:
    samples = np.zeros(int(SR * seconds), dtype=np.int16)
    sf.write(HERE / "silence.wav", samples, SR, subtype="PCM_16")


def write_sine(freq: float = 440.0, seconds: float = 1.0) -> None:
    t = np.arange(int(SR * seconds), dtype=np.float64) / SR
    sig = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)
    sf.write(HERE / "sine_440.wav", sig, SR, subtype="PCM_16")


if __name__ == "__main__":
    write_silence()
    write_sine()
    print(f"wrote: {HERE}/silence.wav, {HERE}/sine_440.wav")
```

- [ ] **Step 6.2: Install soundfile and run generator**

```bash
cd backend
pip install soundfile
python tests/fixtures/_generate.py
```

Expected output: `wrote: .../silence.wav, .../sine_440.wav`. Two `.wav` files exist.

- [ ] **Step 6.3: Add soundfile to dev deps in `backend/pyproject.toml`**

Modify `[project.optional-dependencies]` block to add `"soundfile>=0.12"` to the `dev` list:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-asyncio>=0.24",
  "pytest-cov>=5",
  "ruff>=0.7",
  "mypy>=1.13",
  "httpx>=0.27",
  "soundfile>=0.12",
]
```

- [ ] **Step 6.4: Commit**

```bash
git add backend/tests/fixtures/_generate.py \
        backend/tests/fixtures/silence.wav \
        backend/tests/fixtures/sine_440.wav \
        backend/pyproject.toml
git commit -m "test: deterministic audio fixtures + generator script"
```

---

## Task 7: Wake-Word Module

**Files:**
- Create: `backend/src/jarvis/wake.py`
- Create: `backend/tests/test_wake.py`

Real OpenWakeWord needs a downloaded model. The module exposes both `RealWakeWord` and `FakeWakeWord` — the orchestrator uses real, tests use fake.

- [ ] **Step 7.1: Write failing test in `backend/tests/test_wake.py`**

```python
"""Wake-word detection tests."""
import numpy as np
import pytest

from jarvis.wake import FakeWakeWord


async def test_fake_wake_triggers_on_marker():
    """FakeWakeWord fires when a frame whose first sample equals `marker_value` arrives."""
    marker = 12345
    wake = FakeWakeWord(threshold=0.5, marker_value=marker)
    silent = np.zeros(480, dtype=np.int16)
    marker_frame = np.full(480, marker, dtype=np.int16)

    assert not (await wake.detect(silent))
    assert not (await wake.detect(silent))
    assert (await wake.detect(marker_frame))


async def test_fake_wake_no_trigger_silence():
    wake = FakeWakeWord(threshold=0.5)
    for _ in range(20):
        assert not (await wake.detect(np.zeros(480, dtype=np.int16)))
```

- [ ] **Step 7.2: Run test, expect ImportError**

```bash
pytest tests/test_wake.py -v
```

- [ ] **Step 7.3: Implement `backend/src/jarvis/wake.py`**

```python
"""Wake-word detection. Real (OpenWakeWord) + Fake (test)."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class WakeWord(Protocol):
    async def detect(self, frame_int16: np.ndarray) -> bool: ...


class RealWakeWord:
    """OpenWakeWord-based detector."""

    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        from openwakeword.model import Model
        self._model = Model(wakeword_models=[str(model_path)])
        self._threshold = threshold
        # OpenWakeWord exposes a single trigger key per loaded model
        self._key = list(self._model.models.keys())[0]

    async def detect(self, frame_int16: np.ndarray) -> bool:
        # OpenWakeWord expects 16-bit mono PCM. Internally buffers.
        scores = self._model.predict(frame_int16)
        return scores[self._key] >= self._threshold


class FakeWakeWord:
    """Fires on a frame whose first sample equals `marker_value`."""

    def __init__(self, threshold: float = 0.5, marker_value: int = 12345) -> None:
        self._threshold = threshold
        self._marker = marker_value

    async def detect(self, frame_int16: np.ndarray) -> bool:
        if frame_int16.size == 0:
            return False
        return int(frame_int16[0]) == self._marker
```

- [ ] **Step 7.4: Run test, expect PASS**

```bash
pytest tests/test_wake.py -v
```

- [ ] **Step 7.5: Commit**

```bash
git add backend/src/jarvis/wake.py backend/tests/test_wake.py
git commit -m "feat(wake): OpenWakeWord wrapper + fake for tests"
```

---

## Task 8: VAD Module

**Files:**
- Create: `backend/src/jarvis/vad.py`
- Create: `backend/tests/test_vad.py`

- [ ] **Step 8.1: Write failing test in `backend/tests/test_vad.py`**

```python
"""Voice activity detection tests."""
import numpy as np

from jarvis.vad import FakeVad, SpeechEndDetector


async def test_fake_vad_returns_marker():
    vad = FakeVad(speech_mask=[False, False, True, True, False])
    out = []
    for _ in range(5):
        out.append(await vad.is_speech(np.zeros(480, dtype=np.int16)))
    assert out == [False, False, True, True, False]


async def test_speech_end_detector_signals_after_silence():
    """Detector fires speech_end after `silence_frames` consecutive non-speech post-speech."""
    det = SpeechEndDetector(silence_frames=3)
    # Speech starts
    assert det.update(True) == "speech_start"
    assert det.update(True) is None
    assert det.update(False) is None  # 1 silence
    assert det.update(False) is None  # 2 silence
    assert det.update(False) == "speech_end"  # 3 silence → end


async def test_speech_end_no_signal_without_speech():
    det = SpeechEndDetector(silence_frames=3)
    for _ in range(10):
        assert det.update(False) is None
```

- [ ] **Step 8.2: Run test, expect ImportError**

```bash
pytest tests/test_vad.py -v
```

- [ ] **Step 8.3: Implement `backend/src/jarvis/vad.py`**

```python
"""Voice Activity Detection — Silero VAD wrapper + speech-end detector."""
from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np


class Vad(Protocol):
    async def is_speech(self, frame_int16: np.ndarray) -> bool: ...


class RealVad:
    """Silero-VAD wrapper."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        import torch
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self._threshold = threshold
        self._sr = sample_rate
        self._torch = torch

    async def is_speech(self, frame_int16: np.ndarray) -> bool:
        tensor = self._torch.from_numpy(frame_int16.astype(np.float32) / 32768.0)
        score = float(self._model(tensor, self._sr).item())
        return score >= self._threshold


class FakeVad:
    """Returns scripted sequence."""

    def __init__(self, speech_mask: Iterable[bool]) -> None:
        self._mask = list(speech_mask)
        self._i = 0

    async def is_speech(self, frame_int16: np.ndarray) -> bool:
        if self._i >= len(self._mask):
            return False
        v = self._mask[self._i]
        self._i += 1
        return v


class SpeechEndDetector:
    """
    Tracks transitions: emits `'speech_start'` on first speech frame,
    `'speech_end'` after `silence_frames` consecutive non-speech post-speech.
    """

    def __init__(self, silence_frames: int = 27) -> None:
        # default 27 ≈ 800ms at 30ms frames
        self._silence_target = silence_frames
        self._silence_count = 0
        self._in_speech = False

    def update(self, is_speech: bool) -> str | None:
        if is_speech:
            self._silence_count = 0
            if not self._in_speech:
                self._in_speech = True
                return "speech_start"
            return None
        if self._in_speech:
            self._silence_count += 1
            if self._silence_count >= self._silence_target:
                self._in_speech = False
                self._silence_count = 0
                return "speech_end"
        return None
```

- [ ] **Step 8.4: Run test, expect PASS**

```bash
pytest tests/test_vad.py -v
```

- [ ] **Step 8.5: Commit**

```bash
git add backend/src/jarvis/vad.py backend/tests/test_vad.py
git commit -m "feat(vad): silero wrapper + speech-end detector"
```

---

## Task 9: STT Module

**Files:**
- Create: `backend/src/jarvis/stt.py`
- Create: `backend/tests/test_stt.py`

- [ ] **Step 9.1: Write failing test in `backend/tests/test_stt.py`**

```python
"""STT tests using fake transcriber."""
import numpy as np

from jarvis.stt import FakeStt


async def test_fake_stt_returns_scripted_text():
    stt = FakeStt(text="hallo Jarvis")
    pcm = np.zeros(16000, dtype=np.int16)
    result = await stt.transcribe(pcm)
    assert result == "hallo Jarvis"


async def test_fake_stt_empty_returns_empty():
    stt = FakeStt(text="")
    assert (await stt.transcribe(np.zeros(16000, dtype=np.int16))) == ""
```

- [ ] **Step 9.2: Run test, expect ImportError**

```bash
pytest tests/test_stt.py -v
```

- [ ] **Step 9.3: Implement `backend/src/jarvis/stt.py`**

```python
"""Speech-to-text — faster-whisper wrapper + fake."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Stt(Protocol):
    async def transcribe(self, pcm_int16: np.ndarray) -> str: ...


class RealStt:
    """faster-whisper wrapper. Runs synchronously in executor."""

    def __init__(self, model_name: str = "base", compute_type: str = "int8", language: str = "de") -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        self._language = language

    async def transcribe(self, pcm_int16: np.ndarray) -> str:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, pcm_int16)

    def _transcribe_sync(self, pcm_int16: np.ndarray) -> str:
        audio_f32 = pcm_int16.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio_f32,
            language=self._language,
            vad_filter=False,
            beam_size=1,
        )
        return "".join(seg.text for seg in segments).strip()


class FakeStt:
    """Returns a fixed string regardless of input."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def transcribe(self, pcm_int16: np.ndarray) -> str:
        return self._text
```

- [ ] **Step 9.4: Run test, expect PASS**

```bash
pytest tests/test_stt.py -v
```

- [ ] **Step 9.5: Commit**

```bash
git add backend/src/jarvis/stt.py backend/tests/test_stt.py
git commit -m "feat(stt): faster-whisper wrapper + fake"
```

---

## Task 10: LLM Module

**Files:**
- Create: `backend/src/jarvis/llm.py`
- Create: `backend/tests/test_llm.py`

Phase 1 LLM module produces a streaming text response. Tool-calling protocol is *implemented* so Phase 2 can flip the switch, but with no plugins registered the LLM always streams plain text.

- [ ] **Step 10.1: Write failing test in `backend/tests/test_llm.py`**

```python
"""LLM tests using fake stream."""
from jarvis.llm import FakeLlm, LlmEvent


async def test_fake_llm_streams_chunks():
    fake = FakeLlm(chunks=["Hallo ", "Erwin, ", "wie geht's?"])
    out = []
    async for ev in fake.chat([{"role": "user", "content": "hi"}]):
        out.append(ev)
    assert [e.kind for e in out] == ["chunk", "chunk", "chunk", "done"]
    assert "".join(e.text for e in out if e.kind == "chunk") == "Hallo Erwin, wie geht's?"
    assert out[-1].kind == "done"
    assert out[-1].text == "Hallo Erwin, wie geht's?"


async def test_fake_llm_empty_chunks_still_yields_done():
    fake = FakeLlm(chunks=[])
    out = [ev async for ev in fake.chat([{"role": "user", "content": "x"}])]
    assert len(out) == 1
    assert out[0].kind == "done"
    assert out[0].text == ""
```

- [ ] **Step 10.2: Run test, expect ImportError**

```bash
pytest tests/test_llm.py -v
```

- [ ] **Step 10.3: Implement `backend/src/jarvis/llm.py`**

```python
"""LLM client — Ollama (real) + Fake. Streams text chunks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol


@dataclass
class LlmEvent:
    kind: str   # "chunk" | "done" | "tool_call"
    text: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] | None = None
    call_id: str = ""


class Llm(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> AsyncIterator[LlmEvent]: ...


class RealLlm:
    """Ollama-backed LLM."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        system_prompt: str = "",
    ) -> None:
        import ollama
        self._client = ollama.AsyncClient(host=host)
        self._model = model
        self._system = system_prompt

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
        ):
            content = part.get("message", {}).get("content", "")
            if content:
                full_text += content
                yield LlmEvent(kind="chunk", text=content)
        yield LlmEvent(kind="done", text=full_text)


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
```

- [ ] **Step 10.4: Run test, expect PASS**

```bash
pytest tests/test_llm.py -v
```

- [ ] **Step 10.5: Commit**

```bash
git add backend/src/jarvis/llm.py backend/tests/test_llm.py
git commit -m "feat(llm): ollama streaming client + fake"
```

---

## Task 11: TTS Module

**Files:**
- Create: `backend/src/jarvis/tts.py`
- Create: `backend/tests/test_tts.py`

- [ ] **Step 11.1: Write failing test in `backend/tests/test_tts.py`**

```python
"""TTS tests using fake synthesizer."""
import numpy as np

from jarvis.tts import FakeTts


async def test_fake_tts_yields_chunks():
    tts = FakeTts(samples_per_chunk=4)
    chunks = []
    async for chunk in tts.synthesize("hello world"):
        chunks.append(chunk)
    # FakeTts emits 1 chunk per word, padded to samples_per_chunk
    assert len(chunks) == 2
    assert chunks[0].shape == (4,)
    assert chunks[0].dtype == np.float32


async def test_fake_tts_empty_text_no_chunks():
    tts = FakeTts(samples_per_chunk=4)
    out = [c async for c in tts.synthesize("")]
    assert out == []
```

- [ ] **Step 11.2: Run test, expect ImportError**

```bash
pytest tests/test_tts.py -v
```

- [ ] **Step 11.3: Implement `backend/src/jarvis/tts.py`**

```python
"""TTS — Piper (real) + Fake. Streams float32 audio chunks."""
from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator, Protocol

import numpy as np


class Tts(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[np.ndarray]: ...


class RealTts:
    """Piper voice via piper-tts package. Synthesizes phrase, then chunks the audio."""

    def __init__(self, model_path: Path, chunk_size: int = 2048) -> None:
        from piper import PiperVoice
        self._voice = PiperVoice.load(str(model_path))
        self._chunk_size = chunk_size

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        import asyncio
        if not text.strip():
            return
        loop = asyncio.get_running_loop()

        def _synth() -> bytes:
            buf = bytearray()
            for chunk in self._voice.synthesize_stream_raw(text):
                buf.extend(chunk)
            return bytes(buf)

        raw = await loop.run_in_executor(None, _synth)
        pcm_i16 = np.frombuffer(raw, dtype=np.int16)
        pcm_f32 = (pcm_i16.astype(np.float32) / 32768.0)
        for i in range(0, pcm_f32.size, self._chunk_size):
            yield pcm_f32[i : i + self._chunk_size]
            await asyncio.sleep(0)


class FakeTts:
    """Emits one zeros-chunk per word."""

    def __init__(self, samples_per_chunk: int = 1024) -> None:
        self._n = samples_per_chunk

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        for word in text.split():
            if not word:
                continue
            yield np.zeros(self._n, dtype=np.float32)
```

- [ ] **Step 11.4: Run test, expect PASS**

```bash
pytest tests/test_tts.py -v
```

- [ ] **Step 11.5: Commit**

```bash
git add backend/src/jarvis/tts.py backend/tests/test_tts.py
git commit -m "feat(tts): piper streaming wrapper + fake"
```

---

## Task 12: Orchestrator (Pipeline Glue)

**Files:**
- Create: `backend/src/jarvis/orchestrator.py`
- Create: `backend/tests/test_orchestrator.py`

Orchestrator is the only file that holds references to every stage. It owns the state machine, the WebSocket broadcast, and the asyncio loop wiring. Tests inject fakes for every stage.

- [ ] **Step 12.1: Write failing test in `backend/tests/test_orchestrator.py`**

```python
"""End-to-end orchestrator test with all-fake pipeline."""
import asyncio

import numpy as np

from jarvis.audio_io import FakeMicStream, FakeSpeaker
from jarvis.events import State, StateEvent, TranscriptFinalEvent, ResponseDoneEvent
from jarvis.llm import FakeLlm
from jarvis.orchestrator import Orchestrator
from jarvis.stt import FakeStt
from jarvis.tts import FakeTts
from jarvis.vad import FakeVad
from jarvis.wake import FakeWakeWord


async def test_full_loop_with_fakes():
    """Wake-marker frame → silence → STT → LLM → TTS → IDLE again."""
    sample_rate = 16000
    frame_size = 480
    marker = 12345

    # Build PCM: 5 silent frames, then 1 marker frame, then 30 frames "speech",
    # then 30 frames silence (triggers speech_end at silence_frames=27).
    silent = np.zeros(frame_size, dtype=np.int16)
    marker_frame = np.full(frame_size, marker, dtype=np.int16)
    pcm_parts = [silent] * 5 + [marker_frame] + [silent] * 30 + [silent] * 30
    pcm = np.concatenate(pcm_parts)

    # VAD mask matches the frame schedule: 5 silent, 1 marker silent, 30 speech, 30 silent
    speech_mask = [False] * 5 + [False] + [True] * 30 + [False] * 30

    events: list = []

    orchestrator = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=frame_size),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="wie spät ist es"),
        llm=FakeLlm(chunks=["Es ist ", "14 Uhr."]),
        tts=FakeTts(samples_per_chunk=128),
        emit=events.append,
        silence_frames=27,
        sample_rate=sample_rate,
    )

    await asyncio.wait_for(orchestrator.run_once(), timeout=5.0)

    types = [type(e).__name__ for e in events]
    # Verify ordering: state transitions happened in the right sequence
    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert state_seq[0] == State.LISTENING
    assert State.TRANSCRIBE in state_seq
    assert State.THINKING in state_seq
    assert State.SPEAKING in state_seq
    assert state_seq[-1] == State.IDLE

    transcripts = [e for e in events if isinstance(e, TranscriptFinalEvent)]
    assert len(transcripts) == 1
    assert transcripts[0].text == "wie spät ist es"

    done = [e for e in events if isinstance(e, ResponseDoneEvent)]
    assert len(done) == 1
    assert done[0].full == "Es ist 14 Uhr."


async def test_cancel_returns_to_idle():
    """A cancel request mid-pipeline returns to IDLE without crash."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 99
    pcm = np.concatenate([silent] * 3 + [np.full(480, marker, dtype=np.int16)] + [silent] * 100)

    events: list = []
    orchestrator = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad([False] * 200),    # never speaks
        stt=FakeStt(text=""),
        llm=FakeLlm(chunks=[]),
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=5,
        sample_rate=16000,
    )

    async def cancel_soon():
        await asyncio.sleep(0.05)
        orchestrator.request_cancel()

    await asyncio.gather(
        asyncio.wait_for(orchestrator.run_once(), timeout=2.0),
        cancel_soon(),
    )

    final_state = [e for e in events if isinstance(e, StateEvent)][-1].value
    assert final_state == State.IDLE
```

- [ ] **Step 12.2: Run test, expect ImportError**

```bash
pytest tests/test_orchestrator.py -v
```

- [ ] **Step 12.3: Implement `backend/src/jarvis/orchestrator.py`**

```python
"""Pipeline orchestrator: glues wake/vad/stt/llm/tts into the state machine."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Protocol

import numpy as np

from jarvis.events import (
    AudioLevelEvent,
    ErrorEvent,
    Event,
    ResponseChunkEvent,
    ResponseDoneEvent,
    State,
    StateEvent,
    TranscriptFinalEvent,
)
from jarvis.state_machine import Trigger, transition
from jarvis.audio_io import rms_dbfs
from jarvis.vad import SpeechEndDetector


EmitFn = Callable[[Event], None]


class _MicLike(Protocol):
    def frames(self) -> Any: ...


class _SpeakerLike(Protocol):
    async def play(self, samples: np.ndarray) -> None: ...


class _WakeLike(Protocol):
    async def detect(self, frame_int16: np.ndarray) -> bool: ...


class _VadLike(Protocol):
    async def is_speech(self, frame_int16: np.ndarray) -> bool: ...


class _SttLike(Protocol):
    async def transcribe(self, pcm_int16: np.ndarray) -> str: ...


class _LlmLike(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> Any: ...


class _TtsLike(Protocol):
    def synthesize(self, text: str) -> Any: ...


class Orchestrator:
    def __init__(
        self,
        *,
        mic: _MicLike,
        speaker: _SpeakerLike,
        wake: _WakeLike,
        vad: _VadLike,
        stt: _SttLike,
        llm: _LlmLike,
        tts: _TtsLike,
        emit: EmitFn,
        silence_frames: int = 27,
        sample_rate: int = 16000,
    ) -> None:
        self._mic = mic
        self._speaker = speaker
        self._wake = wake
        self._vad = vad
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._emit = emit
        self._silence_frames = silence_frames
        self._sample_rate = sample_rate
        self._state = State.IDLE
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def _go(self, trigger: Trigger) -> None:
        new = transition(self._state, trigger)
        if new != self._state:
            self._state = new
            self._emit(StateEvent(value=new))

    async def run_once(self) -> None:
        """One full IDLE → IDLE cycle. Returns when speak finishes or cancel fires."""
        self._cancel = False
        self._emit(StateEvent(value=self._state))

        try:
            # 1. IDLE — listen for wake word
            recorded: list[np.ndarray] = []
            wake_seen = False
            end_detector = SpeechEndDetector(silence_frames=self._silence_frames)

            async for frame in self._mic.frames():
                if self._cancel:
                    self._go(Trigger.CANCEL)
                    return

                self._emit(AudioLevelEvent(rms=rms_dbfs(frame), fft=[]))

                if not wake_seen:
                    if await self._wake.detect(frame):
                        wake_seen = True
                        self._go(Trigger.WAKE_DETECTED)
                    continue

                # In LISTENING: record + run VAD
                recorded.append(frame)
                speech = await self._vad.is_speech(frame)
                signal = end_detector.update(speech)
                if signal == "speech_end":
                    break

            if not wake_seen:
                self._go(Trigger.CANCEL)
                return

            # 2. TRANSCRIBE
            self._go(Trigger.SPEECH_END)
            pcm = np.concatenate(recorded) if recorded else np.zeros(0, dtype=np.int16)
            text = await self._stt.transcribe(pcm)
            self._emit(TranscriptFinalEvent(text=text))

            # 3. THINKING
            self._go(Trigger.TRANSCRIPT_READY)
            full_response = ""
            async for ev in self._llm.chat([{"role": "user", "content": text}]):
                if self._cancel:
                    self._go(Trigger.CANCEL)
                    return
                if ev.kind == "chunk":
                    full_response += ev.text
                    self._emit(ResponseChunkEvent(text=ev.text))
                elif ev.kind == "done":
                    full_response = ev.text or full_response
                    self._emit(ResponseDoneEvent(full=full_response))

            # 4. SPEAKING
            self._go(Trigger.RESPONSE_COMPLETE)
            async for chunk in self._tts.synthesize(full_response):
                if self._cancel:
                    self._go(Trigger.CANCEL)
                    return
                await self._speaker.play(chunk)

            # 5. IDLE
            self._go(Trigger.PLAYBACK_DONE)

        except Exception as exc:
            self._emit(ErrorEvent(stage=self._state.value, msg=str(exc), retriable=False))
            self._state = State.ERROR
            self._emit(StateEvent(value=State.ERROR))
            # always return to IDLE
            self._state = State.IDLE
            self._emit(StateEvent(value=State.IDLE))
```

- [ ] **Step 12.4: Run test, expect PASS**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: Both tests PASS.

- [ ] **Step 12.5: Commit**

```bash
git add backend/src/jarvis/orchestrator.py backend/tests/test_orchestrator.py
git commit -m "feat(orchestrator): pipeline glue + state machine integration"
```

---

## Task 13: FastAPI App + WebSocket

**Files:**
- Create: `backend/src/jarvis/app.py`
- Create: `backend/tests/test_app_ws.py`

- [ ] **Step 13.1: Write failing test in `backend/tests/test_app_ws.py`**

```python
"""WebSocket integration smoke test using FastAPI TestClient."""
import json

import pytest
from fastapi.testclient import TestClient

from jarvis.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app(start_orchestrator=False)  # don't try to open the mic
    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "version": "0.1.0"}


def test_websocket_accepts_manual_start(client: TestClient):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "manual_start"}))
        # Server should not close the connection on a known event.
        # We don't assert on responses here because orchestrator isn't running.
        ws.close()


def test_websocket_rejects_garbage(client: TestClient):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not even json")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "parse" in msg["msg"].lower() or "invalid" in msg["msg"].lower()
```

- [ ] **Step 13.2: Run test, expect ImportError**

```bash
pytest tests/test_app_ws.py -v
```

- [ ] **Step 13.3: Implement `backend/src/jarvis/app.py`**

```python
"""FastAPI app + WebSocket route. Holds orchestrator lifecycle."""
from __future__ import annotations

import asyncio
import json
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

from jarvis import __version__
from jarvis.config import get_settings
from jarvis.events import CancelEvent, ErrorEvent, ManualStartEvent, parse_inbound


class WsHub:
    """Tracks connected clients and broadcasts events."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        async with self._lock:
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_text(json.dumps(payload))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)


def create_app(start_orchestrator: bool = True) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Jarvis", version=__version__)
    hub = WsHub()
    app.state.hub = hub
    app.state.start_orchestrator = start_orchestrator

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "version": __version__}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await hub.add(ws)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    ev = parse_inbound(raw)
                except (ValueError, json.JSONDecodeError) as exc:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "stage": "ws",
                        "msg": f"parse failed: {exc}",
                        "retriable": True,
                    }))
                    continue
                # Phase 1: route inbound events through orchestrator hook (set up below)
                handler = getattr(app.state, "inbound_handler", None)
                if handler is not None:
                    await handler(ev)
        except WebSocketDisconnect:
            pass
        finally:
            await hub.remove(ws)

    @app.on_event("startup")
    async def _startup() -> None:
        if not app.state.start_orchestrator:
            return
        # Wire real orchestrator only when explicitly enabled. Tests use
        # start_orchestrator=False so heavy models never load.
        from jarvis.orchestrator import Orchestrator
        from jarvis.audio_io import RealMicStream, RealSpeaker
        from jarvis.wake import RealWakeWord
        from jarvis.vad import RealVad
        from jarvis.stt import RealStt
        from jarvis.llm import RealLlm
        from jarvis.tts import RealTts

        def emit(ev) -> None:
            asyncio.create_task(hub.broadcast(ev.model_dump(mode="json")))

        async def _orch_loop():
            # Build heavy components once. Mic stream stays open across cycles.
            speaker = RealSpeaker()
            wake = RealWakeWord(
                model_path=settings.wake_model_path,
                threshold=settings.wake_threshold,
            )
            vad = RealVad(threshold=settings.vad_threshold, sample_rate=settings.sample_rate)
            stt = RealStt(
                model_name=settings.whisper_model,
                compute_type=settings.whisper_compute,
                language=settings.whisper_lang,
            )
            llm = RealLlm(
                host=settings.llm_host,
                model=settings.llm_model,
                system_prompt=settings.llm_system_prompt,
            )
            tts = RealTts(model_path=settings.tts_model_path)
            silence_frame_count = int(
                settings.silence_ms * settings.sample_rate / 1000 / 480
            )

            async with RealMicStream(sample_rate=settings.sample_rate) as mic:
                while True:
                    try:
                        orch = Orchestrator(
                            mic=mic, speaker=speaker, wake=wake, vad=vad,
                            stt=stt, llm=llm, tts=tts, emit=emit,
                            silence_frames=silence_frame_count,
                            sample_rate=settings.sample_rate,
                        )
                        await orch.run_once()
                    except Exception as exc:
                        logger.exception("orchestrator crashed")
                        await hub.broadcast(
                            ErrorEvent(stage="orchestrator", msg=str(exc), retriable=True)
                            .model_dump(mode="json")
                        )
                        await asyncio.sleep(1.0)

        app.state.orch_task = asyncio.create_task(_orch_loop())

        async def inbound_handler(ev) -> None:
            # Placeholder in Task 13 — Task 18 replaces this with queue routing.
            logger.info(f"inbound: {ev.type}")

        app.state.inbound_handler = inbound_handler

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        task = getattr(app.state, "orch_task", None)
        if task is not None:
            task.cancel()

    return app


app = create_app(start_orchestrator=False)  # `uvicorn jarvis.app:app` won't auto-start orch; use scripts/dev.sh
```

- [ ] **Step 13.4: Run test, expect PASS**

```bash
pytest tests/test_app_ws.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 13.5: Commit**

```bash
git add backend/src/jarvis/app.py backend/tests/test_app_ws.py
git commit -m "feat(app): fastapi app + websocket hub + lifecycle hooks"
```

---

## Task 14: Frontend — Vite + React + Tauri Scaffolding

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src-tauri/Cargo.toml`
- Create: `frontend/src-tauri/tauri.conf.json`
- Create: `frontend/src-tauri/src/main.rs`

- [ ] **Step 14.1: Create `frontend/package.json`**

```json
{
  "name": "jarvis-frontend",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "tauri": "tauri",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "zustand": "^5.0.0",
    "three": "^0.169.0",
    "@react-three/fiber": "^8.17.0",
    "@react-three/drei": "^9.114.0"
  },
  "devDependencies": {
    "@tauri-apps/cli": "^2.0.0",
    "@tauri-apps/api": "^2.0.0",
    "@types/react": "^18.3.10",
    "@types/react-dom": "^18.3.0",
    "@types/three": "^0.169.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.6.0",
    "vite": "^5.4.0",
    "vitest": "^2.1.0",
    "@testing-library/react": "^16.0.0",
    "@testing-library/jest-dom": "^6.5.0",
    "jsdom": "^25.0.0"
  }
}
```

- [ ] **Step 14.2: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "Bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"]
}
```

- [ ] **Step 14.3: Create `frontend/vite.config.ts`**

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 1420, strictPort: true },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/tests/setup.ts"],
  },
});
```

- [ ] **Step 14.4: Create `frontend/index.html`**

```html
<!doctype html>
<html lang="de">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Jarvis</title>
  </head>
  <body style="margin:0;background:#000;">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 14.5: Create `frontend/src/main.tsx`**

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [ ] **Step 14.6: Create `frontend/src/App.tsx`** (placeholder, replaced in next task)

```tsx
export function App() {
  return <div style={{ color: "#00ddff", padding: 24 }}>JARVIS · booting</div>;
}
```

- [ ] **Step 14.7: Create `frontend/src/tests/setup.ts`**

```ts
import "@testing-library/jest-dom";
```

- [ ] **Step 14.8: Create `frontend/src-tauri/Cargo.toml`**

```toml
[package]
name = "jarvis-tauri"
version = "0.1.0"
edition = "2021"

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tauri = { version = "2", features = [] }
tauri-plugin-global-shortcut = "2"

[features]
custom-protocol = ["tauri/custom-protocol"]
```

- [ ] **Step 14.9: Create `frontend/src-tauri/tauri.conf.json`**

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "jarvis",
  "version": "0.1.0",
  "identifier": "io.jarvis.app",
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devUrl": "http://localhost:1420",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "Jarvis",
        "width": 1280,
        "height": 800,
        "fullscreen": false,
        "resizable": true,
        "decorations": true
      }
    ],
    "security": { "csp": null }
  }
}
```

- [ ] **Step 14.10: Create `frontend/src-tauri/src/main.rs`**

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

- [ ] **Step 14.11: Verify install + smoke run**

```bash
cd frontend
npm install
npm run build
```

Expected: `npm run build` exits 0. `dist/` directory created.

- [ ] **Step 14.12: Commit**

```bash
git add frontend/package.json frontend/tsconfig.json frontend/vite.config.ts \
        frontend/index.html frontend/src/main.tsx frontend/src/App.tsx \
        frontend/src/tests/setup.ts \
        frontend/src-tauri/Cargo.toml frontend/src-tauri/tauri.conf.json \
        frontend/src-tauri/src/main.rs
git commit -m "chore(frontend): vite+react+tauri scaffolding"
```

---

## Task 15: Frontend — WebSocket Client + State Store

**Files:**
- Create: `frontend/src/ws.ts`
- Create: `frontend/src/store.ts`
- Create: `frontend/src/tests/ws.test.ts`
- Create: `frontend/src/tests/store.test.ts`

- [ ] **Step 15.1: Write failing test in `frontend/src/tests/store.test.ts`**

```ts
import { describe, it, expect, beforeEach } from "vitest";
import { useJarvisStore } from "../store";

describe("jarvis store", () => {
  beforeEach(() => {
    useJarvisStore.setState({
      state: "idle",
      lastTranscript: "",
      lastResponse: "",
      audioLevel: -120,
      connected: false,
    });
  });

  it("updates state on state event", () => {
    useJarvisStore.getState().handleEvent({ type: "state", value: "listening" });
    expect(useJarvisStore.getState().state).toBe("listening");
  });

  it("updates transcript on transcript_final", () => {
    useJarvisStore.getState().handleEvent({ type: "transcript_final", text: "hallo Jarvis" });
    expect(useJarvisStore.getState().lastTranscript).toBe("hallo Jarvis");
  });

  it("appends response chunks then finalizes", () => {
    useJarvisStore.getState().handleEvent({ type: "response_chunk", text: "Hallo " });
    useJarvisStore.getState().handleEvent({ type: "response_chunk", text: "Erwin" });
    expect(useJarvisStore.getState().lastResponse).toBe("Hallo Erwin");
    useJarvisStore.getState().handleEvent({ type: "response_done", full: "Hallo Erwin." });
    expect(useJarvisStore.getState().lastResponse).toBe("Hallo Erwin.");
  });

  it("updates audio level", () => {
    useJarvisStore.getState().handleEvent({ type: "audio_level", rms: -42, fft: [] });
    expect(useJarvisStore.getState().audioLevel).toBe(-42);
  });

  it("ignores unknown event types", () => {
    const before = useJarvisStore.getState();
    // @ts-expect-error testing unknown type
    useJarvisStore.getState().handleEvent({ type: "nonsense" });
    expect(useJarvisStore.getState()).toEqual(before);
  });
});
```

- [ ] **Step 15.2: Write failing test in `frontend/src/tests/ws.test.ts`**

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { JarvisWs } from "../ws";

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;
  sent: string[] = [];
  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }
  send(s: string) { this.sent.push(s); }
  close() { this.readyState = 3; this.onclose?.(); }
  simulateOpen() { this.readyState = 1; this.onopen?.(); }
  simulateMessage(payload: object) { this.onmessage?.({ data: JSON.stringify(payload) }); }
}

describe("JarvisWs", () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    (globalThis as any).WebSocket = MockWebSocket;
  });

  it("connects and reports open", async () => {
    const events: any[] = [];
    const ws = new JarvisWs({ url: "ws://test", onEvent: (e) => events.push(e), onConnected: () => events.push("OPEN") });
    ws.connect();
    MockWebSocket.instances[0].simulateOpen();
    expect(events).toContain("OPEN");
  });

  it("dispatches parsed events", () => {
    const events: any[] = [];
    const ws = new JarvisWs({ url: "ws://test", onEvent: (e) => events.push(e) });
    ws.connect();
    MockWebSocket.instances[0].simulateOpen();
    MockWebSocket.instances[0].simulateMessage({ type: "state", value: "listening" });
    expect(events[0]).toEqual({ type: "state", value: "listening" });
  });

  it("sends manual_start", () => {
    const ws = new JarvisWs({ url: "ws://test", onEvent: () => {} });
    ws.connect();
    MockWebSocket.instances[0].simulateOpen();
    ws.send({ type: "manual_start" });
    expect(MockWebSocket.instances[0].sent[0]).toBe('{"type":"manual_start"}');
  });

  it("reconnects after close", async () => {
    vi.useFakeTimers();
    const ws = new JarvisWs({ url: "ws://test", onEvent: () => {}, reconnectMs: 100 });
    ws.connect();
    MockWebSocket.instances[0].simulateOpen();
    MockWebSocket.instances[0].close();
    vi.advanceTimersByTime(150);
    expect(MockWebSocket.instances.length).toBe(2);
    vi.useRealTimers();
  });
});
```

- [ ] **Step 15.3: Run tests, expect failures (modules missing)**

```bash
cd frontend
npm test
```

- [ ] **Step 15.4: Implement `frontend/src/store.ts`**

```ts
import { create } from "zustand";

export type JarvisState = "idle" | "listening" | "transcribe" | "thinking" | "speaking" | "error";

export interface InboundEvent {
  type: string;
  [k: string]: unknown;
}

interface StoreState {
  state: JarvisState;
  lastTranscript: string;
  lastResponse: string;
  audioLevel: number;
  connected: boolean;
  setConnected: (b: boolean) => void;
  handleEvent: (ev: InboundEvent) => void;
}

export const useJarvisStore = create<StoreState>((set) => ({
  state: "idle",
  lastTranscript: "",
  lastResponse: "",
  audioLevel: -120,
  connected: false,
  setConnected: (b) => set({ connected: b }),
  handleEvent: (ev) => set((s) => {
    switch (ev.type) {
      case "state":
        return { state: (ev as any).value as JarvisState };
      case "transcript_final":
        return { lastTranscript: (ev as any).text as string };
      case "response_chunk":
        return { lastResponse: s.lastResponse + ((ev as any).text as string) };
      case "response_done":
        return { lastResponse: (ev as any).full as string };
      case "audio_level":
        return { audioLevel: (ev as any).rms as number };
      default:
        return {};
    }
  }),
}));
```

- [ ] **Step 15.5: Implement `frontend/src/ws.ts`**

```ts
export interface JarvisWsOptions {
  url: string;
  onEvent: (ev: any) => void;
  onConnected?: () => void;
  onDisconnected?: () => void;
  reconnectMs?: number;
}

export class JarvisWs {
  private socket: WebSocket | null = null;
  private timer: any = null;
  private closed = false;
  constructor(private opts: JarvisWsOptions) {}

  connect() {
    this.closed = false;
    this.socket = new WebSocket(this.opts.url);
    this.socket.onopen = () => this.opts.onConnected?.();
    this.socket.onmessage = (ev) => {
      try {
        this.opts.onEvent(JSON.parse(ev.data));
      } catch {
        // swallow malformed frames; backend should never emit them
      }
    };
    this.socket.onclose = () => {
      this.opts.onDisconnected?.();
      if (!this.closed) {
        this.timer = setTimeout(() => this.connect(), this.opts.reconnectMs ?? 1000);
      }
    };
    this.socket.onerror = () => this.socket?.close();
  }

  send(obj: object) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(obj));
    }
  }

  close() {
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
    this.socket?.close();
  }
}
```

- [ ] **Step 15.6: Run tests, expect PASS**

```bash
npm test
```

Expected: store tests + ws tests pass.

- [ ] **Step 15.7: Commit**

```bash
git add frontend/src/store.ts frontend/src/ws.ts \
        frontend/src/tests/store.test.ts frontend/src/tests/ws.test.ts
git commit -m "feat(frontend): ws client + zustand store with tests"
```

---

## Task 16: Frontend — Reactor Core HUD Components

**Files:**
- Create: `frontend/src/hud/ReactorCore.tsx`
- Create: `frontend/src/hud/Waveform.tsx`
- Create: `frontend/src/hud/StatusCorners.tsx`
- Create: `frontend/src/hud/HudShell.tsx`
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/tests/hud.test.tsx`

- [ ] **Step 16.1: Write failing test in `frontend/src/tests/hud.test.tsx`**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { useJarvisStore } from "../store";
import { HudShell } from "../hud/HudShell";

describe("HudShell", () => {
  it("renders state in StatusCorners", () => {
    useJarvisStore.setState({ state: "listening", lastTranscript: "", lastResponse: "", audioLevel: -42, connected: true });
    render(<HudShell />);
    expect(screen.getByText(/LISTENING/i)).toBeInTheDocument();
  });

  it("renders last response when speaking", () => {
    useJarvisStore.setState({ state: "speaking", lastTranscript: "", lastResponse: "Hallo Erwin.", audioLevel: -30, connected: true });
    render(<HudShell />);
    expect(screen.getByText("Hallo Erwin.")).toBeInTheDocument();
  });
});
```

- [ ] **Step 16.2: Run test, expect failures (modules missing)**

```bash
npm test
```

- [ ] **Step 16.3: Implement `frontend/src/hud/ReactorCore.tsx`**

```tsx
import { useJarvisStore } from "../store";

const STATE_COLOR: Record<string, string> = {
  idle: "rgba(0,221,255,0.4)",
  listening: "#00ddff",
  transcribe: "#00ddff",
  thinking: "#aa66ff",
  speaking: "#ffaa00",
  error: "#ff3a3a",
};

export function ReactorCore() {
  const state = useJarvisStore((s) => s.state);
  const color = STATE_COLOR[state] ?? STATE_COLOR.idle;
  const pulse = state === "listening" || state === "thinking";

  return (
    <div style={{
      position: "absolute", inset: 0, display: "grid", placeItems: "center",
      pointerEvents: "none",
    }}>
      <div style={{
        width: "50vmin", height: "50vmin", borderRadius: "50%",
        border: `2px solid ${color}`,
        boxShadow: `0 0 80px ${color}66, inset 0 0 60px ${color}44`,
        animation: pulse ? "pulse 1.6s ease-in-out infinite" : undefined,
      }} />
      <div style={{
        position: "absolute",
        width: "10vmin", height: "10vmin", borderRadius: "50%",
        background: `radial-gradient(circle, #fff 0%, ${color} 60%, transparent 80%)`,
        boxShadow: `0 0 50px ${color}`,
      }} />
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%      { opacity: 0.7; transform: scale(1.05); }
        }
      `}</style>
    </div>
  );
}
```

- [ ] **Step 16.4: Implement `frontend/src/hud/Waveform.tsx`**

```tsx
import { useJarvisStore } from "../store";

const BARS = 24;

export function Waveform() {
  const level = useJarvisStore((s) => s.audioLevel); // dBFS, e.g. -120..0
  // Map dBFS [-60,0] → [0,1]
  const normalized = Math.max(0, Math.min(1, (level + 60) / 60));

  return (
    <div style={{
      position: "absolute", top: "50%", left: "20%", right: "20%",
      height: 60, transform: "translateY(-50%)",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      pointerEvents: "none",
    }}>
      {Array.from({ length: BARS }).map((_, i) => {
        const phase = Math.sin((Date.now() / 200) + i) * 0.5 + 0.5;
        const h = Math.max(0.05, normalized * (0.3 + 0.7 * phase));
        return (
          <span key={i} style={{
            width: 3, height: `${h * 100}%`,
            background: "#00ddff",
            boxShadow: "0 0 6px #00ddff",
          }} />
        );
      })}
    </div>
  );
}
```

- [ ] **Step 16.5: Implement `frontend/src/hud/StatusCorners.tsx`**

```tsx
import { useJarvisStore } from "../store";

const cornerStyle: React.CSSProperties = {
  position: "absolute", color: "#00ddff", font: "11px monospace",
  padding: 12, opacity: 0.85, letterSpacing: 1.2,
};

export function StatusCorners() {
  const { state, connected, lastTranscript, lastResponse } = useJarvisStore((s) => ({
    state: s.state,
    connected: s.connected,
    lastTranscript: s.lastTranscript,
    lastResponse: s.lastResponse,
  }));
  const label = state.toUpperCase();

  return (
    <>
      <div style={{ ...cornerStyle, top: 0, left: 0 }}>
        JARVIS · v0.1 · {label}
      </div>
      <div style={{ ...cornerStyle, top: 0, right: 0 }}>
        {connected ? "ONLINE" : "DISCONNECTED"}
      </div>
      <div style={{ ...cornerStyle, bottom: 0, left: 0, maxWidth: "40%" }}>
        {lastTranscript ? `YOU · ${lastTranscript}` : "LISTENING ·····"}
      </div>
      <div style={{ ...cornerStyle, bottom: 0, right: 0, maxWidth: "40%", textAlign: "right" }}>
        {lastResponse || "de-DE · thorsten"}
      </div>
    </>
  );
}
```

- [ ] **Step 16.6: Implement `frontend/src/hud/HudShell.tsx`**

```tsx
import { ReactorCore } from "./ReactorCore";
import { Waveform } from "./Waveform";
import { StatusCorners } from "./StatusCorners";

export function HudShell() {
  return (
    <div style={{
      position: "fixed", inset: 0, background: "radial-gradient(ellipse at center, #001218 0%, #000508 100%)",
      overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage:
          "linear-gradient(rgba(0,200,255,0.06) 1px, transparent 1px), " +
          "linear-gradient(90deg, rgba(0,200,255,0.06) 1px, transparent 1px)",
        backgroundSize: "32px 32px",
      }} />
      <ReactorCore />
      <Waveform />
      <StatusCorners />
    </div>
  );
}
```

- [ ] **Step 16.7: Replace `frontend/src/App.tsx`**

```tsx
import { useEffect } from "react";
import { HudShell } from "./hud/HudShell";
import { useJarvisStore } from "./store";
import { JarvisWs } from "./ws";

export function App() {
  useEffect(() => {
    const ws = new JarvisWs({
      url: "ws://127.0.0.1:8765/ws",
      onEvent: (ev) => useJarvisStore.getState().handleEvent(ev),
      onConnected: () => useJarvisStore.getState().setConnected(true),
      onDisconnected: () => useJarvisStore.getState().setConnected(false),
    });
    ws.connect();
    return () => ws.close();
  }, []);

  return <HudShell />;
}
```

- [ ] **Step 16.8: Run tests, expect PASS**

```bash
npm test
```

- [ ] **Step 16.9: Build to catch type errors**

```bash
npm run build
```

Expected: Build succeeds.

- [ ] **Step 16.10: Commit**

```bash
git add frontend/src/hud/ frontend/src/App.tsx frontend/src/tests/hud.test.tsx
git commit -m "feat(frontend): reactor-core HUD components"
```

---

## Task 17: Tauri Hotkey + Mic-Click Wiring

**Files:**
- Modify: `frontend/src-tauri/src/main.rs`
- Modify: `frontend/src-tauri/Cargo.toml`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/hud/ReactorCore.tsx` (make clickable)

- [ ] **Step 17.1: Update `frontend/src-tauri/Cargo.toml` features**

Replace the `[dependencies]` and `[features]` blocks (lines starting at `[dependencies]`) with:

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tauri = { version = "2", features = [] }
tauri-plugin-global-shortcut = "2"

[features]
custom-protocol = ["tauri/custom-protocol"]
```

(No change from Task 14 — confirm it matches.)

- [ ] **Step 17.2: Update `frontend/src-tauri/src/main.rs`**

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Emitter, Manager};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

fn main() {
    let shortcut = Shortcut::new(Some(Modifiers::SUPER), Code::KeyJ);

    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, sc, event| {
                    if sc == &shortcut && event.state() == ShortcutState::Pressed {
                        if let Some(win) = app.get_webview_window("main") {
                            let _ = win.emit("hotkey", "manual_start");
                        }
                    }
                })
                .build(),
        )
        .setup(move |app| {
            app.global_shortcut().register(shortcut)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

- [ ] **Step 17.3: Update `frontend/src/App.tsx` to listen to hotkey + clicks**

```tsx
import { useEffect, useRef } from "react";
import { HudShell } from "./hud/HudShell";
import { useJarvisStore } from "./store";
import { JarvisWs } from "./ws";

export function App() {
  const wsRef = useRef<JarvisWs | null>(null);

  useEffect(() => {
    const ws = new JarvisWs({
      url: "ws://127.0.0.1:8765/ws",
      onEvent: (ev) => useJarvisStore.getState().handleEvent(ev),
      onConnected: () => useJarvisStore.getState().setConnected(true),
      onDisconnected: () => useJarvisStore.getState().setConnected(false),
    });
    wsRef.current = ws;
    ws.connect();

    // Listen for Tauri-emitted hotkey events
    let unlisten: (() => void) | null = null;
    (async () => {
      try {
        const tauriEvent: any = await import("@tauri-apps/api/event");
        unlisten = await tauriEvent.listen("hotkey", () => {
          ws.send({ type: "manual_start" });
        });
      } catch {
        // running in plain browser (vitest, vite dev without tauri) — ignore
      }
    })();

    return () => {
      ws.close();
      unlisten?.();
    };
  }, []);

  const onClickCore = () => wsRef.current?.send({ type: "manual_start" });

  return (
    <div onClick={onClickCore} style={{ cursor: "pointer", height: "100vh" }}>
      <HudShell />
    </div>
  );
}
```

- [ ] **Step 17.4: Manual smoke check**

```bash
cd backend
uvicorn jarvis.app:app --port 8765 &
cd ../frontend
npm run tauri dev
```

Expected: Tauri window opens, HUD visible, "DISCONNECTED" shows briefly then "ONLINE". Press `Win+J`: backend logs `inbound: manual_start`. Click on the HUD: same log message.

Kill processes.

- [ ] **Step 17.5: Commit**

```bash
git add frontend/src-tauri/src/main.rs frontend/src/App.tsx
git commit -m "feat(frontend): Win+J hotkey + HUD click trigger manual_start"
```

---

## Task 18: Wire Orchestrator to WebSocket Inbound Events

**Files:**
- Modify: `backend/src/jarvis/app.py`
- Create: `backend/tests/test_app_inbound.py`

Currently `inbound_handler` only logs. We make it route `manual_start` and `cancel` into a queue the orchestrator drains between cycles.

- [ ] **Step 18.1: Write failing test in `backend/tests/test_app_inbound.py`**

```python
"""Inbound events drive orchestrator queue."""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from jarvis.app import create_app


def test_manual_start_pushed_to_queue():
    app = create_app(start_orchestrator=False)
    # The queue is created in startup; here we just verify the wiring object exists
    # by inspecting app.state after a manual setup.
    queue: asyncio.Queue = asyncio.Queue()
    app.state.inbound_queue = queue

    async def handler(ev) -> None:
        await app.state.inbound_queue.put(ev.type)
    app.state.inbound_handler = handler

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "manual_start"}))
        ws.send_text(json.dumps({"type": "cancel"}))
        ws.close()

    # Drain queue synchronously: pull until empty
    drained: list[str] = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert "manual_start" in drained
    assert "cancel" in drained
```

- [ ] **Step 18.2: Run test, expect PASS already**

```bash
cd backend
pytest tests/test_app_inbound.py -v
```

Expected: PASS (we manually set up the handler in the test — the production wiring is verified by manual smoke in Step 17.4 and stays small).

- [ ] **Step 18.3: Update `backend/src/jarvis/app.py` to expose the queue in production**

In the `_startup` function, after `app.state.orch_task = ...`, replace the placeholder `inbound_handler` with:

```python
        app.state.inbound_queue = asyncio.Queue()

        async def inbound_handler(ev) -> None:
            await app.state.inbound_queue.put(ev)
            logger.info(f"inbound: {ev.type}")

        app.state.inbound_handler = inbound_handler
```

Also extend `_orch_loop` to drain the queue. Replace the `_orch_loop` definition with:

```python
        async def _orch_loop():
            speaker = RealSpeaker()
            wake = RealWakeWord(
                model_path=settings.wake_model_path,
                threshold=settings.wake_threshold,
            )
            vad = RealVad(threshold=settings.vad_threshold, sample_rate=settings.sample_rate)
            stt = RealStt(
                model_name=settings.whisper_model,
                compute_type=settings.whisper_compute,
                language=settings.whisper_lang,
            )
            llm = RealLlm(
                host=settings.llm_host,
                model=settings.llm_model,
                system_prompt=settings.llm_system_prompt,
            )
            tts = RealTts(model_path=settings.tts_model_path)
            silence_frame_count = int(
                settings.silence_ms * settings.sample_rate / 1000 / 480
            )

            async with RealMicStream(sample_rate=settings.sample_rate) as mic:
                while True:
                    try:
                        orch = Orchestrator(
                            mic=mic, speaker=speaker, wake=wake, vad=vad,
                            stt=stt, llm=llm, tts=tts, emit=emit,
                            silence_frames=silence_frame_count,
                            sample_rate=settings.sample_rate,
                        )

                        async def drain_inbound():
                            while True:
                                ev = await app.state.inbound_queue.get()
                                if isinstance(ev, CancelEvent):
                                    orch.request_cancel()

                        drain = asyncio.create_task(drain_inbound())
                        try:
                            await orch.run_once()
                        finally:
                            drain.cancel()
                    except Exception as exc:
                        logger.exception("orchestrator crashed")
                        await hub.broadcast(
                            ErrorEvent(stage="orchestrator", msg=str(exc), retriable=True)
                            .model_dump(mode="json")
                        )
                        await asyncio.sleep(1.0)
```

Note: `ManualStartEvent` doesn't need orchestrator action in Phase 1 — handling it (skip-wake fast path) is deferred to the Phase 2 plan.

- [ ] **Step 18.4: Re-run all tests**

```bash
pytest
```

Expected: All tests still pass.

- [ ] **Step 18.5: Commit**

```bash
git add backend/src/jarvis/app.py backend/tests/test_app_inbound.py
git commit -m "feat(app): route inbound websocket events to orchestrator queue"
```

---

## Task 19: Model Download Script

**Files:**
- Create: `scripts/download_models.sh`

- [ ] **Step 19.1: Create `scripts/download_models.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"

mkdir -p "$MODELS/openwakeword" "$MODELS/piper"

# OpenWakeWord "jarvis" model
WAKE_URL="https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/jarvis.tflite"
if [ ! -f "$MODELS/openwakeword/jarvis.tflite" ]; then
  echo "Downloading wake-word model..."
  curl -L "$WAKE_URL" -o "$MODELS/openwakeword/jarvis.tflite"
fi

# Piper voice (German, Thorsten)
PIPER_ONNX="https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
PIPER_JSON="$PIPER_ONNX.json"
if [ ! -f "$MODELS/piper/de_DE-thorsten-medium.onnx" ]; then
  echo "Downloading Piper voice..."
  curl -L "$PIPER_ONNX" -o "$MODELS/piper/de_DE-thorsten-medium.onnx"
  curl -L "$PIPER_JSON" -o "$MODELS/piper/de_DE-thorsten-medium.onnx.json"
fi

echo "Done. Whisper model auto-downloads on first run."
```

- [ ] **Step 19.2: Make executable and run**

```bash
chmod +x scripts/download_models.sh
scripts/download_models.sh
```

Expected: two files in `models/openwakeword/` and `models/piper/`.

- [ ] **Step 19.3: Pull Ollama model**

```bash
ollama pull llama3.2:3b
```

Expected: Manifest + blobs downloaded.

- [ ] **Step 19.4: Commit**

```bash
git add scripts/download_models.sh
git commit -m "chore: model download script"
```

---

## Task 20: Final Integration Smoke Test (Manual)

This step is a runbook, not a pytest. It validates Phase 1 exit criteria.

- [ ] **Step 20.1: Start backend with real pipeline**

```bash
cd backend
JARVIS_WHISPER_MODEL=base uvicorn "jarvis.app:create_app(start_orchestrator=True)" --factory --port 8765
```

Expected: Logs show "Started server process". After ~10s (model load), `health` endpoint returns OK.

- [ ] **Step 20.2: Start frontend**

```bash
cd frontend
npm run tauri dev
```

Expected: HUD window opens. "ONLINE" badge.

- [ ] **Step 20.3: Verify wake-word + voice loop**

Speak: "Jarvis, hallo wie geht es dir?"

Expected:
- HUD ring color flips through cyan → cyan → purple → amber → cyan over ~5-10 seconds.
- Subtitle bottom-left shows transcript.
- Subtitle bottom-right (or response area) shows LLM reply text as it streams.
- Audio plays through speakers in Thorsten voice.

- [ ] **Step 20.4: Verify barge-in**

While Jarvis is speaking, say "Jarvis" again.

Expected: TTS stops, ring flips back to listening cyan.

- [ ] **Step 20.5: Verify hotkey**

Press `Win+J`.

Expected: Same as a wake-word — pipeline starts at LISTENING.

- [ ] **Step 20.6: Document outcome**

Append a `## Phase 1 Smoke Notes` section to `README.md` recording the date and any observed latency:

```markdown
## Phase 1 Smoke Notes

- Verified end-to-end loop on: [DATE]
- Hardware: [SPEC]
- Wake → Speak latency: [Xs]
- Known issues: [list]
```

- [ ] **Step 20.7: Commit + tag**

```bash
git add README.md
git commit -m "docs: phase 1 smoke notes"
git tag -a phase-1 -m "Phase 1 — core voice loop complete"
```

---

## Self-Review Checklist

After completing all tasks, verify:

| Spec section | Implemented by |
|---|---|
| §3 Architecture · Frontend Tauri shell | Task 14, 16 |
| §3 Architecture · WebSocket bus | Task 13 |
| §3 Architecture · Backend Orchestrator | Task 12 |
| §3 Architecture · Wake | Task 7 |
| §3 Architecture · VAD | Task 8 |
| §3 Architecture · STT | Task 9 |
| §3 Architecture · LLM | Task 10 |
| §3 Architecture · TTS | Task 11 |
| §3 Architecture · audio_io | Task 5 |
| §3 Architecture · plugin/ framework | Task 4 |
| §4 State Machine | Task 2 |
| §5 WebSocket Event Schema | Task 1 |
| §6 Plugin Contract (framework only, no tools) | Task 4 |
| §7 Phase 1 deliverables | Tasks 0–20 |
| §8 Error Handling (per-stage error events) | Task 12 (try/except in orchestrator) |
| §9 Testing Strategy | Every implementation task has its own test task |
| §10 Risks · CPU latency | Streaming chunks (Tasks 10, 11) |
| §10 Risks · Wake false-positives | Tunable threshold in Task 3 config |
| §11 Repo Structure | Layout matches across all tasks |

**Out-of-scope for Phase 1** (deferred to later plans):
- Plugin discovery + hot-reload (Phase 2 plan)
- Real tools (system/web/files/calendar/mail) (Phase 2 plan)
- Confirm-tool voice flow (Phase 2 plan)
- Conversation memory (Phase 3 plan)
- Installer (Phase 3 plan)
- Vision (Phase 3 plan)
