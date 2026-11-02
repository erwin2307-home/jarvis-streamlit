"""WebSocket event schemas. Pure data types — no I/O."""
from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, Literal, Union, cast

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
    args: dict[str, Any]
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


Event = Annotated[
    Union[
        StateEvent,
        AudioLevelEvent,
        TranscriptPartialEvent,
        TranscriptFinalEvent,
        ToolCallEvent,
        ToolResultEvent,
        ResponseChunkEvent,
        ResponseDoneEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
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


_INBOUND_BY_TYPE: dict[str, type[BaseModel]] = {
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
    return cast("ManualStartEvent | CancelEvent | ConfirmToolEvent", cls.model_validate(data))
