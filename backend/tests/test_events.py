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
