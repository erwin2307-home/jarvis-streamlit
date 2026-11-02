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
