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
