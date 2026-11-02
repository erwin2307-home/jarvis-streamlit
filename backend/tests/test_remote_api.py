"""Remote chat/voice API tests. Uses Fake* backends injected onto
app.state — never hits real Ollama/cloud/Whisper/TTS."""
import base64
import re

import pytest

from jarvis.config import get_settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("JARVIS_REMOTE_API_KEY", "test-secret")
    monkeypatch.setenv("JARVIS_USE_CLOUD", "0")
    get_settings.cache_clear()

    from fastapi.testclient import TestClient
    from jarvis import remote_api
    from jarvis.llm import FakeLlm
    from jarvis.stt import FakeStt
    from jarvis.tts import FakeTts

    remote_api.app.state.llm = FakeLlm(["hallo ", "welt"])
    remote_api.app.state.stt = FakeStt(text="hallo jarvis")
    remote_api.app.state.tts = FakeTts()
    with TestClient(remote_api.app) as c:
        yield c

    get_settings.cache_clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_chat_requires_api_key(client):
    resp = client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 401


def test_chat_rejects_wrong_api_key(client):
    resp = client.post(
        "/chat", json={"message": "hi"}, headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401


def test_chat_returns_reply_with_valid_key(client):
    resp = client.post(
        "/chat", json={"message": "hi"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "hallo welt"


def test_chat_rejects_empty_message(client):
    resp = client.post(
        "/chat", json={"message": "   "}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 400


def test_voice_options_preflight_is_allowed(client):
    # Streamlit Cloud serves a different origin than the tunnel URL, so the
    # browser sends a CORS preflight (OPTIONS) before the real POST /voice.
    # Without CORSMiddleware this 405s and the browser never sends the real
    # request — this reproduces that failure.
    resp = client.options(
        "/voice",
        headers={
            "Origin": "https://example.streamlit.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"


def test_voice_requires_api_key(client, fixtures_dir):
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post("/voice", files={"audio": ("rec.wav", wav_bytes, "audio/wav")})
    assert resp.status_code == 401


def test_voice_returns_transcript_reply_and_audio(client, fixtures_dir):
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "hallo jarvis"
    assert body["reply"] != ""
    audio = base64.b64decode(body["audio_base64"])
    assert len(audio) > 0
    assert body["sample_rate"] > 0


def test_voice_rejects_empty_transcript(client, fixtures_dir):
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="")
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "text",
    ["wie spät ist es", "Wie spät ist es?", "wieviel uhr ist es", "what time is it"],
)
def test_is_time_query_matches_time_questions(client, text):
    from jarvis.remote_api import is_time_query

    assert is_time_query(text) is True


@pytest.mark.parametrize("text", ["wie geht es dir", "baue einen taschenrechner", "hallo"])
def test_is_time_query_does_not_match_other_questions(client, text):
    from jarvis.remote_api import is_time_query

    assert is_time_query(text) is False


_TIME_PATTERN = re.compile(r"\d{2}:\d{2} Uhr")


def test_chat_answers_time_query_deterministically_without_llm(client):
    # A tiny local model can hallucinate on "wie spät ist es" (observed live:
    # answered with London weather instead of the time). Time queries must
    # bypass the LLM entirely rather than rely on it reading the injected
    # time context correctly.
    from jarvis.llm import FakeLlm

    class TrapLlm(FakeLlm):
        pass

    client.app.state.llm = TrapLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat", json={"message": "wie spät ist es"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert "TRAP" not in reply
    assert _TIME_PATTERN.search(reply)


def test_voice_answers_time_query_deterministically_without_llm(client, fixtures_dir):
    from jarvis.llm import FakeLlm
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="wie spät ist es")
    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert "TRAP" not in reply
    assert _TIME_PATTERN.search(reply)


@pytest.mark.parametrize(
    "text,expected_seconds",
    [
        ("timer auf 5 minuten", 300),
        ("stell einen timer auf 90 sekunden", 90),
        ("timer auf 1 stunde 30 minuten", 5400),
        ("timer auf zehn minuten", 600),
        ("timer auf fünf minuten", 300),
    ],
)
def test_parse_timer_seconds_parses_durations(client, text, expected_seconds):
    from jarvis.remote_api import parse_timer_seconds

    assert parse_timer_seconds(text) == expected_seconds


def test_parse_timer_seconds_returns_none_without_a_duration(client):
    from jarvis.remote_api import parse_timer_seconds

    assert parse_timer_seconds("timer") is None


@pytest.mark.parametrize("text", ["timer", "stell einen timer", "Timer auf 5 Minuten"])
def test_is_timer_query_matches_timer_mentions(client, text):
    from jarvis.remote_api import is_timer_query

    assert is_timer_query(text) is True


@pytest.mark.parametrize("text", ["wie spät ist es", "baue einen taschenrechner", "hallo"])
def test_is_timer_query_does_not_match_other_questions(client, text):
    from jarvis.remote_api import is_timer_query

    assert is_timer_query(text) is False


def test_chat_timer_with_duration_sets_timer_seconds_without_llm(client):
    from jarvis.llm import FakeLlm

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat",
        json={"message": "timer auf 5 minuten"},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["timer_seconds"] == 300


def test_chat_timer_without_duration_asks_for_one(client):
    from jarvis.llm import FakeLlm

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat", json={"message": "timer"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["timer_seconds"] is None


def test_voice_timer_with_duration_sets_timer_seconds_without_llm(client, fixtures_dir):
    from jarvis.llm import FakeLlm
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="timer auf 2 minuten")
    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["timer_seconds"] == 120


def test_chat_beadchip_exact_match_bypasses_llm(client):
    from jarvis.llm import FakeLlm
    from jarvis.qa_beadchip import BEADCHIP_QA
    from jarvis.qa_lookup import best_match

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat",
        json={"message": "was ist ein illumina beadchip chip"},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert "TRAP" not in reply
    assert reply == best_match("was ist ein illumina beadchip chip", BEADCHIP_QA)


def test_chat_bare_reagent_abbreviation_bypasses_llm(client):
    from jarvis.llm import FakeLlm
    from jarvis.qa_beadchip import reagent_answer

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat", json={"message": "ra1"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert "TRAP" not in reply
    assert reply == reagent_answer("ra1")


def test_chat_loose_beadchip_question_injects_context_for_llm(client):
    from jarvis.llm import LlmEvent

    captured_messages = []

    class CapturingLlm:
        async def chat(self, messages):
            captured_messages.append(messages)
            yield LlmEvent(kind="done", text="synthesized answer")

    client.app.state.llm = CapturingLlm()
    resp = client.post(
        "/chat",
        json={"message": "erzähl mir was über den beadchip workflow allgemein"},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "synthesized answer"
    assert len(captured_messages) == 1
    system_texts = [m["content"] for m in captured_messages[0] if m.get("role") == "system"]
    assert any("BeadChip" in t for t in system_texts)


def test_chat_unrelated_question_gets_no_beadchip_context(client):
    from jarvis.llm import LlmEvent

    captured_messages = []

    class CapturingLlm:
        async def chat(self, messages):
            captured_messages.append(messages)
            yield LlmEvent(kind="done", text="ok")

    client.app.state.llm = CapturingLlm()
    resp = client.post(
        "/chat", json={"message": "wie geht es dir"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    system_texts = [m["content"] for m in captured_messages[0] if m.get("role") == "system"]
    assert not any("BeadChip" in t for t in system_texts)


def test_voice_beadchip_exact_match_bypasses_llm(client, fixtures_dir):
    from jarvis.llm import FakeLlm
    from jarvis.qa_beadchip import BEADCHIP_QA
    from jarvis.qa_lookup import best_match
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="was ist ein illumina beadchip chip")
    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["reply"] == best_match("was ist ein illumina beadchip chip", BEADCHIP_QA)


def test_voice_bare_reagent_abbreviation_bypasses_llm(client, fixtures_dir):
    from jarvis.llm import FakeLlm
    from jarvis.qa_beadchip import reagent_answer
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="ra1")
    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["reply"] == reagent_answer("ra1")
