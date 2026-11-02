"""Remote chat/voice API tests. Uses Fake* backends injected onto
app.state — never hits real Ollama/cloud/Whisper/TTS."""
import base64

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
