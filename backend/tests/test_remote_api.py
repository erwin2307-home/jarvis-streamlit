"""Remote chat API tests. Uses FakeLlm injected onto app.state — never
hits real Ollama/cloud backends."""
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

    remote_api.app.state.llm = FakeLlm(["hallo ", "welt"])
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
