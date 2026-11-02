"""WebSocket integration smoke test using FastAPI TestClient."""
import json

import pytest
from fastapi.testclient import TestClient

from jarvis.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app(start_orchestrator=False)  # don't try to open the mic
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "version": "0.1.0"}


def test_websocket_accepts_manual_start(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "manual_start"}))
        # Server should not close the connection on a known event.
        # We don't assert on responses here because orchestrator isn't running.
        ws.close()


def test_websocket_rejects_garbage(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not even json")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "parse" in msg["msg"].lower() or "invalid" in msg["msg"].lower()
