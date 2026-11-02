"""Test that inbound WS events reach the orchestrator queue."""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from jarvis.app import create_app
from jarvis.events import CancelEvent, ManualStartEvent


def test_manual_start_pushed_to_queue() -> None:
    app = create_app(start_orchestrator=False)
    queue: asyncio.Queue[object] = asyncio.Queue()
    app.state.inbound_queue = queue

    async def handler(ev: object) -> None:
        await app.state.inbound_queue.put(ev)
    app.state.inbound_handler = handler

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "manual_start"}))
        ws.send_text(json.dumps({"type": "cancel"}))
        ws.close()

    drained: list[object] = []
    while not queue.empty():
        drained.append(queue.get_nowait())

    assert any(isinstance(ev, ManualStartEvent) for ev in drained)
    assert any(isinstance(ev, CancelEvent) for ev in drained)


def test_garbage_does_not_pollute_queue() -> None:
    app = create_app(start_orchestrator=False)
    queue: asyncio.Queue[object] = asyncio.Queue()
    app.state.inbound_queue = queue

    async def handler(ev: object) -> None:
        await app.state.inbound_queue.put(ev)
    app.state.inbound_handler = handler

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        ws.close()

    assert queue.empty()
