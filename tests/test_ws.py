"""WebSocket route: which user id a voice session runs as (it decides whose documents are searched)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from realtime.routes import ws
from realtime.security import is_valid_user_id


class FakeSession:
    instances: list["FakeSession"] = []

    def __init__(self, send_json, send_bytes, user_id, session_id):
        self.user_id = user_id
        self.session_id = session_id
        self.gen_task = None
        self.fed = bytearray()
        FakeSession.instances.append(self)

    async def start(self):
        pass

    async def feed_pcm16(self, pcm):
        self.fed.extend(pcm)


@pytest.fixture
def client(monkeypatch):
    FakeSession.instances = []
    monkeypatch.setattr(ws, "VoiceSession", FakeSession)
    app = FastAPI()
    app.include_router(ws.router)
    return TestClient(app)


def _connect(client, first_message):
    with client.websocket_connect("/ws") as sock:
        sock.send_json(first_message)
        assert sock.receive_json()["type"] == "ready"
        sock.send_bytes(b"\x00\x01" * 160)
    return FakeSession.instances[-1]


class TestUserIdentity:
    def test_valid_browser_id_becomes_the_session_user(self, client):
        session = _connect(client, {"type": "start", "sampleRate": 16000, "userId": "u_0123456789abcdef"})
        assert session.user_id == "u_0123456789abcdef"

    @pytest.mark.parametrize("bad", [None, "", "short", "../../etc/passwd", "has spaces here", "x" * 65, 12345])
    def test_missing_or_malformed_id_gets_a_throwaway_identity(self, client, bad):
        message = {"type": "start", "sampleRate": 16000}
        if bad is not None:
            message["userId"] = bad
        session = _connect(client, message)
        assert session.user_id.startswith("user_")
        assert session.user_id != bad

    def test_ready_message_announces_the_playback_rate(self, client):
        with client.websocket_connect("/ws") as sock:
            sock.send_json({"sampleRate": 16000})
            assert sock.receive_json() == {"type": "ready", "ttsSampleRate": 16000}


class TestIdFormat:
    @pytest.mark.parametrize("good", ["abcdefgh", "u_0123456789abcdef0123456789abcdef", "A-b_C-d_1234", "x" * 64])
    def test_valid(self, good):
        assert is_valid_user_id(good)

    @pytest.mark.parametrize("bad", [None, "", "short", "x" * 65, "has space1", "dots.are.bad", "../traversal", "slash/slash1", "uniécode1"])
    def test_invalid(self, bad):
        assert not is_valid_user_id(bad)
