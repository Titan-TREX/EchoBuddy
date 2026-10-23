"""Tests for realtime/security.py: bearer token, rate limiting and WebSocket access."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from config.settings import settings
from realtime import security
from realtime.routes import auth
from realtime.security import RateLimiter, access_guard, authorize_websocket, limiter


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(auth.router)

    @app.get("/guarded", dependencies=[Depends(access_guard())])
    async def guarded():
        return {"ok": True}

    @app.get("/expensive", dependencies=[Depends(access_guard(cost=5))])
    async def expensive():
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        if not await authorize_websocket(websocket):
            return
        await websocket.accept()
        await websocket.send_json({"hello": True})
        await websocket.close()

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestToken:
    def test_open_when_no_token_configured(self, client):
        assert client.get("/guarded").status_code == 200

    def test_missing_token_is_401(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_token", "s3cret")
        res = client.get("/guarded")
        assert res.status_code == 401
        assert res.headers["www-authenticate"] == "Bearer"

    def test_wrong_token_is_401(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_token", "s3cret")
        assert client.get("/guarded", headers={"Authorization": "Bearer nope"}).status_code == 401

    def test_non_bearer_scheme_is_401(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_token", "s3cret")
        assert client.get("/guarded", headers={"Authorization": "Basic s3cret"}).status_code == 401

    def test_correct_token_is_accepted(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_token", "s3cret")
        assert client.get("/guarded", headers={"Authorization": "Bearer s3cret"}).status_code == 200

    def test_auth_check_endpoint(self, client, monkeypatch):
        assert client.get("/api/auth/check").json() == {"ok": True}
        monkeypatch.setattr(settings, "api_token", "s3cret")
        assert client.get("/api/auth/check").status_code == 401
        assert client.get("/api/auth/check", headers={"Authorization": "Bearer s3cret"}).status_code == 200

    def test_auth_check_is_not_rate_limited(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_requests", 1)
        assert all(client.get("/api/auth/check").status_code == 200 for _ in range(5))


class TestRateLimit:
    def test_blocks_after_the_limit_with_retry_after(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_requests", 3)
        assert [client.get("/guarded").status_code for _ in range(3)] == [200, 200, 200]
        res = client.get("/guarded")
        assert res.status_code == 429
        assert int(res.headers["retry-after"]) >= 1

    def test_cost_counts_against_the_limit(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_requests", 6)
        assert client.get("/expensive").status_code == 200       # 5 of 6
        assert client.get("/expensive").status_code == 429       # would be 10
        assert client.get("/guarded").status_code == 200         # 6 of 6

    def test_clients_are_limited_separately_behind_a_trusted_proxy(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_requests", 1)
        monkeypatch.setattr(settings, "trust_proxy_headers", True)
        a, b = {"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}, {"X-Forwarded-For": "2.2.2.2"}
        assert client.get("/guarded", headers=a).status_code == 200
        assert client.get("/guarded", headers=a).status_code == 429
        assert client.get("/guarded", headers=b).status_code == 200

    def test_forwarded_header_is_ignored_unless_trusted(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_requests", 1)
        assert client.get("/guarded", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
        # A spoofed header must not give an attacker a fresh bucket.
        assert client.get("/guarded", headers={"X-Forwarded-For": "9.9.9.9"}).status_code == 429

    def test_window_expires(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_requests", 1)
        monkeypatch.setattr(settings, "rate_limit_window_seconds", 60)
        clock = {"now": 1000.0}
        monkeypatch.setattr(security.time, "monotonic", lambda: clock["now"])
        rl = RateLimiter()
        assert rl.check("k") == 0
        assert rl.check("k") > 0
        clock["now"] += 61
        assert rl.check("k") == 0

    def test_stale_keys_are_pruned(self, monkeypatch):
        monkeypatch.setattr(security, "_PRUNE_ABOVE_KEYS", 2)
        clock = {"now": 0.0}
        monkeypatch.setattr(security.time, "monotonic", lambda: clock["now"])
        rl = RateLimiter()
        for key in ("a", "b", "c"):
            rl.check(key)
        clock["now"] += settings.rate_limit_window_seconds + 1
        rl.check("d")
        assert set(rl._hits) == {"d"}


class TestWebSocket:
    def test_open_connection_allowed(self, client):
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json() == {"hello": True}

    def test_missing_token_refused(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_token", "s3cret")
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws"):
                pass

    def test_token_via_query_string(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_token", "s3cret")
        with client.websocket_connect("/ws?token=s3cret") as ws:
            assert ws.receive_json() == {"hello": True}

    def test_wrong_query_token_refused(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_token", "s3cret")
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws?token=wrong"):
                pass

    def test_websocket_connections_are_rate_limited(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_requests", 1)
        with client.websocket_connect("/ws"):
            pass
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws"):
                pass


def test_limiter_singleton_is_reset_between_tests():
    assert limiter._hits == {}
