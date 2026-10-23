"""Tests for GET /health and the DATA_DIR storage setting."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import Settings, settings
from rag import ingestor
from realtime.routes import health


@pytest.fixture(autouse=True)
def _fresh_cache():
    health.reset_cache()
    yield
    health.reset_cache()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(health.router)
    return TestClient(app)


class _FakeDb:
    async def execute(self, _sql):
        return None


class TestHealth:
    def test_ok_when_database_answers(self, client, monkeypatch):
        @asynccontextmanager
        async def fake_connection():
            yield _FakeDb()

        monkeypatch.setattr(health, "get_connection", fake_connection)
        monkeypatch.setattr(ingestor, "_collection", object())
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok", "database": "ok", "document_search_ready": True}

    def test_reports_document_search_still_loading(self, client, monkeypatch):
        @asynccontextmanager
        async def fake_connection():
            yield _FakeDb()

        monkeypatch.setattr(health, "get_connection", fake_connection)
        monkeypatch.setattr(ingestor, "_collection", None)
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["document_search_ready"] is False

    def test_503_when_database_is_down(self, client, monkeypatch):
        @asynccontextmanager
        async def broken_connection():
            raise RuntimeError("disk gone: /secret/path")
            yield  # pragma: no cover

        monkeypatch.setattr(health, "get_connection", broken_connection)
        res = client.get("/health")
        assert res.status_code == 503
        assert res.json()["status"] == "degraded"
        assert "secret" not in res.text   # never leak internals to an unauthenticated endpoint

    def test_needs_no_token_and_is_not_rate_limited(self, client, monkeypatch):
        @asynccontextmanager
        async def fake_connection():
            yield _FakeDb()

        monkeypatch.setattr(health, "get_connection", fake_connection)
        monkeypatch.setattr(settings, "api_token", "s3cret")
        monkeypatch.setattr(settings, "rate_limit_requests", 1)
        assert all(client.get("/health").status_code == 200 for _ in range(5))


class TestHealthCache:
    def test_a_flood_of_probes_runs_one_database_check(self, client, monkeypatch):
        checks = []

        @asynccontextmanager
        async def counting_connection():
            checks.append(1)
            yield _FakeDb()

        monkeypatch.setattr(health, "get_connection", counting_connection)
        for _ in range(25):
            assert client.get("/health").status_code == 200
        assert len(checks) == 1

    def test_the_check_runs_again_after_the_cache_expires(self, client, monkeypatch):
        checks = []

        @asynccontextmanager
        async def counting_connection():
            checks.append(1)
            yield _FakeDb()

        monkeypatch.setattr(health, "get_connection", counting_connection)
        monkeypatch.setattr(settings, "health_db_cache_seconds", 0)
        client.get("/health")
        client.get("/health")
        assert len(checks) == 2

    def test_an_outage_is_noticed_once_the_cache_expires(self, client, monkeypatch):
        @asynccontextmanager
        async def ok_connection():
            yield _FakeDb()

        @asynccontextmanager
        async def broken_connection():
            raise RuntimeError("down")
            yield  # pragma: no cover

        monkeypatch.setattr(settings, "health_db_cache_seconds", 0)
        monkeypatch.setattr(health, "get_connection", ok_connection)
        assert client.get("/health").status_code == 200
        monkeypatch.setattr(health, "get_connection", broken_connection)
        assert client.get("/health").status_code == 503


class TestDataDir:
    def test_defaults_keep_the_project_layout(self):
        s = Settings()
        assert s.assets_dir == s.base_dir / "assets"
        assert s.database_path == s.base_dir / "database" / "voice_chatbot.db"
        assert s.chroma_path == s.base_dir / "chroma_db"
        assert s.upload_dir == s.base_dir / "uploaded_docs"

    def test_data_dir_moves_all_four_stores_together(self, tmp_path):
        with patch.dict(os.environ, {"DATA_DIR": str(tmp_path)}):
            s = Settings()
        assert s.assets_dir == tmp_path / "assets"
        assert s.database_path == tmp_path / "database" / "voice_chatbot.db"
        assert s.chroma_path == tmp_path / "chroma_db"
        assert s.upload_dir == tmp_path / "uploaded_docs"
        assert s.schema_path == s.base_dir / "database" / "schema.sql"  # code, not data: stays put

    def test_an_explicit_path_wins_over_data_dir(self, tmp_path):
        env = {"DATA_DIR": str(tmp_path), "CHROMA_PATH": str(tmp_path / "elsewhere")}
        with patch.dict(os.environ, env):
            s = Settings()
        assert s.chroma_path == Path(tmp_path / "elsewhere")
        assert s.assets_dir == tmp_path / "assets"
