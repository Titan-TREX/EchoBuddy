"""
HTTP route tests for realtime/routes (documents, text chat, voice chat), with the
heavy pieces (embedding, LLM, TTS, SQLite, Temporal) replaced by fakes.
"""

from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import settings
from realtime.routes import chat, documents

OWNER = "owner-aaaa1111"
OTHER = "owner-bbbb2222"
AS_OWNER = {"X-User-Id": OWNER}


@pytest.fixture
def docs_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    ingested: list[tuple[str, str]] = []

    def fake_ingest(path: str, owner: str) -> dict:
        ingested.append((path, owner))
        return {"file": path.replace("\\", "/").rsplit("/", 1)[-1], "chunks_added": 3, "total_chunks": 3}

    monkeypatch.setattr(documents, "ingest_file", fake_ingest)
    app = FastAPI()
    app.include_router(documents.router)
    client = TestClient(app)
    client.ingested = ingested
    client.upload_dir = tmp_path
    return client


def _upload(client, filename: str, content: bytes = b"hello", headers=AS_OWNER):
    return client.post(
        "/api/upload-doc", files={"file": (filename, io.BytesIO(content))}, headers=headers
    )


class TestUpload:
    def test_accepts_txt_and_ingests_it_for_the_caller(self, docs_client):
        res = _upload(docs_client, "notes.txt")
        assert res.status_code == 200
        assert res.json()["success"] is True and res.json()["chunks_added"] == 3
        assert (docs_client.upload_dir / OWNER / "notes.txt").read_bytes() == b"hello"
        assert [owner for _, owner in docs_client.ingested] == [OWNER]

    def test_same_filename_from_two_users_does_not_collide(self, docs_client):
        _upload(docs_client, "notes.txt", b"mine", {"X-User-Id": OWNER})
        _upload(docs_client, "notes.txt", b"theirs", {"X-User-Id": OTHER})
        assert (docs_client.upload_dir / OWNER / "notes.txt").read_bytes() == b"mine"
        assert (docs_client.upload_dir / OTHER / "notes.txt").read_bytes() == b"theirs"

    def test_rejects_unsupported_extension(self, docs_client):
        res = _upload(docs_client, "malware.exe")
        assert res.status_code == 400
        assert "Unsupported" in res.json()["error"]
        assert docs_client.ingested == []

    def test_path_traversal_in_filename_cannot_escape_the_owner_folder(self, docs_client):
        res = _upload(docs_client, "../../evil.txt")
        assert res.status_code == 200
        assert (docs_client.upload_dir / OWNER / "evil.txt").exists()
        assert not (docs_client.upload_dir.parent.parent / "evil.txt").exists()

    def test_unsafe_characters_are_sanitised(self, docs_client):
        assert _upload(docs_client, "my<>:doc?.txt").status_code == 200
        assert (docs_client.upload_dir / OWNER / "my___doc_.txt").exists()

    def test_oversized_upload_is_rejected_and_not_kept(self, docs_client, monkeypatch):
        monkeypatch.setattr(settings, "max_upload_mb", 0)
        res = _upload(docs_client, "big.txt", b"x" * 10)
        assert res.status_code == 413
        assert not (docs_client.upload_dir / OWNER / "big.txt").exists()

    def test_ingestion_failure_is_a_clean_error_without_internals(self, docs_client, monkeypatch):
        def boom(_path, _owner):
            raise RuntimeError("chromadb internal detail /secret/path")

        monkeypatch.setattr(documents, "ingest_file", boom)
        res = _upload(docs_client, "notes.txt")
        assert res.status_code == 500
        assert "secret" not in res.json()["error"]
        assert not (docs_client.upload_dir / OWNER / "notes.txt").exists()


class TestOwnerIdentity:
    @pytest.mark.parametrize("headers", [{}, {"X-User-Id": "short"}, {"X-User-Id": "has spaces in it!"}, {"X-User-Id": "../../etc"}])
    def test_missing_or_malformed_id_is_rejected_on_every_documents_route(self, docs_client, headers):
        assert _upload(docs_client, "a.txt", headers=headers).status_code == 400
        assert docs_client.get("/api/docs", headers=headers).status_code == 400
        assert docs_client.delete("/api/docs/a.txt", headers=headers).status_code == 400
        assert docs_client.ingested == []


class TestListAndDelete:
    def test_list_is_scoped_to_the_caller(self, docs_client, monkeypatch):
        seen = []
        monkeypatch.setattr(documents, "list_docs", lambda owner: seen.append(owner) or ["a.txt", "b.pdf"])
        res = docs_client.get("/api/docs", headers=AS_OWNER)
        assert res.json() == {"documents": ["a.txt", "b.pdf"]}
        assert seen == [OWNER]

    def test_delete_removes_the_callers_chunks_and_file_only(self, docs_client, monkeypatch):
        calls = []
        monkeypatch.setattr(documents, "delete_doc", lambda name, owner: calls.append((name, owner)) or 4)
        (docs_client.upload_dir / OWNER).mkdir()
        (docs_client.upload_dir / OTHER).mkdir()
        (docs_client.upload_dir / OWNER / "a.txt").write_text("mine")
        (docs_client.upload_dir / OTHER / "a.txt").write_text("theirs")

        res = docs_client.delete("/api/docs/a.txt", headers=AS_OWNER)

        assert res.json() == {"deleted_chunks": 4, "file": "a.txt"}
        assert calls == [("a.txt", OWNER)]
        assert not (docs_client.upload_dir / OWNER / "a.txt").exists()
        assert (docs_client.upload_dir / OTHER / "a.txt").exists()  # someone else's file untouched


@pytest.fixture
def chat_client(tmp_path, monkeypatch):
    # tmp_path is deliberately outside the project folder, like a mounted DATA_DIR.
    monkeypatch.setattr(settings, "assets_dir", tmp_path)
    events: list[tuple] = []

    async def ensure(user_id, session_id, username):
        events.append(("ensure", session_id))

    async def history(session_id, limit):
        events.append(("history", limit))
        return [{"role": "user", "content": "earlier"}]

    async def log(session_id, role, content, audio_path):
        events.append(("log", role, content))

    async def reply(text, hist, owner=""):
        events.append(("llm", text, list(hist), owner))
        yield "Hi there.", "Hi there."

    async def speech(text):
        events.append(("tts", text))
        yield b"\x00\x01" * 100

    monkeypatch.setattr(chat, "ensure_user_and_session", ensure)
    monkeypatch.setattr(chat, "fetch_recent_history", history)
    monkeypatch.setattr(chat, "log_message", log)
    monkeypatch.setattr(chat, "stream_reply_chunks", reply)
    monkeypatch.setattr(chat, "stream_speech_chunks", speech)

    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)
    client.events = events
    return client


class TestTextChat:
    def test_round_trip(self, chat_client):
        res = chat_client.post("/api/text-chat", json={"text": "hello", "user_id": OWNER, "session_id": "s1"})
        body = res.json()
        assert res.status_code == 200
        assert body["bot_text"] == "Hi there."
        assert body["user_id"] == OWNER and body["session_id"] == "s1"
        assert body["bot_audio_path"].endswith(".wav")

    def test_the_callers_id_is_the_owner_whose_documents_are_searched(self, chat_client):
        chat_client.post("/api/text-chat", json={"text": "hello", "user_id": OWNER})
        llm_event = next(e for e in chat_client.events if e[0] == "llm")
        assert llm_event[3] == OWNER

    def test_history_is_fetched_before_the_user_message_is_logged(self, chat_client):
        """Regression: logging first made the LLM see the user message twice."""
        chat_client.post("/api/text-chat", json={"text": "hello", "session_id": "s1"})
        kinds = [(e[0], e[1]) if e[0] == "log" else (e[0],) for e in chat_client.events]
        assert kinds.index(("history",)) < kinds.index(("log", "user"))
        llm_event = next(e for e in chat_client.events if e[0] == "llm")
        assert llm_event[2] == [{"role": "user", "content": "earlier"}]

    def test_history_limit_comes_from_settings(self, chat_client):
        chat_client.post("/api/text-chat", json={"text": "hello"})
        assert ("history", settings.history_max_turns) in chat_client.events

    def test_ids_are_generated_when_missing(self, chat_client):
        body = chat_client.post("/api/text-chat", json={"text": "hello"}).json()
        assert body["user_id"].startswith("user_") and len(body["session_id"]) == 36

    def test_malformed_user_id_is_rejected(self, chat_client):
        for bad in ("short", "../../x", "has space!!", "x" * 65):
            assert chat_client.post("/api/text-chat", json={"text": "hi", "user_id": bad}).status_code == 422

    def test_rejects_empty_and_oversized_text(self, chat_client):
        assert chat_client.post("/api/text-chat", json={"text": ""}).status_code == 422
        too_long = "x" * (settings.text_chat_max_chars + 1)
        assert chat_client.post("/api/text-chat", json={"text": too_long}).status_code == 422


class TestVoiceChatValidation:
    def test_rejects_unsupported_audio_type(self, chat_client):
        res = chat_client.post("/api/voice-chat", files={"audio": ("clip.exe", io.BytesIO(b"x"))})
        assert res.status_code == 400

    def test_user_and_session_ids_sent_as_form_fields_reach_the_workflow(self, chat_client, monkeypatch):
        """Regression: they were declared as query params, so form values were ignored and
        every batch turn started a brand-new conversation."""
        seen = {}

        class FakeClient:
            async def execute_workflow(self, _run, workflow_input, **_kwargs):
                seen["input"] = workflow_input
                return chat.VoiceChatbotResult(
                    user_text="hi", bot_text="hello", bot_audio_path="assets/bot.mp3", history=[]
                )

        async def fake_client():
            return FakeClient()

        monkeypatch.setattr(chat, "_get_temporal_client", fake_client)
        res = chat_client.post(
            "/api/voice-chat",
            files={"audio": ("clip.webm", io.BytesIO(b"audio"))},
            data={"user_id": OWNER, "session_id": "tab-session"},
        )

        assert res.status_code == 200
        assert seen["input"].user_id == OWNER
        assert seen["input"].session_id == "tab-session"
        assert res.json()["session_id"] == "tab-session"
