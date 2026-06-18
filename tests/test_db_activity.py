"""
Tests for temporal/activities/db_activity.py

Strategy: patch `get_connection` with an in-memory SQLite database so every
test is hermetic, fast, and requires no Temporal runtime.

Covers:
  - ensure_user_and_session: creates rows; idempotent on repeat call.
  - fetch_recent_history:    returns turns oldest-first; respects `limit`.
  - log_message:             inserts user & assistant records; returns row id.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

MINIMAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    audio_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def mem_db():
    """Yield an in-memory aiosqlite connection with the full schema applied."""
    db = await aiosqlite.connect(":memory:")
    await db.execute("PRAGMA foreign_keys = ON;")
    await db.executescript(MINIMAL_SCHEMA)
    await db.commit()
    yield db
    await db.close()


@pytest.fixture
def patched_connection(mem_db, monkeypatch):
    """Monkeypatch get_connection so activities use the in-memory DB."""

    @asynccontextmanager
    async def _fake_get_connection():
        yield mem_db

    monkeypatch.setattr(
        "temporal.activities.db_activity.get_connection", _fake_get_connection
    )


# ---------------------------------------------------------------------------
# ensure_user_and_session
# ---------------------------------------------------------------------------

class TestEnsureUserAndSession:
    @pytest.mark.asyncio
    async def test_creates_user_and_session(self, mem_db, patched_connection):
        from temporal.activities.db_activity import ensure_user_and_session

        await ensure_user_and_session("user-1", "sess-1", "alice")

        cursor = await mem_db.execute("SELECT id, username FROM users WHERE id = ?", ("user-1",))
        row = await cursor.fetchone()
        assert row == ("user-1", "alice")

        cursor = await mem_db.execute("SELECT id, user_id FROM sessions WHERE id = ?", ("sess-1",))
        row = await cursor.fetchone()
        assert row == ("sess-1", "user-1")

    @pytest.mark.asyncio
    async def test_idempotent_on_repeat_call(self, mem_db, patched_connection):
        from temporal.activities.db_activity import ensure_user_and_session

        await ensure_user_and_session("user-2", "sess-2", "bob")
        await ensure_user_and_session("user-2", "sess-2", "bob")  # no-op

        cursor = await mem_db.execute("SELECT COUNT(*) FROM users WHERE id = ?", ("user-2",))
        row = await cursor.fetchone()
        assert row[0] == 1

        cursor = await mem_db.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", ("sess-2",))
        row = await cursor.fetchone()
        assert row[0] == 1

    @pytest.mark.asyncio
    async def test_multiple_sessions_same_user(self, mem_db, patched_connection):
        from temporal.activities.db_activity import ensure_user_and_session

        await ensure_user_and_session("user-3", "sess-3a", "charlie")
        await ensure_user_and_session("user-3", "sess-3b", "charlie")

        cursor = await mem_db.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ?", ("user-3",))
        row = await cursor.fetchone()
        assert row[0] == 2


# ---------------------------------------------------------------------------
# fetch_recent_history
# ---------------------------------------------------------------------------

class TestFetchRecentHistory:
    @pytest_asyncio.fixture(autouse=True)
    async def seed_data(self, mem_db, patched_connection):
        """Insert a user, session, and 5 messages before each test."""
        await mem_db.execute("INSERT INTO users (id, username) VALUES (?, ?)", ("u1", "tester"))
        await mem_db.execute("INSERT INTO sessions (id, user_id) VALUES (?, ?)", ("s1", "u1"))
        messages = [
            ("s1", "user", "msg 1", None),
            ("s1", "assistant", "msg 2", None),
            ("s1", "user", "msg 3", None),
            ("s1", "assistant", "msg 4", None),
            ("s1", "user", "msg 5", None),
        ]
        await mem_db.executemany(
            "INSERT INTO messages (session_id, role, content, audio_path) VALUES (?, ?, ?, ?)",
            messages,
        )
        await mem_db.commit()

    @pytest.mark.asyncio
    async def test_returns_oldest_first(self, patched_connection):
        from temporal.activities.db_activity import fetch_recent_history

        history = await fetch_recent_history("s1", 5)
        assert [t.content for t in history] == ["msg 1", "msg 2", "msg 3", "msg 4", "msg 5"]

    @pytest.mark.asyncio
    async def test_respects_limit(self, patched_connection):
        from temporal.activities.db_activity import fetch_recent_history

        history = await fetch_recent_history("s1", 3)
        assert len(history) == 3
        # Should be the 3 most-recent messages, returned oldest-first
        assert history[0].content == "msg 3"
        assert history[-1].content == "msg 5"

    @pytest.mark.asyncio
    async def test_returns_correct_roles(self, patched_connection):
        from temporal.activities.db_activity import fetch_recent_history

        history = await fetch_recent_history("s1", 5)
        roles = [t.role for t in history]
        assert roles == ["user", "assistant", "user", "assistant", "user"]

    @pytest.mark.asyncio
    async def test_empty_for_unknown_session(self, patched_connection):
        from temporal.activities.db_activity import fetch_recent_history

        history = await fetch_recent_history("nonexistent-session", 10)
        assert history == []

    @pytest.mark.asyncio
    async def test_limit_larger_than_total_messages(self, patched_connection):
        from temporal.activities.db_activity import fetch_recent_history

        history = await fetch_recent_history("s1", 100)
        assert len(history) == 5


# ---------------------------------------------------------------------------
# log_message
# ---------------------------------------------------------------------------

class TestLogMessage:
    @pytest_asyncio.fixture(autouse=True)
    async def seed_user_session(self, mem_db, patched_connection):
        await mem_db.execute("INSERT INTO users (id, username) VALUES (?, ?)", ("u10", "writer"))
        await mem_db.execute("INSERT INTO sessions (id, user_id) VALUES (?, ?)", ("s10", "u10"))
        await mem_db.commit()

    @pytest.mark.asyncio
    async def test_logs_user_message(self, mem_db, patched_connection):
        from temporal.dto import MessageRecord
        from temporal.activities.db_activity import log_message

        record = MessageRecord(
            session_id="s10", role="user", content="What's up?", audio_path="assets/u.wav"
        )
        row_id = await log_message(record)

        assert isinstance(row_id, int)
        assert row_id > 0

        cursor = await mem_db.execute(
            "SELECT role, content, audio_path FROM messages WHERE id = ?", (row_id,)
        )
        row = await cursor.fetchone()
        assert row == ("user", "What's up?", "assets/u.wav")

    @pytest.mark.asyncio
    async def test_logs_assistant_message(self, mem_db, patched_connection):
        from temporal.dto import MessageRecord
        from temporal.activities.db_activity import log_message

        record = MessageRecord(
            session_id="s10",
            role="assistant",
            content="All good!",
            audio_path="assets/bot_xyz.mp3",
        )
        row_id = await log_message(record)

        cursor = await mem_db.execute(
            "SELECT role FROM messages WHERE id = ?", (row_id,)
        )
        row = await cursor.fetchone()
        assert row[0] == "assistant"

    @pytest.mark.asyncio
    async def test_logs_message_without_audio(self, mem_db, patched_connection):
        from temporal.dto import MessageRecord
        from temporal.activities.db_activity import log_message

        record = MessageRecord(
            session_id="s10", role="user", content="Text only", audio_path=None
        )
        row_id = await log_message(record)

        cursor = await mem_db.execute(
            "SELECT audio_path FROM messages WHERE id = ?", (row_id,)
        )
        row = await cursor.fetchone()
        assert row[0] is None

    @pytest.mark.asyncio
    async def test_sequential_ids_increment(self, mem_db, patched_connection):
        from temporal.dto import MessageRecord
        from temporal.activities.db_activity import log_message

        r1 = await log_message(
            MessageRecord(session_id="s10", role="user", content="First")
        )
        r2 = await log_message(
            MessageRecord(session_id="s10", role="assistant", content="Second")
        )
        assert r2 > r1
