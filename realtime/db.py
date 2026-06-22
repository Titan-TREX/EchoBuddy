"""
Direct SQLite persistence for the realtime voice pipeline.

Deliberately does NOT go through Temporal: the batch pipeline in
`temporal/activities/db_activity.py` uses `@activity.defn`, which requires
an active Temporal activity execution context (for `activity.logger`, etc.).
Calling those functions directly from a plain asyncio WebSocket handler
would fail outside that context. This module re-implements the same SQL
against the same schema, using plain `logging` instead, so it works as a
normal async function call with minimal overhead — important here since
every millisecond is in the hot path of a live conversation.

Schema (database/schema.sql) is unchanged and shared with the batch
pipeline; both systems can write to the same database safely.
"""

from __future__ import annotations

import logging

from database.connection import get_connection

logger = logging.getLogger(__name__)


async def ensure_user_and_session(user_id: str, session_id: str, username: str) -> None:
    async with get_connection() as db:
        await db.execute(
            "INSERT OR IGNORE INTO users (id, username) VALUES (?, ?);",
            (user_id, username),
        )
        await db.execute(
            "INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?);",
            (session_id, user_id),
        )
        await db.commit()


async def fetch_recent_history(session_id: str, limit: int) -> list[dict]:
    async with get_connection() as db:
        cursor = await db.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (session_id, limit),
        )
        rows = await cursor.fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


async def log_message(
    session_id: str, role: str, content: str, audio_path: str | None
) -> None:
    async with get_connection() as db:
        await db.execute(
            """
            INSERT INTO messages (session_id, role, content, audio_path)
            VALUES (?, ?, ?, ?);
            """,
            (session_id, role, content, audio_path),
        )
        await db.commit()