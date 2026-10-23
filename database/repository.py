"""
Persistence operations: the only module that issues SQL.

Both pipelines use these functions - the Temporal activities
(`temporal/activities/db_activity.py`) wrap them with `@activity.defn`, and the
realtime pipeline calls them directly. Every query uses `?` placeholders, never
f-strings or `.format()`, so there is no SQL-injection surface. Each function
is a small, idempotent-where-it-matters unit of work, which matters because
Temporal activities run with at-least-once semantics.
"""

from __future__ import annotations

from database.connection import get_connection


async def ensure_user_and_session(user_id: str, session_id: str, username: str) -> None:
    """Idempotently make sure the user and session rows exist (`INSERT OR IGNORE`)."""
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
    """The last `limit` messages of a session as `{"role", "content"}` dicts, oldest first."""
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
) -> int:
    """Persist one chat message. Returns its row id (-1 if the driver reports none)."""
    async with get_connection() as db:
        cursor = await db.execute(
            """
            INSERT INTO messages (session_id, role, content, audio_path)
            VALUES (?, ?, ?, ?);
            """,
            (session_id, role, content, audio_path),
        )
        await db.commit()
        return cursor.lastrowid if cursor.lastrowid is not None else -1
