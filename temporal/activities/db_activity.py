"""
Activity: SQLite persistence operations.

STRICT BOUNDARY: this is the only module in the codebase allowed to issue
SQL. Every query uses `?` parameter placeholders - never f-strings or
`.format()` - eliminating SQL-injection surface area entirely. Each public
function is a small, idempotent unit of work, which matters because
Temporal activities run with *at-least-once* semantics: a worker crash after
the database write but before the activity result is recorded will cause
Temporal to re-execute this activity.
"""

from __future__ import annotations

import logging

from temporalio import activity

from database.connection import get_connection
from temporal.dto import ConversationTurn, MessageRecord

logger = logging.getLogger(__name__)


@activity.defn
async def ensure_user_and_session(user_id: str, session_id: str, username: str) -> None:
    """Idempotently make sure the user and session rows exist.

    Uses ``INSERT OR IGNORE`` so re-running this activity after a worker
    crash never raises a uniqueness error - the second attempt simply
    no-ops.

    Args:
        user_id: Stable identifier for the end user (e.g. generated once per
            browser session by the Streamlit UI).
        session_id: Identifier for this conversation/session.
        username: Human-readable display name; the UI currently reuses
            `user_id` for this, but any unique string is accepted.
    """
    activity.logger.info("Ensuring user=%s / session=%s exist", user_id, session_id)

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


@activity.defn
async def fetch_recent_history(session_id: str, limit: int) -> list[ConversationTurn]:
    """Return the most recent messages for a session, oldest first.

    This gives `generate_reply` enough context to hold a coherent
    multi-turn conversation without re-sending the entire transcript on
    every turn.

    Args:
        session_id: The session to fetch history for.
        limit: Maximum number of most-recent messages to return.

    Returns:
        Up to `limit` `ConversationTurn`s, ordered oldest-first.
    """
    activity.logger.info("Fetching last %d messages for session=%s", limit, session_id)

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

    return [ConversationTurn(role=role, content=content) for role, content in reversed(rows)]


@activity.defn
async def log_message(record: MessageRecord) -> int:
    """Atomically persist a single chat message (user or assistant turn).

    Args:
        record: The message to insert. `record.audio_path` should be a
            relative path under `assets/`, never raw audio bytes.

    Returns:
        The autoincremented `messages.id` of the inserted row (-1 if the
        driver did not report a row id, which should not normally happen).
    """
    activity.logger.info("Logging %s message for session=%s", record.role, record.session_id)

    async with get_connection() as db:
        cursor = await db.execute(
            """
            INSERT INTO messages (session_id, role, content, audio_path)
            VALUES (?, ?, ?, ?);
            """,
            (record.session_id, record.role, record.content, record.audio_path),
        )
        await db.commit()
        return cursor.lastrowid if cursor.lastrowid is not None else -1
