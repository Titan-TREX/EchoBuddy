"""
Activities: SQLite persistence operations.

Thin Temporal wrappers over `database/repository.py`, which holds the SQL. The
repository functions are idempotent where it matters (`INSERT OR IGNORE`),
because activities run with at-least-once semantics: a worker crash after the
write but before the result is recorded makes Temporal re-run the activity.
"""

from __future__ import annotations

from temporalio import activity

from database import repository
from temporal.dto import ConversationTurn, MessageRecord


@activity.defn
async def ensure_user_and_session(user_id: str, session_id: str, username: str) -> None:
    """Idempotently make sure the user and session rows exist.

    Args:
        user_id: Stable identifier for the end user (generated once per browser tab).
        session_id: Identifier for this conversation.
        username: Display name; the UI reuses `user_id`, but any unique string works.
    """
    activity.logger.info("Ensuring user=%s / session=%s exist", user_id, session_id)
    await repository.ensure_user_and_session(user_id, session_id, username)


@activity.defn
async def fetch_recent_history(session_id: str, limit: int) -> list[ConversationTurn]:
    """Return the most recent messages for a session, oldest first.

    Gives `generate_reply` enough context for a coherent multi-turn conversation
    without re-sending the whole transcript every turn.
    """
    activity.logger.info("Fetching last %d messages for session=%s", limit, session_id)
    rows = await repository.fetch_recent_history(session_id, limit)
    return [ConversationTurn(role=row["role"], content=row["content"]) for row in rows]


@activity.defn
async def log_message(record: MessageRecord) -> int:
    """Persist a single chat message and return its `messages.id`.

    `record.audio_path` should be a relative path under `assets/`, never raw audio.
    """
    activity.logger.info("Logging %s message for session=%s", record.role, record.session_id)
    return await repository.log_message(
        record.session_id, record.role, record.content, record.audio_path
    )
