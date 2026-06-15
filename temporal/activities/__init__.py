"""
Execution layer: granular, side-effecting activity implementations.

Each module in this package wraps exactly one external dependency (an AI
provider or the database) behind plain `async def` functions decorated with
`@activity.defn`. Workflows never import the underlying SDKs directly - they
only reference the activity functions re-exported here.

`ALL_ACTIVITIES` is the single registry consumed by `temporal/worker.py` when
constructing the Temporal `Worker`.
"""

from __future__ import annotations

from temporal.activities.db_activity import (
    ensure_user_and_session,
    fetch_recent_history,
    log_message,
)
from temporal.activities.llm_activity import generate_reply
from temporal.activities.stt_activity import transcribe_audio
from temporal.activities.tts_activity import synthesize_speech

ALL_ACTIVITIES = [
    transcribe_audio,
    ensure_user_and_session,
    fetch_recent_history,
    log_message,
    generate_reply,
    synthesize_speech,
]

__all__ = [
    "transcribe_audio",
    "ensure_user_and_session",
    "fetch_recent_history",
    "log_message",
    "generate_reply",
    "synthesize_speech",
    "ALL_ACTIVITIES",
]
