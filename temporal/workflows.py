"""
Orchestration layer: `VoiceChatbotWorkflow`.

STRICT LAW: this module performs NO network I/O, database queries, or API
calls of its own. It only sequences activity executions, holds workflow
state, and configures per-activity timeouts/retries. Every external
interaction lives in `temporal/activities/`.

Workflow code must also be deterministic (Temporal replays history to
recover state), so any import with side effects - reading environment
variables, touching the filesystem, etc. - is wrapped in
`workflow.unsafe.imports_passed_through()`, which tells the sandbox to load
those modules normally rather than re-executing them on every replay.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from config.settings import settings
    from temporal.activities.db_activity import (
        ensure_user_and_session,
        fetch_recent_history,
        log_message,
    )
    from temporal.activities.llm_activity import generate_reply
    from temporal.activities.stt_activity import transcribe_audio
    from temporal.activities.tts_activity import synthesize_speech
    from temporal.dto import MessageRecord, VoiceChatbotInput, VoiceChatbotResult


# Shared timeout/retry configuration for every activity in the pipeline.
#
# - `start_to_close_timeout` bounds a single activity attempt, so a hung
#   third-party API call cannot stall the workflow forever.
# - `retry_policy` retries transient failures (socket timeouts, rate limits,
#   5xx responses) with exponential backoff. Once `maximum_attempts` is
#   exhausted, the activity failure propagates to the workflow caller -
#   programmer errors are never silently masked.
_ACTIVITY_TIMEOUT = timedelta(seconds=settings.activity_start_to_close_timeout_seconds)

_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=settings.activity_retry_initial_interval_seconds),
    backoff_coefficient=settings.activity_retry_backoff_coefficient,
    maximum_attempts=settings.activity_retry_max_attempts,
)


@workflow.defn
class VoiceChatbotWorkflow:
    """Coordinates one voice round-trip: recorded audio in, spoken reply out.

    Pipeline:
        1. Transcribe the user's recorded audio (Deepgram STT).
        2. Ensure the user/session rows exist (idempotent upsert).
        3. Fetch recent conversation history for LLM context.
        4. Persist the user's transcribed message.
        5. Generate the assistant's reply (OpenRouter).
        6. Synthesize the reply to speech (Cartesia TTS).
        7. Persist the assistant's message.
        8. Return everything the UI needs to render the turn.
    """

    @workflow.run
    async def run(self, input: VoiceChatbotInput) -> VoiceChatbotResult:
        transcription = await workflow.execute_activity(
            transcribe_audio,
            input.input_audio_path,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_RETRY_POLICY,
        )

        await workflow.execute_activity(
            ensure_user_and_session,
            args=[input.user_id, input.session_id, input.user_id],
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_RETRY_POLICY,
        )

        # Fetch history BEFORE persisting this utterance: `generate_reply` appends
        # the current user message itself, so logging first would send it twice.
        history = await workflow.execute_activity(
            fetch_recent_history,
            args=[input.session_id, settings.history_max_turns],
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_RETRY_POLICY,
        )

        await workflow.execute_activity(
            log_message,
            MessageRecord(
                session_id=input.session_id,
                role="user",
                content=transcription.text,
                audio_path=input.input_audio_path,
            ),
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_RETRY_POLICY,
        )

        llm_result = await workflow.execute_activity(
            generate_reply,
            args=[transcription.text, history, input.user_id],
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_RETRY_POLICY,
        )

        speech_result = await workflow.execute_activity(
            synthesize_speech,
            llm_result.text,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_RETRY_POLICY,
        )

        await workflow.execute_activity(
            log_message,
            MessageRecord(
                session_id=input.session_id,
                role="assistant",
                content=llm_result.text,
                audio_path=speech_result.audio_path,
            ),
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_RETRY_POLICY,
        )

        return VoiceChatbotResult(
            user_text=transcription.text,
            bot_text=llm_result.text,
            bot_audio_path=speech_result.audio_path,
            history=history,
        )
