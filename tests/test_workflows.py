"""
Tests for temporal/workflows.py  (VoiceChatbotWorkflow)

Temporal workflows run inside a sandboxed event loop managed by the SDK, so
we test them with the official `temporalio.testing.WorkflowEnvironment` (in-
process / time-skipped mode).  No Temporal server is required.

Coverage:
  - Full happy-path run returns a correct VoiceChatbotResult.
  - Activity calls happen in the correct order.
  - Workflow passes the right arguments to each activity.
  - The returned history comes from fetch_recent_history.
"""

from __future__ import annotations


import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from temporal.dto import (
    ConversationTurn,
    LLMResult,
    MessageRecord,
    SpeechResult,
    TranscriptionResult,
    VoiceChatbotInput,
    VoiceChatbotResult,
)
from temporal.workflows import VoiceChatbotWorkflow


# ---------------------------------------------------------------------------
# Stub activity implementations (return canned values instantly)
# ---------------------------------------------------------------------------

STUB_HISTORY = [
    ConversationTurn(role="user", content="Hey"),
    ConversationTurn(role="assistant", content="Hello!"),
]

CALL_LOG: list[str] = []   # shared across stubs in each test


@activity.defn(name="transcribe_audio")
async def stub_transcribe(audio_path: str) -> TranscriptionResult:
    CALL_LOG.append("transcribe_audio")
    return TranscriptionResult(text="stub transcript", confidence=0.99)


@activity.defn(name="ensure_user_and_session")
async def stub_ensure(user_id: str, session_id: str, username: str) -> None:
    CALL_LOG.append("ensure_user_and_session")


@activity.defn(name="log_message")
async def stub_log(record: MessageRecord) -> int:
    CALL_LOG.append(f"log_message:{record.role}")
    return 1


@activity.defn(name="fetch_recent_history")
async def stub_history(session_id: str, limit: int) -> list[ConversationTurn]:
    CALL_LOG.append("fetch_recent_history")
    return STUB_HISTORY


@activity.defn(name="generate_reply")
async def stub_generate(user_text: str, history: list[ConversationTurn], owner: str = "") -> LLMResult:
    CALL_LOG.append("generate_reply")
    return LLMResult(text="stub bot reply")


@activity.defn(name="synthesize_speech")
async def stub_synthesize(text: str) -> SpeechResult:
    CALL_LOG.append("synthesize_speech")
    return SpeechResult(audio_path="assets/bot_stub.mp3")


ALL_STUB_ACTIVITIES = [
    stub_transcribe,
    stub_ensure,
    stub_log,
    stub_history,
    stub_generate,
    stub_synthesize,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_call_log():
    CALL_LOG.clear()
    yield
    CALL_LOG.clear()


# ---------------------------------------------------------------------------
# Workflow tests
# ---------------------------------------------------------------------------

class TestVoiceChatbotWorkflow:
    @pytest.mark.asyncio
    async def test_happy_path_returns_correct_result(self):
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[VoiceChatbotWorkflow],
                activities=ALL_STUB_ACTIVITIES,
            ):
                result = await env.client.execute_workflow(
                    VoiceChatbotWorkflow.run,
                    VoiceChatbotInput(
                        user_id="u1",
                        session_id="s1",
                        input_audio_path="assets/usr_test.wav",
                    ),
                    id="wf-happy",
                    task_queue="test-queue",
                )

        assert isinstance(result, VoiceChatbotResult)
        assert result.user_text == "stub transcript"
        assert result.bot_text == "stub bot reply"
        assert result.bot_audio_path == "assets/bot_stub.mp3"
        assert result.history == STUB_HISTORY

    @pytest.mark.asyncio
    async def test_activities_called_in_correct_order(self):
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[VoiceChatbotWorkflow],
                activities=ALL_STUB_ACTIVITIES,
            ):
                await env.client.execute_workflow(
                    VoiceChatbotWorkflow.run,
                    VoiceChatbotInput(
                        user_id="u2",
                        session_id="s2",
                        input_audio_path="assets/usr_test2.wav",
                    ),
                    id="wf-order",
                    task_queue="test-queue",
                )

        expected_order = [
            "transcribe_audio",
            "ensure_user_and_session",
            "fetch_recent_history",  # before the user message is logged, so it isn't sent twice
            "log_message:user",
            "generate_reply",
            "synthesize_speech",
            "log_message:assistant",
        ]
        assert CALL_LOG == expected_order

    @pytest.mark.asyncio
    async def test_both_log_message_calls_are_made(self):
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[VoiceChatbotWorkflow],
                activities=ALL_STUB_ACTIVITIES,
            ):
                await env.client.execute_workflow(
                    VoiceChatbotWorkflow.run,
                    VoiceChatbotInput(
                        user_id="u3",
                        session_id="s3",
                        input_audio_path="assets/usr_test3.wav",
                    ),
                    id="wf-log",
                    task_queue="test-queue",
                )

        log_calls = [e for e in CALL_LOG if e.startswith("log_message")]
        assert log_calls == ["log_message:user", "log_message:assistant"]

    @pytest.mark.asyncio
    async def test_history_from_fetch_is_returned_in_result(self):
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[VoiceChatbotWorkflow],
                activities=ALL_STUB_ACTIVITIES,
            ):
                result = await env.client.execute_workflow(
                    VoiceChatbotWorkflow.run,
                    VoiceChatbotInput(
                        user_id="u4",
                        session_id="s4",
                        input_audio_path="assets/usr_test4.wav",
                    ),
                    id="wf-hist",
                    task_queue="test-queue",
                )

        assert result.history == STUB_HISTORY
