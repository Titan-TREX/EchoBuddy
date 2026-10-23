"""
Shared pytest fixtures used across the test suite.

Provides:
  - In-memory SQLite DB via aiosqlite
  - A minimal Settings override so tests never touch real .env values
  - Sample DTO instances reused in multiple test modules
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from temporal.dto import (
    ConversationTurn,
    LLMResult,
    MessageRecord,
    SpeechResult,
    TranscriptionResult,
    VoiceChatbotInput,
    VoiceChatbotResult,
)


# ---------------------------------------------------------------------------
# Event-loop fixture (required for async tests with pytest-asyncio)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Sample DTO instances
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_turn() -> ConversationTurn:
    return ConversationTurn(role="user", content="Hello there!")


@pytest.fixture
def sample_history() -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content="Hi"),
        ConversationTurn(role="assistant", content="Hello! How can I help?"),
    ]


@pytest.fixture
def sample_transcription() -> TranscriptionResult:
    return TranscriptionResult(text="What is the weather today?", confidence=0.97)


@pytest.fixture
def sample_llm_result() -> LLMResult:
    return LLMResult(text="It looks sunny and warm outside!")


@pytest.fixture
def sample_speech_result() -> SpeechResult:
    return SpeechResult(audio_path="assets/bot_abc123.mp3")


@pytest.fixture
def sample_message_record() -> MessageRecord:
    return MessageRecord(
        session_id="sess-001",
        role="user",
        content="Hello bot",
        audio_path="assets/usr_aaa111.wav",
    )


@pytest.fixture
def sample_workflow_input() -> VoiceChatbotInput:
    return VoiceChatbotInput(
        user_id="user-001",
        session_id="sess-001",
        input_audio_path="assets/usr_aaa111.wav",
    )


@pytest.fixture
def sample_workflow_result(sample_history) -> VoiceChatbotResult:
    return VoiceChatbotResult(
        user_text="What is the weather today?",
        bot_text="It looks sunny and warm outside!",
        bot_audio_path="assets/bot_abc123.mp3",
        history=sample_history,
    )


# ---------------------------------------------------------------------------
# Minimal mock settings (prevents touching .env or filesystem in unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(tmp_path: Path) -> MagicMock:
    """Return a MagicMock that mimics Settings with safe test values."""
    s = MagicMock()
    s.deepgram_api_key = "test-deepgram-key"
    s.groq_api_key = "test-groq-key"
    s.cartesia_api_key = "test-cartesia-key"
    s.deepgram_model = "nova-2"
    s.deepgram_language = "en"
    s.groq_model = "openai/gpt-oss-120b"
    s.system_prompt = "You are a helpful assistant."
    s.history_max_turns = 10
    s.cartesia_model = "sonic-2"
    s.cartesia_voice_id = "test-voice-id"
    s.cartesia_output_sample_rate = 44100
    s.cartesia_output_bit_rate = 128000
    s.temporal_address = "localhost:7233"
    s.temporal_namespace = "default"
    s.temporal_task_queue = "voice-bot-queue"
    s.base_dir = tmp_path
    s.assets_dir = tmp_path / "assets"
    s.database_path = tmp_path / "database" / "voice_chatbot.db"
    s.schema_path = tmp_path / "database" / "schema.sql"
    s.activity_start_to_close_timeout_seconds = 60
    s.activity_retry_initial_interval_seconds = 1
    s.activity_retry_backoff_coefficient = 2.0
    s.activity_retry_max_attempts = 4
    return s


@pytest.fixture(autouse=True)
def _isolate_access_control(monkeypatch):
    """Tests never inherit API_TOKEN from a developer's .env, and start with a fresh rate limiter."""
    from config.settings import settings
    from realtime.security import limiter

    monkeypatch.setattr(settings, "api_token", "")
    limiter.reset()
    yield
    limiter.reset()
