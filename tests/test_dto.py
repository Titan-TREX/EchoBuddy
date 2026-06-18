"""
Tests for temporal/dto.py

Verifies that every DTO:
  - Can be instantiated with valid fields.
  - Is immutable (frozen=True).
  - Carries correct field values.
  - Handles optional fields and defaults correctly.
"""

from __future__ import annotations

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
# ConversationTurn
# ---------------------------------------------------------------------------

class TestConversationTurn:
    def test_create_user_turn(self):
        turn = ConversationTurn(role="user", content="Hello!")
        assert turn.role == "user"
        assert turn.content == "Hello!"

    def test_create_assistant_turn(self):
        turn = ConversationTurn(role="assistant", content="Hi there!")
        assert turn.role == "assistant"
        assert turn.content == "Hi there!"

    def test_is_frozen(self):
        turn = ConversationTurn(role="user", content="Hello!")
        with pytest.raises((AttributeError, TypeError)):
            turn.role = "assistant"  # type: ignore[misc]

    def test_equality(self):
        t1 = ConversationTurn(role="user", content="Same")
        t2 = ConversationTurn(role="user", content="Same")
        assert t1 == t2

    def test_inequality(self):
        t1 = ConversationTurn(role="user", content="Hello")
        t2 = ConversationTurn(role="assistant", content="Hello")
        assert t1 != t2


# ---------------------------------------------------------------------------
# TranscriptionResult
# ---------------------------------------------------------------------------

class TestTranscriptionResult:
    def test_create_with_confidence(self):
        result = TranscriptionResult(text="Good morning", confidence=0.95)
        assert result.text == "Good morning"
        assert result.confidence == 0.95

    def test_empty_transcript(self):
        result = TranscriptionResult(text="", confidence=0.0)
        assert result.text == ""
        assert result.confidence == 0.0

    def test_is_frozen(self):
        result = TranscriptionResult(text="Hi", confidence=0.9)
        with pytest.raises((AttributeError, TypeError)):
            result.text = "Changed"  # type: ignore[misc]

    def test_high_confidence(self):
        result = TranscriptionResult(text="Perfect audio", confidence=1.0)
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# LLMResult
# ---------------------------------------------------------------------------

class TestLLMResult:
    def test_create(self):
        result = LLMResult(text="The sky is blue.")
        assert result.text == "The sky is blue."

    def test_is_frozen(self):
        result = LLMResult(text="Hello")
        with pytest.raises((AttributeError, TypeError)):
            result.text = "Mutated"  # type: ignore[misc]

    def test_multiline_text(self):
        multiline = "Line one.\nLine two.\nLine three."
        result = LLMResult(text=multiline)
        assert result.text == multiline


# ---------------------------------------------------------------------------
# SpeechResult
# ---------------------------------------------------------------------------

class TestSpeechResult:
    def test_create(self):
        result = SpeechResult(audio_path="assets/bot_abc123.mp3")
        assert result.audio_path == "assets/bot_abc123.mp3"

    def test_is_frozen(self):
        result = SpeechResult(audio_path="assets/bot_xyz.mp3")
        with pytest.raises((AttributeError, TypeError)):
            result.audio_path = "assets/other.mp3"  # type: ignore[misc]

    def test_path_format(self):
        result = SpeechResult(audio_path="assets/bot_f2e4c1.mp3")
        assert result.audio_path.startswith("assets/")
        assert result.audio_path.endswith(".mp3")


# ---------------------------------------------------------------------------
# MessageRecord
# ---------------------------------------------------------------------------

class TestMessageRecord:
    def test_create_with_audio(self):
        record = MessageRecord(
            session_id="sess-1",
            role="user",
            content="Hello",
            audio_path="assets/usr_abc.wav",
        )
        assert record.session_id == "sess-1"
        assert record.role == "user"
        assert record.content == "Hello"
        assert record.audio_path == "assets/usr_abc.wav"

    def test_audio_path_defaults_to_none(self):
        record = MessageRecord(session_id="sess-1", role="assistant", content="Hi")
        assert record.audio_path is None

    def test_is_frozen(self):
        record = MessageRecord(session_id="s", role="user", content="x")
        with pytest.raises((AttributeError, TypeError)):
            record.content = "changed"  # type: ignore[misc]

    def test_assistant_role(self):
        record = MessageRecord(
            session_id="sess-2",
            role="assistant",
            content="Here is your answer",
            audio_path="assets/bot_def.mp3",
        )
        assert record.role == "assistant"


# ---------------------------------------------------------------------------
# VoiceChatbotInput
# ---------------------------------------------------------------------------

class TestVoiceChatbotInput:
    def test_create(self):
        inp = VoiceChatbotInput(
            user_id="user-42",
            session_id="sess-42",
            input_audio_path="assets/usr_xyz.wav",
        )
        assert inp.user_id == "user-42"
        assert inp.session_id == "sess-42"
        assert inp.input_audio_path == "assets/usr_xyz.wav"

    def test_is_frozen(self):
        inp = VoiceChatbotInput(
            user_id="u", session_id="s", input_audio_path="assets/x.wav"
        )
        with pytest.raises((AttributeError, TypeError)):
            inp.user_id = "hacked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VoiceChatbotResult
# ---------------------------------------------------------------------------

class TestVoiceChatbotResult:
    def test_create_with_history(self):
        history = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
        ]
        result = VoiceChatbotResult(
            user_text="Hi",
            bot_text="Hello!",
            bot_audio_path="assets/bot_001.mp3",
            history=history,
        )
        assert result.user_text == "Hi"
        assert result.bot_text == "Hello!"
        assert result.bot_audio_path == "assets/bot_001.mp3"
        assert len(result.history) == 2

    def test_history_defaults_to_empty_list(self):
        result = VoiceChatbotResult(
            user_text="Hey",
            bot_text="Yo",
            bot_audio_path="assets/bot_002.mp3",
        )
        assert result.history == []

    def test_is_frozen(self):
        result = VoiceChatbotResult(
            user_text="a", bot_text="b", bot_audio_path="assets/c.mp3"
        )
        with pytest.raises((AttributeError, TypeError)):
            result.bot_text = "mutated"  # type: ignore[misc]
