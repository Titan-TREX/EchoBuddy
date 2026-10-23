"""
Tests for temporal/activities/llm_activity.py

Mocks the Langfuse-wrapped `AsyncOpenAI` (OpenRouter) so no real API calls are made. Verifies:
  - Correct message list is assembled from history + current utterance.
  - LLMResult.text is taken from response.choices[0].message.content.
  - Fallback reply is returned when the API returns empty content.
  - History is capped at `gemini_max_history_turns`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from temporal.dto import ConversationTurn, LLMResult

_MODULE = "temporal.activities.llm_activity"


def _make_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like a chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# generate_reply – happy path
# ---------------------------------------------------------------------------

class TestGenerateReply:
    @pytest.mark.asyncio
    async def test_returns_llm_result(self):
        mock_response = _make_response("Hello from the bot!")

        with (
            patch(f"{_MODULE}.AsyncOpenAI") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.openrouter_api_key = "test-key"
            mock_settings.openrouter_model = "meta-llama/llama-3.3-70b-instruct"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.gemini_max_history_turns = 10

            instance = AsyncMock()
            instance.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = instance

            from temporal.activities.llm_activity import generate_reply

            result = await generate_reply("What time is it?", [])

        assert isinstance(result, LLMResult)
        assert result.text == "Hello from the bot!"

    @pytest.mark.asyncio
    async def test_appends_user_utterance_as_last_message(self):
        mock_response = _make_response("Sure!")
        captured_messages: list = []

        async def capture_create(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return mock_response

        with (
            patch(f"{_MODULE}.AsyncOpenAI") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.openrouter_api_key = "test-key"
            mock_settings.openrouter_model = "meta-llama/llama-3.3-70b-instruct"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.gemini_max_history_turns = 10

            instance = AsyncMock()
            instance.chat.completions.create = capture_create
            MockClient.return_value = instance

            from temporal.activities.llm_activity import generate_reply
            await generate_reply("My question", [])

        assert captured_messages[-1] == {"role": "user", "content": "My question"}

    @pytest.mark.asyncio
    async def test_history_converted_to_message_dicts(self):
        mock_response = _make_response("Got it")
        captured: list = []

        async def capture(**kwargs):
            captured.extend(kwargs["messages"])
            return mock_response

        history = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
        ]

        with (
            patch(f"{_MODULE}.AsyncOpenAI") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.openrouter_api_key = "k"
            mock_settings.openrouter_model = "meta-llama/llama-3.3-70b-instruct"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.gemini_max_history_turns = 10

            inst = AsyncMock()
            inst.chat.completions.create = capture
            MockClient.return_value = inst

            from temporal.activities.llm_activity import generate_reply
            await generate_reply("Next question", history)

        assert captured[0] == {"role": "user", "content": "Hi"}
        assert captured[1] == {"role": "assistant", "content": "Hello!"}
        assert captured[2] == {"role": "user", "content": "Next question"}

    @pytest.mark.asyncio
    async def test_history_capped_at_max_turns(self):
        mock_response = _make_response("Sure")
        captured: list = []

        async def capture(**kwargs):
            captured.extend(kwargs["messages"])
            return mock_response

        # 15 turns in history, max is 5
        history = [
            ConversationTurn(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
            for i in range(15)
        ]

        with (
            patch(f"{_MODULE}.AsyncOpenAI") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.openrouter_api_key = "k"
            mock_settings.openrouter_model = "meta-llama/llama-3.3-70b-instruct"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.gemini_max_history_turns = 5

            inst = AsyncMock()
            inst.chat.completions.create = capture
            MockClient.return_value = inst

            from temporal.activities.llm_activity import generate_reply
            await generate_reply("new question", history)

        # 5 history messages + 1 current utterance = 6 total
        assert len(captured) == 6


# ---------------------------------------------------------------------------
# generate_reply – fallback / edge cases
# ---------------------------------------------------------------------------

class TestGenerateReplyFallback:
    @pytest.mark.asyncio
    async def test_returns_fallback_on_empty_content(self):
        mock_response = _make_response("")  # empty string → fallback

        with (
            patch(f"{_MODULE}.AsyncOpenAI") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.openrouter_api_key = "k"
            mock_settings.openrouter_model = "meta-llama/llama-3.3-70b-instruct"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.gemini_max_history_turns = 10

            inst = AsyncMock()
            inst.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = inst

            from temporal.activities.llm_activity import generate_reply, _FALLBACK_REPLY
            result = await generate_reply("...", [])

        assert result.text == _FALLBACK_REPLY

    @pytest.mark.asyncio
    async def test_returns_fallback_on_none_content(self):
        mock_response = _make_response(None)  # type: ignore[arg-type]

        with (
            patch(f"{_MODULE}.AsyncOpenAI") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.openrouter_api_key = "k"
            mock_settings.openrouter_model = "meta-llama/llama-3.3-70b-instruct"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.gemini_max_history_turns = 10

            inst = AsyncMock()
            inst.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = inst

            from temporal.activities.llm_activity import generate_reply, _FALLBACK_REPLY
            result = await generate_reply("...", [])

        assert result.text == _FALLBACK_REPLY

    @pytest.mark.asyncio
    async def test_whitespace_only_reply_returns_fallback(self):
        mock_response = _make_response("   ")  # whitespace → stripped to "" → fallback

        with (
            patch(f"{_MODULE}.AsyncOpenAI") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.openrouter_api_key = "k"
            mock_settings.openrouter_model = "meta-llama/llama-3.3-70b-instruct"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.gemini_max_history_turns = 10

            inst = AsyncMock()
            inst.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = inst

            from temporal.activities.llm_activity import generate_reply, _FALLBACK_REPLY
            result = await generate_reply("...", [])

        assert result.text == _FALLBACK_REPLY
