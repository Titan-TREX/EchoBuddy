"""
Tests for temporal/activities/llm_activity.py (the batch pipeline's LLM step).

The OpenRouter client is replaced by a fake, so there are no network calls. Covers:
message assembly (system prompt, history cap, user last), tool calling, answers from
the user's own documents, graceful fallbacks, and that the asking user's id reaches
the document search.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from config.settings import settings
from core import prompting, tool_loop
from temporal.activities import llm_activity
from temporal.dto import ConversationTurn, LLMResult

OWNER = "owner-aaaa1111"


def _choice(content: str | None = "", tool_calls=None) -> NS:
    return NS(
        finish_reason="tool_calls" if tool_calls else "stop",
        message=NS(content=content, tool_calls=tool_calls),
    )


class FakeClient:
    """Stand-in for the OpenAI-style client; replays queued responses and records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return NS(choices=[item])


@pytest.fixture
def run(monkeypatch):
    """Run generate_reply against queued fake responses; returns (result, client)."""
    monkeypatch.setattr(prompting, "retrieve_context", lambda _q, _o: "")

    async def _run(*responses, user_text="Tell me something", history=None, owner=OWNER):
        client = FakeClient(*responses)
        monkeypatch.setattr(llm_activity, "AsyncOpenAI", lambda **_kw: client)
        result = await llm_activity.generate_reply(user_text, history or [], owner)
        return result, client

    return _run


class TestMessages:
    @pytest.mark.asyncio
    async def test_system_prompt_history_then_user(self, run):
        history = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
        ]
        _, client = await run(_choice("Sure."), user_text="Next question", history=history)

        assert client.requests[0]["messages"] == [
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Next question"},
        ]

    @pytest.mark.asyncio
    async def test_history_is_capped(self, run, monkeypatch):
        monkeypatch.setattr(settings, "history_max_turns", 5)
        history = [
            ConversationTurn(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
            for i in range(15)
        ]
        _, client = await run(_choice("Sure."), history=history)

        messages = client.requests[0]["messages"]
        assert len(messages) == 7                      # system + 5 history + current
        assert messages[1]["content"] == "msg 10"      # the most recent 5 are kept

    @pytest.mark.asyncio
    async def test_uses_the_configured_model_and_tools(self, run):
        _, client = await run(_choice("Sure."))
        assert client.requests[0]["model"] == settings.openrouter_model
        assert client.requests[0]["tools"]


class TestReplies:
    @pytest.mark.asyncio
    async def test_direct_answer_needs_only_one_call(self, run):
        result, client = await run(_choice("  Hello from the bot!  "))
        assert result == LLMResult(text="Hello from the bot!")
        assert len(client.requests) == 1

    @pytest.mark.asyncio
    async def test_tool_call_result_feeds_a_final_answer(self, run, monkeypatch):
        calls = []

        async def fake_run_tool(name, arguments):
            calls.append((name, arguments))
            return "12 * 7 = 84"

        monkeypatch.setattr(tool_loop, "run_tool", fake_run_tool)
        tool_call = NS(id="call_1", function=NS(name="calculator", arguments='{"expression": "12*7"}'))

        result, client = await run(
            _choice(None, [tool_call]), _choice("It is 84."), user_text="what is 12 times 7"
        )

        assert calls == [("calculator", '{"expression": "12*7"}')]
        assert result.text == "It is 84."
        final_messages = client.requests[-1]["messages"]
        assert final_messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "12 * 7 = 84"}
        assert "tools" not in client.requests[-1]      # the final call answers, it doesn't call tools again

    @pytest.mark.asyncio
    async def test_time_sensitive_question_forces_a_tool(self, run):
        _, client = await run(_choice("Sunny."), user_text="What's the weather today?")
        assert client.requests[0]["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_falls_back_to_a_plain_answer_when_the_tool_pass_fails(self, run):
        result, client = await run(RuntimeError("tool_use_failed"), _choice("Plain answer."))
        assert result.text == "Plain answer."
        assert "tools" not in client.requests[-1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("content", ["", None, "   "])
    async def test_empty_model_reply_returns_the_fallback(self, run, content):
        result, _ = await run(_choice(content), _choice(content))
        assert result.text == settings.fallback_reply


class TestDocuments:
    @pytest.mark.asyncio
    async def test_the_askers_documents_are_searched_and_injected(self, monkeypatch):
        seen = []

        def fake_retrieve(query, owner):
            seen.append((query, owner))
            return "[Document: memo.txt — excerpt 1]\nThe vault code is 7351."

        monkeypatch.setattr(prompting, "retrieve_context", fake_retrieve)
        client = FakeClient(_choice("7351."))
        monkeypatch.setattr(llm_activity, "AsyncOpenAI", lambda **_kw: client)

        await llm_activity.generate_reply("what is the vault code", [], OWNER)

        assert seen == [("what is the vault code", OWNER)]
        system = client.requests[0]["messages"][0]["content"]
        assert system.startswith(settings.system_prompt)
        assert "vault code is 7351" in system

    @pytest.mark.asyncio
    async def test_no_owner_means_no_document_search_results(self, run, monkeypatch):
        # retrieve_context itself returns "" for an empty owner; the activity must not crash.
        result, _ = await run(_choice("Fine."), owner="")
        assert result.text == "Fine."
