"""
Tests for realtime/llm_stream.py with the LLM client mocked (no network).

Covers keyword matching, message building, the tool-call flow, and the
graceful-failure behaviour (provider errors must never become a 500).
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from config.settings import settings
from core import prompting, tool_loop
from core.prompting import build_messages, needs_tool
from realtime import llm_stream
from realtime.llm_stream import stream_reply_chunks
from temporal.dto import ConversationTurn


class TestNeedsTool:
    @pytest.mark.parametrize(
        "text",
        ["What's the weather in Lahore?", "who won the match", "tell me the news", "what time is it now"],
    )
    def test_time_sensitive_queries_force_a_tool(self, text):
        assert needs_tool(text)

    @pytest.mark.parametrize(
        "text",
        # Substring matching used to flag all of these ("know" ~ "now", "generate" ~ "rate"...).
        ["I know that already", "please generate a poem", "tell me a joke", "hello there", "explain recursion"],
    )
    def test_ordinary_queries_do_not(self, text):
        assert not needs_tool(text)

    def test_keywords_come_from_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "force_tool_keywords", ["banana"])
        prompting._force_tool_pattern.cache_clear()
        try:
            assert needs_tool("I like a Banana")
            assert not needs_tool("what's the weather")
        finally:
            monkeypatch.undo()
            prompting._force_tool_pattern.cache_clear()


class TestBuildMessages:
    def test_system_prompt_history_then_user(self):
        history = [{"role": "user", "content": "hi"}, ConversationTurn(role="assistant", content="hello")]
        msgs = build_messages("how are you", history, "SYS")
        assert msgs == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you"},
        ]

    def test_history_is_capped(self, monkeypatch):
        monkeypatch.setattr(settings, "history_max_turns", 2)
        history = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        contents = [m["content"] for m in build_messages("q", history)]
        assert contents[1:] == ["m3", "m4", "q"]

    def test_default_system_prompt_comes_from_settings(self):
        assert build_messages("q", [])[0]["content"] == settings.system_prompt


def _stream_of(*pieces: str):
    async def gen():
        for piece in pieces:
            yield NS(choices=[NS(delta=NS(content=piece))])
    return gen()


def _fake_client(*, decision=None, stream=None, error=None):
    """A stand-in for the OpenAI-style client. Non-stream calls -> `decision`; stream=True -> `stream`."""
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        if kwargs.get("stream"):
            return stream
        return NS(choices=[decision])

    client = NS(chat=NS(completions=NS(create=create)))
    return client, calls


@pytest.fixture(autouse=True)
def _no_rag(monkeypatch):
    monkeypatch.setattr(prompting, "retrieve_context", lambda _q, _owner: "")


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(llm_stream, "_client", lambda: client)


async def _collect(text="tell me a joke", history=None):
    return [item async for item in stream_reply_chunks(text, history or [])]


class TestStreamReplyChunks:
    @pytest.mark.asyncio
    async def test_streams_sentence_by_sentence(self, monkeypatch):
        decision = NS(finish_reason="stop", message=NS(content="", tool_calls=None))
        client, _ = _fake_client(decision=decision, stream=_stream_of("Hello the", "re. How are", " you? Fine"))
        _patch_client(monkeypatch, client)

        out = await _collect()

        assert [sentence for sentence, _ in out] == ["Hello there.", "How are you?", "Fine"]
        assert out[-1][1] == "Hello there. How are you? Fine"

    @pytest.mark.asyncio
    async def test_tool_call_result_is_sent_back_to_the_model(self, monkeypatch):
        tool_call = NS(id="call_1", function=NS(name="calculator", arguments='{"expression": "2+2"}'))
        decision = NS(finish_reason="tool_calls", message=NS(content=None, tool_calls=[tool_call]))
        client, calls = _fake_client(decision=decision, stream=_stream_of("It is 4."))
        _patch_client(monkeypatch, client)
        run_tool = AsyncMock(return_value="2+2 = 4")
        monkeypatch.setattr(tool_loop, "run_tool", run_tool)

        out = await _collect("what is 2+2")

        run_tool.assert_awaited_once_with("calculator", '{"expression": "2+2"}')
        final_messages = calls[-1]["messages"]
        assert final_messages[-2]["tool_calls"][0]["id"] == "call_1"
        assert final_messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "2+2 = 4"}
        assert out[-1][1] == "It is 4."

    @pytest.mark.asyncio
    async def test_time_sensitive_query_forces_tool_choice_required(self, monkeypatch):
        decision = NS(finish_reason="stop", message=NS(content="", tool_calls=None))
        client, calls = _fake_client(decision=decision, stream=_stream_of("Sunny."))
        _patch_client(monkeypatch, client)

        await _collect("what's the weather today")

        assert calls[0]["tool_choice"] == "required"
        assert calls[0]["temperature"] == settings.llm_tool_temperature

    @pytest.mark.asyncio
    async def test_provider_failure_yields_fallback_instead_of_raising(self, monkeypatch):
        client, _ = _fake_client(error=RuntimeError("404 model not found"))
        _patch_client(monkeypatch, client)

        out = await _collect()

        assert out == [(settings.fallback_reply, settings.fallback_reply)]

    @pytest.mark.asyncio
    async def test_empty_model_reply_yields_fallback(self, monkeypatch):
        decision = NS(finish_reason="stop", message=NS(content="", tool_calls=None))
        client, _ = _fake_client(decision=decision, stream=_stream_of("", "  "))
        _patch_client(monkeypatch, client)

        out = await _collect()

        assert out == [(settings.fallback_reply, settings.fallback_reply)]

    @pytest.mark.asyncio
    async def test_failed_tool_pass_still_answers_without_tools(self, monkeypatch):
        calls: list[dict] = []

        async def create(**kwargs):
            calls.append(kwargs)
            if kwargs.get("stream"):
                return _stream_of("Sure thing.")
            raise RuntimeError("tool_use_failed")

        _patch_client(monkeypatch, NS(chat=NS(completions=NS(create=create))))

        out = await _collect()

        assert out[-1][1] == "Sure thing."
