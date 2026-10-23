"""
Streaming LLM reply for the realtime pipeline and text chat.

Flow:
  1. First call (non-streaming) with tools enabled (`core.tool_loop.decide_tools`).
     -> If the model asks for tools: run them, append the results.
  2. Final call streams the reply sentence by sentence.

Simple questions stay fast (no tool round-trip) while time-sensitive ones get
real capability (search, weather, calculator...). The prompt, history, document
excerpts and the tool-forcing heuristic come from `core.prompting`, shared with the
Temporal batch pipeline.

The client is Groq's OpenAI-compatible API wrapped by Langfuse, so every call is
recorded as a generation nested under the current turn's trace. Models,
temperatures and limits come from `config/settings.py`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from functools import lru_cache

from langfuse.openai import AsyncOpenAI

from config.settings import settings
from config.tracing import init_tracing  # noqa: F401  (registers the Langfuse client)
from core.prompting import build_messages, needs_tool, system_prompt_for
from core.tool_loop import decide_tools, run_tool_calls, wants_tools

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    """One shared client (connection pool) for the process."""
    return AsyncOpenAI(base_url=settings.groq_base_url, api_key=settings.groq_api_key)


async def stream_reply_chunks(
    user_text: str, history: list, owner: str = ""
) -> AsyncIterator[tuple[str, str]]:
    """Yields (sentence, accumulated_full_text).

    `owner` is the id of the user asking; only their uploaded documents are searched.
    If the LLM calls tools they run silently first, then the final answer streams as
    normal. This generator never raises for provider errors: on failure it yields
    `settings.fallback_reply` so the user hears something instead of a server error.
    """
    messages = build_messages(user_text, history, await system_prompt_for(user_text, owner))

    forced = needs_tool(user_text)
    if forced:
        logger.info("Forcing tool_choice=required for query: %r", user_text)
    choice = await decide_tools(_client(), settings.groq_model, messages, forced)

    full_text = ""
    buffer = ""
    try:
        if wants_tools(choice):
            await run_tool_calls(messages, choice)

        stream = await _client().chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=settings.llm_reply_temperature,
            stream=True,
            name="reply-stream",
        )

        async for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content or ""
            if not delta:
                continue
            buffer += delta
            full_text += delta

            while (match := _SENTENCE_END.search(buffer)):
                sentence, buffer = buffer[: match.end()].strip(), buffer[match.end():]
                if sentence:
                    yield sentence, full_text
    except Exception:
        logger.exception("LLM reply failed")

    tail = buffer.strip()
    if tail:
        yield tail, full_text
    elif not full_text.strip():
        yield settings.fallback_reply, settings.fallback_reply
