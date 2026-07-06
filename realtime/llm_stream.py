"""
Streaming LLM with tool calling for the realtime pipeline.

Flow:
  1. First Groq call (non-streaming) with tools enabled.
     → If Groq returns tool_calls: execute each tool, collect results.
     → Send tool results back as a second Groq call (streaming).
  2. If no tools needed: stream the reply directly sentence-by-sentence.

This keeps latency low for simple questions (no tool overhead) while
enabling real capability for complex ones (search, weather, calculator...).
"""

from __future__ import annotations

import re
import logging

from groq import AsyncGroq

from config.settings import settings
from realtime.tools import TOOL_SCHEMAS, run_tool
from temporal.dto import ConversationTurn

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
_ROLE_MAP = {"user": "user", "assistant": "assistant"}


def _build_messages(user_text: str, history: list) -> list[dict]:
    msgs = [{"role": "system", "content": settings.gemini_system_prompt}]

    for turn in history[-settings.gemini_max_history_turns:]:
        # support both dict (realtime db.py) and ConversationTurn dataclass (Temporal)
        if isinstance(turn, dict):
            role, content = turn["role"], turn["content"]
        else:
            role, content = turn.role, turn.content
        msgs.append({"role": _ROLE_MAP.get(role, "user"), "content": content})

    msgs.append({"role": "user", "content": user_text})
    return msgs


def _model() -> str:
    name = getattr(settings, "gemini_model", "llama-3.3-70b-versatile")
    return "llama-3.3-70b-versatile" if "specdec" in name else name


async def stream_reply_chunks(user_text: str, history: list):
    """
    Yields (sentence: str, accumulated_full_text: str).

    If the LLM calls tools, tools are executed silently first, then the
    final answer streams as normal — the caller sees no difference.
    """
    client = AsyncGroq(api_key=settings.gemini_api_key)
    messages = _build_messages(user_text, history)
    model = _model()

    # ── Step 1: first call WITH tools, non-streaming ─────────────────────
    first = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        temperature=0.7,
        max_tokens=1024,
    )

    choice = first.choices[0]

    # ── Step 2: if tool calls requested, execute them ────────────────────
    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
        # append assistant's tool-call message
        messages.append(choice.message)

        for tc in choice.message.tool_calls:
            tool_result = await run_tool(tc.function.name, tc.function.arguments)
            logger.info("Tool %s → %s", tc.function.name, tool_result[:120])
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

        # ── Step 3: second call WITH tool results, streaming ─────────────
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            stream=True,
        )

    else:
        # No tools needed — just stream the first response
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            stream=True,
        )

    # ── Stream sentence-by-sentence ───────────────────────────────────────
    buffer = ""
    full_text = ""

    async for event in stream:
        delta = event.choices[0].delta.content or ""
        if not delta:
            continue
        buffer    += delta
        full_text += delta

        match = _SENTENCE_END.search(buffer)
        while match:
            idx      = match.end()
            sentence = buffer[:idx].strip()
            buffer   = buffer[idx:]
            if sentence:
                yield sentence, full_text
            match = _SENTENCE_END.search(buffer)

    tail = buffer.strip()
    if tail:
        yield tail, full_text