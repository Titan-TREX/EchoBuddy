"""
Streaming LLM with tool calling for the realtime pipeline.

Flow:
  1. First Groq call (non-streaming) with tools enabled.
     → If Groq returns tool_calls: execute each tool, collect results.
     → Send tool results back as a second Groq call (streaming).
  2. If no tools needed: stream the reply directly sentence-by-sentence.

This keeps latency low for simple questions (no tool overhead) while
enabling real capability for complex ones (search, weather, calculator...).

── Tool-choice reliability ───────────────────────────────────────────────
Previously `tool_choice="auto"` at `temperature=0.7` meant the *decision*
of whether to call a tool was itself subject to sampling randomness — the
same type of query (e.g. two different "what's the score of X" questions)
could non-deterministically call web_search one time and skip it the next,
producing a fabricated "I couldn't find that" answer without ever having
searched.

Two changes fix this:
  1. The first (tool-decision) call now uses a low temperature so the
     decision itself is consistent rather than randomly sampled.
  2. A keyword check (`_needs_tool`) forces `tool_choice="required"` for
     obviously time-sensitive/current-info queries (scores, dates,
     weather, prices, schedules, etc.), so the model can no longer choose
     to skip the tool and guess for these cases. Everything else still
     uses "auto" so simple/conversational turns don't pay tool overhead.
"""

from __future__ import annotations

import re
import logging

from groq import AsyncGroq

from config.settings import settings
from realtime.tools import TOOL_SCHEMAS, run_tool
from rag.retriever import retrieve_context
from temporal.dto import ConversationTurn

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
_ROLE_MAP = {"user": "user", "assistant": "assistant"}

# Temperature for the tool-decision call. Deliberately low (near-deterministic)
# since "should I call a tool, and which one" is a decision, not creative
# writing — high temperature here just makes tool use flaky/inconsistent.
_TOOL_DECISION_TEMPERATURE = 0.1

# Temperature for the final, user-facing reply. Kept conversational.
_REPLY_TEMPERATURE = 0.7

# If the user's message contains any of these, force tool_choice="required"
# on the first call so the model cannot silently skip searching and guess
# instead. Keep this list lowercase; matching is done on lowercased text.
_FORCE_TOOL_KEYWORDS = [
    "score", "scored", "result", "results",
    "date", "today", "tomorrow", "yesterday", "time",
    "weather", "temperature", "forecast", "rain", "sunny",
    "current", "latest", "live", "now",
    "price", "rate", "exchange", "dollar", "usd", "pkr",
    "match", "matches", "schedule", "fixture", "fixtures",
    "news", "happened", "update",
]


def _needs_tool(user_text: str) -> bool:
    """Heuristic: does this query smell like it needs live/current info?"""
    text_lower = user_text.lower()
    return any(kw in text_lower for kw in _FORCE_TOOL_KEYWORDS)


def _build_messages(user_text: str, history: list, system_prompt: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt or settings.gemini_system_prompt}]

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
    model  = _model()

    # ── RAG: always inject relevant doc context into system prompt ────────
    # More reliable than a tool call — Groq sees it as part of the prompt,
    # so it cannot fail due to tool_use_failed errors.
    rag_context  = retrieve_context(user_text)
    system_prompt = settings.gemini_system_prompt
    if rag_context:
        system_prompt += (
            "\n\nThe user has uploaded documents. "
            "Use the excerpts below to answer questions about them accurately:\n\n"
            + rag_context
        )
        logger.info("RAG context injected (%d chars)", len(rag_context))

    messages = _build_messages(user_text, history, system_prompt)

    # ── Decide how strongly to push tool use for this turn ────────────────
    # "required" forces the model to call *some* tool rather than skip
    # straight to a (possibly fabricated) answer; "auto" leaves it free for
    # turns that don't look time-sensitive, so plain chit-chat stays fast.
    forced = _needs_tool(user_text)
    tool_choice_param = "required" if forced else "auto"
    if forced:
        logger.info("Forcing tool_choice=required for query: %r", user_text)

    # ── Step 1: first call WITH tools, non-streaming ─────────────────────
    # Groq occasionally returns tool_use_failed (400) with malformed JSON.
    # Catch it and fall through to a plain streaming call without tools.
    tool_choice = None
    try:
        first = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice=tool_choice_param,
            temperature=_TOOL_DECISION_TEMPERATURE,
            max_tokens=1024,
        )
        tool_choice = first.choices[0]
    except Exception as exc:
        logger.warning("Tool-call pass failed (%s) — falling back to no-tools", exc)
        # If we forced "required" and Groq rejected the call for some reason
        # (e.g. it genuinely has no suitable tool to call), retry once with
        # "auto" so the turn doesn't hard-fail into a no-tools guess.
        if forced:
            try:
                first = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=_TOOL_DECISION_TEMPERATURE,
                    max_tokens=1024,
                )
                tool_choice = first.choices[0]
            except Exception as exc2:
                logger.warning("Retry with tool_choice=auto also failed (%s)", exc2)

    # ── Step 2: if tool calls requested, execute them ────────────────────
    if tool_choice and tool_choice.finish_reason == "tool_calls" and tool_choice.message.tool_calls:
        assistant_msg = {
            "role": "assistant",
            "content": tool_choice.message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_choice.message.tool_calls
            ],
        }
        messages.append(assistant_msg)

        for tc in tool_choice.message.tool_calls:
            try:
                tool_result = await run_tool(tc.function.name, tc.function.arguments)
                logger.info("Tool %s → %s", tc.function.name, tool_result[:120])
            except Exception as exc:
                tool_result = f"Tool error: {exc}"
                logger.warning("Tool %s failed: %s", tc.function.name, exc)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

        # ── Step 3: second call WITH tool results, streaming ──────────────
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=_REPLY_TEMPERATURE,
            stream=True,
        )

    else:
        # No tools needed (or tool call failed/unavailable) — stream directly
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=_REPLY_TEMPERATURE,
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