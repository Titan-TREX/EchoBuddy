"""
Activity: Conversational "brain" via Groq Cloud API — with tool calling.

STRICT BOUNDARY: this is the only module in the codebase allowed to import
and call the `groq` SDK. It receives the latest user utterance plus
a short rolling history (fetched by `db_activity.fetch_recent_history`) and
returns the assistant's reply as plain text, ready for text-to-speech.
"""

from __future__ import annotations

import logging

from groq import AsyncGroq
from temporalio import activity

from config.settings import settings
from realtime.tools import TOOL_SCHEMAS, run_tool
from rag.retriever import retrieve_context
from temporal.dto import ConversationTurn, LLMResult

logger = logging.getLogger(__name__)

_ROLE_MAP = {"user": "user", "assistant": "assistant"}
_FALLBACK_REPLY = "Sorry, I didn't quite catch that. Could you say it again?"

_SYSTEM_PROMPT = (
    "You are EchoBuddy, a helpful voice assistant. Keep replies short and conversational "
    "since they will be spoken aloud. "
    "IMPORTANT: You have access to real-time tools — always use them when needed:\n"
    "- For weather questions: ALWAYS call get_weather tool, never say you cannot access weather.\n"
    "- For current news or facts: ALWAYS call web_search tool.\n"
    "- For math calculations: ALWAYS call calculator tool.\n"
    "- For date/time questions: ALWAYS call get_datetime tool.\n"
    "- For definitions or topic summaries: ALWAYS call wikipedia_summary tool.\n"
    "Never say you lack real-time access — use the tools provided."
)


@activity.defn
async def generate_reply(user_text: str, history: list[ConversationTurn]) -> LLMResult:
    """Generate the assistant's next reply using Groq API with tool calling."""
    activity.logger.info("Generating Groq LLM reply for: %r", user_text)

    client = AsyncGroq(api_key=settings.gemini_api_key)

    # Build messages with system prompt + history + user message
    # ── RAG: inject document context into system prompt ─────────────────
    rag_context   = retrieve_context(user_text)
    system_prompt = _SYSTEM_PROMPT
    if rag_context:
        system_prompt += (
            "\n\nThe user has uploaded documents. "
            "Use the excerpts below to answer questions about them accurately:\n\n"
            + rag_context
        )

    messages = [{"role": "system", "content": system_prompt}]
    messages += [
        {"role": _ROLE_MAP.get(turn.role, "user"), "content": turn.content}
        for turn in history[-settings.gemini_max_history_turns:]
    ]
    messages.append({"role": "user", "content": user_text})

    model_name = getattr(settings, "gemini_model", "llama-3.3-70b-versatile")
    if "specdec" in model_name:
        model_name = "llama-3.3-70b-versatile"

    # ── Step 1: first call WITH tools ────────────────────────────────────
    first = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        temperature=0.7,
        max_tokens=1024,
    )

    choice = first.choices[0]
    activity.logger.info("First call finish_reason: %s", choice.finish_reason)

    # ── Step 2: execute tools if called ──────────────────────────────────
    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
        messages.append(choice.message)

        for tc in choice.message.tool_calls:
            activity.logger.info("Executing tool: %s", tc.function.name)
            tool_result = await run_tool(tc.function.name, tc.function.arguments)
            activity.logger.info("Tool result: %s", tool_result[:200])
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

        # ── Step 3: final reply with tool results ─────────────────────────
        final = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
        )
        reply_text = (final.choices[0].message.content or "").strip() or _FALLBACK_REPLY

    else:
        # No tools needed — use direct response
        reply_text = (choice.message.content or "").strip() or _FALLBACK_REPLY

    return LLMResult(text=reply_text)