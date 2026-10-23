"""
Activity: Conversational "brain" via OpenRouter, traced with Langfuse.

STRICT BOUNDARY: this is the only module in the Temporal pipeline allowed to call
the LLM. It receives the latest user utterance, a short rolling history (fetched by
`db_activity.fetch_recent_history`) and the asking user's id, and returns the
assistant's reply as plain text, ready for text-to-speech.

It has the same abilities as the realtime pipeline - tool calling (search, weather,
calculator, date/time) and answers from the user's own uploaded documents - because
both use the shared logic in `core/`. OpenRouter exposes an OpenAI-compatible API,
so the Langfuse drop-in `AsyncOpenAI` wrapper records every call as a generation.
"""

from __future__ import annotations

import logging

from langfuse.openai import AsyncOpenAI
from temporalio import activity

from config.settings import settings
from config.tracing import init_tracing  # noqa: F401  (registers the Langfuse client)
from core.prompting import build_messages, needs_tool, system_prompt_for
from core.tool_loop import decide_tools, run_tool_calls, wants_tools
from temporal.dto import ConversationTurn, LLMResult

logger = logging.getLogger(__name__)


@activity.defn
async def generate_reply(
    user_text: str, history: list[ConversationTurn], owner: str = ""
) -> LLMResult:
    """Generate the assistant's next reply using OpenRouter, with tools and the owner's documents."""
    activity.logger.info("Generating LLM reply for: %r", user_text)

    client = AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )
    messages = build_messages(user_text, history, await system_prompt_for(user_text, owner))

    choice = await decide_tools(client, settings.openrouter_model, messages, needs_tool(user_text))
    if wants_tools(choice):
        await run_tool_calls(messages, choice)
    elif choice is not None and (choice.message.content or "").strip():
        # The model answered directly in the decision call; no need to ask twice.
        return LLMResult(text=choice.message.content.strip())

    response = await client.chat.completions.create(
        model=settings.openrouter_model,
        messages=messages,
        temperature=settings.llm_reply_temperature,
        name="voice-reply",
    )
    reply_text = (response.choices[0].message.content or "").strip() or settings.fallback_reply
    return LLMResult(text=reply_text)
