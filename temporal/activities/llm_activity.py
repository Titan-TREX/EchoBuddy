"""
Activity: Conversational "brain" via OpenRouter, traced with Langfuse.

STRICT BOUNDARY: this is the only module in the codebase allowed to call the
LLM. It receives the latest user utterance plus a short rolling history
(fetched by `db_activity.fetch_recent_history`) and returns the assistant's
reply as plain text, ready for text-to-speech.

OpenRouter exposes an OpenAI-compatible API, so we use the Langfuse drop-in
`AsyncOpenAI` wrapper, which automatically records prompts, replies, latency,
token usage and cost as a Langfuse generation.
"""

from __future__ import annotations

import logging

from langfuse import Langfuse
from langfuse.openai import AsyncOpenAI
from temporalio import activity

from config.settings import settings
from temporal.dto import ConversationTurn, LLMResult

logger = logging.getLogger(__name__)

_ROLE_MAP = {"user": "user", "assistant": "assistant"}

_FALLBACK_REPLY = "Sorry, I didn't quite catch that. Could you say it again?"

# Langfuse reads os.environ, but our config lives in `settings`, so register the
# client explicitly. The langfuse.openai wrapper picks up this singleton.
if settings.langfuse_public_key and settings.langfuse_secret_key:
    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
    )
else:
    logger.warning("Langfuse keys not set; LLM calls will not be traced.")


@activity.defn
async def generate_reply(user_text: str, history: list[ConversationTurn]) -> LLMResult:
    """Generate the assistant's next reply using OpenRouter."""
    activity.logger.info("Generating LLM reply for: %r", user_text)

    client = AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )

    messages = [
        {"role": _ROLE_MAP.get(turn.role, "user"), "content": turn.content}
        for turn in history[-settings.gemini_max_history_turns :]
    ]
    messages.append({"role": "user", "content": user_text})

    response = await client.chat.completions.create(
        model=settings.openrouter_model,
        messages=messages,
        temperature=0.7,
        name="voice-reply",
    )

    reply_text = (response.choices[0].message.content or "").strip() or _FALLBACK_REPLY

    return LLMResult(text=reply_text)
