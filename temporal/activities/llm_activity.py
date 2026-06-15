"""
Activity: Conversational "brain" via Groq Cloud API.

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
from temporal.dto import ConversationTurn, LLMResult

logger = logging.getLogger(__name__)

# Groq expects standard OpenAI roles: "user" and "assistant"
_ROLE_MAP = {"user": "user", "assistant": "assistant"}

_FALLBACK_REPLY = "Sorry, I didn't quite catch that. Could you say it again?"


@activity.defn
async def generate_reply(user_text: str, history: list[ConversationTurn]) -> LLMResult:
    """Generate the assistant's next reply using Groq API."""
    activity.logger.info("Generating Groq LLM reply for: %r", user_text)

    # Groq Client Initialization using the GEMINI_API_KEY environment variable location
    client = AsyncGroq(api_key=settings.gemini_api_key)

    # 1. Convert Chat History into the Groq messages format
    messages = [
        {
            "role": _ROLE_MAP.get(turn.role, "user"), 
            "content": turn.content
        }
        for turn in history[-settings.gemini_max_history_turns :]
    ]
    
    # 2. Append the latest user utterance to the message list
    messages.append({"role": "user", "content": user_text})

    # 3. Groq Async Chat Completion API Call
    # Dynamic settings are used here; if it fails to resolve, it falls back to 'llama-3.3-70b-versatile'.
    model_name = getattr(settings, "gemini_model", "llama-3.3-70b-versatile")
    if "specdec" in model_name:
        model_name = "llama-3.3-70b-versatile"

    response = await client.chat.completions.create(
        model=model_name,  # Stable, running Groq versatile model
        messages=messages,
        temperature=0.7,
    )

    reply_text = (response.choices[0].message.content or "").strip() or _FALLBACK_REPLY

    return LLMResult(text=reply_text)