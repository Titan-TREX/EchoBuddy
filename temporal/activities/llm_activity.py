"""
Activity: Conversational "brain" via Google Gemini.

STRICT BOUNDARY: this is the only module in the codebase allowed to import
and call the `google-genai` SDK. It receives the latest user utterance plus
a short rolling history (fetched by `db_activity.fetch_recent_history`) and
returns the assistant's reply as plain text, ready for text-to-speech.
"""

from __future__ import annotations

import logging

from google import genai
from google.genai import types
from temporalio import activity

from config.settings import settings
from temporal.dto import ConversationTurn, LLMResult

logger = logging.getLogger(__name__)

# Map our internal role names onto the roles the Gemini API expects.
_ROLE_MAP = {"user": "user", "assistant": "model"}

_FALLBACK_REPLY = "Sorry, I didn't quite catch that. Could you say it again?"


@activity.defn
async def generate_reply(user_text: str, history: list[ConversationTurn]) -> LLMResult:
    """Generate the assistant's next reply.

    Args:
        user_text: The freshly transcribed user utterance.
        history: Prior turns (oldest first), used to seed Gemini's context
            window so the conversation feels continuous across turns.

    Returns:
        LLMResult wrapping the model's reply text. Falls back to a short,
        friendly clarification message if Gemini returns an empty response
        (e.g. the user's utterance was empty after transcription).
    """
    activity.logger.info("Generating LLM reply for: %r", user_text)

    client = genai.Client(api_key=settings.gemini_api_key)

    contents: list[types.Content] = [
        types.Content(role=_ROLE_MAP.get(turn.role, "user"), parts=[types.Part(text=turn.content)])
        for turn in history[-settings.gemini_max_history_turns :]
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=settings.gemini_system_prompt,
        ),
    )

    reply_text = (response.text or "").strip() or _FALLBACK_REPLY

    return LLMResult(text=reply_text)
