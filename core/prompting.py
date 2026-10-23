"""
Building the model's input: system prompt (plus the user's own document
excerpts), conversation history, and the "does this need a tool?" heuristic.

Used by both pipelines, so the realtime path and the Temporal batch path behave
the same. Everything tunable comes from `config/settings.py`.
"""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache

from config.settings import settings
from rag.retriever import retrieve_context

logger = logging.getLogger(__name__)

_ROLE_MAP = {"user": "user", "assistant": "assistant"}


@lru_cache(maxsize=1)
def _force_tool_pattern() -> re.Pattern[str]:
    words = sorted((w.strip() for w in settings.force_tool_keywords if w.strip()), key=len, reverse=True)
    if not words:
        return re.compile(r"(?!)")  # matches nothing
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


def needs_tool(user_text: str) -> bool:
    """Heuristic: does this query smell like it needs live/current info?

    A match forces `tool_choice="required"` so the model cannot skip the tool and guess.
    """
    return bool(_force_tool_pattern().search(user_text))


def _turn_fields(turn) -> tuple[str, str]:
    """History rows are dicts (repository) or ConversationTurn objects."""
    if isinstance(turn, dict):
        return turn["role"], turn["content"]
    return turn.role, turn.content


def build_messages(user_text: str, history: list, system_prompt: str | None = None) -> list[dict]:
    """System prompt, then the most recent history turns, then the new user message."""
    messages = [{"role": "system", "content": system_prompt or settings.system_prompt}]
    for turn in history[-settings.history_max_turns:]:
        role, content = _turn_fields(turn)
        messages.append({"role": _ROLE_MAP.get(role, "user"), "content": content})
    messages.append({"role": "user", "content": user_text})
    return messages


async def system_prompt_for(user_text: str, owner: str) -> str:
    """System prompt, plus any relevant excerpts from `owner`'s uploaded documents."""
    # retrieve_context is sync (embedding + Chroma query): keep it off the event loop.
    context = await asyncio.to_thread(retrieve_context, user_text, owner)
    if not context:
        return settings.system_prompt
    logger.info("RAG context injected (%d chars)", len(context))
    return f"{settings.system_prompt}\n\n{settings.rag_prompt}\n\n{context}"
