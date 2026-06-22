"""
Streaming LLM "brain" for the realtime pipeline.

Unlike `temporal/activities/llm_activity.py` (which waits for the full
Groq response before returning), this streams token deltas and yields each
*sentence* the moment it's complete. `session.py` immediately hands each
sentence to TTS, so audio for the first sentence starts generating while
the LLM is still writing the second — this is the main latency win over the
batch pipeline.
"""

from __future__ import annotations

import re

from groq import AsyncGroq

from config.settings import settings

_ROLE_MAP = {"user": "user", "assistant": "assistant"}
_SENTENCE_BOUNDARY = re.compile(r"[.!?](?:\s|$)")


async def stream_reply_chunks(user_text: str, history: list[dict]):
    """Yields (sentence: str, accumulated_full_text: str) tuples as the
    model streams its reply. `accumulated_full_text` on the last yield is
    the complete reply, suitable for logging."""
    client = AsyncGroq(api_key=settings.gemini_api_key)

    messages = [{"role": "system", "content": settings.gemini_system_prompt}]
    messages += [
        {"role": _ROLE_MAP.get(turn["role"], "user"), "content": turn["content"]}
        for turn in history[-settings.gemini_max_history_turns :]
    ]
    messages.append({"role": "user", "content": user_text})

    model_name = getattr(settings, "gemini_model", "llama-3.3-70b-versatile")
    if "specdec" in model_name:
        model_name = "llama-3.3-70b-versatile"

    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        stream=True,
    )

    buffer = ""
    full_text = ""
    async for event in stream:
        delta = event.choices[0].delta.content or ""
        if not delta:
            continue
        buffer += delta
        full_text += delta

        match = _SENTENCE_BOUNDARY.search(buffer)
        while match:
            idx = match.end()
            sentence = buffer[:idx].strip()
            buffer = buffer[idx:]
            if sentence:
                yield sentence, full_text
            match = _SENTENCE_BOUNDARY.search(buffer)

    tail = buffer.strip()
    if tail:
        yield tail, full_text