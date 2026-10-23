"""
The two provider-facing steps of tool calling, shared by both pipelines:

  1. `decide_tools`    - a non-streaming call with the tools attached; the model
                         answers directly or asks for tool calls.
  2. `run_tool_calls`  - execute what it asked for and append the results to the
                         conversation, ready for the final reply call.

The caller supplies the (Langfuse-wrapped) OpenAI-compatible client and model, so
the realtime path (Groq) and the batch path (OpenRouter) use identical logic.
"""

from __future__ import annotations

import logging

from config.settings import settings
from core.tools import TOOL_SCHEMAS, run_tool

logger = logging.getLogger(__name__)


async def decide_tools(client, model: str, messages: list[dict], forced: bool):
    """First call. Returns the model's choice, or None if every attempt failed.

    Providers occasionally reject tool calls (e.g. `tool_use_failed` with malformed
    JSON). The caller then falls back to a plain answer without tools.
    """
    attempts = [("tool-decision", "required" if forced else "auto")]
    if forced:
        # If "required" was rejected, retry once with "auto" before giving up on tools.
        attempts.append(("tool-decision-retry", "auto"))

    for name, tool_choice in attempts:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice=tool_choice,
                temperature=settings.llm_tool_temperature,
                max_tokens=settings.llm_max_tokens,
                name=name,
            )
            return response.choices[0]
        except Exception as exc:
            logger.warning("%s call failed (%s: %s)", name, type(exc).__name__, exc)
    return None


def wants_tools(choice) -> bool:
    return bool(choice is not None and choice.finish_reason == "tool_calls" and choice.message.tool_calls)


async def run_tool_calls(messages: list[dict], choice) -> None:
    """Append the assistant's tool request and every tool result to `messages`."""
    messages.append({
        "role": "assistant",
        "content": choice.message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in choice.message.tool_calls
        ],
    })
    for tc in choice.message.tool_calls:
        result = await run_tool(tc.function.name, tc.function.arguments)
        logger.info("Tool %s -> %s", tc.function.name, result[:120])
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
