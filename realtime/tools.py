"""
Tool definitions and executors for EchoBuddy.

Every tool has:
  - A GROQ_SCHEMA entry  (what Groq sees — decides when to call which tool)
  - An async execute_*() function (what actually runs when called)

Adding a new tool:
  1. Add its schema to TOOL_SCHEMAS
  2. Add its executor to EXECUTORS map
  3. That's it — llm_stream.py picks it up automatically.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import operator
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# ── Tool schemas (sent to Groq so it knows what's available) ────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information, news, facts, or anything "
                "that may have changed recently. Use this for questions about real "
                "events, people, places, or topics you are unsure about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for any city or location. "
                "Use this when the user asks about weather or temperature."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location, e.g. 'Lahore' or 'London, UK'.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression and return the result. "
                "Use for arithmetic, percentages, unit conversions, or any "
                "calculation the user asks for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A safe math expression, e.g. '2 ** 10', '(15 * 8) / 3', 'sqrt(144)'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Return the current date and time in UTC. Use when the user asks what time or date it is.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ── Executors ────────────────────────────────────────────────────────────────

async def execute_web_search(query: str) -> str:
    try:
        from ddgs import DDGS  # noqa: PLC0415
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No results found."
        parts = []
        for r in results:
            parts.append(f"• {r.get('title','')}: {r.get('body','')[:200]}")
        return "\n".join(parts)
    except Exception as exc:
        logger.exception("web_search failed")
        return f"Search error: {exc}"


async def execute_get_weather(location: str) -> str:
    try:
        url = f"https://wttr.in/{location}?format=3"
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url, headers={"User-Agent": "EchoBuddy/1.0"})
            resp.raise_for_status()
            return resp.text.strip()
    except Exception as exc:
        logger.exception("get_weather failed")
        return f"Weather error: {exc}"


# Safe math evaluator — only allows numbers and approved operators/functions.
_SAFE_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.USub, ast.UAdd,
}
_SAFE_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "ceil": math.ceil, "floor": math.floor, "pi": math.pi, "e": math.e,
}

def _safe_eval(expr: str):
    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if type(node) not in _SAFE_NODES:
            raise ValueError(f"Disallowed node: {type(node).__name__}")
    return eval(  # noqa: S307
        compile(tree, "<calc>", "eval"),
        {"__builtins__": {}},
        _SAFE_FUNCS,
    )

async def execute_calculator(expression: str) -> str:
    try:
        result = _safe_eval(expression)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Calculation error: {exc}"


async def execute_get_datetime() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("Current UTC date and time: %A, %d %B %Y, %H:%M:%S UTC")


# ── Dispatcher ───────────────────────────────────────────────────────────────

EXECUTORS = {
    "web_search":        lambda args: execute_web_search(args["query"]),
    "get_weather":       lambda args: execute_get_weather(args["location"]),
    "calculator":        lambda args: execute_calculator(args["expression"]),
    "get_datetime":      lambda args: execute_get_datetime(),
}


async def run_tool(name: str, args_json: str) -> str:
    """Called by llm_stream.py when Groq decides to invoke a tool."""
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError:
        args = {}

    executor = EXECUTORS.get(name)
    if not executor:
        return f"Unknown tool: {name}"

    logger.info("Running tool %s with args %s", name, args)
    return await executor(args)