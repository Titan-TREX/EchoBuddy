"""
Tools the LLM can call: web search, weather, calculator, date/time.

Each tool is one `Tool` entry in the registry below, keeping its schema (what
the model sees) next to its executor (what runs). To add a tool, define an
async executor, add a `Tool(...)` to `_REGISTRY`, and `llm_stream.py` picks it
up automatically. All limits and URLs come from `config/settings.py`.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from config.settings import settings
from config.tracing import observe

logger = logging.getLogger(__name__)

_MAX_EXPRESSION_CHARS = 200


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, str]        # argument name -> description (all strings, all required)
    run: Callable[..., Awaitable[str]]

    @property
    def schema(self) -> dict:
        """OpenAI-style function schema sent to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        arg: {"type": "string", "description": desc}
                        for arg, desc in self.parameters.items()
                    },
                    "required": list(self.parameters),
                },
            },
        }


# ── Executors ────────────────────────────────────────────────────────────────

async def web_search(query: str) -> str:
    def _search() -> list[dict]:
        from ddgs import DDGS  # imported lazily: heavy and only needed on first use

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=settings.web_search_max_results))

    try:
        results = await asyncio.to_thread(_search)  # DDGS is synchronous
    except Exception as exc:
        logger.exception("web_search failed")
        return f"Search error: {exc}"
    if not results:
        return "No results found."
    limit = settings.web_search_snippet_chars
    return "\n".join(f"• {r.get('title', '')}: {r.get('body', '')[:limit]}" for r in results)


async def get_weather(location: str) -> str:
    url = settings.weather_url_template.format(location=quote(location.strip(), safe=""))
    try:
        async with httpx.AsyncClient(timeout=settings.weather_timeout_seconds) as client:
            resp = await client.get(url, headers={"User-Agent": settings.http_user_agent})
            resp.raise_for_status()
            return resp.text.strip()
    except Exception as exc:
        logger.exception("get_weather failed")
        return f"Weather error: {exc}"


async def get_datetime() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("Current UTC date and time: %A, %d %B %Y, %H:%M:%S UTC")


# ── Calculator: a small whitelist evaluator (never uses eval) ────────────────

_BINARY_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "ceil": math.ceil, "floor": math.floor,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"unknown name '{node.id}'")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > settings.calculator_max_exponent:
            raise ValueError("exponent too large")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"unknown function '{node.func.id}'")
        return func(*(_evaluate(arg) for arg in node.args))
    raise ValueError(f"unsupported expression ({type(node).__name__})")


def safe_eval(expression: str) -> Any:
    """Evaluate arithmetic like '2 ** 10', '(15 * 8) / 3' or 'sqrt(144)'."""
    expression = expression.strip()
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise ValueError("expression too long")
    return _evaluate(ast.parse(expression, mode="eval"))


async def calculator(expression: str) -> str:
    try:
        return f"{expression} = {safe_eval(expression)}"
    except Exception as exc:
        return f"Calculation error: {exc}"


# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: list[Tool] = [
    Tool(
        name="web_search",
        description=(
            "Search the web for current information, news, facts, or anything "
            "that may have changed recently. Use this for questions about real "
            "events, people, places, or topics you are unsure about."
        ),
        parameters={"query": "The search query to look up."},
        run=web_search,
    ),
    Tool(
        name="get_weather",
        description=(
            "Get the current weather for any city or location. "
            "Use this when the user asks about weather or temperature."
        ),
        parameters={"location": "City name or location, e.g. 'Lahore' or 'London, UK'."},
        run=get_weather,
    ),
    Tool(
        name="calculator",
        description=(
            "Evaluate a mathematical expression and return the result. "
            "Use for arithmetic, percentages, unit conversions, or any "
            "calculation the user asks for."
        ),
        parameters={"expression": "A math expression, e.g. '2 ** 10', '(15 * 8) / 3', 'sqrt(144)'."},
        run=calculator,
    ),
    Tool(
        name="get_datetime",
        description="Return the current date and time in UTC. Use when the user asks what time or date it is.",
        parameters={},
        run=get_datetime,
    ),
]

TOOLS: dict[str, Tool] = {tool.name: tool for tool in _REGISTRY}
TOOL_SCHEMAS: list[dict] = [tool.schema for tool in _REGISTRY]


@observe(name="tool-call", as_type="tool")
async def run_tool(name: str, args_json: str) -> str:
    """Execute the tool the model asked for; always returns a string for the model."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"Unknown tool: {name}"

    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError:
        return f"Tool error: arguments for {name} were not valid JSON."
    if not isinstance(args, dict):
        return f"Tool error: arguments for {name} must be a JSON object."

    missing = [arg for arg in tool.parameters if arg not in args]
    if missing:
        return f"Tool error: missing argument(s) for {name}: {', '.join(missing)}"

    logger.info("Running tool %s with args %s", name, args)
    try:
        return await tool.run(**{arg: str(args[arg]) for arg in tool.parameters})
    except Exception as exc:
        logger.exception("Tool %s crashed", name)
        return f"Tool error: {exc}"
