"""Tests for core/tools.py: the calculator, tool registry and dispatcher."""

from __future__ import annotations

import json

import pytest

from core import tools
from core.tools import TOOL_SCHEMAS, TOOLS, run_tool, safe_eval


class TestCalculator:
    @pytest.mark.parametrize(
        "expression, expected",
        [
            ("2 + 3 * 4", 14),
            ("(15 * 8) / 3", 40.0),
            ("2 ** 10", 1024),
            ("-5 + 2", -3),
            ("10 // 3", 3),
            ("10 % 3", 1),
            ("sqrt(144)", 12.0),       # regression: function calls used to be rejected
            ("abs(-7)", 7),
            ("round(3.14159, 2)", 3.14),
            ("pi * 2", 6.283185307179586),  # regression: constants used to be rejected
        ],
    )
    def test_evaluates(self, expression, expected):
        assert safe_eval(expression) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo hi')",
            "open('secrets.txt')",
            "().__class__",
            "[1, 2, 3]",
            "'abc'",
            "True + 1",
            "unknown(2)",
            "x + 1",
            "sqrt(x=4)",
            "lambda: 1",
        ],
    )
    def test_rejects_anything_that_is_not_plain_math(self, expression):
        with pytest.raises(ValueError):
            safe_eval(expression)

    def test_huge_exponent_is_rejected(self):
        with pytest.raises(ValueError, match="exponent"):
            safe_eval("9 ** 9 ** 9")  # would hang the server if evaluated

    def test_overlong_expression_is_rejected(self):
        with pytest.raises(ValueError, match="too long"):
            safe_eval("1+" * 300 + "1")

    @pytest.mark.asyncio
    async def test_calculator_reports_errors_as_text(self):
        assert (await tools.calculator("1 / 0")).startswith("Calculation error")
        assert await tools.calculator("2 + 2") == "2 + 2 = 4"


class TestRegistry:
    def test_every_tool_has_a_matching_schema(self):
        assert [s["function"]["name"] for s in TOOL_SCHEMAS] == list(TOOLS)

    def test_schema_shape(self):
        schema = TOOLS["get_weather"].schema
        assert schema["type"] == "function"
        params = schema["function"]["parameters"]
        assert params["required"] == ["location"]
        assert params["properties"]["location"]["type"] == "string"

    def test_tool_without_arguments_has_empty_schema(self):
        params = TOOLS["get_datetime"].schema["function"]["parameters"]
        assert params == {"type": "object", "properties": {}, "required": []}


class TestRunTool:
    @pytest.mark.asyncio
    async def test_runs_known_tool(self):
        assert await run_tool("calculator", json.dumps({"expression": "6 * 7"})) == "6 * 7 = 42"

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        assert await run_tool("nope", "{}") == "Unknown tool: nope"

    @pytest.mark.asyncio
    async def test_invalid_json_arguments(self):
        assert "not valid JSON" in await run_tool("calculator", "{not json")

    @pytest.mark.asyncio
    async def test_non_object_arguments(self):
        assert "JSON object" in await run_tool("calculator", "[1, 2]")

    @pytest.mark.asyncio
    async def test_missing_argument_is_reported_not_raised(self):
        assert "missing argument" in await run_tool("calculator", "{}")

    @pytest.mark.asyncio
    async def test_empty_arguments_for_no_arg_tool(self):
        assert (await run_tool("get_datetime", "")).startswith("Current UTC date and time")

    @pytest.mark.asyncio
    async def test_crashing_tool_returns_error_string(self, monkeypatch):
        async def boom(**_kwargs):
            raise RuntimeError("kaboom")

        monkeypatch.setitem(TOOLS, "calculator", tools.Tool("calculator", "d", {"expression": "e"}, boom))
        assert await run_tool("calculator", json.dumps({"expression": "1"})) == "Tool error: kaboom"


class TestWeather:
    @pytest.mark.asyncio
    async def test_location_is_url_escaped(self, monkeypatch):
        seen = {}

        class FakeResponse:
            text = "Lahore: sunny\n"

            def raise_for_status(self):
                pass

        class FakeClient:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def get(self, url, headers):
                seen["url"] = url
                return FakeResponse()

        monkeypatch.setattr(tools.httpx, "AsyncClient", FakeClient)
        result = await tools.get_weather("Lahore/../admin?x=1")

        assert result == "Lahore: sunny"
        assert "/../" not in seen["url"] and "?x=1" not in seen["url"]
        assert "Lahore%2F..%2Fadmin%3Fx%3D1" in seen["url"]
