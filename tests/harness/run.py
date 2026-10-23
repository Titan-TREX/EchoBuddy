"""LLM quality harness.

Runs every case in `cases.py` through `generate_reply` and scores the replies
in Langfuse (Datasets -> Experiments). Replies are read aloud by TTS, so the
rule checks enforce short, plain-text answers.

Usage (from the project root):
    uv run python -m tests.harness.run
    uv run python -m tests.harness.run --model google/gemini-2.0-flash-001 --name gemini-flash
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys

from langfuse import Evaluation, get_client

from config.settings import settings
from temporal.activities.llm_activity import generate_reply
from temporal.dto import ConversationTurn
from tests.harness.cases import CASES

MAX_WORDS = 60
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*]\s|^\s*\d+\.\s|`)", re.MULTILINE)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


async def task(*, item, **kwargs) -> str:
    data = item["input"] if isinstance(item, dict) else item.input
    history = [ConversationTurn(**turn) for turn in data["history"]]
    result = await generate_reply(data["user_text"], history)
    return result.text


def length_ok(*, output, **kwargs) -> Evaluation:
    words = len(output.split())
    return Evaluation(name="length_ok", value=float(words <= MAX_WORDS), comment=f"{words} words")


def speakable(*, output, **kwargs) -> Evaluation:
    bad = bool(_MARKDOWN.search(output) or _EMOJI.search(output))
    return Evaluation(name="speakable", value=float(not bad), comment="markdown/emoji found" if bad else None)


def contains_expected(*, output, expected_output=None, **kwargs) -> Evaluation | None:
    if not expected_output:
        return None
    hit = expected_output.lower() in output.lower()
    return Evaluation(name="contains_expected", value=float(hit), comment=f"expected {expected_output!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="OpenRouter model slug (default: settings.openrouter_model)")
    parser.add_argument("--name", help="Run name shown in Langfuse")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # report contains emoji; Windows default is cp1252

    if args.model:
        settings.openrouter_model = args.model
    run_name = args.name or settings.openrouter_model.replace("/", "-")

    langfuse = get_client()
    result = langfuse.run_experiment(
        name="voice-reply-quality",
        run_name=run_name,
        description=f"Model: {settings.openrouter_model}",
        data=CASES,
        task=task,
        evaluators=[length_ok, speakable, contains_expected],
        max_concurrency=3,
    )
    print(result.format())
    langfuse.flush()


if __name__ == "__main__":
    main()
