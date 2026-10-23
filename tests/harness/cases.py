"""Test cases for the LLM quality harness.

Each case is a LocalExperimentItem: `input` is passed to the task, and
`expected_output` / `metadata` are available to the evaluators.
"""

from __future__ import annotations

CASES = [
    {
        "input": {"user_text": "Hi there!", "history": []},
        "metadata": {"tag": "greeting"},
    },
    {
        "input": {"user_text": "What's the capital of France?", "history": []},
        "expected_output": "Paris",
        "metadata": {"tag": "fact"},
    },
    {
        "input": {"user_text": "Explain how photosynthesis works.", "history": []},
        "metadata": {"tag": "long-topic"},
    },
    {
        "input": {
            "user_text": "And what about Germany?",
            "history": [
                {"role": "user", "content": "What's the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ],
        },
        "expected_output": "Berlin",
        "metadata": {"tag": "follow-up"},
    },
    {
        "input": {"user_text": "Give me a list of ten healthy breakfast ideas.", "history": []},
        "metadata": {"tag": "list-request"},
    },
    {
        "input": {"user_text": "um... uh...", "history": []},
        "metadata": {"tag": "noisy"},
    },
]
