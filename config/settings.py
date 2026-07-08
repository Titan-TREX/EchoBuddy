"""
Application configuration.

This module is the single source of truth for environment-driven
configuration (API credentials, model names, file paths, Temporal
connection details, and activity retry/timeout tuning). No other module
should call `os.environ` directly - everything flows through the `settings`
singleton defined here.

Values are loaded from a local `.env` file (see `.env.example`) or from real
process environment variables, whichever is present. See `pydantic-settings`
docs for precedence rules.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Third-party API credentials ---------------------------------------------------
    deepgram_api_key: str = "REMOVED-LEAKED-DEEPGRAM-KEY"
    gemini_api_key: str = "REMOVED-LEAKED-GROQ-KEY"
    cartesia_api_key: str = "REMOVED-LEAKED-CARTESIA-KEY"

    # --- Deepgram (Speech-to-Text) ------------------------------------------------------
    deepgram_model: str = "nova-2"
    deepgram_language: str = "en"

    # --- Gemini (LLM "brain") -----------------------------------------------------------
    gemini_model: str = "llama-3.1-8b-instant" 
    gemini_system_prompt: str = (
    "You are a friendly, concise voice assistant. Your replies are converted "
    "directly to speech and read aloud — so ONLY output the final spoken "
    "answer. Never output notes, disclaimers, reasoning, or comments about "
    "your own tools, functions, or process. The user must never hear phrases "
    "like 'I'm assuming', 'the function returns', 'this might be mock data', "
    "or anything that describes how you work internally.\n\n"
    "TOOLS: You have exactly four tools — web_search, get_weather, "
    "calculator, get_datetime. These are your ONLY source of real-world, "
    "current, or time-sensitive information. You have no other knowledge "
    "of the current date, time, weather, scores, or live events — your "
    "training data is outdated for all of these.\n\n"
    "RULE 0 — MANDATORY TOOL USE: For any question about current events, "
    "scores, dates, schedules, weather, or prices, you are REQUIRED to call "
    "a tool before responding. Responding with 'I couldn't find' or similar "
    "WITHOUT first calling web_search is a critical failure. If you skip "
    "calling the tool, you are guessing — which is not allowed."
    "RULE 1 — Always use tools for current info: For any question about "
    "today's date, current time, weather, live scores, schedules, prices, "
    "or recent events, you MUST call the matching tool. Never answer such "
    "questions from memory or guesswork.\n\n"
    "RULE 2 — Trust and use the tool result directly: Whatever the tool "
    "returns IS the real, correct, current answer. Read it and state it "
    "plainly to the user in natural spoken language. Example: if "
    "get_datetime returns 'Current UTC date and time: Wednesday, 08 July "
    "2026, 14:32:10 UTC', your entire reply should be something like "
    "'It's Wednesday, July 8th, 2026.' Nothing more, nothing about how you "
    "got it.\n\n"
    "RULE 3 — Search efficiently: Call web_search at most twice per "
    "question. Use the first useful result. If needed, try one differently "
    "worded query, then answer with the best information found — do not "
    "search a third time.\n\n"
    "RULE 4 — Handle errors plainly: If a tool result is an error message, "
    "tell the user briefly that the information isn't available right now "
    "— do not invent a value and do not explain the technical error.\n\n"
    "RULE 5 — No meta-talk, ever: Never mention that you are 'using a "
    "tool', 'calling a function', or that a result 'might be simulated'. "
    "Just answer as if you naturally know the current, real information."
)
    gemini_max_history_turns: int = 10

    # --- Cartesia (Text-to-Speech) -------------------------------------------------------
    cartesia_model: str = "sonic-2"
    cartesia_voice_id: str = "694f9389-aac1-45b6-b726-9d9369183238"
    cartesia_output_sample_rate: int = 44100
    cartesia_output_bit_rate: int = 128000

    # --- Temporal --------------------------------------------------------------------------
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "voice-bot-queue"

    # --- Filesystem / Database -----------------------------------------------------------
    base_dir: Path = Path(__file__).resolve().parent.parent
    assets_dir: Path = base_dir / "assets"
    database_path: Path = base_dir / "database" / "voice_chatbot.db"
    schema_path: Path = base_dir / "database" / "schema.sql"

    # --- Activity timeout / retry tuning --------------------------------------------------
    activity_start_to_close_timeout_seconds: int = 60
    activity_retry_initial_interval_seconds: int = 1
    activity_retry_backoff_coefficient: float = 2.0
    activity_retry_max_attempts: int = 4


settings = Settings()

# Ensure the transient asset directory exists at import time so activities
# never have to special-case a missing directory on first write.
settings.assets_dir.mkdir(parents=True, exist_ok=True)
