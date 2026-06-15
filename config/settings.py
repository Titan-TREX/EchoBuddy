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
    gemini_model: str = "llama-3.3-70b-versatile"  
    gemini_system_prompt: str = (
        "You are a friendly, concise voice assistant. Keep replies short, "
        "natural, and conversational, since they will be converted to "
        "speech and read aloud to the user."
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
