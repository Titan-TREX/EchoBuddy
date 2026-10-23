"""
Application configuration.

This module is the single source of truth for every tunable in the project:
credentials, model names, prompts, audio/VAD parameters, tool limits, RAG
settings, file paths, Temporal connection details and retry tuning. No other
module should hardcode these values or read `os.environ` directly - everything
flows through the `settings` singleton defined here.

Values are loaded from a local `.env` file (see `.env.example`) or from real
process environment variables, whichever is present. Every field can be
overridden by an environment variable with the same name in UPPER_CASE.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.prompts import (
    DEFAULT_FALLBACK_REPLY,
    DEFAULT_FORCE_TOOL_KEYWORDS,
    DEFAULT_RAG_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
)

_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Third-party API credentials ---------------------------------------------------
    # Empty by default so the app (and its tests) can import without a .env;
    # `missing_credentials()` reports anything unset at startup.
    deepgram_api_key: str = ""
    cartesia_api_key: str = ""
    # Groq key. Legacy name GEMINI_API_KEY is still accepted.
    groq_api_key: str = Field(
        default="", validation_alias=AliasChoices("GROQ_API_KEY", "GEMINI_API_KEY")
    )

    # --- Deepgram (Speech-to-Text) ------------------------------------------------------
    deepgram_model: str = "nova-2"
    deepgram_language: str = "en"

    # --- LLM (realtime + text chat, via Groq's OpenAI-compatible API) -------------------
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Legacy name GEMINI_MODEL is still accepted.
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias=AliasChoices("GROQ_MODEL", "GEMINI_MODEL"),
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        validation_alias=AliasChoices("SYSTEM_PROMPT", "GEMINI_SYSTEM_PROMPT"),
    )
    rag_prompt: str = DEFAULT_RAG_PROMPT
    fallback_reply: str = DEFAULT_FALLBACK_REPLY
    history_max_turns: int = Field(
        default=10,
        validation_alias=AliasChoices("HISTORY_MAX_TURNS", "GEMINI_MAX_HISTORY_TURNS"),
    )
    llm_tool_temperature: float = 0.1    # low: "which tool?" is a decision, not creative writing
    llm_reply_temperature: float = 0.7
    llm_max_tokens: int = 1024
    # Queries containing any of these words force the model to call a tool.
    force_tool_keywords: list[str] = Field(default_factory=lambda: list(DEFAULT_FORCE_TOOL_KEYWORDS))

    # --- LLM (Temporal batch pipeline, via OpenRouter) ----------------------------------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"

    # --- Langfuse (tracing) ---------------------------------------------------------------
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # --- Cartesia (Text-to-Speech) -------------------------------------------------------
    cartesia_model: str = "sonic-2"
    cartesia_voice_id: str = "694f9389-aac1-45b6-b726-9d9369183238"
    cartesia_language: str = "en"
    cartesia_output_sample_rate: int = 44100   # batch pipeline (.mp3)
    cartesia_output_bit_rate: int = 128000
    tts_sample_rate: int = 16000               # realtime pipeline (raw PCM); must match static/index.html

    # --- Realtime audio / voice-activity detection ---------------------------------------
    audio_sample_rate: int = 16000             # rate the VAD and Deepgram expect
    ws_default_source_rate: int = 48000        # browser AudioContext rate if the client doesn't say
    vad_frame_ms: int = 20                     # webrtcvad accepts 10, 20 or 30
    vad_aggressiveness: int = 2                # 0 (lenient) .. 3 (strict)
    vad_silence_timeout_ms: int = 500          # silence that ends an utterance
    vad_pre_speech_padding_ms: int = 300       # audio kept from before speech was detected
    vad_min_utterance_ms: int = 250            # shorter blips are ignored
    vad_min_speech_rms: int = 150              # tune to your mic/room

    # --- Tools ------------------------------------------------------------------------------
    web_search_max_results: int = 4
    web_search_snippet_chars: int = 200
    weather_url_template: str = "https://wttr.in/{location}?format=3"
    weather_timeout_seconds: float = 8.0
    http_user_agent: str = "EchoBuddy/1.0"
    calculator_max_exponent: int = 1000        # guards against 9**9**9 style DoS

    # --- RAG (document search) -------------------------------------------------------------
    chroma_path: Path = _BASE_DIR / "chroma_db"
    chroma_collection: str = "echobuddy_docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50
    rag_top_k: int = 4
    rag_max_distance: float = 1.2              # drop chunks farther than this (lower = stricter)
    rag_batch_size: int = 200                  # chunks written to Chroma per call
    upload_dir: Path = _BASE_DIR / "uploaded_docs"
    allowed_doc_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".txt"])
    max_upload_mb: int = 20
    text_chat_max_chars: int = 2000

    # --- Housekeeping -----------------------------------------------------------------------
    assets_retention_hours: float = 24         # delete generated audio older than this; 0 keeps everything
    assets_cleanup_interval_seconds: float = 3600

    # --- Health check -----------------------------------------------------------------------
    health_db_cache_seconds: float = 5.0       # reuse the database check this long, so probes stay cheap under load

    # --- Server -----------------------------------------------------------------------------
    log_level: str = "INFO"
    static_dir: Path = _BASE_DIR / "static"
    allowed_audio_extensions: list[str] = Field(
        default_factory=lambda: [".webm", ".wav", ".mp3", ".ogg", ".m4a"]
    )
    # Access control. Leave API_TOKEN empty for an open demo (rate limiting still applies);
    # set it to require "Authorization: Bearer <token>" on every API call.
    api_token: str = ""
    rate_limit_requests: int = 30              # requests allowed per client per window
    rate_limit_window_seconds: int = 60
    rate_limit_upload_cost: int = 5            # uploads embed documents, so they count extra
    trust_proxy_headers: bool = False          # True behind Railway/Render: use X-Forwarded-For as the client IP
    # Comma-separated list of allowed origins for the frontend when it's hosted
    # separately from this API (e.g. "https://echobuddy.vercel.app"). Defaults to "*"
    # for local same-origin use.
    cors_origins: str = "*"

    # --- Temporal --------------------------------------------------------------------------
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "voice-bot-queue"
    activity_start_to_close_timeout_seconds: int = 60
    activity_retry_initial_interval_seconds: int = 1
    activity_retry_backoff_coefficient: float = 2.0
    activity_retry_max_attempts: int = 4

    # --- Filesystem / Database -----------------------------------------------------------
    base_dir: Path = _BASE_DIR
    # Root for everything that must survive a redeploy (database, vector store, uploads,
    # audio). Mount ONE volume here in production (e.g. DATA_DIR=/data). Any of the four
    # paths below can still be overridden individually.
    data_dir: Path = _BASE_DIR
    assets_dir: Path = _BASE_DIR / "assets"
    database_path: Path = _BASE_DIR / "database" / "voice_chatbot.db"
    schema_path: Path = _BASE_DIR / "database" / "schema.sql"

    # ----------------------------------------------------------------------------------------

    @model_validator(mode="after")
    def _place_data_under_data_dir(self) -> "Settings":
        for field, relative in (
            ("assets_dir", "assets"),
            ("database_path", "database/voice_chatbot.db"),
            ("chroma_path", "chroma_db"),
            ("upload_dir", "uploaded_docs"),
        ):
            if field not in self.model_fields_set:
                setattr(self, field, self.data_dir / relative)
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def vad_frame_bytes(self) -> int:
        """Bytes in one VAD frame of 16-bit mono PCM."""
        return int(self.audio_sample_rate * self.vad_frame_ms / 1000) * 2

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def missing_credentials(self) -> list[str]:
        """Names of required API keys that are not set."""
        required = {
            "DEEPGRAM_API_KEY": self.deepgram_api_key,
            "GROQ_API_KEY": self.groq_api_key,
            "CARTESIA_API_KEY": self.cartesia_api_key,
        }
        return [name for name, value in required.items() if not value]


settings = Settings()

# Ensure directories that are written to at runtime exist, so no caller has to
# special-case a missing directory on first write.
for _directory in (settings.assets_dir, settings.upload_dir):
    _directory.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    """Apply `settings.log_level` to the root logger (called by each entrypoint)."""
    logging.basicConfig(level=settings.log_level.upper())
