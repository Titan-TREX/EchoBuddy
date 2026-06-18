"""
Tests for config/settings.py

Verifies that:
  - The Settings class loads with expected defaults.
  - Environment variable overrides work correctly.
  - Path fields resolve to Path objects.
  - Assets directory is created at import time.
  - No unexpected mandatory fields are missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from config.settings import Settings


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestSettingsDefaults:
    """Settings should be usable with no .env present (defaults cover everything)."""

    def setup_method(self):
        self.s = Settings()

    def test_deepgram_model_default(self):
        assert self.s.deepgram_model == "nova-2"

    def test_deepgram_language_default(self):
        assert self.s.deepgram_language == "en"

    def test_gemini_model_default(self):
        assert self.s.gemini_model == "llama-3.3-70b-versatile"

    def test_gemini_max_history_turns_default(self):
        assert self.s.gemini_max_history_turns == 10

    def test_cartesia_model_default(self):
        assert self.s.cartesia_model == "sonic-2"

    def test_cartesia_sample_rate_default(self):
        assert self.s.cartesia_output_sample_rate == 44100

    def test_cartesia_bit_rate_default(self):
        assert self.s.cartesia_output_bit_rate == 128000

    def test_temporal_address_default(self):
        assert self.s.temporal_address == "localhost:7233"

    def test_temporal_namespace_default(self):
        assert self.s.temporal_namespace == "default"

    def test_temporal_task_queue_default(self):
        assert self.s.temporal_task_queue == "voice-bot-queue"

    def test_activity_timeout_default(self):
        assert self.s.activity_start_to_close_timeout_seconds == 60

    def test_activity_retry_max_attempts_default(self):
        assert self.s.activity_retry_max_attempts == 4

    def test_activity_retry_backoff_coefficient_default(self):
        assert self.s.activity_retry_backoff_coefficient == 2.0

    def test_activity_retry_initial_interval_default(self):
        assert self.s.activity_retry_initial_interval_seconds == 1


# ---------------------------------------------------------------------------
# Path fields
# ---------------------------------------------------------------------------

class TestSettingsPathFields:
    def setup_method(self):
        self.s = Settings()

    def test_base_dir_is_path(self):
        assert isinstance(self.s.base_dir, Path)

    def test_assets_dir_is_path(self):
        assert isinstance(self.s.assets_dir, Path)

    def test_database_path_is_path(self):
        assert isinstance(self.s.database_path, Path)

    def test_schema_path_is_path(self):
        assert isinstance(self.s.schema_path, Path)

    def test_assets_dir_under_base_dir(self):
        assert str(self.s.assets_dir).startswith(str(self.s.base_dir))

    def test_database_path_under_base_dir(self):
        assert str(self.s.database_path).startswith(str(self.s.base_dir))

    def test_database_filename(self):
        assert self.s.database_path.name == "voice_chatbot.db"

    def test_schema_filename(self):
        assert self.s.schema_path.name == "schema.sql"


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestSettingsEnvOverrides:
    def test_override_temporal_address(self):
        with patch.dict(os.environ, {"TEMPORAL_ADDRESS": "remote-host:7233"}):
            s = Settings()
            assert s.temporal_address == "remote-host:7233"

    def test_override_task_queue(self):
        with patch.dict(os.environ, {"TEMPORAL_TASK_QUEUE": "custom-queue"}):
            s = Settings()
            assert s.temporal_task_queue == "custom-queue"

    def test_override_gemini_max_history_turns(self):
        with patch.dict(os.environ, {"GEMINI_MAX_HISTORY_TURNS": "5"}):
            s = Settings()
            assert s.gemini_max_history_turns == 5

    def test_override_activity_timeout(self):
        with patch.dict(os.environ, {"ACTIVITY_START_TO_CLOSE_TIMEOUT_SECONDS": "120"}):
            s = Settings()
            assert s.activity_start_to_close_timeout_seconds == 120

    def test_override_deepgram_model(self):
        with patch.dict(os.environ, {"DEEPGRAM_MODEL": "nova-3"}):
            s = Settings()
            assert s.deepgram_model == "nova-3"

    def test_override_cartesia_model(self):
        with patch.dict(os.environ, {"CARTESIA_MODEL": "sonic-3"}):
            s = Settings()
            assert s.cartesia_model == "sonic-3"


# ---------------------------------------------------------------------------
# assets_dir creation
# ---------------------------------------------------------------------------

class TestSettingsAssetsDirCreation:
    def test_assets_dir_exists_after_import(self):
        """assets_dir.mkdir() is called at module level; it must exist."""
        s = Settings()
        assert s.assets_dir.exists()
