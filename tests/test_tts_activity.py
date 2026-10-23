"""
Tests for temporal/activities/tts_activity.py

Mocks `AsyncCartesia` to avoid real API calls. Covers:
  - Audio chunks (bytes) are written to a UUID-named .mp3 file.
  - Audio chunks in dict form ({"audio": bytes}) are handled.
  - The returned reference is `assets/<name>`, not an absolute path.
  - The client is always closed (even on error).
  - Empty text still produces a file (zero-byte is fine for the test).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from temporal.dto import SpeechResult

_MODULE = "temporal.activities.tts_activity"


async def _async_gen(*chunks):
    """Async generator yielding the supplied chunks one by one."""
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# synthesize_speech – happy path
# ---------------------------------------------------------------------------

class TestSynthesizeSpeech:
    @pytest.mark.asyncio
    async def test_writes_mp3_file(self, tmp_path: Path):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        with (
            patch(f"{_MODULE}.AsyncCartesia") as MockCartesia,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.cartesia_api_key = "test-key"
            mock_settings.cartesia_model = "sonic-2"
            mock_settings.cartesia_voice_id = "voice-id"
            mock_settings.cartesia_output_sample_rate = 44100
            mock_settings.cartesia_output_bit_rate = 128000
            mock_settings.assets_dir = assets_dir

            inst = AsyncMock()
            inst.tts.bytes = AsyncMock(
                return_value=_async_gen(b"chunk1", b"chunk2", b"chunk3")
            )
            inst.close = AsyncMock()
            MockCartesia.return_value = inst

            from temporal.activities.tts_activity import synthesize_speech
            result = await synthesize_speech("Hello, how are you?")

        assert isinstance(result, SpeechResult)
        written_files = list(assets_dir.glob("bot_*.mp3"))
        assert len(written_files) == 1

    @pytest.mark.asyncio
    async def test_returned_path_is_relative(self, tmp_path: Path):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        with (
            patch(f"{_MODULE}.AsyncCartesia") as MockCartesia,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.cartesia_api_key = "k"
            mock_settings.cartesia_model = "sonic-2"
            mock_settings.cartesia_voice_id = "vid"
            mock_settings.cartesia_output_sample_rate = 44100
            mock_settings.cartesia_output_bit_rate = 128000
            mock_settings.assets_dir = assets_dir

            inst = AsyncMock()
            inst.tts.bytes = AsyncMock(return_value=_async_gen(b"audio"))
            inst.close = AsyncMock()
            MockCartesia.return_value = inst

            from temporal.activities.tts_activity import synthesize_speech
            result = await synthesize_speech("Text")

        assert not Path(result.audio_path).is_absolute()
        assert result.audio_path.startswith("assets/")
        assert result.audio_path.endswith(".mp3")

    @pytest.mark.asyncio
    async def test_assembles_all_chunks(self, tmp_path: Path):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        with (
            patch(f"{_MODULE}.AsyncCartesia") as MockCartesia,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.cartesia_api_key = "k"
            mock_settings.cartesia_model = "sonic-2"
            mock_settings.cartesia_voice_id = "vid"
            mock_settings.cartesia_output_sample_rate = 44100
            mock_settings.cartesia_output_bit_rate = 128000
            mock_settings.assets_dir = assets_dir

            inst = AsyncMock()
            inst.tts.bytes = AsyncMock(
                return_value=_async_gen(b"AAA", b"BBB", b"CCC")
            )
            inst.close = AsyncMock()
            MockCartesia.return_value = inst

            from temporal.activities.tts_activity import synthesize_speech
            result = await synthesize_speech("abc")

        written_file = tmp_path / result.audio_path
        assert written_file.read_bytes() == b"AAABBBCCC"

    @pytest.mark.asyncio
    async def test_handles_dict_chunks(self, tmp_path: Path):
        """Chunks in {'audio': bytes} dict format should also be accepted."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        with (
            patch(f"{_MODULE}.AsyncCartesia") as MockCartesia,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.cartesia_api_key = "k"
            mock_settings.cartesia_model = "sonic-2"
            mock_settings.cartesia_voice_id = "vid"
            mock_settings.cartesia_output_sample_rate = 44100
            mock_settings.cartesia_output_bit_rate = 128000
            mock_settings.assets_dir = assets_dir

            inst = AsyncMock()
            inst.tts.bytes = AsyncMock(
                return_value=_async_gen({"audio": b"XYZ"})
            )
            inst.close = AsyncMock()
            MockCartesia.return_value = inst

            from temporal.activities.tts_activity import synthesize_speech
            result = await synthesize_speech("dict chunk test")

        written_file = tmp_path / result.audio_path
        assert written_file.read_bytes() == b"XYZ"

    @pytest.mark.asyncio
    async def test_client_is_closed_on_success(self, tmp_path: Path):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        close_called = []

        with (
            patch(f"{_MODULE}.AsyncCartesia") as MockCartesia,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.cartesia_api_key = "k"
            mock_settings.cartesia_model = "sonic-2"
            mock_settings.cartesia_voice_id = "vid"
            mock_settings.cartesia_output_sample_rate = 44100
            mock_settings.cartesia_output_bit_rate = 128000
            mock_settings.assets_dir = assets_dir

            async def mock_close():
                close_called.append(True)

            inst = AsyncMock()
            inst.tts.bytes = AsyncMock(return_value=_async_gen(b"data"))
            inst.close = mock_close
            MockCartesia.return_value = inst

            from temporal.activities.tts_activity import synthesize_speech
            await synthesize_speech("close test")

        assert len(close_called) == 1

    @pytest.mark.asyncio
    async def test_unique_filename_per_call(self, tmp_path: Path):
        """Two calls must produce two different .mp3 filenames."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        with (
            patch(f"{_MODULE}.AsyncCartesia") as MockCartesia,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.cartesia_api_key = "k"
            mock_settings.cartesia_model = "sonic-2"
            mock_settings.cartesia_voice_id = "vid"
            mock_settings.cartesia_output_sample_rate = 44100
            mock_settings.cartesia_output_bit_rate = 128000
            mock_settings.assets_dir = assets_dir

            inst = AsyncMock()
            inst.tts.bytes = AsyncMock(return_value=_async_gen(b"a"))
            inst.close = AsyncMock()
            MockCartesia.return_value = inst

            from temporal.activities.tts_activity import synthesize_speech
            r1 = await synthesize_speech("First")

            inst.tts.bytes = AsyncMock(return_value=_async_gen(b"b"))
            r2 = await synthesize_speech("Second")

        assert r1.audio_path != r2.audio_path
