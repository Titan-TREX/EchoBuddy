"""
Tests for temporal/activities/stt_activity.py

Mocks `AsyncDeepgramClient` so no real network calls are made.
Covers:
  - Happy-path transcription returns text + confidence.
  - Empty transcript from Deepgram is handled gracefully.
  - FileNotFoundError is raised when the audio file is missing.
  - Correct API parameters (model, language, smart_format, punctuate) are forwarded.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from temporal.dto import TranscriptionResult

_MODULE = "temporal.activities.stt_activity"


def _make_deepgram_response(transcript: str, confidence: float = 0.95) -> MagicMock:
    """Build a minimal mock that looks like a Deepgram pre-recorded response."""
    alternative = MagicMock()
    alternative.transcript = transcript
    alternative.confidence = confidence

    channel = MagicMock()
    channel.alternatives = [alternative]

    response = MagicMock()
    response.results.channels = [channel]
    return response


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestTranscribeAudio:
    @pytest.mark.asyncio
    async def test_returns_transcription_result(self, tmp_path: Path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake-audio-bytes")

        mock_response = _make_deepgram_response("Hello world", confidence=0.98)

        with (
            patch(f"{_MODULE}.AsyncDeepgramClient") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.deepgram_api_key = "test-key"
            mock_settings.deepgram_model = "nova-2"
            mock_settings.deepgram_language = "en"
            mock_settings.assets_dir = tmp_path

            instance = MagicMock()
            instance.listen.v1.media.transcribe_file = AsyncMock(return_value=mock_response)
            MockClient.return_value = instance

            from temporal.activities.stt_activity import transcribe_audio

            result = await transcribe_audio("test.wav")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.confidence == pytest.approx(0.98)

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_transcript(self, tmp_path: Path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"bytes")

        mock_response = _make_deepgram_response("  hello  ")

        with (
            patch(f"{_MODULE}.AsyncDeepgramClient") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.deepgram_api_key = "k"
            mock_settings.deepgram_model = "nova-2"
            mock_settings.deepgram_language = "en"
            mock_settings.assets_dir = tmp_path

            inst = MagicMock()
            inst.listen.v1.media.transcribe_file = AsyncMock(return_value=mock_response)
            MockClient.return_value = inst

            from temporal.activities.stt_activity import transcribe_audio
            result = await transcribe_audio("test.wav")

        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_passes_correct_model_and_language(self, tmp_path: Path):
        audio_file = tmp_path / "clip.wav"
        audio_file.write_bytes(b"audio")

        mock_response = _make_deepgram_response("text")
        call_kwargs: dict = {}

        async def capture_call(**kwargs):
            call_kwargs.update(kwargs)
            return mock_response

        with (
            patch(f"{_MODULE}.AsyncDeepgramClient") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.deepgram_api_key = "k"
            mock_settings.deepgram_model = "nova-2"
            mock_settings.deepgram_language = "en"
            mock_settings.assets_dir = tmp_path

            inst = MagicMock()
            inst.listen.v1.media.transcribe_file = capture_call
            MockClient.return_value = inst

            from temporal.activities.stt_activity import transcribe_audio
            await transcribe_audio("clip.wav")

        assert call_kwargs.get("model") == "nova-2"
        assert call_kwargs.get("language") == "en"
        assert call_kwargs.get("smart_format") is True
        assert call_kwargs.get("punctuate") is True


# ---------------------------------------------------------------------------
# Empty / missing confidence
# ---------------------------------------------------------------------------

class TestTranscribeAudioEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_transcript_returns_empty_string(self, tmp_path: Path):
        audio_file = tmp_path / "silence.wav"
        audio_file.write_bytes(b"silence")

        mock_response = _make_deepgram_response("", confidence=0.0)

        with (
            patch(f"{_MODULE}.AsyncDeepgramClient") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.deepgram_api_key = "k"
            mock_settings.deepgram_model = "nova-2"
            mock_settings.deepgram_language = "en"
            mock_settings.assets_dir = tmp_path

            inst = MagicMock()
            inst.listen.v1.media.transcribe_file = AsyncMock(return_value=mock_response)
            MockClient.return_value = inst

            from temporal.activities.stt_activity import transcribe_audio
            result = await transcribe_audio("silence.wav")

        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_missing_confidence_defaults_to_zero(self, tmp_path: Path):
        audio_file = tmp_path / "no_conf.wav"
        audio_file.write_bytes(b"audio")

        mock_response = _make_deepgram_response("some text", confidence=None)  # type: ignore

        with (
            patch(f"{_MODULE}.AsyncDeepgramClient") as MockClient,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.deepgram_api_key = "k"
            mock_settings.deepgram_model = "nova-2"
            mock_settings.deepgram_language = "en"
            mock_settings.assets_dir = tmp_path

            inst = MagicMock()
            inst.listen.v1.media.transcribe_file = AsyncMock(return_value=mock_response)
            MockClient.return_value = inst

            from temporal.activities.stt_activity import transcribe_audio
            result = await transcribe_audio("no_conf.wav")

        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# File-not-found
# ---------------------------------------------------------------------------

class TestTranscribeAudioFileNotFound:
    @pytest.mark.asyncio
    async def test_raises_file_not_found_for_missing_file(self, tmp_path: Path):
        with (
            patch(f"{_MODULE}.AsyncDeepgramClient"),
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.deepgram_api_key = "k"
            mock_settings.assets_dir = tmp_path

            from temporal.activities.stt_activity import transcribe_audio

            with pytest.raises(FileNotFoundError):
                await transcribe_audio("nonexistent_file.wav")

    @pytest.mark.asyncio
    async def test_raises_file_not_found_for_wrong_extension(self, tmp_path: Path):
        with (
            patch(f"{_MODULE}.AsyncDeepgramClient"),
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.deepgram_api_key = "k"
            mock_settings.assets_dir = tmp_path

            from temporal.activities.stt_activity import transcribe_audio

            with pytest.raises(FileNotFoundError):
                await transcribe_audio("missing.mp3")
