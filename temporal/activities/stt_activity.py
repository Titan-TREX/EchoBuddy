"""
Activity: Speech-to-Text via Deepgram (Nova-2).

STRICT BOUNDARY: this is the only module in the codebase allowed to import
and call the Deepgram SDK. The workflow passes a local file path; this
activity reads the bytes from disk, sends them to Deepgram's pre-recorded
transcription endpoint, and returns a small, serializable result.

Any network error, timeout, or rate limit raised by the SDK propagates as an
activity failure, which Temporal retries according to the `RetryPolicy`
configured in `temporal/workflows.py`.
"""

from __future__ import annotations

import logging

from deepgram import AsyncDeepgramClient
from temporalio import activity

from config.settings import settings
from config.tracing import observe
from core.assets import asset_path
from temporal.dto import TranscriptionResult

logger = logging.getLogger(__name__)


@activity.defn
@observe(name="stt-deepgram")
async def transcribe_audio(audio_file_path: str) -> TranscriptionResult:
    """Transcribe a recorded user utterance to text using Deepgram Nova-2.

    Args:
        audio_file_path: Path (relative to the project root, or absolute)
            to a `.wav`/`.mp3` file written by the Streamlit UI.

    Returns:
        TranscriptionResult with the best transcript and its confidence
        score. The transcript may be an empty string if Deepgram detected
        no speech.

    Raises:
        FileNotFoundError: if `audio_file_path` does not exist on disk.
        Exception: any Deepgram SDK/network error - surfaced to Temporal so
            the configured RetryPolicy can take over.
    """
    activity.logger.info("Transcribing audio file: %s", audio_file_path)

    path = asset_path(settings.assets_dir, audio_file_path)  # by file name only: no path traversal
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    audio_bytes = path.read_bytes()

    client = AsyncDeepgramClient(api_key=settings.deepgram_api_key)

    response = await client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model=settings.deepgram_model,
        language=settings.deepgram_language,
        smart_format=True,
        punctuate=True,
    )

    channel = response.results.channels[0]
    alternative = channel.alternatives[0]

    transcript = (alternative.transcript or "").strip()
    confidence = float(getattr(alternative, "confidence", 0.0) or 0.0)

    if not transcript:
        activity.logger.warning("Deepgram returned an empty transcript for %s", audio_file_path)

    return TranscriptionResult(text=transcript, confidence=confidence)
