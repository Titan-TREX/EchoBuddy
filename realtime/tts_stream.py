"""
Streaming TTS for the realtime pipeline.

Unlike `temporal/activities/tts_activity.py` (which collects every chunk
and writes one complete .mp3 file), this yields raw PCM16 audio chunks as
Cartesia produces them, and `session.py` forwards each chunk to the browser
the instant it arrives over the WebSocket.

We request raw PCM instead of mp3 deliberately: mp3 chunks generally can't
be decoded incrementally by the Web Audio API (a partial mp3 frame isn't a
valid file), so streamed mp3 would have to wait for the whole clip before
playback could start - defeating the purpose. Raw PCM samples can be
converted straight into an AudioBuffer and scheduled for gapless playback
as each chunk lands. See static/index.html for the playback side.

NOTE: `output_format` field/enum names below match the Cartesia Python SDK
version this was written against. If you get a validation error here,
check your installed `cartesia` package's docs for the current raw-PCM
output_format spec - third-party SDKs shift this kind of thing between
versions (see the README's "Notes on third-party SDKs" section).
"""

from __future__ import annotations

import logging

from cartesia import AsyncCartesia

from config.settings import settings
from config.tracing import get_client

logger = logging.getLogger(__name__)

async def stream_speech_chunks(text: str):
    """Yields raw PCM16LE mono audio bytes as Cartesia synthesizes `text`.

    The sample rate is `settings.tts_sample_rate`, which must match
    PLAYBACK_SAMPLE_RATE in static/index.html.
    """
    if not text.strip():
        return  # Cartesia rejects an empty transcript
    client = AsyncCartesia(api_key=settings.cartesia_api_key)
    span = get_client().start_observation(
        name="tts-cartesia",
        input=text,
        metadata={"model": settings.cartesia_model, "voice_id": settings.cartesia_voice_id},
    )
    audio_bytes = 0
    try:
        audio_generator = await client.tts.bytes(
            model_id=settings.cartesia_model,
            transcript=text,
            voice={"mode": "id", "id": settings.cartesia_voice_id},
            language=settings.cartesia_language,
            output_format={
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": settings.tts_sample_rate,
            },
        )
        async for chunk in audio_generator:
            if isinstance(chunk, dict) and "audio" in chunk:
                audio_bytes += len(chunk["audio"])
                yield chunk["audio"]
            elif isinstance(chunk, bytes):
                audio_bytes += len(chunk)
                yield chunk
    except Exception as exc:
        span.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        span.update(output={"audio_bytes": audio_bytes})
        span.end()
        await client.close()