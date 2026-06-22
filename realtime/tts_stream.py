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

logger = logging.getLogger(__name__)

TTS_SAMPLE_RATE = 16000  # must match PLAYBACK_SAMPLE_RATE in static/index.html


async def stream_speech_chunks(text: str):
    """Yields raw PCM16LE mono audio bytes as Cartesia synthesizes `text`."""
    client = AsyncCartesia(api_key=settings.cartesia_api_key)
    try:
        audio_generator = await client.tts.bytes(
            model_id=settings.cartesia_model,
            transcript=text,
            voice={"mode": "id", "id": settings.cartesia_voice_id},
            language="en",
            output_format={
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": TTS_SAMPLE_RATE,
            },
        )
        async for chunk in audio_generator:
            if isinstance(chunk, dict) and "audio" in chunk:
                yield chunk["audio"]
            elif isinstance(chunk, bytes):
                yield chunk
    finally:
        await client.close()