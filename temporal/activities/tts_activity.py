"""
Activity: Text-to-Speech via Cartesia (Sonic).

STRICT BOUNDARY: this is the only module in the codebase allowed to import
and call the Cartesia SDK. It synthesizes the assistant's reply to an
`.mp3` file under `assets/` using a collision-proof UUID filename, and
returns the *relative* path so it can be persisted to SQLite and rendered by
the Streamlit audio player.
"""

from __future__ import annotations

import logging
import uuid

from cartesia import AsyncCartesia
from temporalio import activity

from config.settings import settings
from config.tracing import observe
from core.assets import asset_ref
from temporal.dto import SpeechResult

logger = logging.getLogger(__name__)


@activity.defn
@observe(name="tts-cartesia")
async def synthesize_speech(text: str) -> SpeechResult:
    """Convert the assistant's reply text into a playable `.mp3` file.

    Args:
        text: The assistant's reply, as produced by `generate_reply`.

    Returns:
        SpeechResult containing the generated audio's path, relative to the
        project root (e.g. ``"assets/bot_f2e4c1.mp3"``).
    """
    activity.logger.info("Synthesizing speech for reply (%d chars)", len(text))

    filename = f"bot_{uuid.uuid4().hex[:8]}.mp3"
    output_path = settings.assets_dir / filename

    client = AsyncCartesia(api_key=settings.cartesia_api_key)
    try:
        audio_chunks: list[bytes] = []
        
        # Step 1: Await the function call to get the async_generator
        audio_generator = await client.tts.bytes(
            model_id=settings.cartesia_model,
            transcript=text,
            voice={"mode": "id", "id": settings.cartesia_voice_id},
            language=settings.cartesia_language,
            output_format={
                "container": "mp3",
                "sample_rate": settings.cartesia_output_sample_rate,
                "bit_rate": settings.cartesia_output_bit_rate,
            },
        )
        
        # Step 2: Iterate over the generator using an async for loop to collect chunks
        async for chunk in audio_generator:
            # Safe checking: Handle cases whether the chunk is a dictionary or direct bytes
            if isinstance(chunk, dict) and "audio" in chunk:
                audio_chunks.append(chunk["audio"])
            elif isinstance(chunk, bytes):
                audio_chunks.append(chunk)

    finally:
        await client.close()

    # Join all chunks together and write them to the file
    output_path.write_bytes(b"".join(audio_chunks))

    return SpeechResult(audio_path=asset_ref(output_path))