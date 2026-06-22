"""
Speech-to-text for the realtime pipeline.

Reuses the same Deepgram batch `transcribe_file` call as
`temporal/activities/stt_activity.py`, but accepts in-memory WAV bytes
directly instead of a file path, since `realtime/session.py` already has
the finished utterance in memory and a disk round-trip just to re-read it
would add latency for no benefit. The same audio is still written to
`assets/` separately for persistence/debugging (see `session.py`).

Note: true streaming STT (Deepgram's live WebSocket API) would shave
further latency off this step, but its SDK surface changes more often
between versions than the batch endpoint, so batch-per-utterance is the
safer starting point. If you want to push further, this is the function to
swap out for a live-streaming connection.
"""

from __future__ import annotations

import logging

from deepgram import AsyncDeepgramClient

from config.settings import settings

logger = logging.getLogger(__name__)


async def transcribe(wav_bytes: bytes) -> str:
    client = AsyncDeepgramClient(api_key=settings.deepgram_api_key)

    response = await client.listen.v1.media.transcribe_file(
        request=wav_bytes,
        model=settings.deepgram_model,
        language=settings.deepgram_language,
        smart_format=True,
        punctuate=True,
    )

    channel = response.results.channels[0]
    alternative = channel.alternatives[0]
    return (alternative.transcript or "").strip()