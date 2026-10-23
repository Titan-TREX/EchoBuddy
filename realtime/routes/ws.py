"""
`WS /ws` - realtime voice pipeline: continuous mic streaming in, server-side
VAD turn-taking, streamed text + PCM16 audio chunks out. See
`realtime/session.py` for the turn state machine.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.settings import settings
from realtime.audio import Resampler
from realtime.security import authorize_websocket, is_valid_user_id
from realtime.session import VoiceSession

logger = logging.getLogger(__name__)

router = APIRouter()

# Sanity bounds for the sample rate a client reports.
_MIN_SOURCE_RATE = 8_000
_MAX_SOURCE_RATE = 192_000


def _source_rate(config: dict) -> int:
    """The browser's AudioContext rate (commonly 48000), validated."""
    try:
        rate = int(config.get("sampleRate", settings.ws_default_source_rate))
    except (TypeError, ValueError):
        return settings.ws_default_source_rate
    return rate if _MIN_SOURCE_RATE <= rate <= _MAX_SOURCE_RATE else settings.ws_default_source_rate


@router.websocket("/ws")
async def voice_ws(websocket: WebSocket) -> None:
    if not await authorize_websocket(websocket):
        return
    await websocket.accept()

    session_id = str(uuid.uuid4())
    session = None

    try:
        config = await websocket.receive_json()
        # The browser's persistent id, so voice chat can find the documents that
        # id uploaded. Anything missing or malformed gets a throwaway id with no documents.
        user_id = config.get("userId")
        if not is_valid_user_id(user_id):
            user_id = f"user_{uuid.uuid4().hex[:8]}"

        session = VoiceSession(
            websocket.send_json, websocket.send_bytes, user_id=user_id, session_id=session_id
        )
        await session.start()

        # The browser rarely runs at exactly 16kHz, so resample server-side to
        # the rate webrtcvad/Deepgram expect.
        resampler = Resampler(_source_rate(config), settings.audio_sample_rate)
        await websocket.send_json({"type": "ready", "ttsSampleRate": settings.tts_sample_rate})

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session.feed_pcm16(resampler.convert(message["bytes"]))
            # Text/JSON control messages are reserved for future use (e.g. mute).

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket session failed: session=%s", session_id)
    finally:
        logger.info("Client disconnected: session=%s", session_id)
        if session and session.gen_task and not session.gen_task.done():
            session.gen_task.cancel()
