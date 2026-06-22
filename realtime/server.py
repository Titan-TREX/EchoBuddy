"""
FastAPI + WebSocket server for the real-time, interruptible voice pipeline.

This is additive: it lives alongside the existing Temporal + Streamlit
system in `ui/` and `temporal/` without modifying either. Run whichever one
you want to demo - they don't conflict (different ports, separate
processes), though both can write to the same `database/voice_chatbot.db`
and `assets/` since they share `config/settings.py`.

Run:
    uv run uvicorn realtime.server:app --reload --port 8000

Then open http://localhost:8000 - no Temporal server or worker needed for
this path.
"""

from __future__ import annotations

import audioop
import logging
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from realtime.session import VoiceSession  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

_STATIC_DIR = _PROJECT_ROOT / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.websocket("/ws")
async def voice_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = str(uuid.uuid4())

    async def send_json(payload: dict) -> None:
        await websocket.send_json(payload)

    async def send_bytes(data: bytes) -> None:
        await websocket.send_bytes(data)

    session = VoiceSession(send_json, send_bytes, user_id, session_id)
    await session.start()

    # The browser's AudioContext rarely runs at exactly 16kHz (commonly
    # 48000), so we resample server-side to the rate webrtcvad/Deepgram
    # expect. `audioop.ratecv` keeps conversion state across calls so
    # consecutive chunks resample cleanly at the boundaries.
    resample_state = None
    source_rate = 48000

    try:
        config = await websocket.receive_json()
        source_rate = int(config.get("sampleRate", 48000))
        await send_json({"type": "ready"})

        while True:
            message = await websocket.receive()

            if message.get("bytes") is not None:
                raw = message["bytes"]
                if source_rate != 16000:
                    converted, resample_state = audioop.ratecv(
                        raw, 2, 1, source_rate, 16000, resample_state
                    )
                else:
                    converted = raw
                await session.feed_pcm16(converted)
            # JSON/text control messages from the client are reserved for
            # future use (e.g. explicit mute) - currently none are sent.

    except WebSocketDisconnect:
        logger.info("Client disconnected: session=%s", session_id)
    finally:
        if session.gen_task and not session.gen_task.done():
            session.gen_task.cancel()