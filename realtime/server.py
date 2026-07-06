"""
FastAPI server hosting BOTH voice pipelines behind a single UI.

  * `POST /api/voice-chat` - batch pipeline. Accepts one recorded audio
    clip, runs it through `VoiceChatbotWorkflow` on Temporal (Deepgram ->
    Groq -> Cartesia), and returns the full turn as JSON once it's done.
    Mirrors what `ui/app.py` (Streamlit) used to do, minus Streamlit.

  * `WS /ws` - realtime pipeline. Unchanged from before: continuous
    mic streaming, server-side VAD turn-taking, streamed text + PCM16
    audio chunks. See `realtime/session.py`.

Both pipelines share `config/settings.py`, the same SQLite database, and
the same `assets/` directory for audio files - they can run side by side
safely. The Temporal pipeline additionally requires a Temporal server
(Dockerized) and a running `temporal/worker.py` process; the realtime
pipeline needs neither.

Run:
    uv run uvicorn realtime.server:app --reload --port 8000

Then open http://localhost:8000
"""

from __future__ import annotations

import audioop
import logging
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from temporalio.client import Client

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from realtime.session import VoiceSession  # noqa: E402
from temporal.dto import VoiceChatbotInput, VoiceChatbotResult  # noqa: E402
from temporal.workflows import VoiceChatbotWorkflow  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

_STATIC_DIR = _PROJECT_ROOT / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Recorded user clips and synthesized bot replies (both pipelines) live
# under settings.assets_dir. Mounting it means the browser can <audio src>
# a bot_audio_path returned by either /api/voice-chat or the workflow
# directly, with no extra download route.
app.mount("/assets", StaticFiles(directory=str(settings.assets_dir)), name="assets")

# One Temporal client, reused across requests instead of reconnecting per
# call. Created lazily on first use and cached here.
_temporal_client: Client | None = None


async def _get_temporal_client() -> Client:
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await Client.connect(
            settings.temporal_address,
            namespace=settings.temporal_namespace,
        )
    return _temporal_client


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.post("/api/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    user_id: str = "",
    session_id: str = "",
) -> JSONResponse:
    """Batch pipeline entrypoint: one recorded clip in, one full turn out.

    The browser sends the clip recorded by `MediaRecorder` plus the
    per-browser-tab `user_id`/`session_id` it's been holding onto, so every
    turn in a tab lands in the same conversation history.
    """
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
    if not session_id:
        session_id = str(uuid.uuid4())

    suffix = Path(audio.filename or "").suffix or ".webm"
    filename = f"usr_{uuid.uuid4().hex[:8]}{suffix}"
    output_path = settings.assets_dir / filename
    output_path.write_bytes(await audio.read())
    relative_audio_path = output_path.relative_to(settings.base_dir).as_posix()

    try:
        client = await _get_temporal_client()
        workflow_input = VoiceChatbotInput(
            user_id=user_id,
            session_id=session_id,
            input_audio_path=relative_audio_path,
        )
        result: VoiceChatbotResult = await client.execute_workflow(
            VoiceChatbotWorkflow.run,
            workflow_input,
            id=f"voice-chat-{uuid.uuid4()}",
            task_queue=settings.temporal_task_queue,
        )
    except Exception:
        logger.exception("Temporal workflow failed for session=%s", session_id)
        raise HTTPException(status_code=502, detail="Voice chat workflow failed.")

    return JSONResponse(
        {
            "user_id": user_id,
            "session_id": session_id,
            "user_text": result.user_text,
            "bot_text": result.bot_text,
            "bot_audio_path": f"/{result.bot_audio_path}",
            "user_audio_path": f"/{relative_audio_path}",
        }
    )


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
            try:
                message = await websocket.receive()
            except RuntimeError:
                break  # client disconnected mid-stream, exit cleanly
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