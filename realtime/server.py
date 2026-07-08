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
import io
import logging
import shutil
import sys
import uuid
import wave
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from temporalio.client import Client

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from rag.ingestor import ingest_file, list_docs, delete_doc  # noqa: E402
from realtime.session import VoiceSession  # noqa: E402

# ── Uploaded documents directory ──────────────────────────────────────────────
_UPLOAD_DIR = _PROJECT_ROOT / "uploaded_docs"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────
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

# =============================================================================
# RAG — Document upload & management
# =============================================================================

@app.post("/api/upload-doc")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a PDF or TXT file, save it, and ingest into ChromaDB.

    The ingested content becomes available to Groq as the 
    tool — no changes to llm_stream.py or llm_activity.py needed.
    """
    allowed = {".pdf", ".txt"}
    suffix  = Path(file.filename or "").suffix.lower()

    if suffix not in allowed:
        return JSONResponse(
            {"error": f"Unsupported file type '{suffix}'. Only PDF and TXT allowed."},
            status_code=400,
        )

    save_path = _UPLOAD_DIR / (file.filename or f"doc_{uuid.uuid4().hex[:8]}{suffix}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_file(str(save_path))
        return JSONResponse({"success": True, **result})
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/docs")
async def get_docs() -> JSONResponse:
    """Return a list of all ingested document filenames."""
    return JSONResponse({"documents": list_docs()})


@app.delete("/api/docs/{filename}")
async def remove_doc(filename: str) -> JSONResponse:
    """Delete all ChromaDB chunks for a specific document."""
    deleted = delete_doc(filename)
    return JSONResponse({"deleted_chunks": deleted, "file": filename})


# =============================================================================
# Text chat — type a message, get a text + audio reply
# =============================================================================

from pydantic import BaseModel  # noqa: E402


class TextChatRequest(BaseModel):
    text:       str
    user_id:    str = ""
    session_id: str = ""


@app.post("/api/text-chat")
async def text_chat(req: TextChatRequest) -> JSONResponse:
    """Text-in, text+audio-out endpoint.

    Uses the same realtime pipeline (llm_stream + tts_stream) so the RAG
    search_documents tool is automatically available here too.
    The reply audio is saved as a WAV file and its URL is returned alongside
    the text so the browser can autoplay it.
    """
    from realtime.db import (  # noqa: E402
        ensure_user_and_session,
        fetch_recent_history,
        log_message,
    )
    from realtime.llm_stream import stream_reply_chunks  # noqa: E402
    from realtime.tts_stream import stream_speech_chunks  # noqa: E402

    user_id    = req.user_id    or f"user_{uuid.uuid4().hex[:8]}"
    session_id = req.session_id or str(uuid.uuid4())

    await ensure_user_and_session(user_id, session_id, user_id)
    history = await fetch_recent_history(session_id, 10)
    await log_message(session_id, "user", req.text, None)

    # ── Collect full LLM reply ────────────────────────────────────────────
    full_text = ""
    async for _sentence, accumulated in stream_reply_chunks(req.text, history):
        full_text = accumulated

    # ── Synthesize reply to WAV ───────────────────────────────────────────
    pcm_chunks: list[bytes] = []
    async for chunk in stream_speech_chunks(full_text):
        pcm_chunks.append(chunk)

    audio_filename = f"bot_{uuid.uuid4().hex[:8]}.wav"
    audio_path     = settings.assets_dir / audio_filename
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"".join(pcm_chunks))
    audio_path.write_bytes(buf.getvalue())

    rel = audio_path.relative_to(settings.base_dir).as_posix()
    await log_message(session_id, "assistant", full_text, rel)

    return JSONResponse({
        "bot_text":       full_text,
        "bot_audio_path": f"/{rel}",
        "user_id":        user_id,
        "session_id":     session_id,
    })