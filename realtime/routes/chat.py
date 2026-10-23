"""
Request/response chat endpoints.

  * `POST /api/voice-chat` - batch pipeline. One recorded clip in, one full turn
    out, orchestrated by `VoiceChatbotWorkflow` on Temporal (needs a Temporal
    server and a running `temporal/worker.py`).
  * `POST /api/text-chat` - typed message in, text + audio reply out, using the
    realtime pipeline (llm_stream + tts_stream) so tools and RAG are available.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from temporalio.client import Client

from config.settings import settings
from config.tracing import observe, propagate_attributes
from core.assets import asset_ref
from database.repository import ensure_user_and_session, fetch_recent_history, log_message
from realtime.audio import pcm_to_wav
from realtime.llm_stream import stream_reply_chunks
from realtime.security import USER_ID_REGEX, access_guard
from realtime.tts_stream import stream_speech_chunks
from temporal.dto import VoiceChatbotInput, VoiceChatbotResult
from temporal.workflows import VoiceChatbotWorkflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"], dependencies=[Depends(access_guard())])

# One Temporal client, reused across requests and created lazily on first use.
_temporal_client: Client | None = None
_temporal_lock = asyncio.Lock()


async def _get_temporal_client() -> Client:
    global _temporal_client
    async with _temporal_lock:
        if _temporal_client is None:
            _temporal_client = await Client.connect(
                settings.temporal_address, namespace=settings.temporal_namespace
            )
        return _temporal_client


# An absent user id is allowed (one is generated); a present one must have the valid shape.
_OPTIONAL_USER_ID = rf"^({USER_ID_REGEX})?$"


def _new_user_id() -> str:
    return f"user_{uuid.uuid4().hex[:8]}"


# ── Batch voice pipeline (Temporal) ──────────────────────────────────────────

@router.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    # Form fields (the browser sends multipart). As plain parameters FastAPI would read
    # them from the query string, silently ignoring them and starting a new session every turn.
    user_id: str = Form("", pattern=_OPTIONAL_USER_ID),
    session_id: str = Form("", max_length=64),
) -> JSONResponse:
    """One recorded clip in, one full turn out.

    The browser sends the clip plus the per-tab `user_id`/`session_id` it holds,
    so every turn in a tab lands in the same conversation history.
    """
    suffix = Path(audio.filename or "").suffix.lower() or ".webm"
    if suffix not in settings.allowed_audio_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type '{suffix}'.")

    user_id = user_id or _new_user_id()
    session_id = session_id or str(uuid.uuid4())

    output_path = settings.assets_dir / f"usr_{uuid.uuid4().hex[:8]}{suffix}"
    await asyncio.to_thread(output_path.write_bytes, await audio.read())
    relative_audio_path = asset_ref(output_path)

    try:
        client = await _get_temporal_client()
        result: VoiceChatbotResult = await client.execute_workflow(
            VoiceChatbotWorkflow.run,
            VoiceChatbotInput(
                user_id=user_id, session_id=session_id, input_audio_path=relative_audio_path
            ),
            id=f"voice-chat-{uuid.uuid4()}",
            task_queue=settings.temporal_task_queue,
        )
    except Exception:
        logger.exception("Temporal workflow failed for session=%s", session_id)
        raise HTTPException(status_code=502, detail="Voice chat workflow failed.")

    return JSONResponse({
        "user_id": user_id,
        "session_id": session_id,
        "user_text": result.user_text,
        "bot_text": result.bot_text,
        "bot_audio_path": f"/{result.bot_audio_path}",
        "user_audio_path": f"/{relative_audio_path}",
    })


# ── Text chat (realtime pipeline) ────────────────────────────────────────────

class TextChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=settings.text_chat_max_chars)
    user_id: str = Field(default="", pattern=_OPTIONAL_USER_ID)
    session_id: str = Field(default="", max_length=64)


@router.post("/text-chat")
async def text_chat(req: TextChatRequest) -> JSONResponse:
    """Text in, text + audio out. One Langfuse trace per turn."""
    user_id = req.user_id or _new_user_id()
    session_id = req.session_id or str(uuid.uuid4())
    with propagate_attributes(user_id=user_id, session_id=session_id):
        return await _text_chat_turn(req.text, user_id, session_id)


@observe(name="text-chat", capture_output=False)
async def _text_chat_turn(text: str, user_id: str, session_id: str) -> JSONResponse:
    await ensure_user_and_session(user_id, session_id, user_id)
    # Fetch history BEFORE logging this message: the LLM layer appends the
    # current user message itself, so logging first would send it twice.
    history = await fetch_recent_history(session_id, settings.history_max_turns)
    await log_message(session_id, "user", text, None)

    full_text = ""
    async for _sentence, accumulated in stream_reply_chunks(text, history, user_id):
        full_text = accumulated

    pcm = b"".join([chunk async for chunk in stream_speech_chunks(full_text)])
    audio_path = settings.assets_dir / f"bot_{uuid.uuid4().hex[:8]}.wav"
    await asyncio.to_thread(audio_path.write_bytes, pcm_to_wav(pcm, settings.tts_sample_rate))

    relative_path = asset_ref(audio_path)
    await log_message(session_id, "assistant", full_text, relative_path)

    return JSONResponse({
        "bot_text": full_text,
        "bot_audio_path": f"/{relative_path}",
        "user_id": user_id,
        "session_id": session_id,
    })
