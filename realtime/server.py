"""
FastAPI app serving BOTH voice pipelines plus text chat and document upload.

Routes live in `realtime/routes/`:
  * chat.py       POST /api/voice-chat (Temporal batch), POST /api/text-chat
  * ws.py         WS   /ws             (realtime voice)
  * documents.py  /api/upload-doc, /api/docs

Both pipelines share `config/settings.py`, the same SQLite database and the
same `assets/` directory, so they can run side by side. The Temporal pipeline
additionally needs a Temporal server and a running `temporal/worker.py`.

Run:
    uv run uvicorn realtime.server:app --reload --port 8000
then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import configure_logging, settings
from config.tracing import flush
from rag.ingestor import get_collection
from realtime.cleanup import cleanup_loop
from realtime.routes import auth, chat, documents, health, ws

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for name in settings.missing_credentials():
        logger.warning("%s is not set - features that need it will fail.", name)

    if not settings.api_token:
        logger.warning("API_TOKEN is not set - the API is open to anyone (rate limiting still applies).")

    # Load the embedding model in the background so the first chat isn't slow.
    async def warm_rag() -> None:
        try:
            await asyncio.to_thread(get_collection)
        except Exception:
            logger.exception("RAG warm-up failed")

    background_tasks = [asyncio.create_task(warm_rag()), asyncio.create_task(cleanup_loop())]
    yield
    for task in background_tasks:
        task.cancel()
    flush()  # send buffered Langfuse traces


app = FastAPI(lifespan=lifespan)

# Lets the frontend call this API from another origin (e.g. static/index.html
# on Vercel talking to this server on Railway/Render). Set via CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(ws.router)

app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
# Recorded user clips and synthesized replies; the browser plays them straight
# from the `bot_audio_path` returned by the chat endpoints.
app.mount("/assets", StaticFiles(directory=str(settings.assets_dir)), name="assets")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(settings.static_dir / "index.html"))
