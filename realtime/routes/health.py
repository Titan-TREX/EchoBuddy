"""
`GET /health`: liveness/readiness for Docker, Railway, Render and uptime monitors.

Open (no token, not rate-limited) so a platform can poll it, and it reveals nothing
sensitive: only whether the database answers and whether the document-search model
has finished loading. Returns 503 if the database check fails.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config.settings import settings
from database.connection import get_connection
from rag import ingestor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# The database check opens a connection, which is slow under load; a flood of probes
# (or a busy server) must not make /health itself time out and get the container restarted.
_cached: tuple[float, str] | None = None
_lock = asyncio.Lock()


def reset_cache() -> None:
    global _cached
    _cached = None


async def _database_status() -> str:
    global _cached
    async with _lock:  # concurrent probes share one check instead of each running their own
        now = time.monotonic()
        if _cached and now - _cached[0] < settings.health_db_cache_seconds:
            return _cached[1]
        try:
            async with get_connection() as db:
                await db.execute("SELECT 1;")
            status = "ok"
        except Exception:
            logger.exception("Health check: database unavailable")
            status = "error"
        _cached = (time.monotonic(), status)
        return status


@router.get("/health")
async def health() -> JSONResponse:
    database = await _database_status()

    body = {
        "status": "ok" if database == "ok" else "degraded",
        "database": database,
        "document_search_ready": ingestor.is_ready(),  # False for ~20s after boot while the model loads
    }
    return JSONResponse(body, status_code=200 if database == "ok" else 503)
