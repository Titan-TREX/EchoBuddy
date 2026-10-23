"""RAG document management. Every operation is scoped to the caller's `X-User-Id`."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse

from config.settings import settings
from rag.ingestor import delete_doc, ingest_file, list_docs
from realtime.security import access_guard, owner_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])

_UNSAFE_CHARS = re.compile(r"[^\w.\- ]")
_CHUNK_BYTES = 1024 * 1024


def _safe_filename(raw_name: str | None, suffix: str) -> str:
    """Strip any directory part and unsafe characters from a client-supplied name."""
    stem = _UNSAFE_CHARS.sub("_", Path(Path(raw_name or "").name).stem).strip(" .")
    return f"{stem or 'doc_' + uuid.uuid4().hex[:8]}{suffix}"


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


@router.post("/upload-doc", dependencies=[Depends(access_guard(settings.rate_limit_upload_cost))])
async def upload_document(
    file: UploadFile = File(...), owner: str = Depends(owner_id)
) -> JSONResponse:
    """Accept an allowed document type, save it, and ingest it for the caller only."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_doc_extensions:
        allowed = ", ".join(settings.allowed_doc_extensions)
        return _error(f"Unsupported file type '{suffix}'. Allowed: {allowed}.", 400)

    # One folder per owner (the id is validated, so it is a safe directory name),
    # so two users uploading "notes.txt" never overwrite each other.
    owner_dir = settings.upload_dir / owner
    owner_dir.mkdir(parents=True, exist_ok=True)
    save_path = owner_dir / _safe_filename(file.filename, suffix)

    # Stream to disk with a hard size cap instead of trusting Content-Length.
    written = 0
    try:
        with open(save_path, "wb") as out:
            while chunk := await file.read(_CHUNK_BYTES):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise ValueError("too large")
                out.write(chunk)
    except ValueError:
        save_path.unlink(missing_ok=True)
        return _error(f"File too large (limit {settings.max_upload_mb} MB).", 413)

    try:
        # Embedding + Chroma writes are CPU/disk bound: keep them off the event loop.
        result = await asyncio.to_thread(ingest_file, str(save_path), owner)
    except Exception:
        logger.exception("Ingestion failed for %s", save_path.name)
        save_path.unlink(missing_ok=True)
        return _error("Could not read that document.", 500)
    return JSONResponse({"success": True, **result})


@router.get("/docs", dependencies=[Depends(access_guard())])
async def get_docs(owner: str = Depends(owner_id)) -> JSONResponse:
    """Filenames of the caller's ingested documents."""
    return JSONResponse({"documents": await asyncio.to_thread(list_docs, owner)})


@router.delete("/docs/{filename}", dependencies=[Depends(access_guard())])
async def remove_doc(filename: str, owner: str = Depends(owner_id)) -> JSONResponse:
    """Delete one of the caller's documents (its chunks and its stored file)."""
    name = Path(filename).name  # never let the URL point outside the owner's folder
    deleted = await asyncio.to_thread(delete_doc, name, owner)
    (settings.upload_dir / owner / name).unlink(missing_ok=True)
    return JSONResponse({"deleted_chunks": deleted, "file": name})
