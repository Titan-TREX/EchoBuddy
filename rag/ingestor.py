"""
RAG document ingestion for EchoBuddy.

Supported formats:
  - PDF  → chunked via PyPDFLoader + RecursiveCharacterTextSplitter
  - TXT  → read directly, then chunked

Documents are stored in a local ChromaDB vector store using
sentence-transformers embeddings (no external API key needed).
The store persists to disk at <project_root>/chroma_db so ingested
documents survive server restarts.

Architecture note:
  This module is called by realtime/server.py (upload endpoint) and
  by realtime/tools.py (search_documents tool executor). Neither
  llm_stream.py nor llm_activity.py are touched — RAG is just another
  tool that Groq decides to call when the user's question involves documents.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# ── ChromaDB / embedding setup ────────────────────────────────────────────────

# Resolve project root without importing settings to avoid circular imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH   = str(_PROJECT_ROOT / "chroma_db")
COLLECTION    = "echobuddy_docs"


def _get_collection():
    """Return (or create) the ChromaDB collection with sentence-transformer embeddings."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(COLLECTION, embedding_function=emb_fn)


# ── Chunking helpers ──────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split plain text into overlapping chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


def _load_pdf(path: str) -> str:
    """Extract full text from a PDF using LangChain's PyPDFLoader."""
    from langchain_community.document_loaders import PyPDFLoader

    pages = PyPDFLoader(path).load()
    return "\n\n".join(p.page_content for p in pages)


def _load_txt(path: str) -> str:
    """Read a plain-text file, trying UTF-8 then latin-1 as fallback."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="latin-1")


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_file(file_path: str) -> dict:
    """Ingest a PDF or TXT file into ChromaDB.

    Args:
        file_path: Absolute path to the file on disk.

    Returns:
        dict with keys: file, chunks_added, total_docs
    """
    path   = Path(file_path)
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if suffix == ".pdf":
        raw_text = _load_pdf(file_path)
    elif suffix == ".txt":
        raw_text = _load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix} (only .pdf and .txt allowed)")

    if not raw_text.strip():
        return {"file": path.name, "chunks_added": 0, "total_docs": 0}

    chunks     = _chunk_text(raw_text)
    collection = _get_collection()
    doc_name   = path.name

    collection.add(
        documents=chunks,
        ids=[str(uuid.uuid4()) for _ in chunks],
        metadatas=[{"source": doc_name, "type": suffix.lstrip(".")} for _ in chunks],
    )

    logger.info("Ingested %d chunks from '%s'", len(chunks), doc_name)
    return {
        "file":         doc_name,
        "chunks_added": len(chunks),
        "total_docs":   collection.count(),
    }


def list_docs() -> list[str]:
    """Return sorted list of ingested document filenames."""
    col = _get_collection()
    if col.count() == 0:
        return []
    res = col.get(include=["metadatas"])
    return sorted({m["source"] for m in res["metadatas"]})


def delete_doc(filename: str) -> int:
    """Remove all chunks belonging to a specific document.

    Returns:
        Number of chunks deleted.
    """
    col = _get_collection()
    res = col.get(where={"source": filename}, include=["metadatas"])
    ids = res.get("ids", [])
    if ids:
        col.delete(ids=ids)
        logger.info("Deleted %d chunks for '%s'", len(ids), filename)
    return len(ids)