"""
RAG document ingestion.

Supported formats (see `settings.allowed_doc_extensions`):
  - PDF -> text via PyPDFLoader
  - TXT -> read directly (UTF-8, falling back to latin-1)

Every chunk is tagged with its owner (the uploading user's id), and every
operation here is filtered by it, so users only ever see their own documents.

Text is split into overlapping chunks and stored in a local ChromaDB vector
store using sentence-transformers embeddings (no external API key needed). The
store persists to `settings.chroma_path`, so ingested documents survive
restarts. Paths, model name, chunk sizes and batch size all come from
`config/settings.py`.

Everything here is synchronous (embedding + disk); async callers should run it
with `asyncio.to_thread`.
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_collection = None
_collection_lock = threading.Lock()


def get_collection():
    """Return the shared ChromaDB collection, creating it on first use.

    Loading the embedding model takes ~25s, so it must happen once per process,
    not once per chat message. The lock stops the startup warm-up and an early
    request from building two Chroma clients on the same path at once.
    """
    global _collection
    with _collection_lock:
        if _collection is None:
            _collection = _build_collection()
        return _collection


def is_ready() -> bool:
    """True once the vector store and embedding model have been loaded."""
    return _collection is not None


def _build_collection():
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    embedder = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    return client.get_or_create_collection(settings.chroma_collection, embedding_function=embedder)


# ── Loading and chunking ──────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap
    )
    return splitter.split_text(text)


def _load_pdf(path: Path) -> str:
    from langchain_community.document_loaders import PyPDFLoader

    pages = PyPDFLoader(str(path)).load()
    return "\n\n".join(page.page_content for page in pages)


def _load_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


_LOADERS = {".pdf": _load_pdf, ".txt": _load_txt}


# ── Public API ────────────────────────────────────────────────────────────────

def _owned(filename: str, owner: str) -> dict:
    return {"$and": [{"source": filename}, {"owner": owner}]}


def ingest_file(file_path: str, owner: str) -> dict:
    """Ingest a PDF or TXT file into ChromaDB on behalf of `owner`.

    Returns a dict with: file, chunks_added, total_chunks (the owner's chunk count).
    Raises FileNotFoundError / ValueError for a missing or unsupported file.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    if suffix not in settings.allowed_doc_extensions or suffix not in _LOADERS:
        allowed = ", ".join(settings.allowed_doc_extensions)
        raise ValueError(f"Unsupported file type '{suffix}' (allowed: {allowed})")

    collection = get_collection()
    chunks = _chunk_text(_LOADERS[suffix](path))
    if not chunks:
        return {"file": path.name, "chunks_added": 0, "total_chunks": _owner_chunk_count(owner)}

    # Re-uploading a file replaces the owner's old chunks instead of duplicating them.
    delete_doc(path.name, owner)

    metadata = {"source": path.name, "owner": owner, "type": suffix.lstrip(".")}
    step = settings.rag_batch_size
    for start in range(0, len(chunks), step):
        batch = chunks[start:start + step]
        collection.add(
            documents=batch,
            ids=[str(uuid.uuid4()) for _ in batch],
            metadatas=[metadata] * len(batch),
        )

    logger.info("Ingested %d chunks from '%s'", len(chunks), path.name)
    return {
        "file": path.name,
        "chunks_added": len(chunks),
        "total_chunks": _owner_chunk_count(owner),
    }


def _owner_chunk_count(owner: str) -> int:
    return len(get_collection().get(where={"owner": owner}, include=[]).get("ids", []))


def has_docs(owner: str) -> bool:
    """Cheap check (no embedding) for whether `owner` has uploaded anything."""
    return bool(get_collection().get(where={"owner": owner}, limit=1, include=[]).get("ids"))


def list_docs(owner: str) -> list[str]:
    """Sorted filenames of the documents `owner` has ingested."""
    metadatas = get_collection().get(where={"owner": owner}, include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metadatas})


def delete_doc(filename: str, owner: str) -> int:
    """Remove all of `owner`'s chunks of one document. Returns the number deleted."""
    collection = get_collection()
    ids = collection.get(where=_owned(filename, owner), include=[]).get("ids", [])
    if ids:
        collection.delete(ids=ids)
        logger.info("Deleted %d chunks for '%s'", len(ids), filename)
    return len(ids)
