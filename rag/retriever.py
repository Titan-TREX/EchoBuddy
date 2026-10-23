"""
RAG context retrieval.

Given a natural-language query, runs a semantic similarity search over the
requesting user's documents in ChromaDB (never anyone else's) and returns the
most relevant chunks as one formatted string. `llm_stream.py` injects that
string into the system prompt.

Stateless and synchronous (embedding + disk); async callers should use
`asyncio.to_thread`. Never raises: any failure is logged and treated as "no
context", so a broken vector store cannot take chat down.
"""

from __future__ import annotations

import logging

from config.settings import settings
from config.tracing import observe

logger = logging.getLogger(__name__)


@observe(name="rag-retrieval", as_type="retriever")
def retrieve_context(query: str, owner: str, n_results: int | None = None) -> str:
    """Semantic search over ingested documents.

    Args:
        query:     The user's question.
        owner:     Id of the user whose documents may be searched.
        n_results: Max chunks to return (default `settings.rag_top_k`).

    Returns:
        Formatted excerpts of chunks closer than `settings.rag_max_distance`,
        or "" if this user has uploaded nothing or nothing is relevant (which
        tells the LLM to answer from general knowledge).
    """
    try:
        from rag.ingestor import get_collection, has_docs  # lazy: keeps app startup fast

        if not owner or not has_docs(owner):
            return ""  # nothing to search: skip embedding the query entirely

        results = get_collection().query(
            query_texts=[query],
            n_results=n_results or settings.rag_top_k,
            where={"owner": owner},
            include=["documents", "metadatas", "distances"],
        )

        lines = []
        for chunk, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            if distance > settings.rag_max_distance:
                continue
            source = (meta or {}).get("source", "unknown")
            lines.append(f"[Document: {source} — excerpt {len(lines) + 1}]\n{chunk}")

        return "\n\n".join(lines)

    except Exception:
        logger.exception("RAG retrieval failed - returning empty context")
        return ""
