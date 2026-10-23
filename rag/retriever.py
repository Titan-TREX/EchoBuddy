"""
RAG context retrieval for EchoBuddy.

Given a natural-language query, this module performs a semantic similarity
search over the ChromaDB vector store and returns the top-N most relevant
document chunks as a formatted string.

Used exclusively by realtime/tools.py as the backend for the
`search_documents` tool. The LLM (Groq) decides when to call this tool —
the retriever itself is stateless and has no knowledge of the conversation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def retrieve_context(query: str, n_results: int = 4) -> str:
    """Semantic search over ingested documents.

    Args:
        query:     The user's question or search query.
        n_results: Number of top chunks to retrieve (default 4).

    Returns:
        A formatted string of the most relevant document excerpts,
        or an empty string if no documents have been ingested yet.
        An empty string tells the LLM to answer from general knowledge.
    """
    try:
        from rag.ingestor import _get_collection  # lazy import to keep startup fast

        col = _get_collection()
        if col.count() == 0:
            return ""

        results = col.query(
            query_texts=[query],
            n_results=min(n_results, col.count()),
        )

        chunks   = results["documents"][0]
        metas    = results["metadatas"][0]

        if not chunks:
            return ""

        lines = []
        for i, (chunk, meta) in enumerate(zip(chunks, metas), start=1):
            source = meta.get("source", "unknown")
            lines.append(f"[Document: {source} — excerpt {i}]\n{chunk}")

        return "\n\n".join(lines)

    except Exception:
        logger.exception("RAG retrieval failed — returning empty context")
        return ""