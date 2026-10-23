"""
RAG isolation tests against a REAL (in-memory) ChromaDB, so the owner filters are
exercised for real. Only the embedding model is swapped for a tiny deterministic
one, so nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import uuid

import chromadb
import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from config.settings import settings
from rag import ingestor, retriever

ALICE = "alice-0000001"
BOB = "bob-000000002"


class HashEmbedding(EmbeddingFunction):
    """Bag-of-words hashing: texts sharing words land close together."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "test-hash-embedding"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(_config: dict) -> "HashEmbedding":
        return HashEmbedding()

    def __call__(self, input: Documents) -> Embeddings:
        dims = 64
        out = []
        for text in input:
            vec = [0.0] * dims
            for word in text.lower().split():
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture(autouse=True)
def store(monkeypatch):
    """Point the ingestor at a fresh in-memory collection for every test."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        f"test_{uuid.uuid4().hex}", embedding_function=HashEmbedding()
    )
    monkeypatch.setattr(ingestor, "_collection", collection)
    monkeypatch.setattr(settings, "rag_max_distance", 10.0)  # relevance cutoff off: isolate ownership
    return collection


def _doc(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestOwnerIsolation:
    def test_retrieval_only_returns_the_askers_documents(self, tmp_path):
        ingestor.ingest_file(_doc(tmp_path, "a.txt", "alice secret launch code is swordfish"), ALICE)
        ingestor.ingest_file(_doc(tmp_path, "b.txt", "bob favourite fruit is banana"), BOB)

        alice = retriever.retrieve_context("what is the launch code", ALICE)
        bob = retriever.retrieve_context("what is the launch code", BOB)

        assert "swordfish" in alice and "banana" not in alice
        assert "banana" in bob and "swordfish" not in bob  # bob never sees alice's text

    def test_user_with_no_documents_gets_no_context(self, tmp_path):
        ingestor.ingest_file(_doc(tmp_path, "a.txt", "alice secret launch code is swordfish"), ALICE)
        assert retriever.retrieve_context("launch code swordfish", "carol-000003") == ""

    def test_empty_owner_gets_nothing(self, tmp_path):
        ingestor.ingest_file(_doc(tmp_path, "a.txt", "alice secret launch code is swordfish"), ALICE)
        assert retriever.retrieve_context("launch code", "") == ""

    def test_asking_for_more_chunks_than_the_owner_has_is_fine(self, tmp_path):
        ingestor.ingest_file(_doc(tmp_path, "a.txt", "one small note about swordfish"), ALICE)
        ingestor.ingest_file(_doc(tmp_path, "b.txt", "bob has lots " * 200), BOB)  # many chunks, not alice's
        assert "swordfish" in retriever.retrieve_context("swordfish", ALICE, n_results=4)

    def test_list_is_per_owner(self, tmp_path):
        ingestor.ingest_file(_doc(tmp_path, "a.txt", "alpha"), ALICE)
        ingestor.ingest_file(_doc(tmp_path, "b.txt", "beta"), BOB)
        assert ingestor.list_docs(ALICE) == ["a.txt"]
        assert ingestor.list_docs(BOB) == ["b.txt"]
        assert ingestor.list_docs("nobody-000004") == []

    def test_same_filename_for_two_owners_stays_separate(self, tmp_path):
        (tmp_path / "x").mkdir()
        (tmp_path / "y").mkdir()
        ingestor.ingest_file(_doc(tmp_path / "x", "notes.txt", "alice notes about swordfish"), ALICE)
        ingestor.ingest_file(_doc(tmp_path / "y", "notes.txt", "bob notes about banana"), BOB)

        assert ingestor.delete_doc("notes.txt", ALICE) > 0

        assert ingestor.list_docs(ALICE) == []
        assert ingestor.list_docs(BOB) == ["notes.txt"]  # bob's same-named file survives
        assert "banana" in retriever.retrieve_context("banana", BOB)

    def test_delete_cannot_touch_another_owners_document(self, tmp_path):
        ingestor.ingest_file(_doc(tmp_path, "a.txt", "alice private text"), ALICE)
        assert ingestor.delete_doc("a.txt", BOB) == 0
        assert ingestor.list_docs(ALICE) == ["a.txt"]

    def test_reupload_replaces_instead_of_duplicating(self, tmp_path):
        path = _doc(tmp_path, "a.txt", "first version about swordfish")
        first = ingestor.ingest_file(path, ALICE)
        second = ingestor.ingest_file(path, ALICE)
        assert second["total_chunks"] == first["total_chunks"]

    def test_chunk_counts_are_per_owner(self, tmp_path):
        ingestor.ingest_file(_doc(tmp_path, "b.txt", "bob words " * 400), BOB)
        result = ingestor.ingest_file(_doc(tmp_path, "a.txt", "alice tiny"), ALICE)
        assert result["total_chunks"] == result["chunks_added"] == 1

    def test_ingest_rejects_unsupported_types(self, tmp_path):
        bad = tmp_path / "x.exe"
        bad.write_text("nope")
        with pytest.raises(ValueError):
            ingestor.ingest_file(str(bad), ALICE)


class TestRelevanceCutoff:
    def test_far_chunks_are_dropped(self, tmp_path, monkeypatch):
        ingestor.ingest_file(_doc(tmp_path, "a.txt", "completely unrelated gardening tips"), ALICE)
        monkeypatch.setattr(settings, "rag_max_distance", 0.0)
        assert retriever.retrieve_context("quantum chromodynamics", ALICE) == ""
