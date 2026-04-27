"""
tests/test_rag.py — Unit tests for the RAG pipeline (chunker, embedder, store, retrieve).

All external calls (ChromaDB, OpenAI embeddings, sentence-transformers) are mocked.
No API key, no GPU, no disk writes required.

Run: uv run pytest tests/test_rag.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.rag.chunker import Chunk, chunk_text
from app.rag.retrieve import format_context, retrieve
from app.rag.store import RetrievedChunk


# ── Chunker ───────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """\
## Account Setup

Visit app.acme.io and click Sign Up.
Enter your work email and create a password.
Confirm your email via the link we send you.

## Billing

We offer three plans: Starter, Pro, and Business.
Billing is monthly by default; annual plans get 20% off.
Invoices are emailed on the 1st of each month.

## Troubleshooting

If you cannot log in, click Forgot Password on the login page.
After 5 failed attempts your account is locked for 15 minutes.
"""


def test_chunk_text_returns_chunks():
    chunks = chunk_text(SAMPLE_TEXT, source="test.md", chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_text_token_budget():
    chunks = chunk_text(SAMPLE_TEXT, source="test.md", chunk_size=80, chunk_overlap=15)
    for c in chunks:
        # Allow minor overshoot from overlap carry-over but stay reasonable
        assert c.token_count <= 120, f"Chunk too large: {c.token_count} tokens"


def test_chunk_ids_are_deterministic():
    chunks_a = chunk_text(SAMPLE_TEXT, source="test.md")
    chunks_b = chunk_text(SAMPLE_TEXT, source="test.md")
    ids_a = [c.chunk_id for c in chunks_a]
    ids_b = [c.chunk_id for c in chunks_b]
    assert ids_a == ids_b


def test_chunk_ids_differ_by_source():
    chunks_a = chunk_text(SAMPLE_TEXT, source="doc_a.md")
    chunks_b = chunk_text(SAMPLE_TEXT, source="doc_b.md")
    assert chunks_a[0].chunk_id != chunks_b[0].chunk_id


def test_chunk_metadata_fields():
    chunks = chunk_text(SAMPLE_TEXT, source="test.md")
    for c in chunks:
        meta = c.metadata()
        assert "source" in meta
        assert "chunk_index" in meta
        assert "token_count" in meta
        assert meta["source"] == "test.md"


def test_chunk_empty_text():
    chunks = chunk_text("   \n\n  ", source="empty.md")
    assert chunks == []


def test_chunk_short_text_single_chunk():
    short = "Hello world."
    chunks = chunk_text(short, source="short.md", chunk_size=400)
    assert len(chunks) == 1
    assert chunks[0].text == short


# ── Embedder (mocked) ─────────────────────────────────────────────────────────

def test_openai_embedder_returns_vectors():
    from app.rag.embedder import OpenAIEmbedder
    from app.config import settings

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

    with patch("app.rag.embedder.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.embeddings.create.return_value = mock_response

        emb = OpenAIEmbedder(settings)
        vecs = emb.embed(["hello world"])

    assert len(vecs) == 1
    assert vecs[0] == [0.1, 0.2, 0.3]


def test_openai_embedder_batches():
    from app.rag.embedder import OpenAIEmbedder
    from app.config import settings

    def make_response(n):
        r = MagicMock()
        r.data = [MagicMock(embedding=[float(i)] * 3) for i in range(n)]
        return r

    with patch("app.rag.embedder.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.embeddings.create.side_effect = [make_response(2)]

        emb = OpenAIEmbedder(settings)
        emb._BATCH_SIZE = 512
        vecs = emb.embed(["a", "b"])

    assert len(vecs) == 2


def test_embedder_empty_input():
    from app.rag.embedder import OpenAIEmbedder
    from app.config import settings

    with patch("app.rag.embedder.OpenAI"):
        emb = OpenAIEmbedder(settings)
        assert emb.embed([]) == []


# ── VectorStore (mocked ChromaDB) ─────────────────────────────────────────────

def _make_store():
    """Return a VectorStore with a fully mocked ChromaDB client."""
    with patch("app.rag.store.chromadb") as mock_chroma:
        mock_client = MagicMock()
        mock_chroma.PersistentClient.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        from app.rag.store import VectorStore
        from app.config import settings
        store = VectorStore(settings)
        store._collection = mock_collection
        return store, mock_collection


def test_store_upsert_calls_collection():
    store, coll = _make_store()
    chunks = chunk_text("some text for testing", source="s.md")
    embeddings = [[0.1, 0.2, 0.3]] * len(chunks)
    store.upsert(chunks, embeddings)
    coll.upsert.assert_called_once()


def test_store_upsert_empty_is_noop():
    store, coll = _make_store()
    store.upsert([], [])
    coll.upsert.assert_not_called()


def test_store_query_returns_retrieved_chunks():
    store, coll = _make_store()
    coll.count.return_value = 3
    coll.query.return_value = {
        "documents": [["chunk text here"]],
        "metadatas": [
            [{"source": "test.md", "chunk_index": 0, "token_count": 10}]
        ],
        "distances": [[0.2]],  # cosine distance 0.2 → similarity 0.9
    }

    results = store.query([0.1, 0.2, 0.3], top_k=1)
    assert len(results) == 1
    assert results[0].text == "chunk text here"
    assert results[0].source == "test.md"
    assert abs(results[0].score - 0.9) < 0.01


def test_store_query_empty_store_returns_nothing():
    store, coll = _make_store()
    coll.count.return_value = 0
    results = store.query([0.1, 0.2], top_k=5)
    assert results == []


# ── format_context ────────────────────────────────────────────────────────────

def test_format_context_includes_source_labels():
    chunks = [
        RetrievedChunk(chunk_id="a", text="The refund policy is 30 days.",
                       source="policy.md", chunk_index=0, score=0.95),
        RetrievedChunk(chunk_id="b", text="Contact support at help@acme.io.",
                       source="contact.md", chunk_index=1, score=0.80),
    ]
    context = format_context(chunks)
    assert "[1]" in context
    assert "[2]" in context
    assert "policy.md" in context
    assert "contact.md" in context


def test_format_context_respects_token_budget():
    long_text = "word " * 500  # ~500 tokens
    chunks = [
        RetrievedChunk(chunk_id=str(i), text=long_text, source="big.md",
                       chunk_index=i, score=1.0 - i * 0.1)
        for i in range(5)
    ]
    context = format_context(chunks, max_tokens=600)
    # Should not include all 5 chunks (that would be ~2500 tokens)
    assert context.count("[") < 5


# ── retrieve (end-to-end mocked) ──────────────────────────────────────────────

def test_retrieve_returns_chunks_when_store_has_data():
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_store = MagicMock()
    mock_store.count.return_value = 5
    mock_store.query.return_value = [
        RetrievedChunk(chunk_id="x", text="Answer here.", source="kb.md",
                       chunk_index=0, score=0.92)
    ]

    results = retrieve("What is the refund policy?", embedder=mock_embedder, store=mock_store)
    assert len(results) == 1
    assert results[0].score == 0.92


def test_retrieve_empty_store_returns_empty():
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2]]
    mock_store = MagicMock()
    mock_store.count.return_value = 0

    results = retrieve("anything", embedder=mock_embedder, store=mock_store)
    assert results == []


def test_retrieve_score_threshold_filters():
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2]]
    mock_store = MagicMock()
    mock_store.count.return_value = 3
    mock_store.query.return_value = [
        RetrievedChunk(chunk_id="a", text="high", source="a.md", chunk_index=0, score=0.90),
        RetrievedChunk(chunk_id="b", text="low", source="b.md", chunk_index=0, score=0.40),
    ]

    results = retrieve("q", score_threshold=0.75, embedder=mock_embedder, store=mock_store)
    assert len(results) == 1
    assert results[0].text == "high"
