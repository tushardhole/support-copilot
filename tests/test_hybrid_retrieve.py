"""
tests/test_hybrid_retrieve.py — Unit tests for the hybrid retrieval pipeline.

All external calls (BM25, dense store, cross-encoder, LLM) are mocked.
No API key, no model download, no disk I/O required.

Run: uv run pytest tests/test_hybrid_retrieve.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.rag.store import RetrievedChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(chunk_id: str, text: str = "sample text", score: float = 0.8,
           embedding: list[float] | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="test.md",
        chunk_index=0,
        score=score,
        embedding=embedding or [0.1, 0.2, 0.3],
    )


# ── BM25Index ─────────────────────────────────────────────────────────────────

def test_bm25_search_returns_ranked_results():
    from app.rag.bm25_index import BM25Index

    mock_store = MagicMock()
    mock_store.get_all.return_value = [
        _chunk("a", text="billing invoice payment refund"),
        _chunk("b", text="login password forgot account"),
        _chunk("c", text="team member invite roles permissions"),
    ]

    index = BM25Index(store=mock_store)
    results = index.search("billing invoice", top_k=3)

    assert len(results) >= 1
    ids = [r[0] for r in results]
    # "billing invoice" should rank chunk "a" first
    assert ids[0] == "a"


def test_bm25_empty_store():
    from app.rag.bm25_index import BM25Index

    mock_store = MagicMock()
    mock_store.get_all.return_value = []

    index = BM25Index(store=mock_store)
    results = index.search("anything", top_k=5)
    assert results == []


def test_bm25_scores_are_floats():
    from app.rag.bm25_index import BM25Index

    mock_store = MagicMock()
    mock_store.get_all.return_value = [_chunk("a", text="hello world")]

    index = BM25Index(store=mock_store)
    results = index.search("hello", top_k=1)
    assert all(isinstance(score, float) for _, score in results)


# ── RRF fusion ────────────────────────────────────────────────────────────────

def test_rrf_fuse_merges_rankings():
    from app.rag.hybrid_retrieve import rrf_fuse

    chunks = {
        "a": _chunk("a"),
        "b": _chunk("b"),
        "c": _chunk("c"),
    }
    # BM25 ranks: a > b > c
    # Dense ranks: c > a > b
    # Expected: "a" appears in both at rank 1 and 2 → should beat "c" (rank 1 only)
    result = rrf_fuse([["a", "b", "c"], ["c", "a", "b"]], chunks)

    assert len(result) == 3
    ids = [c.chunk_id for c in result]
    assert "a" in ids
    assert ids.index("a") < ids.index("b")  # a should beat b


def test_rrf_fuse_score_decreases_with_rank():
    from app.rag.hybrid_retrieve import rrf_fuse

    chunks = {"a": _chunk("a"), "b": _chunk("b")}
    result = rrf_fuse([["a", "b"], ["a", "b"]], chunks)
    assert result[0].chunk_id == "a"
    assert result[0].score > result[1].score


def test_rrf_fuse_handles_missing_chunks():
    from app.rag.hybrid_retrieve import rrf_fuse

    chunks = {"a": _chunk("a")}  # "b" not in all_chunks
    result = rrf_fuse([["a", "b"]], chunks)
    assert len(result) == 1
    assert result[0].chunk_id == "a"


# ── MMR diversity ─────────────────────────────────────────────────────────────

def test_mmr_select_returns_top_k():
    import numpy as np
    from app.rag.hybrid_retrieve import mmr_select

    # Create 5 candidates with distinct embeddings
    candidates = [
        _chunk(f"c{i}", embedding=np.random.rand(8).tolist(), score=1.0 - i * 0.1)
        for i in range(5)
    ]
    query_emb = [0.5] * 8

    selected = mmr_select(candidates, query_emb, top_k=3)
    assert len(selected) == 3


def test_mmr_select_fallback_without_embeddings():
    from app.rag.hybrid_retrieve import mmr_select

    # Chunks with no embeddings → should fall back to simple top-k
    candidates = [_chunk(f"c{i}", embedding=None, score=1.0 - i * 0.1) for i in range(5)]
    selected = mmr_select(candidates, [0.1, 0.2, 0.3], top_k=3)
    assert len(selected) == 3


# ── CrossEncoderReranker ──────────────────────────────────────────────────────

def test_reranker_sorts_by_score():
    from app.rag.rerank import CrossEncoderReranker

    chunks = [_chunk("low", score=0.9), _chunk("high", score=0.5)]

    with patch("app.rag.rerank.CrossEncoder") as MockCE:
        instance = MockCE.return_value
        # cross-encoder gives "high" a higher raw score
        instance.predict.return_value = [0.2, 3.5]  # low=0.2, high=3.5

        r = CrossEncoderReranker()
        r._model = instance
        result = r.rerank("test query", chunks)

    assert result[0].chunk_id == "high"
    assert result[1].chunk_id == "low"


def test_reranker_empty_input():
    from app.rag.rerank import CrossEncoderReranker

    r = CrossEncoderReranker()
    result = r.rerank("query", [])
    assert result == []


def test_reranker_scores_are_sigmoid():
    from app.rag.rerank import CrossEncoderReranker
    import math

    chunks = [_chunk("a")]
    with patch("app.rag.rerank.CrossEncoder") as MockCE:
        instance = MockCE.return_value
        instance.predict.return_value = [0.0]  # logit 0 → sigmoid 0.5

        r = CrossEncoderReranker()
        r._model = instance
        result = r.rerank("q", chunks)

    assert abs(result[0].score - 0.5) < 0.01


# ── hybrid_retrieve (end-to-end mocked) ───────────────────────────────────────

def test_hybrid_retrieve_returns_chunks():
    from app.rag.hybrid_retrieve import hybrid_retrieve

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_store = MagicMock()
    mock_store.count.return_value = 3
    mock_store.query.return_value = [
        _chunk("a", text="billing info", score=0.9, embedding=[0.9, 0.1, 0.0]),
        _chunk("b", text="login help", score=0.7, embedding=[0.1, 0.9, 0.0]),
    ]
    mock_store.get_all.return_value = [
        _chunk("a", text="billing info"),
        _chunk("b", text="login help"),
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda q, chunks, top_k=None: chunks

    results = hybrid_retrieve(
        "billing question",
        top_k=2,
        use_rerank=False,
        use_hyde=False,
        use_query_rewrite=False,
        use_mmr=False,
        embedder=mock_embedder,
        store=mock_store,
        reranker=mock_reranker,
    )
    assert len(results) >= 1


def test_hybrid_retrieve_empty_store():
    from app.rag.hybrid_retrieve import hybrid_retrieve

    mock_embedder = MagicMock()
    mock_store = MagicMock()
    mock_store.count.return_value = 0

    results = hybrid_retrieve("anything", embedder=mock_embedder, store=mock_store)
    assert results == []


# ── VectorStore.get_all ───────────────────────────────────────────────────────

def test_store_get_all_returns_chunks():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.get.return_value = {
        "documents": ["doc one", "doc two"],
        "metadatas": [
            {"source": "a.md", "chunk_index": 0, "token_count": 10},
            {"source": "b.md", "chunk_index": 1, "token_count": 20},
        ],
    }

    with patch("app.rag.store.chromadb") as mock_chroma:
        mock_client = MagicMock()
        mock_chroma.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        from app.rag.store import VectorStore
        from app.config import settings
        store = VectorStore(settings)
        store._collection = mock_collection

        results = store.get_all()

    assert len(results) == 2
    assert results[0].text == "doc one"
    assert results[1].source == "b.md"
