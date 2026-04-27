"""
app/rag/bm25_index.py — BM25 lexical search over the ChromaDB corpus.

Why BM25 alongside dense embeddings?
  Dense embeddings capture semantic meaning but fail on exact term matching.
  Example: user asks about "SAML ACS URL" — the embedding might not match the
  KB chunk even though it contains that exact string. BM25 finds it because it's
  a TF-IDF-style lexical scorer that rewards exact term overlap.

BM25 (Best Matching 25) scoring:
  score(D, Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))
  where:
    f(qi, D)  = term frequency of query term qi in document D
    |D|       = document length (tokens)
    avgdl     = average document length in corpus
    k1, b     = tuning constants (BM25Okapi defaults: k1=1.5, b=0.75)
    IDF       = inverse document frequency (penalises common words)

Trade-off vs dense:
  + Exact match (acronyms, product names, version numbers)
  + No embedding call needed at query time (fast)
  - Lexical mismatch ("price" ≠ "fee")
  - No semantic understanding

Module 3 fuses both with Reciprocal Rank Fusion (RRF) in hybrid_retrieve.py.
"""

from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

from app.rag.store import RetrievedChunk, VectorStore, vector_store as _default_store

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer — keep alphanumeric tokens only."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    In-memory BM25 index built from all chunks in the VectorStore.

    The index is rebuilt on construction. For our KB (< 1k chunks) this is
    instantaneous. In production you'd maintain the BM25 index incrementally
    or rebuild asynchronously after each ingest.
    """

    def __init__(self, store: VectorStore | None = None) -> None:
        store = store or _default_store
        self._chunks: list[RetrievedChunk] = store.get_all()

        if not self._chunks:
            logger.warning("BM25Index: store is empty — no index built.")
            self._bm25: BM25Okapi | None = None
            return

        corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)
        logger.debug("BM25Index built over %d chunks.", len(self._chunks))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Return (chunk_id, bm25_score) pairs ranked best-first.

        Returns up to *top_k* results. Score is not normalised — only useful
        for ranking, not as an absolute relevance measure.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            zip([c.chunk_id for c in self._chunks], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(cid, float(score)) for cid, score in ranked[:top_k]]
