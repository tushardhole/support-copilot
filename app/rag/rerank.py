"""
app/rag/rerank.py — Cross-encoder reranking.

Why rerank?
  The bi-encoder used for initial retrieval (embedder.py) encodes query and
  passage *independently*, then compares their vectors with cosine similarity.
  This is fast (O(1) lookup after indexing) but imprecise — the model has no
  joint attention over (query, passage) pairs.

  A cross-encoder attends to both query and passage simultaneously, producing a
  much more accurate relevance score. The trade-off: it's O(k) forward passes
  at query time (k = number of candidates), not O(1).

  Typical pipeline:
    1. Bi-encoder retrieve top-50 candidates cheaply.
    2. Cross-encoder rerank top-50 to get true top-5.
    3. Serve top-5 to the LLM.

Model chosen: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~22M parameters, fast on CPU (~50ms for 10 candidates)
  - Trained on MS MARCO passage ranking (QA domain) — good default
  - Score range: unbounded logits; higher = more relevant

Interview note:
  "Bi-encoder vs cross-encoder?"
  Bi-encoder: fast retrieval, approximate relevance (good for top-k from millions)
  Cross-encoder: slow, precise (good for reranking top-k from hundreds)
  They are complementary — use both in a two-stage pipeline.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from app.rag.store import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Reranks a list of RetrievedChunk objects using a cross-encoder model.

    The model is lazily loaded on the first call so import time stays fast.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is None:
            logger.info("Loading cross-encoder '%s'…", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder ready.")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Score each (query, chunk.text) pair and return chunks sorted by cross-encoder score.

        Parameters
        ----------
        query:  The user query.
        chunks: Candidate chunks from bi-encoder retrieval (any order).
        top_k:  Return only the top-k reranked chunks. None = return all.

        Returns
        -------
        Chunks sorted by cross-encoder score descending, with `score` updated
        to reflect the cross-encoder's judgment (normalised to [0, 1] via sigmoid).
        """
        if not chunks:
            return []

        self._load()

        pairs = [(query, c.text) for c in chunks]
        raw_scores = self._model.predict(pairs)

        # Apply sigmoid to convert raw logits → [0, 1] for interpretability.
        import math
        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        reranked = sorted(
            zip(chunks, raw_scores),
            key=lambda t: t[1],
            reverse=True,
        )

        result: list[RetrievedChunk] = []
        for chunk, raw in reranked:
            import dataclasses
            updated = dataclasses.replace(chunk, score=round(_sigmoid(float(raw)), 4))
            result.append(updated)

        return result[:top_k] if top_k else result


# Module-level singleton — lazy loads model on first rerank() call.
reranker = CrossEncoderReranker()
