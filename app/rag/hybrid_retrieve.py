"""
app/rag/hybrid_retrieve.py — Production-shaped RAG retrieval.

Builds on naive retrieve.py (M2) with four improvements:

  1. Hybrid retrieval  — BM25 (lexical) + dense (semantic) fused with RRF
  2. Cross-encoder reranking — precise relevance scoring of top candidates
  3. Query transforms  — rewrite (expand/clarify) or HyDE (embed a hypothesis)
  4. MMR diversity     — penalise near-duplicate chunks in the final set

Full pipeline (all steps optional):

  raw_query
    ↓ [optional] query_rewrite  — LLM rewrites query for better retrieval
    ↓ [optional] HyDE           — LLM writes hypothetical answer; embed that
    ↓ BM25 search               — lexical top-k*3
    ↓ Dense search              — semantic top-k*3
    ↓ RRF fusion                — merge by reciprocal rank
    ↓ [optional] cross-encoder  — rerank fused candidates
    ↓ [optional] MMR            — select diverse top-k
    ↓ format_context            — format for prompt with [1][2]... labels

Interview notes inline throughout.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.config import Settings, settings as _default_settings
from app.rag.bm25_index import BM25Index
from app.rag.embedder import Embedder, make_embedder
from app.rag.rerank import CrossEncoderReranker, reranker as _default_reranker
from app.rag.store import RetrievedChunk, VectorStore, vector_store as _default_store

logger = logging.getLogger(__name__)


# ── RRF fusion ────────────────────────────────────────────────────────────────

def rrf_fuse(
    rankings: list[list[str]],
    all_chunks: dict[str, RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion over multiple ranked lists of chunk_ids.

    RRF formula: score(d) = Σ_i  1 / (k + rank_i(d))
    where rank_i(d) is the 1-indexed position of document d in ranking i.
    k=60 is the standard constant (prevents top-1 from dominating).

    Why RRF?
      - No score normalisation needed (BM25 and cosine scales differ)
      - Robust: a chunk ranked 3rd in both lists beats one ranked 1st in one
      - Simple: no learned weights; no training data needed

    Interview note:
      Other fusion methods: CombSUM (sum normalised scores), CombMNZ (CombSUM
      * number of lists that contain d). RRF outperforms them empirically and
      requires zero tuning.
    """
    rrf_scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            rrf_scores[chunk_id] += 1.0 / (k + rank)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    result: list[RetrievedChunk] = []
    for chunk_id, rrf_score in fused:
        if chunk_id in all_chunks:
            import dataclasses
            result.append(dataclasses.replace(all_chunks[chunk_id], score=round(rrf_score, 6)))

    return result


# ── MMR diversity ─────────────────────────────────────────────────────────────

def mmr_select(
    candidates: list[RetrievedChunk],
    query_embedding: list[float],
    *,
    lambda_: float = 0.7,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Maximal Marginal Relevance — select a diverse, relevant subset.

    MMR formula: argmax_c [ λ·sim(q,c) − (1−λ)·max_{s∈S} sim(c,s) ]
    where S is the set of already-selected chunks.

    λ=1 → pure relevance (same as top-k)
    λ=0 → pure diversity
    λ=0.7 → strong relevance bias with diversity penalty (good default)

    Requires candidates to have embeddings populated (include_embeddings=True).
    Falls back to simple top-k if embeddings are missing.

    Interview note:
      "Lost-in-the-middle" is partly caused by near-duplicate chunks: if your
      top-5 are all from the same paragraph, the model sees the same fact 5
      times and misses other relevant facts. MMR prevents this.
    """
    # Filter candidates that have embeddings
    with_emb = [c for c in candidates if c.embedding is not None]

    if len(with_emb) < 2:
        logger.debug("MMR: not enough embeddings — returning simple top-k.")
        return candidates[:top_k]

    q_vec = np.array(query_embedding)
    emb_matrix = np.array([c.embedding for c in with_emb])

    def _cosine_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between vector a and each row of matrix b."""
        norms = np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-10
        return (b @ a) / norms

    # Relevance scores: sim(query, each candidate)
    relevance = _cosine_row(q_vec, emb_matrix)

    selected_indices: list[int] = []
    remaining = list(range(len(with_emb)))

    for _ in range(min(top_k, len(with_emb))):
        if not selected_indices:
            # First pick: highest relevance
            best = max(remaining, key=lambda i: relevance[i])
        else:
            selected_embs = emb_matrix[selected_indices]

            def mmr_score(i: int) -> float:
                sim_to_selected = float(_cosine_row(emb_matrix[i], selected_embs).max())
                return lambda_ * float(relevance[i]) - (1 - lambda_) * sim_to_selected

            best = max(remaining, key=mmr_score)

        selected_indices.append(best)
        remaining.remove(best)

    return [with_emb[i] for i in selected_indices]


# ── Query transforms ──────────────────────────────────────────────────────────

def query_rewrite(query: str, llm_client=None) -> str:
    """
    Use the LLM to rewrite the query for better retrieval.

    When it helps: short/ambiguous queries ("can't login", "billing issue")
    When it hurts: precise technical queries that don't need expansion
    """
    if llm_client is None:
        from app.llm import llm as _llm
        llm_client = _llm

    from app.prompts.loader import load
    prompt = load("query_rewrite", version=1, question=query)
    rewritten, _ = llm_client.chat([
        {"role": "system", "content": "You are a search query optimiser."},
        {"role": "user", "content": prompt},
    ], temperature=0.0, max_tokens=128)
    rewritten = rewritten.strip().strip('"').strip("'")
    logger.debug("query_rewrite: '%s' → '%s'", query, rewritten)
    return rewritten or query


def hyde_embed(query: str, embedder: Embedder, llm_client=None) -> list[float]:
    """
    HyDE: Hypothetical Document Embeddings.

    Generates a hypothetical answer to the query, then embeds that answer.
    The hypothesis lives in the same vector space as the KB chunks, so it
    retrieves more relevant passages than embedding the question directly.

    When it helps: question phrasing differs greatly from KB phrasing.
    When it hurts: the LLM's hypothesis contains hallucinated facts that pull
                   retrieval away from the correct KB chunk.

    Interview note:
      HyDE trades hallucination risk at retrieval time for better semantic
      alignment. A safer variant: embed both the original query AND the
      hypothesis, fuse results with RRF.
    """
    if llm_client is None:
        from app.llm import llm as _llm
        llm_client = _llm

    from app.prompts.loader import load
    prompt = load("hyde", version=1, question=query)
    hypothesis, _ = llm_client.chat([
        {"role": "user", "content": prompt},
    ], temperature=0.3, max_tokens=200)
    hypothesis = hypothesis.strip()
    logger.debug("HyDE hypothesis: %s", hypothesis[:120])
    return embedder.embed([hypothesis])[0]


# ── Main hybrid retrieval function ────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    *,
    top_k: int = 5,
    candidate_multiplier: int = 4,
    use_rerank: bool = True,
    use_hyde: bool = False,
    use_query_rewrite: bool = False,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
    score_threshold: float = 0.0,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    reranker: CrossEncoderReranker | None = None,
    cfg: Settings | None = None,
    llm_client=None,
) -> list[RetrievedChunk]:
    """
    Full production-shaped retrieval pipeline.

    Parameters
    ----------
    query:                The raw user question.
    top_k:                Final number of chunks to return.
    candidate_multiplier: Retrieve top_k * multiplier from each source before fusion/rerank.
    use_rerank:           Apply cross-encoder reranking after RRF fusion.
    use_hyde:             Embed a hypothetical answer instead of the raw query.
    use_query_rewrite:    Rewrite the query with LLM before retrieval.
    use_mmr:              Apply MMR diversity selection as final step.
    mmr_lambda:           MMR relevance-diversity trade-off (0=diversity, 1=relevance).
    score_threshold:      Drop chunks below this RRF score (0 disables).
    """
    cfg = cfg or _default_settings
    embedder = embedder or make_embedder(cfg)
    store = store or _default_store
    reranker = reranker or _default_reranker

    if store.count() == 0:
        logger.warning("Vector store empty — run `python -m app.rag.ingest` first.")
        return []

    n_candidates = top_k * candidate_multiplier

    # ── Step 1: optional query transform ──────────────────────────────────────
    retrieval_query = query
    if use_query_rewrite:
        retrieval_query = query_rewrite(query, llm_client)

    # ── Step 2: build query vector (standard or HyDE) ─────────────────────────
    if use_hyde:
        query_vec = hyde_embed(retrieval_query, embedder, llm_client)
    else:
        query_vec = embedder.embed([retrieval_query])[0]

    # ── Step 3: dense retrieval ───────────────────────────────────────────────
    need_emb_for_mmr = use_mmr and not use_rerank
    dense_chunks = store.query(
        query_vec,
        top_k=n_candidates,
        include_embeddings=need_emb_for_mmr,
    )
    dense_ranking = [c.chunk_id for c in dense_chunks]

    # ── Step 4: BM25 retrieval ────────────────────────────────────────────────
    bm25_index = BM25Index(store)
    bm25_results = bm25_index.search(retrieval_query, top_k=n_candidates)
    bm25_ranking = [chunk_id for chunk_id, _ in bm25_results]

    # ── Step 5: RRF fusion ────────────────────────────────────────────────────
    all_chunks: dict[str, RetrievedChunk] = {c.chunk_id: c for c in dense_chunks}
    # Add any BM25-only chunks (not in dense results) from store
    bm25_only_ids = set(bm25_ranking) - all_chunks.keys()
    if bm25_only_ids:
        all_store_chunks = {c.chunk_id: c for c in store.get_all()}
        for cid in bm25_only_ids:
            if cid in all_store_chunks:
                all_chunks[cid] = all_store_chunks[cid]

    fused = rrf_fuse([bm25_ranking, dense_ranking], all_chunks)

    if score_threshold > 0:
        fused = [c for c in fused if c.score >= score_threshold]

    candidates = fused[:n_candidates]

    # ── Step 6: cross-encoder reranking ──────────────────────────────────────
    if use_rerank and candidates:
        candidates = reranker.rerank(query, candidates, top_k=n_candidates)

    # ── Step 7: MMR diversity selection ──────────────────────────────────────
    if use_mmr and candidates:
        # For MMR we need embeddings; re-fetch if not available (after rerank)
        if all(c.embedding is None for c in candidates[:3]):
            ids_needed = [c.chunk_id for c in candidates]
            refreshed = store.query(query_vec, top_k=len(candidates), include_embeddings=True)
            emb_lookup = {c.chunk_id: c.embedding for c in refreshed}
            import dataclasses
            candidates = [
                dataclasses.replace(c, embedding=emb_lookup.get(c.chunk_id))
                for c in candidates
            ]
        candidates = mmr_select(candidates, query_vec, lambda_=mmr_lambda, top_k=top_k)
    else:
        candidates = candidates[:top_k]

    logger.info(
        "hybrid_retrieve('%s'…) → %d chunks | rerank=%s hyde=%s mmr=%s rewrite=%s",
        query[:50], len(candidates), use_rerank, use_hyde, use_mmr, use_query_rewrite,
    )
    return candidates


def hybrid_retrieve_and_format(
    query: str,
    *,
    top_k: int = 5,
    max_context_tokens: int = 1500,
    **kwargs,
) -> tuple[str, list[RetrievedChunk]]:
    """
    Convenience wrapper: hybrid_retrieve + format_context in one call.

    Returns
    -------
    (context_str, chunks)
    """
    from app.rag.retrieve import format_context
    chunks = hybrid_retrieve(query, top_k=top_k, **kwargs)
    context = format_context(chunks, max_tokens=max_context_tokens)
    return context, chunks
