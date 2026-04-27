"""
app/rag/retrieve.py — Top-k dense retrieval + context assembly.

This is "naive RAG" — embed query → cosine search → top-k chunks → stuff prompt.

Interview note — "Why does naive RAG fail?"
  1. Lexical mismatch: query says "fee" but KB says "price" → no match.
  2. Lost-in-the-middle: stuffing many chunks makes the model ignore the middle ones.
  3. No reranking: embedding similarity ≠ answer relevance for long chunks.
  4. Multi-hop: a question needing two separate KB facts in sequence.
  5. Stale index: KB updated but not re-ingested.

Module 3 (RAG v2) fixes 1 with BM25, 3 with cross-encoder reranking,
and adds query rewriting to help with 4.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings, settings as _default_settings
from app.rag.embedder import Embedder, make_embedder
from app.rag.store import RetrievedChunk, VectorStore, vector_store as _default_store

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    score_threshold: float = 0.0,
    where: dict | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    cfg: Settings | None = None,
) -> list[RetrievedChunk]:
    """
    Embed *query* and return the top-k most similar chunks.

    Parameters
    ----------
    query:           The user question (raw text).
    top_k:           Max results to return.
    score_threshold: Discard chunks below this cosine similarity (0–1).
    where:           ChromaDB metadata filter (e.g. filter by source file).
    embedder:        Override the default embedder (useful in tests).
    store:           Override the default vector store (useful in tests).

    Returns
    -------
    List of RetrievedChunk sorted by score descending.
    """
    cfg = cfg or _default_settings
    embedder = embedder or make_embedder(cfg)
    store = store or _default_store

    if store.count() == 0:
        logger.warning("Vector store is empty. Run `python -m app.rag.ingest` first.")
        return []

    query_vec = embedder.embed([query])[0]
    chunks = store.query(query_vec, top_k=top_k, where=where)

    if score_threshold > 0:
        chunks = [c for c in chunks if c.score >= score_threshold]

    logger.debug("retrieve('%s') → %d chunks (top score=%.3f)", query[:60], len(chunks),
                 chunks[0].score if chunks else 0.0)
    return chunks


def format_context(chunks: list[RetrievedChunk], max_tokens: int = 1500) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.

    Each chunk is labelled with its source so the model can cite it.
    Chunks are included in score order until the token budget is exhausted.

    Interview note — "lost-in-the-middle":
      Research shows LLMs best recall information at the start/end of context.
      A mitigation is to put the most relevant chunk first AND last (bookending).
      M3 addresses this more formally with reranking + MMR diversity.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    lines: list[str] = []
    used_tokens = 0

    for i, chunk in enumerate(chunks, 1):
        source_name = Path(chunk.source).name
        header = f"[{i}] Source: {source_name} (score={chunk.score:.2f})"
        block = f"{header}\n{chunk.text}"
        block_tokens = len(enc.encode(block))

        if used_tokens + block_tokens > max_tokens:
            logger.debug("Context budget hit at chunk %d — stopping.", i)
            break

        lines.append(block)
        used_tokens += block_tokens

    return "\n\n---\n\n".join(lines)


def retrieve_and_format(
    query: str,
    *,
    top_k: int = 5,
    max_context_tokens: int = 1500,
    score_threshold: float = 0.0,
    **kwargs,
) -> tuple[str, list[RetrievedChunk]]:
    """
    Convenience wrapper: retrieve + format in one call.

    Returns
    -------
    (context_str, chunks)  — context_str goes into the prompt;
                             chunks are returned for citation display.
    """
    chunks = retrieve(query, top_k=top_k, score_threshold=score_threshold, **kwargs)
    context = format_context(chunks, max_tokens=max_context_tokens)
    return context, chunks
