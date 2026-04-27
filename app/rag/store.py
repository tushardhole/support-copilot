"""
app/rag/store.py — ChromaDB persistence layer.

ChromaDB chosen because:
  - Zero-ops local persistence (just a directory)
  - HNSW index → sub-millisecond ANN search on millions of vectors
  - Simple Python API, no Docker needed for development
  - Swap path: replace this module with lancedb / pgvector without touching
    ingest.py or retrieve.py (they only call VectorStore methods).

Interview note — when to swap ChromaDB out:
  - pgvector: when you already have Postgres and want SQL + vector in one place
  - Pinecone / Weaviate: when you need managed scale + multi-tenant isolation
  - LanceDB: when you want columnar storage + local-first with cloud sync
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.config import Settings, settings as _default_settings
from app.rag.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "support_kb"


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity query, with its relevance score."""

    chunk_id: str
    text: str
    source: str
    chunk_index: int
    score: float          # cosine similarity (higher = more relevant, max 1.0)
    token_count: int = 0
    embedding: list[float] | None = None  # populated when include_embeddings=True


class VectorStore:
    """
    Thin wrapper around a single ChromaDB collection.

    All writes use upsert so re-ingesting the same documents is idempotent —
    the same chunk_id simply overwrites its previous entry.
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        cfg = cfg or _default_settings
        persist_dir = str(cfg.chroma_persist_dir)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=persist_dir)
        # cosine distance for normalized embeddings (text-embedding-3-* and BGE are normalized)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug("VectorStore ready — collection '%s'", COLLECTION_NAME)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Store chunks with their embeddings.  Idempotent via chunk_id.

        Parameters
        ----------
        chunks:     Chunk objects (text + metadata).
        embeddings: One embedding vector per chunk (same order).
        """
        if not chunks:
            return

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )
        logger.info("Upserted %d chunks into '%s'", len(chunks), COLLECTION_NAME)

    def delete_source(self, source: str) -> None:
        """Remove all chunks from a specific source file (for re-ingestion)."""
        self._collection.delete(where={"source": source})
        logger.info("Deleted chunks for source '%s'", source)

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
        include_embeddings: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Return the top-k most similar chunks to the query embedding.

        Parameters
        ----------
        query_embedding:   Dense vector from the embedder.
        top_k:             Number of results (default 5).
        where:             Optional ChromaDB metadata filter.
        include_embeddings: If True, populate RetrievedChunk.embedding (needed for MMR).
        """
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=include,
        )
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        raw_embeddings = result.get("embeddings", [[]])[0] if include_embeddings else []

        chunks: list[RetrievedChunk] = []
        for i, (doc, meta, dist) in enumerate(zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )):
            # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
            # Convert to similarity score in [0, 1] for intuitive ranking.
            similarity = 1.0 - dist / 2.0
            chunks.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    text=doc,
                    source=str(meta.get("source", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=round(similarity, 4),
                    token_count=int(meta.get("token_count", 0)),
                    embedding=raw_embeddings[i] if raw_embeddings else None,
                )
            )

        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def get_all(self) -> list[RetrievedChunk]:
        """
        Return every chunk in the collection (used by BM25Index to build its corpus).
        Returns an empty list if the collection is empty.
        """
        if self.count() == 0:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        chunks: list[RetrievedChunk] = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            chunks.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    text=doc,
                    source=str(meta.get("source", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=1.0,
                    token_count=int(meta.get("token_count", 0)),
                )
            )
        return chunks

    def count(self) -> int:
        """Number of chunks currently in the collection."""
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Unique source files represented in the collection."""
        if self.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in result["metadatas"]}
        return sorted(sources)


# Module-level singleton
vector_store = VectorStore()
