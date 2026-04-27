"""
app/rag/embedder.py — Provider-agnostic embedding layer.

Two backends, selected by EMBED_PROVIDER in config:

  "openai"  → calls the OpenAI-compatible /embeddings endpoint in batches.
              Requires OPENAI_API_KEY + OPENAI_BASE_URL.
              Pros: no local GPU/RAM, easy to swap models.
              Cons: network latency, cost per token, data leaves your network.

  "local"   → sentence-transformers running locally (no API call).
              Model is downloaded once and cached by HuggingFace.
              Pros: free, offline, low latency after warm-up.
              Cons: first-load is slow; needs RAM / CPU (or GPU).

Interview trade-off:
  "Which embedding model would you choose?"
  → text-embedding-3-small: cheap, fast, 1536-dim, good baseline.
  → text-embedding-3-large: better recall, 3072-dim, 5× more expensive.
  → BAAI/bge-small-en-v1.5: strong local option, 384-dim, ~50ms on CPU.
  → Choose based on: latency SLA, data privacy, retrieval quality target.
"""

from __future__ import annotations

import logging
from typing import Protocol

from openai import OpenAI

from app.config import EmbedProvider, Settings, settings as _default_settings

logger = logging.getLogger(__name__)

Vector = list[float]


# ── Protocol (interface) ──────────────────────────────────────────────────────

class Embedder(Protocol):
    """Any object with an embed() method satisfies this interface."""

    def embed(self, texts: list[str]) -> list[Vector]:
        """Return one embedding vector per input text."""
        ...


# ── OpenAI backend ────────────────────────────────────────────────────────────

class OpenAIEmbedder:
    """
    Calls the /embeddings endpoint in batches to stay within API limits.

    Batch size 512 is safe for text-embedding-3-small (max 2048 inputs/call).
    Reduce if you hit payload size limits with long texts.
    """

    _BATCH_SIZE = 512

    def __init__(self, cfg: Settings) -> None:
        self._client = OpenAI(
            base_url=cfg.openai_base_url,
            api_key=cfg.openai_api_key or "unused",
        )
        self._model = cfg.embed_model

    def embed(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []

        results: list[Vector] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            logger.debug("Embedding batch %d–%d via OpenAI (%s)", i, i + len(batch), self._model)
            response = self._client.embeddings.create(model=self._model, input=batch)
            # API returns items sorted by index — safe to extend in order.
            results.extend(item.embedding for item in response.data)
        return results


# ── Local (sentence-transformers) backend ─────────────────────────────────────

class LocalEmbedder:
    """
    Runs a HuggingFace sentence-transformers model entirely on-device.

    The model is lazily loaded on first call so import time stays fast even
    when the local backend is configured but the embedder isn't used yet.
    """

    def __init__(self, cfg: Settings) -> None:
        self._model_name = cfg.embed_local_model
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading local embedding model '%s'…", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Local embedding model ready.")

    def embed(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        self._load()
        vecs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


# ── Factory ───────────────────────────────────────────────────────────────────

def make_embedder(cfg: Settings | None = None) -> Embedder:
    """Return the configured embedder backend."""
    cfg = cfg or _default_settings
    if cfg.embed_provider == EmbedProvider.local:
        return LocalEmbedder(cfg)
    return OpenAIEmbedder(cfg)


# Module-level singleton — re-use across calls to avoid re-loading local model.
embedder: Embedder = make_embedder()
