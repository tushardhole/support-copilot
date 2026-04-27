"""
app/rag/chunker.py — Recursive, token-aware text chunker.

Why recursive?
  Naive fixed-size splitting ignores sentence and paragraph boundaries, leading
  to chunks that cut mid-sentence and lose context.  Recursive splitting tries
  progressively finer separators (paragraph → line → word → char) so chunks
  stay semantically coherent wherever possible.

Why tokens, not characters?
  LLM context windows are measured in tokens. A chunk that looks small in
  characters might be large in tokens for non-ASCII text. tiktoken gives exact
  token counts for the same BPE vocab the model uses.

Interview trade-offs
--------------------
  chunk_size ↑ → more context per chunk, fewer chunks retrieved, higher token cost
  chunk_size ↓ → more precise retrieval, but may split answers across chunks
  chunk_overlap → prevents answers that straddle a boundary from being missed;
                  costs extra storage / embed calls
  Semantic chunking (M3) can replace this with embedding-based boundary detection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

# cl100k_base is the BPE tokenizer used by gpt-4o, gpt-4, gpt-3.5-turbo,
# text-embedding-3-*, etc.  Using it ensures token counts match the model.
_ENC = tiktoken.get_encoding("cl100k_base")

# Separators tried in order — largest semantic unit first.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """One piece of a document, ready to embed and store."""

    text: str
    source: str          # relative file path or URL
    chunk_index: int
    start_char: int
    token_count: int
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        # Deterministic ID → re-ingesting the same file produces the same IDs.
        # ChromaDB upsert is idempotent when IDs match.
        key = f"{self.source}::{self.chunk_index}::{self.text[:64]}"
        self.chunk_id = hashlib.md5(key.encode()).hexdigest()

    def metadata(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "token_count": self.token_count,
        }


def _token_len(text: str) -> int:
    return len(_ENC.encode(text))


def _split_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """
    Recursively split text using the first separator that keeps chunks ≤ chunk_size.
    Falls back to the next separator when a piece is still too large.
    """
    if _token_len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0]
    next_seps = separators[1:]

    if sep == "":
        # Character-level last resort: hard split by token budget
        tokens = _ENC.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunks.append(_ENC.decode(tokens[start:end]))
            start += chunk_size - chunk_overlap
        return chunks

    pieces = text.split(sep)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = _token_len(piece)

        if piece_tokens > chunk_size and next_seps:
            # This piece alone is too big — recurse with a finer separator.
            if current:
                chunks.append(sep.join(current))
                # Carry overlap: keep last piece(s) that fit within overlap budget
                overlap_buf: list[str] = []
                overlap_tokens = 0
                for p in reversed(current):
                    t = _token_len(p)
                    if overlap_tokens + t <= chunk_overlap:
                        overlap_buf.insert(0, p)
                        overlap_tokens += t
                    else:
                        break
                current = overlap_buf
                current_tokens = overlap_tokens

            sub_chunks = _split_recursive(piece, chunk_size, chunk_overlap, next_seps)
            chunks.extend(sub_chunks[:-1])
            if sub_chunks:
                current = [sub_chunks[-1]]
                current_tokens = _token_len(sub_chunks[-1])
            continue

        if current_tokens + piece_tokens + _token_len(sep) > chunk_size and current:
            chunks.append(sep.join(current))
            # Overlap: retain tail pieces
            overlap_buf = []
            overlap_tokens = 0
            for p in reversed(current):
                t = _token_len(p)
                if overlap_tokens + t <= chunk_overlap:
                    overlap_buf.insert(0, p)
                    overlap_tokens += t
                else:
                    break
            current = overlap_buf
            current_tokens = overlap_tokens

        current.append(piece)
        current_tokens += piece_tokens + _token_len(sep)

    if current:
        chunks.append(sep.join(current))

    return [c for c in chunks if c.strip()]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """
    Split *text* into token-bounded, overlapping chunks.

    Parameters
    ----------
    text:          Full document text.
    source:        Source identifier (file path / URL) stored in metadata.
    chunk_size:    Max tokens per chunk (default 400 ≈ ~300 words).
    chunk_overlap: Token overlap between consecutive chunks (default 80).

    Returns
    -------
    List of Chunk objects with deterministic IDs, ready to embed.
    """
    raw_chunks = _split_recursive(text, chunk_size, chunk_overlap, _SEPARATORS)

    result: list[Chunk] = []
    char_cursor = 0
    for idx, raw in enumerate(raw_chunks):
        start = text.find(raw, char_cursor)
        if start == -1:
            start = char_cursor
        result.append(
            Chunk(
                text=raw,
                source=source,
                chunk_index=idx,
                start_char=start,
                token_count=_token_len(raw),
            )
        )
        char_cursor = start + len(raw)

    return result
