"""
app/rag/ingest.py — Parse → Chunk → Embed → Store pipeline.

Supported formats: .md (markdown), .txt (plain text), .pdf (via pypdf).

Run directly to index the knowledge base:
    uv run python -m app.rag.ingest

Interview note — "What would you improve in a production ingest pipeline?"
  - Incremental ingestion: track file mtimes / hashes; skip unchanged files.
  - Async parallel embedding: embed multiple batches concurrently.
  - Unstructured.io: handles DOCX, HTML, images with OCR.
  - Table extraction: pypdf loses table structure; pdfplumber / camelot help.
  - Metadata enrichment: document date, author, section header from headings.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from app.config import Settings, settings as _default_settings
from app.rag.chunker import Chunk, chunk_text
from app.rag.embedder import Embedder, make_embedder
from app.rag.store import VectorStore, vector_store as _default_store

logger = logging.getLogger(__name__)

# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_markdown(path: Path) -> str:
    """Read markdown / plain-text as-is (keep headings for context)."""
    return path.read_text(encoding="utf-8")


def _parse_pdf(path: Path) -> str:
    """Extract text page-by-page and join with double newlines."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _parse_file(path: Path) -> str | None:
    """Dispatch to the right parser; return None for unsupported formats."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _parse_markdown(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    logger.warning("Unsupported file type '%s' — skipping %s", suffix, path.name)
    return None


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_file(
    path: Path,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    cfg: Settings | None = None,
) -> int:
    """
    Parse, chunk, embed, and store one file.

    Returns
    -------
    Number of chunks ingested (0 if the file was skipped).
    """
    cfg = cfg or _default_settings
    embedder = embedder or make_embedder(cfg)
    store = store or _default_store

    text = _parse_file(path)
    if not text or not text.strip():
        logger.warning("Empty content from '%s' — skipping.", path.name)
        return 0

    # source is the relative path from the KB dir's parent for readable metadata
    source = str(path)
    chunks: list[Chunk] = chunk_text(text, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return 0

    logger.info("'%s' → %d chunks", path.name, len(chunks))

    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    store.upsert(chunks, embeddings)
    return len(chunks)


def ingest_directory(
    kb_dir: Path | str | None = None,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    cfg: Settings | None = None,
) -> dict[str, int]:
    """
    Ingest all supported files in *kb_dir*.

    Returns
    -------
    Dict mapping filename → chunk count.
    """
    cfg = cfg or _default_settings
    embedder = embedder or make_embedder(cfg)
    store = store or VectorStore(cfg)

    if kb_dir is None:
        kb_dir = Path(__file__).resolve().parents[2] / "data" / "kb"
    kb_dir = Path(kb_dir)

    if not kb_dir.exists():
        raise FileNotFoundError(f"KB directory not found: {kb_dir}")

    results: dict[str, int] = {}
    files = sorted(kb_dir.rglob("*"))
    supported = [f for f in files if f.is_file() and f.suffix.lower() in {".md", ".txt", ".pdf"}]

    if not supported:
        logger.warning("No supported files found in '%s'", kb_dir)
        return results

    logger.info("Ingesting %d file(s) from '%s'…", len(supported), kb_dir)
    for fpath in supported:
        count = ingest_file(
            fpath,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedder=embedder,
            store=store,
            cfg=cfg,
        )
        results[fpath.name] = count

    total = sum(results.values())
    logger.info("Ingestion complete — %d chunks across %d file(s).", total, len(results))
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    kb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    summary = ingest_directory(kb_path)

    print("\n── Ingest summary ──────────────────────────")
    for fname, count in summary.items():
        print(f"  {fname:40s} {count:>4d} chunks")
    total = sum(summary.values())
    print(f"  {'TOTAL':40s} {total:>4d} chunks")
