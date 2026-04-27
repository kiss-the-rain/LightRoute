"""Small text chunking helpers used by OCR reranking experiments."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 64) -> list[str]:
    """Split text into overlapping character chunks while dropping empty chunks."""
    clean_text = str(text or "")
    effective_chunk_size = max(int(chunk_size), 1)
    effective_overlap = max(min(int(overlap), effective_chunk_size - 1), 0)
    step = max(effective_chunk_size - effective_overlap, 1)
    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        chunk = clean_text[start : start + effective_chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += step
    return chunks
