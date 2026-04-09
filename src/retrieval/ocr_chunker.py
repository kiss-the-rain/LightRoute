"""OCR chunk construction and chunk-to-page score aggregation utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import re

from src.retrieval.ocr_bge_retriever import load_bge_page_text


def _extract_page_idx_from_path(path_or_name: str) -> int:
    """Extract an integer page index from names like `docid_p80.json`."""
    match = re.search(r"_p(\d+)(?:\.[^.]+)?$", str(path_or_name))
    if not match:
        raise ValueError(f"Unable to extract page index from path: {path_or_name}")
    return int(match.group(1))


@dataclass
class OCRChunkBuilder:
    """Build overlapping OCR text chunks for one document."""

    chunk_size: int = 128
    chunk_stride: int = 96
    page_text_variant: str = "clean_page_text"

    def build_document_chunks(self, doc_id: str, ocr_paths: list[str]) -> list[dict[str, Any]]:
        """Build page-aligned OCR chunks for one document."""
        chunks: list[dict[str, Any]] = []
        for ocr_path in ocr_paths:
            page_id = _extract_page_idx_from_path(ocr_path)
            page_text = load_bge_page_text(ocr_path, variant=self.page_text_variant)
            page_chunks = self.build_page_chunks(doc_id=doc_id, page_id=page_id, page_text=page_text)
            chunks.extend(page_chunks)
        return chunks

    def build_page_chunks(self, doc_id: str, page_id: int, page_text: str) -> list[dict[str, Any]]:
        """Split a single OCR page text into overlapping chunks."""
        tokens = str(page_text or "").split()
        if not tokens:
            return []
        if len(tokens) <= self.chunk_size:
            chunk_text = " ".join(tokens)
            return [
                {
                    "page_id": int(page_id),
                    "chunk_id": f"{doc_id}_p{page_id}_c0",
                    "chunk_text": chunk_text,
                    "chunk_start": 0,
                    "chunk_end": len(tokens),
                    "char_count": len(chunk_text),
                    "word_count": len(tokens),
                }
            ]

        chunks: list[dict[str, Any]] = []
        chunk_index = 0
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + self.chunk_size)
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                break
            chunk_text = " ".join(chunk_tokens)
            chunks.append(
                {
                    "page_id": int(page_id),
                    "chunk_id": f"{doc_id}_p{page_id}_c{chunk_index}",
                    "chunk_text": chunk_text,
                    "chunk_start": int(start),
                    "chunk_end": int(end),
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_tokens),
                }
            )
            if end >= len(tokens):
                break
            start += self.chunk_stride
            chunk_index += 1
        return chunks


def aggregate_chunk_scores_to_page(
    chunk_results: list[dict[str, Any]],
    strategy: str = "max",
    topk: int = 10,
) -> dict[str, list]:
    """Aggregate chunk-level scores back to page-level ranking."""
    page_to_scores: dict[int, list[float]] = defaultdict(list)
    for item in chunk_results:
        page_to_scores[int(item["page_id"])].append(float(item["score"]))

    ranked_items: list[tuple[int, float]] = []
    for page_id, scores in page_to_scores.items():
        sorted_scores = sorted(scores, reverse=True)
        if strategy == "max":
            page_score = sorted_scores[0]
        elif strategy == "top2_mean":
            page_score = sum(sorted_scores[:2]) / max(min(len(sorted_scores), 2), 1)
        elif strategy == "sum":
            page_score = sum(sorted_scores)
        else:
            raise ValueError(f"Unsupported chunk aggregation strategy: {strategy}")
        ranked_items.append((page_id, float(page_score)))

    ranked_items.sort(key=lambda item: item[1], reverse=True)
    ranked_items = ranked_items[:topk]
    return {
        "page_ids": [int(page_id) for page_id, _ in ranked_items],
        "scores": [float(score) for _, score in ranked_items],
        "ranks": list(range(1, len(ranked_items) + 1)),
    }
