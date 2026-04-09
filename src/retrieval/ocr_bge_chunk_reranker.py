"""Chunk-level OCR reranker backed by a local BGE reranker."""

from __future__ import annotations

from typing import Any

from src.retrieval.ocr_bge_reranker import OCRBGEReranker
from src.retrieval.ocr_bge_retriever import format_bge_query


class OCRBGEChunkReranker:
    """Rerank question-chunk pairs and keep chunk metadata for page aggregation."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.reranker = OCRBGEReranker(cfg)

    def rerank(
        self,
        question: str,
        chunk_candidates: list[dict[str, Any]],
        topk: int,
        query_variant: str = "raw_question",
    ) -> dict[str, list]:
        """Rerank chunk candidates with a local cross-encoder-style reranker."""
        if not chunk_candidates:
            return {"chunk_ids": [], "page_ids": [], "scores": [], "ranks": [], "chunks": []}
        self.reranker._ensure_model()
        query_text = format_bge_query(question, variant=query_variant)
        scores = self.reranker._score_pairs(query_text, [str(item.get("chunk_text", "")) for item in chunk_candidates])
        ranked_items = sorted(
            zip(chunk_candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )[:topk]
        return {
            "chunk_ids": [str(item[0]["chunk_id"]) for item in ranked_items],
            "page_ids": [int(item[0]["page_id"]) for item in ranked_items],
            "scores": [float(item[1]) for item in ranked_items],
            "ranks": list(range(1, len(ranked_items) + 1)),
            "chunks": [dict(item[0]) for item in ranked_items],
        }
