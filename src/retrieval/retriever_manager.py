"""Unified retrieval orchestrator for visual, OCR, and merged candidate outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval.candidate_merger import merge_candidates


class RetrieverManager:
    """Run both retrieval branches and produce merged candidate pages."""

    def __init__(self, cfg: Any, visual_retriever: Any, text_retriever: Any) -> None:
        self.cfg = cfg
        self.visual_retriever = visual_retriever
        self.text_retriever = text_retriever

    def retrieve(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Run document-scoped retrieval for a single sample."""
        question = sample["question"]
        doc_id = sample["doc_id"]
        visual_results = self.visual_retriever.retrieve(question, doc_id, int(self.cfg.retrieval.topk_visual))
        text_results = self.text_retriever.retrieve(question, doc_id, int(self.cfg.retrieval.topk_text))

        page_metadata = {}
        for ocr_entry in sample.get("ocr_results", []):
            page_metadata[ocr_entry["page_id"]] = {
                "ocr_avg_confidence": ocr_entry.get("avg_confidence", 0.0),
                "ocr_token_count": ocr_entry.get("token_count", 0),
            }
        for image_path in sample.get("image_paths", []):
            page_id = Path(image_path).stem
            page_metadata.setdefault(page_id, {})

        candidates = merge_candidates(
            visual_results=visual_results,
            text_results=text_results,
            mode=str(self.cfg.retrieval.candidate_merge_mode),
            page_metadata=page_metadata,
        )
        return {
            "visual": visual_results,
            "text": text_results,
            "candidates": candidates,
        }
