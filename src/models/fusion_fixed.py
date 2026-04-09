"""Fixed-weight dual-route fusion baseline."""

from __future__ import annotations

from typing import Any

from src.utils.rank_utils import sort_pages_by_score
from src.utils.score_utils import align_retriever_scores


class FixedFusion:
    """Fixed weighted sum fusion over normalized visual and OCR scores."""

    def __init__(self, alpha: float, normalization_method: str = "minmax") -> None:
        self.alpha = alpha
        self.normalization_method = normalization_method

    def fuse(self, candidates: list[dict[str, Any]]) -> dict[str, list]:
        """Fuse candidate scores with a configurable alpha."""
        visual_scores = [page.get("visual_score", 0.0) for page in candidates]
        text_scores = [page.get("ocr_score", 0.0) for page in candidates]
        aligned_visual = align_retriever_scores(visual_scores, self.normalization_method)
        aligned_text = align_retriever_scores(text_scores, self.normalization_method)

        final_scores = [
            self.alpha * visual_score + (1.0 - self.alpha) * text_score
            for visual_score, text_score in zip(aligned_visual, aligned_text)
        ]
        ranked = sort_pages_by_score([page["page_id"] for page in candidates], final_scores)
        return {
            "page_ids": [item["page_id"] for item in ranked],
            "scores": [item["score"] for item in ranked],
            "ranks": [item["rank"] for item in ranked],
        }
