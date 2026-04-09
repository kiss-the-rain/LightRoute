"""Reciprocal Rank Fusion baseline."""

from __future__ import annotations

from src.utils.rank_utils import reciprocal_rank, sort_pages_by_score


class RRFFusion:
    """Reciprocal Rank Fusion for visual and OCR retrieval lists."""

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, candidates: list[dict[str, float | int | str]]) -> dict[str, list]:
        """Fuse candidate rankings using RRF."""
        scores = []
        page_ids = []
        for candidate in candidates:
            page_ids.append(str(candidate["page_id"]))
            visual_rank = int(candidate.get("visual_rank", 0) or 0)
            ocr_rank = int(candidate.get("ocr_rank", 0) or 0)
            score = 0.0
            if visual_rank > 0:
                score += 1.0 / float(self.k + visual_rank)
            else:
                score += reciprocal_rank(None)
            if ocr_rank > 0:
                score += 1.0 / float(self.k + ocr_rank)
            else:
                score += reciprocal_rank(None)
            scores.append(score)

        ranked = sort_pages_by_score(page_ids, scores)
        return {
            "page_ids": [item["page_id"] for item in ranked],
            "scores": [item["score"] for item in ranked],
            "ranks": [item["rank"] for item in ranked],
        }
