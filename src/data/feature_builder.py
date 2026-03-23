"""Candidate feature construction for fusion and adaptive routing models."""

from __future__ import annotations

from typing import Any

from utils.rank_utils import reciprocal_rank


def build_candidate_features(candidate_pages: list[dict[str, Any]], question_features: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-page fusion features from merged candidate pages."""
    feature_rows: list[dict[str, Any]] = []
    for page in candidate_pages:
        row = {
            "page_id": page["page_id"],
            "visual_score": page.get("visual_score", 0.0),
            "ocr_score": page.get("ocr_score", 0.0),
            "visual_rank": page.get("visual_rank", 0),
            "ocr_rank": page.get("ocr_rank", 0),
            "visual_rr": reciprocal_rank(page.get("visual_rank")),
            "ocr_rr": reciprocal_rank(page.get("ocr_rank")),
            "overlap_flag": int(page.get("overlap_flag", 0)),
            "ocr_avg_confidence": page.get("ocr_avg_confidence", 0.0),
            "ocr_token_count": page.get("ocr_token_count", 0),
        }
        row.update(question_features)
        feature_rows.append(row)
    return feature_rows
