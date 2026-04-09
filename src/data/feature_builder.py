"""Candidate feature construction for baseline and adaptive fusion models."""

from __future__ import annotations

from typing import Any

from src.utils.rank_utils import reciprocal_rank
from src.utils.score_utils import align_retriever_scores


def build_candidate_features(
    candidate_pages: list[dict[str, Any]],
    question_features: dict[str, Any] | None = None,
    quality_features: dict[str, dict[str, Any]] | None = None,
    normalization_method: str = "minmax",
) -> list[dict[str, Any]]:
    """Build per-page fusion features with baseline fields and extension slots."""
    question_features = question_features or {}
    quality_features = quality_features or {}
    vis_scores = [page.get("vis_score", page.get("visual_score", 0.0)) for page in candidate_pages]
    text_scores = [page.get("text_score", page.get("ocr_score", 0.0)) for page in candidate_pages]
    normalized_vis_scores = align_retriever_scores(vis_scores, method=normalization_method)
    normalized_text_scores = align_retriever_scores(text_scores, method=normalization_method)

    feature_rows: list[dict[str, Any]] = []
    for index, page in enumerate(candidate_pages):
        page_quality = quality_features.get(page["page_id"], {})
        vis_rank = int(page.get("vis_rank", page.get("visual_rank", 0)))
        text_rank = int(page.get("text_rank", page.get("ocr_rank", 0)))
        row = {
            "page_id": page["page_id"],
            "normalized_vis_score": normalized_vis_scores[index],
            "normalized_ocr_score": normalized_text_scores[index],
            "vis_rank": vis_rank,
            "ocr_rank": text_rank,
            "reciprocal_vis_rank": reciprocal_rank(vis_rank),
            "reciprocal_ocr_rank": reciprocal_rank(text_rank),
            "overlap_flag": int(page.get("overlap_flag", 0)),
        }
        row.update(page_quality)
        row.update(question_features)
        feature_rows.append(row)
    return feature_rows
