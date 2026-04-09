"""Score-level and rank-level fusion helpers for dual-route retrieval baselines."""

from __future__ import annotations

from typing import Any

from src.utils.rank_utils import sort_pages_by_score


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize one retriever's scores with stable edge-case handling."""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score <= 1e-8:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score + 1e-8) for score in scores]


def fixed_fusion(
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    alpha: float = 0.5,
    topk: int = 10,
) -> dict[str, list]:
    """Fuse OCR and visual retrieval results with normalized weighted score summation."""
    ocr_page_ids = [int(page_id) for page_id in ocr_result.get("page_ids", [])]
    visual_page_ids = [int(page_id) for page_id in visual_result.get("page_ids", [])]
    ocr_scores = normalize_scores([float(score) for score in ocr_result.get("scores", [])])
    visual_scores = normalize_scores([float(score) for score in visual_result.get("scores", [])])

    ocr_score_map = {page_id: score for page_id, score in zip(ocr_page_ids, ocr_scores)}
    visual_score_map = {page_id: score for page_id, score in zip(visual_page_ids, visual_scores)}
    union_page_ids = sorted(set(ocr_page_ids) | set(visual_page_ids))

    final_scores = [
        alpha * visual_score_map.get(page_id, 0.0) + (1.0 - alpha) * ocr_score_map.get(page_id, 0.0)
        for page_id in union_page_ids
    ]
    ranked = sort_pages_by_score(union_page_ids, final_scores)[:topk]
    return {
        "page_ids": [int(item["page_id"]) for item in ranked],
        "scores": [float(item["score"]) for item in ranked],
        "ranks": [int(item["rank"]) for item in ranked],
    }


def rrf_fusion(
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    rrf_k: int = 60,
    topk: int = 10,
) -> dict[str, list]:
    """Fuse OCR and visual rankings with Reciprocal Rank Fusion."""
    ocr_rank_map = {
        int(page_id): int(rank)
        for page_id, rank in zip(ocr_result.get("page_ids", []), ocr_result.get("ranks", []))
    }
    visual_rank_map = {
        int(page_id): int(rank)
        for page_id, rank in zip(visual_result.get("page_ids", []), visual_result.get("ranks", []))
    }
    union_page_ids = sorted(set(ocr_rank_map) | set(visual_rank_map))

    final_scores = []
    for page_id in union_page_ids:
        score = 0.0
        if page_id in visual_rank_map:
            score += 1.0 / float(rrf_k + visual_rank_map[page_id])
        if page_id in ocr_rank_map:
            score += 1.0 / float(rrf_k + ocr_rank_map[page_id])
        final_scores.append(score)

    ranked = sort_pages_by_score(union_page_ids, final_scores)[:topk]
    return {
        "page_ids": [int(item["page_id"]) for item in ranked],
        "scores": [float(item["score"]) for item in ranked],
        "ranks": [int(item["rank"]) for item in ranked],
    }
