"""Ranking helpers for retrieval scores and page ordering."""

from __future__ import annotations

from typing import Iterable


def scores_to_ranks(scores: Iterable[float], reverse: bool = True) -> list[int]:
    """Convert scores to 1-based ranks."""
    indexed_scores = list(enumerate(scores))
    sorted_pairs = sorted(indexed_scores, key=lambda item: item[1], reverse=reverse)
    ranks = [0] * len(indexed_scores)
    for rank, (index, _) in enumerate(sorted_pairs, start=1):
        ranks[index] = rank
    return ranks


def topk_from_scores(page_ids: list[str], scores: list[float], topk: int) -> dict[str, list]:
    """Return the top-k pages sorted by score."""
    ranked_items = sort_pages_by_score(page_ids, scores)
    top_items = ranked_items[:topk]
    return {
        "page_ids": [item["page_id"] for item in top_items],
        "scores": [item["score"] for item in top_items],
        "ranks": [item["rank"] for item in top_items],
    }


def reciprocal_rank(rank: int | None) -> float:
    """Compute reciprocal rank for a 1-based rank."""
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def sort_pages_by_score(page_ids: list[str], scores: list[float]) -> list[dict[str, float | int | str]]:
    """Sort pages by score and return page_id, score, and rank."""
    paired = sorted(zip(page_ids, scores), key=lambda item: item[1], reverse=True)
    return [
        {"page_id": page_id, "score": float(score), "rank": rank}
        for rank, (page_id, score) in enumerate(paired, start=1)
    ]
