"""Retrieval metrics for page-level DocVQA evidence ranking."""

from __future__ import annotations

from typing import Iterable


def recall_at_k(predicted_page_ids: list[str], evidence_page_ids: set[str], k: int) -> float:
    """Compute Recall@K for page retrieval."""
    if not evidence_page_ids:
        return 0.0
    hits = len(set(predicted_page_ids[:k]) & evidence_page_ids)
    return hits / float(len(evidence_page_ids))


def hit_at_k(predicted_page_ids: list[str], evidence_page_ids: set[str], k: int) -> float:
    """Compute Hit@K for page retrieval."""
    return float(bool(set(predicted_page_ids[:k]) & evidence_page_ids))


def mrr(predicted_page_ids: list[str], evidence_page_ids: set[str]) -> float:
    """Compute reciprocal rank of the first relevant page."""
    for rank, page_id in enumerate(predicted_page_ids, start=1):
        if page_id in evidence_page_ids:
            return 1.0 / float(rank)
    return 0.0


def page_level_accuracy(predicted_page_ids: list[str], evidence_page_ids: set[str]) -> float:
    """Compute whether the top-1 page is relevant."""
    return float(bool(predicted_page_ids) and predicted_page_ids[0] in evidence_page_ids)


def compute_retrieval_metrics(predictions: Iterable[dict], ks: list[int]) -> dict[str, float]:
    """Aggregate retrieval metrics over all samples."""
    predictions = list(predictions)
    if not predictions:
        return {}

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = sum(recall_at_k(item["page_ids"], set(item["evidence_page_ids"]), k) for item in predictions) / len(predictions)
        metrics[f"Hit@{k}"] = sum(hit_at_k(item["page_ids"], set(item["evidence_page_ids"]), k) for item in predictions) / len(predictions)
    metrics["MRR"] = sum(mrr(item["page_ids"], set(item["evidence_page_ids"])) for item in predictions) / len(predictions)
    metrics["PageAccuracy"] = sum(page_level_accuracy(item["page_ids"], set(item["evidence_page_ids"])) for item in predictions) / len(predictions)
    return metrics
