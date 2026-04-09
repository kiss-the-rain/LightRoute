"""Retrieval metrics for page-level DocVQA evidence ranking."""

from __future__ import annotations

from typing import Iterable


def recall_at_k(pred_page_ids: list[int], evidence_pages: set[int], k: int) -> float:
    """Compute Recall@K for page retrieval."""
    if not evidence_pages:
        return 0.0
    return float(bool(set(pred_page_ids[:k]) & evidence_pages))


def hit_at_k(pred_page_ids: list[int], evidence_pages: set[int], k: int) -> float:
    """Compute Hit@K for page retrieval."""
    return float(bool(set(pred_page_ids[:k]) & evidence_pages))


def mrr(pred_page_ids: list[int], evidence_pages: set[int]) -> float:
    """Compute reciprocal rank of the first evidence-page hit."""
    for rank, page_id in enumerate(pred_page_ids, start=1):
        if page_id in evidence_pages:
            return 1.0 / float(rank)
    return 0.0


def page_level_accuracy(pred_page_ids: list[int], evidence_pages: set[int]) -> float:
    """Compute whether the top-1 predicted page is relevant."""
    return float(bool(pred_page_ids) and pred_page_ids[0] in evidence_pages)


def evaluate_retrieval(predictions: list[dict], k_values: list[int]) -> dict[str, float]:
    """Aggregate retrieval metrics from OCR-only or other page-level predictions."""
    if not predictions:
        return {}

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = sum(recall_at_k(item["pred_page_ids"], set(item["evidence_pages"]), k) for item in predictions) / len(predictions)
        metrics[f"Hit@{k}"] = sum(hit_at_k(item["pred_page_ids"], set(item["evidence_pages"]), k) for item in predictions) / len(predictions)
    metrics["MRR"] = sum(mrr(item["pred_page_ids"], set(item["evidence_pages"])) for item in predictions) / len(predictions)
    metrics["PageAcc"] = sum(page_level_accuracy(item["pred_page_ids"], set(item["evidence_pages"])) for item in predictions) / len(predictions)
    metrics["num_samples"] = len(predictions)
    return metrics
