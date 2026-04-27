"""Retrieval metrics for page-level DocVQA evidence ranking."""

from __future__ import annotations

from typing import Iterable
import math


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


class StreamingSetRetrievalMetrics:
    """Online aggregation for retrieval metrics over set-valued relevance labels."""

    def __init__(self, k_values: Iterable[int], *, track_top_prefix: bool = False) -> None:
        self.k_values = sorted({int(k) for k in k_values if int(k) > 0})
        self.track_top_prefix = bool(track_top_prefix)
        self.sample_count = 0
        self.recall_sums = {k: 0.0 for k in self.k_values}
        self.hit_sums = {k: 0.0 for k in self.k_values}
        self.top_sums = {k: 0.0 for k in self.k_values} if self.track_top_prefix else {}
        self.mrr_sum = 0.0
        self.page_acc_sum = 0.0
        self.ndcg_sums: dict[int, float] = {}

    def register_ndcg(self, *k_values: int) -> None:
        for k in k_values:
            k = int(k)
            if k > 0:
                self.ndcg_sums.setdefault(k, 0.0)

    def update(self, pred_ids: list[str | int], gold_ids: Iterable[str | int]) -> None:
        gold = {str(item) for item in gold_ids}
        if not gold:
            raise RuntimeError("Streaming retrieval metrics received empty gold labels.")
        pred = [str(item) for item in pred_ids]
        self.sample_count += 1
        for k in self.k_values:
            hits = len(set(pred[:k]) & gold)
            recall_value = float(hits / max(len(gold), 1))
            hit_value = float(hits > 0)
            self.recall_sums[k] += recall_value
            self.hit_sums[k] += hit_value
            if self.track_top_prefix:
                self.top_sums[k] += hit_value
        for rank, pred_id in enumerate(pred, start=1):
            if pred_id in gold:
                self.mrr_sum += 1.0 / float(rank)
                break
        self.page_acc_sum += float(bool(pred) and pred[0] in gold)
        for k in list(self.ndcg_sums.keys()):
            dcg = 0.0
            for index, pred_id in enumerate(pred[:k], start=1):
                if pred_id in gold:
                    dcg += 1.0 / math.log2(index + 1)
            ideal_hits = min(len(gold), k)
            idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
            self.ndcg_sums[k] += dcg / max(idcg, 1e-8)

    def finalize(self) -> dict[str, float]:
        if self.sample_count == 0:
            return {}
        metrics: dict[str, float] = {}
        for k in self.k_values:
            if self.track_top_prefix:
                metrics[f"Top@{k}"] = self.top_sums[k] / self.sample_count
            metrics[f"Recall@{k}"] = self.recall_sums[k] / self.sample_count
            metrics[f"Hit@{k}"] = self.hit_sums[k] / self.sample_count
        metrics["MRR"] = self.mrr_sum / self.sample_count
        metrics["PageAcc"] = self.page_acc_sum / self.sample_count
        metrics["num_samples"] = self.sample_count
        for k, value in self.ndcg_sums.items():
            metrics[f"nDCG@{k}"] = value / self.sample_count
            metrics[f"NDCG@{k}"] = metrics[f"nDCG@{k}"]
        return metrics
