"""Unified evaluation helpers for retrieval-only and end-to-end DocVQA settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.error_analysis import analyze_retrieval_cases
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.inference.infer_retrieval import infer_retrieval_sample
from src.models.fusion_fixed import FixedFusion
from src.models.fusion_rrf import RRFFusion
from src.retrieval.retriever_manager import RetrieverManager
from src.utils.io_utils import save_csv, save_json, save_jsonl


class Evaluator:
    """Evaluate retrieval-only and retrieval-plus-QA pipelines."""

    def __init__(self, cfg: Any, retriever_manager: RetrieverManager) -> None:
        self.cfg = cfg
        self.retriever_manager = retriever_manager

    def evaluate_retrieval(self, dataset: list[dict[str, Any]], output_prefix: str) -> dict[str, dict[str, float]]:
        """Run legacy multi-branch retrieval evaluation."""
        fixed_fusion = FixedFusion(
            alpha=float(self.cfg.fusion.alpha),
            normalization_method=str(self.cfg.retrieval.score_normalization_method),
        )
        rrf_fusion = RRFFusion(k=int(self.cfg.fusion.rrf_k))
        methods = {
            "visual_only": lambda bundle: bundle["visual"],
            "ocr_bm25": lambda bundle: bundle["text"],
            "fixed_fusion": lambda bundle: fixed_fusion.fuse(bundle["candidates"]),
            "rrf_fusion": lambda bundle: rrf_fusion.fuse(bundle["candidates"]),
        }

        prediction_records: dict[str, list[dict[str, Any]]] = {name: [] for name in methods}
        detailed_rows: list[dict[str, Any]] = []
        for sample in dataset:
            bundle = self.retriever_manager.retrieve(sample)
            evidence_pages = [int(page) for page in sample.get("evidence_pages", [])]
            for method_name, method_fn in methods.items():
                result = infer_retrieval_sample(sample, bundle, method_name, method_fn)
                prediction_records[method_name].append(
                    {
                        "qid": sample["qid"],
                        "pred_page_ids": result["page_ids"],
                        "pred_scores": result["scores"],
                        "evidence_pages": evidence_pages,
                    }
                )
                detailed_rows.append(
                    {
                        "qid": sample["qid"],
                        "doc_id": sample["doc_id"],
                        "method": method_name,
                        "question": sample["question"],
                        "evidence_pages": evidence_pages,
                        "pred_page_ids": result["page_ids"],
                        "pred_scores": result["scores"],
                    }
                )

        metrics = {
            method_name: evaluate_retrieval(predictions, list(self.cfg.retrieval.evaluation_topk))
            for method_name, predictions in prediction_records.items()
        }
        _save_retrieval_outputs(output_prefix, prediction_records, metrics, detailed_rows)
        error_cases = analyze_retrieval_cases(
            visual_only_records=prediction_records["visual_only"],
            text_only_records=prediction_records["ocr_bm25"],
            fusion_records=prediction_records["fixed_fusion"],
        )
        save_json(error_cases, Path(self.cfg.paths.metric_dir) / f"{output_prefix}_error_analysis.json")
        return metrics


def evaluate_retrieval_predictions(predictions: list[dict], k_values: list[int]) -> dict[str, float]:
    """Evaluate retrieval predictions and return an aggregate metrics dict."""
    return evaluate_retrieval(predictions, k_values)


def save_retrieval_outputs(
    predictions: list[dict],
    metrics: dict[str, float],
    prediction_path: str | Path,
    metrics_path: str | Path | None = None,
) -> None:
    """Save retrieval predictions and optional metrics to disk."""
    save_jsonl(predictions, prediction_path)
    if metrics_path is not None:
        save_json(metrics, metrics_path)


def _save_retrieval_outputs(
    output_prefix: str,
    prediction_records: dict[str, list[dict[str, Any]]],
    metrics: dict[str, dict[str, float]],
    detailed_rows: list[dict[str, Any]],
) -> None:
    """Legacy helper retained for multi-branch evaluation output."""
    prediction_dir = Path("outputs/predictions")
    metric_dir = Path("outputs/metrics")
    for method_name, rows in prediction_records.items():
        save_jsonl(rows, prediction_dir / f"{output_prefix}_{method_name}.jsonl")
    save_json(metrics, metric_dir / f"{output_prefix}_retrieval_metrics.json")
    csv_rows = []
    for method_name, metric_values in metrics.items():
        row = {"method": method_name}
        row.update(metric_values)
        csv_rows.append(row)
    save_csv(csv_rows, metric_dir / f"{output_prefix}_retrieval_metrics.csv")
    save_jsonl(detailed_rows, prediction_dir / f"{output_prefix}_retrieval_detailed.jsonl")
