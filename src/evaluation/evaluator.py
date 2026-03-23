"""Unified retrieval and QA evaluation entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.retrieval_metrics import compute_retrieval_metrics
from inference.infer_retrieval import infer_retrieval_sample
from models.fusion_fixed import FixedFusion
from models.fusion_rrf import RRFFusion
from retrieval.retriever_manager import RetrieverManager
from utils.io_utils import save_csv, save_json, save_jsonl


class Evaluator:
    """Evaluate retrieval-only and retrieval-plus-QA pipelines."""

    def __init__(self, cfg: Any, retriever_manager: RetrieverManager) -> None:
        self.cfg = cfg
        self.retriever_manager = retriever_manager

    def evaluate_retrieval(
        self,
        dataset: list[dict[str, Any]],
        output_prefix: str,
    ) -> dict[str, dict[str, float]]:
        """Run phase-1 retrieval baselines and save metrics plus predictions."""
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
            evidence_page_ids = {self._normalize_evidence_page_id(sample["doc_id"], page) for page in sample.get("evidence_pages", [])}
            for method_name, method_fn in methods.items():
                result = infer_retrieval_sample(sample, bundle, method_name, method_fn)
                prediction_records[method_name].append(
                    {
                        "qid": sample["qid"],
                        "page_ids": result["page_ids"],
                        "scores": result["scores"],
                        "evidence_page_ids": sorted(evidence_page_ids),
                    }
                )
                detailed_rows.append(
                    {
                        "qid": sample["qid"],
                        "doc_id": sample["doc_id"],
                        "method": method_name,
                        "question": sample["question"],
                        "answer": sample.get("answer", ""),
                        "evidence_page_ids": sorted(evidence_page_ids),
                        "predicted_page_ids": result["page_ids"],
                        "predicted_scores": result["scores"],
                    }
                )

        metrics = {
            method_name: compute_retrieval_metrics(predictions, list(self.cfg.retrieval.evaluation_topk))
            for method_name, predictions in prediction_records.items()
        }
        self._save_retrieval_outputs(output_prefix, prediction_records, metrics, detailed_rows)
        return metrics

    def evaluate_qa(self, dataset: list[dict[str, Any]]) -> dict[str, float]:
        """Placeholder end-to-end QA evaluation entrypoint."""
        return {"status": 0.0, "message": "DocVQA QA evaluation is reserved for Phase 4."}

    def _save_retrieval_outputs(
        self,
        output_prefix: str,
        prediction_records: dict[str, list[dict[str, Any]]],
        metrics: dict[str, dict[str, float]],
        detailed_rows: list[dict[str, Any]],
    ) -> None:
        prediction_dir = Path(self.cfg.paths.prediction_dir)
        metric_dir = Path(self.cfg.paths.metric_dir)
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

    @staticmethod
    def _normalize_evidence_page_id(doc_id: str, evidence_page: Any) -> str:
        if isinstance(evidence_page, str) and "_page_" in evidence_page:
            return evidence_page
        page_number = int(evidence_page) + 1 if int(evidence_page) < 1000 else int(evidence_page)
        return f"{doc_id}_page_{page_number:03d}"
