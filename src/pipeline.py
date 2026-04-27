"""Top-level pipeline orchestration for retrieval, training, and DocVQA inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.dataset import DatasetManager
from src.data.doc_dataset import DocVQADataset
from src.data.ocr_engine import OCREngine
from src.data.pdf_renderer import render_pdf_to_images
from src.evaluation.case_study import export_case_study
from src.evaluation.evaluator import Evaluator
from src.evaluation.qa_metrics import anls, exact_match, f1_score
from src.inference.infer_docvqa import infer_docvqa_sample
from src.retrieval.colpali_retriever import ColPaliRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.index_builder import build_text_index, build_visual_index, collect_document_pages
from src.retrieval.retriever_manager import RetrieverManager
from src.training.train_fusion import (
    train_fusion,
    train_adaptive_fusion_dynamic_rules,
    train_adaptive_fusion_gating_calibrated,
    train_adaptive_fusion_learned_gating,
    train_adaptive_fusion_visual_nemotron_ocr_energy,
    train_fusion_ablation,
    train_fusion_mlp_ocrq_bge,
    train_fusion_mlp_ocrq_chunk,
    train_fusion_visual_colqwen_ocr_chunk,
    train_fusion_mlp_ocrq_chunkplus,
    train_fusion_mlp_ocrq_hybrid,
    train_fusion_mlp_ocrq_nvchunk,
    train_fusion_v2,
)
from src.utils.io_utils import ensure_dir, save_json, save_jsonl
from src.utils.logger import get_logger
from src.utils.seed_utils import set_seed
from src.inference.infer_retrieval import (
    run_adaptive_fusion_on_dataset,
    run_adaptive_fusion_ablation_on_dataset,
    run_adaptive_fusion_dynamic_rules_on_dataset,
    run_adaptive_fusion_gating_calibrated_on_dataset,
    run_adaptive_fusion_learned_gating_on_dataset,
    run_adaptive_fusion_mlp_ocrq_bge_on_dataset,
    run_adaptive_fusion_mlp_ocrq_chunk_on_dataset,
    run_bm25_600_nemotron_bge_mpdocvqa_on_dataset,
    run_bm25_600_nemotron_bge_selective_mpdocvqa_on_dataset,
    run_bm25_600_nemotron_bge_topk_rerank_mpdocvqa_on_dataset,
    run_bm25_600_nemotron_jina_mpdocvqa_on_dataset,
    run_bm25_600_nemotron_jina_chunk_mpdocvqa_on_dataset,
    run_bm25_600_nemotron_bge_dynamic_mpdocvqa_on_dataset,
    run_bm25_600_queryaware_dynamic_mpdocvqa_on_dataset,
    run_bm25_600_queryaware_visual_dominant_mpdocvqa_on_dataset,
    run_bm25_600_selective_confidence_mpdocvqa_on_dataset,
    run_bm25_600_nemotron_bge_vidore_energy_on_dataset,
    run_adaptive_fusion_visual_nemotron_ocr_energy_on_dataset,
    run_adaptive_fusion_visual_colqwen_ocr_chunk_on_dataset,
    run_adaptive_fusion_mlp_ocrq_chunkplus_on_dataset,
    run_adaptive_fusion_mlp_ocrq_hybrid_on_dataset,
    run_adaptive_fusion_mlp_ocrq_nvchunk_on_dataset,
    run_adaptive_fusion_v2_on_dataset,
    run_bm25_vidore_energy_on_dataset,
    run_bm25_retrieval_on_dataset,
    run_ocr_bge_debug_on_dataset,
    run_ocr_bge_chunk_retrieval_on_dataset,
    run_ocr_page_bm25_bge_rerank_on_dataset,
    run_ocr_page_coarse_chunk_on_dataset,
    run_ocr_bge_retrieval_on_dataset,
    run_ocr_hybrid_retrieval_on_dataset,
    run_ocr_jina_chunk_retrieval_on_dataset,
    run_ocr_nv_chunk_retrieval_on_dataset,
    run_fixed_fusion_on_dataset,
    run_rrf_fusion_on_dataset,
    run_visual_colqwen_adaptive_coarse_on_dataset,
    run_visual_colqwen_retrieval_on_dataset,
    run_visual_nemotron_energy_on_dataset,
    run_visual_nemotron_mpdocvqa_on_dataset,
    run_visual_nemotron_energy_sanity_check,
    run_visual_retrieval_on_dataset,
)


def prepare_data_pipeline(cfg: Any) -> None:
    """Prepare processed samples, page images, and OCR caches."""
    logger = _build_logger(cfg, "prepare_data")
    set_seed(int(cfg.seed))
    _ensure_runtime_dirs(cfg)
    dataset_manager = DatasetManager(cfg)
    ocr_engine = OCREngine(backend="dummy")

    for split in cfg.dataset.processed_files.keys():
        processed_path = Path(cfg.dataset.processed_files[split])
        if cfg.dataset.name == "mpdocvqa":
            report_path = Path(cfg.paths.log_dir) / f"mpdocvqa_build_report_{split}.json"
            if _should_skip_split(cfg, processed_path, report_path):
                logger.info(
                    "Skipping split %s because processed file and build report already exist: %s, %s",
                    split,
                    processed_path,
                    report_path,
                )
                continue
        elif _should_skip(cfg, processed_path):
            logger.info("Skipping split %s because processed file exists: %s", split, processed_path)
            continue

        logger.info("Building processed split: %s", split)
        samples = dataset_manager.prepare(split)
        if not samples:
            logger.info("No raw annotations found for split %s. Skipping OCR/render stage.", split)
            continue

        for sample in samples:
            _ensure_sample_page_images(cfg, sample, logger)
            _ensure_sample_ocr(cfg, sample, ocr_engine, logger)

        save_jsonl(samples, processed_path)

    logger.info("Prepare-data pipeline finished.")


def build_indexes_pipeline(cfg: Any) -> None:
    """Build visual and BM25 text retrieval indexes."""
    logger = _build_logger(cfg, "build_indexes")
    _ensure_runtime_dirs(cfg)
    dataset = _load_split_dataset(cfg, str(cfg.experiment.split))
    processed_samples = list(dataset)
    documents = collect_document_pages(cfg, processed_samples)
    visual_retriever = ColPaliRetriever(cfg)
    text_retriever = BM25Retriever(cfg)
    logger.info("Building visual index for %d documents.", len(documents))
    build_visual_index(visual_retriever, documents)
    logger.info("Building text index for %d documents.", len(documents))
    build_text_index(text_retriever, documents)
    logger.info("Index build completed.")


def eval_retrieval_pipeline(cfg: Any) -> None:
    """Run Phase 1 retrieval baselines and compute retrieval metrics."""
    logger = _build_logger(cfg, "eval_retrieval")
    _ensure_runtime_dirs(cfg)
    dataset = _load_split_dataset(cfg, str(cfg.experiment.split))
    processed_samples = list(dataset)
    documents = collect_document_pages(cfg, processed_samples)
    visual_retriever = ColPaliRetriever(cfg)
    text_retriever = BM25Retriever(cfg)
    build_visual_index(visual_retriever, documents)
    build_text_index(text_retriever, documents)
    retriever_manager = RetrieverManager(cfg, visual_retriever, text_retriever)
    evaluator = Evaluator(cfg, retriever_manager)
    output_prefix = f"{cfg.experiment.name}_{cfg.experiment.split}"
    metrics = evaluator.evaluate_retrieval(processed_samples, output_prefix=output_prefix)
    logger.info("Retrieval evaluation completed: %s", metrics)


def eval_bm25_retrieval_pipeline(cfg: Any) -> None:
    """Run OCR-only BM25 retrieval on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_bm25_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_bm25_retrieval_on_dataset(
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved BM25 val predictions to %s", prediction_path)
    logger.info("Saved BM25 val metrics to %s", metrics_path)
    logger.info("BM25 val metrics: %s", metrics)


def infer_bm25_test_pipeline(cfg: Any) -> None:
    """Run OCR-only BM25 retrieval on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_bm25_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_bm25_retrieval_on_dataset(
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved BM25 test predictions to %s", prediction_path)
    if metrics:
        logger.info("BM25 test metrics: %s", metrics)


def eval_ocr_bge_retrieval_pipeline(cfg: Any) -> None:
    """Run OCR BGE dense retrieval on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_ocr_bge_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_ocr_bge_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.ocr_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        use_reranker=False,
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_bge_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_bge_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved OCR BGE val predictions to %s", prediction_path)
    logger.info("Saved OCR BGE val metrics to %s", metrics_path)


def infer_ocr_bge_test_pipeline(cfg: Any) -> None:
    """Run OCR BGE dense retrieval on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_ocr_bge_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_ocr_bge_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.ocr_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        use_reranker=False,
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_bge_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved OCR BGE test predictions to %s", prediction_path)
    if metrics:
        logger.info("OCR BGE test metrics: %s", metrics)


def eval_ocr_bge_rerank_pipeline(cfg: Any) -> None:
    """Run OCR BGE retrieval + rerank on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_ocr_bge_rerank_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_ocr_bge_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.ocr_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        use_reranker=True,
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_bge_rerank_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_bge_rerank_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved OCR BGE rerank val predictions to %s", prediction_path)
    logger.info("Saved OCR BGE rerank val metrics to %s", metrics_path)


def eval_ocr_bge_debug_pipeline(cfg: Any) -> None:
    """Run OCR-BGE debug experiments on the validation split and save analysis outputs."""
    logger = _build_logger(cfg, "eval_ocr_bge_debug_val")
    _ensure_runtime_dirs(cfg)
    summary, run_records, failure_cases = run_ocr_bge_debug_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        use_reranker=False,
    )
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_bge_debug_val_metrics.json"
    runs_path = Path(cfg.paths.output_dir) / "debug" / "ocr_bge_debug_runs.jsonl"
    failures_path = Path(cfg.paths.output_dir) / "debug" / "ocr_bge_failure_cases.jsonl"
    ensure_dir(runs_path.parent)
    save_json(summary, metrics_path)
    save_jsonl(run_records, runs_path)
    save_jsonl(failure_cases, failures_path)
    logger.info("Saved OCR BGE debug summary to %s", metrics_path)
    logger.info("Saved OCR BGE debug runs to %s", runs_path)
    logger.info("Saved OCR BGE debug failure cases to %s", failures_path)


def eval_ocr_bge_rerank_debug_pipeline(cfg: Any) -> None:
    """Run OCR-BGE rerank debug experiments on the validation split and save analysis outputs."""
    logger = _build_logger(cfg, "eval_ocr_bge_rerank_debug_val")
    _ensure_runtime_dirs(cfg)
    summary, run_records, failure_cases = run_ocr_bge_debug_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        use_reranker=True,
    )
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_bge_rerank_debug_val_metrics.json"
    runs_path = Path(cfg.paths.output_dir) / "debug" / "ocr_bge_rerank_debug_runs.jsonl"
    failures_path = Path(cfg.paths.output_dir) / "debug" / "ocr_bge_rerank_failure_cases.jsonl"
    ensure_dir(runs_path.parent)
    save_json(summary, metrics_path)
    save_jsonl(run_records, runs_path)
    save_jsonl(failure_cases, failures_path)
    logger.info("Saved OCR BGE rerank debug summary to %s", metrics_path)
    logger.info("Saved OCR BGE rerank debug runs to %s", runs_path)
    logger.info("Saved OCR BGE rerank debug failure cases to %s", failures_path)


def infer_ocr_bge_rerank_test_pipeline(cfg: Any) -> None:
    """Run OCR BGE retrieval + rerank on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_ocr_bge_rerank_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_ocr_bge_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.ocr_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        use_reranker=True,
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_bge_rerank_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved OCR BGE rerank test predictions to %s", prediction_path)
    if metrics:
        logger.info("OCR BGE rerank test metrics: %s", metrics)


def eval_ocr_bge_chunk_pipeline(cfg: Any) -> None:
    """Run chunk-level OCR BGE retrieval on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_ocr_bge_chunk_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics, failure_cases = run_ocr_bge_chunk_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.ocr_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        use_reranker=False,
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_bge_chunk_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_bge_chunk_val_metrics.json"
    failure_path = Path(cfg.paths.output_dir) / "debug" / "ocr_bge_chunk_failure_cases.jsonl"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    save_jsonl(failure_cases, failure_path)
    logger.info("Saved OCR BGE chunk val predictions to %s", prediction_path)
    logger.info("Saved OCR BGE chunk val metrics to %s", metrics_path)


def infer_ocr_bge_chunk_test_pipeline(cfg: Any) -> None:
    """Run chunk-level OCR BGE retrieval on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_ocr_bge_chunk_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics, failure_cases = run_ocr_bge_chunk_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.ocr_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        use_reranker=False,
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_bge_chunk_test_predictions.jsonl"
    failure_path = Path(cfg.paths.output_dir) / "debug" / "ocr_bge_chunk_test_failure_cases.jsonl"
    save_jsonl(predictions, prediction_path)
    save_jsonl(failure_cases, failure_path)
    logger.info("Saved OCR BGE chunk test predictions to %s", prediction_path)
    if metrics:
        logger.info("OCR BGE chunk test metrics: %s", metrics)


def eval_ocr_nv_chunk_pipeline(cfg: Any) -> None:
    """Run chunk-level OCR retrieval with NV-Embed-v2 on the validation split."""
    logger = _build_logger(cfg, "eval_ocr_nv_chunk_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_ocr_nv_chunk_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.ocr_nv_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_nv_chunk_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_nv_chunk_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved OCR NV chunk val predictions to %s", prediction_path)
    logger.info("Saved OCR NV chunk val metrics to %s", metrics_path)


def eval_ocr_jina_chunk_pipeline(cfg: Any) -> None:
    """Run offline jina-embeddings-v3 chunk OCR retrieval on validation and save outputs."""
    logger = _build_logger(cfg, "eval_ocr_jina_chunk_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_ocr_jina_chunk_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.ocr_jina_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_jina_chunk_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_jina_chunk_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved OCR jina chunk val predictions to %s", prediction_path)
    logger.info("Saved OCR jina chunk val metrics to %s", metrics_path)


def infer_ocr_nv_chunk_test_pipeline(cfg: Any) -> None:
    """Run chunk-level OCR retrieval with NV-Embed-v2 on the test split."""
    logger = _build_logger(cfg, "infer_ocr_nv_chunk_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_ocr_nv_chunk_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.ocr_nv_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_nv_chunk_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved OCR NV chunk test predictions to %s", prediction_path)


def infer_ocr_jina_chunk_test_pipeline(cfg: Any) -> None:
    """Run offline jina-embeddings-v3 chunk OCR retrieval on test and save predictions."""
    logger = _build_logger(cfg, "infer_ocr_jina_chunk_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_ocr_jina_chunk_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.ocr_jina_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_jina_chunk_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved OCR jina chunk test predictions to %s", prediction_path)
    if metrics:
        logger.info("OCR NV chunk test metrics: %s", metrics)


def eval_ocr_bge_chunk_rerank_pipeline(cfg: Any) -> None:
    """Run chunk-level OCR BGE retrieval + rerank on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_ocr_bge_chunk_rerank_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics, failure_cases = run_ocr_bge_chunk_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.ocr_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        use_reranker=True,
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_bge_chunk_rerank_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_bge_chunk_rerank_val_metrics.json"
    failure_path = Path(cfg.paths.output_dir) / "debug" / "ocr_bge_chunk_rerank_failure_cases.jsonl"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    save_jsonl(failure_cases, failure_path)
    logger.info("Saved OCR BGE chunk rerank val predictions to %s", prediction_path)
    logger.info("Saved OCR BGE chunk rerank val metrics to %s", metrics_path)


def infer_ocr_bge_chunk_rerank_test_pipeline(cfg: Any) -> None:
    """Run chunk-level OCR BGE retrieval + rerank on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_ocr_bge_chunk_rerank_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics, failure_cases = run_ocr_bge_chunk_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.ocr_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        use_reranker=True,
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_bge_chunk_rerank_test_predictions.jsonl"
    failure_path = Path(cfg.paths.output_dir) / "debug" / "ocr_bge_chunk_rerank_test_failure_cases.jsonl"
    save_jsonl(predictions, prediction_path)
    save_jsonl(failure_cases, failure_path)
    logger.info("Saved OCR BGE chunk rerank test predictions to %s", prediction_path)
    if metrics:
        logger.info("OCR BGE chunk rerank test metrics: %s", metrics)


def eval_ocr_hybrid_pipeline(cfg: Any) -> None:
    """Run hybrid OCR retrieval on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_ocr_hybrid_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics, failure_cases = run_ocr_hybrid_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.ocr_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_hybrid_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_hybrid_val_metrics.json"
    failure_path = Path(cfg.paths.output_dir) / "debug" / "ocr_hybrid_failure_cases.jsonl"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    save_jsonl(failure_cases, failure_path)
    logger.info("Saved OCR hybrid val predictions to %s", prediction_path)
    logger.info("Saved OCR hybrid val metrics to %s", metrics_path)


def infer_ocr_hybrid_test_pipeline(cfg: Any) -> None:
    """Run hybrid OCR retrieval on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_ocr_hybrid_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics, failure_cases = run_ocr_hybrid_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.ocr_retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_hybrid_test_predictions.jsonl"
    failure_path = Path(cfg.paths.output_dir) / "debug" / "ocr_hybrid_test_failure_cases.jsonl"
    save_jsonl(predictions, prediction_path)
    save_jsonl(failure_cases, failure_path)
    logger.info("Saved OCR hybrid test predictions to %s", prediction_path)
    if metrics:
        logger.info("OCR hybrid test metrics: %s", metrics)


def eval_visual_retrieval_pipeline(cfg: Any) -> None:
    """Run visual-only retrieval on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_visual_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_visual_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "visual_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "visual_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved visual val predictions to %s", prediction_path)
    logger.info("Saved visual val metrics to %s", metrics_path)
    logger.info("Visual val metrics: %s", metrics)


def eval_visual_colqwen_retrieval_pipeline(cfg: Any) -> None:
    """Run ColQwen visual-only retrieval on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_visual_colqwen_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_visual_colqwen_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "visual_colqwen_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "visual_colqwen_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved visual ColQwen val predictions to %s", prediction_path)
    logger.info("Saved visual ColQwen val metrics to %s", metrics_path)
    logger.info("Visual ColQwen val metrics: %s", metrics)


def eval_visual_colqwen_adaptive_coarse_retrieval_pipeline(cfg: Any) -> None:
    """Run adaptive coarse BM25 routing + ColQwen subset reranking on val."""
    logger = _build_logger(cfg, "eval_visual_colqwen_adaptive_coarse_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_visual_colqwen_adaptive_coarse_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "visual_colqwen_adaptive_coarse_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "visual_colqwen_adaptive_coarse_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved visual ColQwen adaptive coarse val predictions to %s", prediction_path)
    logger.info("Saved visual ColQwen adaptive coarse val metrics to %s", metrics_path)
    logger.info("Visual ColQwen adaptive coarse val metrics: %s", metrics)


def eval_visual_nemotron_energy_pipeline(cfg: Any) -> None:
    """Run local nemotron-colembed-vl-8b-v2 visual-only retrieval on ViDoRe V3 Energy."""
    logger = _build_logger(cfg, "eval_visual_nemotron_energy")
    _ensure_runtime_dirs(cfg)
    prediction_path = Path(cfg.paths.prediction_dir) / "visual_nemotron_energy_predictions.jsonl"
    _predictions, metrics = run_visual_nemotron_energy_on_dataset(
        cfg=cfg,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "visual_nemotron_energy_metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Saved visual nemotron energy predictions to %s", prediction_path)
    logger.info("Saved visual nemotron energy metrics to %s", metrics_path)
    logger.info("Visual nemotron energy metrics: %s", metrics)


def eval_bm25_vidore_energy_pipeline(cfg: Any) -> None:
    """Run BM25-only retrieval on ViDoRe Energy and save outputs."""
    logger = _build_logger(cfg, "eval_bm25_vidore_energy")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_bm25_vidore_energy_on_dataset(
        cfg=cfg,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_vidore_energy_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_vidore_energy_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved BM25 ViDoRe Energy predictions to %s", prediction_path)
    logger.info("Saved BM25 ViDoRe Energy metrics to %s", metrics_path)
    logger.info("BM25 ViDoRe Energy metrics: %s", metrics)


def eval_bm25_600_nemotron_bge_vidore_energy_pipeline(cfg: Any) -> None:
    """Run BM25@600 + parallel Nemotron/BGE evaluation on ViDoRe Energy."""
    logger = _build_logger(cfg, "eval_bm25_600_nemotron_bge_vidore_energy")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_bm25_600_nemotron_bge_vidore_energy_on_dataset(
        cfg=cfg,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_600_nemotron_bge_vidore_energy_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_600_nemotron_bge_vidore_energy_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved BM25-600 Nemotron+BGE ViDoRe Energy predictions to %s", prediction_path)
    logger.info("Saved BM25-600 Nemotron+BGE ViDoRe Energy metrics to %s", metrics_path)
    logger.info("BM25-600 Nemotron+BGE ViDoRe Energy metrics: %s", metrics)


def eval_bm25_600_nemotron_bge_mpdocvqa_pipeline(cfg: Any) -> None:
    """Run BM25@600 + parallel Nemotron/BGE evaluation on MP-DocVQA."""
    logger = _build_logger(cfg, "eval_bm25_600_nemotron_bge_mpdocvqa")
    _ensure_runtime_dirs(cfg)
    dataset_path = str(getattr(getattr(cfg, "bm25_600_nemotron_bge_mpdocvqa", {}), "dataset_path", cfg.dataset.val_path))
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_600_nemotron_bge_mpdocvqa_predictions.jsonl"
    _predictions, metrics = run_bm25_600_nemotron_bge_mpdocvqa_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_600_nemotron_bge_mpdocvqa_metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Saved BM25-600 Nemotron+BGE MP-DocVQA predictions to %s", prediction_path)
    logger.info("Saved BM25-600 Nemotron+BGE MP-DocVQA metrics to %s", metrics_path)
    logger.info("BM25-600 Nemotron+BGE MP-DocVQA metrics: %s", metrics)


def eval_bm25_600_nemotron_bge_selective_mpdocvqa_pipeline(cfg: Any) -> None:
    """Run BM25@600 + Nemotron/BGE with visual-confidence selective fusion on MP-DocVQA."""
    logger = _build_logger(cfg, "eval_bm25_600_nemotron_bge_selective_mpdocvqa")
    _ensure_runtime_dirs(cfg)
    logger.info("[Selective Fusion] Using hard-gating branch before existing RRF fusion")
    dataset_path = str(getattr(getattr(cfg, "bm25_600_nemotron_bge_selective_mpdocvqa", {}), "dataset_path", cfg.dataset.val_path))
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_600_nemotron_bge_selective_mpdocvqa_predictions.jsonl"
    _predictions, metrics = run_bm25_600_nemotron_bge_selective_mpdocvqa_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_600_nemotron_bge_selective_mpdocvqa_metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Saved BM25-600 Nemotron+BGE selective MP-DocVQA predictions to %s", prediction_path)
    logger.info("Saved BM25-600 Nemotron+BGE selective MP-DocVQA metrics to %s", metrics_path)
    logger.info("BM25-600 Nemotron+BGE selective MP-DocVQA metrics: %s", metrics)


def eval_bm25_600_nemotron_bge_topk_rerank_mpdocvqa_pipeline(cfg: Any) -> None:
    """Run BM25@600 + Nemotron visual Top-K + BGE OCR rerank on MP-DocVQA."""
    logger = _build_logger(cfg, "eval_bm25_600_nemotron_bge_topk_rerank_mpdocvqa")
    _ensure_runtime_dirs(cfg)
    logger.info("[TopK-Rerank] Using visual Top-K followed by existing BGE OCR reranker")
    dataset_path = str(getattr(getattr(cfg, "bm25_600_nemotron_bge_topk_rerank_mpdocvqa", {}), "dataset_path", cfg.dataset.val_path))
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_600_nemotron_bge_topk_rerank_mpdocvqa_predictions.jsonl"
    _predictions, metrics = run_bm25_600_nemotron_bge_topk_rerank_mpdocvqa_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_600_nemotron_bge_topk_rerank_mpdocvqa_metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Saved BM25-600 Nemotron+BGE TopK-Rerank MP-DocVQA predictions to %s", prediction_path)
    logger.info("Saved BM25-600 Nemotron+BGE TopK-Rerank MP-DocVQA metrics to %s", metrics_path)
    logger.info("BM25-600 Nemotron+BGE TopK-Rerank MP-DocVQA metrics: %s", metrics)


def eval_bm25_600_nemotron_jina_mpdocvqa_pipeline(cfg: Any) -> None:
    """Run BM25@600 + parallel Nemotron/Jina evaluation on MP-DocVQA."""
    logger = _build_logger(cfg, "eval_bm25_600_nemotron_jina_mpdocvqa")
    _ensure_runtime_dirs(cfg)
    dataset_path = str(getattr(getattr(cfg, "bm25_600_nemotron_jina_mpdocvqa", {}), "dataset_path", cfg.dataset.val_path))
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_600_nemotron_jina_mpdocvqa_predictions.jsonl"
    _predictions, metrics = run_bm25_600_nemotron_jina_mpdocvqa_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_600_nemotron_jina_mpdocvqa_metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Saved BM25-600 Nemotron+Jina MP-DocVQA predictions to %s", prediction_path)
    logger.info("Saved BM25-600 Nemotron+Jina MP-DocVQA metrics to %s", metrics_path)
    logger.info("BM25-600 Nemotron+Jina MP-DocVQA metrics: %s", metrics)


def eval_bm25_600_nemotron_jina_chunk_mpdocvqa_pipeline(cfg: Any) -> None:
    """Run BM25@600 + Nemotron + Jina page retrieval + chunk rerank on MP-DocVQA."""
    logger = _build_logger(cfg, "eval_bm25_600_nemotron_jina_chunk_mpdocvqa")
    _ensure_runtime_dirs(cfg)
    logger.info("[Pipeline] Using OCR CHUNK rerank branch")
    dataset_path = str(getattr(getattr(cfg, "bm25_600_nemotron_jina_chunk_mpdocvqa", {}), "dataset_path", cfg.dataset.val_path))
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_600_nemotron_jina_chunk_mpdocvqa_predictions.jsonl"
    _predictions, metrics = run_bm25_600_nemotron_jina_chunk_mpdocvqa_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_600_nemotron_jina_chunk_mpdocvqa_metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Saved BM25-600 Nemotron+Jina-Chunk MP-DocVQA predictions to %s", prediction_path)
    logger.info("Saved BM25-600 Nemotron+Jina-Chunk MP-DocVQA metrics to %s", metrics_path)
    logger.info("BM25-600 Nemotron+Jina-Chunk MP-DocVQA metrics: %s", metrics)


def eval_bm25_600_nemotron_bge_dynamic_mpdocvqa_pipeline(cfg: Any) -> None:
    """Run BM25@600 + Nemotron + OCR with reused colqwen dynamic weighting on MP-DocVQA."""
    logger = _build_logger(cfg, "eval_bm25_600_nemotron_bge_dynamic_mpdocvqa")
    _ensure_runtime_dirs(cfg)
    dataset_path = str(getattr(getattr(cfg, "bm25_600_nemotron_bge_dynamic_mpdocvqa", {}), "dataset_path", cfg.dataset.val_path))
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_600_nemotron_bge_dynamic_mpdocvqa.jsonl"
    _predictions, metrics = run_bm25_600_nemotron_bge_dynamic_mpdocvqa_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_600_nemotron_bge_dynamic_mpdocvqa.json"
    save_json(metrics, metrics_path)
    logger.info("Saved BM25-600 Nemotron+BGE dynamic MP-DocVQA predictions to %s", prediction_path)
    logger.info("Saved BM25-600 Nemotron+BGE dynamic MP-DocVQA metrics to %s", metrics_path)
    logger.info("BM25-600 Nemotron+BGE dynamic MP-DocVQA metrics: %s", metrics)


def eval_bm25_600_queryaware_dynamic_mpdocvqa_pipeline(cfg: Any) -> None:
    """Run BM25@600 + query-aware routing with existing dynamic fusion fallback on MP-DocVQA."""
    logger = _build_logger(cfg, "eval_bm25_600_queryaware_dynamic_mpdocvqa")
    _ensure_runtime_dirs(cfg)
    logger.info("[Router] Using query-aware routing before existing dynamic fusion fallback")
    dataset_path = str(getattr(getattr(cfg, "bm25_600_queryaware_dynamic_mpdocvqa", {}), "dataset_path", cfg.dataset.val_path))
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_600_queryaware_dynamic_mpdocvqa_predictions.jsonl"
    _predictions, metrics = run_bm25_600_queryaware_dynamic_mpdocvqa_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_600_queryaware_dynamic_mpdocvqa_metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Saved BM25-600 query-aware dynamic MP-DocVQA predictions to %s", prediction_path)
    logger.info("Saved BM25-600 query-aware dynamic MP-DocVQA metrics to %s", metrics_path)
    logger.info("BM25-600 query-aware dynamic MP-DocVQA metrics: %s", metrics)


def eval_bm25_600_queryaware_visual_dominant_mpdocvqa_pipeline(cfg: Any) -> None:
    """Run BM25@600 + query-aware visual-dominant OCR correction on MP-DocVQA."""
    logger = _build_logger(cfg, "eval_bm25_600_queryaware_visual_dominant_mpdocvqa")
    _ensure_runtime_dirs(cfg)
    logger.info("[Router] Using query-aware visual-dominant OCR correction")
    dataset_path = str(getattr(getattr(cfg, "bm25_600_queryaware_visual_dominant_mpdocvqa", {}), "dataset_path", cfg.dataset.val_path))
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_600_queryaware_visual_dominant_mpdocvqa_predictions.jsonl"
    _predictions, metrics = run_bm25_600_queryaware_visual_dominant_mpdocvqa_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_600_queryaware_visual_dominant_mpdocvqa_metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Saved BM25-600 query-aware visual-dominant MP-DocVQA predictions to %s", prediction_path)
    logger.info("Saved BM25-600 query-aware visual-dominant MP-DocVQA metrics to %s", metrics_path)
    logger.info("BM25-600 query-aware visual-dominant MP-DocVQA metrics: %s", metrics)


def eval_bm25_600_selective_confidence_mpdocvqa_pipeline(cfg: Any) -> None:
    """Run BM25@600 + selective confidence routing with visual-dominant OCR correction."""
    logger = _build_logger(cfg, "eval_bm25_600_selective_confidence_mpdocvqa")
    _ensure_runtime_dirs(cfg)
    logger.info("[SelectiveConfidence] Using confidence-routed visual-dominant OCR correction")
    dataset_path = str(getattr(getattr(cfg, "bm25_600_selective_confidence_mpdocvqa", {}), "dataset_path", cfg.dataset.val_path))
    prediction_path = Path(cfg.paths.prediction_dir) / "bm25_600_selective_confidence_mpdocvqa_predictions.jsonl"
    _predictions, metrics = run_bm25_600_selective_confidence_mpdocvqa_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "bm25_600_selective_confidence_mpdocvqa_metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Saved BM25-600 selective confidence MP-DocVQA predictions to %s", prediction_path)
    logger.info("Saved BM25-600 selective confidence MP-DocVQA metrics to %s", metrics_path)
    logger.info("BM25-600 selective confidence MP-DocVQA metrics: %s", metrics)


def eval_visual_nemotron_mpdocvqa_pipeline(cfg: Any) -> None:
    """Run pure visual-only Nemotron retrieval on MP-DocVQA."""
    logger = _build_logger(cfg, "eval_visual_nemotron_mpdocvqa")
    _ensure_runtime_dirs(cfg)
    dataset_path = str(getattr(getattr(cfg, "visual_nemotron_mpdocvqa", {}), "dataset_path", cfg.dataset.val_path))
    prediction_path = Path(cfg.paths.prediction_dir) / "visual_nemotron_mpdocvqa_predictions.jsonl"
    _predictions, metrics = run_visual_nemotron_mpdocvqa_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        prediction_path=str(prediction_path),
    )
    metrics_path = Path(cfg.paths.metric_dir) / "visual_nemotron_mpdocvqa_metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Saved visual nemotron MP-DocVQA predictions to %s", prediction_path)
    logger.info("Saved visual nemotron MP-DocVQA metrics to %s", metrics_path)
    logger.info("Visual nemotron MP-DocVQA metrics: %s", metrics)


def debug_visual_nemotron_energy_sanity_pipeline(cfg: Any) -> None:
    """Run a tiny deterministic corpus-id sanity check for local Nemotron visual retrieval."""
    logger = _build_logger(cfg, "debug_visual_nemotron_energy_sanity")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_visual_nemotron_energy_sanity_check(cfg=cfg, num_samples=3, num_negatives=9)
    prediction_path = Path(cfg.paths.prediction_dir) / "visual_nemotron_energy_sanity_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "visual_nemotron_energy_sanity_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved visual nemotron sanity predictions to %s", prediction_path)
    logger.info("Saved visual nemotron sanity metrics to %s", metrics_path)
    logger.info("Visual nemotron sanity metrics: %s", metrics)


def eval_adaptive_fusion_visual_nemotron_ocr_energy_val_pipeline(cfg: Any) -> None:
    """Evaluate local Nemotron visual backbone + existing OCR route + adaptive fusion on ViDoRe Energy."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_visual_nemotron_ocr_energy_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_visual_nemotron_ocr_energy_on_dataset(
        cfg=cfg,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_visual_nemotron_ocr_energy_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_visual_nemotron_ocr_energy_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive_fusion_visual_nemotron_ocr_energy val predictions to %s", prediction_path)
    logger.info("Saved adaptive_fusion_visual_nemotron_ocr_energy val metrics to %s", metrics_path)
    logger.info("Adaptive fusion visual_nemotron_ocr_energy metrics: %s", metrics)


def train_adaptive_fusion_visual_nemotron_ocr_energy_pipeline(cfg: Any) -> None:
    """Train only the fusion MLP for local Nemotron visual backbone + existing OCR route."""
    logger = _build_logger(cfg, "train_adaptive_fusion_visual_nemotron_ocr_energy")
    _ensure_runtime_dirs(cfg)
    metrics = train_adaptive_fusion_visual_nemotron_ocr_energy(cfg)
    logger.info("adaptive_fusion_visual_nemotron_ocr_energy training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def eval_ocr_page_coarse_chunk_pipeline(cfg: Any) -> None:
    """Run OCR page-level coarse routing + chunk rerank on val."""
    logger = _build_logger(cfg, "eval_ocr_page_coarse_chunk_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_ocr_page_coarse_chunk_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_page_coarse_chunk_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_page_coarse_chunk_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved OCR page coarse chunk val predictions to %s", prediction_path)
    logger.info("Saved OCR page coarse chunk val metrics to %s", metrics_path)
    logger.info("OCR page coarse chunk val metrics: %s", metrics)


def eval_ocr_page_bm25_bge_rerank_pipeline(cfg: Any) -> None:
    """Run OCR page-level BM25 -> BGE-M3 -> reranker on val."""
    logger = _build_logger(cfg, "eval_ocr_page_bm25_bge_rerank_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_ocr_page_bm25_bge_rerank_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "ocr_page_bm25_bge_rerank_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "ocr_page_bm25_bge_rerank_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved OCR page BM25 BGE rerank val predictions to %s", prediction_path)
    logger.info("Saved OCR page BM25 BGE rerank val metrics to %s", metrics_path)
    logger.info("OCR page BM25 BGE rerank val metrics: %s", metrics)


def infer_visual_test_pipeline(cfg: Any) -> None:
    """Run visual-only retrieval on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_visual_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_visual_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "visual_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved visual test predictions to %s", prediction_path)
    if metrics:
        logger.info("Visual test metrics: %s", metrics)


def infer_visual_colqwen_test_pipeline(cfg: Any) -> None:
    """Run ColQwen visual-only retrieval on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_visual_colqwen_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_visual_colqwen_retrieval_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "visual_colqwen_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved visual ColQwen test predictions to %s", prediction_path)
    if metrics:
        logger.info("Visual ColQwen test metrics: %s", metrics)


def eval_fixed_fusion_pipeline(cfg: Any) -> None:
    """Run fixed-fusion retrieval on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_fixed_fusion_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_fixed_fusion_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        alpha=float(cfg.fusion.alpha),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "fixed_fusion_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "fixed_fusion_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved fixed fusion val predictions to %s", prediction_path)
    logger.info("Saved fixed fusion val metrics to %s", metrics_path)
    logger.info("Fixed fusion val metrics: %s", metrics)


def infer_fixed_fusion_test_pipeline(cfg: Any) -> None:
    """Run fixed-fusion retrieval on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_fixed_fusion_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_fixed_fusion_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        alpha=float(cfg.fusion.alpha),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "fixed_fusion_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved fixed fusion test predictions to %s", prediction_path)
    if metrics:
        logger.info("Fixed fusion test metrics: %s", metrics)


def eval_rrf_pipeline(cfg: Any) -> None:
    """Run RRF retrieval on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_rrf_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_rrf_fusion_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        rrf_k=int(cfg.fusion.rrf_k),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "rrf_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "rrf_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved RRF val predictions to %s", prediction_path)
    logger.info("Saved RRF val metrics to %s", metrics_path)
    logger.info("RRF val metrics: %s", metrics)


def infer_rrf_test_pipeline(cfg: Any) -> None:
    """Run RRF retrieval on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_rrf_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_rrf_fusion_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        rrf_k=int(cfg.fusion.rrf_k),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "rrf_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved RRF test predictions to %s", prediction_path)
    if metrics:
        logger.info("RRF test metrics: %s", metrics)


def train_adaptive_fusion_pipeline(cfg: Any) -> None:
    """Train the adaptive fusion reranker and save checkpoints/metrics."""
    logger = _build_logger(cfg, "train_adaptive_fusion")
    _ensure_runtime_dirs(cfg)
    metrics = train_fusion(cfg)
    logger.info("Adaptive fusion training completed: %s", metrics)


def eval_adaptive_fusion_pipeline(cfg: Any) -> None:
    """Run adaptive fusion retrieval on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive fusion val predictions to %s", prediction_path)
    logger.info("Saved adaptive fusion val metrics to %s", metrics_path)
    logger.info("Adaptive fusion val metrics: %s", metrics)


def infer_adaptive_fusion_test_pipeline(cfg: Any) -> None:
    """Run adaptive fusion retrieval on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_adaptive_fusion_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved adaptive fusion test predictions to %s", prediction_path)
    if metrics:
        logger.info("Adaptive fusion test metrics: %s", metrics)


def train_adaptive_fusion_v2_pipeline(cfg: Any) -> None:
    """Train the enhanced adaptive fusion reranker and save checkpoints/metrics."""
    logger = _build_logger(cfg, "train_adaptive_fusion_v2")
    _ensure_runtime_dirs(cfg)
    metrics = train_fusion_v2(cfg)
    logger.info("Adaptive fusion v2 training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def eval_adaptive_fusion_v2_pipeline(cfg: Any) -> None:
    """Run enhanced adaptive fusion retrieval on the validation split and save outputs."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_v2_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_v2_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_v2_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_v2_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive fusion v2 val predictions to %s", prediction_path)
    logger.info("Saved adaptive fusion v2 val metrics to %s", metrics_path)
    logger.info("Adaptive fusion v2 val metrics: %s", metrics)


def infer_adaptive_fusion_v2_test_pipeline(cfg: Any) -> None:
    """Run enhanced adaptive fusion retrieval on the test split and save predictions."""
    logger = _build_logger(cfg, "infer_adaptive_fusion_v2_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_v2_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_v2_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved adaptive fusion v2 test predictions to %s", prediction_path)
    if metrics:
        logger.info("Adaptive fusion v2 test metrics: %s", metrics)


def _train_adaptive_ablation_pipeline(cfg: Any, variant: str) -> None:
    """Train one rollback-style adaptive-fusion ablation variant."""
    logger = _build_logger(cfg, f"train_adaptive_fusion_{variant}")
    _ensure_runtime_dirs(cfg)
    metrics = train_fusion_ablation(cfg, variant=variant)
    logger.info("%s training completed. checkpoint=%s", variant, metrics.get("checkpoint_path", ""))


def _resolve_best_adaptive_variant(cfg: Any) -> str:
    """Resolve the configured current-best adaptive fusion variant into the internal ablation slug."""
    variant_name = str(cfg.fusion.current_best_variant)
    return variant_name.removeprefix("adaptive_fusion_")


def _eval_adaptive_ablation_pipeline(cfg: Any, variant: str) -> None:
    """Evaluate one rollback-style adaptive-fusion ablation variant on validation split."""
    logger = _build_logger(cfg, f"eval_adaptive_fusion_{variant}_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_ablation_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        variant=variant,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / f"adaptive_fusion_{variant}_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / f"adaptive_fusion_{variant}_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved %s val predictions to %s", variant, prediction_path)
    logger.info("Saved %s val metrics to %s", variant, metrics_path)


def _infer_adaptive_ablation_test_pipeline(cfg: Any, variant: str) -> None:
    """Run one rollback-style adaptive-fusion ablation variant on test split."""
    logger = _build_logger(cfg, f"infer_adaptive_fusion_{variant}_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_ablation_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        variant=variant,
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / f"adaptive_fusion_{variant}_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved %s test predictions to %s", variant, prediction_path)
    if metrics:
        logger.info("%s test metrics: %s", variant, metrics)


def train_adaptive_fusion_ablate_q_pipeline(cfg: Any) -> None:
    """Train the question-feature-only ablation variant."""
    _train_adaptive_ablation_pipeline(cfg, variant="ablate_q")


def eval_adaptive_fusion_ablate_q_pipeline(cfg: Any) -> None:
    """Evaluate the question-feature-only ablation variant on val."""
    _eval_adaptive_ablation_pipeline(cfg, variant="ablate_q")


def infer_adaptive_fusion_ablate_q_test_pipeline(cfg: Any) -> None:
    """Run the question-feature-only ablation variant on test."""
    _infer_adaptive_ablation_test_pipeline(cfg, variant="ablate_q")


def train_adaptive_fusion_ablate_ocrq_pipeline(cfg: Any) -> None:
    """Train the OCR-quality-only ablation variant."""
    _train_adaptive_ablation_pipeline(cfg, variant="ablate_ocrq")


def eval_adaptive_fusion_ablate_ocrq_pipeline(cfg: Any) -> None:
    """Evaluate the OCR-quality-only ablation variant on val."""
    _eval_adaptive_ablation_pipeline(cfg, variant="ablate_ocrq")


def infer_adaptive_fusion_ablate_ocrq_test_pipeline(cfg: Any) -> None:
    """Run the OCR-quality-only ablation variant on test."""
    _infer_adaptive_ablation_test_pipeline(cfg, variant="ablate_ocrq")


def train_adaptive_fusion_ablate_lex_pipeline(cfg: Any) -> None:
    """Train the lexical-match-only ablation variant."""
    _train_adaptive_ablation_pipeline(cfg, variant="ablate_lex")


def eval_adaptive_fusion_ablate_lex_pipeline(cfg: Any) -> None:
    """Evaluate the lexical-match-only ablation variant on val."""
    _eval_adaptive_ablation_pipeline(cfg, variant="ablate_lex")


def infer_adaptive_fusion_ablate_lex_test_pipeline(cfg: Any) -> None:
    """Run the lexical-match-only ablation variant on test."""
    _infer_adaptive_ablation_test_pipeline(cfg, variant="ablate_lex")


def train_adaptive_fusion_ablate_mlp_pipeline(cfg: Any) -> None:
    """Train the MLP-only ablation variant."""
    _train_adaptive_ablation_pipeline(cfg, variant="ablate_mlp")


def eval_adaptive_fusion_ablate_mlp_pipeline(cfg: Any) -> None:
    """Evaluate the MLP-only ablation variant on val."""
    _eval_adaptive_ablation_pipeline(cfg, variant="ablate_mlp")


def infer_adaptive_fusion_ablate_mlp_test_pipeline(cfg: Any) -> None:
    """Run the MLP-only ablation variant on test."""
    _infer_adaptive_ablation_test_pipeline(cfg, variant="ablate_mlp")


def train_adaptive_fusion_ablate_mlp_q_pipeline(cfg: Any) -> None:
    """Train the MLP + question-aware combination variant."""
    _train_adaptive_ablation_pipeline(cfg, variant="ablate_mlp_q")


def eval_adaptive_fusion_ablate_mlp_q_pipeline(cfg: Any) -> None:
    """Evaluate the MLP + question-aware combination variant on val."""
    _eval_adaptive_ablation_pipeline(cfg, variant="ablate_mlp_q")


def infer_adaptive_fusion_ablate_mlp_q_test_pipeline(cfg: Any) -> None:
    """Run the MLP + question-aware combination variant on test."""
    _infer_adaptive_ablation_test_pipeline(cfg, variant="ablate_mlp_q")


def train_adaptive_fusion_ablate_mlp_ocrq_pipeline(cfg: Any) -> None:
    """Train the MLP + OCR-quality combination variant."""
    _train_adaptive_ablation_pipeline(cfg, variant="ablate_mlp_ocrq")


def eval_adaptive_fusion_ablate_mlp_ocrq_pipeline(cfg: Any) -> None:
    """Evaluate the MLP + OCR-quality combination variant on val."""
    _eval_adaptive_ablation_pipeline(cfg, variant="ablate_mlp_ocrq")


def infer_adaptive_fusion_ablate_mlp_ocrq_test_pipeline(cfg: Any) -> None:
    """Run the MLP + OCR-quality combination variant on test."""
    _infer_adaptive_ablation_test_pipeline(cfg, variant="ablate_mlp_ocrq")


def train_adaptive_fusion_ablate_mlp_lex_pipeline(cfg: Any) -> None:
    """Train the MLP + lexical-overlap combination variant."""
    _train_adaptive_ablation_pipeline(cfg, variant="ablate_mlp_lex")


def eval_adaptive_fusion_ablate_mlp_lex_pipeline(cfg: Any) -> None:
    """Evaluate the MLP + lexical-overlap combination variant on val."""
    _eval_adaptive_ablation_pipeline(cfg, variant="ablate_mlp_lex")


def infer_adaptive_fusion_ablate_mlp_lex_test_pipeline(cfg: Any) -> None:
    """Run the MLP + lexical-overlap combination variant on test."""
    _infer_adaptive_ablation_test_pipeline(cfg, variant="ablate_mlp_lex")


def train_adaptive_fusion_best_pipeline(cfg: Any) -> None:
    """Train the configured current-best adaptive fusion variant."""
    _train_adaptive_ablation_pipeline(cfg, variant=_resolve_best_adaptive_variant(cfg))


def eval_adaptive_fusion_best_val_pipeline(cfg: Any) -> None:
    """Evaluate the configured current-best adaptive fusion variant on val."""
    _eval_adaptive_ablation_pipeline(cfg, variant=_resolve_best_adaptive_variant(cfg))


def infer_adaptive_fusion_best_test_pipeline(cfg: Any) -> None:
    """Run the configured current-best adaptive fusion variant on test."""
    _infer_adaptive_ablation_test_pipeline(cfg, variant=_resolve_best_adaptive_variant(cfg))


def train_adaptive_fusion_mlp_ocrq_bge_pipeline(cfg: Any) -> None:
    """Train the current-best fusion structure with the upgraded OCR BGE route."""
    logger = _build_logger(cfg, "train_adaptive_fusion_mlp_ocrq_bge")
    _ensure_runtime_dirs(cfg)
    metrics = train_fusion_mlp_ocrq_bge(cfg)
    logger.info("adaptive_fusion_mlp_ocrq_bge training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def eval_adaptive_fusion_mlp_ocrq_bge_pipeline(cfg: Any) -> None:
    """Evaluate the BGE-backed mlp_ocrq fusion variant on val."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_mlp_ocrq_bge_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_mlp_ocrq_bge_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_mlp_ocrq_bge_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_mlp_ocrq_bge_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_bge val predictions to %s", prediction_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_bge val metrics to %s", metrics_path)


def infer_adaptive_fusion_mlp_ocrq_bge_test_pipeline(cfg: Any) -> None:
    """Run the BGE-backed mlp_ocrq fusion variant on test."""
    logger = _build_logger(cfg, "infer_adaptive_fusion_mlp_ocrq_bge_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_mlp_ocrq_bge_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_mlp_ocrq_bge_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_bge test predictions to %s", prediction_path)
    if metrics:
        logger.info("adaptive_fusion_mlp_ocrq_bge test metrics: %s", metrics)


def train_adaptive_fusion_mlp_ocrq_chunk_pipeline(cfg: Any) -> None:
    """Train the mlp_ocrq fusion variant with chunk-level OCR rerank."""
    logger = _build_logger(cfg, "train_adaptive_fusion_mlp_ocrq_chunk")
    _ensure_runtime_dirs(cfg)
    metrics = train_fusion_mlp_ocrq_chunk(cfg)
    logger.info("adaptive_fusion_mlp_ocrq_chunk training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def eval_adaptive_fusion_mlp_ocrq_chunk_pipeline(cfg: Any) -> None:
    """Evaluate the chunk-backed mlp_ocrq fusion variant on val."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_mlp_ocrq_chunk_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_mlp_ocrq_chunk_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_mlp_ocrq_chunk_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_mlp_ocrq_chunk_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_chunk val predictions to %s", prediction_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_chunk val metrics to %s", metrics_path)


def infer_adaptive_fusion_mlp_ocrq_chunk_test_pipeline(cfg: Any) -> None:
    """Run the chunk-backed mlp_ocrq fusion variant on test."""
    logger = _build_logger(cfg, "infer_adaptive_fusion_mlp_ocrq_chunk_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_mlp_ocrq_chunk_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_mlp_ocrq_chunk_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_chunk test predictions to %s", prediction_path)
    if metrics:
        logger.info("adaptive_fusion_mlp_ocrq_chunk test metrics: %s", metrics)


def train_adaptive_fusion_visual_colqwen_ocr_chunk_pipeline(cfg: Any) -> None:
    """Train the clean visual_colqwen + OCR chunk fusion baseline."""
    logger = _build_logger(cfg, "train_adaptive_fusion_visual_colqwen_ocr_chunk")
    _ensure_runtime_dirs(cfg)
    metrics = train_fusion_visual_colqwen_ocr_chunk(cfg)
    logger.info("adaptive_fusion_visual_colqwen_ocr_chunk training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def eval_adaptive_fusion_visual_colqwen_ocr_chunk_val_pipeline(cfg: Any) -> None:
    """Evaluate the visual_colqwen + OCR chunk fusion baseline on val."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_visual_colqwen_ocr_chunk_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_visual_colqwen_ocr_chunk_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_visual_colqwen_ocr_chunk_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_visual_colqwen_ocr_chunk_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive_fusion_visual_colqwen_ocr_chunk val predictions to %s", prediction_path)
    logger.info("Saved adaptive_fusion_visual_colqwen_ocr_chunk val metrics to %s", metrics_path)


def infer_adaptive_fusion_visual_colqwen_ocr_chunk_test_pipeline(cfg: Any) -> None:
    """Run the visual_colqwen + OCR chunk fusion baseline on test."""
    logger = _build_logger(cfg, "infer_adaptive_fusion_visual_colqwen_ocr_chunk_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_visual_colqwen_ocr_chunk_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_visual_colqwen_ocr_chunk_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved adaptive_fusion_visual_colqwen_ocr_chunk test predictions to %s", prediction_path)
    if metrics:
        logger.info("adaptive_fusion_visual_colqwen_ocr_chunk test metrics: %s", metrics)


def train_adaptive_fusion_dynamic_rules_pipeline(cfg: Any) -> None:
    """Train the new rule-based dynamic OCR/visual weighting family."""
    logger = _build_logger(cfg, "train_adaptive_fusion_dynamic_rules")
    _ensure_runtime_dirs(cfg)
    metrics = train_adaptive_fusion_dynamic_rules(cfg)
    logger.info("adaptive_fusion_dynamic_rules training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def eval_adaptive_fusion_dynamic_rules_val_pipeline(cfg: Any) -> None:
    """Evaluate the rule-based dynamic OCR/visual weighting family on val."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_dynamic_rules_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_dynamic_rules_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_dynamic_rules_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_dynamic_rules_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive_fusion_dynamic_rules val predictions to %s", prediction_path)
    logger.info("Saved adaptive_fusion_dynamic_rules val metrics to %s", metrics_path)


def train_adaptive_fusion_learned_gating_pipeline(cfg: Any) -> None:
    """Train the learned global-gating fusion family."""
    logger = _build_logger(cfg, "train_adaptive_fusion_learned_gating")
    _ensure_runtime_dirs(cfg)
    metrics = train_adaptive_fusion_learned_gating(cfg)
    logger.info("adaptive_fusion_learned_gating training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def eval_adaptive_fusion_learned_gating_val_pipeline(cfg: Any) -> None:
    """Evaluate the learned global-gating fusion family on val."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_learned_gating_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_learned_gating_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_learned_gating_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_learned_gating_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive_fusion_learned_gating val predictions to %s", prediction_path)
    logger.info("Saved adaptive_fusion_learned_gating val metrics to %s", metrics_path)


def train_adaptive_fusion_gating_calibrated_pipeline(cfg: Any) -> None:
    """Train the calibrated gating family with optional pairwise hard-negative loss."""
    logger = _build_logger(cfg, "train_adaptive_fusion_gating_calibrated")
    _ensure_runtime_dirs(cfg)
    metrics = train_adaptive_fusion_gating_calibrated(cfg)
    logger.info("adaptive_fusion_gating_calibrated training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def eval_adaptive_fusion_gating_calibrated_val_pipeline(cfg: Any) -> None:
    """Evaluate the calibrated gating family on val."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_gating_calibrated_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_gating_calibrated_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    calibration_name = str(getattr(getattr(cfg, "dynamic_fusion", {}), "calibration_option", "raw"))
    loss_type = str(getattr(getattr(cfg, "dynamic_fusion", {}), "loss_type", "pointwise_bce"))
    loss_suffix = "margin" if loss_type == "pairwise_margin" else "pointwise"
    stage_suffix = f"{calibration_name}_{loss_suffix}"
    prediction_path = Path(cfg.paths.prediction_dir) / f"adaptive_fusion_gating_calibrated_{stage_suffix}_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / f"adaptive_fusion_gating_calibrated_{stage_suffix}_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive_fusion_gating_calibrated val predictions to %s", prediction_path)
    logger.info("Saved adaptive_fusion_gating_calibrated val metrics to %s", metrics_path)


def train_adaptive_fusion_mlp_ocrq_chunkplus_pipeline(cfg: Any) -> None:
    """Train the chunkplus fusion variant with chunk-aware/question-aware features."""
    logger = _build_logger(cfg, "train_adaptive_fusion_mlp_ocrq_chunkplus")
    _ensure_runtime_dirs(cfg)
    metrics = train_fusion_mlp_ocrq_chunkplus(cfg)
    logger.info("adaptive_fusion_mlp_ocrq_chunkplus training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def eval_adaptive_fusion_mlp_ocrq_chunkplus_pipeline(cfg: Any) -> None:
    """Evaluate the chunkplus fusion variant on val."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_mlp_ocrq_chunkplus_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_mlp_ocrq_chunkplus_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_mlp_ocrq_chunkplus_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_mlp_ocrq_chunkplus_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_chunkplus val predictions to %s", prediction_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_chunkplus val metrics to %s", metrics_path)


def infer_adaptive_fusion_mlp_ocrq_chunkplus_test_pipeline(cfg: Any) -> None:
    """Run the chunkplus fusion variant on test."""
    logger = _build_logger(cfg, "infer_adaptive_fusion_mlp_ocrq_chunkplus_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_mlp_ocrq_chunkplus_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_mlp_ocrq_chunkplus_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_chunkplus test predictions to %s", prediction_path)
    if metrics:
        logger.info("adaptive_fusion_mlp_ocrq_chunkplus test metrics: %s", metrics)


def train_adaptive_fusion_mlp_ocrq_hybrid_pipeline(cfg: Any) -> None:
    """Train the mlp_ocrq fusion variant with hybrid OCR retrieval."""
    logger = _build_logger(cfg, "train_adaptive_fusion_mlp_ocrq_hybrid")
    _ensure_runtime_dirs(cfg)
    metrics = train_fusion_mlp_ocrq_hybrid(cfg)
    logger.info("adaptive_fusion_mlp_ocrq_hybrid training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def train_adaptive_fusion_mlp_ocrq_nvchunk_pipeline(cfg: Any) -> None:
    """Train the mlp_ocrq fusion variant with NV chunk OCR retrieval."""
    logger = _build_logger(cfg, "train_adaptive_fusion_mlp_ocrq_nvchunk")
    _ensure_runtime_dirs(cfg)
    metrics = train_fusion_mlp_ocrq_nvchunk(cfg)
    logger.info("adaptive_fusion_mlp_ocrq_nvchunk training completed. checkpoint=%s", metrics.get("checkpoint_path", ""))


def eval_adaptive_fusion_mlp_ocrq_hybrid_pipeline(cfg: Any) -> None:
    """Evaluate the hybrid-backed mlp_ocrq fusion variant on val."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_mlp_ocrq_hybrid_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_mlp_ocrq_hybrid_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_mlp_ocrq_hybrid_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_mlp_ocrq_hybrid_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_hybrid val predictions to %s", prediction_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_hybrid val metrics to %s", metrics_path)


def infer_adaptive_fusion_mlp_ocrq_hybrid_test_pipeline(cfg: Any) -> None:
    """Run the hybrid-backed mlp_ocrq fusion variant on test."""
    logger = _build_logger(cfg, "infer_adaptive_fusion_mlp_ocrq_hybrid_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_mlp_ocrq_hybrid_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_mlp_ocrq_hybrid_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_hybrid test predictions to %s", prediction_path)
    if metrics:
        logger.info("adaptive_fusion_mlp_ocrq_hybrid test metrics: %s", metrics)


def eval_adaptive_fusion_mlp_ocrq_nvchunk_pipeline(cfg: Any) -> None:
    """Evaluate the NV-chunk-backed mlp_ocrq fusion variant on val."""
    logger = _build_logger(cfg, "eval_adaptive_fusion_mlp_ocrq_nvchunk_val")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_mlp_ocrq_nvchunk_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.val_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_mlp_ocrq_nvchunk_val_predictions.jsonl"
    metrics_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_mlp_ocrq_nvchunk_val_metrics.json"
    save_jsonl(predictions, prediction_path)
    save_json(metrics, metrics_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_nvchunk val predictions to %s", prediction_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_nvchunk val metrics to %s", metrics_path)


def infer_adaptive_fusion_mlp_ocrq_nvchunk_test_pipeline(cfg: Any) -> None:
    """Run the NV-chunk-backed mlp_ocrq fusion variant on test."""
    logger = _build_logger(cfg, "infer_adaptive_fusion_mlp_ocrq_nvchunk_test")
    _ensure_runtime_dirs(cfg)
    predictions, metrics = run_adaptive_fusion_mlp_ocrq_nvchunk_on_dataset(
        cfg=cfg,
        dataset_path=str(cfg.dataset.test_path),
        topk=int(cfg.retrieval.topk),
        k_values=list(cfg.retrieval.k_values),
        checkpoint_path=str(cfg.experiment.checkpoint_path or ""),
    )
    prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_mlp_ocrq_nvchunk_test_predictions.jsonl"
    save_jsonl(predictions, prediction_path)
    logger.info("Saved adaptive_fusion_mlp_ocrq_nvchunk test predictions to %s", prediction_path)
    if metrics:
        logger.info("adaptive_fusion_mlp_ocrq_nvchunk test metrics: %s", metrics)


def train_fusion_pipeline(cfg: Any) -> None:
    """Train the adaptive fusion model."""
    logger = _build_logger(cfg, "train_fusion")
    _ensure_runtime_dirs(cfg)
    metrics = train_fusion(cfg)
    logger.info("Fusion training completed: %s", metrics)


def infer_docvqa_pipeline(cfg: Any) -> None:
    """Run end-to-end DocVQA inference and save predictions."""
    logger = _build_logger(cfg, "infer_docvqa")
    _ensure_runtime_dirs(cfg)
    dataset = _load_split_dataset(cfg, str(cfg.experiment.split))
    processed_samples = list(dataset)
    documents = collect_document_pages(cfg, processed_samples)
    visual_retriever = ColPaliRetriever(cfg)
    text_retriever = BM25Retriever(cfg)
    build_visual_index(visual_retriever, documents)
    build_text_index(text_retriever, documents)
    retriever_manager = RetrieverManager(cfg, visual_retriever, text_retriever)
    predictions = []
    case_rows = []
    for sample in processed_samples:
        bundle = retriever_manager.retrieve(sample)
        prediction = infer_docvqa_sample(cfg, sample, bundle, fusion_mode="adaptive_fusion", top_n=1)
        predictions.append(prediction)
        case_rows.append(
            {
                "question": sample["question"],
                "evidence_page_ids": [_normalize_evidence_page_id(sample["doc_id"], page) for page in sample.get("evidence_pages", [])],
                "visual_top_k": bundle["visual"].get("page_ids", []),
                "ocr_top_k": bundle["text"].get("page_ids", []),
                "fusion_top_k": prediction["retrieval"].get("page_ids", []),
                "final_answer": prediction["predicted_answer"],
            }
        )
    save_jsonl(predictions, Path(cfg.paths.prediction_dir) / f"{cfg.experiment.name}_{cfg.experiment.split}_docvqa_predictions.jsonl")
    save_json(export_case_study(case_rows), Path(cfg.paths.prediction_dir) / f"{cfg.experiment.name}_{cfg.experiment.split}_case_study.json")
    logger.info("DocVQA inference completed for %d samples.", len(predictions))


def eval_docvqa_pipeline(cfg: Any) -> None:
    """Evaluate end-to-end DocVQA predictions with EM, F1, and ANLS."""
    logger = _build_logger(cfg, "eval_docvqa")
    prediction_path = Path(cfg.paths.prediction_dir) / f"{cfg.experiment.name}_{cfg.experiment.split}_docvqa_predictions.jsonl"
    dataset = _load_split_dataset(cfg, str(cfg.experiment.split))
    predictions = []
    if prediction_path.exists():
        from src.utils.io_utils import load_jsonl

        predictions = load_jsonl(prediction_path)
    else:
        infer_docvqa_pipeline(cfg)
        from src.utils.io_utils import load_jsonl

        predictions = load_jsonl(prediction_path)
    answers_by_qid = {sample["qid"]: sample.get("answer", "") for sample in dataset}
    metrics = {
        "EM": sum(exact_match(item.get("predicted_answer", ""), answers_by_qid.get(item["qid"], "")) for item in predictions) / max(len(predictions), 1),
        "F1": sum(f1_score(item.get("predicted_answer", ""), answers_by_qid.get(item["qid"], "")) for item in predictions) / max(len(predictions), 1),
        "ANLS": sum(anls(item.get("predicted_answer", ""), answers_by_qid.get(item["qid"], "")) for item in predictions) / max(len(predictions), 1),
    }
    save_json(metrics, Path(cfg.paths.metric_dir) / f"{cfg.experiment.name}_{cfg.experiment.split}_docvqa_metrics.json")
    logger.info("DocVQA evaluation completed: %s", metrics)


def _ensure_runtime_dirs(cfg: Any) -> None:
    for key, value in cfg.paths.items():
        if key.endswith("_dir"):
            ensure_dir(value)


def _should_skip(cfg: Any, target_path: Path) -> bool:
    return bool(cfg.runtime.use_cache) and not bool(cfg.runtime.overwrite_cache) and target_path.exists()


def _should_skip_split(cfg: Any, processed_path: Path, report_path: Path) -> bool:
    """Check whether a prepared split can be safely skipped."""
    return (
        bool(cfg.runtime.use_cache)
        and not bool(cfg.runtime.overwrite_cache)
        and processed_path.exists()
        and report_path.exists()
    )


def _build_logger(cfg: Any, stage_name: str):
    log_file = Path(cfg.paths.log_dir) / f"{cfg.experiment.name}_{stage_name}.log"
    return get_logger(stage_name, log_file=log_file)


def _normalize_evidence_page_id(doc_id: str, evidence_page: Any) -> str:
    """Normalize evidence-page annotations into canonical page ids."""
    if isinstance(evidence_page, str) and "_page_" in evidence_page:
        return evidence_page
    if isinstance(evidence_page, str) and "_p" in evidence_page:
        return evidence_page
    return f"{doc_id}_p{int(evidence_page)}"


def _load_split_dataset(cfg: Any, split: str) -> DocVQADataset:
    dataset = DocVQADataset(cfg, split)
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No processed samples found for split '{split}'. Run prepare_data first or place files under {cfg.dataset.processed_files[split]}."
        )
    return dataset


def _ensure_sample_page_images(cfg: Any, sample: dict[str, Any], logger: Any) -> None:
    if sample.get("image_paths"):
        return
    pdf_path = Path(cfg.dataset.raw_pdf_dir) / f"{sample['doc_id']}.pdf"
    if not pdf_path.exists():
        logger.info("No PDF found for doc %s at %s", sample["doc_id"], pdf_path)
        return
    output_dir = Path(cfg.dataset.page_image_dir) / sample["doc_id"]
    render_result = render_pdf_to_images(pdf_path, output_dir)
    sample["image_paths"] = render_result["image_paths"]
    sample["page_ids"] = list(render_result["page_id_map"].keys())


def _ensure_sample_ocr(cfg: Any, sample: dict[str, Any], ocr_engine: OCREngine, logger: Any) -> None:
    ocr_path = Path(sample["ocr_results_path"])
    if _should_skip(cfg, ocr_path):
        return
    image_paths = sample.get("image_paths", [])
    if not image_paths:
        logger.info("No page images available for doc %s. OCR skipped.", sample["doc_id"])
        save_json([], ocr_path)
        return
    ocr_results = ocr_engine.run_ocr_on_document(image_paths)
    save_json(ocr_results, ocr_path)
