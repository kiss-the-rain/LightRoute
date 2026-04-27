"""Training entrypoints for lightweight adaptive fusion retrieval models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data.vidore_energy_loader import load_vidore_energy_dataset
from src.evaluation.retrieval_metrics import StreamingSetRetrievalMetrics, evaluate_retrieval
from src.inference.infer_retrieval import (
    add_ocr_bm25_metric_aliases,
    _accumulate_adaptive_coarse_metrics,
    _accumulate_ocr_page_pipeline_metrics,
    _build_adaptive_coarse_metric_accumulator,
    _build_ocr_page_pipeline_metric_accumulator,
    _retrieve_bm25_for_sample,
    _retrieve_ocr_bge_for_sample,
    _retrieve_ocr_bge_chunk_for_sample,
    _retrieve_ocr_with_bm25_bge_reranker_for_sample,
    _retrieve_ocr_with_page_coarse_for_sample,
    _retrieve_ocr_hybrid_for_sample,
    _retrieve_ocr_nv_chunk_for_sample,
    _retrieve_visual_nemotron_energy_for_sample,
    _retrieve_visual_for_sample,
    _retrieve_visual_with_adaptive_coarse_for_sample,
    _vidore_energy_sample_to_processed_sample,
    summarize_ocr_page_pipeline_metrics,
    summarize_adaptive_coarse_stats,
)
from src.models.adaptive_fusion import AdaptiveFusion, AdaptiveFusionV2
from src.models.gating_mlp import GateNet
from src.models.question_encoder import QuestionEncoder
from src.retrieval.colpali_retriever import ColPaliRetriever
from src.retrieval.colqwen_retriever import ColQwenRetriever
from src.retrieval.bm25_retriever import load_page_ocr_text
from src.retrieval.ocr_bge_chunk_retriever import OCRBGEChunkRetriever
from src.retrieval.ocr_bge_chunk_reranker import OCRBGEChunkReranker
from src.retrieval.ocr_bge_retriever import OCRBGERetriever
from src.retrieval.ocr_bge_reranker import OCRBGEReranker
from src.retrieval.ocr_chunker import OCRChunkBuilder
from src.retrieval.ocr_hybrid_retriever import OCRHybridRetriever
from src.retrieval.ocr_nv_retriever import OCRNVChunkRetriever
from src.retrieval.nemotron_visual_retriever import NemotronVisualRetriever
from src.retrieval.fusion_features import (
    build_dynamic_gating_feature_vector,
    build_candidate_features,
    build_candidate_features_ablate_lex,
    build_candidate_features_ablate_mlp_lex,
    build_candidate_features_ablate_mlp,
    build_candidate_features_ablate_mlp_ocrq,
    build_candidate_features_visual_colqwen_ocr_chunk,
    build_candidate_features_mlp_ocrq_chunkplus,
    build_candidate_features_ablate_mlp_q,
    build_candidate_features_ablate_ocrq,
    build_candidate_features_ablate_q,
    build_candidate_features_v2,
    refresh_candidate_feature_vectors,
)
from src.retrieval.dynamic_weighting import (
    apply_branch_reweighting,
    calibrate_route_scores,
    compute_rule_based_weights,
    derive_gate_targets,
    summarize_weight_debug,
)
from src.training.trainer import Trainer
from src.utils.io_utils import load_jsonl, load_pickle, save_json, save_jsonl, save_pickle
from src.utils.jsonl_writer import JsonlStreamWriter
from src.utils.logger import get_logger
from src.utils.memory_utils import log_ram, release_memory


def _cleanup_stale_visual_nemotron_tmp_dirs(cache_dir: Path, logger: Any) -> None:
    """Remove leftover ephemeral batch shard directories from prior interrupted runs."""
    if not cache_dir.exists():
        return
    cleaned = 0
    for child in cache_dir.iterdir():
        if not child.is_dir():
            continue
        if "_tmp_" not in child.name:
            continue
        shutil.rmtree(child, ignore_errors=True)
        cleaned += 1
    if cleaned > 0:
        logger.info("Removed %d stale visual_nemotron temporary batch directories from %s", cleaned, cache_dir)


def train_fusion(cfg: Any) -> dict[str, Any]:
    """Train the adaptive fusion scorer on train split and evaluate on val split."""
    return _train_fusion_variant(
        cfg=cfg,
        logger_name="train_adaptive_fusion",
        checkpoint_subdir="adaptive_fusion",
        prediction_filename="adaptive_fusion_val_predictions.jsonl",
        metric_filename="adaptive_fusion_val_metrics.json",
        train_metric_filename=f"{cfg.experiment.name}_train_adaptive_fusion_metrics.json",
        model_cls=AdaptiveFusion,
        feature_builder=build_candidate_features,
    )


def train_fusion_v2(cfg: Any) -> dict[str, Any]:
    """Train the enhanced adaptive fusion scorer on train split and evaluate on val split."""
    return _train_fusion_variant(
        cfg=cfg,
        logger_name="train_adaptive_fusion_v2",
        checkpoint_subdir="adaptive_fusion_v2",
        prediction_filename="adaptive_fusion_v2_val_predictions.jsonl",
        metric_filename="adaptive_fusion_v2_val_metrics.json",
        train_metric_filename=f"{cfg.experiment.name}_train_adaptive_fusion_v2_metrics.json",
        model_cls=AdaptiveFusionV2,
        feature_builder=build_candidate_features_v2,
    )


def train_fusion_ablation(cfg: Any, variant: str) -> dict[str, Any]:
    """Train one rollback-style adaptive fusion ablation variant."""
    variant_settings = {
        "ablate_q": {
            "logger_name": "train_adaptive_fusion_ablate_q",
            "checkpoint_subdir": "adaptive_fusion_ablate_q",
            "prediction_filename": "adaptive_fusion_ablate_q_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_ablate_q_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_ablate_q_metrics.json",
            "model_cls": AdaptiveFusion,
            "feature_builder": build_candidate_features_ablate_q,
        },
        "ablate_ocrq": {
            "logger_name": "train_adaptive_fusion_ablate_ocrq",
            "checkpoint_subdir": "adaptive_fusion_ablate_ocrq",
            "prediction_filename": "adaptive_fusion_ablate_ocrq_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_ablate_ocrq_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_ablate_ocrq_metrics.json",
            "model_cls": AdaptiveFusion,
            "feature_builder": build_candidate_features_ablate_ocrq,
        },
        "ablate_lex": {
            "logger_name": "train_adaptive_fusion_ablate_lex",
            "checkpoint_subdir": "adaptive_fusion_ablate_lex",
            "prediction_filename": "adaptive_fusion_ablate_lex_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_ablate_lex_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_ablate_lex_metrics.json",
            "model_cls": AdaptiveFusion,
            "feature_builder": build_candidate_features_ablate_lex,
        },
        "ablate_mlp": {
            "logger_name": "train_adaptive_fusion_ablate_mlp",
            "checkpoint_subdir": "adaptive_fusion_ablate_mlp",
            "prediction_filename": "adaptive_fusion_ablate_mlp_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_ablate_mlp_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_ablate_mlp_metrics.json",
            "model_cls": AdaptiveFusionV2,
            "feature_builder": build_candidate_features_ablate_mlp,
        },
        "ablate_mlp_q": {
            "logger_name": "train_adaptive_fusion_ablate_mlp_q",
            "checkpoint_subdir": "adaptive_fusion_ablate_mlp_q",
            "prediction_filename": "adaptive_fusion_ablate_mlp_q_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_ablate_mlp_q_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_ablate_mlp_q_metrics.json",
            "model_cls": AdaptiveFusionV2,
            "feature_builder": build_candidate_features_ablate_mlp_q,
        },
        "ablate_mlp_ocrq": {
            "logger_name": "train_adaptive_fusion_ablate_mlp_ocrq",
            "checkpoint_subdir": "adaptive_fusion_ablate_mlp_ocrq",
            "prediction_filename": "adaptive_fusion_ablate_mlp_ocrq_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_ablate_mlp_ocrq_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_ablate_mlp_ocrq_metrics.json",
            "model_cls": AdaptiveFusionV2,
            "feature_builder": build_candidate_features_ablate_mlp_ocrq,
        },
        "ablate_mlp_lex": {
            "logger_name": "train_adaptive_fusion_ablate_mlp_lex",
            "checkpoint_subdir": "adaptive_fusion_ablate_mlp_lex",
            "prediction_filename": "adaptive_fusion_ablate_mlp_lex_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_ablate_mlp_lex_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_ablate_mlp_lex_metrics.json",
            "model_cls": AdaptiveFusionV2,
            "feature_builder": build_candidate_features_ablate_mlp_lex,
        },
    }
    if variant not in variant_settings:
        raise ValueError(f"Unsupported adaptive-fusion ablation variant: {variant}")
    return _train_fusion_variant(cfg=cfg, **variant_settings[variant])


def train_fusion_mlp_ocrq_bge(cfg: Any) -> dict[str, Any]:
    """Train the current-best fusion structure using the upgraded OCR BGE route."""
    return train_fusion_mlp_ocrq_with_backend(cfg, ocr_backend="bge_rerank")


def train_fusion_mlp_ocrq_chunk(cfg: Any) -> dict[str, Any]:
    """Train the current-best fusion structure using chunk-level OCR rerank."""
    return train_fusion_mlp_ocrq_with_backend(cfg, ocr_backend="bge_chunk_rerank")


def train_fusion_visual_colqwen_ocr_chunk(cfg: Any) -> dict[str, Any]:
    """Train the stable mlp_ocrq chunk fusion with ColQwen as the visual backbone."""
    return train_fusion_mlp_ocrq_with_backend(
        cfg,
        ocr_backend="ocr_page_bm25_bge_rerank",
        visual_backend="colqwen",
        variant_name="visual_colqwen_ocr_chunk",
        feature_builder=build_candidate_features_visual_colqwen_ocr_chunk,
        use_adaptive_coarse_visual=True,
        use_ocr_page_coarse=True,
    )


def train_fusion_mlp_ocrq_hybrid(cfg: Any) -> dict[str, Any]:
    """Train the current-best fusion structure using the hybrid OCR route."""
    return train_fusion_mlp_ocrq_with_backend(cfg, ocr_backend="ocr_hybrid")


def train_fusion_mlp_ocrq_chunkplus(cfg: Any) -> dict[str, Any]:
    """Train the chunk-aware + question-aware fusion variant on chunk OCR retrieval."""
    return train_fusion_mlp_ocrq_with_backend(
        cfg,
        ocr_backend="bge_chunk_rerank",
        variant_name="mlp_ocrq_chunkplus",
        feature_builder=build_candidate_features_mlp_ocrq_chunkplus,
    )


def train_fusion_mlp_ocrq_nvchunk(cfg: Any) -> dict[str, Any]:
    """Train the current-best fusion structure using NV chunk OCR retrieval."""
    return train_fusion_mlp_ocrq_with_backend(cfg, ocr_backend="nv_chunk")


def train_fusion_mlp_ocrq_with_backend(
    cfg: Any,
    ocr_backend: str,
    visual_backend: str = "colpali",
    variant_name: str | None = None,
    feature_builder: Any = build_candidate_features_ablate_mlp_ocrq,
    use_adaptive_coarse_visual: bool = False,
    use_ocr_page_coarse: bool = False,
) -> dict[str, Any]:
    """Train the current-best fusion structure with a selectable OCR retrieval backend."""
    backend_settings = {
        "bge_rerank": {
            "logger_name": "train_adaptive_fusion_mlp_ocrq_bge",
            "checkpoint_subdir": "adaptive_fusion_mlp_ocrq_bge",
            "prediction_filename": "adaptive_fusion_mlp_ocrq_bge_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_mlp_ocrq_bge_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_mlp_ocrq_bge_metrics.json",
        },
        "bge_chunk_rerank": {
            "logger_name": "train_adaptive_fusion_mlp_ocrq_chunk",
            "checkpoint_subdir": "adaptive_fusion_mlp_ocrq_chunk",
            "prediction_filename": "adaptive_fusion_mlp_ocrq_chunk_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_mlp_ocrq_chunk_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_mlp_ocrq_chunk_metrics.json",
        },
        "ocr_hybrid": {
            "logger_name": "train_adaptive_fusion_mlp_ocrq_hybrid",
            "checkpoint_subdir": "adaptive_fusion_mlp_ocrq_hybrid",
            "prediction_filename": "adaptive_fusion_mlp_ocrq_hybrid_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_mlp_ocrq_hybrid_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_mlp_ocrq_hybrid_metrics.json",
        },
        "nv_chunk": {
            "logger_name": "train_adaptive_fusion_mlp_ocrq_nvchunk",
            "checkpoint_subdir": "adaptive_fusion_mlp_ocrq_nvchunk",
            "prediction_filename": "adaptive_fusion_mlp_ocrq_nvchunk_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_mlp_ocrq_nvchunk_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_mlp_ocrq_nvchunk_metrics.json",
        },
        "ocr_page_bm25_bge_rerank": {
            "logger_name": "train_adaptive_fusion_visual_colqwen_ocr_chunk",
            "checkpoint_subdir": "adaptive_fusion_visual_colqwen_ocr_chunk",
            "prediction_filename": "adaptive_fusion_visual_colqwen_ocr_chunk_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_visual_colqwen_ocr_chunk_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_visual_colqwen_ocr_chunk_metrics.json",
        },
    }
    variant_overrides = {
        "mlp_ocrq_chunkplus": {
            "logger_name": "train_adaptive_fusion_mlp_ocrq_chunkplus",
            "checkpoint_subdir": "adaptive_fusion_mlp_ocrq_chunkplus",
            "prediction_filename": "adaptive_fusion_mlp_ocrq_chunkplus_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_mlp_ocrq_chunkplus_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_mlp_ocrq_chunkplus_metrics.json",
        },
        "visual_colqwen_ocr_chunk": {
            "logger_name": "train_adaptive_fusion_visual_colqwen_ocr_chunk",
            "checkpoint_subdir": "adaptive_fusion_visual_colqwen_ocr_chunk",
            "prediction_filename": "adaptive_fusion_visual_colqwen_ocr_chunk_val_predictions.jsonl",
            "metric_filename": "adaptive_fusion_visual_colqwen_ocr_chunk_val_metrics.json",
            "train_metric_filename": f"{cfg.experiment.name}_train_adaptive_fusion_visual_colqwen_ocr_chunk_metrics.json",
        },
    }
    if variant_name is not None:
        if variant_name not in variant_overrides:
            raise ValueError(f"Unsupported mlp_ocrq variant override: {variant_name}")
        settings = variant_overrides[variant_name]
    else:
        settings = backend_settings.get(ocr_backend)
    if ocr_backend not in backend_settings:
        raise ValueError(f"Unsupported mlp_ocrq OCR backend: {ocr_backend}")
    return _train_fusion_variant(
        cfg=cfg,
        **settings,
        model_cls=AdaptiveFusionV2,
        feature_builder=feature_builder,
        ocr_backend=ocr_backend,
        visual_backend=visual_backend,
        use_adaptive_coarse_visual=use_adaptive_coarse_visual,
        use_ocr_page_coarse=use_ocr_page_coarse,
    )


def _train_fusion_variant(
    cfg: Any,
    logger_name: str,
    checkpoint_subdir: str,
    prediction_filename: str,
    metric_filename: str,
    train_metric_filename: str,
    model_cls: type[AdaptiveFusion] | type[AdaptiveFusionV2],
    feature_builder: Any,
    ocr_backend: str = "bm25",
    visual_backend: str = "colpali",
    use_adaptive_coarse_visual: bool = False,
    use_ocr_page_coarse: bool = False,
) -> dict[str, Any]:
    """Shared training pipeline for adaptive fusion variants."""
    if visual_backend == "old_visual":
        visual_backend = "colpali"
    logger = get_logger(logger_name, log_file=Path(cfg.paths.log_dir) / f"{cfg.experiment.name}_{checkpoint_subdir}.log")
    visual_router_cfg = getattr(cfg, "retrieval_router", None)
    ocr_router_cfg = getattr(cfg, "ocr_router", None)
    ocr_semantic_cfg = getattr(cfg, "ocr_semantic_retrieval", None)
    ocr_reranker_cfg = getattr(cfg, "ocr_reranker", None)
    logger.info(
        "visual_backend=%s ocr_backend=%s visual_adaptive_coarse=%s visual_bm25_coarse=%s "
        "visual_bypass_threshold=%s visual_coarse_topk=%s ocr_page_coarse=%s ocr_bm25_coarse=%s "
        "ocr_bypass_threshold=%s ocr_coarse_topk=%s ocr_semantic_topk=%s ocr_rerank_topk=%s",
        visual_backend,
        ocr_backend,
        bool(use_adaptive_coarse_visual and getattr(visual_router_cfg, "enable_adaptive_coarse", False)),
        bool(getattr(visual_router_cfg, "enable_bm25_coarse", True)),
        int(getattr(visual_router_cfg, "bypass_threshold", 0) or 0),
        int(getattr(visual_router_cfg, "coarse_topk", 0) or 0),
        bool(use_ocr_page_coarse and getattr(ocr_router_cfg, "enable_ocr_page_coarse", False)),
        bool(getattr(ocr_router_cfg, "enable_bm25_coarse", True)),
        int(getattr(ocr_router_cfg, "bypass_threshold", 0) or 0),
        int(getattr(ocr_router_cfg, "coarse_topk", 0) or 0),
        int(getattr(ocr_semantic_cfg, "semantic_topk", 0) or 0),
        int(getattr(ocr_reranker_cfg, "rerank_topk", 0) or 0),
    )
    logger.info("Loading processed train split from %s", cfg.dataset.processed_files["train"])
    train_dataset = load_jsonl(cfg.dataset.processed_files["train"]) if Path(cfg.dataset.processed_files["train"]).exists() else []
    logger.info("Loading processed val split from %s", cfg.dataset.processed_files["val"])
    val_dataset = load_jsonl(cfg.dataset.processed_files["val"]) if Path(cfg.dataset.processed_files["val"]).exists() else []

    max_train_samples = int(getattr(cfg.training, "max_train_samples", 0) or 0)
    max_val_samples = int(getattr(cfg.training, "max_val_samples", 0) or 0)
    if max_train_samples > 0:
        train_dataset = train_dataset[:max_train_samples]
        logger.info("Using train subset for smoke test: %d samples", len(train_dataset))
    if max_val_samples > 0:
        val_dataset = val_dataset[:max_val_samples]
        logger.info("Using val subset for smoke test: %d samples", len(val_dataset))

    if not train_dataset or not val_dataset:
        metrics = {"status": "skipped", "reason": "train/val processed datasets are missing or empty"}
        save_json(metrics, Path(cfg.paths.metric_dir) / train_metric_filename)
        return metrics

    checkpoint_dir = Path(cfg.paths.checkpoint_dir) / checkpoint_subdir
    cfg.training.checkpoint_dir = str(checkpoint_dir)
    question_encoder = QuestionEncoder()
    bm25_text_cache: dict[str, list[str]] = {}
    visual_coarse_text_cache: dict[str, list[str]] = {}
    bm25_retriever_cache: dict[str, Any] = {}
    coarse_bm25_retriever_cache: dict[str, Any] = {}
    ocr_coarse_text_cache: dict[str, list[str]] = {}
    ocr_coarse_retriever_cache: dict[str, Any] = {}
    ocr_bge_retriever = OCRBGERetriever(cfg) if ocr_backend == "bge_rerank" else None
    ocr_bge_reranker = OCRBGEReranker(cfg) if ocr_backend == "bge_rerank" else None
    ocr_page_retriever = OCRBGERetriever(cfg, config_attr="ocr_semantic_retrieval") if ocr_backend == "ocr_page_bm25_bge_rerank" else None
    ocr_page_reranker = (
        OCRBGEReranker(cfg, config_attr="ocr_reranker")
        if ocr_backend == "ocr_page_bm25_bge_rerank" and bool(getattr(ocr_reranker_cfg, "enable_bge_reranker", True))
        else None
    )
    ocr_bge_indexed_docs: set[str] = set()
    chunk_builder = (
        OCRChunkBuilder(
            chunk_size=int(cfg.ocr_chunk_retrieval.chunk_size),
            chunk_stride=int(cfg.ocr_chunk_retrieval.chunk_stride),
            page_text_variant=str(cfg.ocr_chunk_retrieval.page_text_variant),
        )
        if ocr_backend in {"bge_chunk_rerank", "ocr_hybrid", "nv_chunk"}
        else None
    )
    ocr_chunk_retriever = OCRBGEChunkRetriever(cfg) if ocr_backend == "bge_chunk_rerank" else None
    ocr_chunk_reranker = OCRBGEChunkReranker(cfg) if ocr_backend == "bge_chunk_rerank" else None
    ocr_chunk_indexed_docs: set[str] = set()
    ocr_chunk_cache: dict[str, list[dict[str, Any]]] = {}
    ocr_hybrid_retriever = OCRHybridRetriever(cfg) if ocr_backend == "ocr_hybrid" else None
    ocr_nv_chunk_retriever = OCRNVChunkRetriever(cfg) if ocr_backend == "nv_chunk" else None
    ocr_nv_chunk_indexed_docs: set[str] = set()
    visual_retriever = _create_visual_retriever(cfg, visual_backend=visual_backend)
    visual_indexed_docs: set[str] = set()
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] = {}

    train_batches, train_batch_stats = _build_training_batches(
        cfg,
        train_dataset,
        question_encoder,
        logger,
        bm25_text_cache,
        bm25_retriever_cache,
        visual_retriever,
        visual_indexed_docs,
        ocr_quality_cache,
        feature_builder=feature_builder,
        ocr_backend=ocr_backend,
        visual_backend=visual_backend,
        ocr_bge_retriever=ocr_bge_retriever,
        ocr_bge_reranker=ocr_bge_reranker,
        ocr_page_retriever=ocr_page_retriever,
        ocr_page_reranker=ocr_page_reranker,
        ocr_bge_indexed_docs=ocr_bge_indexed_docs,
        chunk_builder=chunk_builder,
        ocr_chunk_retriever=ocr_chunk_retriever,
        ocr_chunk_reranker=ocr_chunk_reranker,
        ocr_chunk_indexed_docs=ocr_chunk_indexed_docs,
        ocr_chunk_cache=ocr_chunk_cache,
        ocr_hybrid_retriever=ocr_hybrid_retriever,
        ocr_nv_chunk_retriever=ocr_nv_chunk_retriever,
        ocr_nv_chunk_indexed_docs=ocr_nv_chunk_indexed_docs,
        coarse_bm25_retriever_cache=coarse_bm25_retriever_cache,
        use_adaptive_coarse_visual=use_adaptive_coarse_visual,
        use_ocr_page_coarse=use_ocr_page_coarse,
        require_positive_candidate=use_adaptive_coarse_visual,
        visual_coarse_text_cache=visual_coarse_text_cache,
        ocr_coarse_text_cache=ocr_coarse_text_cache,
        ocr_coarse_retriever_cache=ocr_coarse_retriever_cache,
    )
    val_batches, val_batch_stats = _build_training_batches(
        cfg,
        val_dataset,
        question_encoder,
        logger,
        bm25_text_cache,
        bm25_retriever_cache,
        visual_retriever,
        visual_indexed_docs,
        ocr_quality_cache,
        feature_builder=feature_builder,
        ocr_backend=ocr_backend,
        visual_backend=visual_backend,
        ocr_bge_retriever=ocr_bge_retriever,
        ocr_bge_reranker=ocr_bge_reranker,
        ocr_page_retriever=ocr_page_retriever,
        ocr_page_reranker=ocr_page_reranker,
        ocr_bge_indexed_docs=ocr_bge_indexed_docs,
        chunk_builder=chunk_builder,
        ocr_chunk_retriever=ocr_chunk_retriever,
        ocr_chunk_reranker=ocr_chunk_reranker,
        ocr_chunk_indexed_docs=ocr_chunk_indexed_docs,
        ocr_chunk_cache=ocr_chunk_cache,
        ocr_hybrid_retriever=ocr_hybrid_retriever,
        ocr_nv_chunk_retriever=ocr_nv_chunk_retriever,
        ocr_nv_chunk_indexed_docs=ocr_nv_chunk_indexed_docs,
        coarse_bm25_retriever_cache=coarse_bm25_retriever_cache,
        use_adaptive_coarse_visual=use_adaptive_coarse_visual,
        use_ocr_page_coarse=use_ocr_page_coarse,
        require_positive_candidate=use_adaptive_coarse_visual,
        visual_coarse_text_cache=visual_coarse_text_cache,
        ocr_coarse_text_cache=ocr_coarse_text_cache,
        ocr_coarse_retriever_cache=ocr_coarse_retriever_cache,
    )
    if not train_batches or not val_batches:
        metrics = {
            "status": "skipped",
            "reason": "adaptive fusion candidate batches are empty",
            "train_batch_stats": train_batch_stats,
            "val_batch_stats": val_batch_stats,
        }
        save_json(metrics, Path(cfg.paths.metric_dir) / train_metric_filename)
        return metrics

    _dump_feature_debug(cfg, checkpoint_subdir, train_batches)

    feature_dim = len(train_batches[0]["features"][0])
    adaptive_model = model_cls(
        feature_dim=feature_dim,
        hidden_dim=int(cfg.fusion.hidden_dim),
        dropout=float(cfg.fusion.dropout),
    )
    trainer = Trainer(cfg, adaptive_model, train_batches, val_batches)
    train_metrics = trainer.train()

    best_checkpoint = load_pickle(trainer.best_checkpoint_path)
    adaptive_model.load_state_dict(best_checkpoint["model_state"])
    val_predictions, retrieval_metrics = _run_adaptive_eval_from_batches(cfg, adaptive_model, val_batches)

    best_name = "adaptive_fusion_best.pkl" if checkpoint_subdir == "adaptive_fusion" else f"{checkpoint_subdir}_best.pkl"
    save_pickle(best_checkpoint, checkpoint_dir / best_name)
    save_jsonl(val_predictions, Path(cfg.paths.prediction_dir) / prediction_filename)
    save_json(retrieval_metrics, Path(cfg.paths.metric_dir) / metric_filename)

    metrics = {
        **train_metrics,
        "checkpoint_path": str(checkpoint_dir / best_name),
        "train_batch_stats": train_batch_stats,
        "val_batch_stats": val_batch_stats,
        "val_retrieval_metrics": retrieval_metrics,
    }
    save_json(metrics, Path(cfg.paths.metric_dir) / train_metric_filename)
    logger.info("%s training completed: %s", checkpoint_subdir, metrics)
    return metrics


def _build_training_batches(
    cfg: Any,
    dataset: list[dict[str, Any]],
    question_encoder: QuestionEncoder,
    logger: Any,
    bm25_text_cache: dict[str, list[str]],
    bm25_retriever_cache: dict[str, Any],
    visual_retriever: Any,
    visual_indexed_docs: set[str],
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]],
    feature_builder: Any,
    ocr_backend: str,
    visual_backend: str,
    ocr_bge_retriever: OCRBGERetriever | None,
    ocr_bge_reranker: OCRBGEReranker | None,
    ocr_page_retriever: OCRBGERetriever | None,
    ocr_page_reranker: OCRBGEReranker | None,
    ocr_bge_indexed_docs: set[str],
    chunk_builder: OCRChunkBuilder | None,
    ocr_chunk_retriever: OCRBGEChunkRetriever | None,
    ocr_chunk_reranker: OCRBGEChunkReranker | None,
    ocr_chunk_indexed_docs: set[str],
    ocr_chunk_cache: dict[str, list[dict[str, Any]]],
    ocr_hybrid_retriever: OCRHybridRetriever | None,
    ocr_nv_chunk_retriever: OCRNVChunkRetriever | None,
    ocr_nv_chunk_indexed_docs: set[str],
    coarse_bm25_retriever_cache: dict[str, Any],
    use_adaptive_coarse_visual: bool = False,
    use_ocr_page_coarse: bool = False,
    require_positive_candidate: bool = False,
    visual_coarse_text_cache: dict[str, list[str]] | None = None,
    ocr_coarse_text_cache: dict[str, list[str]] | None = None,
    ocr_coarse_retriever_cache: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build candidate-level training batches from OCR and visual retrieval results."""
    if visual_backend == "old_visual":
        visual_backend = "colpali"
    if visual_backend not in {"colpali", "colqwen"}:
        raise ValueError(f"Unsupported visual backend for training batches: {visual_backend}")
    batches: list[dict[str, Any]] = []
    stats = {
        "num_samples": len(dataset),
        "num_built_batches": 0,
        "skipped_empty_candidates": 0,
        "skipped_missing_positive": 0,
    }
    visual_coarse_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="retrieval_router", stats_prefix="visual") if use_adaptive_coarse_visual else {}
    ocr_stats_prefix = "ocr_bm25" if ocr_backend == "ocr_page_bm25_bge_rerank" else "ocr"
    ocr_coarse_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="ocr_router", stats_prefix=ocr_stats_prefix) if use_ocr_page_coarse else {}
    ocr_page_pipeline_stats = _build_ocr_page_pipeline_metric_accumulator() if ocr_backend == "ocr_page_bm25_bge_rerank" else {}
    visual_coarse_text_cache = visual_coarse_text_cache or {}
    ocr_coarse_text_cache = ocr_coarse_text_cache or {}
    ocr_coarse_retriever_cache = ocr_coarse_retriever_cache or {}
    split_name = "train" if dataset and str(dataset[0].get("qid", "")).startswith("train_") else "val"
    for sample in tqdm(dataset, desc=f"BuildAdaptiveBatches {split_name}", unit="sample"):
        doc_id = str(sample["doc_id"])
        if ocr_backend == "bge_rerank":
            assert ocr_bge_retriever is not None and ocr_bge_reranker is not None
            ocr_result, _ = _retrieve_ocr_bge_for_sample(
                cfg,
                sample,
                topk=int(cfg.retrieval.topk),
                retriever=ocr_bge_retriever,
                indexed_docs=ocr_bge_indexed_docs,
                reranker=ocr_bge_reranker,
                coarse_topn=int(cfg.ocr_retrieval.coarse_topn),
            )
            ocr_page_texts = ocr_bge_retriever.get_document_page_texts(doc_id)
        elif ocr_backend == "ocr_page_bm25_bge_rerank":
            assert ocr_page_retriever is not None
            ocr_result, ocr_stats = _retrieve_ocr_with_bm25_bge_reranker_for_sample(
                cfg=cfg,
                sample=sample,
                topk=int(cfg.retrieval.topk),
                retriever=ocr_page_retriever,
                indexed_docs=ocr_bge_indexed_docs,
                reranker=ocr_page_reranker,
                bm25_doc_text_cache=ocr_coarse_text_cache,
                bm25_doc_retriever_cache=ocr_coarse_retriever_cache,
            )
            if use_ocr_page_coarse:
                _accumulate_adaptive_coarse_metrics(ocr_coarse_stats, ocr_stats, stats_prefix=ocr_stats_prefix)
            _accumulate_ocr_page_pipeline_metrics(ocr_page_pipeline_stats, ocr_stats)
            ocr_page_texts = ocr_coarse_text_cache.get(doc_id)
        elif ocr_backend == "bge_chunk_rerank":
            assert chunk_builder is not None and ocr_chunk_retriever is not None and ocr_chunk_reranker is not None
            if use_ocr_page_coarse:
                ocr_result, ocr_stats = _retrieve_ocr_with_page_coarse_for_sample(
                    cfg=cfg,
                    sample=sample,
                    topk=int(cfg.retrieval.topk),
                    chunk_builder=chunk_builder,
                    retriever=ocr_chunk_retriever,
                    indexed_docs=ocr_chunk_indexed_docs,
                    reranker=ocr_chunk_reranker,
                    chunk_cache=ocr_chunk_cache,
                    coarse_doc_text_cache=ocr_coarse_text_cache,
                    coarse_doc_retriever_cache=ocr_coarse_retriever_cache,
                    coarse_topn=int(cfg.ocr_chunk_retrieval.coarse_topn),
                    aggregation_strategy=str(cfg.ocr_chunk_retrieval.aggregate_strategy),
                    query_variant=str(cfg.ocr_chunk_retrieval.query_variant),
                )
                _accumulate_adaptive_coarse_metrics(ocr_coarse_stats, ocr_stats, stats_prefix=ocr_stats_prefix)
                ocr_page_texts = ocr_coarse_text_cache.get(doc_id)
            else:
                ocr_result, _ = _retrieve_ocr_bge_chunk_for_sample(
                    cfg=cfg,
                    sample=sample,
                    topk=int(cfg.retrieval.topk),
                    chunk_builder=chunk_builder,
                    retriever=ocr_chunk_retriever,
                    indexed_docs=ocr_chunk_indexed_docs,
                    reranker=ocr_chunk_reranker,
                    chunk_cache=ocr_chunk_cache,
                    coarse_topn=int(cfg.ocr_chunk_retrieval.coarse_topn),
                    aggregation_strategy=str(cfg.ocr_chunk_retrieval.aggregate_strategy),
                    query_variant=str(cfg.ocr_chunk_retrieval.query_variant),
                    collect_chunk_stats=False,
                )
                ocr_page_texts = bm25_text_cache.get(doc_id)
            if ocr_page_texts is None:
                ocr_page_texts = [load_page_ocr_text(path) for path in sample.get("ocr_paths", [])]
                bm25_text_cache[doc_id] = ocr_page_texts
        elif ocr_backend == "ocr_hybrid":
            assert chunk_builder is not None and ocr_hybrid_retriever is not None
            ocr_result, _ = _retrieve_ocr_hybrid_for_sample(
                cfg=cfg,
                sample=sample,
                topk=int(cfg.retrieval.topk),
                hybrid_retriever=ocr_hybrid_retriever,
                chunk_builder=chunk_builder,
                chunk_cache=ocr_chunk_cache,
                bm25_text_cache=bm25_text_cache,
            )
            ocr_page_texts = bm25_text_cache.get(doc_id, [])
        elif ocr_backend == "nv_chunk":
            assert chunk_builder is not None and ocr_nv_chunk_retriever is not None
            ocr_result, _ = _retrieve_ocr_nv_chunk_for_sample(
                cfg=cfg,
                sample=sample,
                topk=int(cfg.retrieval.topk),
                chunk_builder=chunk_builder,
                retriever=ocr_nv_chunk_retriever,
                indexed_docs=ocr_nv_chunk_indexed_docs,
                chunk_cache=ocr_chunk_cache,
                coarse_topn=int(cfg.ocr_nv_retrieval.coarse_topn),
                aggregation_strategy=str(cfg.ocr_chunk_retrieval.aggregate_strategy),
            )
            ocr_page_texts = bm25_text_cache.get(doc_id)
            if ocr_page_texts is None:
                ocr_page_texts = [load_page_ocr_text(path) for path in sample.get("ocr_paths", [])]
                bm25_text_cache[doc_id] = ocr_page_texts
        else:
            ocr_result, _ = _retrieve_bm25_for_sample(sample, int(cfg.retrieval.topk), bm25_text_cache, bm25_retriever_cache)
            ocr_page_texts = bm25_text_cache.get(doc_id, [])
        if use_adaptive_coarse_visual:
            visual_result, visual_stats = _retrieve_visual_with_adaptive_coarse_for_sample(
                cfg=cfg,
                sample=sample,
                retriever=visual_retriever,
                indexed_docs=visual_indexed_docs,
                coarse_doc_text_cache=visual_coarse_text_cache,
                coarse_doc_retriever_cache=coarse_bm25_retriever_cache,
            )
            _accumulate_adaptive_coarse_metrics(visual_coarse_stats, visual_stats, stats_prefix="visual")
        else:
            visual_result, _ = _retrieve_visual_for_sample(
                cfg,
                sample,
                int(cfg.retrieval.topk),
                visual_retriever,
                visual_indexed_docs,
            )
        candidate_rows = feature_builder(
            sample,
            ocr_result,
            visual_result,
            ocr_page_texts=ocr_page_texts,
            question_encoder=question_encoder,
            ocr_quality_cache=ocr_quality_cache,
        )
        if not candidate_rows:
            logger.warning("Skipping empty adaptive-fusion candidate set. qid=%s doc_id=%s", sample.get("qid"), sample.get("doc_id"))
            stats["skipped_empty_candidates"] += 1
            continue
        evidence_pages = {int(page) for page in sample.get("evidence_pages", [])}
        labels = [1.0 if int(row["page_id"]) in evidence_pages else 0.0 for row in candidate_rows]
        if require_positive_candidate and evidence_pages and not any(label > 0 for label in labels):
            logger.warning(
                "Skipping batch without positive candidate after adaptive routing. qid=%s doc_id=%s",
                sample.get("qid"),
                sample.get("doc_id"),
            )
            stats["skipped_missing_positive"] += 1
            continue
        batches.append(
            {
                "qid": sample["qid"],
                "doc_id": sample["doc_id"],
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "candidate_rows": candidate_rows,
                "features": [list(row["feature_vector"]) for row in candidate_rows],
                "labels": labels,
            }
        )
    stats["num_built_batches"] = len(batches)
    if use_adaptive_coarse_visual:
        stats.update(summarize_adaptive_coarse_stats(visual_coarse_stats, len(dataset), stats_prefix="visual"))
    if use_ocr_page_coarse:
        stats.update(summarize_adaptive_coarse_stats(ocr_coarse_stats, len(dataset), stats_prefix=ocr_stats_prefix))
        if ocr_stats_prefix == "ocr_bm25":
            stats.update(add_ocr_bm25_metric_aliases(stats))
    if ocr_page_pipeline_stats:
        stats.update(summarize_ocr_page_pipeline_metrics(ocr_page_pipeline_stats, len(dataset)))
    return batches, stats


def _create_visual_retriever(cfg: Any, visual_backend: str) -> Any:
    """Create the requested visual retriever backend for fusion training."""
    if visual_backend in {"colpali", "old_visual"}:
        return ColPaliRetriever(cfg, require_engine=True)
    if visual_backend == "colqwen":
        return ColQwenRetriever(cfg)
    raise ValueError(f"Unsupported visual backend for fusion training: {visual_backend}")


def _run_adaptive_eval_from_batches(
    cfg: Any,
    model: AdaptiveFusion,
    val_batches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a trained adaptive fusion model from cached validation batches."""
    predictions: list[dict[str, Any]] = []

    for batch in tqdm(val_batches, desc="AdaptiveEval val", unit="sample"):
        candidate_rows = batch.get("candidate_rows", [])
        if not candidate_rows:
            continue
        ranked = model.rank_candidates(candidate_rows)
        predictions.append(
            {
                "qid": batch["qid"],
                "doc_id": batch["doc_id"],
                "question": batch.get("question", ""),
                "evidence_pages": [int(page) for page in batch.get("evidence_pages", [])],
                "pred_page_ids": ranked["page_ids"][: int(cfg.retrieval.topk)],
                "pred_scores": ranked["scores"][: int(cfg.retrieval.topk)],
                "topk": int(cfg.retrieval.topk),
            }
        )

    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, list(cfg.retrieval.k_values)) if labeled_predictions else {}
    metrics["num_samples"] = len(predictions)
    return predictions, metrics


def _dump_feature_debug(cfg: Any, variant_name: str, batches: list[dict[str, Any]], limit: int = 64) -> None:
    """Save a small feature dump for manual sanity checks on order and scale."""
    debug_rows: list[dict[str, Any]] = []
    for batch in batches:
        candidate_rows = batch.get("candidate_rows", [])
        labels = batch.get("labels", [])
        for row, label in zip(candidate_rows, labels):
            debug_rows.append(
                {
                    "qid": batch.get("qid", ""),
                    "doc_id": batch.get("doc_id", ""),
                    "page_id": int(row.get("page_id", -1)),
                    "label": float(label),
                    "feature_names": list(row.get("feature_names", [])),
                    "feature_vector": [float(value) for value in row.get("feature_vector", [])],
                }
            )
            if len(debug_rows) >= limit:
                break
        if len(debug_rows) >= limit:
            break

    if not debug_rows:
        return
    debug_path = Path(cfg.paths.output_dir) / "debug" / f"fusion_feature_debug_{variant_name}.jsonl"
    save_jsonl(debug_rows, debug_path)


def train_adaptive_fusion_dynamic_rules(cfg: Any) -> dict[str, Any]:
    """Train a new rule-based dynamic weighting family on top of the frozen dual-branch retrievers."""
    return _train_dynamic_fusion_family(
        cfg=cfg,
        stage_name="dynamic_rules",
        logger_name="train_adaptive_fusion_dynamic_rules",
        checkpoint_subdir="adaptive_fusion_dynamic_rules",
        prediction_filename="adaptive_fusion_dynamic_rules_val_predictions.jsonl",
        metric_filename="adaptive_fusion_dynamic_rules_val_metrics.json",
        train_metric_filename=f"{cfg.experiment.name}_train_adaptive_fusion_dynamic_rules_metrics.json",
    )


def train_adaptive_fusion_learned_gating(cfg: Any) -> dict[str, Any]:
    """Train a learned global gate plus the existing lightweight MLP scorer."""
    return _train_dynamic_fusion_family(
        cfg=cfg,
        stage_name="learned_gating",
        logger_name="train_adaptive_fusion_learned_gating",
        checkpoint_subdir="adaptive_fusion_learned_gating",
        prediction_filename="adaptive_fusion_learned_gating_val_predictions.jsonl",
        metric_filename="adaptive_fusion_learned_gating_val_metrics.json",
        train_metric_filename=f"{cfg.experiment.name}_train_adaptive_fusion_learned_gating_metrics.json",
    )


def train_adaptive_fusion_gating_calibrated(cfg: Any) -> dict[str, Any]:
    """Train the calibrated gating family with optional pairwise hard-negative loss."""
    calibration_name = str(getattr(getattr(cfg, "dynamic_fusion", {}), "calibration_option", "raw"))
    loss_type = str(getattr(getattr(cfg, "dynamic_fusion", {}), "loss_type", "pointwise_bce"))
    loss_suffix = "margin" if loss_type == "pairwise_margin" else "pointwise"
    stage_suffix = f"{calibration_name}_{loss_suffix}"
    return _train_dynamic_fusion_family(
        cfg=cfg,
        stage_name="gating_calibrated",
        logger_name="train_adaptive_fusion_gating_calibrated",
        checkpoint_subdir=f"adaptive_fusion_gating_calibrated_{stage_suffix}",
        prediction_filename=f"adaptive_fusion_gating_calibrated_{stage_suffix}_val_predictions.jsonl",
        metric_filename=f"adaptive_fusion_gating_calibrated_{stage_suffix}_val_metrics.json",
        train_metric_filename=f"{cfg.experiment.name}_train_adaptive_fusion_gating_calibrated_{stage_suffix}_metrics.json",
    )


def train_adaptive_fusion_visual_nemotron_ocr_energy(cfg: Any) -> dict[str, Any]:
    """Train only the fusion MLP on frozen Nemotron visual signals + existing OCR route."""
    logger_name = "train_adaptive_fusion_visual_nemotron_ocr_energy"
    checkpoint_subdir = "adaptive_fusion_visual_nemotron_ocr_energy"
    logger = get_logger(logger_name, log_file=Path(cfg.paths.log_dir) / f"{cfg.experiment.name}_{checkpoint_subdir}.log")
    nemotron_cfg = cfg.visual_nemotron_energy
    logger.info("Training adaptive_fusion_visual_nemotron_ocr_energy")
    logger.info("local visual model=%s dataset=%s device=%s", nemotron_cfg.model_name, nemotron_cfg.dataset_path, nemotron_cfg.device)
    logger.info(
        "Running OCR branch unchanged: backend=%s page_coarse=%s bm25_coarse=%s semantic_topk=%s rerank_topk=%s",
        "ocr_page_bm25_bge_rerank",
        bool(getattr(cfg.ocr_router, "enable_ocr_page_coarse", False)),
        bool(getattr(cfg.ocr_router, "enable_bm25_coarse", True)),
        int(getattr(cfg.ocr_semantic_retrieval, "semantic_topk", 0) or 0),
        int(getattr(cfg.ocr_reranker, "rerank_topk", 0) or 0),
    )

    samples, dataset_summary = load_vidore_energy_dataset(str(nemotron_cfg.dataset_path), max_samples=0)
    train_samples, val_samples, split_summary = _split_vidore_energy_samples(cfg, samples)
    del samples
    release_memory()
    logger.info("ViDoRe Energy dataset summary: %s", dataset_summary)
    logger.info("ViDoRe Energy train/val split summary: %s", split_summary)
    log_ram(logger, "after_vidore_dataset_split")
    if not train_samples or not val_samples:
        metrics = {
            "status": "skipped",
            "reason": "ViDoRe Energy train/val split is empty",
            "dataset_summary": dataset_summary,
            "split_summary": split_summary,
        }
        save_json(metrics, Path(cfg.paths.metric_dir) / f"{cfg.experiment.name}_train_{checkpoint_subdir}_metrics.json")
        return metrics

    train_payload = None
    val_payload = None
    try:
        train_payload = _load_or_build_visual_nemotron_energy_batches(cfg, train_samples, split_name="train", logger=logger)
        val_payload = _load_or_build_visual_nemotron_energy_batches(cfg, val_samples, split_name="val", logger=logger)
        if not train_payload["batch_paths"] or not val_payload["batch_paths"]:
            metrics = {
                "status": "skipped",
                "reason": "visual_nemotron_ocr_energy batches are empty",
                "dataset_summary": dataset_summary,
                "split_summary": split_summary,
                "train_batch_stats": train_payload["stats"],
                "val_batch_stats": val_payload["stats"],
            }
            save_json(metrics, Path(cfg.paths.metric_dir) / f"{cfg.experiment.name}_train_{checkpoint_subdir}_metrics.json")
            return metrics

        feature_dim = int(train_payload["feature_dim"])
        model = AdaptiveFusionV2(
            feature_dim=feature_dim,
            hidden_dim=int(cfg.fusion.hidden_dim),
            dropout=float(cfg.fusion.dropout),
        )
        checkpoint_dir = Path(cfg.paths.checkpoint_dir) / checkpoint_subdir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = checkpoint_dir / f"{checkpoint_subdir}_best.pkl"
        last_checkpoint_path = checkpoint_dir / f"{checkpoint_subdir}_last.pkl"
        prediction_path = Path(cfg.paths.prediction_dir) / "adaptive_fusion_visual_nemotron_ocr_energy_val_predictions.jsonl"
        metric_path = Path(cfg.paths.metric_dir) / "adaptive_fusion_visual_nemotron_ocr_energy_val_metrics.json"
        train_metric_path = Path(cfg.paths.metric_dir) / f"{cfg.experiment.name}_train_{checkpoint_subdir}_metrics.json"

        best_mrr = float("-inf")
        best_epoch = -1
        best_val_metrics: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        patience = int(cfg.training.early_stopping_patience)
        total_epochs = int(cfg.training.epochs)
        epoch_iterator = tqdm(range(total_epochs), desc="visual_nemotron_ocr_energy epochs", unit="epoch")
        for epoch in epoch_iterator:
            log_ram(logger, f"before_train_epoch_{epoch + 1}")
            train_metrics = _run_visual_nemotron_energy_train_epoch(cfg, model, train_payload)
            log_ram(logger, f"after_train_epoch_{epoch + 1}")
            val_metrics = _eval_visual_nemotron_energy_batches(cfg, model, val_payload, prediction_path=str(prediction_path))
            log_ram(logger, f"after_val_epoch_{epoch + 1}")
            epoch_metrics = {
                "epoch": epoch + 1,
                **train_metrics,
                "val_mrr": float(val_metrics.get("MRR", 0.0)),
                "val_recall@1": float(val_metrics.get("Recall@1", 0.0)),
            }
            history.append(epoch_metrics)
            checkpoint = {
                "epoch": epoch + 1,
                "metric": float(val_metrics.get("MRR", 0.0)),
                "metric_name": "val_mrr",
                "branch_name": "adaptive_fusion_visual_nemotron_ocr_energy",
                "model_state": model.state_dict(),
            }
            log_ram(logger, f"before_checkpoint_save_epoch_{epoch + 1}")
            save_pickle(checkpoint, last_checkpoint_path)
            if float(val_metrics.get("MRR", 0.0)) > best_mrr:
                best_mrr = float(val_metrics.get("MRR", 0.0))
                best_epoch = epoch + 1
                best_val_metrics = dict(val_metrics)
                save_pickle(checkpoint, best_checkpoint_path)
                save_json(val_metrics, metric_path)
            log_ram(logger, f"after_checkpoint_save_epoch_{epoch + 1}")
            logger.info("Epoch %d visual_nemotron_ocr_energy metrics: %s", epoch + 1, epoch_metrics)
            epoch_iterator.set_postfix(
                train_loss=f"{float(train_metrics.get('train_loss', 0.0)):.4f}",
                val_mrr=f"{float(val_metrics.get('MRR', 0.0)):.4f}",
            )
            if epoch + 1 - best_epoch >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        best_checkpoint = load_pickle(best_checkpoint_path)
        model.load_state_dict(best_checkpoint["model_state"])
        run_post_train_eval = bool(getattr(nemotron_cfg, "run_post_train_eval", False))
        if run_post_train_eval:
            log_ram(logger, "before_post_train_val")
            best_val_metrics = _eval_visual_nemotron_energy_batches(cfg, model, val_payload, prediction_path=str(prediction_path))
            save_json(best_val_metrics, metric_path)
            log_ram(logger, "after_post_train_val")
        metrics = {
            "best_val_mrr": best_mrr,
            "best_epoch": best_epoch,
            "history": history,
            "checkpoint_path": str(best_checkpoint_path),
            "dataset_summary": dataset_summary,
            "split_summary": split_summary,
            "train_batch_stats": train_payload["stats"],
            "val_batch_stats": val_payload["stats"],
            "val_retrieval_metrics": best_val_metrics,
        }
        save_json(metrics, train_metric_path)
        logger.info("Best checkpoint selected by val_mrr. checkpoint=%s", best_checkpoint_path)
        logger.info("adaptive_fusion_visual_nemotron_ocr_energy training completed: %s", metrics)
        return metrics
    finally:
        if train_payload is not None:
            _cleanup_visual_nemotron_batch_payload(train_payload)
        if val_payload is not None:
            _cleanup_visual_nemotron_batch_payload(val_payload)
        release_memory(train_payload, val_payload)


def _split_vidore_energy_samples(cfg: Any, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create a deterministic train/val split for the local Energy subset when no official split is exposed."""
    sorted_samples = sorted(samples, key=lambda item: str(item.get("qid", "")))
    val_ratio = float(getattr(getattr(cfg, "visual_nemotron_energy", {}), "val_ratio", 0.2))
    val_ratio = min(max(val_ratio, 0.05), 0.5)
    val_count = max(1, int(round(len(sorted_samples) * val_ratio))) if len(sorted_samples) > 1 else 0
    train_samples = sorted_samples[:-val_count] if val_count else sorted_samples
    val_samples = sorted_samples[-val_count:] if val_count else []
    max_train_samples = int(getattr(cfg.training, "max_train_samples", 0) or 0)
    max_val_samples = int(getattr(cfg.training, "max_val_samples", 0) or 0)
    if max_train_samples > 0:
        train_samples = train_samples[:max_train_samples]
    if max_val_samples > 0:
        val_samples = val_samples[:max_val_samples]
    return train_samples, val_samples, {
        "split_policy": "deterministic_qid_sorted_80_20",
        "val_ratio": val_ratio,
        "num_total_samples": len(samples),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
    }


def _visual_nemotron_energy_config_signature(cfg: Any, samples: list[dict[str, Any]], split_name: str) -> str:
    """Build a cache signature for branch-specific feature rows."""
    nemotron_cfg = getattr(cfg, "visual_nemotron_energy", {})
    payload = {
        "branch": "adaptive_fusion_visual_nemotron_ocr_energy",
        "split_name": split_name,
        "sample_qids": [str(item.get("qid", "")) for item in samples],
        "model_name": str(getattr(nemotron_cfg, "model_name", "")),
        "dataset_path": str(getattr(nemotron_cfg, "dataset_path", "")),
        "ocr_backend": "ocr_page_bm25_bge_rerank",
        "ocr_page_coarse": bool(getattr(getattr(cfg, "ocr_router", {}), "enable_ocr_page_coarse", False)),
        "ocr_bm25_coarse": bool(getattr(getattr(cfg, "ocr_router", {}), "enable_bm25_coarse", True)),
        "semantic_topk": int(getattr(getattr(cfg, "ocr_semantic_retrieval", {}), "semantic_topk", 0) or 0),
        "rerank_topk": int(getattr(getattr(cfg, "ocr_reranker", {}), "rerank_topk", 0) or 0),
        "feature_builder": "build_candidate_features_visual_colqwen_ocr_chunk",
    }
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:10]


def _load_or_build_visual_nemotron_energy_batches(
    cfg: Any,
    samples: list[dict[str, Any]],
    split_name: str,
    logger: Any,
) -> dict[str, Any]:
    """Load or build disk-backed branch-specific candidate feature batches."""
    nemotron_cfg = getattr(cfg, "visual_nemotron_energy", {})
    feature_cache_enabled = bool(getattr(nemotron_cfg, "enable_feature_cache", True))
    feature_cache_save_enabled = bool(getattr(nemotron_cfg, "enable_feature_cache_save", False))
    batch_cache_save_enabled = bool(getattr(nemotron_cfg, "enable_batch_cache_save", feature_cache_save_enabled))
    signature = _visual_nemotron_energy_config_signature(cfg, samples, split_name)
    cache_dir = Path(cfg.paths.cache_dir) / "fusion_visual_nemotron_ocr_energy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_base_dir = cache_dir / f"{split_name}_{signature}"
    manifest_path = cache_base_dir / "manifest.pkl"
    if feature_cache_enabled and manifest_path.exists():
        logger.info("Loaded visual_nemotron_ocr_energy %s batch manifest from cache: %s", split_name, manifest_path)
        payload = load_pickle(manifest_path)
        payload["stats"]["feature_cache_hits"] = int(payload["stats"].get("feature_cache_hits", 0)) + 1
        return payload
    if not feature_cache_enabled:
        logger.info("visual_nemotron_ocr_energy feature cache disabled; using ephemeral %s batch shards.", split_name)
        _cleanup_stale_visual_nemotron_tmp_dirs(cache_dir, logger)
        cache_base_dir = Path(
            tempfile.mkdtemp(
                dir=str(cache_dir),
                prefix=f"{split_name}_{signature}_tmp_",
            )
        )
    log_ram(logger, f"before_build_{split_name}_batch_shards")
    payload = _build_visual_nemotron_energy_batch_shards(
        cfg,
        samples,
        split_name=split_name,
        logger=logger,
        cache_base_dir=cache_base_dir,
        persist_manifest=feature_cache_enabled and batch_cache_save_enabled,
    )
    payload["stats"].update(
        {
            "feature_cache_hits": 0,
            "feature_cache_misses": 1,
            "cache_path": str(cache_base_dir),
            "config_signature": signature,
            "split_name": split_name,
            "feature_cache_enabled": feature_cache_enabled,
            "feature_cache_save_enabled": feature_cache_save_enabled,
            "batch_cache_save_enabled": batch_cache_save_enabled,
        }
    )
    if feature_cache_enabled and batch_cache_save_enabled:
        save_pickle(payload, manifest_path)
    log_ram(logger, f"after_build_{split_name}_batch_shards")
    return payload


def _build_visual_nemotron_energy_batch_shards(
    cfg: Any,
    samples: list[dict[str, Any]],
    split_name: str,
    logger: Any,
    cache_base_dir: Path,
    persist_manifest: bool,
) -> dict[str, Any]:
    """Construct disk-backed training/eval batches from frozen Nemotron visual and existing OCR route outputs."""
    nemotron_cfg = cfg.visual_nemotron_energy
    question_encoder = QuestionEncoder()
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] = {}
    visual_retriever = NemotronVisualRetriever(
        model_path=str(nemotron_cfg.model_name),
        device=str(nemotron_cfg.device),
        local_files_only=bool(getattr(nemotron_cfg, "local_files_only", True)),
        cache_dir=str(getattr(nemotron_cfg, "cache_dir", "outputs/cache/visual_nemotron_energy")),
        cache_namespace="vidore_energy",
        enable_cache=bool(getattr(nemotron_cfg, "enable_embedding_cache", True)),
        batch_size=int(getattr(nemotron_cfg, "batch_size", 4)),
        query_batch_size=int(getattr(nemotron_cfg, "query_encode_batch_size", getattr(nemotron_cfg, "batch_size", 4))),
        page_batch_size=int(getattr(nemotron_cfg, "page_encode_batch_size", getattr(nemotron_cfg, "batch_size", 4))),
        enable_cache_save=bool(getattr(nemotron_cfg, "enable_embedding_cache_save", True)),
        in_memory_cache_limit=int(getattr(nemotron_cfg, "cache_chunk_size", 128) or 128),
    )
    ocr_page_retriever = OCRBGERetriever(cfg, config_attr="ocr_semantic_retrieval")
    ocr_page_reranker = OCRBGEReranker(cfg, config_attr="ocr_reranker") if bool(getattr(cfg.ocr_reranker, "enable_bge_reranker", True)) else None
    ocr_indexed_docs: set[str] = set()
    ocr_coarse_doc_text_cache: dict[str, list[str]] = {}
    ocr_coarse_doc_retriever_cache: dict[str, Any] = {}
    ocr_coarse_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="ocr_router", stats_prefix="ocr_bm25")
    ocr_page_pipeline_stats = _build_ocr_page_pipeline_metric_accumulator()
    shutil.rmtree(cache_base_dir, ignore_errors=True)
    batch_dir = cache_base_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_paths: list[str] = []
    stats: dict[str, Any] = {
        "num_samples": len(samples),
        "num_built_batches": 0,
        "skipped_empty_candidates": 0,
        "skipped_missing_positive": 0,
        "visual_embedding_cache_hits": 0,
        "visual_embedding_cache_misses": 0,
    }
    feature_dim = 0
    logger.info("Building visual_nemotron_ocr_energy %s batches with frozen retrievers: %d samples", split_name, len(samples))
    for raw_sample in tqdm(samples, desc=f"NemotronFusionBatches {split_name}", unit="sample"):
        sample = _vidore_energy_sample_to_processed_sample(raw_sample)
        evidence_pages = {int(page) for page in sample.get("evidence_pages", [])}
        visual_result, visual_stats = _retrieve_visual_nemotron_energy_for_sample(raw_sample, visual_retriever)
        stats["visual_embedding_cache_hits"] += int(visual_stats.get("embedding_cache_hits", 0))
        stats["visual_embedding_cache_misses"] += int(visual_stats.get("embedding_cache_misses", 0))
        ocr_result, ocr_stats = _retrieve_ocr_with_bm25_bge_reranker_for_sample(
            cfg=cfg,
            sample=sample,
            topk=int(cfg.retrieval.topk),
            retriever=ocr_page_retriever,
            indexed_docs=ocr_indexed_docs,
            reranker=ocr_page_reranker,
            bm25_doc_text_cache=ocr_coarse_doc_text_cache,
            bm25_doc_retriever_cache=ocr_coarse_doc_retriever_cache,
        )
        _accumulate_adaptive_coarse_metrics(ocr_coarse_stats, ocr_stats, stats_prefix="ocr_bm25")
        _accumulate_ocr_page_pipeline_metrics(ocr_page_pipeline_stats, ocr_stats)
        candidate_rows = build_candidate_features_visual_colqwen_ocr_chunk(
            sample,
            ocr_result,
            visual_result,
            ocr_page_texts=ocr_coarse_doc_text_cache.get(str(sample["doc_id"])),
            question_encoder=question_encoder,
            ocr_quality_cache=ocr_quality_cache,
        )
        if not candidate_rows:
            stats["skipped_empty_candidates"] += 1
            continue
        labels = [1.0 if int(row["page_id"]) in evidence_pages else 0.0 for row in candidate_rows]
        if not any(label > 0.5 for label in labels):
            stats["skipped_missing_positive"] += 1
            continue
        batch = {
            "qid": sample["qid"],
            "doc_id": sample["doc_id"],
            "question": sample.get("question", ""),
            "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
            "candidate_rows": candidate_rows,
            "labels": labels,
        }
        if feature_dim == 0 and batch["candidate_rows"]:
            feature_dim = len(batch["candidate_rows"][0].get("feature_vector", []))
        batch_path = batch_dir / f"batch_{len(batch_paths):06d}.pkl"
        save_pickle(batch, batch_path)
        batch_paths.append(str(batch_path))
        release_memory(batch, candidate_rows, labels, sample, visual_result, ocr_result)
        if len(batch_paths) % 64 == 0:
            log_ram(logger, f"during_{split_name}_batch_shard_build_{len(batch_paths)}")
    visual_retriever.save_cache()
    if not persist_manifest:
        logger.info("visual_nemotron_ocr_energy %s batch shards built in ephemeral mode: %s", split_name, cache_base_dir)
    stats["num_built_batches"] = len(batch_paths)
    stats.update(summarize_ocr_page_pipeline_metrics(ocr_page_pipeline_stats, len(samples)))
    stats.update(add_ocr_bm25_metric_aliases(summarize_adaptive_coarse_stats(ocr_coarse_stats, len(samples), stats_prefix="ocr_bm25")))
    stats.update(visual_retriever.cache_stats())
    return {
        "batch_paths": batch_paths,
        "feature_dim": int(feature_dim),
        "cache_base_dir": str(cache_base_dir),
        "persist_manifest": bool(persist_manifest),
        "stats": stats,
    }


def _iter_visual_nemotron_energy_batches(payload: dict[str, Any]) -> Any:
    for batch_path in payload.get("batch_paths", []):
        yield load_pickle(batch_path)


def _cleanup_visual_nemotron_batch_payload(payload: dict[str, Any]) -> None:
    cache_base_dir = str(payload.get("cache_base_dir", "") or "")
    persist_manifest = bool(payload.get("persist_manifest", False))
    if not cache_base_dir or persist_manifest:
        return
    shutil.rmtree(cache_base_dir, ignore_errors=True)


def _run_visual_nemotron_energy_train_epoch(
    cfg: Any,
    model: AdaptiveFusionV2,
    train_payload: dict[str, Any],
) -> dict[str, Any]:
    """Train one epoch of the branch-specific fusion MLP."""
    learning_rate = float(cfg.training.learning_rate)
    weight_decay = float(cfg.training.weight_decay)
    total_loss = 0.0
    total_batches = 0
    total_candidates = 0
    positive_samples = 0
    for batch in tqdm(_iter_visual_nemotron_energy_batches(train_payload), desc="visual_nemotron_ocr_energy train", unit="sample", leave=False):
        features = np.asarray([list(row["feature_vector"]) for row in batch["candidate_rows"]], dtype=float)
        labels = np.asarray(batch["labels"], dtype=float)
        loss = model.train_step(features, labels, learning_rate=learning_rate, weight_decay=weight_decay)
        total_loss += float(loss)
        total_batches += 1
        total_candidates += len(labels)
        if any(label > 0.5 for label in batch.get("labels", [])):
            positive_samples += 1
        release_memory(features, labels, batch)
    return {
        "train_loss": total_loss / max(total_batches, 1),
        "num_positive_samples": positive_samples,
        "num_train_batches": total_batches,
        "avg_candidates_per_sample": float(total_candidates / max(total_batches, 1)),
    }


def _eval_visual_nemotron_energy_batches(
    cfg: Any,
    model: AdaptiveFusionV2,
    payload: dict[str, Any],
    prediction_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate branch-specific fusion MLP over disk-backed candidate batches."""
    writer: JsonlStreamWriter | None = JsonlStreamWriter(prediction_path) if prediction_path else None
    online_metrics = StreamingSetRetrievalMetrics(list(cfg.retrieval.k_values))
    online_metrics.register_ndcg(10)
    for batch in tqdm(_iter_visual_nemotron_energy_batches(payload), desc="visual_nemotron_ocr_energy eval", unit="sample", leave=False):
        ranked = model.rank_candidates(batch["candidate_rows"])
        prediction = {
            "qid": batch["qid"],
            "doc_id": batch["doc_id"],
            "question": batch.get("question", ""),
            "evidence_pages": [int(page) for page in batch.get("evidence_pages", [])],
            "pred_page_ids": [int(page_id) for page_id in ranked["page_ids"][: int(cfg.retrieval.topk)]],
            "pred_scores": [float(score) for score in ranked["scores"][: int(cfg.retrieval.topk)]],
            "topk": int(cfg.retrieval.topk),
        }
        online_metrics.update(prediction["pred_page_ids"], prediction["evidence_pages"])
        if writer is not None:
            writer.write(prediction)
        release_memory(ranked, batch)
    if writer is not None:
        writer.close()
    metrics = online_metrics.finalize()
    metrics["NDCG@10"] = float(metrics.get("nDCG@10", 0.0))
    return metrics


def _ndcg_at_k_for_predictions(predictions: list[dict[str, Any]], k: int) -> float:
    """Compute binary NDCG@K for branch validation predictions."""
    if not predictions:
        return 0.0
    total = 0.0
    for item in predictions:
        evidence = {int(page) for page in item.get("evidence_pages", [])}
        if not evidence:
            continue
        dcg = 0.0
        for index, page_id in enumerate(item.get("pred_page_ids", [])[:k], start=1):
            if int(page_id) in evidence:
                dcg += 1.0 / np.log2(index + 1)
        ideal_hits = min(len(evidence), k)
        idcg = sum(1.0 / np.log2(index + 1) for index in range(1, ideal_hits + 1))
        total += dcg / max(idcg, 1e-8)
    return float(total / len(predictions))


def _train_dynamic_fusion_family(
    cfg: Any,
    stage_name: str,
    logger_name: str,
    checkpoint_subdir: str,
    prediction_filename: str,
    metric_filename: str,
    train_metric_filename: str,
) -> dict[str, Any]:
    """Shared staged training loop for dynamic OCR/visual weighting experiments."""
    logger = get_logger(logger_name, log_file=Path(cfg.paths.log_dir) / f"{cfg.experiment.name}_{checkpoint_subdir}.log")
    logger.info("Starting staged dynamic fusion training. stage=%s", stage_name)
    logger.info(
        "dynamic settings: rule_variant=%s calibration=%s loss_type=%s gate_hidden_dim=%s cache=%s",
        str(getattr(getattr(cfg, "dynamic_fusion", {}), "rule_variant", "combined")),
        str(getattr(getattr(cfg, "dynamic_fusion", {}), "calibration_option", "raw")),
        str(getattr(getattr(cfg, "dynamic_fusion", {}), "loss_type", "pointwise_bce")),
        int(getattr(getattr(cfg, "dynamic_fusion", {}), "gate_hidden_dim", 16)),
        bool(getattr(getattr(cfg, "dynamic_fusion", {}), "enable_feature_cache", True)),
    )

    logger.info("Loading processed train split from %s", cfg.dataset.processed_files["train"])
    train_dataset = load_jsonl(cfg.dataset.processed_files["train"]) if Path(cfg.dataset.processed_files["train"]).exists() else []
    logger.info("Loading processed val split from %s", cfg.dataset.processed_files["val"])
    val_dataset = load_jsonl(cfg.dataset.processed_files["val"]) if Path(cfg.dataset.processed_files["val"]).exists() else []

    max_train_samples = int(getattr(cfg.training, "max_train_samples", 0) or 0)
    max_val_samples = int(getattr(cfg.training, "max_val_samples", 0) or 0)
    if max_train_samples > 0:
        train_dataset = train_dataset[:max_train_samples]
        logger.info("Using train subset for smoke test: %d samples", len(train_dataset))
    if max_val_samples > 0:
        val_dataset = val_dataset[:max_val_samples]
        logger.info("Using val subset for smoke test: %d samples", len(val_dataset))

    if not train_dataset or not val_dataset:
        metrics = {"status": "skipped", "reason": "train/val processed datasets are missing or empty"}
        save_json(metrics, Path(cfg.paths.metric_dir) / train_metric_filename)
        return metrics

    checkpoint_dir = Path(cfg.paths.checkpoint_dir) / checkpoint_subdir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base_cache_train = _load_or_build_dynamic_stage_inputs(cfg, train_dataset, stage_name, split_name="train", logger=logger)
    base_cache_val = _load_or_build_dynamic_stage_inputs(cfg, val_dataset, stage_name, split_name="val", logger=logger)
    train_batches = base_cache_train["batches"]
    val_batches = base_cache_val["batches"]
    train_batch_stats = dict(base_cache_train["stats"])
    val_batch_stats = dict(base_cache_val["stats"])
    if not train_batches or not val_batches:
        metrics = {
            "status": "skipped",
            "reason": "dynamic fusion stage inputs are empty",
            "train_batch_stats": train_batch_stats,
            "val_batch_stats": val_batch_stats,
        }
        save_json(metrics, Path(cfg.paths.metric_dir) / train_metric_filename)
        return metrics

    feature_dim = len(train_batches[0]["candidate_rows"][0].get("feature_vector", []))
    scorer = AdaptiveFusionV2(
        feature_dim=feature_dim,
        hidden_dim=int(cfg.fusion.hidden_dim),
        dropout=float(cfg.fusion.dropout),
    )
    gate_model = None
    if stage_name in {"learned_gating", "gating_calibrated"}:
        gate_dim = len(train_batches[0].get("gating_feature_vector", []))
        gate_model = GateNet(
            input_dim=gate_dim,
            hidden_dim=int(getattr(getattr(cfg, "dynamic_fusion", {}), "gate_hidden_dim", 16)),
            min_weight=float(getattr(getattr(cfg, "dynamic_fusion", {}), "min_weight", 0.2)),
            max_weight=float(getattr(getattr(cfg, "dynamic_fusion", {}), "max_weight", 0.8)),
        )

    best_mrr = float("-inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    patience = int(cfg.training.early_stopping_patience)
    total_epochs = int(cfg.training.epochs)
    best_checkpoint_path = checkpoint_dir / f"{checkpoint_subdir}_best.pkl"
    last_checkpoint_path = checkpoint_dir / f"{checkpoint_subdir}_last.pkl"

    epoch_iterator = tqdm(range(total_epochs), desc=f"{stage_name} epochs", unit="epoch")
    for epoch in epoch_iterator:
        train_epoch_metrics = _run_dynamic_train_epoch(cfg, scorer, gate_model, train_batches, stage_name, epoch)
        val_predictions, val_retrieval_metrics, val_dynamic_metrics = _run_dynamic_eval_from_batches(
            cfg,
            scorer,
            gate_model,
            val_batches,
            stage_name=stage_name,
            desc=f"{stage_name} val",
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            **train_epoch_metrics,
            "val_mrr": float(val_retrieval_metrics.get("MRR", 0.0)),
            "val_recall@1": float(val_retrieval_metrics.get("Recall@1", 0.0)),
            "val_loss_proxy": float(train_epoch_metrics.get("train_loss", 0.0)),
            **{f"val_{key}": value for key, value in val_dynamic_metrics.items()},
        }
        history.append(epoch_metrics)
        self_checkpoint = {
            "epoch": epoch + 1,
            "metric": float(val_retrieval_metrics.get("MRR", 0.0)),
            "stage_name": stage_name,
            "model_state": scorer.state_dict(),
            "gate_state": gate_model.state_dict() if gate_model is not None else None,
        }
        save_pickle(self_checkpoint, last_checkpoint_path)
        if float(val_retrieval_metrics.get("MRR", 0.0)) > best_mrr:
            best_mrr = float(val_retrieval_metrics.get("MRR", 0.0))
            best_epoch = epoch + 1
            save_pickle(self_checkpoint, best_checkpoint_path)
            save_jsonl(val_predictions, Path(cfg.paths.prediction_dir) / prediction_filename)
            save_json(
                {**val_retrieval_metrics, **val_dynamic_metrics},
                Path(cfg.paths.metric_dir) / metric_filename,
            )
        logger.info("Epoch %d staged dynamic metrics: %s", epoch + 1, epoch_metrics)
        epoch_iterator.set_postfix(
            train_loss=f"{float(train_epoch_metrics.get('train_loss', 0.0)):.4f}",
            val_mrr=f"{float(val_retrieval_metrics.get('MRR', 0.0)):.4f}",
        )
        if epoch + 1 - best_epoch >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    best_checkpoint = load_pickle(best_checkpoint_path)
    scorer.load_state_dict(best_checkpoint["model_state"])
    if gate_model is not None and best_checkpoint.get("gate_state") is not None:
        gate_model.load_state_dict(best_checkpoint["gate_state"])
    val_predictions, retrieval_metrics, val_dynamic_metrics = _run_dynamic_eval_from_batches(
        cfg,
        scorer,
        gate_model,
        val_batches,
        stage_name=stage_name,
        desc=f"{stage_name} best-val",
    )
    save_jsonl(val_predictions, Path(cfg.paths.prediction_dir) / prediction_filename)
    save_json({**retrieval_metrics, **val_dynamic_metrics}, Path(cfg.paths.metric_dir) / metric_filename)
    metrics = {
        "best_val_mrr": best_mrr,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint_path": str(best_checkpoint_path),
        "train_batch_stats": train_batch_stats,
        "val_batch_stats": val_batch_stats,
        "val_retrieval_metrics": retrieval_metrics,
        "val_dynamic_metrics": val_dynamic_metrics,
    }
    save_json(metrics, Path(cfg.paths.metric_dir) / train_metric_filename)
    logger.info("%s training completed: %s", checkpoint_subdir, metrics)
    return metrics


def _load_or_build_dynamic_stage_inputs(
    cfg: Any,
    dataset: list[dict[str, Any]],
    stage_name: str,
    split_name: str,
    logger: Any,
) -> dict[str, Any]:
    """Load or build cached dynamic-fusion base signals and stage-prepared inputs."""
    dynamic_cfg = getattr(cfg, "dynamic_fusion", {})
    cache_enabled = bool(getattr(dynamic_cfg, "enable_feature_cache", True))
    signature = _dynamic_config_signature(cfg, stage_name)
    base_cache_path = Path(cfg.paths.cache_dir) / f"{cfg.experiment.name}_{stage_name}_{split_name}_layer1_{signature}.pkl"
    prepared_cache_path = Path(cfg.paths.cache_dir) / f"{cfg.experiment.name}_{stage_name}_{split_name}_layer2_{signature}.pkl"
    if cache_enabled and prepared_cache_path.exists():
        logger.info("Loaded dynamic stage inputs from cache: %s", prepared_cache_path)
        payload = load_pickle(prepared_cache_path)
        payload["stats"]["feature_cache_hits"] = int(payload["stats"].get("feature_cache_hits", 0)) + 1
        return payload

    if cache_enabled and base_cache_path.exists():
        base_payload = load_pickle(base_cache_path)
        base_payload["stats"]["base_cache_hits"] = int(base_payload["stats"].get("base_cache_hits", 0)) + 1
    else:
        base_batches, base_stats = _build_dynamic_base_batches(cfg, dataset, logger)
        base_payload = {"batches": base_batches, "stats": {**base_stats, "base_cache_hits": 0, "base_cache_misses": 1}}
        if cache_enabled:
            save_pickle(base_payload, base_cache_path)

    prepared_batches = [_prepare_stage_input_batch(batch, cfg, stage_name) for batch in base_payload["batches"]]
    prepared_payload = {
        "batches": prepared_batches,
        "stats": {
            **base_payload["stats"],
            "feature_cache_hits": 0,
            "feature_cache_misses": 1,
            "stage_name": stage_name,
        },
    }
    if cache_enabled:
        save_pickle(prepared_payload, prepared_cache_path)
    return prepared_payload


def _dynamic_config_signature(cfg: Any, stage_name: str) -> str:
    dynamic_cfg = getattr(cfg, "dynamic_fusion", {})
    payload = {
        "stage_name": stage_name,
        "rule_variant": getattr(dynamic_cfg, "rule_variant", "combined"),
        "calibration_option": getattr(dynamic_cfg, "calibration_option", "raw"),
        "loss_type": getattr(dynamic_cfg, "loss_type", "pointwise_bce"),
        "hard_negative_k": getattr(dynamic_cfg, "pairwise_hard_negative_k", 3),
        "min_weight": getattr(dynamic_cfg, "min_weight", 0.2),
        "max_weight": getattr(dynamic_cfg, "max_weight", 0.8),
        "query_bias_scale": getattr(dynamic_cfg, "query_bias_scale", 0.18),
        "page_correction_scale": getattr(dynamic_cfg, "page_correction_scale", 0.08),
        "confidence_correction_scale": getattr(dynamic_cfg, "confidence_correction_scale", 0.08),
    }
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:10]


def _build_dynamic_base_batches(cfg: Any, dataset: list[dict[str, Any]], logger: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build base candidate rows for the new dynamic-fusion family."""
    question_encoder = QuestionEncoder()
    bm25_text_cache: dict[str, list[str]] = {}
    visual_coarse_text_cache: dict[str, list[str]] = {}
    bm25_retriever_cache: dict[str, Any] = {}
    coarse_bm25_retriever_cache: dict[str, Any] = {}
    ocr_coarse_text_cache: dict[str, list[str]] = {}
    ocr_coarse_retriever_cache: dict[str, Any] = {}
    ocr_page_retriever = OCRBGERetriever(cfg, config_attr="ocr_semantic_retrieval")
    ocr_page_reranker = OCRBGEReranker(cfg, config_attr="ocr_reranker") if bool(getattr(cfg.ocr_reranker, "enable_bge_reranker", True)) else None
    ocr_indexed_docs: set[str] = set()
    visual_retriever = ColQwenRetriever(cfg)
    visual_indexed_docs: set[str] = set()
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] = {}
    return _build_training_batches(
        cfg,
        dataset,
        question_encoder,
        logger,
        bm25_text_cache,
        bm25_retriever_cache,
        visual_retriever,
        visual_indexed_docs,
        ocr_quality_cache,
        feature_builder=build_candidate_features_visual_colqwen_ocr_chunk,
        ocr_backend="ocr_page_bm25_bge_rerank",
        visual_backend="colqwen",
        ocr_bge_retriever=None,
        ocr_bge_reranker=None,
        ocr_page_retriever=ocr_page_retriever,
        ocr_page_reranker=ocr_page_reranker,
        ocr_bge_indexed_docs=ocr_indexed_docs,
        chunk_builder=None,
        ocr_chunk_retriever=None,
        ocr_chunk_reranker=None,
        ocr_chunk_indexed_docs=set(),
        ocr_chunk_cache={},
        ocr_hybrid_retriever=None,
        ocr_nv_chunk_retriever=None,
        ocr_nv_chunk_indexed_docs=set(),
        coarse_bm25_retriever_cache=coarse_bm25_retriever_cache,
        use_adaptive_coarse_visual=True,
        use_ocr_page_coarse=True,
        require_positive_candidate=True,
        visual_coarse_text_cache=visual_coarse_text_cache,
        ocr_coarse_text_cache=ocr_coarse_text_cache,
        ocr_coarse_retriever_cache=ocr_coarse_retriever_cache,
    )


def _prepare_stage_input_batch(batch: dict[str, Any], cfg: Any, stage_name: str) -> dict[str, Any]:
    """Assemble cached base signals and sample-level gating inputs for a staged experiment."""
    stage_batch = {
        "qid": batch["qid"],
        "doc_id": batch["doc_id"],
        "question": batch.get("question", ""),
        "evidence_pages": list(batch.get("evidence_pages", [])),
        "labels": list(batch.get("labels", [])),
        "candidate_rows": batch.get("candidate_rows", []),
    }
    gating_features = build_dynamic_gating_feature_vector({"question": batch.get("question", "")}, stage_batch["candidate_rows"])
    stage_batch["gating_feature_names"] = list(gating_features["feature_names"])
    stage_batch["gating_feature_vector"] = list(gating_features["feature_vector"])
    stage_batch["gating_feature_map"] = dict(gating_features["feature_map"])
    if stage_name == "gating_calibrated":
        calibrated_rows, calibration_stats = calibrate_route_scores(
            stage_batch["candidate_rows"],
            option=str(getattr(getattr(cfg, "dynamic_fusion", {}), "calibration_option", "raw")),
        )
        stage_batch["candidate_rows"] = refresh_candidate_feature_vectors(calibrated_rows)
        stage_batch["calibration_stats"] = calibration_stats
    else:
        stage_batch["calibration_stats"] = {}
    return stage_batch


def _run_dynamic_train_epoch(
    cfg: Any,
    scorer: AdaptiveFusionV2,
    gate_model: GateNet | None,
    train_batches: list[dict[str, Any]],
    stage_name: str,
    epoch: int,
) -> dict[str, Any]:
    dynamic_cfg = getattr(cfg, "dynamic_fusion", {})
    learning_rate = float(cfg.training.learning_rate)
    gate_learning_rate = float(getattr(dynamic_cfg, "gate_learning_rate", learning_rate))
    weight_decay = float(cfg.training.weight_decay)
    loss_type = str(getattr(dynamic_cfg, "loss_type", "pointwise_bce"))
    margin = float(getattr(dynamic_cfg, "pairwise_margin", 1.0))
    hard_negative_k = int(getattr(dynamic_cfg, "pairwise_hard_negative_k", 3))
    total_loss = 0.0
    total_gate_loss = 0.0
    total_batches = 0
    total_pairs = 0
    weight_debug: list[dict[str, Any]] = []
    iterator = tqdm(train_batches, desc=f"{stage_name} train epoch {epoch + 1}", unit="sample", leave=False)
    for batch in iterator:
        prepared = _prepare_dynamic_batch_for_scoring(cfg, batch, stage_name, gate_model, train_gate=True, gate_learning_rate=gate_learning_rate, weight_decay=weight_decay)
        features = np.asarray([list(row["feature_vector"]) for row in prepared["weighted_rows"]], dtype=float)
        labels = np.asarray(batch["labels"], dtype=float)
        if loss_type == "pairwise_margin":
            loss, num_pairs = scorer.train_step_pairwise(
                features,
                labels,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                margin=margin,
                hard_negative_k=hard_negative_k,
            )
            total_pairs += int(num_pairs)
        else:
            loss = scorer.train_step(
                features,
                labels,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        total_loss += float(loss)
        total_gate_loss += float(prepared.get("gate_loss", 0.0))
        total_batches += 1
        weight_debug.append(prepared["weight_debug"])
        iterator.set_postfix(
            loss=f"{float(total_loss / max(total_batches, 1)):.4f}",
            alpha_v=f"{float(prepared['weight_debug'].get('alpha_v', 0.5)):.2f}",
            alpha_o=f"{float(prepared['weight_debug'].get('alpha_o', 0.5)):.2f}",
        )
    summary = summarize_weight_debug(weight_debug)
    summary.update(
        {
            "train_loss": total_loss / max(total_batches, 1),
            "train_gate_loss": total_gate_loss / max(total_batches, 1),
            "loss_type": loss_type,
            "num_positive_samples": sum(1 for batch in train_batches if any(label > 0 for label in batch.get("labels", []))),
            "num_negative_pairs": total_pairs,
        }
    )
    return summary


def _prepare_dynamic_batch_for_scoring(
    cfg: Any,
    batch: dict[str, Any],
    stage_name: str,
    gate_model: GateNet | None,
    train_gate: bool,
    gate_learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    dynamic_cfg = getattr(cfg, "dynamic_fusion", {})
    candidate_rows = batch.get("candidate_rows", [])
    alpha_v = 0.5
    alpha_o = 0.5
    weight_debug: dict[str, Any] = {}
    gate_loss = 0.0
    if stage_name == "dynamic_rules":
        alpha_v, alpha_o, debug_meta = compute_rule_based_weights(
            candidate_rows,
            {"question": batch.get("question", "")},
            variant=str(getattr(dynamic_cfg, "rule_variant", "combined")),
            cfg=cfg,
        )
        weight_debug = {"alpha_v": alpha_v, "alpha_o": alpha_o, **debug_meta}
    else:
        assert gate_model is not None
        target_weights = derive_gate_targets(
            candidate_rows,
            batch.get("evidence_pages", []),
            min_weight=float(getattr(dynamic_cfg, "min_weight", 0.2)),
            max_weight=float(getattr(dynamic_cfg, "max_weight", 0.8)),
        )
        if train_gate:
            gate_loss = gate_model.train_step(
                batch.get("gating_feature_vector", []),
                target_weights=target_weights,
                learning_rate=gate_learning_rate,
                weight_decay=float(getattr(dynamic_cfg, "gate_weight_decay", weight_decay)),
            )
        alpha_v, alpha_o = gate_model.predict_weights(batch.get("gating_feature_vector", []))
        buckets = compute_rule_based_weights(
            candidate_rows,
            {"question": batch.get("question", "")},
            variant="combined",
            cfg=cfg,
        )[2]
        weight_debug = {
            "alpha_v": alpha_v,
            "alpha_o": alpha_o,
            "target_alpha_v": float(target_weights[0]),
            "target_alpha_o": float(target_weights[1]),
            **{key: value for key, value in buckets.items() if key in {"query_bucket", "page_bucket"}},
        }
    weighted_rows = apply_branch_reweighting(candidate_rows, alpha_v=alpha_v, alpha_o=alpha_o)
    weighted_rows = refresh_candidate_feature_vectors(weighted_rows)
    return {
        "weighted_rows": weighted_rows,
        "weight_debug": weight_debug,
        "gate_loss": gate_loss,
    }


def _run_dynamic_eval_from_batches(
    cfg: Any,
    scorer: AdaptiveFusionV2,
    gate_model: GateNet | None,
    val_batches: list[dict[str, Any]],
    stage_name: str,
    desc: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    weight_debug: list[dict[str, Any]] = []
    calibration_debug: dict[str, list[dict[str, float]]] = {"visual_pre": [], "visual_post": [], "ocr_pre": [], "ocr_post": []}
    for batch in tqdm(val_batches, desc=desc, unit="sample", leave=False):
        prepared = _prepare_dynamic_batch_for_scoring(
            cfg,
            batch,
            stage_name=stage_name,
            gate_model=gate_model,
            train_gate=False,
            gate_learning_rate=0.0,
            weight_decay=0.0,
        )
        weight_debug.append(prepared["weight_debug"])
        if batch.get("calibration_stats"):
            for key in calibration_debug:
                calibration_debug[key].append(batch["calibration_stats"].get(key, {}))
        ranked = scorer.rank_candidates(prepared["weighted_rows"])
        predictions.append(
            {
                "qid": batch["qid"],
                "doc_id": batch["doc_id"],
                "question": batch.get("question", ""),
                "evidence_pages": [int(page) for page in batch.get("evidence_pages", [])],
                "pred_page_ids": ranked["page_ids"][: int(cfg.retrieval.topk)],
                "pred_scores": ranked["scores"][: int(cfg.retrieval.topk)],
                "topk": int(cfg.retrieval.topk),
            }
        )
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    retrieval_metrics = evaluate_retrieval(labeled_predictions, list(cfg.retrieval.k_values)) if labeled_predictions else {}
    retrieval_metrics["num_samples"] = len(predictions)
    dynamic_metrics = summarize_weight_debug(weight_debug)
    dynamic_metrics["rule_variant_name"] = str(getattr(getattr(cfg, "dynamic_fusion", {}), "rule_variant", "combined"))
    if any(calibration_debug[key] for key in calibration_debug):
        dynamic_metrics["calibration_debug"] = {
            key: {
                "mean": float(np.mean([item.get("mean", 0.0) for item in values])) if values else 0.0,
                "std": float(np.mean([item.get("std", 0.0) for item in values])) if values else 0.0,
                "min": float(np.mean([item.get("min", 0.0) for item in values])) if values else 0.0,
                "max": float(np.mean([item.get("max", 0.0) for item in values])) if values else 0.0,
            }
            for key, values in calibration_debug.items()
        }
    return predictions, retrieval_metrics, dynamic_metrics
