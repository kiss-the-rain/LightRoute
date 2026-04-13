"""Training entrypoints for lightweight adaptive fusion retrieval models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.evaluation.retrieval_metrics import evaluate_retrieval
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
    _retrieve_visual_for_sample,
    _retrieve_visual_with_adaptive_coarse_for_sample,
    summarize_ocr_page_pipeline_metrics,
    summarize_adaptive_coarse_stats,
)
from src.models.adaptive_fusion import AdaptiveFusion, AdaptiveFusionV2
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
from src.retrieval.fusion_features import (
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
)
from src.training.trainer import Trainer
from src.utils.io_utils import load_jsonl, load_pickle, save_json, save_jsonl, save_pickle
from src.utils.logger import get_logger


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
        "visual_backend=%s ocr_backend=%s visual_adaptive_coarse=%s visual_bypass_threshold=%s visual_coarse_topk=%s "
        "ocr_page_coarse=%s ocr_bypass_threshold=%s ocr_coarse_topk=%s ocr_semantic_topk=%s ocr_rerank_topk=%s",
        visual_backend,
        ocr_backend,
        bool(use_adaptive_coarse_visual and getattr(visual_router_cfg, "enable_adaptive_coarse", False)),
        int(getattr(visual_router_cfg, "bypass_threshold", 0) or 0),
        int(getattr(visual_router_cfg, "coarse_topk", 0) or 0),
        bool(use_ocr_page_coarse and getattr(ocr_router_cfg, "enable_ocr_page_coarse", False)),
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
