"""Retrieval-only inference helpers with an OCR-only BM25 baseline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from src.data.vidore_energy_loader import load_vidore_energy_dataset
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.retrieval.colpali_retriever import ColPaliRetriever
from src.retrieval.colqwen_retriever import ColQwenRetriever
from src.models.adaptive_fusion import AdaptiveFusion, AdaptiveFusionV2
from src.models.gating_mlp import GateNet
from src.models.fusion_fixed import FixedFusion
from src.models.fusion_rrf import RRFFusion
from src.models.question_encoder import QuestionEncoder
from src.retrieval.adaptive_coarse_router import (
    add_adaptive_coarse_recall_stats,
    filter_sample_to_page_ids,
    route_document_pages_with_adaptive_coarse,
    summarize_adaptive_coarse_stats,
)
from src.retrieval.bm25_retriever import BM25Retriever, load_page_ocr_text
from src.retrieval.ocr_bge_chunk_retriever import OCRBGEChunkRetriever
from src.retrieval.ocr_bge_chunk_reranker import OCRBGEChunkReranker
from src.retrieval.ocr_bge_retriever import (
    OCRBGERetriever,
    format_bge_query,
    load_bge_page_text,
    summarize_page_text_lengths,
)
from src.retrieval.ocr_bge_reranker import OCRBGEReranker
from src.retrieval.ocr_chunker import OCRChunkBuilder, aggregate_chunk_scores_to_page
from src.retrieval.ocr_hybrid_retriever import OCRHybridRetriever
from src.retrieval.ocr_jina_retriever import OCRJinaChunkRetriever
from src.retrieval.ocr_nv_retriever import OCRNVChunkRetriever
from src.retrieval.ocr_page_pipeline import (
    run_ocr_page_pipeline_for_sample,
    run_ocr_page_pipeline_for_subset,
)
from src.retrieval.nemotron_visual_retriever import NemotronVisualRetriever
from src.retrieval.fusion_features import (
    build_candidate_features as build_adaptive_candidate_features,
    build_dynamic_gating_feature_vector,
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
    summarize_weight_debug,
)
from src.retrieval.fusion import fixed_fusion, rrf_fusion
from src.utils.io_utils import ensure_dir, load_jsonl, load_pickle, save_jsonl
from src.utils.logger import get_logger


def infer_retrieval_sample(
    sample: dict[str, Any],
    retrieval_bundle: dict[str, Any],
    mode: str,
    runner: Callable[[dict[str, Any]], dict[str, list]],
) -> dict[str, Any]:
    """Run one retrieval or fusion mode for a single sample."""
    result = runner(retrieval_bundle)
    return {
        "qid": sample["qid"],
        "doc_id": sample["doc_id"],
        "mode": mode,
        "page_ids": result.get("page_ids", []),
        "scores": result.get("scores", []),
        "ranks": result.get("ranks", []),
    }


def _extract_page_idx_from_path(path_or_name: str) -> int:
    """Extract an integer page index from names like `docid_p80.jpg`."""
    match = re.search(r"_p(\d+)(?:\.[^.]+)?$", str(path_or_name))
    if not match:
        raise ValueError(f"Unable to extract page index from path: {path_or_name}")
    return int(match.group(1))


def infer_candidate_feature_dim(candidate_features: Any) -> int:
    """Infer candidate feature dimensionality for adaptive fusion initialization."""
    if candidate_features is None:
        raise ValueError("candidate_features is None.")
    if not candidate_features:
        raise ValueError("candidate_features is empty.")

    first_item = candidate_features[0]
    if isinstance(first_item, dict):
        if "feature_vector" in first_item:
            return len(first_item["feature_vector"])
        return len([value for key, value in first_item.items() if key != "page_id" and isinstance(value, (int, float))])
    if isinstance(first_item, (list, tuple)):
        return len(first_item)
    if hasattr(candidate_features, "shape"):
        shape = getattr(candidate_features, "shape")
        if len(shape) == 1:
            return int(shape[0])
        if len(shape) >= 2:
            return int(shape[-1])

    raise TypeError(
        "Unable to infer candidate feature dimension from type "
        f"{type(candidate_features).__name__}."
    )


def get_or_create_question_encoder(question_encoder: QuestionEncoder | None) -> QuestionEncoder:
    """Return a reusable question encoder instance."""
    return question_encoder or QuestionEncoder()


def get_or_create_fixed_fusion(cfg: Any, fixed_fusion: FixedFusion | None) -> FixedFusion:
    """Return a reusable fixed fusion module."""
    return fixed_fusion or FixedFusion(
        alpha=float(cfg.fusion.alpha),
        normalization_method=str(cfg.retrieval.score_normalization_method),
    )


def get_or_create_rrf_fusion(cfg: Any, rrf_fusion: RRFFusion | None) -> RRFFusion:
    """Return a reusable RRF fusion module."""
    return rrf_fusion or RRFFusion(k=int(cfg.fusion.rrf_k))


def _retrieve_bm25_for_sample(
    sample: dict[str, Any],
    topk: int,
    doc_text_cache: dict[str, list[str]],
    doc_retriever_cache: dict[str, BM25Retriever],
) -> tuple[dict[str, list], dict[str, int]]:
    """Run BM25 retrieval for one sample with doc-level OCR text and retriever caching."""
    doc_id = str(sample["doc_id"])
    stats = {
        "doc_text_cache_hits": 0,
        "doc_text_cache_misses": 0,
        "doc_retriever_cache_hits": 0,
        "doc_retriever_cache_misses": 0,
    }

    if doc_id in doc_text_cache:
        page_texts = doc_text_cache[doc_id]
        stats["doc_text_cache_hits"] += 1
    else:
        page_texts = [load_page_ocr_text(path) for path in sample.get("ocr_paths", [])]
        doc_text_cache[doc_id] = page_texts
        stats["doc_text_cache_misses"] += 1

    if doc_id in doc_retriever_cache:
        retriever = doc_retriever_cache[doc_id]
        stats["doc_retriever_cache_hits"] += 1
    else:
        retriever = BM25Retriever()
        page_ids = list(range(len(page_texts)))
        retriever.build_index(page_texts=page_texts, page_ids=page_ids)
        doc_retriever_cache[doc_id] = retriever
        stats["doc_retriever_cache_misses"] += 1

    return retriever.retrieve(sample["question"], topk=topk), stats


def _retrieve_visual_for_sample(
    cfg: Any,
    sample: dict[str, Any],
    topk: int,
    retriever: Any,
    indexed_docs: set[str],
) -> tuple[dict[str, list], dict[str, int]]:
    """Run visual retrieval for one sample with doc-level page-embedding caching."""
    doc_id = str(sample["doc_id"])
    image_paths = [str(path) for path in sample.get("image_paths", [])]
    stats = {"doc_visual_cache_hits": 0, "doc_visual_cache_misses": 0}

    if doc_id in indexed_docs:
        stats["doc_visual_cache_hits"] += 1
    else:
        page_ids = [_extract_page_idx_from_path(path) for path in image_paths]
        retriever.build_document_index(doc_id=doc_id, image_paths=image_paths, page_ids=page_ids)
        indexed_docs.add(doc_id)
        stats["doc_visual_cache_misses"] += 1

    return retriever.retrieve(sample["question"], doc_id=doc_id, topk=topk), stats


def _retrieve_visual_with_adaptive_coarse_for_sample(
    cfg: Any,
    sample: dict[str, Any],
    retriever: Any,
    indexed_docs: set[str],
    coarse_doc_text_cache: dict[str, list[str]],
    coarse_doc_retriever_cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Route visual candidates with BM25 then rerank the routed subset with ColQwen."""
    if not isinstance(retriever, ColQwenRetriever):
        raise TypeError("Adaptive coarse visual routing requires ColQwenRetriever.")

    doc_id = str(sample["doc_id"])
    image_paths = [str(path) for path in sample.get("image_paths", [])]
    route_result, route_stats = route_document_pages_with_adaptive_coarse(
        cfg=cfg,
        sample=sample,
        doc_text_cache=coarse_doc_text_cache,
        doc_retriever_cache=coarse_doc_retriever_cache,
        router_cfg=getattr(cfg, "retrieval_router", None),
        stats_prefix="visual",
        enabled_attr="enable_adaptive_coarse",
    )

    stats = {
        "doc_visual_cache_hits": 0,
        "doc_visual_cache_misses": 0,
        **route_stats,
    }

    if doc_id in indexed_docs:
        stats["doc_visual_cache_hits"] += 1
    else:
        page_ids = [_extract_page_idx_from_path(path) for path in image_paths]
        retriever.build_document_index(doc_id=doc_id, image_paths=image_paths, page_ids=page_ids)
        indexed_docs.add(doc_id)
        stats["doc_visual_cache_misses"] += 1

    candidate_page_ids = [int(page_id) for page_id in route_result.get("page_ids", [])]
    visual_result = retriever.retrieve_subset(
        sample["question"],
        doc_id=doc_id,
        candidate_page_ids=candidate_page_ids,
        topk=None,
    )
    visual_result["adaptive_coarse"] = route_result.get("metadata", {})
    visual_result["routed_page_ids"] = list(candidate_page_ids)
    add_adaptive_coarse_recall_stats(stats, candidate_page_ids, sample.get("evidence_pages", []), stats_prefix="visual")
    return visual_result, stats


def _retrieve_ocr_with_page_coarse_for_sample(
    cfg: Any,
    sample: dict[str, Any],
    topk: int,
    chunk_builder: OCRChunkBuilder,
    retriever: OCRBGEChunkRetriever,
    indexed_docs: set[str],
    reranker: OCRBGEChunkReranker | None = None,
    chunk_cache: dict[str, list[dict[str, Any]]] | None = None,
    coarse_doc_text_cache: dict[str, list[str]] | None = None,
    coarse_doc_retriever_cache: dict[str, Any] | None = None,
    coarse_topn: int | None = None,
    aggregation_strategy: str = "max",
    query_variant: str = "instruction_query",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run OCR page-level coarse routing, then chunk retrieval/rerank on the routed subset only."""
    coarse_doc_text_cache = coarse_doc_text_cache or {}
    coarse_doc_retriever_cache = coarse_doc_retriever_cache or {}
    route_result, route_stats = route_document_pages_with_adaptive_coarse(
        cfg=cfg,
        sample=sample,
        doc_text_cache=coarse_doc_text_cache,
        doc_retriever_cache=coarse_doc_retriever_cache,
        router_cfg=getattr(cfg, "ocr_router", None),
        stats_prefix="ocr",
        enabled_attr="enable_ocr_page_coarse",
    )
    routed_page_ids = [int(page_id) for page_id in route_result.get("page_ids", [])]
    subset_sample = filter_sample_to_page_ids(sample, routed_page_ids)
    subset_suffix = _build_subset_suffix(routed_page_ids)
    ocr_result, ocr_stats = _retrieve_ocr_bge_chunk_for_sample(
        cfg=cfg,
        sample=subset_sample,
        topk=topk,
        chunk_builder=chunk_builder,
        retriever=retriever,
        indexed_docs=indexed_docs,
        reranker=reranker,
        chunk_cache=chunk_cache,
        coarse_topn=coarse_topn,
        aggregation_strategy=aggregation_strategy,
        query_variant=query_variant,
        collect_chunk_stats=False,
        doc_key_suffix=subset_suffix,
    )
    stats = {**route_stats, **ocr_stats}
    add_adaptive_coarse_recall_stats(stats, routed_page_ids, sample.get("evidence_pages", []), stats_prefix="ocr")
    ocr_result["ocr_page_coarse"] = route_result.get("metadata", {})
    ocr_result["routed_page_ids"] = routed_page_ids
    return ocr_result, stats


def _retrieve_ocr_with_bm25_bge_reranker_for_sample(
    cfg: Any,
    sample: dict[str, Any],
    topk: int,
    retriever: OCRBGERetriever,
    indexed_docs: set[str],
    reranker: OCRBGEReranker | None = None,
    bm25_doc_text_cache: dict[str, list[str]] | None = None,
    bm25_doc_retriever_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the lightweight OCR page pipeline: BM25 coarse -> BGE-M3 -> page reranker."""
    return run_ocr_page_pipeline_for_sample(
        cfg=cfg,
        sample=sample,
        topk=topk,
        retriever=retriever,
        indexed_docs=indexed_docs,
        reranker=reranker,
        bm25_doc_text_cache=bm25_doc_text_cache,
        bm25_doc_retriever_cache=bm25_doc_retriever_cache,
    )


def _build_adaptive_coarse_metric_accumulator(cfg: Any, router_attr: str, stats_prefix: str) -> dict[str, Any]:
    """Create a fresh branch-specific adaptive coarse metrics accumulator from config."""
    router_cfg = getattr(cfg, router_attr, None)
    return {
        f"{stats_prefix}_num_pages_before_coarse": 0.0,
        f"{stats_prefix}_num_pages_after_coarse": 0.0,
        f"{stats_prefix}_num_bypass_samples": 0,
        f"{stats_prefix}_num_coarse_samples": 0,
        f"{stats_prefix}_coarse_page_text_cache_hits": 0,
        f"{stats_prefix}_coarse_page_text_cache_misses": 0,
        f"{stats_prefix}_coarse_index_cache_hits": 0,
        f"{stats_prefix}_coarse_index_cache_misses": 0,
        f"{stats_prefix}_coarse_method": str(getattr(router_cfg, "coarse_method", "bm25")),
        f"{stats_prefix}_bm25_coarse_enabled": bool(getattr(router_cfg, "enable_bm25_coarse", True)),
        f"{stats_prefix}_bypass_threshold": int(getattr(router_cfg, "bypass_threshold", 0) or 0),
        f"{stats_prefix}_coarse_topk": int(getattr(router_cfg, "coarse_topk", 0) or 0),
    }


def _accumulate_adaptive_coarse_metrics(accumulator: dict[str, Any], sample_stats: dict[str, Any], stats_prefix: str) -> None:
    """Add one sample's branch-specific adaptive coarse stats into an accumulator."""
    for key in (
        f"{stats_prefix}_num_pages_before_coarse",
        f"{stats_prefix}_num_pages_after_coarse",
        f"{stats_prefix}_num_bypass_samples",
        f"{stats_prefix}_num_coarse_samples",
        f"{stats_prefix}_coarse_page_text_cache_hits",
        f"{stats_prefix}_coarse_page_text_cache_misses",
        f"{stats_prefix}_coarse_index_cache_hits",
        f"{stats_prefix}_coarse_index_cache_misses",
        f"{stats_prefix}_coarse_recall@10",
        f"{stats_prefix}_coarse_recall@20",
        f"{stats_prefix}_coarse_recall@50",
    ):
        if key in sample_stats:
            accumulator[key] = accumulator.get(key, 0.0) + float(sample_stats[key])


def _build_ocr_page_pipeline_metric_accumulator() -> dict[str, Any]:
    """Create a fresh accumulator for OCR page-level semantic and rerank stages."""
    return {
        "ocr_bge_embedding_cache_hits": 0,
        "ocr_bge_embedding_cache_misses": 0,
        "ocr_num_pages_after_bge": 0.0,
        "ocr_rerank_calls": 0,
        "ocr_num_pages_after_rerank": 0.0,
    }


def _accumulate_ocr_page_pipeline_metrics(accumulator: dict[str, Any], sample_stats: dict[str, Any]) -> None:
    """Accumulate one sample's OCR page pipeline stats."""
    for key in (
        "ocr_bge_embedding_cache_hits",
        "ocr_bge_embedding_cache_misses",
        "ocr_num_pages_after_bge",
        "ocr_rerank_calls",
        "ocr_num_pages_after_rerank",
    ):
        if key in sample_stats:
            accumulator[key] = accumulator.get(key, 0.0) + float(sample_stats[key])


def summarize_ocr_page_pipeline_metrics(stats: dict[str, Any], num_samples: int) -> dict[str, Any]:
    """Convert accumulated OCR page pipeline counters into report-ready metrics."""
    denominator = float(max(num_samples, 1))
    summary = dict(stats)
    summary["avg_num_pages_after_ocr_bge"] = float(stats.get("ocr_num_pages_after_bge", 0.0)) / denominator
    summary["avg_num_pages_after_ocr_rerank"] = float(stats.get("ocr_num_pages_after_rerank", 0.0)) / denominator
    return summary


def add_ocr_bm25_metric_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add stable OCR BM25 stage metric aliases for the new page-level OCR route."""
    aliased = dict(metrics)
    if "avg_num_pages_before_ocr_bm25_coarse" in aliased:
        aliased["avg_num_pages_before_ocr_bm25"] = aliased["avg_num_pages_before_ocr_bm25_coarse"]
    if "avg_num_pages_after_ocr_bm25_coarse" in aliased:
        aliased["avg_num_pages_after_ocr_bm25"] = aliased["avg_num_pages_after_ocr_bm25_coarse"]
    return aliased


def _build_subset_suffix(page_ids: list[int]) -> str:
    """Create a stable cache-key suffix for one routed page subset."""
    if not page_ids:
        return "empty"
    return "subset_" + "_".join(str(int(page_id)) for page_id in sorted(set(page_ids)))


def _retrieve_ocr_bge_for_sample(
    cfg: Any,
    sample: dict[str, Any],
    topk: int,
    retriever: OCRBGERetriever,
    indexed_docs: set[str],
    reranker: OCRBGEReranker | None = None,
    coarse_topn: int | None = None,
    page_text_variant: str = "legacy_raw",
    query_variant: str = "raw_question",
    doc_text_cache: dict[str, list[str]] | None = None,
    return_debug: bool = False,
) -> tuple[dict[str, list], dict[str, int]] | tuple[dict[str, list], dict[str, int], dict[str, Any]]:
    """Run BGE-based OCR retrieval with optional reranking for one sample."""
    doc_id = str(sample["doc_id"])
    doc_key = f"{page_text_variant}:{doc_id}"
    stats = {
        "doc_ocr_bge_cache_hits": 0,
        "doc_ocr_bge_cache_misses": 0,
        "ocr_rerank_calls": 0,
    }
    debug_info = {
        "coarse_page_ids": [],
        "coarse_scores": [],
        "reranked_page_ids": [],
        "query_text": "",
        "doc_key": doc_key,
    }

    if doc_text_cache is None:
        doc_text_cache = {}

    if doc_key in indexed_docs:
        stats["doc_ocr_bge_cache_hits"] += 1
    else:
        page_texts = _build_ocr_bge_page_texts(sample, page_text_variant, doc_text_cache)
        page_ids = [_extract_page_idx_from_path(path) for path in sample.get("ocr_paths", [])[: len(page_texts)]]
        retriever.build_document_index(doc_id=doc_key, page_texts=page_texts, page_ids=page_ids)
        indexed_docs.add(doc_key)
        stats["doc_ocr_bge_cache_misses"] += 1

    retrieve_topk = int(coarse_topn or topk)
    query_text = format_bge_query(sample["question"], variant=query_variant)
    debug_info["query_text"] = query_text
    dense_result = retriever.retrieve(query_text, doc_id=doc_key, topk=retrieve_topk)
    debug_info["coarse_page_ids"] = [int(page_id) for page_id in dense_result.get("page_ids", [])]
    debug_info["coarse_scores"] = [float(score) for score in dense_result.get("scores", [])]
    if reranker is None:
        result = {
            "page_ids": dense_result["page_ids"][:topk],
            "scores": dense_result["scores"][:topk],
            "ranks": list(range(1, min(len(dense_result["page_ids"]), topk) + 1)),
        }
        if return_debug:
            debug_info["reranked_page_ids"] = [int(page_id) for page_id in result["page_ids"]]
            return result, stats, debug_info
        return result, stats

    stats["ocr_rerank_calls"] += 1
    page_texts = retriever.get_document_page_texts(doc_key)
    page_id_to_text = {
        int(page_id): str(text)
        for page_id, text in zip(retriever.index[doc_key]["page_ids"], page_texts)
    }
    candidates = [
        {
            "page_id": int(page_id),
            "text": page_id_to_text.get(int(page_id), ""),
            "score": float(score),
        }
        for page_id, score in zip(dense_result.get("page_ids", []), dense_result.get("scores", []))
    ]
    reranked = reranker.rerank(query_text, candidates, topk=topk)
    debug_info["reranked_page_ids"] = [int(page_id) for page_id in reranked.get("page_ids", [])]
    if return_debug:
        return reranked, stats, debug_info
    return reranked, stats


def _build_ocr_bge_page_texts(
    sample: dict[str, Any],
    page_text_variant: str,
    doc_text_cache: dict[str, list[str]],
) -> list[str]:
    """Build or reuse one document's OCR page texts under a specific text variant."""
    doc_id = str(sample["doc_id"])
    doc_key = f"{page_text_variant}:{doc_id}"
    if doc_key in doc_text_cache:
        return doc_text_cache[doc_key]

    if page_text_variant == "legacy_raw":
        page_texts = [load_page_ocr_text(path) for path in sample.get("ocr_paths", [])]
    else:
        page_texts = [load_bge_page_text(path, variant=page_text_variant) for path in sample.get("ocr_paths", [])]
    doc_text_cache[doc_key] = page_texts
    return page_texts


def _compute_recall_at_k_from_page_ids(pred_page_ids: list[int], evidence_pages: list[int], k: int) -> float:
    """Compute a single-sample hit-style recall for one ranked page-id list."""
    evidence_set = {int(page) for page in evidence_pages}
    if not evidence_set:
        return 0.0
    return 1.0 if any(int(page_id) in evidence_set for page_id in pred_page_ids[:k]) else 0.0


def _update_length_summary(aggregate: dict[str, float], page_texts: list[str], variant: str) -> None:
    """Accumulate page-text length statistics across documents."""
    summary = summarize_page_text_lengths(page_texts, variant)
    num_pages = max(len(page_texts), 1)
    aggregate["num_docs"] += 1.0
    aggregate["num_pages"] += float(num_pages)
    aggregate["total_tokens"] += float(summary["avg_token_count"]) * num_pages
    aggregate["max_tokens"] = max(float(aggregate["max_tokens"]), float(summary["max_token_count"]))
    aggregate["truncated_pages"] += float(summary["truncated_ratio"]) * num_pages


def _finalize_length_summary(aggregate: dict[str, float], variant: str) -> dict[str, float | str]:
    """Finalize aggregated page-text length statistics for JSON output."""
    num_pages = max(float(aggregate["num_pages"]), 1.0)
    return {
        "page_text_variant": variant,
        "avg_token_count": float(aggregate["total_tokens"] / num_pages),
        "max_token_count": float(aggregate["max_tokens"]),
        "truncated_ratio": float(aggregate["truncated_pages"] / num_pages),
        "num_docs": int(aggregate["num_docs"]),
        "num_pages": int(aggregate["num_pages"]),
    }


def _build_ocr_chunks_for_sample(
    sample: dict[str, Any],
    chunk_builder: OCRChunkBuilder,
    chunk_cache: dict[str, list[dict[str, Any]]],
    cache_key_suffix: str | None = None,
) -> list[dict[str, Any]]:
    """Build or reuse chunked OCR text for one document."""
    doc_id = str(sample["doc_id"])
    suffix = f":{cache_key_suffix}" if cache_key_suffix else ""
    chunk_key = f"{chunk_builder.page_text_variant}:{chunk_builder.chunk_size}:{chunk_builder.chunk_stride}:{doc_id}{suffix}"
    if chunk_key not in chunk_cache:
        chunk_cache[chunk_key] = chunk_builder.build_document_chunks(doc_id=doc_id, ocr_paths=[str(path) for path in sample.get("ocr_paths", [])])
    return chunk_cache[chunk_key]


def _compute_percentile(values: list[int], percentile: float) -> float:
    """Compute a simple percentile from integer values without external deps."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((percentile / 100.0) * (len(sorted_values) - 1)))
    index = max(0, min(index, len(sorted_values) - 1))
    return float(sorted_values[index])


def _summarize_chunk_stats(
    doc_id: str,
    chunks: list[dict[str, Any]],
    token_counter: OCRBGERetriever,
    collect_debug: bool = False,
) -> dict[str, Any]:
    """Summarize chunk-count and chunk-length statistics with explicit units."""
    if not chunks:
        return {
            "avg_chunks_per_page": 0.0,
            "avg_char_count_per_chunk": 0.0,
            "avg_word_count_per_chunk": 0.0,
            "avg_model_token_count_per_chunk": 0.0,
            "num_pages": 0,
            "num_chunks": 0,
            "total_char_count": 0,
            "total_word_count": 0,
            "total_model_token_count": 0,
            "max_char_count_per_chunk": 0.0,
            "max_word_count_per_chunk": 0.0,
            "max_model_token_count_per_chunk": 0.0,
            "chunk_char_count_p50": 0.0,
            "chunk_char_count_p90": 0.0,
            "chunk_char_count_p99": 0.0,
            "chunk_word_count_p50": 0.0,
            "chunk_word_count_p90": 0.0,
            "chunk_word_count_p99": 0.0,
            "chunk_model_token_count_p50": 0.0,
            "chunk_model_token_count_p90": 0.0,
            "chunk_model_token_count_p99": 0.0,
            "char_counts": [],
            "word_counts": [],
            "model_token_counts": [],
            "sample_rows": [],
            "abnormal_rows": [],
        }

    page_to_count: dict[int, int] = defaultdict(int)
    char_counts: list[int] = []
    word_counts: list[int] = []
    model_token_counts: list[int] = []
    sample_rows: list[dict[str, Any]] = []
    abnormal_rows: list[dict[str, Any]] = []
    sampled_pages: set[int] = set()

    for chunk in chunks:
        page_id = int(chunk["page_id"])
        text = str(chunk.get("chunk_text", ""))
        char_count = int(chunk.get("char_count", len(text)))
        word_count = int(chunk.get("word_count", len(text.split())))
        model_token_count = token_counter.count_model_tokens(text)
        chunk["char_count"] = char_count
        chunk["word_count"] = word_count
        chunk["model_token_count"] = model_token_count
        page_to_count[page_id] += 1
        char_counts.append(char_count)
        word_counts.append(word_count)
        model_token_counts.append(model_token_count)

        row = {
            "doc_id": doc_id,
            "page_id": page_id,
            "chunk_id": str(chunk["chunk_id"]),
            "char_count": char_count,
            "word_count": word_count,
            "model_token_count": model_token_count,
            "text_preview": text[:120],
        }
        if collect_debug and page_id not in sampled_pages:
            page_rows = [candidate for candidate in chunks if int(candidate["page_id"]) == page_id][:3]
            sample_rows.extend(
                {
                    "doc_id": doc_id,
                    "page_id": int(candidate["page_id"]),
                    "chunk_id": str(candidate["chunk_id"]),
                    "char_count": int(candidate.get("char_count", len(str(candidate.get("chunk_text", ""))))),
                    "word_count": int(candidate.get("word_count", len(str(candidate.get("chunk_text", "")).split()))),
                    "model_token_count": int(candidate.get("model_token_count", token_counter.count_model_tokens(str(candidate.get("chunk_text", ""))))),
                    "text_preview": str(candidate.get("chunk_text", ""))[:120],
                }
                for candidate in page_rows
            )
            sampled_pages.add(page_id)
        if model_token_count > 1024 or word_count > 500:
            abnormal_rows.append(row)

    return {
        "avg_chunks_per_page": float(sum(page_to_count.values()) / max(len(page_to_count), 1)),
        "avg_char_count_per_chunk": float(sum(char_counts) / max(len(char_counts), 1)),
        "avg_word_count_per_chunk": float(sum(word_counts) / max(len(word_counts), 1)),
        "avg_model_token_count_per_chunk": float(sum(model_token_counts) / max(len(model_token_counts), 1)),
        "num_pages": int(len(page_to_count)),
        "num_chunks": int(len(chunks)),
        "total_char_count": int(sum(char_counts)),
        "total_word_count": int(sum(word_counts)),
        "total_model_token_count": int(sum(model_token_counts)),
        "max_char_count_per_chunk": float(max(char_counts) if char_counts else 0.0),
        "max_word_count_per_chunk": float(max(word_counts) if word_counts else 0.0),
        "max_model_token_count_per_chunk": float(max(model_token_counts) if model_token_counts else 0.0),
        "chunk_char_count_p50": _compute_percentile(char_counts, 50),
        "chunk_char_count_p90": _compute_percentile(char_counts, 90),
        "chunk_char_count_p99": _compute_percentile(char_counts, 99),
        "chunk_word_count_p50": _compute_percentile(word_counts, 50),
        "chunk_word_count_p90": _compute_percentile(word_counts, 90),
        "chunk_word_count_p99": _compute_percentile(word_counts, 99),
        "chunk_model_token_count_p50": _compute_percentile(model_token_counts, 50),
        "chunk_model_token_count_p90": _compute_percentile(model_token_counts, 90),
        "chunk_model_token_count_p99": _compute_percentile(model_token_counts, 99),
        "char_counts": char_counts,
        "word_counts": word_counts,
        "model_token_counts": model_token_counts,
        "sample_rows": sample_rows,
        "abnormal_rows": abnormal_rows,
    }


def _compute_chunk_recall_at_k(chunk_page_ids: list[int], evidence_pages: list[int], k: int) -> float:
    """Compute whether any evidence page appears within the top-k chunk candidates."""
    return _compute_recall_at_k_from_page_ids(chunk_page_ids, evidence_pages, k)


def run_bm25_retrieval_on_dataset(dataset_path: str, topk: int = 10, k_values: list[int] | None = None) -> tuple[list[dict], dict]:
    """Run document-internal OCR-only BM25 retrieval over one processed dataset split."""
    logger = get_logger("bm25_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running BM25 retrieval on %s with %d samples and topk=%d", dataset_path, len(records), topk)
    predictions: list[dict] = []
    doc_text_cache: dict[str, list[str]] = {}
    doc_retriever_cache: dict[str, BM25Retriever] = {}
    doc_text_cache_hits = 0
    doc_text_cache_misses = 0
    doc_retriever_cache_hits = 0
    doc_retriever_cache_misses = 0

    for sample in tqdm(records, desc=f"BM25 {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        result, stats = _retrieve_bm25_for_sample(sample, topk, doc_text_cache, doc_retriever_cache)
        doc_text_cache_hits += stats["doc_text_cache_hits"]
        doc_text_cache_misses += stats["doc_text_cache_misses"]
        doc_retriever_cache_hits += stats["doc_retriever_cache_hits"]
        doc_retriever_cache_misses += stats["doc_retriever_cache_misses"]
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": result["page_ids"],
                "pred_scores": result["scores"],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    cache_stats = {
        "num_samples": len(records),
        "num_unique_docs": len(doc_text_cache),
        "doc_text_cache_hits": doc_text_cache_hits,
        "doc_text_cache_misses": doc_text_cache_misses,
        "doc_retriever_cache_hits": doc_retriever_cache_hits,
        "doc_retriever_cache_misses": doc_retriever_cache_misses,
    }
    metrics.update(cache_stats)
    logger.info("BM25 retrieval cache stats: %s", cache_stats)
    logger.info("BM25 retrieval metrics: %s", metrics)
    return predictions, metrics


def run_visual_retrieval_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run visual-only page retrieval within each document using a ColPali retriever."""
    logger = get_logger("visual_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running visual retrieval on %s with %d samples and topk=%d", dataset_path, len(records), topk)

    retriever = ColPaliRetriever(cfg, require_engine=True)
    predictions: list[dict] = []
    indexed_docs: set[str] = set()
    doc_visual_cache_hits = 0
    doc_visual_cache_misses = 0

    for sample in tqdm(records, desc=f"Visual {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        result, stats = _retrieve_visual_for_sample(cfg, sample, topk, retriever, indexed_docs)
        doc_visual_cache_hits += stats["doc_visual_cache_hits"]
        doc_visual_cache_misses += stats["doc_visual_cache_misses"]
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in result["page_ids"]],
                "pred_scores": result["scores"],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    cache_stats = {
        "num_samples": len(records),
        "num_unique_docs": len(indexed_docs),
        "doc_visual_cache_hits": doc_visual_cache_hits,
        "doc_visual_cache_misses": doc_visual_cache_misses,
    }
    metrics.update(cache_stats)
    logger.info("Visual retrieval cache stats: %s", cache_stats)
    logger.info("Visual retrieval metrics: %s", metrics)
    return predictions, metrics


def run_visual_colqwen_retrieval_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run visual-only page retrieval within each document using a ColQwen retriever."""
    logger = get_logger("visual_colqwen_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running visual ColQwen retrieval on %s with %d samples and topk=%d", dataset_path, len(records), topk)

    retriever = ColQwenRetriever(cfg)
    predictions: list[dict] = []
    indexed_docs: set[str] = set()
    doc_visual_cache_hits = 0
    doc_visual_cache_misses = 0

    for sample in tqdm(records, desc=f"VisualColQwen {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        result, stats = _retrieve_visual_for_sample(cfg, sample, topk, retriever, indexed_docs)
        doc_visual_cache_hits += stats["doc_visual_cache_hits"]
        doc_visual_cache_misses += stats["doc_visual_cache_misses"]
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in result["page_ids"]],
                "pred_scores": result["scores"],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    cache_stats = {
        "num_samples": len(records),
        "num_unique_docs": len(indexed_docs),
        "doc_visual_cache_hits": doc_visual_cache_hits,
        "doc_visual_cache_misses": doc_visual_cache_misses,
    }
    metrics.update(cache_stats)
    logger.info("Visual ColQwen retrieval cache stats: %s", cache_stats)
    logger.info("Visual ColQwen retrieval metrics: %s", metrics)
    return predictions, metrics


def run_visual_nemotron_energy_on_dataset(
    cfg: Any,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run local nemotron-colembed-vl visual-only retrieval on ViDoRe V3 Energy."""
    logger = get_logger("visual_nemotron_energy_retrieval")
    nemotron_cfg = cfg.visual_nemotron_energy
    dataset_path = str(nemotron_cfg.dataset_path)
    model_path = str(nemotron_cfg.model_name)
    max_samples = int(getattr(cfg.training, "max_val_samples", 0) or 0)
    samples, dataset_summary = load_vidore_energy_dataset(dataset_path, max_samples=max_samples)
    logger.info(
        "Running visual nemotron energy retrieval. model=%s dataset=%s device=%s samples=%d topk=%d",
        model_path,
        dataset_path,
        str(nemotron_cfg.device),
        len(samples),
        topk,
    )

    retriever = NemotronVisualRetriever(
        model_path=model_path,
        device=str(nemotron_cfg.device),
        local_files_only=bool(getattr(nemotron_cfg, "local_files_only", True)),
        cache_dir=str(getattr(nemotron_cfg, "cache_dir", "outputs/cache/visual_nemotron_energy")),
        enable_cache=bool(getattr(nemotron_cfg, "enable_embedding_cache", True)),
        batch_size=int(getattr(nemotron_cfg, "batch_size", 4)),
    )
    if not bool(getattr(nemotron_cfg, "enable_embedding_cache", True)):
        logger.info("Nemotron embedding cache disabled; all page embeddings will be recomputed.")
    predictions: list[dict] = []
    total_candidates = 0
    constant_score_samples = 0
    start_time = time.perf_counter()
    for sample_index, sample in enumerate(tqdm(samples, desc="VisualNemotronEnergy", unit="sample")):
        candidates = list(sample.get("candidates", []))
        total_candidates += len(candidates)
        query_text = _get_vidore_energy_query_text(sample)
        result = retriever.retrieve(query_text, candidates, topk=topk)
        pred_internal_page_ids = [int(page_id) for page_id in result["page_ids"]]
        pred_corpus_ids = _map_vidore_page_ids_to_corpus_ids(sample, pred_internal_page_ids)
        gold_corpus_ids = [str(corpus_id) for corpus_id in sample.get("gold_corpus_ids", sample.get("positive_corpus_ids", []))]
        _validate_vidore_corpus_id_space(sample, pred_corpus_ids, gold_corpus_ids)
        if len(pred_corpus_ids) != len(result["scores"]):
            raise RuntimeError(
                "ViDoRe Energy prediction alignment failed: pred_corpus_ids and pred_scores have different lengths. "
                f"query_id={sample.get('query_id', sample.get('qid', ''))}"
            )
        score_span = max(result["scores"]) - min(result["scores"]) if len(result["scores"]) > 1 else 0.0
        if len(result["scores"]) > 1 and abs(score_span) < 1e-8:
            constant_score_samples += 1
        if sample_index < 3:
            _log_visual_nemotron_debug_sample(
                logger=logger,
                sample_index=sample_index,
                sample=sample,
                query_text=query_text,
                result=result,
                pred_corpus_ids=pred_corpus_ids,
                gold_corpus_ids=gold_corpus_ids,
                candidates=candidates,
            )
        prediction = {
            "qid": sample["qid"],
            "query_id": sample.get("query_id", sample["qid"]),
            "doc_id": sample.get("doc_id", "vidore_v3_energy"),
            "query": query_text,
            "gold_corpus_ids": gold_corpus_ids,
            "pred_corpus_ids": pred_corpus_ids,
            "pred_scores": [float(score) for score in result["scores"]],
            "topk": topk,
        }
        _assert_vidore_prediction_uses_corpus_ids(prediction)
        predictions.append(prediction)
    retriever.save_cache()
    runtime_seconds = time.perf_counter() - start_time

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("gold_corpus_ids")]
    metrics = _evaluate_vidore_corpus_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics["NDCG@10"] = _ndcg_corpus_at_k(labeled_predictions, k=10) if labeled_predictions else 0.0
    metrics.update(
        {
            "num_samples": len(predictions),
            "num_labeled_samples": len(labeled_predictions),
            "num_unique_pages": int(dataset_summary.get("num_candidate_images", 0)),
            "avg_candidate_set_size": float(total_candidates / max(len(samples), 1)),
            "id_space": "corpus_id",
            "constant_score_samples": int(constant_score_samples),
            "constant_score_sample_ratio": float(constant_score_samples / max(len(predictions), 1)),
            "runtime_seconds": runtime_seconds,
            "model_path": model_path,
            "dataset_path": dataset_path,
            **retriever.cache_stats(),
        }
    )
    if predictions and constant_score_samples / max(len(predictions), 1) > 0.2:
        logger.warning(
            "Visual nemotron scores are nearly constant for %.2f%% of samples. Check model scoring/image input.",
            100.0 * constant_score_samples / max(len(predictions), 1),
        )
    logger.info("Visual nemotron energy dataset summary: %s", dataset_summary)
    logger.info("Visual nemotron retrieval metrics: %s", metrics)
    return predictions, metrics


def run_bm25_vidore_energy_on_dataset(
    cfg: Any,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run BM25-only retrieval on ViDoRe Energy in canonical corpus_id space."""
    logger = get_logger("bm25_vidore_energy_retrieval")
    bm25_cfg = getattr(cfg, "bm25_vidore_energy", {})
    dataset_path = str(getattr(bm25_cfg, "dataset_path", getattr(cfg.visual_nemotron_energy, "dataset_path", "")))
    max_samples = int(getattr(cfg.training, "max_val_samples", 0) or 0)
    samples, dataset_summary = load_vidore_energy_dataset(dataset_path, max_samples=max_samples)
    if not samples:
        raise RuntimeError(f"ViDoRe Energy BM25 evaluation found no samples: dataset_path={dataset_path}")

    candidates = list(samples[0].get("candidates", []))
    if not candidates:
        raise RuntimeError("ViDoRe Energy BM25 evaluation found no candidate pages.")
    page_ids = [int(candidate.get("page_id", index)) for index, candidate in enumerate(candidates)]
    page_texts = [str(candidate.get("ocr_text", "") or candidate.get("markdown", "") or "") for candidate in candidates]
    if not any(text.strip() for text in page_texts):
        raise RuntimeError("ViDoRe Energy BM25 evaluation found no usable page text in corpus.markdown / ocr_text.")

    retriever = BM25Retriever()
    retriever.build_index(page_texts=page_texts, page_ids=page_ids)
    configured_k_values = list(k_values) if k_values is not None else list(getattr(cfg.retrieval, "k_values", [1, 5, 10]))
    sweep_topks = sorted(
        {
            int(value)
            for value in (list(getattr(bm25_cfg, "sweep_topks", [])) + configured_k_values + [int(topk)])
            if int(value) > 0
        }
    )
    max_topk = min(max(sweep_topks or [int(topk)]), len(page_ids))
    logger.info(
        "Running BM25-only ViDoRe Energy retrieval. dataset=%s samples=%d max_topk=%d sweep_topks=%s",
        dataset_path,
        len(samples),
        max_topk,
        sweep_topks,
    )

    predictions: list[dict[str, Any]] = []
    total_candidates = 0
    start_time = time.perf_counter()
    for sample_index, sample in enumerate(tqdm(samples, desc="BM25ViDoReEnergy", unit="sample")):
        query_text = _get_vidore_energy_query_text(sample)
        result = retriever.retrieve(query_text, topk=max_topk)
        pred_internal_page_ids = [int(page_id) for page_id in result["page_ids"]]
        pred_corpus_ids = _map_vidore_page_ids_to_corpus_ids(sample, pred_internal_page_ids)
        gold_corpus_ids = [str(corpus_id) for corpus_id in sample.get("gold_corpus_ids", sample.get("positive_corpus_ids", []))]
        _validate_vidore_corpus_id_space(sample, pred_corpus_ids, gold_corpus_ids)
        prediction = {
            "qid": str(sample.get("query_id", sample.get("qid", ""))),
            "query_id": str(sample.get("query_id", sample.get("qid", ""))),
            "query": query_text,
            "gold_corpus_ids": gold_corpus_ids,
            "pred_corpus_ids": pred_corpus_ids,
            "pred_scores": [float(score) for score in result.get("scores", [])[: len(pred_corpus_ids)]],
            "topk": max_topk,
        }
        _assert_vidore_prediction_uses_corpus_ids(prediction)
        predictions.append(prediction)
        total_candidates += len(sample.get("candidates", []))
        if sample_index < 3:
            logger.info(
                "[DEBUG] BM25ViDoRe sample %d: query_id=%s gold=%s pred_top10=%s top10_scores=%s hit@1=%s hit@5=%s hit@10=%s",
                sample_index,
                sample.get("query_id", sample.get("qid", "")),
                gold_corpus_ids[:10],
                pred_corpus_ids[:10],
                prediction["pred_scores"][:10],
                bool(set(gold_corpus_ids) & set(pred_corpus_ids[:1])),
                bool(set(gold_corpus_ids) & set(pred_corpus_ids[:5])),
                bool(set(gold_corpus_ids) & set(pred_corpus_ids[:10])),
            )

    runtime_seconds = float(time.perf_counter() - start_time)
    labeled_predictions = [item for item in predictions if item.get("gold_corpus_ids")]
    metrics = _evaluate_vidore_corpus_retrieval(labeled_predictions, configured_k_values) if labeled_predictions else {}
    metrics["NDCG@10"] = _ndcg_corpus_at_k(labeled_predictions, k=10) if labeled_predictions else 0.0
    sweep_rows: list[dict[str, Any]] = []
    for cutoff in sweep_topks:
        truncated = [
            {
                **item,
                "pred_corpus_ids": list(item.get("pred_corpus_ids", [])[:cutoff]),
                "pred_scores": list(item.get("pred_scores", [])[:cutoff]),
            }
            for item in labeled_predictions
        ]
        cutoff_metrics = _evaluate_vidore_corpus_retrieval(truncated, [cutoff]) if truncated else {}
        mrr_at_k = sum(_vidore_corpus_mrr(item) for item in truncated) / len(truncated) if truncated else 0.0
        sweep_rows.append(
            {
                "topk": int(cutoff),
                f"Recall@{cutoff}": float(cutoff_metrics.get(f"Recall@{cutoff}", 0.0)),
                f"Hit@{cutoff}": float(cutoff_metrics.get(f"Hit@{cutoff}", 0.0)),
                "MRR": float(mrr_at_k),
            }
        )
    metrics.update(
        {
            "num_labeled_samples": len(labeled_predictions),
            "num_unique_pages": len(candidates),
            "avg_candidate_set_size": float(total_candidates / max(len(samples), 1)),
            "id_space": "corpus_id",
            "runtime_seconds": runtime_seconds,
            "dataset_path": dataset_path,
            "topk_runtime_note": (
                "BM25 standalone runtime is dominated by scoring the full corpus per query; returned topk mainly "
                "changes downstream retained candidates rather than BM25 scoring cost itself."
            ),
            "topk_sweep": sweep_rows,
            **dataset_summary,
        }
    )
    logger.info("BM25 ViDoRe Energy metrics: %s", metrics)
    return predictions, metrics


def run_bm25_600_nemotron_bge_vidore_energy_on_dataset(
    cfg: Any,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run BM25@600 -> parallel Nemotron/BGE -> candidate union -> fusion MLP on ViDoRe Energy."""
    logger = get_logger("bm25_600_nemotron_bge_vidore_energy_retrieval")
    branch_cfg = getattr(cfg, "bm25_600_nemotron_bge_vidore_energy", {})
    nemotron_cfg = cfg.visual_nemotron_energy
    dataset_path = str(getattr(branch_cfg, "dataset_path", getattr(nemotron_cfg, "dataset_path", "")))
    coarse_topk = int(getattr(branch_cfg, "coarse_topk", 600) or 600)
    if coarse_topk != 600:
        raise RuntimeError(
            f"BM25-600 Nemotron+BGE branch requires coarse_topk=600, but got coarse_topk={coarse_topk}."
        )
    final_merge_strategy = str(getattr(branch_cfg, "final_merge_strategy", "adaptive_fusion_mlp"))
    if final_merge_strategy != "adaptive_fusion_mlp":
        raise ValueError(
            f"Unsupported final_merge_strategy for bm25_600_nemotron_bge_vidore_energy: {final_merge_strategy}"
        )
    max_samples = int(getattr(cfg.training, "max_val_samples", 0) or 0)
    samples, dataset_summary = load_vidore_energy_dataset(dataset_path, max_samples=max_samples)
    if not samples:
        raise RuntimeError(f"ViDoRe Energy dual-branch evaluation found no samples: dataset_path={dataset_path}")

    corpus_candidates = list(samples[0].get("candidates", []))
    if not corpus_candidates:
        raise RuntimeError("ViDoRe Energy dual-branch evaluation found no candidate pages in the corpus pool.")
    corpus_page_ids = [int(candidate.get("page_id", index)) for index, candidate in enumerate(corpus_candidates)]
    corpus_texts = [str(candidate.get("ocr_text", "") or candidate.get("markdown", "") or "") for candidate in corpus_candidates]
    if not any(text.strip() for text in corpus_texts):
        raise RuntimeError("ViDoRe Energy dual-branch evaluation found no usable corpus.markdown / ocr_text text.")

    logger.info("Running BM25 coarse filter with top@600")
    logger.info(
        "Running BM25-600 Nemotron+BGE ViDoRe Energy retrieval. dataset=%s samples=%d visual_model=%s "
        "ocr_backend=%s final_merge_strategy=%s",
        dataset_path,
        len(samples),
        str(nemotron_cfg.model_name),
        "ocr_page_bm25_bge_rerank",
        final_merge_strategy,
    )
    logger.info("Running nemotron visual branch on BM25-filtered candidates")
    logger.info("Running BGE OCR branch on BM25-filtered candidates")
    logger.info("Final stage reuses candidate union + fusion MLP from adaptive_fusion_visual_nemotron_ocr_energy")
    fusion_checkpoint = _resolve_visual_nemotron_fusion_checkpoint(
        cfg,
        str(getattr(branch_cfg, "fusion_checkpoint_path", "") or ""),
    )
    fusion_model = _load_adaptive_ablation_mlp_model_from_checkpoint(
        cfg,
        variant="visual_nemotron_ocr_energy",
        checkpoint_path=fusion_checkpoint,
        checkpoint_subdir="adaptive_fusion_visual_nemotron_ocr_energy",
    )

    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(page_texts=corpus_texts, page_ids=corpus_page_ids)
    candidate_by_page_id = {int(candidate["page_id"]): candidate for candidate in corpus_candidates}
    page_text_by_id = {int(candidate["page_id"]): str(candidate.get("ocr_text", "") or candidate.get("markdown", "") or "") for candidate in corpus_candidates}

    visual_retriever = NemotronVisualRetriever(
        model_path=str(nemotron_cfg.model_name),
        device=str(nemotron_cfg.device),
        local_files_only=bool(getattr(nemotron_cfg, "local_files_only", True)),
        cache_dir=str(getattr(nemotron_cfg, "cache_dir", "outputs/cache/visual_nemotron_energy")),
        enable_cache=bool(getattr(nemotron_cfg, "enable_embedding_cache", True)),
        batch_size=int(getattr(nemotron_cfg, "batch_size", 4)),
    )
    if not bool(getattr(nemotron_cfg, "enable_embedding_cache", True)):
        logger.info("Nemotron embedding cache disabled; all page embeddings will be recomputed.")

    ocr_retriever = OCRBGERetriever(cfg, config_attr="ocr_semantic_retrieval")
    ocr_reranker = OCRBGEReranker(cfg, config_attr="ocr_reranker") if bool(getattr(cfg.ocr_reranker, "enable_bge_reranker", True)) else None
    ocr_doc_id = "__vidore_energy_full_corpus__"
    semantic_topk = int(getattr(cfg.ocr_semantic_retrieval, "semantic_topk", 30) or 30)
    rerank_topk = int(getattr(cfg.ocr_reranker, "rerank_topk", int(cfg.retrieval.topk)) or int(cfg.retrieval.topk))
    ocr_indexed_docs: set[str] = set()
    ocr_page_text_cache = {ocr_doc_id: list(corpus_texts)}
    question_encoder = QuestionEncoder()
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] = {}

    if k_values is None:
        k_values = [1, 5, 10]

    predictions: list[dict[str, Any]] = []
    total_candidates = 0
    coarse_hits = {10: 0.0, 50: 0.0, 100: 0.0, 600: 0.0}
    coarse_before_total = 0.0
    coarse_after_total = 0.0
    visual_runtime = 0.0
    ocr_runtime = 0.0
    fusion_runtime = 0.0
    cache_stats: dict[str, Any] = {
        "skipped_empty_candidates": 0,
        "visual_embedding_cache_hits": 0,
        "visual_embedding_cache_misses": 0,
    }
    start_time = time.perf_counter()

    for sample_index, sample in enumerate(tqdm(samples, desc="BM25_600_Nemotron_BGE_ViDoReEnergy", unit="sample")):
        query_text = _get_vidore_energy_query_text(sample)
        gold_corpus_ids = [str(corpus_id) for corpus_id in sample.get("gold_corpus_ids", sample.get("positive_corpus_ids", []))]
        coarse_before_total += float(len(corpus_candidates))
        bm25_result = bm25_retriever.retrieve(query_text, topk=min(coarse_topk, len(corpus_candidates)))
        bm25_page_ids = [int(page_id) for page_id in bm25_result.get("page_ids", [])]
        expected_pool_size = min(coarse_topk, len(corpus_candidates))
        if len(bm25_page_ids) != expected_pool_size:
            raise RuntimeError(
                f"BM25 candidate pool size mismatch for query_id={sample.get('query_id', sample.get('qid', ''))}: "
                f"expected={expected_pool_size} actual={len(bm25_page_ids)}"
            )
        coarse_after_total += float(len(bm25_page_ids))
        bm25_corpus_ids = _map_vidore_page_ids_to_corpus_ids(sample, bm25_page_ids)
        _validate_vidore_corpus_id_space(sample, bm25_corpus_ids, gold_corpus_ids)
        for cutoff in coarse_hits:
            if set(gold_corpus_ids) & set(bm25_corpus_ids[: min(cutoff, len(bm25_corpus_ids))]):
                coarse_hits[cutoff] += 1.0

        filtered_candidates = [candidate_by_page_id[page_id] for page_id in bm25_page_ids]
        total_candidates += len(filtered_candidates)

        visual_start = time.perf_counter()
        before_hits = int(visual_retriever.cache_hits)
        before_misses = int(visual_retriever.cache_misses)
        visual_result = visual_retriever.retrieve(query_text, filtered_candidates, topk=len(filtered_candidates))
        visual_runtime += float(time.perf_counter() - visual_start)
        cache_stats["visual_embedding_cache_hits"] += int(visual_retriever.cache_hits - before_hits)
        cache_stats["visual_embedding_cache_misses"] += int(visual_retriever.cache_misses - before_misses)
        visual_pred_corpus_ids = _map_vidore_page_ids_to_corpus_ids(sample, [int(page_id) for page_id in visual_result.get("page_ids", [])])

        ocr_start = time.perf_counter()
        ocr_subset_sample = {
            "doc_id": ocr_doc_id,
            "question": query_text,
            "query": query_text,
            "page_ids": list(corpus_page_ids),
            "page_texts": list(corpus_texts),
        }
        ocr_result, ocr_stats = run_ocr_page_pipeline_for_subset(
            cfg=cfg,
            sample=ocr_subset_sample,
            candidate_page_ids=bm25_page_ids,
            topk=topk,
            retriever=ocr_retriever,
            indexed_docs=ocr_indexed_docs,
            reranker=ocr_reranker,
            page_text_cache=ocr_page_text_cache,
            doc_id_override=ocr_doc_id,
            query_text=query_text,
            query_variant="raw_question",
            semantic_topk=min(semantic_topk, len(bm25_page_ids)),
            rerank_topk=min(rerank_topk, len(bm25_page_ids)),
        )
        ocr_runtime += float(time.perf_counter() - ocr_start)
        ocr_pred_corpus_ids = _map_vidore_page_ids_to_corpus_ids(sample, [int(page_id) for page_id in ocr_result.get("page_ids", [])])

        fusion_start = time.perf_counter()
        processed_sample = _vidore_energy_sample_to_processed_sample(sample)
        subset_sample = dict(processed_sample)
        subset_sample["page_ids"] = list(bm25_page_ids)
        subset_sample["page_texts"] = [page_text_by_id.get(int(page_id), "") for page_id in bm25_page_ids]
        subset_sample["page_id_to_corpus_id"] = {
            str(page_id): str(processed_sample.get("page_id_to_corpus_id", {}).get(str(page_id), ""))
            for page_id in bm25_page_ids
            if str(page_id) in processed_sample.get("page_id_to_corpus_id", {})
        }
        candidate_rows = build_candidate_features_visual_colqwen_ocr_chunk(
            subset_sample,
            ocr_result,
            visual_result,
            ocr_page_texts=corpus_texts,
            question_encoder=question_encoder,
            ocr_quality_cache=ocr_quality_cache,
        )
        if not candidate_rows:
            cache_stats["skipped_empty_candidates"] += 1
            continue
        ranked = fusion_model.rank_candidates(candidate_rows)
        final_page_ids = [int(page_id) for page_id in ranked.get("page_ids", [])]
        final_pred_corpus_ids = _map_vidore_page_ids_to_corpus_ids(sample, final_page_ids)
        final_scores = [float(score) for score in ranked.get("scores", [])[: len(final_pred_corpus_ids)]]
        fusion_runtime += float(time.perf_counter() - fusion_start)

        if final_scores and np.allclose(final_scores, final_scores[0]):
            logger.warning(
                "Dual-branch final scores are constant for query_id=%s; merge strategy=%s",
                sample.get("query_id", sample.get("qid", "")),
                final_merge_strategy,
            )

        prediction = {
            "qid": str(sample.get("query_id", sample.get("qid", ""))),
            "query_id": str(sample.get("query_id", sample.get("qid", ""))),
            "doc_id": "vidore_v3_energy",
            "query": query_text,
            "gold_corpus_ids": gold_corpus_ids,
            "pred_corpus_ids": final_pred_corpus_ids,
            "pred_scores": final_scores,
            "topk": int(topk),
        }
        _assert_vidore_prediction_uses_corpus_ids(prediction)
        predictions.append(prediction)

        if sample_index < 3:
            logger.info(
                "[DEBUG] DualBM25NemotronBGE sample %d: query_id=%s query_preview=%r gold=%s "
                "bm25_top10=%s visual_top10=%s ocr_top10=%s final_top10=%s final_hit@10=%s",
                sample_index,
                sample.get("query_id", sample.get("qid", "")),
                query_text[:200],
                gold_corpus_ids[:10],
                bm25_corpus_ids[:10],
                visual_pred_corpus_ids[:10],
                ocr_pred_corpus_ids[:10],
                final_pred_corpus_ids[:10],
                bool(set(gold_corpus_ids) & set(final_pred_corpus_ids[:10])),
            )

    runtime_seconds = float(time.perf_counter() - start_time)
    visual_retriever.save_cache()
    labeled_predictions = [item for item in predictions if item.get("gold_corpus_ids")]
    metrics = _evaluate_vidore_corpus_retrieval(labeled_predictions, list(k_values)) if labeled_predictions else {}
    metrics["NDCG@10"] = _ndcg_corpus_at_k(labeled_predictions, k=10) if labeled_predictions else 0.0
    metrics.update(
        {
            "num_labeled_samples": len(labeled_predictions),
            "num_unique_pages": len(corpus_candidates),
            "avg_candidate_set_size": float(total_candidates / max(len(samples), 1)),
            "id_space": "corpus_id",
            "runtime_seconds": runtime_seconds,
            "bm25_runtime_seconds": float(runtime_seconds - visual_runtime - ocr_runtime - fusion_runtime),
            "visual_runtime_seconds": float(visual_runtime),
            "ocr_runtime_seconds": float(ocr_runtime),
            "fusion_runtime_seconds": float(fusion_runtime),
            "avg_num_pages_before_coarse": float(coarse_before_total / max(len(samples), 1)),
            "avg_num_pages_after_coarse": float(coarse_after_total / max(len(samples), 1)),
            "coarse_topk": int(coarse_topk),
            "bm25_coarse_recall@10": float(coarse_hits[10] / max(len(samples), 1)),
            "bm25_coarse_recall@50": float(coarse_hits[50] / max(len(samples), 1)),
            "bm25_coarse_recall@100": float(coarse_hits[100] / max(len(samples), 1)),
            "bm25_coarse_recall@600": float(coarse_hits[600] / max(len(samples), 1)),
            "visual_model_path": str(nemotron_cfg.model_name),
            "dataset_path": dataset_path,
            "final_merge_strategy": final_merge_strategy,
            "fusion_checkpoint_path": fusion_checkpoint,
            **cache_stats,
            **visual_retriever.cache_stats(),
            **dataset_summary,
        }
    )
    logger.info("Merging dual-branch candidates by corpus_id and scoring them with fusion MLP")
    logger.info("BM25-600 Nemotron+BGE ViDoRe Energy metrics: %s", metrics)
    return predictions, metrics


def run_bm25_600_nemotron_bge_mpdocvqa_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run per-sample BM25@600 -> Nemotron/BGE -> RRF on MP-DocVQA canonical page ids."""
    logger = get_logger("bm25_600_nemotron_bge_mpdocvqa_retrieval")
    branch_cfg = getattr(cfg, "bm25_600_nemotron_bge_mpdocvqa", {})
    nemotron_cfg = cfg.visual_nemotron_energy
    coarse_topk = int(getattr(branch_cfg, "coarse_topk", 600) or 600)
    if coarse_topk != 600:
        raise RuntimeError(
            f"BM25-600 Nemotron+BGE MP-DocVQA branch requires coarse_topk=600, but got coarse_topk={coarse_topk}."
        )
    final_merge_strategy = str(getattr(branch_cfg, "final_merge_strategy", "rrf"))
    if final_merge_strategy != "rrf":
        raise ValueError(
            f"Unsupported final_merge_strategy for bm25_600_nemotron_bge_mpdocvqa: {final_merge_strategy}"
        )
    canonical_page_id_format = str(getattr(branch_cfg, "page_id_format", getattr(cfg.dataset, "page_id_template", "{doc_id}_p{page_index}")))
    records = load_jsonl(dataset_path)
    max_samples = int(getattr(cfg.training, "max_val_samples", 0) or 0)
    if max_samples > 0:
        records = records[:max_samples]
    if not records:
        raise RuntimeError(f"MP-DocVQA evaluation found no processed samples: dataset_path={dataset_path}")

    logger.info("Loaded MP-DocVQA dataset %s with %d samples", dataset_path, len(records))
    logger.info("Using canonical page id format: %s", canonical_page_id_format)
    logger.info("Running BM25 coarse filter with top@600")
    logger.info(
        "Running BM25-600 Nemotron+BGE MP-DocVQA retrieval. dataset=%s samples=%d visual_model=%s ocr_backend=%s final_merge_strategy=%s",
        dataset_path,
        len(records),
        str(nemotron_cfg.model_name),
        "ocr_page_bm25_bge_rerank",
        final_merge_strategy,
    )

    visual_retriever = NemotronVisualRetriever(
        model_path=str(nemotron_cfg.model_name),
        device=str(nemotron_cfg.device),
        local_files_only=bool(getattr(nemotron_cfg, "local_files_only", True)),
        cache_dir=str(getattr(nemotron_cfg, "cache_dir", "outputs/cache/visual_nemotron_energy")),
        enable_cache=bool(getattr(nemotron_cfg, "enable_embedding_cache", True)),
        batch_size=int(getattr(nemotron_cfg, "batch_size", 4)),
    )
    ocr_retriever = OCRBGERetriever(cfg, config_attr="ocr_semantic_retrieval")
    ocr_reranker = OCRBGEReranker(cfg, config_attr="ocr_reranker") if bool(getattr(cfg.ocr_reranker, "enable_bge_reranker", True)) else None
    ocr_indexed_docs: set[str] = set()
    ocr_page_text_cache: dict[str, list[str]] = {}
    if k_values is None:
        k_values = [1, 5, 10]
    k_values = sorted({int(k) for k in list(k_values) + [20, 50] if int(k) > 0})
    rrf_k = int(getattr(branch_cfg, "rrf_k", getattr(cfg.fusion, "rrf_k", 60)) or 60)

    predictions: list[dict[str, Any]] = []
    runtime_start = time.perf_counter()
    total_candidates = 0
    coarse_before_total = 0.0
    coarse_after_total = 0.0
    coarse_hits = {10: 0.0, 50: 0.0, 100: 0.0, 600: 0.0}
    visual_runtime = 0.0
    ocr_runtime = 0.0
    merge_runtime = 0.0
    cache_stats: dict[str, Any] = {
        "visual_embedding_cache_hits": 0,
        "visual_embedding_cache_misses": 0,
        "ocr_bge_embedding_cache_hits": 0,
        "ocr_bge_embedding_cache_misses": 0,
        "ocr_rerank_calls": 0,
        "skipped_empty_candidates": 0,
    }

    for sample_index, sample in enumerate(tqdm(records, desc="BM25_600_Nemotron_BGE_MPDocVQA", unit="sample")):
        candidates, page_texts, gold_page_ids, page_id_to_canonical = _build_mpdocvqa_candidate_pool(cfg, sample)
        total_candidates += len(candidates)
        coarse_before_total += float(len(candidates))
        bm25_retriever = BM25Retriever()
        bm25_retriever.build_index(page_texts=page_texts, page_ids=list(range(len(candidates))))
        bm25_result = bm25_retriever.retrieve(str(sample["question"]), topk=min(coarse_topk, len(candidates)))
        bm25_page_ids = [int(page_id) for page_id in bm25_result.get("page_ids", [])]
        expected_pool_size = min(coarse_topk, len(candidates))
        if len(candidates) > coarse_topk and len(bm25_page_ids) != coarse_topk:
            raise RuntimeError(
                f"BM25 candidate pool size mismatch for qid={sample.get('qid', '')}: expected={coarse_topk} actual={len(bm25_page_ids)}"
            )
        coarse_after_total += float(len(bm25_page_ids))
        bm25_pred_page_ids = [page_id_to_canonical[int(page_id)] for page_id in bm25_page_ids]
        _validate_mpdocvqa_page_id_space(sample, bm25_pred_page_ids, gold_page_ids, list(page_id_to_canonical.values()))
        for cutoff in coarse_hits:
            if set(gold_page_ids) & set(bm25_pred_page_ids[: min(cutoff, len(bm25_pred_page_ids))]):
                coarse_hits[cutoff] += 1.0

        filtered_candidates = [candidates[page_id] for page_id in bm25_page_ids]
        if not filtered_candidates:
            cache_stats["skipped_empty_candidates"] += 1
            continue

        visual_start = time.perf_counter()
        before_hits = int(visual_retriever.cache_hits)
        before_misses = int(visual_retriever.cache_misses)
        visual_result = visual_retriever.retrieve(str(sample["question"]), filtered_candidates, topk=len(filtered_candidates))
        visual_runtime += float(time.perf_counter() - visual_start)
        cache_stats["visual_embedding_cache_hits"] += int(visual_retriever.cache_hits - before_hits)
        cache_stats["visual_embedding_cache_misses"] += int(visual_retriever.cache_misses - before_misses)
        visual_pred_page_ids = [page_id_to_canonical[int(page_id)] for page_id in visual_result.get("page_ids", [])]

        ocr_start = time.perf_counter()
        ocr_sample = dict(sample)
        ocr_sample["page_ids"] = list(range(len(candidates)))
        ocr_sample["page_texts"] = list(page_texts)
        ocr_result, ocr_stats = run_ocr_page_pipeline_for_subset(
            cfg=cfg,
            sample=ocr_sample,
            candidate_page_ids=bm25_page_ids,
            topk=topk,
            retriever=ocr_retriever,
            indexed_docs=ocr_indexed_docs,
            reranker=ocr_reranker,
            page_text_cache=ocr_page_text_cache,
            query_text=str(sample["question"]),
            query_variant="raw_question",
            semantic_topk=min(int(getattr(cfg.ocr_semantic_retrieval, "semantic_topk", 30) or 30), len(bm25_page_ids)),
            rerank_topk=min(int(getattr(cfg.ocr_reranker, "rerank_topk", topk) or topk), len(bm25_page_ids)),
        )
        ocr_runtime += float(time.perf_counter() - ocr_start)
        cache_stats["ocr_bge_embedding_cache_hits"] += int(ocr_stats.get("ocr_bge_embedding_cache_hits", 0))
        cache_stats["ocr_bge_embedding_cache_misses"] += int(ocr_stats.get("ocr_bge_embedding_cache_misses", 0))
        cache_stats["ocr_rerank_calls"] += int(ocr_stats.get("ocr_rerank_calls", 0))
        ocr_pred_page_ids = [page_id_to_canonical[int(page_id)] for page_id in ocr_result.get("page_ids", [])]

        merge_start = time.perf_counter()
        final_pred_page_ids, final_scores = _merge_page_rankings_with_rrf(
            visual_pred_page_ids=visual_pred_page_ids,
            visual_scores=[float(score) for score in visual_result.get("scores", [])],
            ocr_pred_page_ids=ocr_pred_page_ids,
            ocr_scores=[float(score) for score in ocr_result.get("scores", [])],
            rrf_k=rrf_k,
        )
        merge_runtime += float(time.perf_counter() - merge_start)
        _validate_mpdocvqa_page_id_space(sample, final_pred_page_ids, gold_page_ids, list(page_id_to_canonical.values()))
        if final_scores and np.allclose(final_scores, final_scores[0]):
            logger.warning("MP-DocVQA final merged scores are constant for qid=%s", sample.get("qid", ""))

        prediction = {
            "qid": str(sample.get("qid", "")),
            "doc_id": str(sample.get("doc_id", "")),
            "question": str(sample.get("question", "")),
            "gold_page_ids": list(gold_page_ids),
            "pred_page_ids": list(final_pred_page_ids[: max(k_values)]),
            "pred_scores": [float(score) for score in final_scores[: max(k_values)]],
            "topk": int(topk),
        }
        predictions.append(prediction)

        if sample_index < 3:
            logger.info(
                "[DEBUG] DualBM25NemotronBGEMPDocVQA sample %d: qid=%s doc_id=%s question_preview=%r gold=%s bm25_top10=%s visual_top10=%s ocr_top10=%s final_top10=%s final_hit@10=%s",
                sample_index,
                sample.get("qid", ""),
                sample.get("doc_id", ""),
                str(sample.get("question", ""))[:200],
                gold_page_ids[:10],
                bm25_pred_page_ids[:10],
                visual_pred_page_ids[:10],
                ocr_pred_page_ids[:10],
                final_pred_page_ids[:10],
                bool(set(gold_page_ids) & set(final_pred_page_ids[:10])),
            )

    visual_retriever.save_cache()
    total_runtime = float(time.perf_counter() - runtime_start)
    labeled_predictions = [item for item in predictions if item.get("gold_page_ids")]
    metrics = _evaluate_mpdocvqa_page_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics["nDCG@5"] = _ndcg_page_id_at_k(labeled_predictions, k=5) if labeled_predictions else 0.0
    metrics["nDCG@10"] = _ndcg_page_id_at_k(labeled_predictions, k=10) if labeled_predictions else 0.0
    metrics["nDCG@20"] = _ndcg_page_id_at_k(labeled_predictions, k=20) if labeled_predictions else 0.0
    metrics.update(
        {
            "num_labeled_samples": len(labeled_predictions),
            "num_unique_docs": len({str(item.get("doc_id", "")) for item in records}),
            "num_unique_pages": int(sum(len(item.get("image_paths", [])) for item in records)),
            "avg_candidate_set_size": float(total_candidates / max(len(records), 1)),
            "id_space": canonical_page_id_format,
            "id_space_description": "canonical MP-DocVQA page ids derived from dataset.page_id_template",
            "runtime_seconds": total_runtime,
            "bm25_runtime_seconds": max(0.0, total_runtime - visual_runtime - ocr_runtime - merge_runtime),
            "visual_runtime_seconds": float(visual_runtime),
            "ocr_runtime_seconds": float(ocr_runtime),
            "merge_runtime_seconds": float(merge_runtime),
            "avg_num_pages_before_coarse": float(coarse_before_total / max(len(records), 1)),
            "avg_num_pages_after_coarse": float(coarse_after_total / max(len(records), 1)),
            "coarse_topk": int(coarse_topk),
            "bm25_coarse_recall@10": float(coarse_hits[10] / max(len(records), 1)),
            "bm25_coarse_recall@50": float(coarse_hits[50] / max(len(records), 1)),
            "bm25_coarse_recall@100": float(coarse_hits[100] / max(len(records), 1)),
            "bm25_coarse_recall@600": float(coarse_hits[600] / max(len(records), 1)),
            "dataset_path": dataset_path,
            "visual_model_path": str(nemotron_cfg.model_name),
            "ocr_backend": "ocr_page_bm25_bge_rerank",
            "final_merge_strategy": final_merge_strategy,
            "rrf_k": int(rrf_k),
            **cache_stats,
        }
    )
    logger.info("Merging dual-branch candidates by canonical page_id")
    logger.info("BM25-600 Nemotron+BGE MP-DocVQA metrics: %s", metrics)
    return predictions, metrics


def run_visual_nemotron_mpdocvqa_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run pure visual-only Nemotron retrieval on MP-DocVQA canonical page ids."""
    logger = get_logger("visual_nemotron_mpdocvqa_retrieval")
    branch_cfg = getattr(cfg, "visual_nemotron_mpdocvqa", {})
    nemotron_cfg = cfg.visual_nemotron_energy
    canonical_page_id_format = str(getattr(branch_cfg, "page_id_format", getattr(cfg.dataset, "page_id_template", "{doc_id}_p{page_index}")))
    records = load_jsonl(dataset_path)
    max_samples = int(getattr(cfg.training, "max_val_samples", 0) or 0)
    if max_samples > 0:
        records = records[:max_samples]
    if not records:
        raise RuntimeError(f"MP-DocVQA visual-only evaluation found no processed samples: dataset_path={dataset_path}")

    logger.info("Loaded MP-DocVQA dataset %s with %d samples", dataset_path, len(records))
    logger.info("Using canonical page id format: %s", canonical_page_id_format)
    logger.info(
        "Running visual-only nemotron retrieval on MP-DocVQA. model=%s device=%s samples=%d",
        str(nemotron_cfg.model_name),
        str(nemotron_cfg.device),
        len(records),
    )

    visual_retriever = NemotronVisualRetriever(
        model_path=str(nemotron_cfg.model_name),
        device=str(nemotron_cfg.device),
        local_files_only=bool(getattr(nemotron_cfg, "local_files_only", True)),
        cache_dir=str(getattr(nemotron_cfg, "cache_dir", "outputs/cache/visual_nemotron_energy")),
        enable_cache=bool(getattr(nemotron_cfg, "enable_embedding_cache", True)),
        batch_size=int(getattr(nemotron_cfg, "batch_size", 4)),
    )

    if k_values is None:
        k_values = [1, 5, 10]
    k_values = sorted({int(k) for k in list(k_values) + [20] if int(k) > 0})

    predictions: list[dict[str, Any]] = []
    runtime_start = time.perf_counter()
    total_candidates = 0
    cache_stats: dict[str, Any] = {
        "visual_embedding_cache_hits": 0,
        "visual_embedding_cache_misses": 0,
        "constant_score_samples": 0,
    }

    for sample_index, sample in enumerate(tqdm(records, desc="VisualNemotronMPDocVQA", unit="sample")):
        candidates, _page_texts, gold_page_ids, page_id_to_canonical = _build_mpdocvqa_candidate_pool(cfg, sample)
        total_candidates += len(candidates)
        before_hits = int(visual_retriever.cache_hits)
        before_misses = int(visual_retriever.cache_misses)
        result = visual_retriever.retrieve(str(sample["question"]), candidates, topk=len(candidates))
        cache_stats["visual_embedding_cache_hits"] += int(visual_retriever.cache_hits - before_hits)
        cache_stats["visual_embedding_cache_misses"] += int(visual_retriever.cache_misses - before_misses)

        pred_page_ids = [page_id_to_canonical[int(page_id)] for page_id in result.get("page_ids", [])]
        pred_scores = [float(score) for score in result.get("scores", [])[: len(pred_page_ids)]]
        _validate_mpdocvqa_page_id_space(sample, pred_page_ids, gold_page_ids, list(page_id_to_canonical.values()))
        if pred_scores and np.allclose(pred_scores, pred_scores[0]):
            cache_stats["constant_score_samples"] += 1
            logger.warning("MP-DocVQA visual-only scores are constant for qid=%s", sample.get("qid", ""))

        prediction = {
            "qid": str(sample.get("qid", "")),
            "doc_id": str(sample.get("doc_id", "")),
            "question": str(sample.get("question", "")),
            "gold_page_ids": list(gold_page_ids),
            "pred_page_ids": list(pred_page_ids[: max(k_values)]),
            "pred_scores": list(pred_scores[: max(k_values)]),
            "topk": int(topk),
        }
        predictions.append(prediction)

        if sample_index < 3:
            logger.info(
                "[DEBUG] VisualNemotronMPDocVQA sample %d: qid=%s doc_id=%s question_preview=%r gold=%s pred_top10=%s top10_scores=%s final_hit@10=%s",
                sample_index,
                sample.get("qid", ""),
                sample.get("doc_id", ""),
                str(sample.get("question", ""))[:200],
                gold_page_ids[:10],
                pred_page_ids[:10],
                pred_scores[:10],
                bool(set(gold_page_ids) & set(pred_page_ids[:10])),
            )

    visual_retriever.save_cache()
    total_runtime = float(time.perf_counter() - runtime_start)
    labeled_predictions = [item for item in predictions if item.get("gold_page_ids")]
    metrics = _evaluate_mpdocvqa_page_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics["nDCG@5"] = _ndcg_page_id_at_k(labeled_predictions, k=5) if labeled_predictions else 0.0
    metrics["nDCG@10"] = _ndcg_page_id_at_k(labeled_predictions, k=10) if labeled_predictions else 0.0
    metrics["nDCG@20"] = _ndcg_page_id_at_k(labeled_predictions, k=20) if labeled_predictions else 0.0
    metrics.update(
        {
            "num_labeled_samples": len(labeled_predictions),
            "num_unique_docs": len({str(item.get("doc_id", "")) for item in records}),
            "num_unique_pages": int(sum(len(item.get("image_paths", [])) for item in records)),
            "avg_candidate_set_size": float(total_candidates / max(len(records), 1)),
            "id_space": canonical_page_id_format,
            "id_space_description": "canonical MP-DocVQA page ids derived from dataset.page_id_template",
            "runtime_seconds": total_runtime,
            "model_path": str(nemotron_cfg.model_name),
            "dataset_path": dataset_path,
            **cache_stats,
            **visual_retriever.cache_stats(),
        }
    )
    logger.info("Visual nemotron MP-DocVQA metrics: %s", metrics)
    return predictions, metrics


def run_visual_nemotron_energy_sanity_check(
    cfg: Any,
    num_samples: int = 3,
    num_negatives: int = 9,
) -> tuple[list[dict], dict]:
    """Run a tiny deterministic corpus_id-space sanity check for Nemotron visual retrieval."""
    logger = get_logger("visual_nemotron_energy_sanity")
    nemotron_cfg = cfg.visual_nemotron_energy
    samples, dataset_summary = load_vidore_energy_dataset(str(nemotron_cfg.dataset_path), max_samples=0)
    retriever = NemotronVisualRetriever(
        model_path=str(nemotron_cfg.model_name),
        device=str(nemotron_cfg.device),
        local_files_only=bool(getattr(nemotron_cfg, "local_files_only", True)),
        cache_dir=str(getattr(nemotron_cfg, "cache_dir", "outputs/cache/visual_nemotron_energy")),
        enable_cache=bool(getattr(nemotron_cfg, "enable_embedding_cache", True)),
        batch_size=int(getattr(nemotron_cfg, "batch_size", 4)),
    )
    sanity_predictions: list[dict] = []
    for sample_index, sample in enumerate(samples[:num_samples]):
        query_text = _get_vidore_energy_query_text(sample)
        candidates = list(sample.get("candidates", []))
        positives = [str(corpus_id) for corpus_id in sample.get("positive_corpus_ids", [])]
        if not positives:
            raise RuntimeError(f"ViDoRe sanity sample has no positive_corpus_ids: query_id={sample.get('query_id')}")
        positive_id = positives[0]
        positive_candidates = [candidate for candidate in candidates if str(candidate.get("corpus_id")) == positive_id]
        negative_candidates = [
            candidate for candidate in candidates if str(candidate.get("corpus_id")) not in set(positives)
        ][:num_negatives]
        tiny_candidates = positive_candidates[:1] + negative_candidates
        if len(tiny_candidates) <= 1:
            raise RuntimeError(f"ViDoRe sanity sample has insufficient negatives: query_id={sample.get('query_id')}")
        result = retriever.retrieve(query_text, tiny_candidates, topk=len(tiny_candidates))
        pred_internal_page_ids = [int(page_id) for page_id in result["page_ids"]]
        pred_corpus_ids = _map_vidore_page_ids_to_corpus_ids(sample, pred_internal_page_ids)
        scores_by_corpus_id = {
            corpus_id: float(score) for corpus_id, score in zip(pred_corpus_ids, result["scores"], strict=False)
        }
        logger.info(
            "Nemotron sanity sample %d: query_id=%s query=%r gold_corpus_id=%s candidate_corpus_ids=%s ranked_corpus_ids=%s scores=%s",
            sample_index,
            sample.get("query_id", sample.get("qid", "")),
            query_text[:200],
            positive_id,
            [str(candidate.get("corpus_id")) for candidate in tiny_candidates],
            pred_corpus_ids,
            [float(score) for score in result["scores"]],
        )
        sanity_predictions.append(
            {
                "qid": sample.get("qid", sample.get("query_id", "")),
                "query": query_text,
                "gold_corpus_ids": [positive_id],
                "pred_corpus_ids": pred_corpus_ids,
                "pred_scores": [float(score) for score in result["scores"]],
                "scores_by_corpus_id": scores_by_corpus_id,
                "gold_in_top1": bool(pred_corpus_ids and pred_corpus_ids[0] == positive_id),
                "gold_in_top10": positive_id in pred_corpus_ids[:10],
            }
        )
    retriever.save_cache()
    metrics = _evaluate_vidore_corpus_retrieval(sanity_predictions, [1, 5, 10]) if sanity_predictions else {}
    metrics.update(
        {
            "stage_name": "visual_nemotron_energy_sanity",
            "id_space": "corpus_id",
            "num_sanity_samples": len(sanity_predictions),
            "num_negatives_per_sample": int(num_negatives),
            "dataset_path": str(nemotron_cfg.dataset_path),
            "model_path": str(nemotron_cfg.model_name),
            "dataset_summary": dataset_summary,
            **retriever.cache_stats(),
        }
    )
    logger.info("Visual nemotron sanity metrics: %s", metrics)
    return sanity_predictions, metrics


def run_adaptive_fusion_visual_nemotron_ocr_energy_on_dataset(
    cfg: Any,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run visual_nemotron + existing OCR page route + stable adaptive-fusion MLP on ViDoRe Energy."""
    logger = get_logger("adaptive_fusion_visual_nemotron_ocr_energy_retrieval")
    nemotron_cfg = cfg.visual_nemotron_energy
    dataset_path = str(nemotron_cfg.dataset_path)
    model_path = str(nemotron_cfg.model_name)
    max_samples = int(getattr(cfg.training, "max_val_samples", 0) or 0)
    samples, dataset_summary = load_vidore_energy_dataset(dataset_path, max_samples=max_samples)
    logger.info(
        "Running adaptive fusion visual_nemotron_ocr_energy. model=%s dataset=%s device=%s samples=%d topk=%d",
        model_path,
        dataset_path,
        str(nemotron_cfg.device),
        len(samples),
        topk,
    )
    logger.info(
        "OCR branch reused unchanged: backend=%s page_coarse=%s bm25_coarse=%s semantic_topk=%s rerank_topk=%s",
        "ocr_page_bm25_bge_rerank",
        bool(getattr(cfg.ocr_router, "enable_ocr_page_coarse", False)),
        bool(getattr(cfg.ocr_router, "enable_bm25_coarse", True)),
        int(getattr(cfg.ocr_semantic_retrieval, "semantic_topk", 0) or 0),
        int(getattr(cfg.ocr_reranker, "rerank_topk", 0) or 0),
    )

    fusion_checkpoint = _resolve_visual_nemotron_fusion_checkpoint(cfg, checkpoint_path)
    model = _load_adaptive_ablation_mlp_model_from_checkpoint(
        cfg,
        variant="visual_nemotron_ocr_energy",
        checkpoint_path=fusion_checkpoint,
        checkpoint_subdir="adaptive_fusion_visual_nemotron_ocr_energy",
    )
    question_encoder = QuestionEncoder()
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] = {}
    visual_retriever = NemotronVisualRetriever(
        model_path=model_path,
        device=str(nemotron_cfg.device),
        local_files_only=bool(getattr(nemotron_cfg, "local_files_only", True)),
        cache_dir=str(getattr(nemotron_cfg, "cache_dir", "outputs/cache/visual_nemotron_energy")),
        enable_cache=bool(getattr(nemotron_cfg, "enable_embedding_cache", True)),
        batch_size=int(getattr(nemotron_cfg, "batch_size", 4)),
    )
    if not bool(getattr(nemotron_cfg, "enable_embedding_cache", True)):
        logger.info("Nemotron embedding cache disabled; all page embeddings will be recomputed.")
    ocr_page_retriever = OCRBGERetriever(cfg, config_attr="ocr_semantic_retrieval")
    ocr_page_reranker = OCRBGEReranker(cfg, config_attr="ocr_reranker") if bool(getattr(cfg.ocr_reranker, "enable_bge_reranker", True)) else None
    ocr_indexed_docs: set[str] = set()
    ocr_coarse_doc_text_cache: dict[str, list[str]] = {}
    ocr_coarse_doc_retriever_cache: dict[str, Any] = {}
    ocr_coarse_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="ocr_router", stats_prefix="ocr_bm25")
    ocr_page_pipeline_stats = _build_ocr_page_pipeline_metric_accumulator()
    predictions: list[dict] = []
    cache_stats: dict[str, Any] = {
        "skipped_empty_candidates": 0,
        "visual_embedding_cache_hits": 0,
        "visual_embedding_cache_misses": 0,
    }
    total_candidates = 0
    start_time = time.perf_counter()

    for raw_sample in tqdm(samples, desc="FusionNemotronEnergy", unit="sample"):
        sample = _vidore_energy_sample_to_processed_sample(raw_sample)
        total_candidates += len(raw_sample.get("candidates", []))
        visual_result, visual_stats = _retrieve_visual_nemotron_energy_for_sample(
            sample=raw_sample,
            retriever=visual_retriever,
        )
        cache_stats["visual_embedding_cache_hits"] += int(visual_stats.get("embedding_cache_hits", 0))
        cache_stats["visual_embedding_cache_misses"] += int(visual_stats.get("embedding_cache_misses", 0))

        ocr_result, ocr_stats = _retrieve_ocr_with_bm25_bge_reranker_for_sample(
            cfg=cfg,
            sample=sample,
            topk=topk,
            retriever=ocr_page_retriever,
            indexed_docs=ocr_indexed_docs,
            reranker=ocr_page_reranker,
            bm25_doc_text_cache=ocr_coarse_doc_text_cache,
            bm25_doc_retriever_cache=ocr_coarse_doc_retriever_cache,
        )
        _accumulate_adaptive_coarse_metrics(ocr_coarse_stats, ocr_stats, stats_prefix="ocr_bm25")
        _accumulate_ocr_page_pipeline_metrics(ocr_page_pipeline_stats, ocr_stats)
        for key, value in ocr_stats.items():
            if key in ocr_coarse_stats or key in ocr_page_pipeline_stats:
                continue
            cache_stats[key] = cache_stats.get(key, 0) + value

        candidate_rows = build_candidate_features_visual_colqwen_ocr_chunk(
            sample,
            ocr_result,
            visual_result,
            ocr_page_texts=ocr_coarse_doc_text_cache.get(str(sample["doc_id"])),
            question_encoder=question_encoder,
            ocr_quality_cache=ocr_quality_cache,
        )
        if not candidate_rows:
            cache_stats["skipped_empty_candidates"] += 1
            continue
        ranked = model.rank_candidates(candidate_rows)
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": sample["doc_id"],
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in ranked["page_ids"][:topk]],
                "pred_scores": [float(score) for score in ranked["scores"][:topk]],
                "topk": topk,
            }
        )

    visual_retriever.save_cache()
    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics["NDCG@10"] = _ndcg_at_k(labeled_predictions, k=10) if labeled_predictions else 0.0
    metrics.update(
        {
            "stage_name": "adaptive_fusion_visual_nemotron_ocr_energy",
            "visual_backend": "nemotron-colembed-vl-8b-v2",
            "ocr_backend": "ocr_page_bm25_bge_rerank",
            "num_samples": len(predictions),
            "num_loaded_samples": len(samples),
            "num_labeled_samples": len(labeled_predictions),
            "num_unique_pages": int(dataset_summary.get("num_candidate_images", 0)),
            "avg_candidate_set_size": float(total_candidates / max(len(samples), 1)),
            "runtime_seconds": time.perf_counter() - start_time,
            "model_path": model_path,
            "dataset_path": dataset_path,
            "fusion_checkpoint_path": fusion_checkpoint,
            **cache_stats,
            **visual_retriever.cache_stats(),
        }
    )
    metrics.update(summarize_ocr_page_pipeline_metrics(ocr_page_pipeline_stats, len(samples)))
    metrics.update(add_ocr_bm25_metric_aliases(summarize_adaptive_coarse_stats(ocr_coarse_stats, len(samples), stats_prefix="ocr_bm25")))
    logger.info("ViDoRe Energy dataset summary: %s", dataset_summary)
    logger.info("Adaptive fusion visual_nemotron_ocr_energy metrics: %s", metrics)
    return predictions, metrics


def _get_vidore_energy_query_text(sample: dict[str, Any]) -> str:
    """Return canonical ViDoRe query text, prioritizing queries.query over legacy question."""
    query_text = str(sample.get("query", "") or "").strip()
    if not query_text:
        query_text = str(sample.get("question", "") or "").strip()
    if not query_text:
        raise RuntimeError(
            "ViDoRe Energy sample has empty query text. "
            f"query_id={sample.get('query_id', sample.get('qid', ''))}"
        )
    return query_text


def _map_vidore_page_ids_to_corpus_ids(sample: dict[str, Any], page_ids: list[int]) -> list[str]:
    """Map internal integer page_ids back to canonical ViDoRe corpus_id strings."""
    page_id_to_corpus_id = {str(key): str(value) for key, value in dict(sample.get("page_id_to_corpus_id", {})).items()}
    if not page_id_to_corpus_id:
        page_id_to_corpus_id = {
            str(candidate.get("page_id")): str(candidate.get("corpus_id"))
            for candidate in sample.get("candidates", [])
            if candidate.get("page_id") is not None and candidate.get("corpus_id") is not None
        }
    corpus_ids: list[str] = []
    missing_page_ids: list[int] = []
    for page_id in page_ids:
        corpus_id = page_id_to_corpus_id.get(str(int(page_id)))
        if corpus_id is None or corpus_id == "":
            missing_page_ids.append(int(page_id))
            continue
        corpus_ids.append(corpus_id)
    if missing_page_ids:
        raise RuntimeError(
            "ViDoRe Energy prediction id-space mismatch: internal page_id cannot be mapped to corpus_id. "
            f"query_id={sample.get('query_id', sample.get('qid', ''))} missing_page_ids={missing_page_ids[:10]}"
        )
    return corpus_ids


def _validate_vidore_corpus_id_space(
    sample: dict[str, Any],
    pred_corpus_ids: list[str],
    gold_corpus_ids: list[str],
) -> None:
    """Fail fast when ViDoRe predictions/gold labels are not in corpus_id space."""
    logger = get_logger("visual_nemotron_energy_retrieval")
    if not gold_corpus_ids:
        raise RuntimeError(f"ViDoRe Energy sample has no gold_corpus_ids: query_id={sample.get('query_id', sample.get('qid', ''))}")
    candidate_corpus_ids = {str(candidate.get("corpus_id")) for candidate in sample.get("candidates", [])}
    unknown_gold = [corpus_id for corpus_id in gold_corpus_ids if corpus_id not in candidate_corpus_ids]
    unknown_pred = [corpus_id for corpus_id in pred_corpus_ids if corpus_id not in candidate_corpus_ids]
    if unknown_gold:
        logger.warning(
            "ViDoRe Energy gold corpus_id not found in candidate pool. query_id=%s unknown_gold=%s",
            sample.get("query_id", sample.get("qid", "")),
            unknown_gold[:10],
        )
    if unknown_pred:
        raise RuntimeError(
            "ViDoRe Energy corpus_id alignment failed. "
            f"query_id={sample.get('query_id', sample.get('qid', ''))} "
            f"unknown_pred={unknown_pred[:10]}"
        )


def _assert_vidore_prediction_uses_corpus_ids(prediction: dict[str, Any]) -> None:
    """Forbid legacy page-id fields in ViDoRe Energy visual-only predictions."""
    forbidden_keys = {"pred_page_ids", "evidence_pages", "pred_internal_page_ids", "evidence_internal_page_ids"}
    present = sorted(key for key in forbidden_keys if key in prediction)
    if present:
        raise RuntimeError(f"Do not use page_id in ViDoRe evaluation. Forbidden prediction keys: {present}")
    if not prediction.get("pred_corpus_ids") or not prediction.get("gold_corpus_ids"):
        raise RuntimeError("ViDoRe Energy prediction must contain pred_corpus_ids and gold_corpus_ids.")


def _evaluate_vidore_corpus_retrieval(predictions: list[dict[str, Any]], k_values: list[int]) -> dict[str, float]:
    """Compute ViDoRe Energy metrics strictly in corpus_id space."""
    if not predictions:
        return {}
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = sum(
            float(bool(set(map(str, item["pred_corpus_ids"][:k])) & set(map(str, item["gold_corpus_ids"]))))
            for item in predictions
        ) / len(predictions)
        metrics[f"Hit@{k}"] = metrics[f"Recall@{k}"]
    metrics["MRR"] = sum(_vidore_corpus_mrr(item) for item in predictions) / len(predictions)
    metrics["PageAcc"] = sum(
        float(bool(item.get("pred_corpus_ids")) and str(item["pred_corpus_ids"][0]) in set(map(str, item["gold_corpus_ids"])))
        for item in predictions
    ) / len(predictions)
    metrics["num_samples"] = len(predictions)
    return metrics


def _vidore_corpus_mrr(item: dict[str, Any]) -> float:
    gold = set(map(str, item.get("gold_corpus_ids", [])))
    for rank, corpus_id in enumerate(item.get("pred_corpus_ids", []), start=1):
        if str(corpus_id) in gold:
            return 1.0 / float(rank)
    return 0.0


def _log_visual_nemotron_debug_sample(
    logger: Any,
    sample_index: int,
    sample: dict[str, Any],
    query_text: str,
    result: dict[str, Any],
    pred_corpus_ids: list[str],
    gold_corpus_ids: list[str],
    candidates: list[dict[str, Any]],
) -> None:
    """Branch-local diagnostics for the first few ViDoRe Nemotron visual samples."""
    candidate_example = candidates[0] if candidates else {}
    image_info = _describe_vidore_candidate_image(candidate_example)
    top_scores = [float(score) for score in result.get("scores", [])[:10]]
    logger.info(
        "VisualNemotron debug sample %d: query_id=%s query_preview=%r query_len=%d query_empty=%s",
        sample_index,
        sample.get("query_id", sample.get("qid", "")),
        query_text[:200],
        len(query_text),
        not bool(query_text),
    )
    logger.info(
        "[DEBUG] VisualNemotron ids sample %d: query_id=%s gold=%s pred=%s hit@1=%s hit@5=%s hit@10=%s "
        "top10_scores=%s internal_top10_page_ids=%s page_id_to_corpus_id_preview=%s",
        sample_index,
        sample.get("query_id", sample.get("qid", "")),
        gold_corpus_ids[:10],
        pred_corpus_ids[:10],
        bool(set(gold_corpus_ids) & set(pred_corpus_ids[:1])),
        bool(set(gold_corpus_ids) & set(pred_corpus_ids[:5])),
        bool(set(gold_corpus_ids) & set(pred_corpus_ids[:10])),
        top_scores,
        [int(page_id) for page_id in result.get("page_ids", [])[:10]],
        list(dict(sample.get("page_id_to_corpus_id", {})).items())[:5],
    )
    logger.info(
        "VisualNemotron debug image sample %d: corpus_id=%s page_id=%s image_source=%s image_info=%s",
        sample_index,
        candidate_example.get("corpus_id", ""),
        candidate_example.get("page_id", ""),
        candidate_example.get("image_source_type", ""),
        image_info,
    )


def _describe_vidore_candidate_image(candidate: dict[str, Any]) -> dict[str, Any]:
    image = candidate.get("image")
    if image is not None:
        return {
            "input_type": type(image).__name__,
            "is_pil": image.__class__.__module__.startswith("PIL."),
            "mode": str(getattr(image, "mode", "")),
            "size": tuple(getattr(image, "size", ())),
        }
    image_path = candidate.get("image_path")
    info: dict[str, Any] = {
        "input_type": "path",
        "image_path": str(image_path or ""),
        "path_exists": bool(Path(str(image_path)).exists()) if image_path else False,
    }
    if image_path:
        try:
            from PIL import Image

            with Image.open(str(image_path)) as loaded:
                info["mode"] = str(loaded.mode)
                info["size"] = tuple(loaded.size)
        except Exception as exc:
            info["open_error"] = str(exc)
    return info


def _retrieve_visual_nemotron_energy_for_sample(
    sample: dict[str, Any],
    retriever: NemotronVisualRetriever,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Run Nemotron visual retrieval over all local ViDoRe Energy candidate images."""
    candidates = list(sample.get("candidates", []))
    before_hits = retriever.cache_hits
    before_misses = retriever.cache_misses
    result = retriever.retrieve(_get_vidore_energy_query_text(sample), candidates, topk=len(candidates))
    result["visual_backend"] = "nemotron-colembed-vl-8b-v2"
    return result, {
        "embedding_cache_hits": int(retriever.cache_hits - before_hits),
        "embedding_cache_misses": int(retriever.cache_misses - before_misses),
    }


def _vidore_energy_sample_to_processed_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Map local ViDoRe Energy loader samples into the processed-sample shape used by OCR/fusion helpers."""
    candidates = list(sample.get("candidates", []))
    page_ids = [int(candidate.get("page_id", index)) for index, candidate in enumerate(candidates)]
    image_paths: list[str] = []
    for index, candidate in enumerate(candidates):
        image_path = candidate.get("image_path")
        if image_path:
            image_paths.append(str(image_path))
        else:
            image_paths.append(f"vidore_energy_{sample.get('qid', 'query')}_p{page_ids[index]}.jpg")
    qid = str(sample.get("qid", "vidore_energy"))
    page_texts = [str(candidate.get("ocr_text", "") or "") for candidate in candidates]
    return {
        "qid": qid,
        "doc_id": f"vidore_energy:{qid}",
        "query": _get_vidore_energy_query_text(sample),
        "question": _get_vidore_energy_query_text(sample),
        "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
        "positive_corpus_ids": [str(corpus_id) for corpus_id in sample.get("positive_corpus_ids", [])],
        "page_id_to_corpus_id": dict(sample.get("page_id_to_corpus_id", {})),
        "image_paths": image_paths,
        "ocr_paths": [],
        "page_texts": page_texts,
        "page_ids": page_ids,
    }


def _resolve_visual_nemotron_fusion_checkpoint(cfg: Any, checkpoint_path: str | None) -> str:
    """Resolve the fusion checkpoint for the isolated Nemotron visual experiment branch."""
    if checkpoint_path:
        return str(checkpoint_path)
    branch_checkpoint = (
        Path(cfg.paths.checkpoint_dir)
        / "adaptive_fusion_visual_nemotron_ocr_energy"
        / "adaptive_fusion_visual_nemotron_ocr_energy_best.pkl"
    )
    if branch_checkpoint.exists():
        return str(branch_checkpoint)
    nemotron_cfg = getattr(cfg, "visual_nemotron_energy", None)
    configured = str(getattr(nemotron_cfg, "fusion_checkpoint_path", "") or "")
    if configured:
        return configured
    experiment_checkpoint = str(getattr(getattr(cfg, "experiment", {}), "checkpoint_path", "") or "")
    if experiment_checkpoint:
        return experiment_checkpoint
    return str(
        Path(cfg.paths.checkpoint_dir)
        / "adaptive_fusion_visual_colqwen_ocr_chunk"
        / "adaptive_fusion_visual_colqwen_ocr_chunk_best.pkl"
    )


def _ndcg_at_k(predictions: list[dict], k: int) -> float:
    """Compute binary NDCG@K for page/image retrieval predictions."""
    if not predictions:
        return 0.0
    total = 0.0
    for item in predictions:
        evidence = {str(page) for page in item.get("evidence_pages", [])}
        if not evidence:
            continue
        dcg = 0.0
        for index, page_id in enumerate(item.get("pred_page_ids", [])[:k], start=1):
            if str(page_id) in evidence:
                dcg += 1.0 / np.log2(index + 1)
        ideal_hits = min(len(evidence), k)
        idcg = sum(1.0 / np.log2(index + 1) for index in range(1, ideal_hits + 1))
        total += dcg / max(idcg, 1e-8)
    return float(total / len(predictions))


def _ndcg_corpus_at_k(predictions: list[dict[str, Any]], k: int) -> float:
    """Compute binary NDCG@K for ViDoRe corpus_id predictions."""
    if not predictions:
        return 0.0
    total = 0.0
    for item in predictions:
        gold = set(map(str, item.get("gold_corpus_ids", [])))
        if not gold:
            continue
        dcg = 0.0
        for index, corpus_id in enumerate(item.get("pred_corpus_ids", [])[:k], start=1):
            if str(corpus_id) in gold:
                dcg += 1.0 / np.log2(index + 1)
        ideal_hits = min(len(gold), k)
        idcg = sum(1.0 / np.log2(index + 1) for index in range(1, ideal_hits + 1))
        total += dcg / max(idcg, 1e-8)
    return float(total / len(predictions))


def _merge_vidore_rankings_with_rrf(
    visual_pred_corpus_ids: list[str],
    visual_scores: list[float],
    ocr_pred_corpus_ids: list[str],
    ocr_scores: list[float],
    rrf_k: int,
) -> tuple[list[str], list[float]]:
    """Merge visual and OCR rankings in corpus_id space with reciprocal-rank fusion."""
    score_by_corpus_id: dict[str, float] = defaultdict(float)
    visual_rank_map = {str(corpus_id): rank for rank, corpus_id in enumerate(visual_pred_corpus_ids, start=1)}
    ocr_rank_map = {str(corpus_id): rank for rank, corpus_id in enumerate(ocr_pred_corpus_ids, start=1)}
    visual_score_map = {str(corpus_id): float(score) for corpus_id, score in zip(visual_pred_corpus_ids, visual_scores)}
    ocr_score_map = {str(corpus_id): float(score) for corpus_id, score in zip(ocr_pred_corpus_ids, ocr_scores)}

    for corpus_id, rank in visual_rank_map.items():
        score_by_corpus_id[corpus_id] += 1.0 / float(rrf_k + rank)
    for corpus_id, rank in ocr_rank_map.items():
        score_by_corpus_id[corpus_id] += 1.0 / float(rrf_k + rank)

    ranked_items = sorted(
        score_by_corpus_id.items(),
        key=lambda item: (
            -float(item[1]),
            visual_rank_map.get(item[0], 10**9),
            ocr_rank_map.get(item[0], 10**9),
            -float(visual_score_map.get(item[0], -1e9)),
            -float(ocr_score_map.get(item[0], -1e9)),
            str(item[0]),
        ),
    )
    return [str(corpus_id) for corpus_id, _ in ranked_items], [float(score) for _, score in ranked_items]


def _merge_page_rankings_with_rrf(
    visual_pred_page_ids: list[str],
    visual_scores: list[float],
    ocr_pred_page_ids: list[str],
    ocr_scores: list[float],
    rrf_k: int,
) -> tuple[list[str], list[float]]:
    """Merge two page-id rankings with reciprocal-rank fusion."""
    score_by_page_id: dict[str, float] = defaultdict(float)
    visual_rank_map = {str(page_id): rank for rank, page_id in enumerate(visual_pred_page_ids, start=1)}
    ocr_rank_map = {str(page_id): rank for rank, page_id in enumerate(ocr_pred_page_ids, start=1)}
    visual_score_map = {str(page_id): float(score) for page_id, score in zip(visual_pred_page_ids, visual_scores)}
    ocr_score_map = {str(page_id): float(score) for page_id, score in zip(ocr_pred_page_ids, ocr_scores)}
    for page_id, rank in visual_rank_map.items():
        score_by_page_id[page_id] += 1.0 / float(rrf_k + rank)
    for page_id, rank in ocr_rank_map.items():
        score_by_page_id[page_id] += 1.0 / float(rrf_k + rank)
    ranked_items = sorted(
        score_by_page_id.items(),
        key=lambda item: (
            -item[1],
            visual_rank_map.get(item[0], 10**9),
            ocr_rank_map.get(item[0], 10**9),
            -float(visual_score_map.get(item[0], -1e9)),
            -float(ocr_score_map.get(item[0], -1e9)),
            str(item[0]),
        ),
    )
    return [str(page_id) for page_id, _ in ranked_items], [float(score) for _, score in ranked_items]


def _canonical_mpdocvqa_page_id(cfg: Any, doc_id: str, page_index: int) -> str:
    """Normalize one MP-DocVQA page index into a stable page-id string."""
    template = str(getattr(getattr(cfg, "dataset", {}), "page_id_template", "{doc_id}_p{page_index}"))
    return template.format(doc_id=str(doc_id), page_index=int(page_index))


def _build_mpdocvqa_candidate_pool(
    cfg: Any,
    sample: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str], dict[int, str]]:
    """Build canonicalized page candidates for one MP-DocVQA sample."""
    question = str(sample.get("question", "") or "").strip()
    if not question:
        raise RuntimeError(f"MP-DocVQA sample has empty question: qid={sample.get('qid', '')}")
    doc_id = str(sample.get("doc_id", "")).strip()
    image_paths = [str(path) for path in sample.get("image_paths", [])]
    if not image_paths:
        raise RuntimeError(f"MP-DocVQA sample has no image_paths: qid={sample.get('qid', '')} doc_id={doc_id}")
    ocr_paths = list(sample.get("ocr_paths", []))
    page_texts = [load_page_ocr_text(str(path)) if index < len(ocr_paths) else "" for index, path in enumerate(image_paths)]
    page_id_to_canonical: dict[int, str] = {
        int(page_index): _canonical_mpdocvqa_page_id(cfg, doc_id, int(page_index))
        for page_index in range(len(image_paths))
    }
    candidates = [
        {
            "page_id": int(page_index),
            "canonical_page_id": page_id_to_canonical[int(page_index)],
            "doc_id": doc_id,
            "image_path": str(image_path),
            "ocr_text": str(page_texts[page_index]),
            "image_source_type": "path",
        }
        for page_index, image_path in enumerate(image_paths)
    ]
    gold_page_ids = [
        _canonical_mpdocvqa_page_id(cfg, doc_id, int(page_index))
        for page_index in sample.get("evidence_pages", [])
    ]
    return candidates, page_texts, gold_page_ids, page_id_to_canonical


def _validate_mpdocvqa_page_id_space(
    sample: dict[str, Any],
    pred_page_ids: list[str],
    gold_page_ids: list[str],
    candidate_page_ids: list[str],
) -> None:
    """Fail fast when MP-DocVQA predictions/gold do not share the same canonical page-id space."""
    candidate_set = {str(page_id) for page_id in candidate_page_ids}
    pred_set = {str(page_id) for page_id in pred_page_ids}
    gold_set = {str(page_id) for page_id in gold_page_ids}
    if not pred_set.issubset(candidate_set):
        raise RuntimeError(
            f"MP-DocVQA predicted page ids are outside the candidate id space: qid={sample.get('qid', '')} "
            f"doc_id={sample.get('doc_id', '')} unknown={sorted(pred_set - candidate_set)[:10]}"
        )
    if gold_set and not gold_set.issubset(candidate_set):
        raise RuntimeError(
            f"MP-DocVQA gold page ids are outside the candidate id space: qid={sample.get('qid', '')} "
            f"doc_id={sample.get('doc_id', '')} unknown={sorted(gold_set - candidate_set)[:10]}"
        )


def _mpdocvqa_recall_at_k(pred_page_ids: list[str], gold_page_ids: set[str], k: int) -> float:
    """Compute multi-relevant Recall@K in canonical MP-DocVQA page-id space."""
    if not gold_page_ids:
        return 0.0
    hits = len(set(map(str, pred_page_ids[:k])) & gold_page_ids)
    return float(hits / max(len(gold_page_ids), 1))


def _mpdocvqa_hit_at_k(pred_page_ids: list[str], gold_page_ids: set[str], k: int) -> float:
    """Compute Hit/Top@K in canonical MP-DocVQA page-id space."""
    return float(bool(set(map(str, pred_page_ids[:k])) & gold_page_ids))


def _mpdocvqa_mrr(pred_page_ids: list[str], gold_page_ids: set[str]) -> float:
    """Compute reciprocal rank of the first relevant page."""
    for rank, page_id in enumerate(pred_page_ids, start=1):
        if str(page_id) in gold_page_ids:
            return 1.0 / float(rank)
    return 0.0


def _evaluate_mpdocvqa_page_retrieval(predictions: list[dict[str, Any]], k_values: list[int]) -> dict[str, float]:
    """Aggregate page-level retrieval metrics for canonical MP-DocVQA page ids."""
    if not predictions:
        return {}
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"Top@{k}"] = sum(
            _mpdocvqa_hit_at_k(item.get("pred_page_ids", []), set(map(str, item.get("gold_page_ids", []))), int(k))
            for item in predictions
        ) / len(predictions)
        metrics[f"Recall@{k}"] = sum(
            _mpdocvqa_recall_at_k(item.get("pred_page_ids", []), set(map(str, item.get("gold_page_ids", []))), int(k))
            for item in predictions
        ) / len(predictions)
        metrics[f"Hit@{k}"] = metrics[f"Top@{k}"]
    metrics["MRR"] = sum(
        _mpdocvqa_mrr(item.get("pred_page_ids", []), set(map(str, item.get("gold_page_ids", []))))
        for item in predictions
    ) / len(predictions)
    metrics["PageAcc"] = sum(
        _mpdocvqa_hit_at_k(item.get("pred_page_ids", []), set(map(str, item.get("gold_page_ids", []))), 1)
        for item in predictions
    ) / len(predictions)
    metrics["num_samples"] = len(predictions)
    return metrics


def _ndcg_page_id_at_k(predictions: list[dict[str, Any]], k: int) -> float:
    """Compute binary nDCG@K for canonical page-id predictions."""
    if not predictions:
        return 0.0
    total = 0.0
    counted = 0
    for item in predictions:
        gold = set(map(str, item.get("gold_page_ids", [])))
        if not gold:
            raise RuntimeError("nDCG computation received an MP-DocVQA sample with empty gold_page_ids.")
        dcg = 0.0
        for index, page_id in enumerate(item.get("pred_page_ids", [])[:k], start=1):
            if str(page_id) in gold:
                dcg += 1.0 / np.log2(index + 1)
        ideal_hits = min(len(gold), k)
        idcg = sum(1.0 / np.log2(index + 1) for index in range(1, ideal_hits + 1))
        total += dcg / max(idcg, 1e-8)
        counted += 1
    if counted == 0:
        raise RuntimeError("nDCG computation received no labeled MP-DocVQA predictions.")
    return float(total / counted)


def run_visual_colqwen_adaptive_coarse_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run adaptive coarse BM25 routing followed by ColQwen subset reranking."""
    logger = get_logger("visual_colqwen_adaptive_coarse_retrieval")
    records = load_jsonl(dataset_path)
    logger.info(
        "Running visual ColQwen adaptive coarse retrieval on %s with %d samples and topk=%d",
        dataset_path,
        len(records),
        topk,
    )
    router_cfg = getattr(cfg, "retrieval_router", None)
    logger.info(
        "visual_backend=%s adaptive_coarse=%s visual_bm25_coarse=%s bypass_threshold=%s coarse_topk=%s",
        "colqwen",
        bool(getattr(router_cfg, "enable_adaptive_coarse", False)),
        bool(getattr(router_cfg, "enable_bm25_coarse", True)),
        int(getattr(router_cfg, "bypass_threshold", 0) or 0),
        int(getattr(router_cfg, "coarse_topk", 0) or 0),
    )

    retriever = ColQwenRetriever(cfg)
    predictions: list[dict] = []
    indexed_docs: set[str] = set()
    coarse_doc_text_cache: dict[str, list[str]] = {}
    coarse_doc_retriever_cache: dict[str, Any] = {}
    cache_stats = {
        "num_samples": len(records),
        "num_unique_docs": 0,
        "doc_visual_cache_hits": 0,
        "doc_visual_cache_misses": 0,
    }
    adaptive_coarse_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="retrieval_router", stats_prefix="visual")

    for sample in tqdm(records, desc=f"VisualColQwenAdaptiveCoarse {Path(dataset_path).stem}", unit="sample"):
        result, stats = _retrieve_visual_with_adaptive_coarse_for_sample(
            cfg=cfg,
            sample=sample,
            retriever=retriever,
            indexed_docs=indexed_docs,
            coarse_doc_text_cache=coarse_doc_text_cache,
            coarse_doc_retriever_cache=coarse_doc_retriever_cache,
        )
        cache_stats["doc_visual_cache_hits"] += int(stats.get("doc_visual_cache_hits", 0))
        cache_stats["doc_visual_cache_misses"] += int(stats.get("doc_visual_cache_misses", 0))
        _accumulate_adaptive_coarse_metrics(adaptive_coarse_stats, stats, stats_prefix="visual")
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": str(sample["doc_id"]),
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in result["page_ids"][:topk]],
                "pred_scores": [float(score) for score in result["scores"][:topk]],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    cache_stats["num_unique_docs"] = len({str(item["doc_id"]) for item in records})
    metrics.update(cache_stats)
    metrics.update(summarize_adaptive_coarse_stats(adaptive_coarse_stats, len(records), stats_prefix="visual"))
    logger.info("Visual ColQwen adaptive coarse cache stats: %s", cache_stats)
    logger.info("Visual ColQwen adaptive coarse metrics: %s", metrics)
    return predictions, metrics


def run_ocr_bge_retrieval_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    use_reranker: bool = False,
) -> tuple[list[dict], dict]:
    """Run OCR dense retrieval with optional BGE reranking over one dataset split."""
    logger_name = "ocr_bge_rerank_retrieval" if use_reranker else "ocr_bge_retrieval"
    logger = get_logger(logger_name)
    records = load_jsonl(dataset_path)
    logger.info(
        "Running %s on %s with %d samples and topk=%d",
        "ocr_bge_rerank" if use_reranker else "ocr_bge",
        dataset_path,
        len(records),
        topk,
    )
    retriever = OCRBGERetriever(cfg)
    reranker = OCRBGEReranker(cfg) if use_reranker else None
    indexed_docs: set[str] = set()
    predictions: list[dict] = []
    cache_stats = {
        "num_samples": len(records),
        "num_unique_docs": 0,
        "doc_ocr_bge_cache_hits": 0,
        "doc_ocr_bge_cache_misses": 0,
        "ocr_rerank_calls": 0,
    }

    for sample in tqdm(records, desc=f"{'OCRBGERerank' if use_reranker else 'OCRBGE'} {Path(dataset_path).stem}", unit="sample"):
        result, stats = _retrieve_ocr_bge_for_sample(
            cfg,
            sample,
            topk=topk,
            retriever=retriever,
            indexed_docs=indexed_docs,
            reranker=reranker,
            coarse_topn=int(cfg.ocr_retrieval.coarse_topn),
        )
        for key, value in stats.items():
            cache_stats[key] += value
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": str(sample["doc_id"]),
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in result["page_ids"]],
                "pred_scores": [float(score) for score in result["scores"]],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    cache_stats["num_unique_docs"] = len(indexed_docs)
    metrics.update(cache_stats)
    logger.info("%s metrics: %s", logger_name, metrics)
    return predictions, metrics


def _retrieve_ocr_bge_chunk_for_sample(
    cfg: Any,
    sample: dict[str, Any],
    topk: int,
    chunk_builder: OCRChunkBuilder,
    retriever: OCRBGEChunkRetriever,
    indexed_docs: set[str],
    reranker: OCRBGEChunkReranker | None = None,
    chunk_cache: dict[str, list[dict[str, Any]]] | None = None,
    coarse_topn: int | None = None,
    aggregation_strategy: str = "max",
    query_variant: str = "instruction_query",
    return_debug: bool = False,
    collect_chunk_stats: bool = False,
    doc_key_suffix: str | None = None,
) -> tuple[dict[str, list], dict[str, int]] | tuple[dict[str, list], dict[str, int], dict[str, Any]]:
    """Run chunk-level OCR retrieval with optional reranking and page aggregation."""
    doc_id = str(sample["doc_id"])
    suffix = f":{doc_key_suffix}" if doc_key_suffix else ""
    doc_key = f"{chunk_builder.page_text_variant}:{chunk_builder.chunk_size}:{chunk_builder.chunk_stride}:{doc_id}{suffix}"
    if chunk_cache is None:
        chunk_cache = {}
    stats = {
        "doc_ocr_chunk_cache_hits": 0,
        "doc_ocr_chunk_cache_misses": 0,
        "ocr_chunk_rerank_calls": 0,
    }
    debug_info = {
        "coarse_chunk_page_ids": [],
        "coarse_page_ids": [],
        "reranked_page_ids": [],
        "chunk_stats": {},
    }

    if doc_key in indexed_docs:
        stats["doc_ocr_chunk_cache_hits"] += 1
    else:
        chunks = _build_ocr_chunks_for_sample(sample, chunk_builder, chunk_cache, cache_key_suffix=doc_key_suffix)
        retriever.build_document_index(doc_id=doc_key, chunks=chunks)
        indexed_docs.add(doc_key)
        stats["doc_ocr_chunk_cache_misses"] += 1

    chunks = retriever.index.get(doc_key, {}).get("chunks", [])
    if collect_chunk_stats:
        debug_info["chunk_stats"] = _summarize_chunk_stats(
            doc_id=doc_id,
            chunks=chunks,
            token_counter=retriever.encoder,
            collect_debug=bool(getattr(cfg.ocr_chunk_retrieval, "debug_chunk_stats", True)),
        )
    else:
        debug_info["chunk_stats"] = {}
    retrieve_topk = int(coarse_topn or topk)
    chunk_result = retriever.retrieve(
        query=sample["question"],
        doc_id=doc_key,
        topk=retrieve_topk,
        query_variant=query_variant,
    )
    coarse_page_chunk_scores: dict[int, list[float]] = defaultdict(list)
    for page_id, score in zip(chunk_result.get("page_ids", []), chunk_result.get("scores", [])):
        coarse_page_chunk_scores[int(page_id)].append(float(score))
    debug_info["coarse_chunk_page_ids"] = [int(page_id) for page_id in chunk_result.get("page_ids", [])]
    coarse_page_result = aggregate_chunk_scores_to_page(
        [
            {"page_id": page_id, "score": score}
            for page_id, score in zip(chunk_result.get("page_ids", []), chunk_result.get("scores", []))
        ],
        strategy=aggregation_strategy,
        topk=retrieve_topk,
    )
    debug_info["coarse_page_ids"] = [int(page_id) for page_id in coarse_page_result.get("page_ids", [])]

    if reranker is None:
        final_result = {
            "page_ids": coarse_page_result["page_ids"][:topk],
            "scores": coarse_page_result["scores"][:topk],
            "ranks": list(range(1, min(len(coarse_page_result["page_ids"]), topk) + 1)),
            "page_chunk_scores": {page_id: scores for page_id, scores in coarse_page_chunk_scores.items()},
        }
        debug_info["reranked_page_ids"] = [int(page_id) for page_id in final_result["page_ids"]]
        if return_debug:
            return final_result, stats, debug_info
        return final_result, stats

    stats["ocr_chunk_rerank_calls"] += 1
    reranked_chunk_result = reranker.rerank(
        question=sample["question"],
        chunk_candidates=chunk_result["chunks"],
        topk=retrieve_topk,
        query_variant=query_variant,
    )
    reranked_page_chunk_scores: dict[int, list[float]] = defaultdict(list)
    for page_id, score in zip(reranked_chunk_result.get("page_ids", []), reranked_chunk_result.get("scores", [])):
        reranked_page_chunk_scores[int(page_id)].append(float(score))
    final_result = aggregate_chunk_scores_to_page(
        [
            {"page_id": page_id, "score": score}
            for page_id, score in zip(reranked_chunk_result.get("page_ids", []), reranked_chunk_result.get("scores", []))
        ],
        strategy=aggregation_strategy,
        topk=topk,
    )
    final_result["page_chunk_scores"] = {page_id: scores for page_id, scores in reranked_page_chunk_scores.items()}
    debug_info["reranked_page_ids"] = [int(page_id) for page_id in final_result["page_ids"]]
    if return_debug:
        return final_result, stats, debug_info
    return final_result, stats


def run_ocr_page_coarse_chunk_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run OCR page-level coarse routing followed by OCR chunk rerank on the routed subset."""
    logger = get_logger("ocr_page_coarse_chunk_retrieval")
    records = load_jsonl(dataset_path)
    logger.info(
        "Running OCR page coarse + chunk rerank on %s with %d samples and topk=%d",
        dataset_path,
        len(records),
        topk,
    )
    ocr_router_cfg = getattr(cfg, "ocr_router", None)
    logger.info(
        "ocr_backend=%s ocr_page_coarse=%s ocr_bm25_coarse=%s bypass_threshold=%s coarse_topk=%s",
        "bge_chunk_rerank",
        bool(getattr(ocr_router_cfg, "enable_ocr_page_coarse", False)),
        bool(getattr(ocr_router_cfg, "enable_bm25_coarse", True)),
        int(getattr(ocr_router_cfg, "bypass_threshold", 0) or 0),
        int(getattr(ocr_router_cfg, "coarse_topk", 0) or 0),
    )

    chunk_builder = OCRChunkBuilder(
        chunk_size=int(cfg.ocr_chunk_retrieval.chunk_size),
        chunk_stride=int(cfg.ocr_chunk_retrieval.chunk_stride),
        page_text_variant=str(cfg.ocr_chunk_retrieval.page_text_variant),
    )
    retriever = OCRBGEChunkRetriever(cfg)
    reranker = OCRBGEChunkReranker(cfg)
    indexed_docs: set[str] = set()
    coarse_doc_text_cache: dict[str, list[str]] = {}
    coarse_doc_retriever_cache: dict[str, Any] = {}
    chunk_cache: dict[str, list[dict[str, Any]]] = {}
    predictions: list[dict] = []
    cache_stats = {
        "num_samples": len(records),
        "num_unique_docs": 0,
        "doc_ocr_chunk_cache_hits": 0,
        "doc_ocr_chunk_cache_misses": 0,
        "ocr_chunk_rerank_calls": 0,
    }
    adaptive_coarse_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="ocr_router", stats_prefix="ocr")

    for sample in tqdm(records, desc=f"OCRPageCoarseChunk {Path(dataset_path).stem}", unit="sample"):
        result, stats = _retrieve_ocr_with_page_coarse_for_sample(
            cfg=cfg,
            sample=sample,
            topk=topk,
            chunk_builder=chunk_builder,
            retriever=retriever,
            indexed_docs=indexed_docs,
            reranker=reranker,
            chunk_cache=chunk_cache,
            coarse_doc_text_cache=coarse_doc_text_cache,
            coarse_doc_retriever_cache=coarse_doc_retriever_cache,
            coarse_topn=int(cfg.ocr_chunk_retrieval.coarse_topn),
            aggregation_strategy=str(cfg.ocr_chunk_retrieval.aggregate_strategy),
            query_variant=str(cfg.ocr_chunk_retrieval.query_variant),
        )
        for key in ("doc_ocr_chunk_cache_hits", "doc_ocr_chunk_cache_misses", "ocr_chunk_rerank_calls"):
            cache_stats[key] += int(stats.get(key, 0))
        _accumulate_adaptive_coarse_metrics(adaptive_coarse_stats, stats, stats_prefix="ocr")
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": str(sample["doc_id"]),
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in result["page_ids"][:topk]],
                "pred_scores": [float(score) for score in result["scores"][:topk]],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    cache_stats["num_unique_docs"] = len(indexed_docs)
    metrics.update(cache_stats)
    metrics.update(summarize_adaptive_coarse_stats(adaptive_coarse_stats, len(records), stats_prefix="ocr"))
    logger.info("OCR page coarse + chunk cache stats: %s", cache_stats)
    logger.info("OCR page coarse + chunk metrics: %s", metrics)
    return predictions, metrics


def run_ocr_page_bm25_bge_rerank_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run the lightweight OCR page route: BM25 coarse -> BGE-M3 -> page reranker."""
    logger = get_logger("ocr_page_bm25_bge_rerank_retrieval")
    records = load_jsonl(dataset_path)
    logger.info(
        "Running OCR page BM25->BGE->rerank on %s with %d samples and topk=%d",
        dataset_path,
        len(records),
        topk,
    )
    ocr_router_cfg = getattr(cfg, "ocr_router", None)
    semantic_cfg = getattr(cfg, "ocr_semantic_retrieval", None)
    reranker_cfg = getattr(cfg, "ocr_reranker", None)
    logger.info(
        "ocr_backend=%s ocr_page_coarse=%s ocr_bm25_coarse=%s ocr_bypass_threshold=%s ocr_coarse_topk=%s "
        "ocr_semantic_topk=%s ocr_rerank_topk=%s",
        "ocr_page_bm25_bge_rerank",
        bool(getattr(ocr_router_cfg, "enable_ocr_page_coarse", False)),
        bool(getattr(ocr_router_cfg, "enable_bm25_coarse", True)),
        int(getattr(ocr_router_cfg, "bypass_threshold", 0) or 0),
        int(getattr(ocr_router_cfg, "coarse_topk", 0) or 0),
        int(getattr(semantic_cfg, "semantic_topk", 0) or 0),
        int(getattr(reranker_cfg, "rerank_topk", 0) or 0),
    )

    retriever = OCRBGERetriever(cfg, config_attr="ocr_semantic_retrieval")
    reranker = OCRBGEReranker(cfg, config_attr="ocr_reranker") if bool(getattr(reranker_cfg, "enable_bge_reranker", True)) else None
    indexed_docs: set[str] = set()
    coarse_doc_text_cache: dict[str, list[str]] = {}
    coarse_doc_retriever_cache: dict[str, Any] = {}
    predictions: list[dict] = []
    cache_stats = {
        "num_samples": len(records),
        "num_unique_docs": 0,
    }
    ocr_bm25_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="ocr_router", stats_prefix="ocr_bm25")
    ocr_pipeline_stats = _build_ocr_page_pipeline_metric_accumulator()

    for sample in tqdm(records, desc=f"OCRPageBM25BGERerank {Path(dataset_path).stem}", unit="sample"):
        result, stats = _retrieve_ocr_with_bm25_bge_reranker_for_sample(
            cfg=cfg,
            sample=sample,
            topk=topk,
            retriever=retriever,
            indexed_docs=indexed_docs,
            reranker=reranker,
            bm25_doc_text_cache=coarse_doc_text_cache,
            bm25_doc_retriever_cache=coarse_doc_retriever_cache,
        )
        _accumulate_adaptive_coarse_metrics(ocr_bm25_stats, stats, stats_prefix="ocr_bm25")
        _accumulate_ocr_page_pipeline_metrics(ocr_pipeline_stats, stats)
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": str(sample["doc_id"]),
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in result["page_ids"][:topk]],
                "pred_scores": [float(score) for score in result["scores"][:topk]],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    cache_stats["num_unique_docs"] = len(indexed_docs)
    metrics.update(cache_stats)
    metrics.update(summarize_ocr_page_pipeline_metrics(ocr_pipeline_stats, len(records)))
    metrics.update(add_ocr_bm25_metric_aliases(summarize_adaptive_coarse_stats(ocr_bm25_stats, len(records), stats_prefix="ocr_bm25")))
    logger.info("OCR page BM25->BGE->rerank metrics: %s", metrics)
    return predictions, metrics


def run_ocr_bge_chunk_retrieval_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    use_reranker: bool = False,
) -> tuple[list[dict], dict, list[dict[str, Any]]]:
    """Run chunk-level OCR BGE retrieval with optional reranking over one dataset split."""
    logger_name = "ocr_bge_chunk_rerank_retrieval" if use_reranker else "ocr_bge_chunk_retrieval"
    logger = get_logger(logger_name)
    records = load_jsonl(dataset_path)
    chunk_cfg = cfg.ocr_chunk_retrieval
    logger.info(
        "Running %s on %s with %d samples, topk=%d, chunk_size=%d, stride=%d, agg=%s",
        logger_name,
        dataset_path,
        len(records),
        topk,
        int(chunk_cfg.chunk_size),
        int(chunk_cfg.chunk_stride),
        str(chunk_cfg.aggregate_strategy),
    )
    chunk_builder = OCRChunkBuilder(
        chunk_size=int(chunk_cfg.chunk_size),
        chunk_stride=int(chunk_cfg.chunk_stride),
        page_text_variant=str(chunk_cfg.page_text_variant),
    )
    retriever = OCRBGEChunkRetriever(cfg)
    reranker = OCRBGEChunkReranker(cfg) if use_reranker else None
    chunk_cache: dict[str, list[dict[str, Any]]] = {}
    indexed_docs: set[str] = set()
    predictions: list[dict] = []
    failure_cases: list[dict[str, Any]] = []
    chunk_coarse_hits = {k: 0.0 for k in [10, 20, 50]}
    page_coarse_hits = {k: 0.0 for k in [10, 20, 30, 50]}
    chunk_stat_totals = defaultdict(float)
    chunk_stat_doc_count = 0
    chunk_debug_rows: list[dict[str, Any]] = []
    abnormal_chunk_rows: list[dict[str, Any]] = []
    all_chunk_char_counts: list[int] = []
    all_chunk_word_counts: list[int] = []
    all_chunk_model_token_counts: list[int] = []
    labeled_count = 0
    stats_doc_keys_seen: set[str] = set()

    for sample in tqdm(records, desc=f"{'OCRBGEChunkRerank' if use_reranker else 'OCRBGEChunk'} {Path(dataset_path).stem}", unit="sample"):
        result, _stats, debug_info = _retrieve_ocr_bge_chunk_for_sample(
            cfg=cfg,
            sample=sample,
            topk=topk,
            chunk_builder=chunk_builder,
            retriever=retriever,
            indexed_docs=indexed_docs,
            reranker=reranker,
            chunk_cache=chunk_cache,
            coarse_topn=int(chunk_cfg.coarse_topn),
            aggregation_strategy=str(chunk_cfg.aggregate_strategy),
            query_variant=str(chunk_cfg.query_variant),
            return_debug=True,
            collect_chunk_stats=True,
        )
        evidence_pages = [int(page) for page in sample.get("evidence_pages", [])]
        doc_id = str(sample["doc_id"])
        stats_doc_key = f"{chunk_builder.page_text_variant}:{chunk_builder.chunk_size}:{chunk_builder.chunk_stride}:{doc_id}"
        if stats_doc_key not in stats_doc_keys_seen:
            stats_doc_keys_seen.add(stats_doc_key)
            chunk_stat_doc_count += 1
            for key, value in debug_info["chunk_stats"].items():
                if key in {"sample_rows", "abnormal_rows", "char_counts", "word_counts", "model_token_counts"}:
                    continue
                if key.startswith("max_"):
                    chunk_stat_totals[key] = max(float(chunk_stat_totals[key]), float(value))
                elif key.endswith("_p50") or key.endswith("_p90") or key.endswith("_p99"):
                    continue
                else:
                    chunk_stat_totals[key] += float(value)
            chunk_debug_rows.extend(debug_info["chunk_stats"].get("sample_rows", []))
            abnormal_chunk_rows.extend(debug_info["chunk_stats"].get("abnormal_rows", []))
            all_chunk_char_counts.extend(int(value) for value in debug_info["chunk_stats"].get("char_counts", []))
            all_chunk_word_counts.extend(int(value) for value in debug_info["chunk_stats"].get("word_counts", []))
            all_chunk_model_token_counts.extend(int(value) for value in debug_info["chunk_stats"].get("model_token_counts", []))
        if evidence_pages:
            labeled_count += 1
            for k in chunk_coarse_hits:
                chunk_coarse_hits[k] += _compute_chunk_recall_at_k(debug_info["coarse_chunk_page_ids"], evidence_pages, k)
            for k in page_coarse_hits:
                page_coarse_hits[k] += _compute_recall_at_k_from_page_ids(debug_info["coarse_page_ids"], evidence_pages, k)

        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": str(sample["doc_id"]),
                "question": sample["question"],
                "evidence_pages": evidence_pages,
                "pred_page_ids": [int(page_id) for page_id in result["page_ids"]],
                "pred_scores": [float(score) for score in result["scores"]],
                "topk": topk,
            }
        )

        correct_in_candidates = _compute_recall_at_k_from_page_ids(debug_info["coarse_page_ids"], evidence_pages, int(chunk_cfg.coarse_topn)) > 0.0
        if len(failure_cases) < 200 and evidence_pages and not correct_in_candidates:
            failure_cases.append(
                {
                    "qid": sample["qid"],
                    "doc_id": str(sample["doc_id"]),
                    "evidence_pages": evidence_pages,
                    "bge_chunk_candidate_pages": debug_info["coarse_page_ids"],
                    "reranked_pages": debug_info["reranked_page_ids"],
                    "correct_in_candidates": bool(correct_in_candidates),
                }
            )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    denom = max(labeled_count, 1)
    doc_denom = max(chunk_stat_doc_count, 1)
    metrics.update(
        {
            "num_samples": len(predictions),
            "num_unique_docs": len(indexed_docs),
            **{f"coarse_chunk_recall@{k}": float(chunk_coarse_hits[k] / denom) for k in chunk_coarse_hits},
            **{f"coarse_page_recall@{k}": float(page_coarse_hits[k] / denom) for k in page_coarse_hits},
            "avg_chunks_per_page": float(chunk_stat_totals["num_chunks"] / max(chunk_stat_totals["num_pages"], 1.0)),
            "avg_char_count_per_chunk": float(chunk_stat_totals["total_char_count"] / max(chunk_stat_totals["num_chunks"], 1.0)),
            "avg_word_count_per_chunk": float(chunk_stat_totals["total_word_count"] / max(chunk_stat_totals["num_chunks"], 1.0)),
            "avg_model_token_count_per_chunk": float(chunk_stat_totals["total_model_token_count"] / max(chunk_stat_totals["num_chunks"], 1.0)),
            "avg_tokens_per_chunk": float(chunk_stat_totals["total_word_count"] / max(chunk_stat_totals["num_chunks"], 1.0)),
            "max_char_count_per_chunk": float(chunk_stat_totals["max_char_count_per_chunk"]),
            "max_word_count_per_chunk": float(chunk_stat_totals["max_word_count_per_chunk"]),
            "max_model_token_count_per_chunk": float(chunk_stat_totals["max_model_token_count_per_chunk"]),
            "max_tokens_per_chunk": float(chunk_stat_totals["max_word_count_per_chunk"]),
            "chunk_char_count_p50": _compute_percentile(all_chunk_char_counts, 50),
            "chunk_char_count_p90": _compute_percentile(all_chunk_char_counts, 90),
            "chunk_char_count_p99": _compute_percentile(all_chunk_char_counts, 99),
            "chunk_word_count_p50": _compute_percentile(all_chunk_word_counts, 50),
            "chunk_word_count_p90": _compute_percentile(all_chunk_word_counts, 90),
            "chunk_word_count_p99": _compute_percentile(all_chunk_word_counts, 99),
            "chunk_model_token_count_p50": _compute_percentile(all_chunk_model_token_counts, 50),
            "chunk_model_token_count_p90": _compute_percentile(all_chunk_model_token_counts, 90),
            "chunk_model_token_count_p99": _compute_percentile(all_chunk_model_token_counts, 99),
            "chunk_size": int(chunk_cfg.chunk_size),
            "chunk_stride": int(chunk_cfg.chunk_stride),
            "aggregation_strategy": str(chunk_cfg.aggregate_strategy),
            "page_text_variant": str(chunk_cfg.page_text_variant),
            "query_variant": str(chunk_cfg.query_variant),
            "coarse_topn": int(chunk_cfg.coarse_topn),
        }
    )
    if bool(getattr(cfg.ocr_chunk_retrieval, "debug_chunk_stats", True)):
        debug_dir = Path(cfg.paths.output_dir) / "debug"
        ensure_dir(debug_dir)
        if chunk_debug_rows:
            save_jsonl(chunk_debug_rows, debug_dir / "chunk_debug_samples.jsonl")
        if abnormal_chunk_rows:
            save_jsonl(abnormal_chunk_rows, debug_dir / "abnormal_chunks.jsonl")
    logger.info("%s metrics: %s", logger_name, metrics)
    return predictions, metrics, failure_cases


def run_ocr_hybrid_retrieval_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict, list[dict[str, Any]]]:
    """Run hybrid OCR retrieval that unions BM25 page candidates and BGE chunk candidates."""
    logger = get_logger("ocr_hybrid_retrieval")
    records = load_jsonl(dataset_path)
    chunk_cfg = cfg.ocr_chunk_retrieval
    logger.info("Running ocr_hybrid on %s with %d samples and topk=%d", dataset_path, len(records), topk)
    hybrid = OCRHybridRetriever(cfg)
    chunk_builder = OCRChunkBuilder(
        chunk_size=int(chunk_cfg.chunk_size),
        chunk_stride=int(chunk_cfg.chunk_stride),
        page_text_variant=str(chunk_cfg.page_text_variant),
    )
    chunk_cache: dict[str, list[dict[str, Any]]] = {}
    failure_cases: list[dict[str, Any]] = []
    predictions: list[dict] = []

    for sample in tqdm(records, desc=f"OCRHybrid {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        page_texts = [load_page_ocr_text(path) for path in sample.get("ocr_paths", [])]
        page_ids = [_extract_page_idx_from_path(path) for path in sample.get("ocr_paths", [])]
        chunks = _build_ocr_chunks_for_sample(sample, chunk_builder, chunk_cache)
        result = hybrid.retrieve(
            question=sample["question"],
            doc_id=doc_id,
            page_texts=page_texts,
            page_ids=page_ids,
            chunks=chunks,
            topk=topk,
            coarse_chunk_topn=int(chunk_cfg.coarse_topn),
            bm25_topn=int(chunk_cfg.hybrid_bm25_topn),
            aggregation_strategy=str(chunk_cfg.aggregate_strategy),
            query_variant=str(chunk_cfg.query_variant),
        )
        evidence_pages = [int(page) for page in sample.get("evidence_pages", [])]
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": evidence_pages,
                "pred_page_ids": [int(page_id) for page_id in result["page_ids"]],
                "pred_scores": [float(score) for score in result["scores"]],
                "topk": topk,
            }
        )
        correct_in_candidates = _compute_recall_at_k_from_page_ids(
            list({*result.get("bm25_page_ids", []), *result.get("chunk_page_ids", [])}),
            evidence_pages,
            max(int(chunk_cfg.hybrid_bm25_topn), int(chunk_cfg.coarse_topn)),
        ) > 0.0
        if len(failure_cases) < 200 and evidence_pages and not correct_in_candidates:
            failure_cases.append(
                {
                    "qid": sample["qid"],
                    "doc_id": doc_id,
                    "evidence_pages": evidence_pages,
                    "BM25 candidate pages": [int(page_id) for page_id in result.get("bm25_page_ids", [])],
                    "BGE chunk candidate pages": [int(page_id) for page_id in result.get("chunk_page_ids", [])],
                    "reranked pages": [int(page_id) for page_id in result.get("reranked_page_ids", [])],
                    "correct_in_candidates": bool(correct_in_candidates),
                }
            )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics.update(
        {
            "num_samples": len(predictions),
            "chunk_size": int(chunk_cfg.chunk_size),
            "chunk_stride": int(chunk_cfg.chunk_stride),
            "aggregation_strategy": str(chunk_cfg.aggregate_strategy),
            "page_text_variant": str(chunk_cfg.page_text_variant),
            "query_variant": str(chunk_cfg.query_variant),
            "coarse_topn": int(chunk_cfg.coarse_topn),
            "hybrid_bm25_topn": int(chunk_cfg.hybrid_bm25_topn),
        }
    )
    logger.info("ocr_hybrid metrics: %s", metrics)
    return predictions, metrics, failure_cases


def _retrieve_ocr_hybrid_for_sample(
    cfg: Any,
    sample: dict[str, Any],
    topk: int,
    hybrid_retriever: OCRHybridRetriever,
    chunk_builder: OCRChunkBuilder,
    chunk_cache: dict[str, list[dict[str, Any]]],
    bm25_text_cache: dict[str, list[str]],
) -> tuple[dict[str, list], dict[str, int]]:
    """Run the hybrid OCR retrieval route for one sample with shared caches."""
    doc_id = str(sample["doc_id"])
    page_texts = bm25_text_cache.get(doc_id)
    if page_texts is None:
        page_texts = [load_page_ocr_text(path) for path in sample.get("ocr_paths", [])]
        bm25_text_cache[doc_id] = page_texts
    page_ids = [_extract_page_idx_from_path(path) for path in sample.get("ocr_paths", [])]
    chunks = _build_ocr_chunks_for_sample(sample, chunk_builder, chunk_cache)
    result = hybrid_retriever.retrieve(
        question=sample["question"],
        doc_id=doc_id,
        page_texts=page_texts,
        page_ids=page_ids,
        chunks=chunks,
        topk=topk,
        coarse_chunk_topn=int(cfg.ocr_chunk_retrieval.coarse_topn),
        bm25_topn=int(cfg.ocr_chunk_retrieval.hybrid_bm25_topn),
        aggregation_strategy=str(cfg.ocr_chunk_retrieval.aggregate_strategy),
        query_variant=str(cfg.ocr_chunk_retrieval.query_variant),
    )
    return result, {"doc_ocr_hybrid_calls": 1}


def _retrieve_ocr_nv_chunk_for_sample(
    cfg: Any,
    sample: dict[str, Any],
    topk: int,
    chunk_builder: OCRChunkBuilder,
    retriever: OCRNVChunkRetriever,
    indexed_docs: set[str],
    chunk_cache: dict[str, list[dict[str, Any]]] | None = None,
    coarse_topn: int | None = None,
    aggregation_strategy: str = "max",
) -> tuple[dict[str, list], dict[str, int]]:
    """Run chunk-level OCR retrieval using NV-Embed-v2 and aggregate to page scores."""
    doc_id = str(sample["doc_id"])
    doc_key = f"{chunk_builder.page_text_variant}:{chunk_builder.chunk_size}:{chunk_builder.chunk_stride}:{doc_id}"
    if chunk_cache is None:
        chunk_cache = {}
    stats = {
        "doc_ocr_nv_chunk_cache_hits": 0,
        "doc_ocr_nv_chunk_cache_misses": 0,
    }
    if doc_key in indexed_docs:
        stats["doc_ocr_nv_chunk_cache_hits"] += 1
    else:
        chunks = _build_ocr_chunks_for_sample(sample, chunk_builder, chunk_cache)
        retriever.build_document_index(doc_id=doc_key, chunks=chunks)
        indexed_docs.add(doc_key)
        stats["doc_ocr_nv_chunk_cache_misses"] += 1

    retrieve_topk = int(coarse_topn or topk)
    chunk_result = retriever.retrieve(
        query=sample["question"],
        doc_id=doc_key,
        topk=retrieve_topk,
    )
    page_chunk_scores: dict[int, list[float]] = defaultdict(list)
    for page_id, score in zip(chunk_result.get("page_ids", []), chunk_result.get("scores", [])):
        page_chunk_scores[int(page_id)].append(float(score))
    page_result = aggregate_chunk_scores_to_page(
        [
            {"page_id": page_id, "score": score}
            for page_id, score in zip(chunk_result.get("page_ids", []), chunk_result.get("scores", []))
        ],
        strategy=aggregation_strategy,
        topk=topk,
    )
    page_result["page_chunk_scores"] = {page_id: scores for page_id, scores in page_chunk_scores.items()}
    return page_result, stats


def _retrieve_ocr_jina_chunk_for_sample(
    cfg: Any,
    sample: dict[str, Any],
    topk: int,
    chunk_builder: OCRChunkBuilder,
    retriever: OCRJinaChunkRetriever,
    indexed_docs: set[str],
    chunk_cache: dict[str, list[dict[str, Any]]] | None = None,
    coarse_topn: int | None = None,
    aggregation_strategy: str = "max",
) -> tuple[dict[str, list], dict[str, int]]:
    """Run chunk-level OCR retrieval using offline jina-embeddings-v3 and aggregate to page scores."""
    doc_id = str(sample["doc_id"])
    doc_key = f"{chunk_builder.page_text_variant}:{chunk_builder.chunk_size}:{chunk_builder.chunk_stride}:{doc_id}"
    if chunk_cache is None:
        chunk_cache = {}
    stats = {
        "doc_ocr_jina_chunk_cache_hits": 0,
        "doc_ocr_jina_chunk_cache_misses": 0,
    }
    if doc_key in indexed_docs:
        stats["doc_ocr_jina_chunk_cache_hits"] += 1
    else:
        chunks = _build_ocr_chunks_for_sample(sample, chunk_builder, chunk_cache)
        retriever.build_document_index(doc_id=doc_key, chunks=chunks)
        indexed_docs.add(doc_key)
        stats["doc_ocr_jina_chunk_cache_misses"] += 1

    retrieve_topk = int(coarse_topn or topk)
    chunk_result = retriever.retrieve(
        query=sample["question"],
        doc_id=doc_key,
        topk=retrieve_topk,
    )
    page_result = aggregate_chunk_scores_to_page(
        [
            {"page_id": page_id, "score": score}
            for page_id, score in zip(chunk_result.get("page_ids", []), chunk_result.get("scores", []))
        ],
        strategy=aggregation_strategy,
        topk=topk,
    )
    return page_result, stats


def run_ocr_nv_chunk_retrieval_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run chunk-level OCR retrieval with NV-Embed-v2 over one dataset split."""
    logger = get_logger("ocr_nv_chunk_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running ocr_nv_chunk on %s with %d samples and topk=%d", dataset_path, len(records), topk)
    chunk_cfg = cfg.ocr_chunk_retrieval
    nv_cfg = cfg.ocr_nv_retrieval
    chunk_builder = OCRChunkBuilder(
        chunk_size=int(chunk_cfg.chunk_size),
        chunk_stride=int(chunk_cfg.chunk_stride),
        page_text_variant=str(chunk_cfg.page_text_variant),
    )
    retriever = OCRNVChunkRetriever(cfg)
    chunk_cache: dict[str, list[dict[str, Any]]] = {}
    indexed_docs: set[str] = set()
    predictions: list[dict] = []
    cache_stats = {
        "num_samples": len(records),
        "num_unique_docs": 0,
        "doc_ocr_nv_chunk_cache_hits": 0,
        "doc_ocr_nv_chunk_cache_misses": 0,
    }
    for sample in tqdm(records, desc=f"OCRNVChunk {Path(dataset_path).stem}", unit="sample"):
        result, stats = _retrieve_ocr_nv_chunk_for_sample(
            cfg=cfg,
            sample=sample,
            topk=topk,
            chunk_builder=chunk_builder,
            retriever=retriever,
            indexed_docs=indexed_docs,
            chunk_cache=chunk_cache,
            coarse_topn=int(nv_cfg.coarse_topn),
            aggregation_strategy=str(chunk_cfg.aggregate_strategy),
        )
        for key, value in stats.items():
            cache_stats[key] += value
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": str(sample["doc_id"]),
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in result["page_ids"]],
                "pred_scores": [float(score) for score in result["scores"]],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    cache_stats["num_unique_docs"] = len(indexed_docs)
    metrics.update(
        {
            **cache_stats,
            "chunk_size": int(chunk_cfg.chunk_size),
            "chunk_stride": int(chunk_cfg.chunk_stride),
            "aggregation_strategy": str(chunk_cfg.aggregate_strategy),
            "page_text_variant": str(chunk_cfg.page_text_variant),
            "coarse_topn": int(nv_cfg.coarse_topn),
        }
    )
    logger.info("ocr_nv_chunk metrics: %s", metrics)
    return predictions, metrics


def run_ocr_jina_chunk_retrieval_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run chunk-level OCR retrieval with offline jina-embeddings-v3 over one dataset split."""
    logger = get_logger("ocr_jina_chunk_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running ocr_jina_chunk on %s with %d samples and topk=%d", dataset_path, len(records), topk)
    chunk_cfg = cfg.ocr_chunk_retrieval
    jina_cfg = cfg.ocr_jina_retrieval
    chunk_builder = OCRChunkBuilder(
        chunk_size=int(chunk_cfg.chunk_size),
        chunk_stride=int(chunk_cfg.chunk_stride),
        page_text_variant=str(chunk_cfg.page_text_variant),
    )
    retriever = OCRJinaChunkRetriever(cfg)
    chunk_cache: dict[str, list[dict[str, Any]]] = {}
    indexed_docs: set[str] = set()
    predictions: list[dict] = []
    cache_stats = {
        "num_samples": len(records),
        "num_unique_docs": 0,
        "doc_ocr_jina_chunk_cache_hits": 0,
        "doc_ocr_jina_chunk_cache_misses": 0,
    }
    for sample in tqdm(records, desc=f"OCRJinaChunk {Path(dataset_path).stem}", unit="sample"):
        result, stats = _retrieve_ocr_jina_chunk_for_sample(
            cfg=cfg,
            sample=sample,
            topk=topk,
            chunk_builder=chunk_builder,
            retriever=retriever,
            indexed_docs=indexed_docs,
            chunk_cache=chunk_cache,
            coarse_topn=int(jina_cfg.coarse_topn),
            aggregation_strategy=str(chunk_cfg.aggregate_strategy),
        )
        for key, value in stats.items():
            cache_stats[key] += value
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": str(sample["doc_id"]),
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in result["page_ids"]],
                "pred_scores": [float(score) for score in result["scores"]],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    cache_stats["num_unique_docs"] = len(indexed_docs)
    metrics.update(
        {
            **cache_stats,
            "chunk_size": int(chunk_cfg.chunk_size),
            "chunk_stride": int(chunk_cfg.chunk_stride),
            "aggregation_strategy": str(chunk_cfg.aggregate_strategy),
            "page_text_variant": str(chunk_cfg.page_text_variant),
            "coarse_topn": int(jina_cfg.coarse_topn),
            "query_task": str(jina_cfg.query_task),
            "passage_task": str(jina_cfg.passage_task),
        }
    )
    logger.info("ocr_jina_chunk metrics: %s", metrics)
    return predictions, metrics


def run_ocr_bge_debug_on_dataset(
    cfg: Any,
    dataset_path: str,
    use_reranker: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run OCR-BGE debug experiments across page-text, query, and coarse-topN variants."""
    logger_name = "ocr_bge_rerank_debug" if use_reranker else "ocr_bge_debug"
    logger = get_logger(logger_name)
    records = load_jsonl(dataset_path)
    topk = int(cfg.ocr_retrieval.topk)
    k_values = list(cfg.retrieval.k_values)
    logger.info(
        "Running %s on %s with %d samples and topk=%d",
        logger_name,
        dataset_path,
        len(records),
        topk,
    )

    retriever = OCRBGERetriever(cfg)
    reranker = OCRBGEReranker(cfg) if use_reranker else None
    doc_text_cache: dict[str, list[str]] = {}
    indexed_docs: set[str] = set()
    coarse_topn_values = [10, 20, 30, 50]
    coarse_matrix = [
        ("raw_page_text", "raw_question"),
        ("clean_page_text", "raw_question"),
        ("clean_page_text", "instruction_query"),
        ("clean_trunc_page_text", "instruction_query"),
    ]

    selected_matrix = coarse_matrix
    if use_reranker:
        coarse_summary, coarse_runs, _ = run_ocr_bge_debug_on_dataset(cfg, dataset_path, use_reranker=False)
        ranked_runs = sorted(
            coarse_runs,
            key=lambda item: (float(item.get("MRR", 0.0)), float(item.get("Recall@10", 0.0))),
            reverse=True,
        )
        selected_matrix = []
        seen_pairs: set[tuple[str, str]] = set()
        for item in ranked_runs:
            pair = (str(item["page_text_variant"]), str(item["query_variant"]))
            if pair in seen_pairs:
                continue
            selected_matrix.append(pair)
            seen_pairs.add(pair)
            if len(selected_matrix) == 2:
                break
        if not selected_matrix:
            selected_matrix = coarse_matrix[:2]
        logger.info("Selected rerank debug settings from coarse stage: %s", selected_matrix)

    run_records: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []

    for page_text_variant, query_variant in selected_matrix:
        topn_grid = coarse_topn_values if use_reranker else [max(coarse_topn_values)]
        for coarse_topn in topn_grid:
            predictions: list[dict[str, Any]] = []
            length_aggregate = defaultdict(float)
            num_unique_docs = 0
            doc_keys_seen: set[str] = set()
            coarse_hits = {k: 0.0 for k in coarse_topn_values}
            labeled_count = 0

            for sample in tqdm(
                records,
                desc=f"{'OCRBGERerankDebug' if use_reranker else 'OCRBGEDebug'} {page_text_variant} {query_variant} top{coarse_topn}",
                unit="sample",
            ):
                doc_id = str(sample["doc_id"])
                doc_key = f"{page_text_variant}:{doc_id}"
                if doc_key not in doc_keys_seen:
                    doc_keys_seen.add(doc_key)
                    page_texts = _build_ocr_bge_page_texts(sample, page_text_variant, doc_text_cache)
                    _update_length_summary(length_aggregate, page_texts, page_text_variant)
                    num_unique_docs += 1

                result, _stats, debug_info = _retrieve_ocr_bge_for_sample(
                    cfg=cfg,
                    sample=sample,
                    topk=topk,
                    retriever=retriever,
                    indexed_docs=indexed_docs,
                    reranker=reranker,
                    coarse_topn=coarse_topn,
                    page_text_variant=page_text_variant,
                    query_variant=query_variant,
                    doc_text_cache=doc_text_cache,
                    return_debug=True,
                )
                evidence_pages = [int(page) for page in sample.get("evidence_pages", [])]
                coarse_page_ids = [int(page_id) for page_id in debug_info["coarse_page_ids"]]
                reranked_page_ids = [int(page_id) for page_id in result["page_ids"]]
                if evidence_pages:
                    labeled_count += 1
                    for k in coarse_topn_values:
                        coarse_hits[k] += _compute_recall_at_k_from_page_ids(coarse_page_ids, evidence_pages, k)
                predictions.append(
                    {
                        "qid": sample["qid"],
                        "doc_id": doc_id,
                        "question": sample["question"],
                        "evidence_pages": evidence_pages,
                        "pred_page_ids": reranked_page_ids,
                        "pred_scores": [float(score) for score in result["scores"]],
                        "topk": topk,
                    }
                )
                correct_in_coarse = _compute_recall_at_k_from_page_ids(coarse_page_ids, evidence_pages, coarse_topn) > 0.0
                correct_in_rerank_top10 = _compute_recall_at_k_from_page_ids(reranked_page_ids, evidence_pages, topk) > 0.0
                if len(failure_cases) < 200 and (not correct_in_coarse or not correct_in_rerank_top10):
                    failure_cases.append(
                        {
                            "qid": sample["qid"],
                            "doc_id": doc_id,
                            "question": sample["question"],
                            "evidence_pages": evidence_pages,
                            "query_variant": query_variant,
                            "page_text_variant": page_text_variant,
                            "coarse_topn": coarse_topn,
                            "coarse_page_ids": coarse_page_ids,
                            "reranked_page_ids": reranked_page_ids,
                            "correct_in_coarse": bool(correct_in_coarse),
                            "correct_in_rerank_top10": bool(correct_in_rerank_top10),
                        }
                    )

            labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
            metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
            coarse_denominator = max(labeled_count, 1)
            coarse_stats = {f"coarse_recall@{k}": float(coarse_hits[k] / coarse_denominator) for k in coarse_topn_values}

            run_record = {
                "mode": "ocr_bge_rerank_debug" if use_reranker else "ocr_bge_debug",
                "page_text_variant": page_text_variant,
                "query_variant": query_variant,
                "coarse_topn": int(coarse_topn),
                "num_samples": len(labeled_predictions),
                "num_unique_docs": num_unique_docs,
                **metrics,
                **coarse_stats,
                **_finalize_length_summary(length_aggregate, page_text_variant),
            }
            run_records.append(run_record)
            logger.info("OCR BGE debug run: %s", run_record)

    best_run = max(run_records, key=lambda item: (float(item.get("MRR", 0.0)), float(item.get("Recall@10", 0.0)))) if run_records else {}
    summary = {
        "mode": "ocr_bge_rerank_debug" if use_reranker else "ocr_bge_debug",
        "dataset_path": dataset_path,
        "num_runs": len(run_records),
        "best_run": best_run,
        "selected_matrix": [
            {"page_text_variant": pair[0], "query_variant": pair[1]}
            for pair in selected_matrix
        ],
    }
    return summary, run_records, failure_cases


def run_fixed_fusion_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    alpha: float = 0.5,
) -> tuple[list[dict], dict]:
    """Run fixed weighted fusion over OCR-only and visual-only retrieval results."""
    logger = get_logger("fixed_fusion_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running fixed fusion on %s with %d samples, topk=%d, alpha=%.3f", dataset_path, len(records), topk, alpha)

    bm25_text_cache: dict[str, list[str]] = {}
    bm25_retriever_cache: dict[str, BM25Retriever] = {}
    visual_retriever = ColPaliRetriever(cfg, require_engine=True)
    visual_indexed_docs: set[str] = set()
    predictions: list[dict] = []
    cache_stats = {
        "doc_text_cache_hits": 0,
        "doc_text_cache_misses": 0,
        "doc_retriever_cache_hits": 0,
        "doc_retriever_cache_misses": 0,
        "doc_visual_cache_hits": 0,
        "doc_visual_cache_misses": 0,
    }

    for sample in tqdm(records, desc=f"FixedFusion {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        ocr_result, ocr_stats = _retrieve_bm25_for_sample(sample, topk, bm25_text_cache, bm25_retriever_cache)
        visual_result, visual_stats = _retrieve_visual_for_sample(cfg, sample, topk, visual_retriever, visual_indexed_docs)
        for key, value in ocr_stats.items():
            cache_stats[key] += value
        for key, value in visual_stats.items():
            cache_stats[key] += value

        fusion_result = fixed_fusion(ocr_result=ocr_result, visual_result=visual_result, alpha=alpha, topk=topk)
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in fusion_result["page_ids"]],
                "pred_scores": fusion_result["scores"],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics.update(
        {
            "num_samples": len(records),
            "num_unique_docs": len(set(item["doc_id"] for item in records)),
            **cache_stats,
        }
    )
    fixed_cache_log = {
        "num_samples": metrics["num_samples"],
        "num_unique_docs": metrics["num_unique_docs"],
        **{key: metrics[key] for key in cache_stats},
    }
    logger.info("Fixed fusion cache stats: %s", fixed_cache_log)
    logger.info("Fixed fusion metrics: %s", metrics)
    return predictions, metrics


def run_rrf_fusion_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    rrf_k: int = 60,
) -> tuple[list[dict], dict]:
    """Run Reciprocal Rank Fusion over OCR-only and visual-only retrieval results."""
    logger = get_logger("rrf_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running RRF fusion on %s with %d samples, topk=%d, rrf_k=%d", dataset_path, len(records), topk, rrf_k)

    bm25_text_cache: dict[str, list[str]] = {}
    bm25_retriever_cache: dict[str, BM25Retriever] = {}
    visual_retriever = ColPaliRetriever(cfg, require_engine=True)
    visual_indexed_docs: set[str] = set()
    predictions: list[dict] = []
    cache_stats = {
        "doc_text_cache_hits": 0,
        "doc_text_cache_misses": 0,
        "doc_retriever_cache_hits": 0,
        "doc_retriever_cache_misses": 0,
        "doc_visual_cache_hits": 0,
        "doc_visual_cache_misses": 0,
    }

    for sample in tqdm(records, desc=f"RRF {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        ocr_result, ocr_stats = _retrieve_bm25_for_sample(sample, topk, bm25_text_cache, bm25_retriever_cache)
        visual_result, visual_stats = _retrieve_visual_for_sample(cfg, sample, topk, visual_retriever, visual_indexed_docs)
        for key, value in ocr_stats.items():
            cache_stats[key] += value
        for key, value in visual_stats.items():
            cache_stats[key] += value

        fusion_result = rrf_fusion(ocr_result=ocr_result, visual_result=visual_result, rrf_k=rrf_k, topk=topk)
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": [int(page_id) for page_id in fusion_result["page_ids"]],
                "pred_scores": fusion_result["scores"],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics.update(
        {
            "num_samples": len(records),
            "num_unique_docs": len(set(item["doc_id"] for item in records)),
            **cache_stats,
        }
    )
    rrf_cache_log = {
        "num_samples": metrics["num_samples"],
        "num_unique_docs": metrics["num_unique_docs"],
        **{key: metrics[key] for key in cache_stats},
    }
    logger.info("RRF fusion cache stats: %s", rrf_cache_log)
    logger.info("RRF fusion metrics: %s", metrics)
    return predictions, metrics


def run_adaptive_fusion_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run trained adaptive fusion reranking over a dataset split."""
    logger = get_logger("adaptive_fusion_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running adaptive fusion on %s with %d samples and topk=%d", dataset_path, len(records), topk)

    model = _load_adaptive_model_from_checkpoint(cfg, checkpoint_path)
    question_encoder = QuestionEncoder()
    bm25_text_cache: dict[str, list[str]] = {}
    bm25_retriever_cache: dict[str, BM25Retriever] = {}
    visual_retriever = ColPaliRetriever(cfg, require_engine=True)
    visual_indexed_docs: set[str] = set()
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] = {}
    predictions: list[dict] = []
    skipped_empty_candidates = 0
    cache_stats = {
        "doc_text_cache_hits": 0,
        "doc_text_cache_misses": 0,
        "doc_retriever_cache_hits": 0,
        "doc_retriever_cache_misses": 0,
        "doc_visual_cache_hits": 0,
        "doc_visual_cache_misses": 0,
    }

    for sample in tqdm(records, desc=f"AdaptiveFusion {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        ocr_result, ocr_stats = _retrieve_bm25_for_sample(sample, topk, bm25_text_cache, bm25_retriever_cache)
        visual_result, visual_stats = _retrieve_visual_for_sample(cfg, sample, topk, visual_retriever, visual_indexed_docs)
        for key, value in ocr_stats.items():
            cache_stats[key] += value
        for key, value in visual_stats.items():
            cache_stats[key] += value

        candidate_rows = build_adaptive_candidate_features(
            sample,
            ocr_result,
            visual_result,
            question_encoder=question_encoder,
            ocr_quality_cache=ocr_quality_cache,
        )
        if not candidate_rows:
            skipped_empty_candidates += 1
            logger.warning(
                "Skipping adaptive fusion sample with empty candidates. qid=%s doc_id=%s mode=adaptive_fusion",
                sample.get("qid"),
                sample.get("doc_id"),
            )
            continue

        ranked = model.rank_candidates(candidate_rows)
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": ranked["page_ids"][:topk],
                "pred_scores": ranked["scores"][:topk],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics.update(
        {
            "num_samples": len(predictions),
            "num_unique_docs": len(set(item["doc_id"] for item in records)),
            "skipped_empty_candidates": skipped_empty_candidates,
            **cache_stats,
        }
    )
    logger.info("Adaptive fusion metrics: %s", metrics)
    return predictions, metrics


def run_adaptive_fusion_v2_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run enhanced adaptive fusion reranking over a dataset split."""
    logger = get_logger("adaptive_fusion_v2_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running adaptive fusion v2 on %s with %d samples and topk=%d", dataset_path, len(records), topk)

    model = _load_adaptive_model_v2_from_checkpoint(cfg, checkpoint_path)
    question_encoder = QuestionEncoder()
    bm25_text_cache: dict[str, list[str]] = {}
    bm25_retriever_cache: dict[str, BM25Retriever] = {}
    visual_retriever = ColPaliRetriever(cfg, require_engine=True)
    visual_indexed_docs: set[str] = set()
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] = {}
    predictions: list[dict] = []
    skipped_empty_candidates = 0
    cache_stats = {
        "doc_text_cache_hits": 0,
        "doc_text_cache_misses": 0,
        "doc_retriever_cache_hits": 0,
        "doc_retriever_cache_misses": 0,
        "doc_visual_cache_hits": 0,
        "doc_visual_cache_misses": 0,
    }

    for sample in tqdm(records, desc=f"AdaptiveFusionV2 {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        ocr_result, ocr_stats = _retrieve_bm25_for_sample(sample, topk, bm25_text_cache, bm25_retriever_cache)
        visual_result, visual_stats = _retrieve_visual_for_sample(cfg, sample, topk, visual_retriever, visual_indexed_docs)
        for key, value in ocr_stats.items():
            cache_stats[key] += value
        for key, value in visual_stats.items():
            cache_stats[key] += value

        ocr_page_texts = bm25_text_cache.get(doc_id, [])
        candidate_rows = build_candidate_features_v2(
            sample,
            ocr_result,
            visual_result,
            ocr_page_texts=ocr_page_texts,
            question_encoder=question_encoder,
            ocr_quality_cache=ocr_quality_cache,
        )
        if not candidate_rows:
            skipped_empty_candidates += 1
            logger.warning(
                "Skipping adaptive fusion v2 sample with empty candidates. qid=%s doc_id=%s mode=adaptive_fusion_v2",
                sample.get("qid"),
                sample.get("doc_id"),
            )
            continue

        ranked = model.rank_candidates(candidate_rows)
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": ranked["page_ids"][:topk],
                "pred_scores": ranked["scores"][:topk],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics.update(
        {
            "num_samples": len(predictions),
            "num_unique_docs": len(set(item["doc_id"] for item in records)),
            "skipped_empty_candidates": skipped_empty_candidates,
            **cache_stats,
        }
    )
    logger.info("Adaptive fusion v2 metrics: %s", metrics)
    return predictions, metrics


def run_adaptive_fusion_ablation_on_dataset(
    cfg: Any,
    dataset_path: str,
    variant: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run one rollback-style adaptive fusion ablation over a dataset split."""
    variant_settings = {
        "ablate_q": {
            "logger_name": "adaptive_fusion_ablate_q_retrieval",
            "feature_builder": build_candidate_features_ablate_q,
            "model_loader": _load_adaptive_ablation_model_from_checkpoint,
        },
        "ablate_ocrq": {
            "logger_name": "adaptive_fusion_ablate_ocrq_retrieval",
            "feature_builder": build_candidate_features_ablate_ocrq,
            "model_loader": _load_adaptive_ablation_model_from_checkpoint,
        },
        "ablate_lex": {
            "logger_name": "adaptive_fusion_ablate_lex_retrieval",
            "feature_builder": build_candidate_features_ablate_lex,
            "model_loader": _load_adaptive_ablation_model_from_checkpoint,
        },
        "ablate_mlp": {
            "logger_name": "adaptive_fusion_ablate_mlp_retrieval",
            "feature_builder": build_candidate_features_ablate_mlp,
            "model_loader": _load_adaptive_ablation_mlp_model_from_checkpoint,
        },
        "ablate_mlp_q": {
            "logger_name": "adaptive_fusion_ablate_mlp_q_retrieval",
            "feature_builder": build_candidate_features_ablate_mlp_q,
            "model_loader": _load_adaptive_ablation_mlp_model_from_checkpoint,
        },
        "ablate_mlp_ocrq": {
            "logger_name": "adaptive_fusion_ablate_mlp_ocrq_retrieval",
            "feature_builder": build_candidate_features_ablate_mlp_ocrq,
            "model_loader": _load_adaptive_ablation_mlp_model_from_checkpoint,
        },
        "ablate_mlp_lex": {
            "logger_name": "adaptive_fusion_ablate_mlp_lex_retrieval",
            "feature_builder": build_candidate_features_ablate_mlp_lex,
            "model_loader": _load_adaptive_ablation_mlp_model_from_checkpoint,
        },
    }
    if variant not in variant_settings:
        raise ValueError(f"Unsupported adaptive-fusion ablation variant: {variant}")
    setting = variant_settings[variant]
    logger = get_logger(setting["logger_name"])
    records = load_jsonl(dataset_path)
    logger.info("Running %s on %s with %d samples and topk=%d", variant, dataset_path, len(records), topk)

    model = setting["model_loader"](cfg, variant, checkpoint_path)
    question_encoder = QuestionEncoder()
    bm25_text_cache: dict[str, list[str]] = {}
    bm25_retriever_cache: dict[str, BM25Retriever] = {}
    visual_retriever = ColPaliRetriever(cfg, require_engine=True)
    visual_indexed_docs: set[str] = set()
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] = {}
    predictions: list[dict] = []
    skipped_empty_candidates = 0
    cache_stats = {
        "doc_text_cache_hits": 0,
        "doc_text_cache_misses": 0,
        "doc_retriever_cache_hits": 0,
        "doc_retriever_cache_misses": 0,
        "doc_visual_cache_hits": 0,
        "doc_visual_cache_misses": 0,
    }

    for sample in tqdm(records, desc=f"{variant} {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        ocr_result, ocr_stats = _retrieve_bm25_for_sample(sample, topk, bm25_text_cache, bm25_retriever_cache)
        visual_result, visual_stats = _retrieve_visual_for_sample(cfg, sample, topk, visual_retriever, visual_indexed_docs)
        for key, value in ocr_stats.items():
            cache_stats[key] += value
        for key, value in visual_stats.items():
            cache_stats[key] += value

        candidate_rows = setting["feature_builder"](
            sample,
            ocr_result,
            visual_result,
            ocr_page_texts=bm25_text_cache.get(doc_id, []),
            question_encoder=question_encoder,
            ocr_quality_cache=ocr_quality_cache,
        )
        if not candidate_rows:
            skipped_empty_candidates += 1
            logger.warning(
                "Skipping %s sample with empty candidates. qid=%s doc_id=%s",
                variant,
                sample.get("qid"),
                sample.get("doc_id"),
            )
            continue

        ranked = model.rank_candidates(candidate_rows)
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": ranked["page_ids"][:topk],
                "pred_scores": ranked["scores"][:topk],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics.update(
        {
            "num_samples": len(predictions),
            "num_unique_docs": len(set(item["doc_id"] for item in records)),
            "skipped_empty_candidates": skipped_empty_candidates,
            **cache_stats,
        }
    )
    logger.info("%s metrics: %s", variant, metrics)
    return predictions, metrics


def run_adaptive_fusion_mlp_ocrq_bge_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run the current-best fusion structure with page-level BGE rerank OCR retrieval."""
    return _run_adaptive_fusion_mlp_ocrq_with_backend_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=topk,
        k_values=k_values,
        checkpoint_path=checkpoint_path,
        variant="mlp_ocrq_bge",
        checkpoint_subdir="adaptive_fusion_mlp_ocrq_bge",
        ocr_backend="bge_rerank",
    )


def run_adaptive_fusion_mlp_ocrq_chunk_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run the current-best fusion structure with chunk-level OCR rerank."""
    return _run_adaptive_fusion_mlp_ocrq_with_backend_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=topk,
        k_values=k_values,
        checkpoint_path=checkpoint_path,
        variant="mlp_ocrq_chunk",
        checkpoint_subdir="adaptive_fusion_mlp_ocrq_chunk",
        ocr_backend="bge_chunk_rerank",
    )


def run_adaptive_fusion_visual_colqwen_ocr_chunk_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run the clean visual_colqwen + OCR chunk adaptive fusion baseline."""
    return _run_adaptive_fusion_mlp_ocrq_with_backend_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=topk,
        k_values=k_values,
        checkpoint_path=checkpoint_path,
        variant="visual_colqwen_ocr_chunk",
        checkpoint_subdir="adaptive_fusion_visual_colqwen_ocr_chunk",
        ocr_backend="ocr_page_bm25_bge_rerank",
        visual_backend="colqwen",
        feature_builder=build_candidate_features_visual_colqwen_ocr_chunk,
        use_adaptive_coarse_visual=True,
        use_ocr_page_coarse=True,
    )


def run_adaptive_fusion_dynamic_rules_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Evaluate the staged rule-based dynamic fusion family."""
    return _run_dynamic_fusion_family_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=topk,
        k_values=k_values,
        checkpoint_path=checkpoint_path,
        stage_name="dynamic_rules",
        checkpoint_subdir="adaptive_fusion_dynamic_rules",
    )


def run_adaptive_fusion_learned_gating_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Evaluate the staged learned global-gating family."""
    return _run_dynamic_fusion_family_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=topk,
        k_values=k_values,
        checkpoint_path=checkpoint_path,
        stage_name="learned_gating",
        checkpoint_subdir="adaptive_fusion_learned_gating",
    )


def run_adaptive_fusion_gating_calibrated_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Evaluate the calibrated gating family with score normalization and optional margin loss."""
    dynamic_cfg = getattr(cfg, "dynamic_fusion", {})
    calibration_name = str(getattr(dynamic_cfg, "calibration_option", "raw"))
    loss_type = str(getattr(dynamic_cfg, "loss_type", "pointwise_bce"))
    loss_suffix = "margin" if loss_type == "pairwise_margin" else "pointwise"
    stage_suffix = f"{calibration_name}_{loss_suffix}"
    return _run_dynamic_fusion_family_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=topk,
        k_values=k_values,
        checkpoint_path=checkpoint_path,
        stage_name="gating_calibrated",
        checkpoint_subdir=f"adaptive_fusion_gating_calibrated_{stage_suffix}",
    )


def _run_dynamic_fusion_family_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int,
    k_values: list[int] | None,
    checkpoint_path: str | None,
    stage_name: str,
    checkpoint_subdir: str,
) -> tuple[list[dict], dict]:
    """Run one staged dynamic fusion family on a dataset split without changing old branches."""
    logger = get_logger(f"adaptive_fusion_{stage_name}_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running staged dynamic fusion on %s with %d samples stage=%s topk=%d", dataset_path, len(records), stage_name, topk)
    logger.info(
        "visual_backend=%s ocr_backend=%s visual_adaptive_coarse=%s visual_bm25_coarse=%s "
        "ocr_page_coarse=%s ocr_bm25_coarse=%s calibration=%s loss_type=%s",
        "colqwen",
        "ocr_page_bm25_bge_rerank",
        bool(getattr(cfg.retrieval_router, "enable_adaptive_coarse", False)),
        bool(getattr(cfg.retrieval_router, "enable_bm25_coarse", True)),
        bool(getattr(cfg.ocr_router, "enable_ocr_page_coarse", False)),
        bool(getattr(cfg.ocr_router, "enable_bm25_coarse", True)),
        str(getattr(getattr(cfg, "dynamic_fusion", {}), "calibration_option", "raw")),
        str(getattr(getattr(cfg, "dynamic_fusion", {}), "loss_type", "pointwise_bce")),
    )

    scorer, gate_model = _load_dynamic_fusion_bundle(
        cfg,
        stage_name=stage_name,
        checkpoint_subdir=checkpoint_subdir,
        checkpoint_path=checkpoint_path,
    )
    question_encoder = QuestionEncoder()
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] = {}
    visual_retriever = ColQwenRetriever(cfg)
    visual_indexed_docs: set[str] = set()
    visual_coarse_doc_text_cache: dict[str, list[str]] = {}
    visual_coarse_doc_retriever_cache: dict[str, Any] = {}
    ocr_page_retriever = OCRBGERetriever(cfg, config_attr="ocr_semantic_retrieval")
    ocr_page_reranker = OCRBGEReranker(cfg, config_attr="ocr_reranker") if bool(getattr(cfg.ocr_reranker, "enable_bge_reranker", True)) else None
    ocr_indexed_docs: set[str] = set()
    ocr_coarse_doc_text_cache: dict[str, list[str]] = {}
    ocr_coarse_doc_retriever_cache: dict[str, Any] = {}
    predictions: list[dict] = []
    cache_stats = {
        "doc_visual_cache_hits": 0,
        "doc_visual_cache_misses": 0,
        "skipped_empty_candidates": 0,
    }
    visual_coarse_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="retrieval_router", stats_prefix="visual")
    ocr_coarse_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="ocr_router", stats_prefix="ocr_bm25")
    ocr_page_pipeline_stats = _build_ocr_page_pipeline_metric_accumulator()
    weight_debug: list[dict[str, Any]] = []
    calibration_debug: dict[str, list[dict[str, float]]] = {"visual_pre": [], "visual_post": [], "ocr_pre": [], "ocr_post": []}

    for sample in tqdm(records, desc=f"DynamicFusion {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        ocr_result, ocr_stats = _retrieve_ocr_with_bm25_bge_reranker_for_sample(
            cfg=cfg,
            sample=sample,
            topk=topk,
            retriever=ocr_page_retriever,
            indexed_docs=ocr_indexed_docs,
            reranker=ocr_page_reranker,
            bm25_doc_text_cache=ocr_coarse_doc_text_cache,
            bm25_doc_retriever_cache=ocr_coarse_doc_retriever_cache,
        )
        visual_result, visual_stats = _retrieve_visual_with_adaptive_coarse_for_sample(
            cfg=cfg,
            sample=sample,
            retriever=visual_retriever,
            indexed_docs=visual_indexed_docs,
            coarse_doc_text_cache=visual_coarse_doc_text_cache,
            coarse_doc_retriever_cache=visual_coarse_doc_retriever_cache,
        )
        _accumulate_adaptive_coarse_metrics(visual_coarse_stats, visual_stats, stats_prefix="visual")
        _accumulate_adaptive_coarse_metrics(ocr_coarse_stats, ocr_stats, stats_prefix="ocr_bm25")
        _accumulate_ocr_page_pipeline_metrics(ocr_page_pipeline_stats, ocr_stats)
        for key, value in ocr_stats.items():
            if key in ocr_coarse_stats or key in ocr_page_pipeline_stats:
                continue
            cache_stats[key] = cache_stats.get(key, 0) + value
        for key, value in visual_stats.items():
            if key in visual_coarse_stats:
                continue
            cache_stats[key] = cache_stats.get(key, 0) + value

        candidate_rows = build_candidate_features_visual_colqwen_ocr_chunk(
            sample,
            ocr_result,
            visual_result,
            ocr_page_texts=ocr_coarse_doc_text_cache.get(doc_id),
            question_encoder=question_encoder,
            ocr_quality_cache=ocr_quality_cache,
        )
        if not candidate_rows:
            cache_stats["skipped_empty_candidates"] += 1
            continue
        if stage_name == "gating_calibrated":
            candidate_rows, calibration_stats = calibrate_route_scores(
                candidate_rows,
                option=str(getattr(getattr(cfg, "dynamic_fusion", {}), "calibration_option", "raw")),
            )
            for key in calibration_debug:
                calibration_debug[key].append(calibration_stats.get(key, {}))
        gating_features = build_dynamic_gating_feature_vector({"question": sample.get("question", "")}, candidate_rows)
        if stage_name == "dynamic_rules":
            alpha_v, alpha_o, debug_meta = compute_rule_based_weights(
                candidate_rows,
                sample,
                variant=str(getattr(getattr(cfg, "dynamic_fusion", {}), "rule_variant", "combined")),
                cfg=cfg,
            )
        else:
            assert gate_model is not None
            alpha_v, alpha_o = gate_model.predict_weights(gating_features["feature_vector"])
            debug_meta = compute_rule_based_weights(candidate_rows, sample, variant="combined", cfg=cfg)[2]
        weight_debug.append({"alpha_v": alpha_v, "alpha_o": alpha_o, **debug_meta})
        weighted_rows = refresh_candidate_feature_vectors(
            apply_branch_reweighting(candidate_rows, alpha_v=alpha_v, alpha_o=alpha_o)
        )
        ranked = scorer.rank_candidates(weighted_rows)
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": ranked["page_ids"][:topk],
                "pred_scores": ranked["scores"][:topk],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics.update(
        {
            "num_samples": len(predictions),
            "num_unique_docs": len(set(item["doc_id"] for item in records)),
            **cache_stats,
        }
    )
    metrics.update(summarize_adaptive_coarse_stats(visual_coarse_stats, len(records), stats_prefix="visual"))
    metrics.update(summarize_adaptive_coarse_stats(ocr_coarse_stats, len(records), stats_prefix="ocr_bm25"))
    metrics.update(add_ocr_bm25_metric_aliases(metrics))
    metrics.update(summarize_ocr_page_pipeline_metrics(ocr_page_pipeline_stats, len(records)))
    dynamic_metrics = summarize_weight_debug(weight_debug)
    dynamic_metrics["rule_variant_name"] = str(getattr(getattr(cfg, "dynamic_fusion", {}), "rule_variant", "combined"))
    if any(calibration_debug[key] for key in calibration_debug):
        dynamic_metrics["calibration_debug"] = {
            key: {
                "mean": float(sum(item.get("mean", 0.0) for item in values) / max(len(values), 1)),
                "std": float(sum(item.get("std", 0.0) for item in values) / max(len(values), 1)),
                "min": float(sum(item.get("min", 0.0) for item in values) / max(len(values), 1)),
                "max": float(sum(item.get("max", 0.0) for item in values) / max(len(values), 1)),
            }
            for key, values in calibration_debug.items()
        }
    metrics.update(dynamic_metrics)
    logger.info("adaptive_fusion_%s metrics: %s", stage_name, metrics)
    return predictions, metrics

def run_adaptive_fusion_mlp_ocrq_hybrid_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run the current-best fusion structure with hybrid OCR retrieval."""
    return _run_adaptive_fusion_mlp_ocrq_with_backend_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=topk,
        k_values=k_values,
        checkpoint_path=checkpoint_path,
        variant="mlp_ocrq_hybrid",
        checkpoint_subdir="adaptive_fusion_mlp_ocrq_hybrid",
        ocr_backend="ocr_hybrid",
    )


def run_adaptive_fusion_mlp_ocrq_chunkplus_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run the chunkplus adaptive fusion variant with chunk-aware OCR features."""
    return _run_adaptive_fusion_mlp_ocrq_with_backend_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=topk,
        k_values=k_values,
        checkpoint_path=checkpoint_path,
        variant="mlp_ocrq_chunkplus",
        checkpoint_subdir="adaptive_fusion_mlp_ocrq_chunkplus",
        ocr_backend="bge_chunk_rerank",
        feature_builder=build_candidate_features_mlp_ocrq_chunkplus,
    )


def run_adaptive_fusion_mlp_ocrq_nvchunk_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int = 10,
    k_values: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run the current-best fusion structure with NV chunk OCR retrieval."""
    return _run_adaptive_fusion_mlp_ocrq_with_backend_on_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        topk=topk,
        k_values=k_values,
        checkpoint_path=checkpoint_path,
        variant="mlp_ocrq_nvchunk",
        checkpoint_subdir="adaptive_fusion_mlp_ocrq_nvchunk",
        ocr_backend="nv_chunk",
    )


def _run_adaptive_fusion_mlp_ocrq_with_backend_on_dataset(
    cfg: Any,
    dataset_path: str,
    topk: int,
    k_values: list[int] | None,
    checkpoint_path: str | None,
    variant: str,
    checkpoint_subdir: str,
    ocr_backend: str,
    visual_backend: str = "colpali",
    feature_builder: Any = build_candidate_features_ablate_mlp_ocrq,
    use_adaptive_coarse_visual: bool = False,
    use_ocr_page_coarse: bool = False,
) -> tuple[list[dict], dict]:
    """Run the mlp_ocrq fusion structure with a selectable OCR backend."""
    if visual_backend == "old_visual":
        visual_backend = "colpali"
    if visual_backend not in {"colpali", "colqwen"}:
        raise ValueError(f"Unsupported visual backend for adaptive fusion inference: {visual_backend}")
    logger = get_logger(f"adaptive_fusion_{variant}_retrieval")
    records = load_jsonl(dataset_path)
    logger.info("Running adaptive_fusion_%s on %s with %d samples and topk=%d", variant, dataset_path, len(records), topk)
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

    model = _load_adaptive_ablation_mlp_model_from_checkpoint(
        cfg,
        variant=variant,
        checkpoint_path=checkpoint_path,
        checkpoint_subdir=checkpoint_subdir,
    )
    question_encoder = QuestionEncoder()
    bm25_text_cache: dict[str, list[str]] = {}
    visual_coarse_doc_text_cache: dict[str, list[str]] = {}
    visual_coarse_doc_retriever_cache: dict[str, Any] = {}
    ocr_coarse_doc_text_cache: dict[str, list[str]] = {}
    ocr_coarse_doc_retriever_cache: dict[str, Any] = {}
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] = {}
    visual_retriever = _create_visual_retriever_for_fusion(cfg, visual_backend=visual_backend)
    visual_indexed_docs: set[str] = set()
    ocr_retriever = OCRBGERetriever(cfg) if ocr_backend == "bge_rerank" else None
    ocr_reranker = OCRBGEReranker(cfg) if ocr_backend == "bge_rerank" else None
    ocr_page_retriever = OCRBGERetriever(cfg, config_attr="ocr_semantic_retrieval") if ocr_backend == "ocr_page_bm25_bge_rerank" else None
    ocr_page_reranker = (
        OCRBGEReranker(cfg, config_attr="ocr_reranker")
        if ocr_backend == "ocr_page_bm25_bge_rerank" and bool(getattr(ocr_reranker_cfg, "enable_bge_reranker", True))
        else None
    )
    ocr_indexed_docs: set[str] = set()
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
    predictions: list[dict] = []
    cache_stats = {
        "doc_visual_cache_hits": 0,
        "doc_visual_cache_misses": 0,
        "skipped_empty_candidates": 0,
    }
    visual_coarse_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="retrieval_router", stats_prefix="visual") if use_adaptive_coarse_visual else {}
    ocr_stats_prefix = "ocr_bm25" if ocr_backend == "ocr_page_bm25_bge_rerank" else "ocr"
    ocr_coarse_stats = _build_adaptive_coarse_metric_accumulator(cfg, router_attr="ocr_router", stats_prefix=ocr_stats_prefix) if use_ocr_page_coarse else {}
    ocr_page_pipeline_stats = _build_ocr_page_pipeline_metric_accumulator() if ocr_backend == "ocr_page_bm25_bge_rerank" else {}

    for sample in tqdm(records, desc=f"AdaptiveFusion{variant} {Path(dataset_path).stem}", unit="sample"):
        doc_id = str(sample["doc_id"])
        if ocr_backend == "bge_rerank":
            assert ocr_retriever is not None and ocr_reranker is not None
            ocr_result, ocr_stats = _retrieve_ocr_bge_for_sample(
                cfg,
                sample,
                topk=topk,
                retriever=ocr_retriever,
                indexed_docs=ocr_indexed_docs,
                reranker=ocr_reranker,
                coarse_topn=int(cfg.ocr_retrieval.coarse_topn),
            )
            ocr_page_texts = ocr_retriever.get_document_page_texts(doc_id)
        elif ocr_backend == "ocr_page_bm25_bge_rerank":
            assert ocr_page_retriever is not None
            ocr_result, ocr_stats = _retrieve_ocr_with_bm25_bge_reranker_for_sample(
                cfg=cfg,
                sample=sample,
                topk=topk,
                retriever=ocr_page_retriever,
                indexed_docs=ocr_indexed_docs,
                reranker=ocr_page_reranker,
                bm25_doc_text_cache=ocr_coarse_doc_text_cache,
                bm25_doc_retriever_cache=ocr_coarse_doc_retriever_cache,
            )
            if use_ocr_page_coarse:
                _accumulate_adaptive_coarse_metrics(ocr_coarse_stats, ocr_stats, stats_prefix=ocr_stats_prefix)
            _accumulate_ocr_page_pipeline_metrics(ocr_page_pipeline_stats, ocr_stats)
            ocr_page_texts = ocr_coarse_doc_text_cache.get(doc_id)
        elif ocr_backend == "bge_chunk_rerank":
            assert chunk_builder is not None and ocr_chunk_retriever is not None and ocr_chunk_reranker is not None
            if use_ocr_page_coarse:
                ocr_result, ocr_stats = _retrieve_ocr_with_page_coarse_for_sample(
                    cfg=cfg,
                    sample=sample,
                    topk=topk,
                    chunk_builder=chunk_builder,
                    retriever=ocr_chunk_retriever,
                    indexed_docs=ocr_chunk_indexed_docs,
                    reranker=ocr_chunk_reranker,
                    chunk_cache=ocr_chunk_cache,
                    coarse_doc_text_cache=ocr_coarse_doc_text_cache,
                    coarse_doc_retriever_cache=ocr_coarse_doc_retriever_cache,
                    coarse_topn=int(cfg.ocr_chunk_retrieval.coarse_topn),
                    aggregation_strategy=str(cfg.ocr_chunk_retrieval.aggregate_strategy),
                    query_variant=str(cfg.ocr_chunk_retrieval.query_variant),
                )
                _accumulate_adaptive_coarse_metrics(ocr_coarse_stats, ocr_stats, stats_prefix=ocr_stats_prefix)
                ocr_page_texts = ocr_coarse_doc_text_cache.get(doc_id)
            else:
                ocr_result, ocr_stats = _retrieve_ocr_bge_chunk_for_sample(
                    cfg=cfg,
                    sample=sample,
                    topk=topk,
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
            ocr_result, ocr_stats = _retrieve_ocr_hybrid_for_sample(
                cfg=cfg,
                sample=sample,
                topk=topk,
                hybrid_retriever=ocr_hybrid_retriever,
                chunk_builder=chunk_builder,
                chunk_cache=ocr_chunk_cache,
                bm25_text_cache=bm25_text_cache,
            )
            ocr_page_texts = bm25_text_cache.get(doc_id, [])
        elif ocr_backend == "nv_chunk":
            assert chunk_builder is not None and ocr_nv_chunk_retriever is not None
            ocr_result, ocr_stats = _retrieve_ocr_nv_chunk_for_sample(
                cfg=cfg,
                sample=sample,
                topk=topk,
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
            raise ValueError(f"Unsupported mlp_ocrq OCR backend at inference time: {ocr_backend}")

        if use_adaptive_coarse_visual:
            visual_result, visual_stats = _retrieve_visual_with_adaptive_coarse_for_sample(
                cfg=cfg,
                sample=sample,
                retriever=visual_retriever,
                indexed_docs=visual_indexed_docs,
                coarse_doc_text_cache=visual_coarse_doc_text_cache,
                coarse_doc_retriever_cache=visual_coarse_doc_retriever_cache,
            )
        else:
            visual_result, visual_stats = _retrieve_visual_for_sample(cfg, sample, topk, visual_retriever, visual_indexed_docs)
        for key, value in ocr_stats.items():
            if use_ocr_page_coarse and key in ocr_coarse_stats:
                continue
            if key in ocr_page_pipeline_stats:
                continue
            cache_stats[key] = cache_stats.get(key, 0) + value
        for key, value in visual_stats.items():
            if use_adaptive_coarse_visual and key in visual_coarse_stats:
                continue
            cache_stats[key] = cache_stats.get(key, 0) + value
        if use_adaptive_coarse_visual:
            _accumulate_adaptive_coarse_metrics(visual_coarse_stats, visual_stats, stats_prefix="visual")
        candidate_rows = feature_builder(
            sample,
            ocr_result,
            visual_result,
            ocr_page_texts=ocr_page_texts,
            question_encoder=question_encoder,
            ocr_quality_cache=ocr_quality_cache,
        )
        if not candidate_rows:
            cache_stats["skipped_empty_candidates"] += 1
            continue
        ranked = model.rank_candidates(candidate_rows)
        predictions.append(
            {
                "qid": sample["qid"],
                "doc_id": doc_id,
                "question": sample["question"],
                "evidence_pages": [int(page) for page in sample.get("evidence_pages", [])],
                "pred_page_ids": ranked["page_ids"][:topk],
                "pred_scores": ranked["scores"][:topk],
                "topk": topk,
            }
        )

    if k_values is None:
        k_values = [1, 5, 10]
    labeled_predictions = [item for item in predictions if item.get("evidence_pages")]
    metrics = evaluate_retrieval(labeled_predictions, k_values) if labeled_predictions else {}
    metrics.update(
        {
            "num_samples": len(records) if (use_adaptive_coarse_visual or use_ocr_page_coarse) else len(predictions),
            "num_unique_docs": len(set(item["doc_id"] for item in records)),
            **cache_stats,
        }
    )
    if use_adaptive_coarse_visual:
        metrics.update(summarize_adaptive_coarse_stats(visual_coarse_stats, len(records), stats_prefix="visual"))
    if use_ocr_page_coarse:
        metrics.update(summarize_adaptive_coarse_stats(ocr_coarse_stats, len(records), stats_prefix=ocr_stats_prefix))
        if ocr_stats_prefix == "ocr_bm25":
            metrics.update(add_ocr_bm25_metric_aliases(metrics))
    if ocr_page_pipeline_stats:
        metrics.update(summarize_ocr_page_pipeline_metrics(ocr_page_pipeline_stats, len(records)))
    logger.info("adaptive_fusion_%s metrics: %s", variant, metrics)
    return predictions, metrics


def _create_visual_retriever_for_fusion(cfg: Any, visual_backend: str) -> Any:
    """Create the visual retriever backend requested by a fusion inference variant."""
    if visual_backend in {"colpali", "old_visual"}:
        return ColPaliRetriever(cfg, require_engine=True)
    if visual_backend == "colqwen":
        return ColQwenRetriever(cfg)
    raise ValueError(f"Unsupported visual backend for adaptive fusion inference: {visual_backend}")


def _load_adaptive_model_from_checkpoint(cfg: Any, checkpoint_path: str | None = None) -> AdaptiveFusion:
    """Load a trained adaptive fusion model from checkpoint."""
    resolved_path = checkpoint_path or str(cfg.experiment.checkpoint_path or (Path(cfg.paths.checkpoint_dir) / "adaptive_fusion" / "adaptive_fusion_best.pkl"))
    checkpoint = load_pickle(resolved_path)
    model_state = checkpoint["model_state"]
    model = AdaptiveFusion(
        feature_dim=int(model_state.get("feature_dim", 18)),
        hidden_dim=int(model_state.get("hidden_dim", cfg.fusion.hidden_dim)),
        dropout=float(model_state.get("dropout", cfg.fusion.dropout)),
    )
    model.load_state_dict(model_state)
    return model


def _load_adaptive_model_v2_from_checkpoint(cfg: Any, checkpoint_path: str | None = None) -> AdaptiveFusionV2:
    """Load a trained enhanced adaptive fusion model from checkpoint."""
    resolved_path = checkpoint_path or str(
        cfg.experiment.checkpoint_path or (Path(cfg.paths.checkpoint_dir) / "adaptive_fusion_v2" / "adaptive_fusion_v2_best.pkl")
    )
    checkpoint = load_pickle(resolved_path)
    model_state = checkpoint["model_state"]
    model = AdaptiveFusionV2(
        feature_dim=int(model_state.get("feature_dim", 46)),
        hidden_dim=int(model_state.get("hidden_dim", cfg.fusion.hidden_dim)),
        dropout=float(model_state.get("dropout", cfg.fusion.dropout)),
    )
    model.load_state_dict(model_state)
    return model


def _load_adaptive_ablation_model_from_checkpoint(cfg: Any, variant: str, checkpoint_path: str | None = None) -> AdaptiveFusion:
    """Load an ablation model that keeps the v1 MLP structure."""
    resolved_path = checkpoint_path or str(Path(cfg.paths.checkpoint_dir) / f"adaptive_fusion_{variant}" / f"adaptive_fusion_{variant}_best.pkl")
    checkpoint = load_pickle(resolved_path)
    model_state = checkpoint["model_state"]
    model = AdaptiveFusion(
        feature_dim=int(model_state.get("feature_dim", 18)),
        hidden_dim=int(model_state.get("hidden_dim", cfg.fusion.hidden_dim)),
        dropout=float(model_state.get("dropout", cfg.fusion.dropout)),
    )
    model.load_state_dict(model_state)
    return model


def _load_adaptive_ablation_mlp_model_from_checkpoint(
    cfg: Any,
    variant: str,
    checkpoint_path: str | None = None,
    checkpoint_subdir: str | None = None,
) -> AdaptiveFusionV2:
    """Load an ablation model that only upgrades the MLP structure."""
    subdir = checkpoint_subdir or f"adaptive_fusion_{variant}"
    resolved_path = checkpoint_path or str(Path(cfg.paths.checkpoint_dir) / subdir / f"{subdir}_best.pkl")
    checkpoint = load_pickle(resolved_path)
    model_state = checkpoint["model_state"]
    model = AdaptiveFusionV2(
        feature_dim=int(model_state.get("feature_dim", 18)),
        hidden_dim=int(model_state.get("hidden_dim", cfg.fusion.hidden_dim)),
        dropout=float(model_state.get("dropout", cfg.fusion.dropout)),
    )
    model.load_state_dict(model_state)
    return model


def _load_dynamic_fusion_bundle(
    cfg: Any,
    stage_name: str,
    checkpoint_subdir: str,
    checkpoint_path: str | None = None,
) -> tuple[AdaptiveFusionV2, GateNet | None]:
    """Load the staged dynamic fusion scorer and optional global gate."""
    resolved_path = checkpoint_path or str(Path(cfg.paths.checkpoint_dir) / checkpoint_subdir / f"{checkpoint_subdir}_best.pkl")
    checkpoint = load_pickle(resolved_path)
    model_state = checkpoint["model_state"]
    scorer = AdaptiveFusionV2(
        feature_dim=int(model_state.get("feature_dim", 22)),
        hidden_dim=int(model_state.get("hidden_dim", cfg.fusion.hidden_dim)),
        dropout=float(model_state.get("dropout", cfg.fusion.dropout)),
    )
    scorer.load_state_dict(model_state)
    gate_state = checkpoint.get("gate_state")
    if stage_name == "dynamic_rules" or not gate_state:
        return scorer, None
    gate_model = GateNet(
        input_dim=int(gate_state.get("input_dim", 0)),
        hidden_dim=int(gate_state.get("hidden_dim", getattr(getattr(cfg, "dynamic_fusion", {}), "gate_hidden_dim", 16))),
        min_weight=float(gate_state.get("min_weight", getattr(getattr(cfg, "dynamic_fusion", {}), "min_weight", 0.2))),
        max_weight=float(gate_state.get("max_weight", getattr(getattr(cfg, "dynamic_fusion", {}), "max_weight", 0.8))),
    )
    gate_model.load_state_dict(gate_state)
    return scorer, gate_model


def infer_retrieval(
    cfg: Any,
    sample: dict[str, Any],
    retrieval_bundle: dict[str, Any],
    mode: str = "fixed_fusion",
    question_encoder: QuestionEncoder | None = None,
    adaptive_model: AdaptiveFusion | None = None,
    fixed_fusion: FixedFusion | None = None,
    rrf_fusion: RRFFusion | None = None,
) -> dict[str, Any]:
    """Run retrieval-only inference for one sample under the selected fusion mode."""
    if mode == "visual_only":
        return infer_retrieval_sample(sample, retrieval_bundle, mode, lambda bundle: bundle["visual"])
    if mode in {"ocr_bm25", "text_only"}:
        return infer_retrieval_sample(sample, retrieval_bundle, mode, lambda bundle: bundle["text"])
    if mode == "rrf_fusion":
        fusion = get_or_create_rrf_fusion(cfg, rrf_fusion)
        return infer_retrieval_sample(sample, retrieval_bundle, mode, lambda bundle: fusion.fuse(bundle["candidates"]))
    if mode == "adaptive_fusion":
        encoder = get_or_create_question_encoder(question_encoder)
        candidate_features = build_adaptive_candidate_features(
            sample,
            retrieval_bundle["text"],
            retrieval_bundle["visual"],
            question_encoder=encoder,
        )
        if not candidate_features:
            raise ValueError(
                "Adaptive fusion received no candidate features. "
                f"qid={sample.get('qid')} doc_id={sample.get('doc_id')} mode={mode} candidate_count=0"
            )
        try:
            feature_dim = infer_candidate_feature_dim(candidate_features)
        except Exception as exc:
            raise ValueError(
                "Failed to infer adaptive fusion feature dimension. "
                f"qid={sample.get('qid')} doc_id={sample.get('doc_id')} mode={mode} "
                f"candidate_count={len(candidate_features)}"
            ) from exc

        model = adaptive_model or AdaptiveFusion(
            feature_dim=feature_dim,
            hidden_dim=int(cfg.fusion.hidden_dim),
            dropout=float(cfg.fusion.dropout),
        )
        try:
            result = model.rank_candidates(candidate_features)
        except Exception as exc:
            raise RuntimeError(
                "Adaptive fusion ranking failed. "
                f"qid={sample.get('qid')} doc_id={sample.get('doc_id')} mode={mode} "
                f"feature_dim={feature_dim} candidate_count={len(candidate_features)}"
            ) from exc
        return {
            "qid": sample["qid"],
            "doc_id": sample["doc_id"],
            "mode": mode,
            "page_ids": result["page_ids"],
            "scores": result["scores"],
            "ranks": result["ranks"],
            "routing_weights": result.get("routing_weights", {}),
        }
    fusion = get_or_create_fixed_fusion(cfg, fixed_fusion)
    return infer_retrieval_sample(sample, retrieval_bundle, mode, lambda bundle: fusion.fuse(bundle["candidates"]))
