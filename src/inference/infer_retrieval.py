"""Retrieval-only inference helpers with an OCR-only BM25 baseline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Callable

from tqdm import tqdm

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.retrieval.colpali_retriever import ColPaliRetriever
from src.retrieval.colqwen_retriever import ColQwenRetriever
from src.models.adaptive_fusion import AdaptiveFusion, AdaptiveFusionV2
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
from src.retrieval.ocr_page_pipeline import run_ocr_page_pipeline_for_sample
from src.retrieval.fusion_features import (
    build_candidate_features as build_adaptive_candidate_features,
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
        "visual_backend=%s adaptive_coarse=%s bypass_threshold=%s coarse_topk=%s",
        "colqwen",
        bool(getattr(router_cfg, "enable_adaptive_coarse", False)),
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
        "ocr_backend=%s ocr_page_coarse=%s bypass_threshold=%s coarse_topk=%s",
        "bge_chunk_rerank",
        bool(getattr(ocr_router_cfg, "enable_ocr_page_coarse", False)),
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
        "ocr_backend=%s ocr_bm25_coarse=%s ocr_bypass_threshold=%s ocr_coarse_topk=%s ocr_semantic_topk=%s ocr_rerank_topk=%s",
        "ocr_page_bm25_bge_rerank",
        bool(getattr(ocr_router_cfg, "enable_ocr_page_coarse", False)),
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
