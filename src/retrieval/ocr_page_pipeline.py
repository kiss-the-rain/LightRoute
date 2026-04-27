"""Lightweight page-level OCR retrieval pipeline for the upgraded layered branch."""

from __future__ import annotations

import logging
from typing import Any

from src.retrieval.adaptive_coarse_router import (
    add_adaptive_coarse_recall_stats,
    extract_sample_page_ids,
    route_document_pages_with_adaptive_coarse,
)
from src.retrieval.bm25_retriever import load_page_ocr_text
from src.retrieval.ocr_bge_retriever import OCRBGERetriever, format_bge_query
from src.retrieval.ocr_bge_reranker import OCRBGEReranker
from src.utils.chunking import chunk_text

logger = logging.getLogger(__name__)


def _resolve_ocr_query_text(
    sample: dict[str, Any],
    *,
    query_text: str | None = None,
    query_variant: str = "raw_question",
) -> str:
    """Resolve and format the query text used by OCR BGE retrieval/rerank."""
    raw_query = str(query_text or sample.get("question") or sample.get("query") or "").strip()
    if not raw_query:
        return ""
    return format_bge_query(raw_query, variant=query_variant)


def run_ocr_page_bm25_coarse(
    cfg: Any,
    sample: dict[str, Any],
    doc_text_cache: dict[str, list[str]],
    doc_retriever_cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run page-level BM25 coarse routing for the OCR branch."""
    route_result, route_stats = route_document_pages_with_adaptive_coarse(
        cfg=cfg,
        sample=sample,
        doc_text_cache=doc_text_cache,
        doc_retriever_cache=doc_retriever_cache,
        router_cfg=getattr(cfg, "ocr_router", None),
        stats_prefix="ocr_bm25",
        enabled_attr="enable_ocr_page_coarse",
    )
    routed_page_ids = [int(page_id) for page_id in route_result.get("page_ids", [])]
    add_adaptive_coarse_recall_stats(route_stats, routed_page_ids, sample.get("evidence_pages", []), stats_prefix="ocr_bm25")
    return route_result, route_stats


def run_ocr_page_bge_m3(
    cfg: Any,
    sample: dict[str, Any],
    retriever: OCRBGERetriever,
    indexed_docs: set[str],
    candidate_page_ids: list[int],
    page_text_cache: dict[str, list[str]],
    query_text: str | None = None,
    query_variant: str = "raw_question",
    doc_id_override: str | None = None,
    semantic_topk: int | None = None,
) -> tuple[dict[str, list], dict[str, Any]]:
    """Run page-level dense retrieval with BGE-M3 on a routed OCR page subset."""
    doc_id = str(doc_id_override or sample["doc_id"])
    doc_key = f"ocr_page:{doc_id}"
    stats = {
        "ocr_bge_embedding_cache_hits": 0,
        "ocr_bge_embedding_cache_misses": 0,
        "ocr_num_pages_after_bge": 0,
    }
    page_texts = page_text_cache.get(doc_id)
    if page_texts is None:
        explicit_page_texts = list(sample.get("page_texts", []))
        if explicit_page_texts:
            page_texts = [str(text or "") for text in explicit_page_texts]
        else:
            page_texts = [load_page_ocr_text(path) for path in sample.get("ocr_paths", [])]
        image_paths = list(sample.get("image_paths", []))
        if image_paths and len(page_texts) < len(image_paths):
            page_texts.extend(["" for _ in range(len(image_paths) - len(page_texts))])
        page_text_cache[doc_id] = page_texts
    page_ids = extract_sample_page_ids(sample, len(page_texts))
    if len(page_ids) < len(page_texts):
        page_texts = page_texts[: len(page_ids)]
    elif len(page_ids) > len(page_texts):
        page_ids = page_ids[: len(page_texts)]

    if doc_key in indexed_docs:
        stats["ocr_bge_embedding_cache_hits"] += 1
    else:
        retriever.build_document_index(doc_id=doc_key, page_texts=page_texts, page_ids=page_ids)
        indexed_docs.add(doc_key)
        stats["ocr_bge_embedding_cache_misses"] += 1

    semantic_cfg = getattr(cfg, "ocr_semantic_retrieval", None)
    effective_topk = semantic_topk
    if effective_topk is None:
        effective_topk = int(getattr(semantic_cfg, "semantic_topk", 0) or 0)
    if effective_topk <= 0:
        effective_topk = None
    formatted_query = _resolve_ocr_query_text(sample, query_text=query_text, query_variant=query_variant)

    dense_result = retriever.retrieve_subset(
        query=formatted_query,
        doc_id=doc_key,
        candidate_page_ids=[int(page_id) for page_id in candidate_page_ids],
        topk=effective_topk,
    )
    stats["ocr_num_pages_after_bge"] = len(dense_result.get("page_ids", []))
    return dense_result, stats


def run_ocr_page_reranker(
    cfg: Any,
    sample: dict[str, Any],
    reranker: OCRBGEReranker | None,
    retriever: OCRBGERetriever,
    retriever_doc_id: str,
    semantic_result: dict[str, list],
    query_text: str | None = None,
    query_variant: str = "raw_question",
    rerank_topk: int | None = None,
) -> tuple[dict[str, list], dict[str, Any]]:
    """Rerank page-level OCR candidates with bge-reranker-v2-m3."""
    reranker_cfg = getattr(cfg, "ocr_reranker", None)
    enable_reranker = bool(getattr(reranker_cfg, "enable_bge_reranker", reranker is not None))
    effective_topk = rerank_topk
    if effective_topk is None:
        effective_topk = int(getattr(reranker_cfg, "rerank_topk", 0) or 0)
    if effective_topk <= 0:
        effective_topk = len(semantic_result.get("page_ids", []))

    stats = {
        "ocr_rerank_calls": 0,
        "ocr_num_pages_after_rerank": 0,
    }
    if not semantic_result.get("page_ids"):
        return {"page_ids": [], "scores": [], "ranks": []}, stats

    if not enable_reranker or reranker is None:
        final_result = {
            "page_ids": [int(page_id) for page_id in semantic_result.get("page_ids", [])[:effective_topk]],
            "scores": [float(score) for score in semantic_result.get("scores", [])[:effective_topk]],
            "ranks": list(range(1, min(len(semantic_result.get("page_ids", [])), effective_topk) + 1)),
        }
        stats["ocr_num_pages_after_rerank"] = len(final_result["page_ids"])
        return final_result, stats

    page_ids = retriever.index[retriever_doc_id]["page_ids"]
    page_texts = retriever.get_document_page_texts(retriever_doc_id)
    page_id_to_text = {int(page_id): str(text) for page_id, text in zip(page_ids, page_texts)}
    candidates = [
        {
            "page_id": int(page_id),
            "text": page_id_to_text.get(int(page_id), ""),
            "score": float(score),
        }
        for page_id, score in zip(semantic_result.get("page_ids", []), semantic_result.get("scores", []))
    ]
    formatted_query = _resolve_ocr_query_text(sample, query_text=query_text, query_variant=query_variant)
    reranked = reranker.rerank(formatted_query, candidates, topk=effective_topk)
    stats["ocr_rerank_calls"] = 1
    stats["ocr_num_pages_after_rerank"] = len(reranked.get("page_ids", []))
    return reranked, stats


def run_ocr_page_pipeline_for_subset(
    cfg: Any,
    sample: dict[str, Any],
    candidate_page_ids: list[int],
    topk: int,
    retriever: OCRBGERetriever,
    indexed_docs: set[str],
    reranker: OCRBGEReranker | None = None,
    page_text_cache: dict[str, list[str]] | None = None,
    doc_id_override: str | None = None,
    query_text: str | None = None,
    query_variant: str = "raw_question",
    semantic_topk: int | None = None,
    rerank_topk: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the OCR page BGE+rereank stack on a preselected page subset."""
    page_text_cache = page_text_cache or {}
    effective_doc_id = str(doc_id_override or sample["doc_id"])
    doc_key = f"ocr_page:{effective_doc_id}"
    normalized_candidate_page_ids = [int(page_id) for page_id in candidate_page_ids]
    semantic_cfg = getattr(cfg, "ocr_semantic_retrieval", None)
    if bool(getattr(semantic_cfg, "enable_bge_m3", True)):
        semantic_result, semantic_stats = run_ocr_page_bge_m3(
            cfg=cfg,
            sample=sample,
            retriever=retriever,
            indexed_docs=indexed_docs,
            candidate_page_ids=normalized_candidate_page_ids,
            page_text_cache=page_text_cache,
            query_text=query_text,
            query_variant=query_variant,
            doc_id_override=effective_doc_id,
            semantic_topk=semantic_topk,
        )
    else:
        page_texts = page_text_cache.get(effective_doc_id)
        if page_texts is None:
            explicit_page_texts = list(sample.get("page_texts", []))
            if explicit_page_texts:
                page_texts = [str(text or "") for text in explicit_page_texts]
            else:
                page_texts = [load_page_ocr_text(path) for path in sample.get("ocr_paths", [])]
            page_text_cache[effective_doc_id] = page_texts
        if doc_key not in indexed_docs:
            page_ids = list(sample.get("page_ids", normalized_candidate_page_ids))
            retriever.build_document_index(doc_id=doc_key, page_texts=page_texts, page_ids=page_ids[: len(page_texts)])
            indexed_docs.add(doc_key)
        effective_semantic_topk = len(normalized_candidate_page_ids) if semantic_topk is None else min(int(semantic_topk), len(normalized_candidate_page_ids))
        semantic_result = {
            "page_ids": list(normalized_candidate_page_ids[:effective_semantic_topk]),
            "scores": [0.0 for _ in range(effective_semantic_topk)],
            "ranks": list(range(1, effective_semantic_topk + 1)),
        }
        semantic_stats = {
            "ocr_bge_embedding_cache_hits": 0,
            "ocr_bge_embedding_cache_misses": 0,
            "ocr_num_pages_after_bge": len(semantic_result["page_ids"]),
        }

    effective_rerank_topk = rerank_topk
    if effective_rerank_topk is None:
        effective_rerank_topk = int(getattr(getattr(cfg, "ocr_reranker", None), "rerank_topk", 0) or topk)
    reranked_result, rerank_stats = run_ocr_page_reranker(
        cfg=cfg,
        sample=sample,
        reranker=reranker,
        retriever=retriever,
        retriever_doc_id=doc_key,
        semantic_result=semantic_result,
        query_text=query_text,
        query_variant=query_variant,
        rerank_topk=effective_rerank_topk,
    )
    reranked_result["ocr_bge_stage"] = {
        "semantic_topk": int(semantic_topk or getattr(getattr(cfg, "ocr_semantic_retrieval", None), "semantic_topk", 0) or 0),
        "num_pages_after_bge": int(semantic_stats.get("ocr_num_pages_after_bge", 0)),
    }
    reranked_result["ocr_reranker_stage"] = {
        "rerank_topk": int(effective_rerank_topk or 0),
        "num_pages_after_rerank": int(rerank_stats.get("ocr_num_pages_after_rerank", 0)),
    }
    reranked_result["routed_page_ids"] = list(normalized_candidate_page_ids)
    stats = {}
    stats.update(semantic_stats)
    stats.update(rerank_stats)
    return reranked_result, stats


def run_ocr_page_pipeline_for_subset_with_chunk_rerank(
    cfg: Any,
    sample: dict[str, Any],
    candidate_page_ids: list[int],
    topk: int,
    retriever: OCRBGERetriever,
    indexed_docs: set[str],
    reranker: OCRBGEReranker | None = None,
    page_text_cache: dict[str, list[str]] | None = None,
    doc_id_override: str | None = None,
    query_text: str | None = None,
    query_variant: str = "raw_question",
    semantic_topk: int | None = None,
    rerank_topk: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run page retrieval first, then rerank bounded OCR chunks and aggregate back to pages."""
    page_text_cache = page_text_cache or {}
    effective_doc_id = str(doc_id_override or sample["doc_id"])
    doc_key = f"ocr_page:{effective_doc_id}"
    normalized_candidate_page_ids = [int(page_id) for page_id in candidate_page_ids]
    semantic_result, semantic_stats = run_ocr_page_bge_m3(
        cfg=cfg,
        sample=sample,
        retriever=retriever,
        indexed_docs=indexed_docs,
        candidate_page_ids=normalized_candidate_page_ids,
        page_text_cache=page_text_cache,
        query_text=query_text,
        query_variant=query_variant,
        doc_id_override=effective_doc_id,
        semantic_topk=semantic_topk,
    )
    logger.info("[Pipeline] Using OCR CHUNK rerank branch")

    chunk_cfg = getattr(cfg, "ocr_jina_chunk_reranker", None)
    chunk_size = int(getattr(chunk_cfg, "chunk_size", 256) or 256)
    chunk_overlap = int(getattr(chunk_cfg, "chunk_overlap", 64) or 64)
    chunk_topk_pages = int(getattr(chunk_cfg, "chunk_topk_pages", 20) or 20)
    chunk_max_per_page = int(getattr(chunk_cfg, "chunk_max_per_page", 10) or 10)
    max_chunks_per_query = int(getattr(chunk_cfg, "max_chunks_per_query", chunk_topk_pages * chunk_max_per_page) or 0)
    if max_chunks_per_query <= 0:
        max_chunks_per_query = chunk_topk_pages * chunk_max_per_page

    effective_rerank_topk = rerank_topk
    if effective_rerank_topk is None:
        effective_rerank_topk = int(getattr(getattr(cfg, "ocr_jina_reranker", None), "rerank_topk", 0) or topk)
    if effective_rerank_topk <= 0:
        effective_rerank_topk = len(semantic_result.get("page_ids", []))

    stats = {
        "ocr_rerank_calls": 0,
        "ocr_num_pages_after_rerank": 0,
        "ocr_chunk_rerank_enabled": True,
        "ocr_chunk_topk_pages": int(chunk_topk_pages),
        "ocr_chunk_max_per_page": int(chunk_max_per_page),
        "ocr_chunk_total_chunks": 0,
        "ocr_chunk_rerank_calls": 0,
        "ocr_chunk_scores_changed": False,
    }
    stats.update(semantic_stats)
    if not semantic_result.get("page_ids"):
        logger.error("Chunk pipeline empty -> fallback detected: no semantic pages for doc_id=%s", effective_doc_id)
        raise RuntimeError("Chunk pipeline failed: no semantic pages available for chunk rerank.")

    page_ids = retriever.index[doc_key]["page_ids"]
    retriever_page_texts = retriever.get_document_page_texts(doc_key)
    sample_page_texts = [str(text or "") for text in sample.get("page_texts", [])]
    page_id_to_text: dict[int, str] = {
        int(page_id): str(text or "")
        for page_id, text in enumerate(sample_page_texts)
    }
    for page_id, retriever_text in zip(page_ids, retriever_page_texts):
        int_page_id = int(page_id)
        sample_text = sample_page_texts[int_page_id] if 0 <= int_page_id < len(sample_page_texts) else ""
        page_id_to_text[int_page_id] = str(sample_text or retriever_text or "")
    semantic_scores = {
        int(page_id): float(score)
        for page_id, score in zip(semantic_result.get("page_ids", []), semantic_result.get("scores", []))
    }
    top_semantic_page_ids = [int(page_id) for page_id in semantic_result.get("page_ids", [])[:chunk_topk_pages]]

    chunk_candidates: list[dict[str, Any]] = []
    total_chunks = 0
    for page_id in top_semantic_page_ids:
        text = (
            page_id_to_text.get(int(page_id), "")
            or ""
        )
        if not str(text).strip():
            continue
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )[:chunk_max_per_page]
        total_chunks += len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            chunk_candidates.append(
                {
                    "page_id": int(page_id),
                    "chunk_index": int(chunk_index),
                    "text": chunk,
                    "score": float(semantic_scores.get(int(page_id), 0.0)),
                }
            )
            if len(chunk_candidates) >= max_chunks_per_query:
                break
        if len(chunk_candidates) >= max_chunks_per_query:
            break

    if total_chunks == 0 or not chunk_candidates:
        non_empty_page_text_count = sum(1 for text in page_id_to_text.values() if str(text).strip())
        logger.error(
            "Chunk pipeline empty -> fallback detected: doc_id=%s semantic_page_ids=%s len_page_texts=%d non_empty_page_text_count=%d",
            effective_doc_id,
            top_semantic_page_ids[:10],
            len(page_id_to_text),
            non_empty_page_text_count,
        )
        raise RuntimeError("Chunk pipeline failed: no chunks generated. Check OCR text field.")

    if reranker is None:
        logger.error("Chunk pipeline empty -> fallback detected: reranker is None")
        raise RuntimeError("Chunk fallback not allowed: reranker is None.")

    before_scores = [
        float(semantic_scores.get(int(page_id), 0.0))
        for page_id in semantic_result.get("page_ids", [])
    ]

    formatted_query = _resolve_ocr_query_text(sample, query_text=query_text, query_variant=query_variant)
    logger.info("[Chunk Debug] total_chunks=%d", len(chunk_candidates))
    chunk_reranked = reranker.rerank(formatted_query, chunk_candidates, topk=len(chunk_candidates))
    logger.info("[Chunk Debug] rerank_calls=%d", len(chunk_candidates))
    stats["ocr_rerank_calls"] = len(chunk_candidates)
    stats["ocr_chunk_rerank_calls"] = 1
    stats["ocr_chunk_total_chunks"] = int(total_chunks)

    page_score_map: dict[int, float] = {}
    for page_id, score in zip(chunk_reranked.get("page_ids", []), chunk_reranked.get("scores", [])):
        int_page_id = int(page_id)
        page_score_map[int_page_id] = max(float(score), page_score_map.get(int_page_id, float("-inf")))

    candidate_page_ids_order = [int(page_id) for page_id in semantic_result.get("page_ids", [])]
    scored_pages = [
        (
            page_id,
            float(page_score_map.get(page_id, semantic_scores.get(page_id, 0.0))),
            0 if page_id in page_score_map else 1,
        )
        for page_id in candidate_page_ids_order
    ]
    scored_pages.sort(key=lambda item: (item[2], -item[1]))
    scored_pages = scored_pages[:effective_rerank_topk]
    after_scores = [float(score) for _, score, _ in scored_pages]
    if before_scores[: len(after_scores)] == after_scores:
        logger.warning("Chunk rerank did not change scores for doc_id=%s", effective_doc_id)
    else:
        stats["ocr_chunk_scores_changed"] = True
    final_result = {
        "page_ids": [int(page_id) for page_id, _, _ in scored_pages],
        "scores": [float(score) for _, score, _ in scored_pages],
        "ranks": list(range(1, len(scored_pages) + 1)),
        "ocr_bge_stage": {
            "semantic_topk": int(semantic_topk or getattr(getattr(cfg, "ocr_jina_semantic_retrieval", None), "semantic_topk", 0) or 0),
            "num_pages_after_bge": int(semantic_stats.get("ocr_num_pages_after_bge", 0)),
        },
        "ocr_reranker_stage": {
            "rerank_topk": int(effective_rerank_topk or 0),
            "num_pages_after_rerank": len(scored_pages),
            "chunk_rerank": True,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "num_chunks": int(len(chunk_candidates)),
        },
        "routed_page_ids": list(normalized_candidate_page_ids),
    }
    stats["ocr_num_pages_after_rerank"] = len(final_result["page_ids"])
    return final_result, stats


def run_ocr_page_pipeline_for_sample(
    cfg: Any,
    sample: dict[str, Any],
    topk: int,
    retriever: OCRBGERetriever,
    indexed_docs: set[str],
    reranker: OCRBGEReranker | None = None,
    bm25_doc_text_cache: dict[str, list[str]] | None = None,
    bm25_doc_retriever_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run BM25 page coarse -> BGE-M3 page retrieval -> page reranker for one sample."""
    bm25_doc_text_cache = bm25_doc_text_cache or {}
    bm25_doc_retriever_cache = bm25_doc_retriever_cache or {}
    route_result, route_stats = run_ocr_page_bm25_coarse(
        cfg=cfg,
        sample=sample,
        doc_text_cache=bm25_doc_text_cache,
        doc_retriever_cache=bm25_doc_retriever_cache,
    )
    routed_page_ids = [int(page_id) for page_id in route_result.get("page_ids", [])]
    reranked_result, subset_stats = run_ocr_page_pipeline_for_subset(
        cfg=cfg,
        sample=sample,
        candidate_page_ids=routed_page_ids,
        topk=topk,
        retriever=retriever,
        indexed_docs=indexed_docs,
        reranker=reranker,
        page_text_cache=bm25_doc_text_cache,
    )
    reranked_result["ocr_page_coarse"] = route_result.get("metadata", {})
    reranked_result["routed_page_ids"] = routed_page_ids
    stats = {}
    stats.update(route_stats)
    stats.update(subset_stats)
    return reranked_result, stats
