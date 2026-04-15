"""Lightweight page-level OCR retrieval pipeline for the upgraded layered branch."""

from __future__ import annotations

from typing import Any

from src.retrieval.adaptive_coarse_router import (
    add_adaptive_coarse_recall_stats,
    extract_sample_page_ids,
    route_document_pages_with_adaptive_coarse,
)
from src.retrieval.bm25_retriever import load_page_ocr_text
from src.retrieval.ocr_bge_retriever import OCRBGERetriever
from src.retrieval.ocr_bge_reranker import OCRBGEReranker


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
    semantic_topk: int | None = None,
) -> tuple[dict[str, list], dict[str, Any]]:
    """Run page-level dense retrieval with BGE-M3 on a routed OCR page subset."""
    doc_id = str(sample["doc_id"])
    doc_key = f"ocr_page:{doc_id}"
    stats = {
        "ocr_bge_embedding_cache_hits": 0,
        "ocr_bge_embedding_cache_misses": 0,
        "ocr_num_pages_after_bge": 0,
    }
    page_texts = page_text_cache.get(doc_id)
    if page_texts is None:
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

    dense_result = retriever.retrieve_subset(
        query=str(sample.get("question", "")),
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
    reranked = reranker.rerank(str(sample.get("question", "")), candidates, topk=effective_topk)
    stats["ocr_rerank_calls"] = 1
    stats["ocr_num_pages_after_rerank"] = len(reranked.get("page_ids", []))
    return reranked, stats


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
    doc_key = f"ocr_page:{sample['doc_id']}"
    semantic_cfg = getattr(cfg, "ocr_semantic_retrieval", None)
    if bool(getattr(semantic_cfg, "enable_bge_m3", True)):
        semantic_result, semantic_stats = run_ocr_page_bge_m3(
            cfg=cfg,
            sample=sample,
            retriever=retriever,
            indexed_docs=indexed_docs,
            candidate_page_ids=routed_page_ids,
            page_text_cache=bm25_doc_text_cache,
        )
    else:
        page_texts = bm25_doc_text_cache.get(str(sample["doc_id"])) or [load_page_ocr_text(path) for path in sample.get("ocr_paths", [])]
        page_ids = extract_sample_page_ids(sample, len(page_texts))
        if doc_key not in indexed_docs:
            retriever.build_document_index(doc_id=doc_key, page_texts=page_texts, page_ids=page_ids)
            indexed_docs.add(doc_key)
        semantic_result = {
            "page_ids": list(routed_page_ids),
            "scores": [float(score) for score in route_result.get("scores", [])[: len(routed_page_ids)]],
            "ranks": list(range(1, len(routed_page_ids) + 1)),
        }
        semantic_stats = {
            "ocr_bge_embedding_cache_hits": 0,
            "ocr_bge_embedding_cache_misses": 0,
            "ocr_num_pages_after_bge": len(routed_page_ids),
        }
    reranker_cfg = getattr(cfg, "ocr_reranker", None)
    reranked_result, rerank_stats = run_ocr_page_reranker(
        cfg=cfg,
        sample=sample,
        reranker=reranker,
        retriever=retriever,
        retriever_doc_id=doc_key,
        semantic_result=semantic_result,
        rerank_topk=int(getattr(reranker_cfg, "rerank_topk", 0) or topk),
    )
    reranked_result["ocr_page_coarse"] = route_result.get("metadata", {})
    reranked_result["ocr_bge_stage"] = {
        "semantic_topk": int(getattr(getattr(cfg, "ocr_semantic_retrieval", None), "semantic_topk", 0) or 0),
        "num_pages_after_bge": int(semantic_stats.get("ocr_num_pages_after_bge", 0)),
    }
    reranked_result["ocr_reranker_stage"] = {
        "rerank_topk": int(getattr(getattr(cfg, "ocr_reranker", None), "rerank_topk", 0) or 0),
        "num_pages_after_rerank": int(rerank_stats.get("ocr_num_pages_after_rerank", 0)),
    }
    reranked_result["routed_page_ids"] = routed_page_ids
    stats = {}
    stats.update(route_stats)
    stats.update(semantic_stats)
    stats.update(rerank_stats)
    return reranked_result, stats
