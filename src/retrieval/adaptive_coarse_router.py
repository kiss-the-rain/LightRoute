"""Adaptive coarse retrieval helpers for layered branch routing."""

from __future__ import annotations

import re
from typing import Any

from src.retrieval.bm25_retriever import BM25Retriever, load_page_ocr_text


def route_document_pages_with_adaptive_coarse(
    cfg: Any,
    sample: dict[str, Any],
    doc_text_cache: dict[str, list[str]],
    doc_retriever_cache: dict[str, Any],
    router_cfg: Any | None = None,
    stats_prefix: str = "",
    enabled_attr: str = "enable_adaptive_coarse",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Route one document's page pool through bypass-or-BM25 adaptive coarse retrieval."""
    router_cfg = router_cfg or getattr(cfg, "retrieval_router", None)
    enabled = bool(getattr(router_cfg, enabled_attr, False))
    coarse_method = str(getattr(router_cfg, "coarse_method", "bm25"))
    bypass_threshold = int(getattr(router_cfg, "bypass_threshold", 0) or 0)
    coarse_topk = int(getattr(router_cfg, "coarse_topk", 0) or 0)

    if coarse_method != "bm25":
        raise ValueError(f"Unsupported adaptive coarse method: {coarse_method}")

    doc_id = str(sample["doc_id"])
    stats = {
        _metric_key(stats_prefix, "adaptive_coarse_enabled"): float(enabled),
        _metric_key(stats_prefix, "num_pages_before_coarse"): 0,
        _metric_key(stats_prefix, "num_pages_after_coarse"): 0,
        _metric_key(stats_prefix, "num_bypass_samples"): 0,
        _metric_key(stats_prefix, "num_coarse_samples"): 0,
        _metric_key(stats_prefix, "coarse_page_text_cache_hits"): 0,
        _metric_key(stats_prefix, "coarse_page_text_cache_misses"): 0,
        _metric_key(stats_prefix, "coarse_index_cache_hits"): 0,
        _metric_key(stats_prefix, "coarse_index_cache_misses"): 0,
    }

    page_texts = doc_text_cache.get(doc_id)
    if page_texts is None:
        ocr_paths = list(sample.get("ocr_paths", []))
        image_paths = list(sample.get("image_paths", []))
        if ocr_paths:
            page_texts = [load_page_ocr_text(path) for path in ocr_paths]
        else:
            page_texts = ["" for _ in image_paths]
        if image_paths and len(page_texts) < len(image_paths):
            page_texts.extend(["" for _ in range(len(image_paths) - len(page_texts))])
        doc_text_cache[doc_id] = page_texts
        stats[_metric_key(stats_prefix, "coarse_page_text_cache_misses")] += 1
    else:
        stats[_metric_key(stats_prefix, "coarse_page_text_cache_hits")] += 1

    page_ids = _extract_sample_page_ids(sample, len(page_texts))
    if len(page_ids) < len(page_texts):
        page_texts = page_texts[: len(page_ids)]
    elif len(page_ids) > len(page_texts):
        page_ids = page_ids[: len(page_texts)]

    num_pages = len(page_ids)
    stats[_metric_key(stats_prefix, "num_pages_before_coarse")] = num_pages
    metadata = {
        "num_pages_before": num_pages,
        "num_pages_after": num_pages,
        "bypassed": True,
        "coarse_method": coarse_method,
        "bypass_threshold": bypass_threshold,
        "coarse_topk": coarse_topk,
    }

    if num_pages == 0:
        return {"page_ids": [], "scores": [], "ranks": [], "metadata": metadata}, stats

    if not enabled or num_pages <= max(bypass_threshold, 0):
        stats[_metric_key(stats_prefix, "num_bypass_samples")] = 1
        stats[_metric_key(stats_prefix, "num_pages_after_coarse")] = num_pages
        return {
            "page_ids": list(page_ids),
            "scores": [0.0 for _ in page_ids],
            "ranks": list(range(1, len(page_ids) + 1)),
            "metadata": metadata,
        }, stats

    if doc_id in doc_retriever_cache:
        retriever = doc_retriever_cache[doc_id]
        stats[_metric_key(stats_prefix, "coarse_index_cache_hits")] += 1
    else:
        retriever = BM25Retriever(cfg)
        retriever.build_index(page_texts=page_texts, page_ids=page_ids)
        doc_retriever_cache[doc_id] = retriever
        stats[_metric_key(stats_prefix, "coarse_index_cache_misses")] += 1

    effective_topk = min(num_pages, coarse_topk) if coarse_topk > 0 else num_pages
    routed = retriever.retrieve(sample["question"], topk=effective_topk)
    stats[_metric_key(stats_prefix, "num_coarse_samples")] = 1
    stats[_metric_key(stats_prefix, "num_pages_after_coarse")] = len(routed.get("page_ids", []))
    metadata.update(
        {
            "num_pages_after": len(routed.get("page_ids", [])),
            "bypassed": False,
        }
    )
    routed["metadata"] = metadata
    return routed, stats


def add_adaptive_coarse_recall_stats(
    stats: dict[str, Any],
    routed_page_ids: list[int],
    evidence_pages: list[int] | None,
    cutoffs: tuple[int, ...] = (10, 20, 50),
    stats_prefix: str = "",
) -> None:
    """Accumulate coarse recall counts in-place when evidence labels are available."""
    if not evidence_pages:
        return
    evidence_set = {int(page_id) for page_id in evidence_pages}
    routed_list = [int(page_id) for page_id in routed_page_ids]
    for cutoff in cutoffs:
        key = _metric_key(stats_prefix, f"coarse_recall@{cutoff}")
        hit = float(any(page_id in evidence_set for page_id in routed_list[:cutoff]))
        stats[key] = float(stats.get(key, 0.0)) + hit


def summarize_adaptive_coarse_stats(stats: dict[str, Any], num_samples: int, stats_prefix: str = "") -> dict[str, Any]:
    """Convert accumulated adaptive coarse counters into report-ready metrics."""
    denominator = float(max(num_samples, 1))
    summary = dict(stats)
    before_key = _metric_key(stats_prefix, "num_pages_before_coarse")
    after_key = _metric_key(stats_prefix, "num_pages_after_coarse")
    bypass_key = _metric_key(stats_prefix, "num_bypass_samples")
    summary[_metric_key(stats_prefix, "avg_num_pages_before_coarse")] = float(stats.get(before_key, 0.0)) / denominator
    summary[_metric_key(stats_prefix, "avg_num_pages_after_coarse")] = float(stats.get(after_key, 0.0)) / denominator
    summary[_metric_key(stats_prefix, "bypass_ratio")] = float(stats.get(bypass_key, 0.0)) / denominator
    for cutoff in (10, 20, 50):
        key = _metric_key(stats_prefix, f"coarse_recall@{cutoff}")
        if key in summary:
            summary[key] = float(summary[key]) / denominator
    return summary


def extract_sample_page_ids(sample: dict[str, Any], max_len: int) -> list[int]:
    """Derive canonical page ids from page assets while preserving original document numbering."""
    candidate_paths = sample.get("image_paths") or sample.get("ocr_paths") or []
    page_ids: list[int] = []
    for path in list(candidate_paths)[:max_len]:
        page_ids.append(_extract_page_idx_from_path(path))
    if not page_ids:
        page_ids = list(range(max_len))
    return page_ids


def filter_sample_to_page_ids(sample: dict[str, Any], allowed_page_ids: list[int]) -> dict[str, Any]:
    """Return a shallow sample copy keeping only assets whose page id is in the allowed subset."""
    allowed = {int(page_id) for page_id in allowed_page_ids}
    filtered = dict(sample)
    for field in ("image_paths", "ocr_paths"):
        values = list(sample.get(field, []))
        filtered[field] = [value for value in values if _extract_page_idx_from_path(str(value)) in allowed]
    return filtered


def _extract_sample_page_ids(sample: dict[str, Any], max_len: int) -> list[int]:
    """Backward-compatible internal alias."""
    return extract_sample_page_ids(sample, max_len)


def _metric_key(prefix: str, base_name: str) -> str:
    """Prefix a metric key when a branch-specific namespace is requested."""
    return f"{prefix}_{base_name}" if prefix else base_name


def _extract_page_idx_from_path(path_or_name: str) -> int:
    """Extract an integer page index from names like `docid_p80.jpg`."""
    match = re.search(r"_p(\d+)(?:\.[^.]+)?$", str(path_or_name))
    if not match:
        raise ValueError(f"Unable to extract page index from path: {path_or_name}")
    return int(match.group(1))
