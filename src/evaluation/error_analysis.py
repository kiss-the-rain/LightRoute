"""Basic retrieval error-analysis helpers for baseline comparison."""

from __future__ import annotations

from typing import Any


def analyze_retrieval_cases(
    visual_only_records: list[dict[str, Any]],
    text_only_records: list[dict[str, Any]],
    fusion_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Export simple retrieval case buckets useful for paper analysis."""
    visual_by_qid = {row["qid"]: row for row in visual_only_records}
    text_by_qid = {row["qid"]: row for row in text_only_records}
    fusion_by_qid = {row["qid"]: row for row in fusion_records}

    buckets = {
        "visual_only_success_cases": [],
        "ocr_only_success_cases": [],
        "fusion_improved_cases": [],
    }
    for qid, visual_row in visual_by_qid.items():
        text_row = text_by_qid.get(qid)
        fusion_row = fusion_by_qid.get(qid)
        if text_row is None or fusion_row is None:
            continue

        visual_hit = _is_success(visual_row)
        text_hit = _is_success(text_row)
        fusion_hit = _is_success(fusion_row)

        if visual_hit and not text_hit:
            buckets["visual_only_success_cases"].append(_build_case_summary(visual_row, text_row, fusion_row))
        if text_hit and not visual_hit:
            buckets["ocr_only_success_cases"].append(_build_case_summary(visual_row, text_row, fusion_row))
        if fusion_hit and not (visual_hit and text_hit):
            buckets["fusion_improved_cases"].append(_build_case_summary(visual_row, text_row, fusion_row))
    return buckets


def _is_success(row: dict[str, Any]) -> bool:
    evidence = set(row.get("evidence_page_ids", []))
    return bool(set(row.get("page_ids", [])[:1]) & evidence)


def _build_case_summary(
    visual_row: dict[str, Any],
    text_row: dict[str, Any],
    fusion_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "qid": visual_row["qid"],
        "evidence_page_ids": visual_row.get("evidence_page_ids", []),
        "visual_topk": visual_row.get("page_ids", []),
        "ocr_topk": text_row.get("page_ids", []),
        "fusion_topk": fusion_row.get("page_ids", []),
    }
