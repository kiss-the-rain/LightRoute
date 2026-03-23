"""Candidate page merging for dual-route retrieval outputs."""

from __future__ import annotations

from typing import Any


def merge_candidates(
    visual_results: dict[str, list],
    text_results: dict[str, list],
    mode: str = "union",
    page_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge visual and OCR candidate lists into a unified page candidate set."""
    if mode not in {"union", "intersection"}:
        raise ValueError(f"Unsupported candidate merge mode: {mode}")

    page_metadata = page_metadata or {}
    merged: dict[str, dict[str, Any]] = {}

    for page_id, score, rank in zip(
        visual_results.get("page_ids", []),
        visual_results.get("scores", []),
        visual_results.get("ranks", []),
    ):
        merged.setdefault(page_id, {"page_id": page_id})
        merged[page_id].update({"visual_score": float(score), "visual_rank": int(rank)})

    for page_id, score, rank in zip(
        text_results.get("page_ids", []),
        text_results.get("scores", []),
        text_results.get("ranks", []),
    ):
        merged.setdefault(page_id, {"page_id": page_id})
        merged[page_id].update({"ocr_score": float(score), "ocr_rank": int(rank)})

    candidates: list[dict[str, Any]] = []
    for page_id, payload in merged.items():
        has_visual = "visual_rank" in payload
        has_text = "ocr_rank" in payload
        if mode == "intersection" and not (has_visual and has_text):
            continue
        metadata = page_metadata.get(page_id, {})
        payload.setdefault("visual_score", 0.0)
        payload.setdefault("ocr_score", 0.0)
        payload.setdefault("visual_rank", 0)
        payload.setdefault("ocr_rank", 0)
        payload["overlap_flag"] = int(has_visual and has_text)
        payload["ocr_avg_confidence"] = metadata.get("ocr_avg_confidence", 0.0)
        payload["ocr_token_count"] = metadata.get("ocr_token_count", 0)
        candidates.append(payload)

    candidates.sort(key=lambda item: (item["visual_rank"] or 10**9, item["ocr_rank"] or 10**9))
    return candidates
