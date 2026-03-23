"""Retrieval-only inference helpers for Phase 1 baselines."""

from __future__ import annotations

from typing import Any, Callable


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
