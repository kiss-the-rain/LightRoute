"""Lightweight query-aware routing helpers for MP-DocVQA retrieval."""

from __future__ import annotations

from typing import Any, Dict


def extract_query_features(query: str) -> Dict[str, Any]:
    """Extract simple lexical query features for routing decisions."""
    q = str(query or "").lower()
    return {
        "has_number": any(c.isdigit() for c in q),
        "has_percentage": "%" in q or "percent" in q,
        "has_year": any(y in q for y in ["201", "202"]),
        "is_count_question": "how many" in q,
        "is_amount_question": any(k in q for k in ["amount", "total", "cost"]),
        "is_definition": any(k in q for k in ["what is", "define"]),
        "is_reasoning": any(k in q for k in ["why", "explain"]),
        "length": len(q.split()),
    }


def classify_query_type(features: Dict[str, Any]) -> str:
    """Classify the query into a coarse routing bucket."""
    if (
        features["has_number"]
        or features["has_percentage"]
        or features["has_year"]
        or features["is_count_question"]
        or features["is_amount_question"]
    ):
        return "ocr_query"

    if features["is_reasoning"] or features["length"] > 12:
        return "visual_query"

    return "generic_query"


def compute_visual_confidence(visual_results: list[dict[str, Any]]) -> float:
    """Return top1-top2 visual score margin."""
    if len(visual_results) < 2:
        return 0.0
    s1 = float(visual_results[0].get("score", 0.0))
    s2 = float(visual_results[1].get("score", 0.0))
    return float(s1 - s2)
