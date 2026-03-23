"""Adaptive fusion placeholder reserved for later phases."""

from __future__ import annotations

from typing import Any


class AdaptiveFusion:
    """Future quality-aware and question-aware adaptive fusion model."""

    def rank(self, candidates: list[dict[str, Any]], question_features: dict[str, Any]) -> dict[str, Any]:
        return {
            "page_ids": [candidate["page_id"] for candidate in candidates],
            "scores": [0.0 for _ in candidates],
            "ranks": list(range(1, len(candidates) + 1)),
            "routing_weights": {},
        }
