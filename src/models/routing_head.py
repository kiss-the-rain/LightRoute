"""Soft routing head placeholder for later adaptive fusion phases."""

from __future__ import annotations

from typing import Any


class RoutingHead:
    """Return soft routing weights for visual, text, and hybrid paths."""

    def predict(self, features: dict[str, Any]) -> dict[str, float]:
        return {"visual": 0.33, "text": 0.33, "hybrid": 0.34}
