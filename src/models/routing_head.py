"""Soft routing head for modality weighting."""

from __future__ import annotations

from typing import Any


class RoutingHead:
    """Return soft routing weights for visual, text, and hybrid paths."""

    def predict(self, features: dict[str, Any]) -> dict[str, float]:
        """Predict normalized routing weights from heuristic features."""
        visual_signal = float(features.get("normalized_vis_score", features.get("ocr_reliability", 0.5)))
        text_signal = float(features.get("normalized_ocr_score", features.get("ocr_reliability", 0.5)))
        hybrid_signal = float(features.get("overlap_flag", 0.0)) + 0.5
        total = max(visual_signal + text_signal + hybrid_signal, 1e-8)
        return {
            "visual": visual_signal / total,
            "text": text_signal / total,
            "hybrid": hybrid_signal / total,
        }
