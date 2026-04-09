"""OCR quality feature extraction for adaptive routing."""

from __future__ import annotations

from typing import Any


class QualityEstimator:
    """Extract lightweight OCR quality descriptors."""

    def extract(self, page_payload: dict[str, Any]) -> dict[str, float]:
        """Return quality features plus an optional reliability score."""
        token_count = float(page_payload.get("ocr_token_count", 0))
        avg_confidence = float(page_payload.get("ocr_avg_confidence", 0.0))
        text_density = float(page_payload.get("text_density", token_count))
        reliability = avg_confidence if token_count > 0 else 0.0
        return {
            "quality_feature_vector": [avg_confidence, token_count, text_density],
            "ocr_avg_confidence": avg_confidence,
            "ocr_token_count": token_count,
            "text_density": text_density,
            "ocr_reliability": reliability,
        }
