"""Unified OCR interface with dummy and optional Tesseract/Paddle backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from utils.text_utils import clean_ocr_text, tokenize_text

try:
    import pytesseract
    from PIL import Image
except ImportError:  # pragma: no cover
    pytesseract = None
    Image = None


@dataclass
class OCREngine:
    """Thin OCR wrapper that standardizes page-level OCR output."""

    backend: str = "dummy"

    def run_ocr_on_page(self, image_path: str | Path) -> dict[str, Any]:
        """Run OCR on a single page image and return normalized OCR output."""
        image_path = Path(image_path)
        page_id = image_path.stem
        backend = self.backend.lower()
        if backend == "dummy":
            return self._run_dummy(image_path, page_id)
        if backend == "tesseract":
            return self._run_tesseract(image_path, page_id)
        if backend == "paddleocr":
            raise NotImplementedError("PaddleOCR backend is reserved for a later integration phase.")
        raise ValueError(f"Unsupported OCR backend: {self.backend}")

    def run_ocr_on_document(self, image_paths: list[str] | list[Path]) -> list[dict[str, Any]]:
        """Run OCR over an ordered page image list."""
        return [self.run_ocr_on_page(image_path) for image_path in image_paths]

    @staticmethod
    def _run_dummy(image_path: Path, page_id: str) -> dict[str, Any]:
        text = clean_ocr_text(image_path.stem.replace("_", " "))
        score = 0.5
        return {
            "page_id": page_id,
            "full_text": text,
            "blocks": [{"text": text, "bbox": [0, 0, 1, 1], "score": score}] if text else [],
            "avg_confidence": score if text else 0.0,
            "token_count": len(tokenize_text(text)),
        }

    @staticmethod
    def _run_tesseract(image_path: Path, page_id: str) -> dict[str, Any]:
        if pytesseract is None or Image is None:
            raise ImportError("pytesseract and Pillow are required for the Tesseract backend.")

        image = Image.open(image_path)
        raw = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        blocks: list[dict[str, Any]] = []
        confidences: list[float] = []
        texts: list[str] = []
        for index, text in enumerate(raw["text"]):
            cleaned_text = clean_ocr_text(text)
            if not cleaned_text:
                continue
            confidence = max(float(raw["conf"][index]), 0.0) / 100.0
            x1 = int(raw["left"][index])
            y1 = int(raw["top"][index])
            width = int(raw["width"][index])
            height = int(raw["height"][index])
            blocks.append(
                {
                    "text": cleaned_text,
                    "bbox": [x1, y1, x1 + width, y1 + height],
                    "score": confidence,
                }
            )
            confidences.append(confidence)
            texts.append(cleaned_text)

        full_text = clean_ocr_text(" ".join(texts))
        return {
            "page_id": page_id,
            "full_text": full_text,
            "blocks": blocks,
            "avg_confidence": mean(confidences) if confidences else 0.0,
            "token_count": len(tokenize_text(full_text)),
        }
