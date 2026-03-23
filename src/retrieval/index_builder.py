"""Shared retrieval index construction and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io_utils import load_json


def collect_document_pages(cfg: Any, processed_samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Collect document page metadata and OCR text into a unified page store."""
    documents: dict[str, dict[str, dict[str, Any]]] = {}
    for sample in processed_samples:
        doc_id = sample["doc_id"]
        doc_pages = documents.setdefault(doc_id, {})
        ocr_path = Path(sample["ocr_results_path"])
        ocr_results = load_json(ocr_path) if ocr_path.exists() else []
        ocr_by_page = {item["page_id"]: item for item in ocr_results}
        for image_path in sample.get("image_paths", []):
            page_id = Path(image_path).stem
            ocr_entry = ocr_by_page.get(page_id, {})
            doc_pages[page_id] = {
                "page_id": page_id,
                "doc_id": doc_id,
                "image_path": image_path,
                "ocr_text": ocr_entry.get("full_text", ""),
                "surrogate_text": ocr_entry.get("full_text", ""),
                "ocr_avg_confidence": ocr_entry.get("avg_confidence", 0.0),
                "ocr_token_count": ocr_entry.get("token_count", 0),
            }

    return {doc_id: list(page_map.values()) for doc_id, page_map in documents.items()}


def build_visual_index(visual_retriever: Any, documents: dict[str, list[dict[str, Any]]]) -> None:
    """Build the visual retrieval index."""
    visual_retriever.build_index(documents)


def build_text_index(text_retriever: Any, documents: dict[str, list[dict[str, Any]]]) -> None:
    """Build the OCR text retrieval index."""
    text_retriever.build_index(documents)
