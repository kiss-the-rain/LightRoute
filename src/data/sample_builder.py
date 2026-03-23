"""Raw DocVQA annotation conversion into a unified sample format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io_utils import ensure_dir, load_json, load_jsonl, save_jsonl


class SampleBuilder:
    """Convert raw annotation files into the project's canonical JSONL format."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def build_split(self, split: str) -> list[dict[str, Any]]:
        """Build a processed split file from the configured raw annotation source."""
        raw_path = Path(self.cfg.dataset.split_files[split])
        if not raw_path.exists():
            return []

        if raw_path.suffix == ".jsonl":
            annotations = load_jsonl(raw_path)
        else:
            payload = load_json(raw_path)
            annotations = payload.get("data", payload if isinstance(payload, list) else [])

        samples = [self._normalize_annotation(item, split) for item in annotations]
        processed_path = Path(self.cfg.dataset.processed_files[split])
        ensure_dir(processed_path.parent)
        save_jsonl(samples, processed_path)
        return samples

    def _normalize_annotation(self, item: dict[str, Any], split: str) -> dict[str, Any]:
        doc_id = str(item.get("doc_id") or item.get("document_id") or item.get("image") or item.get("pdf") or "unknown_doc")
        doc_id = Path(doc_id).stem
        evidence_pages = item.get("evidence_pages") or item.get("answers_page_idx") or item.get("page_idx") or []
        if isinstance(evidence_pages, int):
            evidence_pages = [evidence_pages]

        image_dir = Path(self.cfg.dataset.page_image_dir) / doc_id
        image_paths = sorted(str(path) for path in image_dir.glob(f"*{self.cfg.dataset.image_ext}"))
        page_ids = [Path(path).stem for path in image_paths]

        return {
            "qid": str(item.get("qid") or item.get("question_id") or f"{split}_{doc_id}_{item.get('id', 0)}"),
            "question": item.get("question", ""),
            "answer": item.get("answer", self.cfg.dataset.default_answer),
            "doc_id": doc_id,
            "page_ids": page_ids,
            "evidence_pages": list(evidence_pages),
            "image_paths": image_paths,
            "ocr_results_path": str(Path(self.cfg.dataset.ocr_dir) / f"{doc_id}{self.cfg.dataset.ocr_ext}"),
            "split": split,
        }
