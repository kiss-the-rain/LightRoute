"""Dataset wrapper that resolves sample-linked assets for retrieval experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from src.utils.io_utils import load_json, load_jsonl


class DocVQADataset:
    """Lightweight dataset for processed DocVQA samples."""

    def __init__(self, cfg: Any, split: str) -> None:
        self.cfg = cfg
        self.split = split
        self.samples = load_jsonl(cfg.dataset.processed_files[split]) if Path(cfg.dataset.processed_files[split]).exists() else []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.samples[index])
        if "ocr_results_path" in sample:
            ocr_path = Path(sample["ocr_results_path"])
            sample["ocr_results"] = load_json(ocr_path) if ocr_path.exists() else []
        else:
            ocr_results = []
            for ocr_path_str in sample.get("ocr_paths", []):
                ocr_path = Path(ocr_path_str)
                ocr_results.append(load_json(ocr_path) if ocr_path.exists() else {})
            sample["ocr_results"] = ocr_results
        sample.setdefault("image_paths", [])
        sample.setdefault("ocr_paths", [])
        sample.setdefault("evidence_pages", [])
        sample.setdefault("answer", "")
        return sample

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]
