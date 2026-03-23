"""Compatibility helpers for dataset preparation and loading."""

from __future__ import annotations

from typing import Any

from data.doc_dataset import DocVQADataset
from data.sample_builder import SampleBuilder


class DatasetManager:
    """Backward-compatible wrapper around sample building and dataset loading."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.builder = SampleBuilder(cfg)

    def prepare(self, split: str) -> list[dict[str, Any]]:
        return self.builder.build_split(split)

    def load(self, split: str) -> DocVQADataset:
        return DocVQADataset(self.cfg, split)
