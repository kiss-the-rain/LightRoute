"""Generic trainer skeleton for later adaptive-fusion learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io_utils import save_json


class Trainer:
    """Generic training loop placeholder with checkpoint hooks."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def train(self) -> dict[str, Any]:
        metrics = {"status": "not_implemented"}
        save_json(metrics, Path(self.cfg.paths.metric_dir) / "train_placeholder.json")
        return metrics
