"""Fusion-training entrypoint reserved for later adaptive-fusion phases."""

from __future__ import annotations

from typing import Any

from training.trainer import Trainer


def train_fusion(cfg: Any) -> dict[str, Any]:
    return Trainer(cfg).train()
