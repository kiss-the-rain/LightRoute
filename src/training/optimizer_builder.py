"""Optimizer construction utilities for training fusion models."""

from __future__ import annotations

from typing import Any


def build_optimizer(parameters: Any, cfg: Any) -> dict[str, float | str]:
    """Build a lightweight optimizer config used by the NumPy trainer."""
    return {
        "name": str(cfg.training.optimizer).lower(),
        "learning_rate": float(cfg.training.learning_rate),
        "weight_decay": float(cfg.training.weight_decay),
    }
