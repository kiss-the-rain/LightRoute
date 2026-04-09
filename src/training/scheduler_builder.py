"""Learning-rate scheduler construction utilities."""

from __future__ import annotations

from typing import Any


def build_scheduler(optimizer: dict[str, float | str], cfg: Any) -> dict[str, float | str]:
    """Build a lightweight scheduler config used by the NumPy trainer."""
    return {
        "name": str(cfg.training.scheduler).lower(),
        "base_learning_rate": float(optimizer["learning_rate"]),
        "epochs": int(cfg.training.epochs),
    }
