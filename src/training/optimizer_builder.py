"""Optimizer builder reserved for later adaptive-fusion training."""

from __future__ import annotations

from typing import Any

import torch


def build_optimizer(parameters: Any, cfg: Any) -> torch.optim.Optimizer:
    optimizer_name = str(cfg.training.optimizer).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=float(cfg.training.learning_rate), weight_decay=float(cfg.training.weight_decay))
    return torch.optim.AdamW(parameters, lr=float(cfg.training.learning_rate), weight_decay=float(cfg.training.weight_decay))
