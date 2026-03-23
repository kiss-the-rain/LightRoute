"""Learning-rate scheduler builder reserved for later phases."""

from __future__ import annotations

from typing import Any

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Any) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler_name = str(cfg.training.scheduler).lower()
    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(cfg.training.epochs), 1))
    return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(int(cfg.training.epochs), 1))
