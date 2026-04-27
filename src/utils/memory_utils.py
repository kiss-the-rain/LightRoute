"""Lightweight memory logging and cleanup helpers."""

from __future__ import annotations

import gc
from typing import Any


def _resident_gb() -> float:
    try:
        import psutil

        process = psutil.Process()
        return float(process.memory_info().rss / (1024**3))
    except Exception:
        return -1.0


def log_ram(logger: Any, tag: str) -> None:
    """Log current resident RAM for the active process."""
    resident_gb = _resident_gb()
    if resident_gb < 0:
        logger.info("[RAM] %s rss_gb=unavailable", tag)
        return
    logger.info("[RAM] %s rss_gb=%.3f", tag, resident_gb)


def release_memory(*objs: Any) -> None:
    """Best-effort Python and CUDA memory cleanup."""
    for obj in objs:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
