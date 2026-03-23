"""Score normalization and alignment helpers for heterogeneous retrievers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def min_max_normalize(scores: Iterable[float]) -> list[float]:
    """Apply min-max normalization to a score list."""
    array = np.asarray(list(scores), dtype=float)
    if array.size == 0:
        return []
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value == min_value:
        return [1.0 for _ in array]
    return ((array - min_value) / (max_value - min_value)).tolist()


def z_score_normalize(scores: Iterable[float]) -> list[float]:
    """Apply z-score normalization to a score list."""
    array = np.asarray(list(scores), dtype=float)
    if array.size == 0:
        return []
    std = float(array.std())
    if std == 0.0:
        return [0.0 for _ in array]
    return ((array - array.mean()) / std).tolist()


def softmax_normalize(scores: Iterable[float]) -> list[float]:
    """Apply softmax normalization to a score list."""
    array = np.asarray(list(scores), dtype=float)
    if array.size == 0:
        return []
    shifted = array - array.max()
    exp_scores = np.exp(shifted)
    return (exp_scores / exp_scores.sum()).tolist()


def clip_scores(scores: Iterable[float], min_value: float = 0.0, max_value: float = 1.0) -> list[float]:
    """Clip scores into a bounded range."""
    array = np.asarray(list(scores), dtype=float)
    if array.size == 0:
        return []
    return np.clip(array, min_value, max_value).tolist()


def align_retriever_scores(scores: Iterable[float], method: str = "minmax") -> list[float]:
    """Normalize retriever scores onto a comparable scale."""
    method = method.lower()
    if method == "minmax":
        return min_max_normalize(scores)
    if method == "zscore":
        return z_score_normalize(scores)
    if method == "softmax":
        return softmax_normalize(scores)
    if method == "none":
        return list(scores)
    raise ValueError(f"Unsupported score alignment method: {method}")
