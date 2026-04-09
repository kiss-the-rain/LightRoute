"""Training losses for fusion and routing models."""

from __future__ import annotations

import numpy as np


def bce_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary cross entropy loss from logits."""
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    probabilities = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    labels = labels.astype(float)
    return float(-(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)).mean())


def ranking_loss(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    margin: float = 1.0,
) -> float:
    """Compute a margin ranking loss for positive versus negative pages."""
    return float(np.maximum(0.0, margin - positive_scores + negative_scores).mean())


def listwise_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute a simple listwise cross-entropy style loss over candidate sets."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / np.clip(exp_scores.sum(axis=-1, keepdims=True), 1e-8, None)
    normalized_labels = labels.astype(float)
    normalized_labels = normalized_labels / np.clip(normalized_labels.sum(axis=-1, keepdims=True), 1.0, None)
    return float(-(normalized_labels * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(axis=-1).mean())
