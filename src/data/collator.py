"""Batch collation helpers for future fusion-model training."""

from __future__ import annotations

from typing import Any

import numpy as np


def collate_feature_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate numeric features, optional embeddings, and labels."""
    numeric_features = np.asarray([item.get("features", []) for item in batch], dtype=float)
    question_embeddings = np.asarray([item.get("question_embedding", []) for item in batch], dtype=float)
    labels = np.asarray([item.get("label", 0) for item in batch], dtype=float)
    return {
        "numeric_features": numeric_features,
        "question_embeddings": question_embeddings,
        "labels": labels,
    }
