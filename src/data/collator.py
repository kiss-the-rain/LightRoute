"""Basic batch collation helpers for retrieval feature training."""

from __future__ import annotations

from typing import Any

import numpy as np


def collate_feature_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate retrieval features and labels for a basic training batch."""
    retrieval_features = np.asarray([item.get("retrieval_features", item.get("features", [])) for item in batch], dtype=float)
    labels = np.asarray([item.get("label", 0) for item in batch], dtype=float)
    return {
        "retrieval_features": retrieval_features,
        "labels": labels,
    }
