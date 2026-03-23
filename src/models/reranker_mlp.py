"""Lightweight MLP reranker placeholder for future adaptive fusion training."""

from __future__ import annotations

from typing import Any


class RerankerMLP:
    """Placeholder reranker interface."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def predict(self, features: list[dict[str, Any]]) -> list[float]:
        return [0.0 for _ in features]
