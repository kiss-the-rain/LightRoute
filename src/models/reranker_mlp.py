"""Lightweight MLP reranker for candidate feature scoring."""

from __future__ import annotations

from typing import Any

import numpy as np


class RerankerMLP:
    """A small 2-layer NumPy MLP reranker with batch inference support."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        scale = 0.02
        self.w1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.w2 = np.random.randn(hidden_dim, 1) * scale
        self.b2 = np.zeros(1, dtype=float)

    def forward(self, feature_array: np.ndarray) -> np.ndarray:
        """Run the MLP and return a score per candidate."""
        hidden = np.maximum(0.0, feature_array @ self.w1 + self.b1)
        return (hidden @ self.w2 + self.b2).reshape(-1)

    def predict(self, features: list[list[float]] | list[dict[str, Any]]) -> list[float]:
        """Run batch inference on dense numeric feature vectors."""
        if not features:
            return []
        if isinstance(features[0], dict):
            feature_vectors = [list(item.values()) for item in features]  # type: ignore[union-attr]
        else:
            feature_vectors = features  # type: ignore[assignment]
        feature_array = np.asarray(feature_vectors, dtype=float)
        return self.forward(feature_array).tolist()

    def train_step(self, feature_array: np.ndarray, labels: np.ndarray, learning_rate: float, weight_decay: float) -> float:
        """Run one SGD step with BCE loss and return the batch loss."""
        hidden_pre = feature_array @ self.w1 + self.b1
        hidden = np.maximum(0.0, hidden_pre)
        logits = (hidden @ self.w2 + self.b2).reshape(-1)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        probabilities = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
        labels = labels.astype(float)
        loss = float(-(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)).mean())

        grad_logits = (probabilities - labels)[:, None] / max(len(labels), 1)
        grad_w2 = hidden.T @ grad_logits + weight_decay * self.w2
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ self.w2.T
        grad_hidden[hidden_pre <= 0.0] = 0.0
        grad_w1 = feature_array.T @ grad_hidden + weight_decay * self.w1
        grad_b1 = grad_hidden.sum(axis=0)

        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        return loss

    def state_dict(self) -> dict[str, list]:
        """Return serializable model weights."""
        return {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    def load_state_dict(self, state_dict: dict[str, list]) -> None:
        """Restore model weights from a serialized state dictionary."""
        self.w1 = np.asarray(state_dict["w1"], dtype=float)
        self.b1 = np.asarray(state_dict["b1"], dtype=float)
        self.w2 = np.asarray(state_dict["w2"], dtype=float)
        self.b2 = np.asarray(state_dict["b2"], dtype=float)

    def parameters(self) -> list[np.ndarray]:
        """Return parameter arrays for optimizer compatibility."""
        return [self.w1, self.b1, self.w2, self.b2]
