"""Lightweight adaptive fusion scorer for candidate-level retrieval reranking."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.reranker_mlp import RerankerMLP
from src.utils.rank_utils import sort_pages_by_score


class AdaptiveFusion:
    """Question-aware and quality-aware adaptive fusion scorer."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.scorer = RerankerMLP(input_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)

    def predict_logits(self, feature_vectors: list[list[float]]) -> list[float]:
        """Predict one relevance logit per candidate page."""
        return self.scorer.predict(feature_vectors)

    def predict(self, feature_vectors: list[list[float]]) -> list[float]:
        """Backward-compatible prediction alias used by the generic trainer."""
        return self.predict_logits(feature_vectors)

    def train_step(self, feature_array: np.ndarray, labels: np.ndarray, learning_rate: float, weight_decay: float) -> float:
        """Run one training step over a batch of candidate features."""
        return self.scorer.train_step(feature_array, labels, learning_rate=learning_rate, weight_decay=weight_decay)

    def rank_candidates(self, candidate_rows: list[dict[str, Any]]) -> dict[str, list]:
        """Score and rank candidate pages from feature rows containing `feature_vector`."""
        if not candidate_rows:
            return {"page_ids": [], "scores": [], "ranks": []}
        feature_vectors = [list(row["feature_vector"]) for row in candidate_rows]
        logits = self.predict_logits(feature_vectors)
        ranked = sort_pages_by_score([int(row["page_id"]) for row in candidate_rows], logits)
        return {
            "page_ids": [int(item["page_id"]) for item in ranked],
            "scores": [float(item["score"]) for item in ranked],
            "ranks": [int(item["rank"]) for item in ranked],
        }

    def state_dict(self) -> dict[str, Any]:
        """Return serializable model state."""
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "scorer": self.scorer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore model state from a serialized checkpoint."""
        self.feature_dim = int(state_dict.get("feature_dim", self.feature_dim))
        self.hidden_dim = int(state_dict.get("hidden_dim", self.hidden_dim))
        self.dropout = float(state_dict.get("dropout", self.dropout))
        self.scorer.load_state_dict(state_dict["scorer"])

    def parameters(self) -> list[np.ndarray]:
        """Expose parameter arrays for trainer compatibility."""
        return self.scorer.parameters()


class AdaptiveFusionV2:
    """Enhanced adaptive fusion scorer with two hidden layers and layer normalization."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        scale = 0.02
        self.w1 = np.random.randn(feature_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b2 = np.zeros(hidden_dim, dtype=float)
        self.w3 = np.random.randn(hidden_dim, 1) * scale
        self.b3 = np.zeros(1, dtype=float)

    def _layer_norm(self, hidden: np.ndarray) -> np.ndarray:
        mean = hidden.mean(axis=-1, keepdims=True)
        var = hidden.var(axis=-1, keepdims=True)
        return (hidden - mean) / np.sqrt(var + 1e-6)

    def _relu(self, hidden: np.ndarray) -> np.ndarray:
        return np.maximum(hidden, 0.0)

    def forward(self, feature_array: np.ndarray) -> np.ndarray:
        """Run the enhanced scorer and return one logit per candidate."""
        hidden1 = self._relu(feature_array @ self.w1 + self.b1)
        hidden1 = self._layer_norm(hidden1)
        hidden2 = self._relu(hidden1 @ self.w2 + self.b2)
        hidden2 = self._layer_norm(hidden2)
        return (hidden2 @ self.w3 + self.b3).reshape(-1)

    def predict_logits(self, feature_vectors: list[list[float]]) -> list[float]:
        """Predict one relevance logit per candidate page."""
        feature_array = np.asarray(feature_vectors, dtype=float)
        if feature_array.size == 0:
            return []
        return self.forward(feature_array).tolist()

    def predict(self, feature_vectors: list[list[float]]) -> list[float]:
        """Backward-compatible prediction alias used by the generic trainer."""
        return self.predict_logits(feature_vectors)

    def train_step(self, feature_array: np.ndarray, labels: np.ndarray, learning_rate: float, weight_decay: float) -> float:
        """Run one SGD step with BCE loss over the enhanced scorer."""
        hidden1_pre = feature_array @ self.w1 + self.b1
        hidden1 = self._relu(hidden1_pre)
        hidden1 = self._layer_norm(hidden1)
        hidden2_pre = hidden1 @ self.w2 + self.b2
        hidden2 = self._relu(hidden2_pre)
        hidden2 = self._layer_norm(hidden2)
        logits = (hidden2 @ self.w3 + self.b3).reshape(-1)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        probabilities = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
        labels = labels.astype(float)
        loss = float(-(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)).mean())

        grad_logits = (probabilities - labels)[:, None] / max(len(labels), 1)
        grad_w3 = hidden2.T @ grad_logits + weight_decay * self.w3
        grad_b3 = grad_logits.sum(axis=0)
        grad_hidden2 = grad_logits @ self.w3.T
        grad_hidden2 *= (hidden2_pre > 0).astype(float)
        grad_w2 = hidden1.T @ grad_hidden2 + weight_decay * self.w2
        grad_b2 = grad_hidden2.sum(axis=0)
        grad_hidden1 = grad_hidden2 @ self.w2.T
        grad_hidden1 *= (hidden1_pre > 0).astype(float)
        grad_w1 = feature_array.T @ grad_hidden1 + weight_decay * self.w1
        grad_b1 = grad_hidden1.sum(axis=0)

        self.w3 -= learning_rate * grad_w3
        self.b3 -= learning_rate * grad_b3
        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        return loss

    def rank_candidates(self, candidate_rows: list[dict[str, Any]]) -> dict[str, list]:
        """Score and rank candidate pages from feature rows containing `feature_vector`."""
        if not candidate_rows:
            return {"page_ids": [], "scores": [], "ranks": []}
        feature_vectors = [list(row["feature_vector"]) for row in candidate_rows]
        logits = self.predict_logits(feature_vectors)
        ranked = sort_pages_by_score([int(row["page_id"]) for row in candidate_rows], logits)
        return {
            "page_ids": [int(item["page_id"]) for item in ranked],
            "scores": [float(item["score"]) for item in ranked],
            "ranks": [int(item["rank"]) for item in ranked],
        }

    def state_dict(self) -> dict[str, Any]:
        """Return serializable model state."""
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3.tolist(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore model state from a serialized checkpoint."""
        self.feature_dim = int(state_dict.get("feature_dim", self.feature_dim))
        self.hidden_dim = int(state_dict.get("hidden_dim", self.hidden_dim))
        self.dropout = float(state_dict.get("dropout", self.dropout))
        self.w1 = np.asarray(state_dict["w1"], dtype=float)
        self.b1 = np.asarray(state_dict["b1"], dtype=float)
        self.w2 = np.asarray(state_dict["w2"], dtype=float)
        self.b2 = np.asarray(state_dict["b2"], dtype=float)
        self.w3 = np.asarray(state_dict["w3"], dtype=float)
        self.b3 = np.asarray(state_dict["b3"], dtype=float)

    def parameters(self) -> list[np.ndarray]:
        """Expose parameter arrays for trainer compatibility."""
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
