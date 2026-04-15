"""Lightweight gating MLP for sample-level OCR/visual mixture weights."""

from __future__ import annotations

from typing import Any

import numpy as np


class GateNet:
    """Small MLP that predicts a 2-way OCR/visual mixture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        min_weight: float = 0.2,
        max_weight: float = 0.8,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        scale = 0.02
        self.w1 = np.random.randn(self.input_dim, self.hidden_dim) * scale
        self.b1 = np.zeros(self.hidden_dim, dtype=float)
        self.w2 = np.random.randn(self.hidden_dim, 2) * scale
        self.b2 = np.zeros(2, dtype=float)

    @staticmethod
    def _relu(hidden: np.ndarray) -> np.ndarray:
        return np.maximum(hidden, 0.0)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.clip(exp_scores.sum(axis=-1, keepdims=True), 1e-8, None)

    def _postprocess_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / np.clip(weights.sum(axis=-1, keepdims=True), 1e-8, None)
        return weights

    def predict_weights(self, feature_vector: list[float] | np.ndarray) -> tuple[float, float]:
        feature_array = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        hidden = self._relu(feature_array @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        weights = self._postprocess_weights(self._softmax(logits))[0]
        return float(weights[0]), float(weights[1])

    def train_step(
        self,
        feature_vector: list[float] | np.ndarray,
        target_weights: tuple[float, float] | list[float] | np.ndarray,
        learning_rate: float,
        weight_decay: float,
    ) -> float:
        feature_array = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        targets = np.asarray(target_weights, dtype=float).reshape(1, 2)
        hidden_pre = feature_array @ self.w1 + self.b1
        hidden = self._relu(hidden_pre)
        logits = hidden @ self.w2 + self.b2
        probabilities = self._softmax(logits)
        probabilities = np.clip(probabilities, 1e-8, 1.0)
        loss = float(-(targets * np.log(probabilities)).sum(axis=-1).mean())

        grad_logits = (probabilities - targets)
        grad_w2 = hidden.T @ grad_logits + weight_decay * self.w2
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ self.w2.T
        grad_hidden *= (hidden_pre > 0).astype(float)
        grad_w1 = feature_array.T @ grad_hidden + weight_decay * self.w1
        grad_b1 = grad_hidden.sum(axis=0)

        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.input_dim = int(state_dict.get("input_dim", self.input_dim))
        self.hidden_dim = int(state_dict.get("hidden_dim", self.hidden_dim))
        self.min_weight = float(state_dict.get("min_weight", self.min_weight))
        self.max_weight = float(state_dict.get("max_weight", self.max_weight))
        self.w1 = np.asarray(state_dict["w1"], dtype=float)
        self.b1 = np.asarray(state_dict["b1"], dtype=float)
        self.w2 = np.asarray(state_dict["w2"], dtype=float)
        self.b2 = np.asarray(state_dict["b2"], dtype=float)
