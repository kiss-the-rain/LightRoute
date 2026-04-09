"""Generic trainer with train/validation loops, checkpointing, and early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.training.losses import bce_loss
from src.training.optimizer_builder import build_optimizer
from src.training.scheduler_builder import build_scheduler
from src.utils.io_utils import load_pickle, save_json, save_pickle
from src.utils.logger import get_logger


class Trainer:
    """Generic trainer for lightweight reranking modules."""

    def __init__(self, cfg: Any, model: Any, train_data: list[dict[str, Any]], val_data: list[dict[str, Any]]) -> None:
        self.cfg = cfg
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = build_optimizer(self.model.parameters(), cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.logger = get_logger("trainer", log_file=Path(cfg.paths.log_dir) / f"{cfg.experiment.name}_train_fusion.log")
        checkpoint_dir = getattr(cfg.training, "checkpoint_dir", cfg.paths.checkpoint_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_checkpoint_path = self.checkpoint_dir / f"{cfg.experiment.name}_best.pkl"
        self.last_checkpoint_path = self.checkpoint_dir / f"{cfg.experiment.name}_last.pkl"

    def train(self) -> dict[str, Any]:
        """Run training with validation, checkpointing, and early stopping."""
        best_metric = float("-inf")
        best_epoch = -1
        patience = int(self.cfg.training.early_stopping_patience)
        history: list[dict[str, float | int]] = []

        total_epochs = int(self.cfg.training.epochs)
        epoch_iterator = tqdm(range(total_epochs), desc="AdaptiveFusion epochs", unit="epoch")
        for epoch in epoch_iterator:
            train_metrics = self.train_loop(epoch)
            val_metrics = self.val_loop(epoch)
            epoch_metrics = {"epoch": epoch + 1, **train_metrics, **val_metrics}
            history.append(epoch_metrics)
            self.logger.info("Epoch %d metrics: %s", epoch + 1, epoch_metrics)
            epoch_iterator.set_postfix(
                train_loss=f"{float(train_metrics['train_loss']):.4f}",
                val_loss=f"{float(val_metrics['val_loss']):.4f}",
                val_acc=f"{float(val_metrics['val_accuracy']):.4f}",
            )

            current_metric = float(val_metrics["val_accuracy"])
            self.save_checkpoint(self.last_checkpoint_path, epoch + 1, current_metric)
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1
                self.save_checkpoint(self.best_checkpoint_path, epoch + 1, best_metric)

            if epoch + 1 - best_epoch >= patience:
                self.logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

        metrics = {"best_val_accuracy": best_metric, "best_epoch": best_epoch, "history": history}
        save_json(metrics, Path(self.cfg.paths.metric_dir) / f"{self.cfg.experiment.name}_train_metrics.json")
        return metrics

    def train_loop(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        total_loss = 0.0
        total_examples = 0
        total_correct = 0
        learning_rate = self._get_learning_rate(epoch)
        batch_iterator = tqdm(
            self.train_data,
            desc=f"Train epoch {epoch + 1}",
            unit="batch",
            leave=False,
        )
        for batch in batch_iterator:
            features = np.asarray(batch["features"], dtype=float)
            labels = np.asarray(batch["labels"], dtype=float)
            loss = self.model.train_step(
                features,
                labels,
                learning_rate=learning_rate,
                weight_decay=float(self.optimizer["weight_decay"]),
            )
            logits = np.asarray(self.model.predict(features.tolist()), dtype=float)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            total_loss += float(loss) * labels.size
            predictions = (probabilities >= 0.5).astype(float)
            total_correct += int((predictions == labels).sum())
            total_examples += int(labels.size)
            batch_iterator.set_postfix(
                loss=f"{float(loss):.4f}",
                acc=f"{(total_correct / max(total_examples, 1)):.4f}",
            )
        return {
            "train_loss": total_loss / max(total_examples, 1),
            "train_accuracy": total_correct / max(total_examples, 1),
        }

    def val_loop(self, epoch: int) -> dict[str, float]:
        """Run one validation epoch."""
        total_loss = 0.0
        total_examples = 0
        total_correct = 0
        batch_iterator = tqdm(
            self.val_data,
            desc=f"Val epoch {epoch + 1}",
            unit="batch",
            leave=False,
        )
        for batch in batch_iterator:
            features = np.asarray(batch["features"], dtype=float)
            labels = np.asarray(batch["labels"], dtype=float)
            logits = np.asarray(self.model.predict(features.tolist()), dtype=float)
            loss = bce_loss(logits, labels)
            total_loss += float(loss) * labels.size
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            predictions = (probabilities >= 0.5).astype(float)
            total_correct += int((predictions == labels).sum())
            total_examples += int(labels.size)
            batch_iterator.set_postfix(
                loss=f"{float(loss):.4f}",
                acc=f"{(total_correct / max(total_examples, 1)):.4f}",
            )
        return {
            "val_loss": total_loss / max(total_examples, 1),
            "val_accuracy": total_correct / max(total_examples, 1),
        }

    def save_checkpoint(self, path: str | Path, epoch: int, metric: float) -> None:
        """Save a checkpoint with model and optimizer state."""
        checkpoint = {
            "epoch": epoch,
            "metric": metric,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_pickle(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load a checkpoint and restore model and optimizer state."""
        checkpoint = load_pickle(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer = checkpoint["optimizer_state"]
        return checkpoint

    def _get_learning_rate(self, epoch: int) -> float:
        """Return a lightweight scheduled learning rate."""
        scheduler_name = str(self.scheduler["name"])
        base_lr = float(self.scheduler["base_learning_rate"])
        total_epochs = max(int(self.scheduler["epochs"]), 1)
        if scheduler_name == "linear":
            return base_lr * max(0.1, 1.0 - epoch / total_epochs)
        if scheduler_name == "cosine":
            cosine_ratio = 0.5 * (1.0 + np.cos(np.pi * epoch / total_epochs))
            return base_lr * max(cosine_ratio, 0.1)
        return base_lr
