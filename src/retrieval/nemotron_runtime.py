"""Shared memory-safe runtime helpers for Nemotron branches."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io_utils import ensure_dir
from src.utils.logger import get_logger


class NemotronPageEmbeddingStore:
    """Disk-backed page embedding store with bounded in-memory residency."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        enabled: bool = True,
        allow_save: bool = True,
        in_memory_limit: int = 128,
        warn_entry_threshold: int = 20000,
    ) -> None:
        self.cache_dir = ensure_dir(cache_dir)
        self.enabled = bool(enabled)
        self.allow_save = bool(allow_save)
        self.in_memory_limit = max(1, int(in_memory_limit))
        self.warn_entry_threshold = int(warn_entry_threshold)
        self.logger = get_logger("nemotron_embedding_store")
        self._hot_cache: OrderedDict[str, Any] = OrderedDict()
        self._num_known_disk_entries = sum(1 for _ in self.cache_dir.glob("**/*.npz")) if self.enabled else 0
        self._warned_large_store = False

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        if key in self._hot_cache:
            value = self._hot_cache.pop(key)
            self._hot_cache[key] = value
            return value
        path = self._path_for_key(key)
        if not path.exists():
            return None
        value = self._load_embedding(path)
        self._remember(key, value)
        return value

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._remember(key, value)
        if not self.allow_save:
            return
        path = self._path_for_key(key)
        ensure_dir(path.parent)
        if not path.exists():
            self._num_known_disk_entries += 1
        self._save_embedding(path, value)
        if not self._warned_large_store and self._num_known_disk_entries >= self.warn_entry_threshold:
            self._warned_large_store = True
            self.logger.warning(
                "Nemotron embedding store is large: approx_disk_entries=%d cache_dir=%s",
                self._num_known_disk_entries,
                self.cache_dir,
            )

    def stats(self) -> dict[str, int]:
        return {
            "num_hot_cached_embeddings": int(len(self._hot_cache)),
            "num_known_disk_embeddings": int(self._num_known_disk_entries),
        }

    def flush(self) -> None:
        self._hot_cache.clear()

    def _remember(self, key: str, value: Any) -> None:
        if key in self._hot_cache:
            self._hot_cache.pop(key)
        self._hot_cache[key] = value
        while len(self._hot_cache) > self.in_memory_limit:
            self._hot_cache.popitem(last=False)

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.npz"

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _save_embedding(self, path: Path, value: Any) -> None:
        array = self._as_numpy(value)
        np.savez_compressed(path, embedding=array.astype(np.float16, copy=False), shape=np.asarray(array.shape, dtype=np.int32))

    @staticmethod
    def _load_embedding(path: Path) -> np.ndarray:
        with np.load(path, allow_pickle=False) as payload:
            array = payload["embedding"]
        return np.asarray(array, dtype=np.float32)
