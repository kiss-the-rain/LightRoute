"""Streaming JSONL writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.io_utils import ensure_dir


class JsonlStreamWriter:
    """Append JSON records incrementally without holding them all in memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.handle = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False))
        self.handle.write("\n")

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "JsonlStreamWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
