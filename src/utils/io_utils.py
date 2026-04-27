"""File-system and serialization helpers used across the project."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Any, path: str | Path) -> None:
    """Save a JSON file with UTF-8 encoding."""
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield JSONL records one by one."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Save an iterable of dictionaries as JSONL."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def load_pickle(path: str | Path) -> Any:
    """Load a pickle object from disk."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_pickle(data: Any, path: str | Path) -> None:
    """Save a pickle object to disk."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("wb") as handle:
        pickle.dump(data, handle)


def save_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Save a list of dictionaries to CSV."""
    rows = list(rows)
    target = Path(path)
    ensure_dir(target.parent)
    if not rows:
        target.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
