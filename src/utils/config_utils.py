"""Configuration loading helpers with nested YAML include support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml


class ConfigNode(dict):
    """Dictionary with recursive attribute-style access."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def copy(self) -> "ConfigNode":
        return ConfigNode({key: _to_config_node(value) for key, value in self.items()})

    def to_dict(self) -> dict[str, Any]:
        return {key: _to_plain_dict(value) for key, value in self.items()}


def _load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _iter_include_paths(config: dict[str, Any], base_dir: Path) -> Iterator[Path]:
    includes = config.get("includes", [])
    if isinstance(includes, str):
        includes = [includes]
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = (base_dir / include_path).resolve()
        yield include_path


def _to_config_node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({key: _to_config_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_config_node(item) for item in value]
    return value


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value


def load_config(path: str | Path) -> ConfigNode:
    """Load YAML config with recursive includes and attribute access."""
    config_path = Path(path).resolve()
    raw_config = _load_yaml_file(config_path)

    merged: dict[str, Any] = {}
    for include_path in _iter_include_paths(raw_config, config_path.parent):
        included_config = load_config(include_path).to_dict()
        merged = _deep_update(merged, included_config)

    raw_config.pop("includes", None)
    merged = _deep_update(merged, raw_config)
    return _to_config_node(merged)
