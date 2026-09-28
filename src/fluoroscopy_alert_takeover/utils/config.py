"""Configuration loading.

Ref: Sec. 4.2-4.10, pp. 14-21, for the protocol values the configs carry. Every value a
config sets that the manuscript does not print is marked ``# engineering default`` at
its definition site and recorded in ``verification_report.json``.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from .logging import get_logger

_LOG = get_logger("config")

Mapping = dict[str, Any]


def load_yaml(path: str | Path) -> Mapping:
    """Read one YAML document into a plain mapping."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"config not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"config root must be a mapping: {resolved}")
    return dict(loaded)


def deep_merge(base: Mapping, overlay: Mapping) -> Mapping:
    """Recursively merge ``overlay`` onto ``base`` without mutating either."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def compose(paths: Sequence[str | Path], overrides: Sequence[str] | None = None) -> Mapping:
    """Compose the listed YAML files left to right, then apply ``key=value`` overrides."""
    merged: Mapping = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml(path))
    for override in overrides or []:
        merged = apply_override(merged, override)
    return merged


def _coerce(raw: str) -> Any:
    return yaml.safe_load(raw)


def apply_override(config: Mapping, override: str) -> Mapping:
    """Apply a single ``dotted.path=value`` override, coercing the value through YAML."""
    if "=" not in override:
        raise ValueError(f"override must be key=value: {override!r}")
    key, _, raw = override.partition("=")
    parts = [segment for segment in key.split(".") if segment]
    if not parts:
        raise ValueError(f"override has an empty key: {override!r}")
    updated = copy.deepcopy(config)
    cursor: Mapping = updated
    for segment in parts[:-1]:
        child = cursor.get(segment)
        if not isinstance(child, dict):
            child = {}
            cursor[segment] = child
        cursor = child
    cursor[parts[-1]] = _coerce(raw)
    _LOG.debug("override applied: %s", override)
    return updated


def dig(config: Mapping, dotted: str, default: Any = None) -> Any:
    """Read ``dotted`` from a nested mapping, returning ``default`` when absent."""
    cursor: Any = config
    for segment in dotted.split("."):
        if not isinstance(cursor, dict) or segment not in cursor:
            return default
        cursor = cursor[segment]
    return cursor


def require(config: Mapping, dotted: str) -> Any:
    """Read ``dotted`` and fail loudly when it is absent."""
    sentinel = object()
    value = dig(config, dotted, sentinel)
    if value is sentinel:
        raise KeyError(f"missing config key: {dotted}")
    return value


__all__ = ["Mapping", "apply_override", "compose", "deep_merge", "dig", "load_yaml", "require"]
