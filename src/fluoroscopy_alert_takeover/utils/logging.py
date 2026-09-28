"""Logging setup.

Ref: none - release-internal plumbing.
"""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
# The handlers already installed on the package root, kept in a mutable container so the
# configuration function does not need a module-level rebinding.
_INSTALLED: list[logging.Handler] = []


def configure_logging(level: str = "INFO") -> None:
    """Install a single stream handler on the package root the first time it is called."""
    root = logging.getLogger("fluoroscopy_alert_takeover")
    if _INSTALLED:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.setLevel(level.upper())
    root.addHandler(handler)
    root.propagate = False
    _INSTALLED.append(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring the package root if needed."""
    configure_logging()
    return logging.getLogger(f"fluoroscopy_alert_takeover.{name}")


__all__ = ["configure_logging", "get_logger"]
