"""Atomic checkpoint write and read, carrying the seed.

Ref: none - release-internal plumbing. Every resumed run restores the seed recorded with
the weights, so a resumed analysis reproduces the interrupted one.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint through a temporary sibling and an atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "epoch": int(epoch),
        "seed": int(seed),
        "extra": dict(extra or {}),
    }
    # The temporary file must outlive its handle so the rename is the only visible step.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        dir=target.parent, delete=False, suffix=".pt.tmp"
    )
    handle.close()
    try:
        torch.save(payload, handle.name)
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def load_checkpoint(
    path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> tuple[int, int, dict[str, Any]]:
    """Load weights into ``model`` and return ``(epoch, seed, extra)``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload["epoch"]), int(payload["seed"]), dict(payload.get("extra", {}))


__all__ = ["load_checkpoint", "save_checkpoint"]
