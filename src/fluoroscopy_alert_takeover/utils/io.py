"""Atomic artefact IO and content hashing.

Ref: none - release-internal plumbing. Checkpoints and report files are written through a
temporary sibling and moved into place, so a killed run never leaves a half file behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Hash a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temporary file is managed by hand: it has to stay on disk after the handle is
    # closed so the bytes can be moved into place in one step.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        dir=path.parent, delete=False, suffix=".tmp"
    )
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def write_text(path: str | Path, text: str) -> Path:
    """Write text atomically and return the file that was written."""
    target = Path(path)
    _atomic_write_bytes(target, text.encode("utf-8"))
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON atomically with sorted keys so the artefact is byte-stable."""
    target = Path(path)
    serialised = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    _atomic_write_bytes(target, (serialised + "\n").encode("utf-8"))
    return target


def read_json(path: str | Path) -> Any:
    """Read a JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_arrays(path: str | Path, arrays: dict[str, npt.NDArray[np.generic]]) -> Path:
    """Write an uncompressed npz atomically and return the file that was written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Managed by hand for the same reason as the byte writer above.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        dir=target.parent, delete=False, suffix=".npz.tmp"
    )
    handle.close()
    try:
        with Path(handle.name).open("wb") as sink:
            np.savez(sink, **arrays)  # type: ignore[arg-type]
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def read_arrays(path: str | Path) -> dict[str, npt.NDArray[np.generic]]:
    """Read an npz written by :func:`write_arrays`."""
    with np.load(Path(path), allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


__all__ = [
    "read_arrays",
    "read_json",
    "sha256_file",
    "write_arrays",
    "write_json",
    "write_text",
]
