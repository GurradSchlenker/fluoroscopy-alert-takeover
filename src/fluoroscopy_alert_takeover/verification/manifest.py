"""Integrity manifest of the release tree.

Ref: none - release-internal. The manifest lists the SHA-256 of every file that ships, so a
reader can tell whether a copy of the release is the copy that was verified. It covers the
two verification artefacts as well as the tree, and excludes only itself, because a file
cannot contain its own hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..utils.io import sha256_file

EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "runs",
        "data",
        ".venv",
        "venv",
        "build",
        "dist",
        "node_modules",
    }
)
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".npz", ".npy", ".log")
MANIFEST_NAME = "integrity_manifest.json"


@dataclass
class Manifest:
    """Hashes of the release tree.

    Every key in ``entries`` is a path relative to the release root, and ``root`` is the
    placeholder ``"."``. The absolute location is deliberately not recorded: the manifest
    has to describe the tree wherever a reader happens to have unpacked it.
    """

    root: str = "."
    entries: dict[str, str] = field(default_factory=dict)

    @property
    def file_count(self) -> int:
        return len(self.entries)

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "paths_are": "relative to the release root",
            "file_count": self.file_count,
            "algorithm": "sha256",
            "entries": dict(sorted(self.entries.items())),
            "excluded": {
                "directories": sorted(EXCLUDED_DIRECTORIES),
                "suffixes": list(EXCLUDED_SUFFIXES),
                "files": [MANIFEST_NAME],
            },
        }


def build_manifest(root: Path, extra_excluded: frozenset[str] = frozenset()) -> Manifest:
    """Hash every tracked file under ``root``."""
    resolved = Path(root).resolve()
    manifest = Manifest()
    for path in sorted(resolved.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(resolved)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if relative.name == MANIFEST_NAME or relative.name in extra_excluded:
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        manifest.entries[str(relative)] = sha256_file(path)
    return manifest


def verify_manifest(manifest: Manifest, root: Path) -> dict[str, object]:
    """Re-hash the tree and report files added, removed or changed since the manifest."""
    resolved = Path(root).resolve()
    current = build_manifest(resolved)
    recorded = set(manifest.entries)
    present = set(current.entries)
    changed = sorted(
        name for name in recorded & present if manifest.entries[name] != current.entries[name]
    )
    return {
        "matches": not (recorded ^ present) and not changed,
        "added": sorted(present - recorded),
        "removed": sorted(recorded - present),
        "changed": changed,
        "recorded_files": len(recorded),
        "present_files": len(present),
    }


__all__ = [
    "EXCLUDED_DIRECTORIES",
    "EXCLUDED_SUFFIXES",
    "MANIFEST_NAME",
    "Manifest",
    "build_manifest",
    "verify_manifest",
]
