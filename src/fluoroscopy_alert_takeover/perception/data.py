"""Frame store and the person-window dataset of the alert policy.

Ref: Sec. 2.2, p. 3 - the predicate is applied to the clinical fluoroscopic stream;
Sec. 4.6, p. 19 - the policy reads the navigation record and the fluoroscopic stream.

The record-level cine frames are held under site data-sharing agreements and are not
distributed; the store below is the reader for the on-disk layout only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..cohort.cine import read_cine_index
from ..cohort.layout import CohortLayout
from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray

FRAMES_DIR = "frames"
MASKS_DIR = "masks"


class WindowSample(TypedDict):
    """One training item: a frame sequence and its three targets."""

    frames: Tensor
    mask: Tensor
    frame_alert: Tensor
    window_alert: Tensor
    mask_observed: Tensor
    site: Tensor
    row: Tensor


@dataclass(frozen=True)
class FrameStore:
    """Cine frames and reference masks, one array per procedure."""

    root: Path
    image_size: tuple[int, int] = (128, 128)

    def frames(self, procedure_id: int) -> FloatArray:
        path = self.root / FRAMES_DIR / f"{procedure_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"frame array missing for procedure {procedure_id}: {path}")
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)

    def masks(self, procedure_id: int) -> BoolArray:
        path = self.root / MASKS_DIR / f"{procedure_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"mask array missing for procedure {procedure_id}: {path}")
        return np.asarray(np.load(path, allow_pickle=False) > 0, dtype=np.bool_)

    def resize(self, frames: FloatArray) -> FloatArray:
        """Nearest-neighbour reshape onto the analysis grid."""
        height, width = self.image_size
        if frames.shape[-2:] == (height, width):
            return frames
        tensor = torch.as_tensor(frames, dtype=torch.float32)
        resized = torch.nn.functional.interpolate(
            tensor[:, None], size=(height, width), mode="nearest"
        )
        return np.asarray(resized[:, 0].numpy(), dtype=np.float64)


def _even_indices(count: int, wanted: int) -> np.ndarray:
    """``wanted`` frame positions spread evenly over ``count`` frames.

    When ``wanted`` exceeds ``count`` the linspace repeats positions, so a short cine run
    still fills the analysis grid.
    """
    if count <= 0:
        raise ValueError("a window sample needs at least one frame")
    return np.floor(np.linspace(0, count - 1, wanted)).astype(np.int64)


def site_order(table: PersonWindowTable) -> dict[str, int]:
    """Deterministic site to index mapping, ordered by site label."""
    return {site: index for index, site in enumerate(sorted({str(value) for value in table.site}))}


class FluoroscopyWindowDataset(Dataset[WindowSample]):
    """Person-windows as frame sequences with predicate, frame-alert and window-alert targets.

    The window-alert target is the exposure state of the frozen policy edition, not the
    model's own score, so the policy is trained against the edition that the analysis
    contrast uses.
    """

    def __init__(
        self,
        table: PersonWindowTable,
        store: FrameStore,
        window_frames: int = 8,
        rows: BoolArray | None = None,
        sites: dict[str, int] | None = None,
    ) -> None:
        if window_frames < 1:
            raise ValueError("window_frames must be positive")
        self.table = table
        self.store = store
        self.window_frames = window_frames
        self.sites = site_order(table) if sites is None else dict(sites)
        self.index = (
            np.arange(table.size, dtype=np.int64)
            if rows is None
            else np.nonzero(rows)[0].astype(np.int64)
        )
        self._cache: dict[int, tuple[FloatArray, BoolArray]] = {}
        self._cine: dict[int, tuple[FloatArray, BoolArray, BoolArray]] = {}

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def _cine_columns(self, procedure_id: int) -> tuple[FloatArray, BoolArray, BoolArray]:
        if procedure_id not in self._cine:
            layout = CohortLayout(root=self.store.root)
            cine = read_cine_index(layout.cine_index(procedure_id))
            self._cine[procedure_id] = (cine.t_s, cine.device_in_target, cine.mask_annotated)
        return self._cine[procedure_id]

    def _arrays(self, procedure_id: int) -> tuple[FloatArray, BoolArray]:
        if procedure_id not in self._cache:
            frames = self.store.resize(self.store.frames(procedure_id))
            masks = self.store.resize(self.store.masks(procedure_id).astype(np.float64)) > 0.5
            self._cache[procedure_id] = (frames, masks)
        return self._cache[procedure_id]

    def __getitem__(self, position: int) -> WindowSample:
        row = int(self.index[position])
        procedure_id = int(self.table.procedure_id[row])
        frames, masks = self._arrays(procedure_id)
        times, frame_label, annotated = self._cine_columns(procedure_id)
        start = float(self.table.window_start_s[row])
        end = start + float(self.table.window_length_s[row])
        inside = np.nonzero((times >= start) & (times < end))[0]
        if inside.size == 0:
            inside = np.asarray([int(np.argmin(np.abs(times - start)))], dtype=np.int64)
        picked = inside[_even_indices(inside.size, self.window_frames)]
        images = []
        for position_index in picked:
            index = int(position_index)
            channel = np.asarray(frames[index], dtype=np.float64)[None]
            images.append(channel)
        stacked = np.stack(images, axis=0)
        mask_stack = np.stack([masks[int(index)] for index in picked], axis=0)
        observed = np.asarray([bool(annotated[int(index)]) for index in picked], dtype=np.bool_)
        labels = np.asarray([bool(frame_label[int(index)]) for index in picked], dtype=np.bool_)
        site = str(self.table.site[row])
        if site not in self.sites:
            raise KeyError(f"site outside the dataset's site order: {site}")
        return WindowSample(
            frames=torch.as_tensor(stacked, dtype=torch.float32),
            mask=torch.as_tensor(mask_stack, dtype=torch.float32)[:, None],
            frame_alert=torch.as_tensor(labels.astype(np.float32)),
            window_alert=torch.as_tensor([float(self.table.alert[row])], dtype=torch.float32),
            mask_observed=torch.as_tensor(observed.astype(np.float32)),
            site=torch.as_tensor([float(self.sites[site])], dtype=torch.float32),
            row=torch.as_tensor([row], dtype=torch.int64),
        )


def collate(samples: list[WindowSample]) -> WindowSample:
    """Stack window samples into a batch."""
    return WindowSample(
        frames=torch.stack([sample["frames"] for sample in samples]),
        mask=torch.stack([sample["mask"] for sample in samples]),
        frame_alert=torch.stack([sample["frame_alert"] for sample in samples]),
        window_alert=torch.stack([sample["window_alert"] for sample in samples]),
        mask_observed=torch.stack([sample["mask_observed"] for sample in samples]),
        site=torch.stack([sample["site"] for sample in samples]),
        row=torch.stack([sample["row"] for sample in samples]),
    )


__all__ = [
    "FRAMES_DIR",
    "MASKS_DIR",
    "FluoroscopyWindowDataset",
    "FrameStore",
    "WindowSample",
    "collate",
    "site_order",
]
