"""Cine layer: the fluoroscopic frame index and the spatial-predicate label.

Ref: Sec. 2.2, p. 3 - the spatial predicate is applied to the clinical fluoroscopic
stream; Sec. 4.5, pp. 18-19 - the cine database is one of the linked layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.types import BoolArray, FloatArray
from .tables import read_csv_rows

_RUN_GAP_S = 1.0


@dataclass(frozen=True)
class CineIndex:
    """Frame times, acquisition quality, the predicate label and the annotation flag."""

    t_s: FloatArray
    subtraction_quality: FloatArray
    device_in_target: BoolArray
    mask_annotated: BoolArray

    def __len__(self) -> int:
        return int(self.t_s.shape[0])

    def frames_prior(self, when: float) -> float:
        """Frames acquired strictly before ``when``."""
        return float(np.count_nonzero(self.t_s < when))

    def frame_rate(self, start_s: float, end_s: float) -> float:
        """Frames per second inside the window."""
        span = end_s - start_s
        if span <= 0.0:
            return 0.0
        inside = np.count_nonzero((self.t_s >= start_s) & (self.t_s < end_s))
        return float(inside) / span

    def mean_subtraction_quality(self, start_s: float, end_s: float) -> float:
        """Mean acquisition quality of the frames inside the window."""
        mask = (self.t_s >= start_s) & (self.t_s < end_s)
        if not bool(mask.any()):
            return 0.0
        return float(np.mean(self.subtraction_quality[mask]))

    def runs_before(self, when: float) -> float:
        """Number of cine runs that started strictly before ``when``."""
        if self.t_s.size == 0:
            return 0.0
        breaks = np.nonzero(np.diff(self.t_s) > _RUN_GAP_S)[0]
        starts = np.concatenate([[0], breaks + 1])
        return float(np.count_nonzero(self.t_s[starts] < when))

    def run_duration(self, start_s: float, end_s: float) -> float:
        """Length of the cine run covering the window.

        A run is a maximal group of frames separated by less than ``_RUN_GAP_S``; the
        returned value is the part of that run's span that falls inside the window.
        """
        if self.t_s.size == 0:
            return 0.0
        breaks = np.nonzero(np.diff(self.t_s) > _RUN_GAP_S)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [self.t_s.size - 1]])
        for begin, stop in zip(starts, ends, strict=True):
            run_start = float(self.t_s[begin])
            run_end = float(self.t_s[stop])
            if run_end >= start_s and run_start < end_s:
                return max(0.0, min(run_end, end_s) - max(run_start, start_s))
        return 0.0

    def predicate_label(self, start_s: float, end_s: float) -> float:
        """Share of frames in the window that the spatial predicate calls positive."""
        mask = (self.t_s >= start_s) & (self.t_s < end_s)
        if not bool(mask.any()):
            return 0.0
        return float(np.mean(self.device_in_target[mask]))

    def frames_in_window(self, start_s: float, end_s: float) -> FloatArray:
        """Frame times inside the window, used by the perception fixture loader."""
        return self.t_s[(self.t_s >= start_s) & (self.t_s < end_s)]


def read_cine_index(path: str | Path) -> CineIndex:
    """Parse a cine index CSV."""
    rows = read_csv_rows(path)
    times = np.asarray([float(row["t_s"]) for row in rows], dtype=np.float64)
    quality = np.asarray([float(row["subtraction_quality"]) for row in rows], dtype=np.float64)
    label = np.asarray(
        [row["device_in_target"].strip().lower() in {"1", "true", "yes", "y"} for row in rows],
        dtype=np.bool_,
    )
    annotated = np.asarray(
        [row["mask_annotated"].strip().lower() in {"1", "true", "yes", "y"} for row in rows],
        dtype=np.bool_,
    )
    order = np.argsort(times, kind="stable")
    return CineIndex(
        t_s=times[order],
        subtraction_quality=quality[order],
        device_in_target=label[order],
        mask_annotated=annotated[order],
    )


__all__ = ["CineIndex", "read_cine_index"]
