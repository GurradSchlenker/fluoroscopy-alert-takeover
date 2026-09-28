"""Navigation-log layer: timestamped device and alert events.

Ref: Sec. 4.3, p. 15 - the alert state is read from logged events carrying timestamps;
Sec. 4.6, p. 19 - "The detection policy is based on certain criteria of the navigation
record". The log carries no score that decides the exposure state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.types import FloatArray, IntArray, StrArray

_MISSING = float("nan")


@dataclass(frozen=True)
class NavigationLog:
    """One procedure's navigation events, sorted by time."""

    t_s: FloatArray
    kind: StrArray
    score: FloatArray
    x_mm: FloatArray
    y_mm: FloatArray
    stage_index: IntArray

    def __len__(self) -> int:
        return int(self.t_s.shape[0])

    def _select(self, kinds: tuple[str, ...]) -> np.ndarray:
        return np.isin(self.kind, np.asarray(kinds, dtype=np.str_))

    def onsets(self) -> FloatArray:
        """Alert onsets in the frozen policy edition."""
        return np.sort(self.t_s[self._select(("alert",))])

    def scored_times(self) -> tuple[FloatArray, FloatArray]:
        """Every window score the deployed policy logged, in time order.

        Ref: Algorithm 5, p. 20 - the operating threshold is swept by regenerating the
        alert stream from the frozen policy, which needs a score per candidate window.
        """
        mask = self._select(("score", "alert"))
        order = np.argsort(self.t_s[mask], kind="stable")
        return self.t_s[mask][order], self.score[mask][order]

    def max_score(self, start_s: float, end_s: float) -> float:
        """Highest policy score inside a window; ``0.0`` when the window was not scored."""
        times, scores = self.scored_times()
        inside = (times >= start_s) & (times < end_s) & np.isfinite(scores)
        if not bool(inside.any()):
            return 0.0
        return float(np.max(scores[inside]))

    def hand_back_times(self) -> FloatArray:
        """Logged hand-back events."""
        return np.sort(self.t_s[self._select(("hand_back",))])

    def takeover_time(self) -> float:
        """First logged takeover, or ``nan`` when the surgeon never took over."""
        selected = self.t_s[self._select(("takeover",))]
        if selected.size == 0:
            return _MISSING
        return float(np.min(selected))

    def count_before(self, kind: str, when: float) -> int:
        """Number of events of one kind strictly before ``when``."""
        selected = self.t_s[self._select((kind,)) & (self.t_s < when)]
        return int(selected.size)

    def stage_before(self, when: float) -> int:
        """Highest stage index entered before ``when``; ``0`` when none was logged."""
        mask = self._select(("stage",)) & (self.t_s < when)
        if not bool(mask.any()):
            return 0
        return int(np.max(self.stage_index[mask]))

    def device_track(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Device positions in time order."""
        mask = self._select(("device",))
        order = np.argsort(self.t_s[mask])
        return self.t_s[mask][order], self.x_mm[mask][order], self.y_mm[mask][order]

    def device_path_before(self, when: float) -> float:
        """Cumulative device travel strictly before ``when``, in millimetres.

        A pre-window quantity: the guidewire has already moved by the amount it has moved,
        whatever the alert does later.
        """
        times, xs, ys = self.device_track()
        keep = times < when
        if int(np.count_nonzero(keep)) < 2:
            return 0.0
        steps = np.hypot(np.diff(xs[keep]), np.diff(ys[keep]))
        return float(np.sum(steps))


def read_navigation_log(path: str | Path) -> NavigationLog:
    """Parse a JSON Lines navigation log.

    Rows that omit a field take the schema default: ``score``, ``x_mm`` and ``y_mm``
    become ``nan`` and ``stage_index`` becomes ``-1``.
    """
    times: list[float] = []
    kinds: list[str] = []
    scores: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    stages: list[int] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed log row") from error
            if "t_s" not in row or "kind" not in row:
                raise ValueError(f"{path}:{line_number}: log row needs t_s and kind")
            times.append(float(row["t_s"]))
            kinds.append(str(row["kind"]))
            scores.append(float(row.get("score", _MISSING)))
            xs.append(float(row.get("x_mm", _MISSING)))
            ys.append(float(row.get("y_mm", _MISSING)))
            stages.append(int(row.get("stage_index", -1)))
    order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
    return NavigationLog(
        t_s=np.asarray(times, dtype=np.float64)[order],
        kind=np.asarray(kinds, dtype=np.str_)[order],
        score=np.asarray(scores, dtype=np.float64)[order],
        x_mm=np.asarray(xs, dtype=np.float64)[order],
        y_mm=np.asarray(ys, dtype=np.float64)[order],
        stage_index=np.asarray(stages, dtype=np.int64)[order],
    )


def kinematic_window(
    times: FloatArray,
    xs: FloatArray,
    ys: FloatArray,
    start_s: float,
    end_s: float,
    dwell_speed: float,
) -> tuple[float, float, int, float]:
    """Kinematic summaries of one window.

    Returns ``(advance_rate_mm_per_s, tip_path_curvature, motion_reversals,
    dwell_time_s)``. The manuscript names the kinematic block (Sec. 4.6, p. 19) but not
    its members; these four plus the cine run length make up the released schema.
    """
    inside = (times >= start_s) & (times < end_s) & np.isfinite(xs) & np.isfinite(ys)
    window_times = times[inside]
    window_x = xs[inside]
    window_y = ys[inside]
    if window_times.size < 2:
        return 0.0, 0.0, 0, float(end_s - start_s)

    deltas = np.diff(window_times)
    steps = np.hypot(np.diff(window_x), np.diff(window_y))
    span = float(window_times[-1] - window_times[0])
    path_length = float(steps.sum())
    advance_rate = path_length / span if span > 0.0 else 0.0

    headings = np.arctan2(np.diff(window_y), np.diff(window_x))
    turns = np.abs(np.diff(headings))
    turns = np.where(turns > np.pi, 2.0 * np.pi - turns, turns)
    curvature = float(turns.sum() / path_length) if path_length > 0.0 else 0.0

    mean_heading = float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))
    along = steps * np.cos(headings - mean_heading)
    signs = np.sign(along)
    signs = signs[signs != 0.0]
    reversals = int(np.count_nonzero(np.diff(signs) != 0.0))

    speeds = steps / np.where(deltas > 0.0, deltas, np.inf)
    dwell = float(deltas[speeds < dwell_speed].sum())
    return advance_rate, curvature, reversals, dwell


__all__ = ["NavigationLog", "kinematic_window", "read_navigation_log"]
