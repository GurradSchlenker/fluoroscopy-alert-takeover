"""Person-window assembly and exposure-state assignment.

Ref: Sec. 4.2, p. 14 (time zero, follow-up, and the two treatment strategies);
Sec. 4.3, p. 15 (the alert state is defined through logged events with timestamps
rather than through a hidden score, so that consistency is probeable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.types import BoolArray, FloatArray, IntArray


@dataclass(frozen=True)
class WindowSpec:
    """Layout of the pre-specified analysis periods along a procedure timeline.

    The manuscript fixes the *shape* of the protocol (a pre-established period, a
    time-zero, follow-up to takeover or procedure end) but prints neither the period
    length, the stride, nor the alert-to-response window inside which a hand-back counts
    as a response to the alert. All three are engineering defaults.
    """

    length_s: float = 30.0
    stride_s: float = 15.0
    response_s: float = 30.0
    min_coverage: float = 0.5


def lay_windows(
    duration_s: float, spec: WindowSpec, offset_s: float = 0.0
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` pairs covering ``duration_s`` in fixed-width periods.

    A trailing period is kept only when it covers at least ``min_coverage`` of a full
    window, so a procedure does not contribute a stub window that no alert could reach.
    """
    if duration_s <= 0.0:
        return []
    if spec.length_s <= 0.0 or spec.stride_s <= 0.0:
        raise ValueError("window length and stride must be positive")
    windows: list[tuple[float, float]] = []
    start = offset_s
    limit = offset_s + duration_s
    while start < limit:
        end = min(start + spec.length_s, limit)
        if end - start >= spec.min_coverage * spec.length_s:
            windows.append((start, end))
        start += spec.stride_s
    return windows


def assign_alert_state(onsets: FloatArray, start_s: float, end_s: float) -> bool:
    """Exposure state of one window: an alert logged inside it, or none.

    Ref: Sec. 4.3, p. 15 - the state is read from timestamped log events inside the
    frozen policy edition, never from a latent score.
    """
    inside = onsets[(onsets >= start_s) & (onsets < end_s) & np.isfinite(onsets)]
    return bool(inside.size > 0)


def alert_onset_in_window(onsets: FloatArray, start_s: float, end_s: float) -> float:
    """First logged alert onset inside the window, or ``nan`` when the window is unexposed."""
    inside = onsets[(onsets >= start_s) & (onsets < end_s) & np.isfinite(onsets)]
    if inside.size == 0:
        return float("nan")
    return float(np.min(inside))


def window_alert_states(onsets: FloatArray, starts: FloatArray, ends: FloatArray) -> BoolArray:
    """Vectorised :func:`assign_alert_state` over aligned window boundaries."""
    if not (onsets.shape[0] == starts.shape[0] == ends.shape[0]):
        raise ValueError("onsets, starts and ends must be aligned")
    return (onsets >= starts) & (onsets < ends) & np.isfinite(onsets)


def first_onset_per_window(onsets: FloatArray, starts: FloatArray, ends: FloatArray) -> FloatArray:
    """First onset inside each window, ``nan`` where the window is unexposed."""
    inside = (onsets >= starts) & (onsets < ends)
    flagged = np.where(inside, onsets, np.inf)
    first = np.min(flagged, axis=1) if flagged.ndim > 1 else flagged
    return np.asarray(np.where(np.isfinite(first), first, np.nan), dtype=np.float64)


def expand_windows(
    procedure_id: IntArray,
    duration_s: FloatArray,
    onsets_by_procedure: dict[int, FloatArray],
    spec: WindowSpec,
) -> tuple[IntArray, FloatArray, FloatArray, BoolArray, FloatArray]:
    """Build the person-window rows of a batch of procedures.

    Returns the procedure index, window start, window length, exposure state and alert
    onset for every window, in procedure order.
    """
    if not (procedure_id.shape[0] == duration_s.shape[0]):
        raise ValueError("procedure_id and duration_s must be aligned")
    rows_proc: list[int] = []
    rows_start: list[float] = []
    rows_length: list[float] = []
    rows_alert: list[bool] = []
    rows_onset: list[float] = []
    for index, procedure in enumerate(procedure_id):
        onsets = onsets_by_procedure.get(int(procedure), np.empty(0, dtype=np.float64))
        for start, end in lay_windows(float(duration_s[index]), spec):
            rows_proc.append(int(procedure))
            rows_start.append(start)
            rows_length.append(end - start)
            rows_alert.append(assign_alert_state(onsets, start, end))
            rows_onset.append(alert_onset_in_window(onsets, start, end))
    return (
        np.asarray(rows_proc, dtype=np.int64),
        np.asarray(rows_start, dtype=np.float64),
        np.asarray(rows_length, dtype=np.float64),
        np.asarray(rows_alert, dtype=np.bool_),
        np.asarray(rows_onset, dtype=np.float64),
    )


def follow_up_end(window_start_s: float, window_length_s: float, takeover_s: float) -> float:
    """Follow-up ends at the takeover event or the end of the period, whichever is first.

    Ref: Sec. 4.2, p. 14.
    """
    period_end = window_start_s + window_length_s
    if not np.isfinite(takeover_s):
        return period_end
    return float(min(takeover_s, period_end))


__all__ = [
    "WindowSpec",
    "alert_onset_in_window",
    "assign_alert_state",
    "expand_windows",
    "first_onset_per_window",
    "follow_up_end",
    "lay_windows",
    "window_alert_states",
]
