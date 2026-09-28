"""Covariate derivation for the four blocks.

Ref: Sec. 4.6, p. 19 - the analysis gate requires the imaging block, the kinematic block,
the process block and the performer block to be present beforehand; they are built from
the navigation record and the cine layer, never from anything measured after the outcome.

Only the block *names* are printed by the manuscript. The members below are the released
analytic schema and are recorded as engineering defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol.schema import CASE_MIX, COVARIATE_BLOCKS
from .cine import CineIndex
from .navigation_log import NavigationLog, kinematic_window
from .tables import ProcedureRecord

_DWELL_SPEED_MM_PER_S = 5.0


@dataclass(frozen=True)
class WindowContext:
    """Everything needed to build the covariates of one window of one procedure."""

    record: ProcedureRecord
    log: NavigationLog
    cine: CineIndex
    annual_procedure_volume: float
    operator_case_volume_prior_30d: float
    console_sessions_this_week: float
    dwell_speed_mm_per_s: float = _DWELL_SPEED_MM_PER_S


def window_covariates(context: WindowContext, start_s: float, end_s: float) -> dict[str, float]:
    """Build every covariate column of one person-window."""
    record = context.record
    times, xs, ys = context.log.device_track()
    advance_rate, curvature, reversals, dwell = kinematic_window(
        times, xs, ys, start_s, end_s, context.dwell_speed_mm_per_s
    )

    values: dict[str, float] = {
        "calcification_grade": record.calcification_grade,
        "lesion_length_mm": record.lesion_length_mm,
        "reference_diameter_mm": record.reference_diameter_mm,
        "cine_frames_prior": context.cine.frames_prior(start_s),
        "subtraction_quality": context.cine.mean_subtraction_quality(start_s, end_s),
        "advance_rate_mm_per_s": advance_rate,
        "tip_path_curvature": curvature,
        "motion_reversals": float(reversals),
        "run_duration_s": context.cine.run_duration(start_s, end_s),
        "dwell_time_s": dwell,
        "elapsed_procedure_time_s": float(start_s),
        "stage_index": float(context.log.stage_before(start_s)),
        "guidewire_exchanges": float(context.log.count_before("exchange", start_s)),
        "cine_runs_prior": context.cine.runs_before(start_s),
        "prior_device_advance_mm": context.log.device_path_before(start_s),
        "operator_experience_years": record.operator_experience_years,
        "annual_procedure_volume": context.annual_procedure_volume,
        "operator_case_volume_prior_30d": context.operator_case_volume_prior_30d,
        "console_sessions_this_week": context.console_sessions_this_week,
        "age_years": record.age_years,
        "female": float(record.female),
        "diabetes_mellitus": float(record.diabetes_mellitus),
        "chronic_limb_threatening_ischaemia": float(record.chronic_limb_threatening_ischaemia),
    }
    absent = [
        name for block in COVARIATE_BLOCKS.values() for name in block if name not in values
    ] + [name for name in CASE_MIX if name not in values]
    if absent:
        raise KeyError(f"covariate derivation left columns unset: {absent}")
    return values


def operator_history(
    record: ProcedureRecord, operator_records: list[ProcedureRecord], window_days: int
) -> tuple[float, float]:
    """Case volume of the operator in the preceding window and in the current week.

    Returns ``(case_volume_prior_30d, console_sessions_this_week)``. Both are counted from
    the operating-report layer only, so they are fixed before the window opens.
    """
    prior = [
        other
        for other in operator_records
        if other.operator_id == record.operator_id
        and 0 < record.procedure_day - other.procedure_day <= window_days
    ]
    week = [
        other
        for other in operator_records
        if other.operator_id == record.operator_id
        and 0 <= record.procedure_day - other.procedure_day < 7
    ]
    return float(len(prior)), float(len(week))


def block_histories(records: list[ProcedureRecord]) -> dict[int, tuple[float, float]]:
    """Per-procedure operator history for a whole cohort, computed in one pass."""
    by_operator: dict[int, list[ProcedureRecord]] = {}
    for record in records:
        by_operator.setdefault(record.operator_id, []).append(record)
    for bucket in by_operator.values():
        bucket.sort(key=lambda item: (item.procedure_day, item.procedure_id))
    history: dict[int, tuple[float, float]] = {}
    for record in records:
        history[record.procedure_id] = operator_history(record, by_operator[record.operator_id], 30)
    return history


__all__ = [
    "WindowContext",
    "block_histories",
    "operator_history",
    "window_covariates",
]
