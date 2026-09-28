"""Analytic person-window schema and the four covariate blocks.

Ref: Sec. 4.5, pp. 18-19 (person-window table and the provenance layering);
Sec. 4.6, p. 19 ("imaging block, kinematic block, process block, and performer
block"); Sec. 2.5, p. 4 (the hand-back window).

The manuscript names the four blocks but prints neither their members nor their
columns; the members below are the released analytic schema and are engineering
defaults, recorded as such in ``verification_report.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from ..utils.types import BoolArray, FloatArray, IntArray, StrArray

IMAGING_BLOCK = (
    "calcification_grade",
    "lesion_length_mm",
    "reference_diameter_mm",
    "cine_frames_prior",
    "subtraction_quality",
)
KINEMATIC_BLOCK = (
    "advance_rate_mm_per_s",
    "tip_path_curvature",
    "motion_reversals",
    "run_duration_s",
    "dwell_time_s",
)
# Every member is fixed before the analysis period opens. Counters of earlier alerts or
# earlier takeovers are deliberately absent: they are descendants of the exposure inside
# the same procedure, and adjusting for them would block part of the effect being estimated.
WORKFLOW_BLOCK = (
    "elapsed_procedure_time_s",
    "stage_index",
    "guidewire_exchanges",
    "cine_runs_prior",
    "prior_device_advance_mm",
)
OPERATOR_BLOCK = (
    "operator_experience_years",
    "annual_procedure_volume",
    "operator_case_volume_prior_30d",
    "console_sessions_this_week",
)

COVARIATE_BLOCKS: dict[str, tuple[str, ...]] = {
    "imaging": IMAGING_BLOCK,
    "kinematic": KINEMATIC_BLOCK,
    "workflow": WORKFLOW_BLOCK,
    "operator": OPERATOR_BLOCK,
}

# Table 1, pp. 6-7, prints a standardised mean difference for every characteristic it
# lists, including the case-mix rows that sit outside the four blocks named in Sec. 4.6
# p. 19. They therefore enter the balance evaluation, but not the block-level gate
# distance, which Sec. 4.6 defines over the four blocks.
CASE_MIX: tuple[str, ...] = (
    "age_years",
    "female",
    "diabetes_mellitus",
    "chronic_limb_threatening_ischaemia",
)

# Table 4 Panel A, p. 11: the five pre-specified negative-control outcomes.
NEGATIVE_CONTROLS: tuple[str, ...] = (
    "emergency_representation_7d",
    "unrelated_diagnosis_admission",
    "contralateral_limb_intervention",
    "preprocedure_imaging_count",
    "anaesthesia_related_event",
)

# Sec. 2.4, p. 3: the thirty-day safety and treatment-success outcomes.
SAFETY_OUTCOMES: tuple[str, ...] = (
    "technical_success",
    "treatment_success",
    "limb_event",
    "major_adverse_cardiac_event",
    "death",
    "amputation",
    "transfusion_or_surgery",
)

STRATUM_LABELS: tuple[str, ...] = (
    "complexity_class",
    "infrapopliteal",
    "operator_experience_years",
    "annual_procedure_volume",
    "site",
    "region",
)


def _check_equal_lengths(table: PersonWindowTable) -> None:
    sizes = {table.alert.shape[0]}
    for field in fields(table):
        value = getattr(table, field.name)
        if isinstance(value, np.ndarray):
            sizes.add(value.shape[0])
        elif isinstance(value, dict):
            sizes.update(column.shape[0] for column in value.values())
    if len(sizes) != 1:
        raise ValueError(f"person-window columns disagree on length: {sorted(sizes)}")


@dataclass(frozen=True)
class PersonWindowTable:
    """One row per analytic person-window.

    ``alert`` is the frozen-policy exposure state, ``takeover`` the primary outcome and
    ``completion`` the secondary outcome; ``hand_back`` is the mediator and is derived
    from ``hand_back_delay_s`` against ``base_window_s`` (Sec. 2.5, p. 4).
    """

    patient_id: IntArray
    procedure_id: IntArray
    operator_id: IntArray
    site: StrArray
    region: StrArray
    window_index: IntArray
    window_start_s: FloatArray
    window_length_s: FloatArray
    alert: BoolArray
    alert_score: FloatArray
    alert_onset_s: FloatArray
    hand_back_delay_s: FloatArray
    base_window_s: float
    takeover: BoolArray
    completion: BoolArray
    hand_back: BoolArray
    complexity_class: IntArray
    infrapopliteal: BoolArray
    operator_experience_years: FloatArray
    annual_procedure_volume: FloatArray
    covariates: dict[str, FloatArray]
    negative_controls: dict[str, FloatArray]
    safety: dict[str, FloatArray]
    workflow: dict[str, FloatArray]
    linkage_complete: BoolArray

    def __post_init__(self) -> None:
        _check_equal_lengths(self)
        required = [name for block in COVARIATE_BLOCKS.values() for name in block] + list(CASE_MIX)
        missing = [name for name in required if name not in self.covariates]
        if missing:
            raise ValueError(f"covariate block members absent from the table: {missing}")
        absent = [name for name in NEGATIVE_CONTROLS if name not in self.negative_controls]
        if absent:
            raise ValueError(f"negative-control outcomes absent from the table: {absent}")
        absent_safety = [name for name in SAFETY_OUTCOMES if name not in self.safety]
        if absent_safety:
            raise ValueError(f"safety outcomes absent from the table: {absent_safety}")

    @property
    def size(self) -> int:
        return int(self.alert.shape[0])

    @property
    def n_patients(self) -> int:
        return int(np.unique(self.patient_id).shape[0])

    @property
    def n_sites(self) -> int:
        return int(np.unique(self.site).shape[0])

    def block(self, name: str) -> FloatArray:
        """Stack one covariate block into a dense ``(n, p)`` matrix."""
        if name not in COVARIATE_BLOCKS:
            raise KeyError(f"unknown covariate block: {name}")
        return self.block_columns(COVARIATE_BLOCKS[name])

    def block_columns(self, names: tuple[str, ...]) -> FloatArray:
        """Stack named covariate columns into a dense ``(n, p)`` matrix."""
        return np.column_stack([self.covariates[name] for name in names])

    def design(self, block_names: tuple[str, ...] | None = None) -> FloatArray:
        """Stack the requested blocks (default: all four) into ``(n, p)``."""
        selected = tuple(COVARIATE_BLOCKS) if block_names is None else block_names
        columns: list[FloatArray] = []
        for name in selected:
            columns.append(self.block(name))
        return np.hstack(columns)

    def full_design(self, block_names: tuple[str, ...] | None = None) -> FloatArray:
        """The requested blocks plus case-mix: the design matrix the models are fitted on.

        The case-mix rows are always carried, so a reduced adjustment set in the Table 2
        ablation removes a history block rather than also dropping the case mix.
        """
        return np.hstack([self.design(block_names), self.block_columns(CASE_MIX)])

    def balance_groups(self) -> dict[str, tuple[str, ...]]:
        """Named column groups for balance evaluation, matching Table 1's row structure."""
        groups: dict[str, tuple[str, ...]] = dict(COVARIATE_BLOCKS)
        groups["case_mix"] = CASE_MIX
        return groups

    def column_names(self, block_names: tuple[str, ...] | None = None) -> list[str]:
        """Names of the columns returned by :meth:`design`, in the same order."""
        selected = tuple(COVARIATE_BLOCKS) if block_names is None else block_names
        names: list[str] = []
        for name in selected:
            names.extend(COVARIATE_BLOCKS[name])
        return names

    def outcome(self, name: str) -> FloatArray:
        """Read a named outcome as a float vector."""
        if name in self.negative_controls:
            return self.negative_controls[name]
        if name in self.safety:
            return self.safety[name]
        if name == "takeover":
            return self.takeover.astype(np.float64)
        if name == "completion":
            return self.completion.astype(np.float64)
        if name == "hand_back":
            return self.hand_back.astype(np.float64)
        raise KeyError(f"unknown outcome: {name}")

    def subset(self, mask: BoolArray) -> PersonWindowTable:
        """Return the rows selected by ``mask`` as a new table."""

        def take(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value[mask]
            if isinstance(value, dict):
                return {key: column[mask] for key, column in value.items()}
            return value

        return replace(
            self, **{field.name: take(getattr(self, field.name)) for field in fields(self)}
        )

    def with_mediator_window(self, multiplier: float) -> PersonWindowTable:
        """Rebuild the mediator at a multiplied alert-to-response window.

        Ref: Table 4 Panel B, p. 11 - the window-length attenuation prediction. Only the
        mediator changes; exposure and outcome are held at their pre-specified values.
        """
        width = multiplier * self.base_window_s
        responded = np.isfinite(self.hand_back_delay_s) & (self.hand_back_delay_s <= width)
        return replace(self, hand_back=responded)


def empty_table(size: int, base_window_s: float) -> PersonWindowTable:
    """Allocate a zeroed table of ``size`` rows.

    Used by the cohort builder to describe a cohort before any record is read, and by the
    regression tests that check a table rejects a column of the wrong length.
    """
    zeros_f = np.zeros(size, dtype=np.float64)
    zeros_b = np.zeros(size, dtype=np.bool_)
    members = [name for block in COVARIATE_BLOCKS.values() for name in block] + list(CASE_MIX)
    zero_columns = {name: zeros_f.copy() for name in members}
    return PersonWindowTable(
        patient_id=np.zeros(size, dtype=np.int64),
        procedure_id=np.zeros(size, dtype=np.int64),
        operator_id=np.zeros(size, dtype=np.int64),
        site=np.full(size, "", dtype=np.str_),
        region=np.full(size, "", dtype=np.str_),
        window_index=np.zeros(size, dtype=np.int64),
        window_start_s=zeros_f.copy(),
        window_length_s=zeros_f.copy(),
        alert=zeros_b.copy(),
        alert_score=zeros_f.copy(),
        alert_onset_s=zeros_f.copy(),
        hand_back_delay_s=zeros_f.copy(),
        base_window_s=base_window_s,
        takeover=zeros_b.copy(),
        completion=zeros_b.copy(),
        hand_back=zeros_b.copy(),
        complexity_class=np.zeros(size, dtype=np.int64),
        infrapopliteal=zeros_b.copy(),
        operator_experience_years=zeros_f.copy(),
        annual_procedure_volume=zeros_f.copy(),
        covariates=zero_columns,
        negative_controls={name: zeros_f.copy() for name in NEGATIVE_CONTROLS},
        safety={name: zeros_f.copy() for name in SAFETY_OUTCOMES},
        workflow={},
        linkage_complete=np.ones(size, dtype=np.bool_),
    )


__all__ = [
    "CASE_MIX",
    "COVARIATE_BLOCKS",
    "IMAGING_BLOCK",
    "KINEMATIC_BLOCK",
    "NEGATIVE_CONTROLS",
    "OPERATOR_BLOCK",
    "SAFETY_OUTCOMES",
    "STRATUM_LABELS",
    "WORKFLOW_BLOCK",
    "PersonWindowTable",
    "empty_table",
]
