"""The estimator, adjustment-set and nuisance ablation of Table 2.

Ref: Table 2, p. 8, and its caption: "All rows estimate the same prespecified causal
contrast on the certified-overlap region; only the estimator or the adjustment set
changes. ... Estimator spread, not estimator ranking, is the quantity of interest: the two
singly-robust estimators disagree by 2.1 pp, while the doubly-robust estimate moves by less
than 0.2 pp when either nuisance family is replaced by a machine-learning alternative."

The primary specification is the cross-fitted AIPW estimator on the full history, so the
release asserts that this row reproduces the primary contrast exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exposure.family import DEFAULT_BLOCKS
from ..exposure.model import fit_balance_first_exposure, in_sample_propensity
from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray, IntArray
from .aipw import aipw_contrast
from .crossfit import clip_propensity, cross_fitted_nuisance, table_design
from .influence import InfluenceEstimate
from .singly_robust import ipw_contrast, outcome_regression_contrast
from .tmle import tmle_contrast

ADJUSTMENT_SETS: dict[str, tuple[str, ...] | None] = {
    "full_history": None,
    "imaging_only": ("imaging",),
    "workflow_only": ("workflow",),
    "operator_only": ("operator",),
}

ESTIMATORS: tuple[str, ...] = ("aipw", "tmle", "ipw", "outcome_regression")


@dataclass(frozen=True)
class AblationSpec:
    """One row of the Table 2 grid."""

    label: str
    estimator: str
    adjustment: str
    nuisance: str = "glm"
    cross_fit: bool = True
    gated: bool = True

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator: {self.estimator}")
        if self.adjustment not in ADJUSTMENT_SETS:
            raise ValueError(f"unknown adjustment set: {self.adjustment}")
        if self.estimator != "aipw" and self.nuisance != "glm":
            raise ValueError("the nuisance family is only varied for the AIPW rows")


ABLATION_GRID: tuple[AblationSpec, ...] = (
    AblationSpec("aipw_cross_fitted_full_history", "aipw", "full_history"),
    AblationSpec("tmle_cross_fitted_full_history", "tmle", "full_history"),
    AblationSpec("ipw_only_full_history", "ipw", "full_history"),
    AblationSpec("outcome_regression_only_full_history", "outcome_regression", "full_history"),
    AblationSpec("aipw_reduced_imaging_block", "aipw", "imaging_only"),
    AblationSpec("aipw_reduced_workflow_block", "aipw", "workflow_only"),
    AblationSpec("aipw_reduced_operator_block", "aipw", "operator_only"),
    AblationSpec("aipw_single_fold_no_cross_fitting", "aipw", "full_history", cross_fit=False),
    AblationSpec("aipw_glm_nuisance", "aipw", "full_history", nuisance="glm"),
    AblationSpec(
        "aipw_gradient_boosted_nuisance", "aipw", "full_history", nuisance="gradient_boosted"
    ),
    AblationSpec("aipw_random_forest_nuisance", "aipw", "full_history", nuisance="random_forest"),
    AblationSpec("aipw_ungated_full_cohort", "aipw", "full_history", gated=False),
)


@dataclass(frozen=True)
class AblationRow:
    """Both causal contrasts of one Table 2 row."""

    spec: AblationSpec
    takeover: InfluenceEstimate
    completion: InfluenceEstimate

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.spec.label,
            "estimator": self.spec.estimator,
            "adjustment": self.spec.adjustment,
            "nuisance": self.spec.nuisance,
            "cross_fitted": self.spec.cross_fit,
            "gated": self.spec.gated,
            "takeover": self.takeover.as_dict(),
            "completion": self.completion.as_dict(),
        }


def _propensity_for(
    table: PersonWindowTable, folds: IntArray, spec: AblationSpec
) -> tuple[FloatArray, str, float]:
    """Exposure propensity of one ablation row.

    The family is re-selected on the adjustment set the row uses, because a reduced
    history changes which family balances best; the region itself does not move, which is
    what the caption of Table 2 requires.
    """
    blocks = ADJUSTMENT_SETS[spec.adjustment]
    if spec.cross_fit:
        selected = fit_balance_first_exposure(
            table, folds, block_names=DEFAULT_BLOCKS if blocks is None else blocks
        )
        return selected.propensity, selected.family, selected.strength
    return in_sample_propensity(table, "regularised_logistic", 0.1), "regularised_logistic", 0.1


def run_ablation(
    table: PersonWindowTable,
    folds: IntArray,
    region: BoolArray,
    outcome_name: str,
    spec: AblationSpec,
    clip_gamma: float,
) -> InfluenceEstimate:
    """Estimate one Table 2 row."""
    blocks = ADJUSTMENT_SETS[spec.adjustment]
    raw_propensity, _, _ = _propensity_for(table, folds, spec)
    trimmed = clip_propensity(raw_propensity, clip_gamma)
    alert = table.alert.astype(np.int64)
    outcome = table.outcome(outcome_name)
    nuisance = cross_fitted_nuisance(
        design=table_design(table, blocks),
        alert=alert,
        outcome=outcome,
        propensity=trimmed.propensity,
        folds=folds,
        learner=spec.nuisance,
        cross_fit=spec.cross_fit,
    )
    analysis_region = region if spec.gated else np.ones(table.size, dtype=np.bool_)
    name = f"{outcome_name}:{spec.label}"
    if spec.estimator == "aipw":
        return aipw_contrast(
            name,
            alert.astype(np.float64),
            outcome,
            nuisance.propensity,
            nuisance.mu_treated,
            nuisance.mu_control,
            analysis_region,
        )
    if spec.estimator == "tmle":
        return tmle_contrast(
            name,
            alert.astype(np.float64),
            outcome,
            nuisance.propensity,
            nuisance.mu_treated,
            nuisance.mu_control,
            analysis_region,
        )
    if spec.estimator == "ipw":
        return ipw_contrast(
            name, alert.astype(np.float64), outcome, nuisance.propensity, analysis_region
        )
    return outcome_regression_contrast(
        name, nuisance.mu_treated, nuisance.mu_control, analysis_region
    )


def run_ablation_grid(
    table: PersonWindowTable,
    folds: IntArray,
    region: BoolArray,
    outcomes: tuple[str, str],
    clip_gamma: float,
    grid: tuple[AblationSpec, ...] = ABLATION_GRID,
) -> list[AblationRow]:
    """Run every row of the grid for both reported outcomes."""
    rows: list[AblationRow] = []
    for spec in grid:
        rows.append(
            AblationRow(
                spec=spec,
                takeover=run_ablation(table, folds, region, outcomes[0], spec, clip_gamma),
                completion=run_ablation(table, folds, region, outcomes[1], spec, clip_gamma),
            )
        )
    return rows


def estimator_spread(rows: list[AblationRow], outcome: str) -> float:
    """Disagreement of the two singly-robust estimators, in percentage points.

    Ref: Table 2, p. 8 - "the two singly-robust estimators disagree by 2.1 pp". The
    quantity is the absolute gap between the IPW-only and outcome-regression-only rows.
    """
    ipw = next(row for row in rows if row.spec.label == "ipw_only_full_history")
    regression = next(
        row for row in rows if row.spec.label == "outcome_regression_only_full_history"
    )
    return float(abs(100.0 * _pick(ipw, outcome).point - 100.0 * _pick(regression, outcome).point))


def nuisance_drift(rows: list[AblationRow], outcome: str) -> float:
    """Largest move of the doubly-robust estimate across the three nuisance families."""
    labels = (
        "aipw_glm_nuisance",
        "aipw_gradient_boosted_nuisance",
        "aipw_random_forest_nuisance",
        "aipw_cross_fitted_full_history",
    )
    points = [
        100.0 * _pick(next(row for row in rows if row.spec.label == label), outcome).point
        for label in labels
    ]
    return float(max(points) - min(points))


def ablation_deficit_price(rows: list[AblationRow], outcome: str) -> float:
    """Gated contrast minus ungated contrast, in percentage points (Algorithm 1, step 12)."""
    gated = next(row for row in rows if row.spec.label == "aipw_cross_fitted_full_history")
    ungated = next(row for row in rows if row.spec.label == "aipw_ungated_full_cohort")
    return float(100.0 * _pick(gated, outcome).point - 100.0 * _pick(ungated, outcome).point)


def _pick(row: AblationRow, outcome: str) -> InfluenceEstimate:
    if outcome in {"takeover", "primary"}:
        return row.takeover
    if outcome in {"completion", "secondary"}:
        return row.completion
    raise KeyError(f"unknown outcome of the ablation grid: {outcome}")


__all__ = [
    "ABLATION_GRID",
    "ADJUSTMENT_SETS",
    "ESTIMATORS",
    "AblationRow",
    "AblationSpec",
    "ablation_deficit_price",
    "estimator_spread",
    "nuisance_drift",
    "run_ablation",
    "run_ablation_grid",
]
