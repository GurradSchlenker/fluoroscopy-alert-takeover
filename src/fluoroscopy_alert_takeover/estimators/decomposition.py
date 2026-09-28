"""The cross-fitted decomposition of Algorithm 3.

Ref: Algorithm 3, p. 18, steps 1-10. The region's rows are partitioned into patient-level
folds; on each fitting half the exposure propensity, the mediator rule and the four
cell regressions ``E[Y | A = a, M = m, H]`` are refitted; the held-out rows accumulate the
influence-function contribution; the propensity is clipped and the trimmed mass recorded;
and the three arms of Eq. (1) are assembled into ``psi``, ``psi_dir`` and ``psi_med`` with
``theta = psi_med / psi``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..estimands.interventional import (
    PathEstimates,
    augmented_psi_terms,
    decompose,
    mediator_positivity,
    psi_estimate,
)
from ..protocol.folds import fold_masks
from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray, IntArray
from .nuisance import fit_outcome

CELLS: tuple[tuple[int, int], ...] = ((1, 1), (1, 0), (0, 1), (0, 0))


@dataclass(frozen=True)
class MediationNuisance:
    """Out-of-fold mediator rule and cell regressions of the decomposition."""

    propensity: FloatArray
    mediator_exposed: FloatArray
    mediator_control: FloatArray
    cells: dict[str, FloatArray]
    learner: str

    def cell(self, alert: int, mediator: int) -> FloatArray:
        """One ``E[Y | A = alert, M = mediator, H]`` vector."""
        return self.cells[f"{alert}_{mediator}"]

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome_learner": self.learner,
            "mediator_rate_exposed": float(np.mean(self.mediator_exposed)),
            "mediator_rate_control": float(np.mean(self.mediator_control)),
        }


def _fit_cell(
    learner: str,
    design: FloatArray,
    outcome: FloatArray,
    design_eval: FloatArray,
    fallback: float,
) -> FloatArray:
    """Fit one regression, falling back to the arm mean when the cell cannot support one."""
    if design.shape[0] < 2 or np.unique(outcome).shape[0] < 2:
        return np.full(design_eval.shape[0], fallback, dtype=np.float64)
    return fit_outcome(learner, design, outcome, design_eval)


def cross_fitted_mediation_nuisance(
    table: PersonWindowTable,
    propensity: FloatArray,
    folds: IntArray,
    outcome_name: str,
    learner: str,
) -> MediationNuisance:
    """Cross-fit the mediator rule and the four cell regressions of Eq. (1)."""
    design = table.full_design()
    alert = table.alert.astype(np.int64)
    mediator = table.hand_back.astype(np.int64)
    outcome = table.outcome(outcome_name)
    n_folds = int(np.unique(folds).shape[0])

    mediator_exposed = np.full(table.size, np.nan, dtype=np.float64)
    mediator_control = np.full(table.size, np.nan, dtype=np.float64)
    cells = {
        f"{arm}_{value}": np.full(table.size, np.nan, dtype=np.float64) for arm, value in CELLS
    }

    for held_out, fitted in fold_masks(folds, n_folds):
        for arm in (1, 0):
            arm_rows = fitted[alert[fitted] == arm]
            if arm_rows.size == 0:
                raise ValueError("a fold left one alert arm empty; reduce the fold count")
            arm_mean = float(np.mean(mediator[arm_rows]))
            rule = _fit_cell(
                learner,
                design[arm_rows],
                mediator[arm_rows].astype(np.float64),
                design[held_out],
                arm_mean,
            )
            if arm == 1:
                mediator_exposed[held_out] = rule
            else:
                mediator_control[held_out] = rule
        for arm, value in CELLS:
            arm_rows = fitted[alert[fitted] == arm]
            cell_rows = fitted[(alert[fitted] == arm) & (mediator[fitted] == value)]
            fallback = float(np.mean(outcome[arm_rows])) if arm_rows.size > 0 else float("nan")
            if not np.isfinite(fallback):
                raise ValueError("a fold left an arm empty; reduce the fold count")
            cells[f"{arm}_{value}"][held_out] = _fit_cell(
                learner, design[cell_rows], outcome[cell_rows], design[held_out], fallback
            )

    for vector in (mediator_exposed, mediator_control, *cells.values()):
        if not np.isfinite(vector).all():
            raise RuntimeError("cross-fitting left rows unscored")
    return MediationNuisance(
        propensity=np.asarray(propensity, dtype=np.float64),
        mediator_exposed=mediator_exposed,
        mediator_control=mediator_control,
        cells=cells,
        learner=learner,
    )


def cross_fitted_decomposition(
    table: PersonWindowTable,
    nuisance: MediationNuisance,
    mask: BoolArray,
    outcome_name: str,
) -> tuple[PathEstimates, dict[str, float]]:
    """Assemble the three arms of Eq. (1) and the fraction on the certified region."""
    alert = table.alert.astype(np.float64)
    mediator = table.hand_back.astype(np.float64)
    outcome = table.outcome(outcome_name)
    propensity = nuisance.propensity

    observed = augmented_psi_terms(
        1,
        alert,
        outcome,
        mediator,
        propensity,
        nuisance.cell(1, 0),
        nuisance.cell(1, 1),
        nuisance.mediator_exposed,
        nuisance.mediator_exposed,
    )
    crossed = augmented_psi_terms(
        1,
        alert,
        outcome,
        mediator,
        propensity,
        nuisance.cell(1, 0),
        nuisance.cell(1, 1),
        nuisance.mediator_control,
        nuisance.mediator_exposed,
    )
    control = augmented_psi_terms(
        0,
        alert,
        outcome,
        mediator,
        propensity,
        nuisance.cell(0, 0),
        nuisance.cell(0, 1),
        nuisance.mediator_control,
        nuisance.mediator_control,
    )
    paths = decompose(
        arm_observed=psi_estimate(observed, mask),
        arm_crossed=psi_estimate(crossed, mask),
        arm_control=psi_estimate(control, mask),
    )
    diagnostics = mediator_positivity(nuisance.mediator_exposed, mask)
    return paths, diagnostics


__all__ = [
    "CELLS",
    "MediationNuisance",
    "cross_fitted_decomposition",
    "cross_fitted_mediation_nuisance",
]
