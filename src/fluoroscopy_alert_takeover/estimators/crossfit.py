"""Cross-fitting plumbing: trimming and the assembled nuisance pair.

Ref: Algorithm 3, steps 1-6, p. 18. Step 1 builds patient-level folds, steps 3-4 fit and
accumulate out of fold, and step 6 clips the propensity and records the trimmed mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray, IntArray
from .nuisance import CrossFittedNuisance, cross_fitted_outcome, in_sample_outcome

_EPS = 1e-6


@dataclass(frozen=True)
class TrimResult:
    """The clipped propensity and the mass the clip removed."""

    propensity: FloatArray
    gamma: float
    trimmed_mass: float

    def as_dict(self) -> dict[str, float]:
        return {"gamma": self.gamma, "trimmed_mass": self.trimmed_mass}


def clip_propensity(propensity: FloatArray, gamma: float) -> TrimResult:
    """Clip ``pi`` to ``[gamma, 1 - gamma]`` and record the trimmed mass.

    Ref: Algorithm 3, step 6, p. 18. The trimmed mass is the share of rows whose
    propensity moved, which is the quantity that the positivity discussion in Sec. 2.3
    (p. 3) reports as a percentage of the analytical windows.
    """
    if not 0.0 < gamma < 0.5:
        raise ValueError("the trimming bound must lie in (0, 0.5)")
    moved = float(np.mean((propensity < gamma) | (propensity > 1.0 - gamma)))
    clipped = np.asarray(np.clip(propensity, gamma, 1.0 - gamma), dtype=np.float64)
    return TrimResult(propensity=clipped, gamma=gamma, trimmed_mass=moved)


def table_design(table: PersonWindowTable, blocks: tuple[str, ...] | None) -> FloatArray:
    """Design matrix of a table for one adjustment set."""
    return table.full_design(blocks)


def cross_fitted_nuisance(
    design: FloatArray,
    alert: IntArray,
    outcome: FloatArray,
    propensity: FloatArray,
    folds: IntArray,
    learner: str,
    cross_fit: bool = True,
) -> CrossFittedNuisance:
    """Assemble the propensity and the two arm regressions of Eq. (2).

    The regressions are fitted on the folds the row is not in; ``cross_fit=False`` fits
    each arm regression and scores the same rows, which is the no-cross-fitting row of the
    Table 2 ablation.
    """
    if not (design.shape[0] == alert.shape[0] == outcome.shape[0] == propensity.shape[0]):
        raise ValueError("design, exposure, outcome and propensity must be aligned")
    if cross_fit:
        mu_treated, mu_control = cross_fitted_outcome(design, alert, outcome, folds, learner)
    else:
        mu_treated, mu_control = in_sample_outcome(design, alert, outcome, learner)
    return CrossFittedNuisance(
        propensity=np.asarray(np.clip(propensity, _EPS, 1.0 - _EPS), dtype=np.float64),
        mu_treated=mu_treated,
        mu_control=mu_control,
        folds=folds,
        learner=learner,
    )


def region_indices(mask: BoolArray) -> IntArray:
    """Row indices of a region."""
    return np.nonzero(mask)[0].astype(np.int64)


def region_design(design: FloatArray, mask: BoolArray) -> FloatArray:
    """Design matrix restricted to a region."""
    return np.asarray(design[mask], dtype=np.float64)


__all__ = [
    "TrimResult",
    "clip_propensity",
    "cross_fitted_nuisance",
    "region_design",
    "region_indices",
    "table_design",
]
