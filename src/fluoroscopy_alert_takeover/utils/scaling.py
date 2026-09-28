"""Column scaling for the nuisance regressions.

Ref: none - release-internal numerics. The four covariate blocks carry different units and
spreads, so every penalised or tree-based nuisance is fitted on standardised columns whose
centre and scale come from the fitting rows alone; a cross-fitted model must not see the
held-out rows through its scaling.
"""

from __future__ import annotations

import numpy as np

from .types import FloatArray


def fit_scaler(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Column centre and scale of a fitting matrix."""
    if matrix.ndim != 2:
        raise ValueError("the scaling matrix must be two dimensional")
    return (
        np.asarray(matrix.mean(axis=0), dtype=np.float64),
        np.asarray(matrix.std(axis=0), dtype=np.float64),
    )


def standardise_columns(matrix: FloatArray, centre: FloatArray, scale: FloatArray) -> FloatArray:
    """Standardise a matrix against supplied centre and scale, leaving flat columns alone."""
    safe = np.where(scale > 0.0, scale, 1.0)
    return np.asarray((matrix - centre) / safe, dtype=np.float64)


def standardise_pair(design: FloatArray, design_eval: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Standardise a design and an evaluation matrix on the design's own moments."""
    if design.ndim != 2 or design_eval.ndim != 2:
        raise ValueError("both design matrices must be two dimensional")
    centre, scale = fit_scaler(design)
    return standardise_columns(design, centre, scale), standardise_columns(
        design_eval, centre, scale
    )


__all__ = ["fit_scaler", "standardise_columns", "standardise_pair"]
