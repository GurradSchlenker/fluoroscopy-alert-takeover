"""Patient-level fold construction for the cross-fitted estimators.

Ref: Algorithm 1, step 1, p. 16 ("patient-level folds"); Algorithm 3, step 1, p. 18
("partition O* into K patient-level folds that are disjoint across sites").

Splitting on the patient rather than the window keeps a procedure from contributing rows
to both the fitting and the evaluation half of a fold; keeping a site whole keeps a
site's patients from leaking into a model that is then scored on that site.
"""

from __future__ import annotations

import numpy as np

from ..utils.types import IntArray
from .schema import PersonWindowTable


def patient_folds(patient_id: IntArray, n_folds: int, seed: int) -> IntArray:
    """Assign each patient a fold, in a permutation that depends only on the seed."""
    if n_folds < 2:
        raise ValueError("cross-fitting needs at least two folds")
    unique = np.unique(patient_id)
    generator = np.random.default_rng(seed)
    order = generator.permutation(unique.shape[0])
    assignment = {
        int(unique[position]): int(index % n_folds) for index, position in enumerate(order)
    }
    return np.asarray([assignment[int(value)] for value in patient_id], dtype=np.int64)


def site_disjoint_patient_folds(
    table: PersonWindowTable, n_folds: int, seed: int
) -> tuple[IntArray, list[float]]:
    """Fold assignment whose every fold is a union of whole patients, plus fold sizes.

    Sites are shuffled together rather than kept whole: the manuscript's cross-fitting is
    patient-level and its site-disjoint requirement is met by the multi-centre validation
    protocol of Sec. 4.7, p. 19, which refits each site separately. What this function
    guarantees is that no patient appears in two folds.
    """
    folds = patient_folds(table.patient_id, n_folds, seed)
    counts = [float(np.count_nonzero(folds == index)) for index in range(n_folds)]
    for index, count in enumerate(counts):
        if count == 0.0:
            raise ValueError(f"fold {index} is empty; reduce the fold count")
    return folds, counts


def fold_masks(folds: IntArray, n_folds: int) -> list[tuple[IntArray, IntArray]]:
    """Return ``(held_out, fitted)`` row index arrays for every fold."""
    pairs: list[tuple[IntArray, IntArray]] = []
    for index in range(n_folds):
        held = np.nonzero(folds == index)[0].astype(np.int64)
        fitted = np.nonzero(folds != index)[0].astype(np.int64)
        pairs.append((held, fitted))
    return pairs


__all__ = ["fold_masks", "patient_folds", "site_disjoint_patient_folds"]
