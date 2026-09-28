"""The honest split of Algorithm 4, step 1.

Ref: Algorithm 4, p. 18 - "split O* into a structural half and an estimation half by
patient". Splitting on the patient keeps a procedure's windows on one side of the split,
so a leaf is never built from the rows it is then scored on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.types import BoolArray, IntArray


@dataclass(frozen=True)
class HonestSplit:
    """Row masks of the structural and estimation halves."""

    structural: BoolArray
    estimation: BoolArray

    @property
    def structural_rows(self) -> int:
        return int(np.count_nonzero(self.structural))

    @property
    def estimation_rows(self) -> int:
        return int(np.count_nonzero(self.estimation))

    def as_dict(self) -> dict[str, float]:
        return {
            "structural_rows": float(self.structural_rows),
            "estimation_rows": float(self.estimation_rows),
        }


def honest_split(patient_id: IntArray, region: BoolArray, seed: int) -> HonestSplit:
    """Assign whole patients to one of the two halves.

    A patient carries all of its windows to one side, which is the unit the algorithm
    names; the halves cannot be separated per row without breaking that.
    """
    if patient_id.shape[0] != region.shape[0]:
        raise ValueError("patient_id and region must be aligned")
    patients = np.unique(patient_id[region])
    if patients.shape[0] < 2:
        raise ValueError("the honest split needs at least two patients in the region")
    generator = np.random.default_rng(seed)
    order = generator.permutation(patients.shape[0])
    chosen_patients = patients[order[: patients.shape[0] // 2]]
    membership = np.isin(patient_id, chosen_patients)
    structural = np.asarray(region & membership, dtype=np.bool_)
    estimation = np.asarray(region & ~membership, dtype=np.bool_)
    if not bool(structural.any()) or not bool(estimation.any()):
        raise ValueError("the honest split produced an empty half")
    return HonestSplit(structural=structural, estimation=estimation)


__all__ = ["HonestSplit", "honest_split"]
