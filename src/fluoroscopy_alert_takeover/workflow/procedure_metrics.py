"""Procedure-level workflow quantities.

Ref: Sec. 2.7, p. 5 - "The outcomes of the deployment were ambivalent, and the two measures
relating to radiation were defined in advance in accordance with the study design as those
measurements that were expected to show no impact thereafter"; the four printed quantities
are fluoroscopy time, contrast volume, procedure duration and the integration cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..estimators.influence import InfluenceEstimate
from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray

METRICS: tuple[str, ...] = (
    "fluoroscopy_time_min",
    "contrast_volume_ml",
    "procedure_duration_min",
    "integration_cost_s",
)


@dataclass(frozen=True)
class MetricSummary:
    """Mean and standard deviation of one workflow quantity on a region."""

    metric: str
    mean: float
    sd: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "sd": self.sd, "n": float(self.n)}


def describe(table: PersonWindowTable, metric: str, mask: BoolArray | None = None) -> MetricSummary:
    """Mean and standard deviation of one quantity, at procedure level.

    The printed values are procedure-level, so windows of one procedure are collapsed to
    that procedure's single value before the summary is taken.
    """
    if metric not in table.workflow:
        raise KeyError(f"workflow quantity not in the table: {metric}")
    selected = np.ones(table.size, dtype=np.bool_) if mask is None else mask
    values = table.workflow[metric][selected]
    procedures = table.procedure_id[selected]
    if values.size == 0:
        raise ValueError("the region is empty")
    _, first = np.unique(procedures, return_index=True)
    collapsed = values[np.sort(first)]
    return MetricSummary(
        metric=metric,
        mean=float(np.mean(collapsed)),
        sd=float(np.std(collapsed, ddof=1)),
        n=int(collapsed.size),
    )


def workflow_contrast(
    name: str,
    quantity: FloatArray,
    alert: FloatArray,
    propensity: FloatArray,
    mu_treated: FloatArray,
    mu_control: FloatArray,
    mask: BoolArray,
) -> InfluenceEstimate:
    """Doubly-robust difference of a continuous workflow quantity.

    The same one-step form as Eq. (2) applies to a continuous outcome, so the two
    radiation measures that the protocol pre-declared as null are estimated with the same
    estimator as the primary contrast rather than with a separate test.
    """
    safe_exposed = np.where(propensity < 1e-9, 1e-9, propensity)
    safe_control = np.where((1.0 - propensity) < 1e-9, 1e-9, 1.0 - propensity)
    terms = (
        alert * (quantity - mu_treated) / safe_exposed
        - (1.0 - alert) * (quantity - mu_control) / safe_control
        + mu_treated
        - mu_control
    )
    region = terms[mask]
    if region.size == 0:
        raise ValueError("the certified region is empty")
    point = float(np.mean(region))
    return InfluenceEstimate(
        name=name, point=point, influence=np.asarray(region - point, dtype=np.float64)
    )


def unadjusted_difference(
    quantity: FloatArray, alert: BoolArray, mask: BoolArray
) -> dict[str, float]:
    """Crude between-group difference, reported beside the adjusted one."""
    exposed = quantity[mask & alert]
    control = quantity[mask & ~alert]
    if exposed.size == 0 or control.size == 0:
        raise ValueError("both arms are needed for a difference")
    return {
        "exposed_mean": float(np.mean(exposed)),
        "control_mean": float(np.mean(control)),
        "difference": float(np.mean(exposed) - np.mean(control)),
    }


def integration_cost(table: PersonWindowTable) -> dict[str, float]:
    """Median integration cost in seconds, and the share of procedures that paid it.

    Ref: Sec. 2.7, p. 5 - "median integration has included 4.2 s into the process
    technology"; "Based on the breakdown of time motion, it can be inferred that even
    though the cost of integration was absorbed by the stages of setting up and docking,
    the process of motion of the apparatus did not change".
    """
    if "integration_cost_s" not in table.workflow:
        raise KeyError("the table carries no integration cost")
    costs = table.workflow["integration_cost_s"]
    procedures = table.procedure_id
    _, first = np.unique(procedures, return_index=True)
    collapsed = costs[np.sort(first)]
    return {
        "median_s": float(np.median(collapsed)),
        "share_with_cost": float(np.mean(collapsed > 0.0)),
        "n_procedures": float(collapsed.size),
    }


__all__ = [
    "METRICS",
    "MetricSummary",
    "describe",
    "integration_cost",
    "unadjusted_difference",
    "workflow_contrast",
]
