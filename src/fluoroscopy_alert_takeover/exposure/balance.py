"""Covariate balance: standardised mean differences, weighted and unweighted.

Ref: Table 1, pp. 6-7 - each characteristic carries an absolute standardised mean
difference before and after the certified-overlap restriction and weighting;
Algorithm 1, p. 16 - the exposure-model family is chosen by the maximum absolute
standardised mean difference over the four blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..protocol.schema import PersonWindowTable
from ..utils.types import BoolArray, FloatArray

_EPS = 1e-12


def standardised_mean_difference(
    values: FloatArray, treated: BoolArray, weights: FloatArray | None = None
) -> float:
    """Absolute standardised mean difference between the two alert states.

    The denominator is the root of the pooled within-group variance, each group's variance
    being its own weighted variance, which is the convention the manuscript's ``|SMD|``
    column follows. A column with no spread in either state returns ``0`` when the means
    agree and ``inf`` when they do not.
    """
    if values.shape[0] != treated.shape[0]:
        raise ValueError("values and treated must be aligned")
    if weights is None:
        weights = np.ones(values.shape[0], dtype=np.float64)
    if weights.shape[0] != values.shape[0]:
        raise ValueError("weights must be aligned with values")
    if float(weights.sum()) <= 0.0:
        raise ValueError("weights must carry positive mass")
    exposed_weight = weights[treated]
    control_weight = weights[~treated]
    if float(exposed_weight.sum()) <= 0.0 or float(control_weight.sum()) <= 0.0:
        raise ValueError("both alert states need positive weight inside the region")
    mean_treated = float(np.sum(exposed_weight * values[treated]) / np.sum(exposed_weight))
    mean_control = float(np.sum(control_weight * values[~treated]) / np.sum(control_weight))
    var_treated = float(
        np.sum(exposed_weight * (values[treated] - mean_treated) ** 2) / np.sum(exposed_weight)
    )
    var_control = float(
        np.sum(control_weight * (values[~treated] - mean_control) ** 2) / np.sum(control_weight)
    )
    denominator = np.sqrt(0.5 * (var_treated + var_control))
    if denominator <= _EPS:
        return 0.0 if abs(mean_treated - mean_control) <= _EPS else float("inf")
    return float(abs(mean_treated - mean_control) / denominator)


@dataclass(frozen=True)
class BalanceRow:
    """One characteristic's balance, matching the row Table 1 prints for it."""

    group: str
    column: str
    smd_before: float
    smd_after: float


def balance_rows(
    table: PersonWindowTable, weights: FloatArray | None = None, mask: BoolArray | None = None
) -> list[BalanceRow]:
    """Balance of every characteristic in the four blocks and the case-mix group.

    ``weights`` are the analysis weights (the inverse-propensity weights of the certified
    region); ``mask`` restricts to that region. Both default to the full cohort, which is
    the "before" column.
    """
    selected = np.ones(table.size, dtype=np.bool_) if mask is None else mask
    subset = table.subset(selected)
    unweighted = np.ones(subset.size, dtype=np.float64)
    weighted = unweighted if weights is None else weights[selected]
    rows: list[BalanceRow] = []
    for group, columns in subset.balance_groups().items():
        for column in columns:
            values = subset.covariates[column]
            rows.append(
                BalanceRow(
                    group=group,
                    column=column,
                    smd_before=standardised_mean_difference(values, subset.alert, unweighted),
                    smd_after=standardised_mean_difference(values, subset.alert, weighted),
                )
            )
    return rows


def block_distance(rows: list[BalanceRow], column: str, block_names: tuple[str, ...]) -> float:
    """Maximum balance distance over a named set of blocks.

    Ref: Algorithm 1, steps 3 and 8, p. 16 - the family is selected and the gate compared
    on ``max_B SMD`` and on ``kappa_k``, both of which are a maximum over blocks.
    """
    values = [getattr(row, column) for row in rows if row.group in block_names]
    if not values:
        raise ValueError(f"no balance rows for blocks {block_names}")
    return float(max(values))


def worst_row(rows: list[BalanceRow], column: str) -> BalanceRow:
    """The characteristic with the largest value of ``column``."""
    if not rows:
        raise ValueError("no balance rows to compare")
    return max(rows, key=lambda row: getattr(row, column))


__all__ = [
    "BalanceRow",
    "balance_rows",
    "block_distance",
    "standardised_mean_difference",
    "worst_row",
]
