"""The certified-overlap gate.

Ref: Algorithm 1, pp. 16 - steps 5-13 of the box, verbatim:

    for k = 1 to G:  O_k = {(i,w) : eps_k <= pi <= 1 - eps_k}; e_k = (sum pi)^2 / sum pi^2;
                     kappa_k = normalised distance between the alert-state covariate
                     marginals on O_k
    if max_k kappa_k <= kappa*: eps* = max{eps_k : kappa_k <= kappa*}
    else: eps* = arg max_k kappa_k and flag the deficit
    O* = {(i,w) : eps* <= pi <= 1 - eps*}; price = contrast on O* - contrast on ungated set

The manuscript prints the shape of the gate grid and the existence of the tolerance but
not their values; both are engineering defaults here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exposure.balance import BalanceRow, balance_rows
from ..protocol.schema import PersonWindowTable
from ..utils.logging import get_logger
from ..utils.types import BoolArray, FloatArray
from .effective_sample_size import effective_sample_size

_LOG = get_logger("overlap.gate")

# Engineering default: 25 gates on [0.01, 0.25], and the conventional 0.10 absolute
# standardised mean difference as the balance tolerance kappa*.
DEFAULT_GRID: tuple[float, ...] = tuple(np.round(np.linspace(0.01, 0.25, 25), 4))
DEFAULT_TOLERANCE = 0.10


@dataclass(frozen=True)
class GateSweep:
    """The sweep of the gate grid: retained mass, effective sample size and distance."""

    epsilon: FloatArray
    retained: FloatArray
    effective_sample_size: FloatArray
    kappa: FloatArray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "epsilon": [float(value) for value in self.epsilon],
            "retained": [float(value) for value in self.retained],
            "effective_sample_size": [float(value) for value in self.effective_sample_size],
            "kappa": [float(value) for value in self.kappa],
        }


@dataclass(frozen=True)
class CertifiedGate:
    """The frozen gate, its region and the record of the sweep that selected it."""

    epsilon: float
    mask: BoolArray
    sweep: GateSweep
    tolerance: float
    deficit_flagged: bool

    @property
    def retained(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def retained_share(self) -> float:
        return float(np.mean(self.mask))

    def as_dict(self) -> dict[str, object]:
        return {
            "epsilon": self.epsilon,
            "tolerance": self.tolerance,
            "deficit_flagged": self.deficit_flagged,
            "retained": self.retained,
            "retained_share": self.retained_share,
            "sweep": self.sweep.as_dict(),
        }


def gate_mask(propensity: FloatArray, epsilon: float) -> BoolArray:
    """The region ``{eps <= pi <= 1 - eps}``."""
    if not 0.0 <= epsilon < 0.5:
        raise ValueError("epsilon must lie in [0, 0.5)")
    return np.asarray((propensity >= epsilon) & (propensity <= 1.0 - epsilon), dtype=np.bool_)


def covariate_marginal_distance(
    table: PersonWindowTable,
    propensity: FloatArray,
    mask: BoolArray,
    block_names: tuple[str, ...],
) -> float:
    """``kappa_k``: the largest alert-state covariate marginal distance on the region.

    The distance is the maximum absolute standardised mean difference over the covariates
    of the named blocks, computed with the inverse-propensity weighting that the region
    itself induces, so a region that is balanced only because it is small is not rewarded.
    """
    if not bool(mask.any()):
        return float("inf")
    exposed = int(np.count_nonzero(table.alert[mask]))
    control = int(np.count_nonzero(~table.alert[mask]))
    if exposed == 0 or control == 0:
        return float("inf")
    weights = np.zeros(table.size, dtype=np.float64)
    clipped = np.clip(propensity[mask], 1e-6, 1.0 - 1e-6)
    weights[mask] = np.where(table.alert[mask], 1.0 / clipped, 1.0 / (1.0 - clipped))
    rows: list[BalanceRow] = balance_rows(table, weights=weights, mask=mask)
    selected = [row for row in rows if row.group in block_names]
    if not selected:
        raise ValueError(f"no covariates in the requested blocks: {block_names}")
    return float(max(row.smd_after for row in selected))


def sweep_gate(
    table: PersonWindowTable,
    propensity: FloatArray,
    grid: tuple[float, ...] = DEFAULT_GRID,
    block_names: tuple[str, ...] = ("imaging", "kinematic", "workflow", "operator"),
) -> GateSweep:
    """Evaluate the gate grid, returning the trajectory Algorithm 1 tracks."""
    if sorted(grid) != list(grid):
        raise ValueError("the gate grid must be increasing")
    retained: list[float] = []
    sizes: list[float] = []
    kappas: list[float] = []
    for epsilon in grid:
        mask = gate_mask(propensity, epsilon)
        retained.append(float(np.mean(mask)))
        sizes.append(effective_sample_size(propensity, mask))
        kappas.append(covariate_marginal_distance(table, propensity, mask, block_names))
    return GateSweep(
        epsilon=np.asarray(grid, dtype=np.float64),
        retained=np.asarray(retained, dtype=np.float64),
        effective_sample_size=np.asarray(sizes, dtype=np.float64),
        kappa=np.asarray(kappas, dtype=np.float64),
    )


def build_certified_overlap_gate(
    table: PersonWindowTable,
    propensity: FloatArray,
    grid: tuple[float, ...] = DEFAULT_GRID,
    tolerance: float = DEFAULT_TOLERANCE,
    block_names: tuple[str, ...] = ("imaging", "kinematic", "workflow", "operator"),
) -> CertifiedGate:
    """Build ``O*``, freeze ``eps*`` and flag a positivity deficit when balance fails."""
    sweep = sweep_gate(table, propensity, grid, block_names)
    acceptable = sweep.kappa <= tolerance
    if bool(acceptable.any()):
        # step 10, first branch: the strictest gate that still balances.
        epsilon = float(sweep.epsilon[acceptable].max())
        deficit = False
    else:
        # step 10, second branch: no gate balances, so take the least unbalanced one and
        # flag the deficit rather than reporting a region the design cannot certify.
        epsilon = float(sweep.epsilon[int(np.argmax(sweep.kappa))])
        deficit = True
        _LOG.warning(
            "no gate satisfied the balance tolerance %.3f; deficit flagged at eps=%.4f",
            tolerance,
            epsilon,
        )
    return CertifiedGate(
        epsilon=epsilon,
        mask=gate_mask(propensity, epsilon),
        sweep=sweep,
        tolerance=tolerance,
        deficit_flagged=deficit,
    )


def positivity_deficit_price(gated: float, ungated: float) -> float:
    """``price`` of Algorithm 1, step 12: the contrast lost to the certified region."""
    return float(gated - ungated)


def boundary_mass(propensity: FloatArray, epsilon: float) -> float:
    """Share of rows the gate removes, reported with the frozen gate."""
    return float(1.0 - np.mean(gate_mask(propensity, epsilon)))


__all__ = [
    "DEFAULT_GRID",
    "DEFAULT_TOLERANCE",
    "CertifiedGate",
    "GateSweep",
    "boundary_mass",
    "build_certified_overlap_gate",
    "covariate_marginal_distance",
    "gate_mask",
    "positivity_deficit_price",
    "sweep_gate",
]
