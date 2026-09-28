"""The four identification assumptions and one probe for each.

Ref: Sec. 4.3, p. 15 ("it is necessary that four assumptions be made ... stating the
assumptions along with the analysis required for each assumption probing it"); the
consistency probe is "the percentage of windows having finally identifiable exposure
status in the frozen policy" and the positivity probe is the overlapping gate of
Algorithm 1.

Exchangeability and no-interference are probed through the falsification panel
(``falsification.protocol``), so this module returns the two probes that are computed
directly on the table and re-exports the pairing for the other two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.types import BoolArray, FloatArray
from .schema import PersonWindowTable


@dataclass(frozen=True)
class ProbeResult:
    """One probe, its pre-specified verdict rule and the observed value."""

    assumption: str
    probe: str
    value: float
    threshold: float
    verdict: str

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def identifiable_exposure_state(table: PersonWindowTable) -> BoolArray:
    """Windows whose exposure state the frozen policy can adjudicate.

    A window is adjudicable when its linkage is complete and, if it is exposed, an alert
    onset was logged; an exposed window with no onset is an inconsistency between the
    policy edition and the log rather than a window with a hidden score.
    """
    adjudicable = table.linkage_complete.copy()
    exposed_without_onset = table.alert & ~np.isfinite(table.alert_onset_s)
    return adjudicable & ~exposed_without_onset


def consistency_probe(table: PersonWindowTable, threshold: float) -> ProbeResult:
    """Share of windows with an identifiable exposure state in the frozen policy."""
    if table.size == 0:
        raise ValueError("consistency probe needs a non-empty table")
    share = float(np.mean(identifiable_exposure_state(table)))
    return ProbeResult(
        assumption="consistency",
        probe="identifiable_exposure_state_share",
        value=share,
        threshold=threshold,
        verdict="PASS" if share >= threshold else "FAIL",
    )


def positivity_probe(propensity: FloatArray, epsilon: float, min_ess: float) -> ProbeResult:
    """Overlap of the exposure model at the frozen gate.

    The probe reports the effective sample size divided by the sample size at the gate,
    which is the quantity the gate of Algorithm 1 tracks through the sweep.
    """
    clipped = np.clip(propensity, epsilon, 1.0 - epsilon)
    share = float((clipped * 2.0 * (1.0 - clipped)).mean())
    return ProbeResult(
        assumption="positivity",
        probe="gate_normalised_effective_sample_size",
        value=share,
        threshold=min_ess,
        verdict="PASS" if share >= min_ess else "FAIL",
    )


def exchangeability_probe(negative_control_p_values: FloatArray, alpha: float) -> ProbeResult:
    """Falsification pairing on the outcome side.

    The verdict is the share of negative-control outcomes whose interval covers the null;
    the manuscript requires all five to cover it (Table 4 Panel A, p. 11).
    """
    if negative_control_p_values.size == 0:
        raise ValueError("exchangeability probe needs at least one negative control")
    covered = float(np.mean(negative_control_p_values > alpha))
    return ProbeResult(
        assumption="exchangeability",
        probe="negative_control_null_coverage",
        value=covered,
        threshold=1.0,
        verdict="PASS" if covered >= 1.0 else "FAIL",
    )


__all__ = [
    "ProbeResult",
    "consistency_probe",
    "exchangeability_probe",
    "identifiable_exposure_state",
    "positivity_probe",
]
