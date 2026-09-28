"""The three-arm observational reader comparison.

Ref: Sec. 2.8, pp. 5-6 and Table 1, p. 7 - sensitivity 0.612 / 0.847 / 0.839 and
specificity 0.923 / 0.881 / 0.902 for clinician alone, alert alone and the team; the two
relative risks "Team vs alert alone, relative risk 0.99 (95% CI 0.96 to 1.02)" and
"Team vs clinician alone, relative risk 1.37 (95% CI 1.29 to 1.45)".

The comparison is observational, so the arms are described and never merged: the release
returns the arm-level operating points and the pairwise relative risks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from ..cohort.tables import read_csv_rows

ARM_NAMES: tuple[str, ...] = ("clinician_alone", "alert_alone", "team")


@dataclass(frozen=True)
class ArmCounts:
    """One reader arm as a two-by-two table against the reference standard."""

    name: str
    positive_reference: int
    negative_reference: int
    positive_calls: int
    true_positive: int
    true_negative: int

    def __post_init__(self) -> None:
        if self.positive_reference <= 0 or self.negative_reference <= 0:
            raise ValueError("both reference groups are needed")
        if not 0 <= self.true_positive <= self.positive_reference:
            raise ValueError("true positives outside the reference group")
        if not 0 <= self.true_negative <= self.negative_reference:
            raise ValueError("true negatives outside the reference group")

    @property
    def sensitivity(self) -> float:
        return self.true_positive / self.positive_reference

    @property
    def specificity(self) -> float:
        return self.true_negative / self.negative_reference

    def as_dict(self) -> dict[str, float]:
        return {
            "positive_reference": float(self.positive_reference),
            "negative_reference": float(self.negative_reference),
            "positive_calls": float(self.positive_calls),
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
        }


def relative_risk(
    numerator: ArmCounts, denominator: ArmCounts, level: float = 0.95
) -> dict[str, float]:
    """Ratio of the two arms' sensitivities with a log-scale interval.

    A ratio of proportions is the quantity Table 1 reports, and the log-scale interval is
    the standard one for it.
    """
    p_numerator = numerator.sensitivity
    p_denominator = denominator.sensitivity
    if p_denominator <= 0.0:
        raise ValueError("the denominator arm has no sensitivity to compare against")
    ratio = p_numerator / p_denominator
    var_ratio = ((1.0 - p_numerator) / (p_numerator * numerator.positive_reference)) + (
        (1.0 - p_denominator) / (p_denominator * denominator.positive_reference)
    )
    spread = float(np.sqrt(max(var_ratio, 0.0)))
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    return {
        "ratio": float(ratio),
        "se_log": spread,
        "ci_low": float(ratio * np.exp(-z * spread)),
        "ci_high": float(ratio * np.exp(z * spread)),
    }


def reader_comparison(arms: list[ArmCounts]) -> dict[str, object]:
    """Arm-level operating points plus every pairwise relative risk among the arms."""
    if len(arms) < 2:
        raise ValueError("the comparison needs at least two arms")
    by_name = {arm.name: arm for arm in arms}
    risks: dict[str, dict[str, float]] = {}
    for name, arm in by_name.items():
        for other_name, other in by_name.items():
            if name == other_name:
                continue
            if name != "team":
                # Table 1 reports the team against each single arm, not the reverse.
                continue
            risks[f"team_vs_{other_name}"] = relative_risk(arm, other)
    return {
        "arms": {arm.name: arm.as_dict() for arm in arms},
        "relative_risk": risks,
    }


def load_reader_arms(path: str | Path) -> list[ArmCounts]:
    """Read the reader-comparison layer of the cohort.

    Ref: Sec. 2.8, p. 5 - the comparison is a separate observational layer over the same
    procedures, so its counts are stored beside the person-window table rather than inside
    it.
    """
    arms: list[ArmCounts] = []
    for row in read_csv_rows(path):
        arms.append(
            ArmCounts(
                name=row["arm"].strip(),
                positive_reference=int(float(row["positive_reference"])),
                negative_reference=int(float(row["negative_reference"])),
                positive_calls=int(float(row["positive_calls"])),
                true_positive=int(float(row["true_positive"])),
                true_negative=int(float(row["true_negative"])),
            )
        )
    if not arms:
        raise ValueError(f"the reader layer is empty: {path}")
    return arms


def sensitivity_from_estimate(point: float, reference: int) -> float:
    """Sensitivity implied by a printed operating point and a reference size."""
    if not 0.0 <= point <= 1.0:
        raise ValueError("a sensitivity must lie in [0, 1]")
    if reference <= 0:
        raise ValueError("the reference group must be non-empty")
    return float(point * reference)


__all__ = [
    "ARM_NAMES",
    "ArmCounts",
    "load_reader_arms",
    "reader_comparison",
    "relative_risk",
    "sensitivity_from_estimate",
]
