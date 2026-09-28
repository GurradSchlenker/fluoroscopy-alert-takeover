"""The pre-specified success criteria.

Ref: Sec. 4.10, p. 21 - "the main performance criterion decided upon was a technical success
rate of equal to or higher than 88.0%. The effectiveness conditions imposed included that
the risk of failure of the provided treatment should be constituted in a way that the 95%
confidence interval should not extend to the zero level of the positive area, while the
maximum gap should not exceed 10%"; Sec. 2.4, p. 3 - all three criteria met.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..estimators.influence import InfluenceEstimate
from ..utils.types import BoolArray

TECHNICAL_SUCCESS_FLOOR = 0.88
GAP_CEILING_PP = 10.0


def wilson_interval(successes: int, total: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a proportion."""
    if total <= 0:
        raise ValueError("a proportion needs a non-empty denominator")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie inside the denominator")
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    share = successes / total
    denominator = 1.0 + z * z / total
    centre = (share + z * z / (2.0 * total)) / denominator
    spread = (
        z
        * float(np.sqrt(share * (1.0 - share) / total + z * z / (4.0 * total * total)))
        / denominator
    )
    return centre - spread, centre + spread


def proportion_of_flag(flag: BoolArray, level: float = 0.95) -> dict[str, float]:
    """Share and Wilson interval of a binary outcome vector."""
    total = int(flag.shape[0])
    successes = int(np.count_nonzero(flag))
    low, high = wilson_interval(successes, total, level)
    return {"rate": successes / total, "low": low, "high": high, "n": float(total)}


@dataclass(frozen=True)
class Criterion:
    """One pre-specified criterion and its verdict."""

    name: str
    observed: float
    threshold: float
    relation: str
    met: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "observed": self.observed,
            "threshold": self.threshold,
            "relation": self.relation,
            "met": self.met,
        }


def success_criteria(
    technical_success: dict[str, float],
    completion: InfluenceEstimate,
    subgroup_gap: float,
    floor: float = TECHNICAL_SUCCESS_FLOOR,
    ceiling: float = GAP_CEILING_PP,
    level: float = 0.95,
) -> dict[str, object]:
    """Evaluate the three criteria and return them with the joint verdict."""
    low, high = completion.interval(level)
    technical = Criterion(
        name="technical_success_at_or_above_floor",
        observed=technical_success["rate"],
        threshold=floor,
        relation=">=",
        met=bool(technical_success["rate"] >= floor),
    )
    completion_criterion = Criterion(
        name="completion_interval_excludes_null",
        observed=completion.point,
        threshold=0.0,
        relation="ci_low>0",
        met=bool(low > 0.0),
    )
    gap_criterion = Criterion(
        name="subgroup_gap_within_ceiling",
        observed=subgroup_gap,
        threshold=ceiling,
        relation="<=",
        met=bool(subgroup_gap <= ceiling),
    )
    criteria = [technical, completion_criterion, gap_criterion]
    return {
        "criteria": [criterion.as_dict() for criterion in criteria],
        "completion_interval": {"low": low, "high": high},
        "all_met": all(criterion.met for criterion in criteria),
    }


__all__ = [
    "GAP_CEILING_PP",
    "TECHNICAL_SUCCESS_FLOOR",
    "Criterion",
    "proportion_of_flag",
    "success_criteria",
    "wilson_interval",
]
