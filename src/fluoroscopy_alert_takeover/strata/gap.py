"""The pre-specified performance-gap ceiling.

Ref: Table 3, pp. 9-10 - "Maximum subgroup gap (pp) 3.8 (ceiling 10.0) - within ceiling";
Sec. 4.10, p. 21 - "the maximum gap should not exceed 10%".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.types import FloatArray

DEFAULT_CEILING_PP = 10.0


@dataclass(frozen=True)
class SubgroupGap:
    """Range of the subgroup point estimates against the ceiling."""

    ceiling: float
    lowest_label: str
    highest_label: str
    lowest: float
    highest: float

    @property
    def gap(self) -> float:
        return float(self.highest - self.lowest)

    @property
    def within_ceiling(self) -> bool:
        return self.gap <= self.ceiling

    def as_dict(self) -> dict[str, object]:
        return {
            "ceiling": self.ceiling,
            "lowest_label": self.lowest_label,
            "highest_label": self.highest_label,
            "lowest": self.lowest,
            "highest": self.highest,
            "gap": self.gap,
            "within_ceiling": self.within_ceiling,
        }


def maximum_subgroup_gap(
    labels: list[str], estimates: FloatArray, ceiling: float = DEFAULT_CEILING_PP
) -> SubgroupGap:
    """Largest difference between any two subgroup point estimates, in percentage points."""
    values = np.asarray(estimates, dtype=np.float64)
    if values.size != len(labels):
        raise ValueError("labels and estimates must be aligned")
    if values.size < 2:
        raise ValueError("a subgroup gap needs at least two subgroups")
    if ceiling <= 0.0:
        raise ValueError("the ceiling must be positive")
    high = int(np.argmax(values))
    low = int(np.argmin(values))
    return SubgroupGap(
        ceiling=ceiling,
        lowest_label=labels[low],
        highest_label=labels[high],
        lowest=float(values[low]),
        highest=float(values[high]),
    )


def ceiling_report(gap: SubgroupGap) -> dict[str, object]:
    """The ``within ceiling`` verdict of Table 3's footer, as a record."""
    return {"gap": gap.gap, "ceiling": gap.ceiling, "within_ceiling": gap.within_ceiling}


__all__ = ["DEFAULT_CEILING_PP", "SubgroupGap", "ceiling_report", "maximum_subgroup_gap"]
