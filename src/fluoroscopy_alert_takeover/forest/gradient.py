"""The pre-specified gradient test of the mediated fraction.

Ref: Algorithm 4, step 10, p. 18 - "test the pre-specified gradient of theta_g across
complexity classes"; Table 3, pp. 9-10 - the fraction rises from class I through class III
to the infrapopliteal stratum and the caption states that it "rises monotonically across
the same strata".

The strata are disjoint, so their estimates are combined through their standard errors
rather than through one shared influence vector: the slope's variance is the weighted sum
of the stratum variances.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import stats

_EPS = 1e-12


@dataclass(frozen=True)
class GradientTest:
    """Least-squares trend of an ordered sequence of stratum estimates."""

    labels: list[str]
    points: list[float]
    standard_errors: list[float]
    slope: float
    standard_error: float
    z: float
    p_value_one_sided: float
    monotone: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "points": list(self.points),
            "standard_errors": list(self.standard_errors),
            "slope": self.slope,
            "se": self.standard_error,
            "z": self.z,
            "p_value_one_sided": self.p_value_one_sided,
            "monotone": self.monotone,
        }


def monotone_gradient_test(
    labels: list[str],
    points: list[float],
    standard_errors: list[float],
    spacing: list[float] | None = None,
) -> GradientTest:
    """Slope of the stratum estimates on the order index, with its own standard error.

    ``spacing`` carries the positions of the strata on the axis the gradient is claimed
    over; it defaults to the equally spaced order index, which is what the complexity
    classes are.
    """
    if not (len(labels) == len(points) == len(standard_errors)) or len(points) < 2:
        raise ValueError("a gradient needs at least two labelled strata with standard errors")
    if any(error < 0.0 for error in standard_errors):
        raise ValueError("a standard error cannot be negative")
    steps = np.asarray(
        spacing if spacing is not None else np.arange(len(points), dtype=np.float64),
        dtype=np.float64,
    )
    centred = steps - float(np.mean(steps))
    denominator = float(np.sum(centred**2))
    if denominator <= _EPS:
        raise ValueError("the order index carries no spread")
    weights = centred / denominator
    values = np.asarray(points, dtype=np.float64)
    errors = np.asarray(standard_errors, dtype=np.float64)
    slope = float(np.sum(weights * values))
    variance = float(np.sum((weights**2) * (errors**2)))
    se = float(np.sqrt(variance))
    z = slope / se if se > _EPS else float("nan")
    monotone = bool(all(later >= earlier for earlier, later in itertools.pairwise(points)))
    return GradientTest(
        labels=list(labels),
        points=[float(value) for value in points],
        standard_errors=[float(value) for value in standard_errors],
        slope=slope,
        standard_error=se,
        z=float(z),
        p_value_one_sided=float(stats.norm.sf(z)) if np.isfinite(z) else float("nan"),
        monotone=monotone,
    )


__all__ = ["GradientTest", "monotone_gradient_test"]
