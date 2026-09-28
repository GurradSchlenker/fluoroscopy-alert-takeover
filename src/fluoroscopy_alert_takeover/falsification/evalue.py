"""E-values for the primary contrasts.

Ref: Algorithm 2, step 6, p. 16 - "compute E from the primary contrast and ECI at the
confidence-interval limit"; Sec. 2.3, p. 3 - "The E-value corresponding to the estimate,
known as 2.06, could be interpretable as the lower limit of confidence interval for 1.62";
Table 4 Panel C, p. 11.

The E-value is defined on the risk-ratio scale, so the risk difference is converted with
the observed outcome rate in the unexposed region. The manuscript's printed value is not
reproduced by that conversion; the verifier records the gap as a manuscript-arithmetic
finding rather than adjusting either number.
"""

from __future__ import annotations

import numpy as np

from ..utils.types import FloatArray

_EPS = 1e-12


def risk_ratio_from_difference(risk_difference: float, unexposed_risk: float) -> float:
    """Risk ratio implied by a risk difference at a given unexposed risk."""
    if not 0.0 < unexposed_risk < 1.0:
        raise ValueError("the unexposed risk must lie strictly inside (0, 1)")
    exposed_risk = unexposed_risk + risk_difference
    if exposed_risk <= 0.0:
        return 0.0
    return float(exposed_risk / unexposed_risk)


def e_value_from_risk_ratio(risk_ratio: float) -> float:
    """VanderWeele-Ding E-value for a risk ratio at or above one.

    A risk ratio below one is inverted before the formula is applied, which is the
    standard treatment for a protective contrast.
    """
    if risk_ratio <= 0.0:
        return 1.0
    value = risk_ratio if risk_ratio >= 1.0 else 1.0 / risk_ratio
    return float(value + np.sqrt(value * (value - 1.0)))


def e_value_risk_difference(risk_difference: float, unexposed_risk: float) -> float:
    """E-value of a risk difference."""
    return e_value_from_risk_ratio(risk_ratio_from_difference(risk_difference, unexposed_risk))


def e_value_ci_limit(
    risk_difference: float, interval: tuple[float, float], unexposed_risk: float
) -> float:
    """E-value at the confidence limit closest to the null.

    Ref: Algorithm 2, step 6. The limit closest to the null is the one that requires the
    weakest unmeasured confounder, so it is the conservative choice.
    """
    low, high = interval
    if low <= 0.0 <= high:
        return 1.0
    limit = low if abs(low) < abs(high) else high
    del risk_difference
    return e_value_risk_difference(limit, unexposed_risk)


def unexposed_risk(table_alert: FloatArray, table_outcome: FloatArray, mask: FloatArray) -> float:
    """Observed outcome rate among the unexposed rows of a region."""
    selected = (mask > 0.5) & (table_alert <= 0.5)
    if not bool(selected.any()):
        raise ValueError("the region has no unexposed rows")
    return float(np.mean(table_outcome[selected]))


__all__ = [
    "e_value_ci_limit",
    "e_value_from_risk_ratio",
    "e_value_risk_difference",
    "risk_ratio_from_difference",
    "unexposed_risk",
]
