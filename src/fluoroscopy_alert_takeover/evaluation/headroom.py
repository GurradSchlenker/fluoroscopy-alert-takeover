"""The achievable ceiling of the alert model.

Ref: Table 1, p. 7 - the row "Achievable ceiling (headroom analysis) 0.964", printed
without a definition.

The manuscript never says what the ceiling measures, so the reading below is a release
engineering default and is recorded as such: the ceiling is the AUROC the same policy would
reach if the spatial predicate were read off the reference channel rather than predicted,
which bounds what a better predictor of the same target could add.
"""

from __future__ import annotations

from ..utils.types import BoolArray, FloatArray
from .discrimination import auroc


def achievable_ceiling(oracle_scores: FloatArray, labels: BoolArray, floor: float = 0.5) -> float:
    """AUROC of the reference channel, clipped below the chance floor.

    ``oracle_scores`` is the reference-channel reading of the same target the policy
    predicts. A ceiling below chance means the reference channel is misaligned with the
    label, which is reported as-is rather than silently flipped.
    """
    value = auroc(oracle_scores, labels)
    return float(max(value, floor))


def headroom(ceiling: float, achieved: float) -> dict[str, float]:
    """Distance between the achieved discrimination and the ceiling."""
    if not 0.0 <= achieved <= 1.0 or not 0.0 <= ceiling <= 1.0:
        raise ValueError("both quantities must be areas under a curve")
    if ceiling < achieved:
        raise ValueError("the ceiling cannot lie below the achieved discrimination")
    return {"ceiling": ceiling, "achieved": achieved, "headroom": ceiling - achieved}


__all__ = ["achievable_ceiling", "headroom"]
