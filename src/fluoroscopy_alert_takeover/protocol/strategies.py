"""The two pre-specified treatment strategies of the emulated trial.

Ref: Sec. 4.2, p. 14 ("the treatment strategies involve alert-exposed and
alert-unexposed analytical periods that happen in a pre-established time period
determined in advance").

The strategies are analysed as a single-period contrast on the person-window table, so
what has to be frozen is the *edition* of the alert policy: the exposure state of every
window is read from the locked policy and never re-derived after outcomes are opened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Strategy(StrEnum):
    """The two arms of the emulated trial."""

    ALERT_EXPOSED = "alert_exposed"
    ALERT_UNEXPOSED = "alert_unexposed"


@dataclass(frozen=True)
class AlertPolicyEdition:
    """A frozen edition of the navigation alert policy.

    ``edition`` is the lock identifier recorded in the pre-specified analysis plan;
    ``threshold`` is the operating threshold of the policy edition, which Algorithm 5
    selects over the abstention band (p. 20).
    """

    edition: str
    threshold: float
    abstention_low: float = 0.0
    abstention_high: float = 0.0
    policy_sha256: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold outside [0, 1]")
        if self.abstention_low > self.abstention_high:
            raise ValueError("abstention band is inverted")

    @property
    def abstains(self) -> bool:
        """Whether the edition carries a non-empty abstention band."""
        return self.abstention_high > self.abstention_low


@dataclass(frozen=True)
class TargetTrialProtocol:
    """Everything about the trial that must be fixed before outcomes are opened."""

    strategies: tuple[Strategy, Strategy] = (Strategy.ALERT_UNEXPOSED, Strategy.ALERT_EXPOSED)
    time_zero: str = "analysis_period_start"
    follow_up: str = "takeover_or_period_end"
    primary_outcome: str = "takeover"
    secondary_outcome: str = "completion"
    mediator: str = "hand_back"
    policy: AlertPolicyEdition = field(default_factory=lambda: AlertPolicyEdition("frozen", 0.35))
    window_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)

    def arm_label(self, alert: bool) -> Strategy:
        return Strategy.ALERT_EXPOSED if alert else Strategy.ALERT_UNEXPOSED


def strategy_contrast_arm(protocol: TargetTrialProtocol) -> tuple[str, str]:
    """Return the ``(control, treated)`` arm names of the target contrast."""
    control, treated = protocol.strategies
    return control.value, treated.value


__all__ = ["AlertPolicyEdition", "Strategy", "TargetTrialProtocol", "strategy_contrast_arm"]
