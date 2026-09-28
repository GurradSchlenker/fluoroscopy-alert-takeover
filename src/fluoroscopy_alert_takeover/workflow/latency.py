"""Stage latency and throughput accounting.

Ref: Table A1, p. 25 - per-stage median latency with its interquartile range, throughput
and memory, the observed alert-to-action latency that defines the intra-procedural budget,
and the end-to-end path's share of that budget; and the offline analysis figure of 18.4 min
per 10,000 procedures on 8 CPU cores.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..utils.types import FloatArray

STAGES: tuple[str, ...] = (
    "frame_ingest",
    "spatial_predicate_primary",
    "spatial_predicate_state_space",
    "alert_state_construction",
    "propensity_and_mediator_scoring",
    "end_to_end_alert_path",
)


@dataclass(frozen=True)
class StageLatency:
    """Wall-clock record of one pipeline stage."""

    stage: str
    milliseconds: FloatArray
    memory_gb: float

    @property
    def median(self) -> float:
        return float(np.median(self.milliseconds))

    @property
    def iqr(self) -> tuple[float, float]:
        return float(np.quantile(self.milliseconds, 0.25)), float(
            np.quantile(self.milliseconds, 0.75)
        )

    @property
    def throughput(self) -> float:
        """Windows per second implied by the median latency."""
        if self.median <= 0.0:
            return float("inf")
        return 1000.0 / self.median

    def as_dict(self) -> dict[str, float]:
        low, high = self.iqr
        return {
            "median_ms": self.median,
            "iqr_low_ms": low,
            "iqr_high_ms": high,
            "throughput_windows_per_s": self.throughput,
            "memory_gb": self.memory_gb,
        }


@dataclass
class StageLedger:
    """Collects raw timings while the analysis runs."""

    samples: dict[str, list[float]] = field(default_factory=dict)
    memory: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, seconds: float, memory_gb: float = 0.0) -> None:
        self.samples.setdefault(stage, []).append(float(seconds) * 1000.0)
        self.memory[stage] = max(self.memory.get(stage, 0.0), float(memory_gb))

    def latencies(self) -> list[StageLatency]:
        return [
            StageLatency(
                stage=stage,
                milliseconds=np.asarray(values, dtype=np.float64),
                memory_gb=self.memory.get(stage, 0.0),
            )
            for stage, values in self.samples.items()
        ]

    def total_median_seconds(self, stages: tuple[str, ...]) -> float:
        """Sum of the median latencies of the named stages, in seconds."""
        total = 0.0
        for stage in stages:
            values = self.samples.get(stage)
            if values:
                total += float(np.median(values)) / 1000.0
        return total


def timed(stage: str, ledger: StageLedger, call: Callable[[], float]) -> float:
    """Run ``call`` once, record its wall clock under ``stage`` and return its value."""
    started = time.perf_counter()
    value = call()
    ledger.record(stage, time.perf_counter() - started)
    return value


def observed_action_budget(samples_ms: FloatArray) -> dict[str, float]:
    """The alert-to-action latency that defines the deployable budget.

    Ref: Table A1, p. 25 - "Observed alert-to-action latency (definition of the budget)
    Median 2,400 ms, IQR 1,600-3,800 ms".
    """
    if samples_ms.size == 0:
        raise ValueError("the budget needs at least one observed latency")
    return {
        "median_ms": float(np.median(samples_ms)),
        "iqr_low_ms": float(np.quantile(samples_ms, 0.25)),
        "iqr_high_ms": float(np.quantile(samples_ms, 0.75)),
        "n": float(samples_ms.size),
    }


def budget_fraction(end_to_end_ms: float, budget_median_ms: float) -> float:
    """Share of the intra-procedural budget the end-to-end path consumes."""
    if budget_median_ms <= 0.0:
        raise ValueError("the budget must be positive")
    return float(end_to_end_ms / budget_median_ms)


def offline_analysis_rate(
    seconds: float, procedures: int, cores: int, reference_cores: int = 8
) -> float:
    """Seconds per 10,000 procedures, rescaled to the reported core count.

    Ref: Table A1, p. 25 - "Offline analysis: causal estimation 18.4 min per 10,000
    procedures on 8 CPU cores". The figure below rescales a measured run to those cores so
    the two are comparable; it is an accounting identity, not a prediction.
    """
    if procedures <= 0 or cores <= 0:
        raise ValueError("procedure count and core count must be positive")
    per_procedure = seconds / float(procedures)
    scaled = per_procedure * (float(cores) / float(reference_cores))
    return float(scaled * 10000.0)


__all__ = [
    "STAGES",
    "StageLatency",
    "StageLedger",
    "budget_fraction",
    "observed_action_budget",
    "offline_analysis_rate",
    "timed",
]
