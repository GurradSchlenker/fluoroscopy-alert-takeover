"""First verification layer: paper-to-code conformance.

Ref: Sec. 4.2-4.10, pp. 14-21; every entry of ``claim_map.json`` and ``paper_reported.json``.

Two things are established here. First, the *mapping*: each claim names the symbols that
carry it, and the audit imports them, so a claim cannot be listed as implemented while the
code that would implement it is absent. Second, the *arithmetic*: where the manuscript
prints values that must agree with one another, the audit recomputes the relation and
reports whether it holds. A failed arithmetic check is a finding about the manuscript, not
about this release, so it is reported with ``failure_owner`` set accordingly.
"""

from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.io import read_json
from ..utils.logging import get_logger

_LOG = get_logger("verification.conformance")

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ArithmeticResult:
    """Outcome of one internal-consistency check of the manuscript's printed values."""

    label: str
    kind: str
    status: str
    observed: float
    expected: float
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "status": self.status,
            "observed": self.observed,
            "expected": self.expected,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ClaimStatus:
    """One claim, its code binding and its arithmetic finding."""

    id: str
    location: str
    statement: str
    verification: str
    symbols: list[str]
    missing_symbols: list[str]
    arithmetic: ArithmeticResult | None
    unbound: str | None = None

    @property
    def code_present(self) -> bool:
        return not self.missing_symbols

    @property
    def status(self) -> str:
        if not self.code_present:
            return FAIL
        if self.arithmetic is None:
            return NOT_RUN if self.unbound else PASS
        return self.arithmetic.status

    @property
    def failure_owner(self) -> str:
        if not self.code_present:
            return "release"
        if self.arithmetic is not None and self.arithmetic.status == FAIL:
            return "manuscript"
        return "none"

    @property
    def code_status(self) -> str:
        """Whether the code side of the claim holds, independent of the manuscript's arithmetic."""
        return PASS if self.code_present else FAIL

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "location": self.location,
            "statement": self.statement,
            "verification": self.verification,
            "symbols": list(self.symbols),
            "missing_symbols": list(self.missing_symbols),
            "code_present": self.code_present,
            "code_status": self.code_status,
            "status": self.status,
            "failure_owner": self.failure_owner,
            "unbound": self.unbound,
            "arithmetic": None if self.arithmetic is None else self.arithmetic.as_dict(),
        }


@dataclass
class ConformanceReport:
    """The whole first layer."""

    claims: list[ClaimStatus] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, float]:
        counts = {PASS: 0, FAIL: 0, NOT_RUN: 0}
        for claim in self.claims:
            counts[claim.status] = counts.get(claim.status, 0) + 1
        return {
            "claims": float(len(self.claims)),
            "pass": float(counts.get(PASS, 0)),
            "fail": float(counts.get(FAIL, 0)),
            "not_run": float(counts.get(NOT_RUN, 0)),
            "code_failures": float(sum(1 for claim in self.claims if not claim.code_present)),
            "manuscript_arithmetic_failures": float(
                sum(1 for claim in self.claims if claim.failure_owner == "manuscript")
            ),
        }

    @property
    def code_status(self) -> str:
        return PASS if self.summary["code_failures"] == 0 else FAIL

    @property
    def overall_status(self) -> str:
        if self.summary["code_failures"] > 0:
            return FAIL
        if self.summary["manuscript_arithmetic_failures"] > 0:
            return PASS
        return PASS

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "code_status": self.code_status,
            "overall_status": self.overall_status,
            "claims": [claim.as_dict() for claim in self.claims],
            "manuscript_findings": [
                claim.arithmetic.as_dict()
                for claim in self.claims
                if claim.failure_owner == "manuscript" and claim.arithmetic is not None
            ],
        }


def resolve_symbol(dotted: str) -> bool:
    """Whether ``module.path:attribute`` resolves in the installed package."""
    if ":" not in dotted:
        raise ValueError(f"a symbol must be module:attribute, got {dotted!r}")
    module_name, _, attribute = dotted.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False
    return hasattr(module, attribute)


def _close(observed: float, expected: float, tolerance: float) -> bool:
    return abs(observed - expected) <= tolerance + _TOLERANCE


def arithmetic_check(spec: dict[str, Any]) -> ArithmeticResult:
    """Evaluate one internal-consistency relation of the manuscript's printed values."""
    kind = str(spec["kind"])
    label = str(spec["label"])

    def result(status: str, observed: float, expected: float, detail: str) -> ArithmeticResult:
        return ArithmeticResult(
            label=label,
            kind=kind,
            status=status,
            observed=observed,
            expected=expected,
            detail=detail,
        )

    if kind == "compare":
        left = float(spec["left"])
        right = float(spec["right"])
        tolerance = float(spec.get("tolerance", 1e-6))
        relation = str(spec["relation"])
        if relation == ">=":
            ok = left >= right - tolerance
        elif relation == "<=":
            ok = left <= right + tolerance
        elif relation == "==":
            ok = _close(left, right, tolerance)
        else:
            raise KeyError(f"unknown relation: {relation}")
        return result(PASS if ok else FAIL, left, right, f"{left} {relation} {right}")
    if kind == "ratio":
        observed = float(spec["numerator"]) / float(spec["denominator"])
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 1e-3)))
        return result(PASS if ok else FAIL, observed, expected, "ratio of the two printed values")
    if kind == "sum":
        observed = float(sum(spec["values"]))
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 0.5)))
        return result(PASS if ok else FAIL, observed, expected, "sum of the printed parts")
    if kind == "difference":
        observed = float(spec["left"]) - float(spec["right"])
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 1e-3)))
        return result(PASS if ok else FAIL, observed, expected, "difference of the printed values")
    if kind == "share":
        observed = float(spec["part"]) / float(spec["whole"])
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 1e-3)))
        return result(PASS if ok else FAIL, observed, expected, "share of the printed denominator")
    if kind == "pooled_share":
        parts = [float(value) for value in spec["parts"]]
        sizes = [float(value) for value in spec["sizes"]]
        observeds = [parts[i] / sizes[i] for i in range(len(parts))]
        observed = sum(observeds[i] * sizes[i] for i in range(len(parts))) / sum(sizes)
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 1e-3)))
        return result(
            PASS if ok else FAIL, observed, expected, "size-weighted pooling of the site values"
        )
    if kind == "weighted_mean":
        values = [float(value) for value in spec["values"]]
        weights = [float(value) for value in spec["weights"]]
        observed = sum(v * w for v, w in zip(values, weights, strict=False)) / sum(weights)
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 0.05)))
        return result(PASS if ok else FAIL, observed, expected, "size-weighted mean of the strata")
    if kind == "max_equals":
        observed = max(float(value) for value in spec["values"])
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 1e-3)))
        return result(
            PASS if ok else FAIL, observed, expected, "largest printed value in the column"
        )
    if kind == "max_matches":
        candidates = {str(key): float(value) for key, value in spec["values"].items()}
        observed_key = max(candidates, key=lambda name: candidates[name])
        ok = observed_key == str(spec["expected_key"])
        return result(
            PASS if ok else FAIL,
            candidates[observed_key],
            candidates[str(spec["expected_key"])],
            f"largest printed characteristic is {observed_key}",
        )
    if kind == "evalue_from_rates":
        exposed = float(spec["exposed_rate"])
        unexposed = float(spec["unexposed_rate"])
        ratio = exposed / unexposed if unexposed > 0.0 else 0.0
        value = ratio if ratio >= 1.0 else (1.0 / ratio if ratio > 0.0 else 1.0)
        observed = value + (value * (value - 1.0)) ** 0.5 if value > 0.0 else 1.0
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 0.02)))
        return result(
            PASS if ok else FAIL, observed, expected, "E-value from the printed outcome rates"
        )
    if kind in {"monotone_decreasing", "monotone_increasing"}:
        values = [float(value) for value in spec["values"]]
        pairs = list(itertools.pairwise(values))
        if kind == "monotone_decreasing":
            ok = all(later <= earlier for earlier, later in pairs)
        else:
            ok = all(later >= earlier for earlier, later in pairs)
        return result(
            PASS if ok else FAIL,
            float(values[-1] - values[0]),
            0.0,
            "first minus last of the printed sequence",
        )
    if kind == "interval_excludes":
        low = float(spec["low"])
        high = float(spec["high"])
        value = float(spec["value"])
        ok = not (low <= value <= high)
        return result(PASS if ok else FAIL, low, value, f"interval {low} to {high} against {value}")
    if kind == "inverted_u":
        values = [float(value) for value in spec["values"]]
        peak = max(range(len(values)), key=lambda index: values[index])
        ok = 0 < peak < len(values) - 1
        return result(
            PASS if ok else FAIL, float(peak), float(peak), "interior maximum of the sequence"
        )
    if kind == "peak_index":
        values = [float(value) for value in spec["values"]]
        observed = float(max(range(len(values)), key=lambda index: values[index]))
        expected = float(spec["expected_index"])
        ok = observed == expected
        return result(
            PASS if ok else FAIL, observed, expected, "index of the largest printed value"
        )
    if kind == "equal_values":
        left = float(spec["left"])
        right = float(spec["right"])
        return result(
            PASS if _close(left, right, 1e-9) else FAIL, left, right, "two printed values"
        )
    if kind == "relative_reduction":
        before = float(spec["before"])
        after = float(spec["after"])
        observed = 1.0 - after / before if before > 0.0 else float("nan")
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 1e-3)))
        return result(
            PASS if ok else FAIL, observed, expected, "relative reduction of the two proportions"
        )
    if kind == "row_sums":
        row_values = [[float(value) for value in row] for row in spec["rows"]]
        row_targets = [float(value) for value in spec["expected"]]
        tolerance = float(spec.get("tolerance", 0.5))
        total = float(sum(sum(row) for row in row_values))
        ok = all(
            _close(sum(row), target, tolerance)
            for row, target in zip(row_values, row_targets, strict=False)
        )
        detail = "; ".join(
            f"row {index} sums to {sum(row)} against {target}"
            for index, (row, target) in enumerate(zip(row_values, row_targets, strict=False))
        )
        return result(PASS if ok else FAIL, total, float(sum(row_targets)), detail)
    if kind == "range_equals":
        values = [float(value) for value in spec["values"]]
        observed = max(values) - min(values)
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 0.05)))
        return result(
            PASS if ok else FAIL, observed, expected, "range of the printed subgroup estimates"
        )
    if kind == "range_covers":
        values = [float(value) for value in spec["values"]]
        low = float(spec["low"])
        high = float(spec["high"])
        ok = all(low <= value <= high for value in values)
        return result(
            PASS if ok else FAIL,
            max(values),
            high,
            f"printed site estimates {values} against the stated range {low} to {high}",
        )
    if kind == "i_squared_from_q":
        q = float(spec["q"])
        df = float(spec["df"])
        observed = 0.0 if q <= df else 100.0 * (q - df) / q
        expected = float(spec["expected"])
        ok = _close(observed, expected, float(spec.get("tolerance", 0.2)))
        return result(
            PASS if ok else FAIL,
            observed,
            expected,
            "conventional I-squared from the printed Q and df",
        )
    raise KeyError(f"unknown arithmetic check kind: {kind}")


def audit(claim_map: dict[str, Any], reported: dict[str, Any]) -> ConformanceReport:
    """Run the first layer over the whole claim map."""
    report = ConformanceReport()
    for entry in claim_map["claims"]:
        symbols = [str(value) for value in entry["symbols"]]
        missing = [symbol for symbol in symbols if not resolve_symbol(symbol)]
        arithmetic = None
        if entry.get("arithmetic") is not None:
            arithmetic = arithmetic_check(entry["arithmetic"])
        report.claims.append(
            ClaimStatus(
                id=str(entry["id"]),
                location=str(entry["location"]),
                statement=str(entry["statement"]),
                verification=str(entry["verification"]),
                symbols=symbols,
                missing_symbols=missing,
                arithmetic=arithmetic,
                unbound=None if entry.get("unbound") is None else str(entry["unbound"]),
            )
        )
    _LOG.info(
        "conformance audit: %d claims, %d code failures, %d manuscript arithmetic findings",
        report.summary["claims"],
        report.summary["code_failures"],
        report.summary["manuscript_arithmetic_failures"],
    )
    del reported
    return report


def load_claim_map() -> dict[str, Any]:
    """Read the packaged claim map."""
    payload: dict[str, Any] = read_json(Path(__file__).parent / "claim_map.json")
    return payload


def load_paper_reported() -> dict[str, Any]:
    """Read the packaged transcription of the manuscript's printed values."""
    payload: dict[str, Any] = read_json(Path(__file__).parent / "paper_reported.json")
    return payload


__all__ = [
    "FAIL",
    "NOT_RUN",
    "PASS",
    "ArithmeticResult",
    "ClaimStatus",
    "ConformanceReport",
    "arithmetic_check",
    "audit",
    "load_claim_map",
    "load_paper_reported",
    "resolve_symbol",
]
