"""Assembly of the three verification artefacts.

Ref: none - release-internal. ``claim_to_code.json`` records the paper-to-code map and its
conformance status, ``verification_report.json`` records both verification layers with the
numbers they produced, and ``integrity_manifest.json`` records the hashes of the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils.io import read_json, write_json
from .conformance import ConformanceReport
from .execution import ExecutionReport
from .manifest import Manifest, build_manifest, verify_manifest

CLAIM_TO_CODE = "claim_to_code.json"
VERIFICATION_REPORT = "verification_report.json"
MANIFEST = "integrity_manifest.json"

# Values this release had to choose because the manuscript does not print them. They are
# listed explicitly so a reviewer can see exactly which numbers are the release's own.
ENGINEERING_DEFAULTS: dict[str, str] = {
    "protocol.window_length_s": (
        "30.0 s analysis period; the manuscript fixes the shape of the period, not its length"
    ),
    "protocol.window_stride_s": (
        "30.0 s stride, so the periods are disjoint and a follow-up cannot reach the next one"
    ),
    "protocol.response_window_s": (
        "30.0 s alert-to-response window; Sec. 2.5 names the interval without printing it"
    ),
    "protocol.consistency_floor": "0.90 share of windows with an identifiable exposure state",
    "gate.grid": "25 gates on [0.01, 0.25]; Algorithm 1 names a grid G without printing it",
    "gate.tolerance": "0.10 absolute standardised mean difference as kappa*",
    "gate.ess_floor": "0.20 normalised effective sample size at the frozen gate",
    "estimator.n_folds": "5 patient-level folds; Algorithm 3 names K without printing it",
    "estimator.clip_gamma": "0.01 propensity trimming bound, the gamma of Algorithm 3",
    "estimator.outcome_learner": "logistic regression on the covariate history",
    "strata.forest_trees": "200 causal trees; Algorithm 4 names T without printing it",
    "strata.forest_min_leaf": "50 rows per leaf",
    "strata.forest_max_depth": "4 levels",
    "operating_point.thresholds": "0.15, 0.25, 0.35, 0.45, 0.55, as printed in Table A3",
    "operating_point.abstention_low": "0.30 lower edge of the abstention band, not printed",
    "operating_point.abstention_high": "0.40 upper edge of the abstention band, not printed",
    "operating_point.burden_ceiling": "0.60 emission-rate ceiling of the operating-point criterion",
    "inference.bootstrap_replicates": "1,000 replicates, as printed in Table 4 Panel E",
    "alert.optimiser": (
        "AdamW at 3e-4 with cosine decay over 30 epochs at batch 8; no schedule is printed"
    ),
    "alert.precision": "fp32; the Table A1 footprint is a memory figure, not a precision",
    "schema.covariate_members": (
        "the members of the four named blocks; the manuscript names the blocks only"
    ),
    "schema.case_mix": (
        "age, sex, diabetes and chronic limb-threatening ischaemia enter the balance "
        "evaluation but not the block-level gate distance"
    ),
    "evaluation.achievable_ceiling": (
        "the ceiling is read as the AUROC of the reference channel on the same target"
    ),
}


@dataclass(frozen=True)
class VerificationBundle:
    """The two report artefacts, ready to write."""

    claim_to_code: dict[str, object]
    verification_report: dict[str, object]


def build_bundle(conformance: ConformanceReport, execution: ExecutionReport) -> VerificationBundle:
    """Assemble the two report artefacts from both verification layers.

    The integrity manifest is not assembled here: it has to be built from the tree *after*
    the reports are on disk, or it would record the digests of the previous run's bytes.
    """
    conformance_payload = conformance.as_dict()
    execution_payload = execution.as_dict()
    claim_payload: dict[str, object] = {
        "layer": "mapping and internal consistency of the manuscript's printed values",
        "summary": conformance_payload["summary"],
        "code_status": conformance_payload["code_status"],
        "overall_status": conformance_payload["overall_status"],
        "claims": conformance_payload["claims"],
        "manuscript_arithmetic_findings": conformance_payload["manuscript_findings"],
    }
    report: dict[str, object] = {
        "layer_one_paper_to_code": conformance_payload,
        "layer_two_execution": execution_payload,
        "engineering_defaults": ENGINEERING_DEFAULTS,
        "summary": {
            "code_status": conformance.code_status,
            "execution_code_status": execution.code_status,
            "overall_status": execution.overall_status,
            "claims_checked": conformance.summary["claims"],
            "claims_with_code_failures": conformance.summary["code_failures"],
            "manuscript_arithmetic_findings": conformance.summary["manuscript_arithmetic_failures"],
            "execution_checks": execution.summary["checks"],
            "execution_failures": execution.summary["fail"],
            "execution_blocked": execution.summary["blocked"],
            "execution_not_run": execution.summary["not_run"],
        },
        "notes": [
            "A manuscript-arithmetic finding is a discrepancy inside the manuscript's own "
            "printed values. It is not a defect of this release and is reported with "
            "failure_owner set to 'manuscript'.",
            "A blocked check needs the cohort, which is not distributed. Those checks are "
            "recorded as BLOCKED rather than omitted.",
            "The execution layer reads a cohort root when one is staged. The tree ships no "
            "data-producing code, so the checks that need a table are recorded as BLOCKED "
            "here and are exercised by the suite instead.",
        ],
    }
    return VerificationBundle(claim_to_code=claim_payload, verification_report=report)


def write_bundle(
    bundle: VerificationBundle, release_root: Path
) -> tuple[dict[str, Path], Manifest]:
    """Write the three artefacts, hashing the tree at the moment it is complete.

    The ordering is the point: the reports land first, then the manifest is built from the
    tree that holds them, then the manifest lands. Building it earlier would record the
    previous run's digest for every report it covers.
    """
    root = Path(release_root)
    written = {
        "claim_to_code": write_json(root / CLAIM_TO_CODE, bundle.claim_to_code),
        "verification_report": write_json(root / VERIFICATION_REPORT, bundle.verification_report),
    }
    manifest = build_manifest(root)
    written["manifest"] = write_json(root / MANIFEST, manifest.as_dict())
    return written, manifest


def verify_written_manifest(root: Path) -> dict[str, object]:
    """Rebuild the manifest and compare it with the one on disk."""
    payload = read_json(Path(root) / MANIFEST)
    recorded = Manifest(entries=dict(payload["entries"]))
    return verify_manifest(recorded, Path(root))


__all__ = [
    "CLAIM_TO_CODE",
    "ENGINEERING_DEFAULTS",
    "MANIFEST",
    "VERIFICATION_REPORT",
    "VerificationBundle",
    "build_bundle",
    "verify_written_manifest",
    "write_bundle",
]
