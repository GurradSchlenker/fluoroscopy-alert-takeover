"""Run both verification layers and write the three artefacts.

Ref: none - release-internal. Layer one checks the paper-to-code mapping and the internal
consistency of the manuscript's printed values; layer two executes the release. The command
writes ``claim_to_code.json``, ``verification_report.json`` and
``integrity_manifest.json`` into the release root.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..utils.logging import get_logger
from ..verification.conformance import audit, load_claim_map, load_paper_reported
from ..verification.execution import run_execution_verification
from ..verification.manifest import MANIFEST_NAME
from ..verification.report import build_bundle, write_bundle
from .common import EXIT_FAILED, EXIT_OK, cohort_root_from, prepare_context

_LOG = get_logger("cli.verify")

_RELEASE_ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    """Run the two layers, write the artefacts and report the summary on stdout."""
    context = prepare_context("verify", "Run the verification layers", argv)
    conformance = audit(load_claim_map(), load_paper_reported())
    root = cohort_root_from(context.config)
    cohort = root if root.is_dir() else None
    if cohort is None:
        _LOG.warning("no cohort root at %s; cohort-dependent checks will be BLOCKED", root)
    execution = run_execution_verification(cohort)

    bundle = build_bundle(conformance, execution)
    written, manifest = write_bundle(bundle, _RELEASE_ROOT)
    summary = _summary(bundle.verification_report)
    context.record(
        {
            "status": execution.overall_status,
            "written": {name: str(path) for name, path in written.items()},
            **summary,
        }
    )
    print(f"overall: {summary['overall_status']}")
    print(f"code: {summary['code_status']}, execution: {summary['execution_code_status']}")
    print(
        "claims {c} (code failures {f}), manuscript arithmetic findings {m}".format(
            c=summary["claims_checked"],
            f=summary["claims_with_code_failures"],
            m=summary["manuscript_arithmetic_findings"],
        )
    )
    print(
        "execution checks {checks} (failures {fails}, blocked {blocked}, not run {notrun})".format(
            checks=summary["execution_checks"],
            fails=summary["execution_failures"],
            blocked=summary["execution_blocked"],
            notrun=summary["execution_not_run"],
        )
    )
    print(f"wrote {MANIFEST_NAME} over {manifest.file_count} files")
    if summary["execution_failures"] or summary["claims_with_code_failures"]:
        return EXIT_FAILED
    return EXIT_OK


def _summary(report: dict[str, object]) -> dict[str, object]:
    """Pull the headline block out of the assembled report."""
    block = report.get("summary")
    if not isinstance(block, dict):
        raise TypeError("the verification report carries no summary block")
    return {str(key): value for key, value in block.items()}


if __name__ == "__main__":
    raise SystemExit(main())


_ = sys
