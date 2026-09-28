"""The verification layers themselves.

Layer one is checked for coverage and for the absence of code-side failures; layer two runs
on the session sample, with the slowest checks delegated to the pipeline identities.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fluoroscopy_alert_takeover.verification.conformance import (
    FAIL,
    NOT_RUN,
    PASS,
    arithmetic_check,
    audit,
    load_claim_map,
    load_paper_reported,
    resolve_symbol,
)
from fluoroscopy_alert_takeover.verification.execution import (
    IDENTITY_TOLERANCE,
    run_execution_verification,
)
from fluoroscopy_alert_takeover.verification.manifest import build_manifest, verify_manifest
from fluoroscopy_alert_takeover.verification.report import (
    ENGINEERING_DEFAULTS,
    build_bundle,
    verify_written_manifest,
    write_bundle,
)


def test_every_claim_resolves_to_code() -> None:
    report = audit(load_claim_map(), load_paper_reported())
    assert report.summary["claims"] >= 60
    assert report.summary["code_failures"] == 0
    assert report.code_status == PASS
    for claim in report.claims:
        assert claim.symbols, claim.id
        assert claim.status in {PASS, FAIL, NOT_RUN}


def test_claim_map_covers_every_equation_algorithm_and_table() -> None:
    entries = load_claim_map()["claims"]
    locations = " ".join(entry["location"] for entry in entries)
    for marker in (
        "Eq. (1)",
        "Eq. (2)",
        "Algorithm 1",
        "Algorithm 2",
        "Algorithm 3",
        "Algorithm 4",
        "Algorithm 5",
    ):
        assert marker in locations, marker
    for marker in ("Table 1", "Table 2", "Table 3", "Table 4", "Table A1", "Table A2", "Table A3"):
        assert marker in locations, marker


def test_paper_reported_has_no_placeholder_values() -> None:
    reported = load_paper_reported()
    assert reported["cohort"]["analytic_windows"] == 9864
    assert reported["contrasts_pp"]["takeover_aipw"]["point"] == 3.2
    assert reported["decomposition_pp"]["fraction"]["point"] == 0.73
    assert reported["alert_model"]["auroc"]["point"] == 0.918
    assert set(reported["attenuation"]) == {"1.0", "2.0", "3.0", "4.0"}


def test_resolve_symbol_rejects_an_absent_attribute() -> None:
    assert resolve_symbol("fluoroscopy_alert_takeover.utils.io:sha256_file")
    assert not resolve_symbol("fluoroscopy_alert_takeover.utils.io:absent_symbol")
    assert not resolve_symbol("fluoroscopy_alert_takeover.absent:thing")
    with pytest.raises(ValueError):
        resolve_symbol("no-colon")


def test_arithmetic_checks_are_exact() -> None:
    assert (
        arithmetic_check(
            {"kind": "sum", "values": [1.9, 0.7], "expected": 2.6, "label": "x"}
        ).status
        == PASS
    )
    assert (
        arithmetic_check(
            {
                "kind": "ratio",
                "numerator": 1.9,
                "denominator": 2.6,
                "expected": 0.73,
                "tolerance": 0.005,
                "label": "x",
            }
        ).status
        == PASS
    )
    failing = arithmetic_check({"kind": "sum", "values": [1.0, 1.0], "expected": 3.0, "label": "x"})
    assert failing.status == FAIL
    with pytest.raises(KeyError):
        arithmetic_check({"kind": "absent", "label": "x"})


def test_arithmetic_checks_cover_every_declared_kind() -> None:
    from fluoroscopy_alert_takeover.verification.conformance import load_claim_map as loader

    kinds = {
        entry["arithmetic"]["kind"]
        for entry in loader()["claims"]
        if entry.get("arithmetic") is not None
    }
    assert {"sum", "ratio", "share", "compare", "pooled_share", "weighted_mean"} <= kinds
    assert {"max_equals", "max_matches", "monotone_decreasing", "monotone_increasing"} <= kinds
    assert {
        "interval_excludes",
        "inverted_u",
        "peak_index",
        "row_sums",
        "i_squared_from_q",
    } <= kinds


def test_manuscript_findings_are_attributed_to_the_manuscript() -> None:
    report = audit(load_claim_map(), load_paper_reported())
    findings = report.as_dict()["manuscript_findings"]
    assert findings, "the audit must record the discrepancies it finds inside the printed values"
    for finding in findings:
        assert finding["status"] == FAIL
    assert report.overall_status == PASS
    assert report.summary["manuscript_arithmetic_failures"] == len(findings)


def test_execution_layer_on_the_sample(sample_root: Path) -> None:
    report = run_execution_verification(sample_root)
    names = {check.name for check in report.checks}
    assert {
        "read_cohort_layers",
        "tensor_forward_shapes",
        "loss_and_backward",
        "parameter_update",
        "checkpoint_round_trip",
        "single_batch_overfit",
        "training_loop",
        "decomposition_additivity",
        "eq2_matches_hand_written_terms",
        "kish_effective_sample_size",
        "holm_step_down",
        "benjamini_hochberg",
        "wilson_interval",
        "cochran_q",
    } <= names
    assert report.summary["fail"] == 0, [check for check in report.checks if check.status == FAIL]
    assert report.code_status == PASS


def test_execution_layer_blocks_without_a_cohort() -> None:
    report = run_execution_verification(None)
    assert report.summary["blocked"] > 0
    assert report.summary["fail"] == 0
    assert report.overall_status == "PARTIALLY_VERIFIED"
    states = {check.name: check.status for check in report.checks}
    assert states["read_cohort_layers"] == "BLOCKED"
    assert states["tensor_forward_shapes"] == PASS


def test_identity_tolerance_is_tight() -> None:
    assert IDENTITY_TOLERANCE <= 1e-9


def test_manifest_round_trip(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    manifest = build_manifest(tmp_path)
    assert manifest.file_count == 2
    assert verify_manifest(manifest, tmp_path)["matches"]
    (tmp_path / "b.txt").write_text("changed", encoding="utf-8")
    report = verify_manifest(manifest, tmp_path)
    assert not report["matches"]
    assert report["changed"] == ["b.txt"]
    (tmp_path / "c.txt").write_text("gamma", encoding="utf-8")
    assert verify_manifest(manifest, tmp_path)["added"] == ["c.txt"]


def test_manifest_skips_its_own_artefacts(tmp_path: Path) -> None:
    (tmp_path / "integrity_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "verification_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "claim_to_code.json").write_text("{}", encoding="utf-8")
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
    manifest = build_manifest(tmp_path)
    assert manifest.file_count == 1
    assert "real.py" in manifest.entries


def test_manifest_records_no_absolute_path(tmp_path: Path) -> None:
    """The manifest has to describe the tree wherever a reader unpacked it."""
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    payload = build_manifest(tmp_path).as_dict()
    assert payload["root"] == "."
    assert all(not key.startswith("/") for key in payload["entries"])
    assert not any(str(tmp_path) in str(value) for value in payload.values())


def test_bundle_records_the_engineering_defaults(tmp_path: Path) -> None:
    conformance = audit(load_claim_map(), load_paper_reported())
    execution = run_execution_verification(None)
    manifest = build_manifest(tmp_path)
    bundle = build_bundle(conformance, execution, manifest)
    assert bundle.verification_report["engineering_defaults"] == ENGINEERING_DEFAULTS
    assert bundle.claim_to_code["summary"]["claims"] >= 60
    assert bundle.manifest["file_count"] == 0
    written = write_bundle(bundle, tmp_path)
    assert all(path.is_file() for path in written.values())
    assert verify_written_manifest(tmp_path)["matches"]
