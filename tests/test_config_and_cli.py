"""Configuration composition and the command-line surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluoroscopy_alert_takeover.cli.common import (
    EXIT_BLOCKED,
    EXIT_OK,
    analysis_config_from,
    gamma_grid_from,
    load_table,
    prepare_context,
    window_spec_from,
)
from fluoroscopy_alert_takeover.cli.prepare import main as prepare_main
from fluoroscopy_alert_takeover.utils.config import (
    apply_override,
    compose,
    deep_merge,
    dig,
    load_yaml,
    require,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"


def test_config_chain_composes() -> None:
    config = compose(
        [CONFIG_ROOT / "protocol" / "default.yaml", CONFIG_ROOT / "estimator" / "default.yaml"]
    )
    assert dig(config, "protocol.primary_outcome") == "takeover"
    assert dig(config, "estimator.n_folds") == 5
    assert dig(config, "absent.key", "fallback") == "fallback"
    with pytest.raises(KeyError):
        require(config, "absent.key")


def test_overrides_win_over_the_file() -> None:
    base = load_yaml(CONFIG_ROOT / "estimator" / "default.yaml")
    merged = apply_override(base, "estimator.n_folds=9")
    assert dig(merged, "estimator.n_folds") == 9
    assert dig(base, "estimator.n_folds") == 5
    with pytest.raises(ValueError):
        apply_override(base, "no-equals-sign")
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}


def test_every_experiment_config_parses() -> None:
    for path in sorted((CONFIG_ROOT / "experiment").glob("*.yaml")):
        payload = load_yaml(path)
        assert isinstance(payload, dict)
        assert "protocol" in payload, path.name
        assert dig(payload, "cohort.root") == "data/cohort", path.name


def test_analysis_config_reads_the_experiment() -> None:
    experiment = CONFIG_ROOT / "experiment" / "primary.yaml"
    config = compose(
        [
            CONFIG_ROOT / "protocol" / "default.yaml",
            CONFIG_ROOT / "estimator" / "default.yaml",
            CONFIG_ROOT / "gate" / "default.yaml",
            CONFIG_ROOT / "operating_point" / "default.yaml",
            CONFIG_ROOT / "strata" / "default.yaml",
            CONFIG_ROOT / "inference" / "default.yaml",
            CONFIG_ROOT / "mediation" / "default.yaml",
            CONFIG_ROOT / "falsification" / "default.yaml",
            experiment,
        ]
    )
    analysis = analysis_config_from(config)
    assert analysis.primary_outcome == "takeover"
    assert analysis.secondary_outcome == "completion"
    assert analysis.n_folds == 5
    assert analysis.abstention.low == 0.30
    assert analysis.thresholds == (0.15, 0.25, 0.35, 0.45, 0.55)
    assert analysis.window_multipliers == (1.0, 2.0, 3.0, 4.0)
    assert len(gamma_grid_from(config)) > 10


def test_prepare_context_installs_a_run_directory(tmp_path: Path) -> None:
    context = prepare_context(
        "prepare",
        "context test",
        [
            "--experiment",
            str(CONFIG_ROOT / "experiment" / "smoke.yaml"),
            "--root",
            str(CONFIG_ROOT),
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )
    assert context.run_dir.is_dir()
    assert context.seed == 20260101
    path = context.record({"status": "PASS"})
    assert path.is_file()


def test_window_spec_follows_the_protocol(tmp_path: Path) -> None:
    context = prepare_context(
        "prepare",
        "spec test",
        [
            "--experiment",
            str(CONFIG_ROOT / "experiment" / "smoke.yaml"),
            "--root",
            str(CONFIG_ROOT),
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )
    spec = window_spec_from(context.config)
    assert spec.length_s == 30.0
    assert spec.response_s == 30.0


def test_load_table_reports_a_missing_cohort(tmp_path: Path) -> None:
    context = prepare_context(
        "prepare",
        "missing cohort",
        [
            "--experiment",
            str(CONFIG_ROOT / "experiment" / "smoke.yaml"),
            "--root",
            str(CONFIG_ROOT),
            "--run-dir",
            str(tmp_path / "run"),
            f"cohort.root={tmp_path / 'absent'}",
        ],
    )
    with pytest.raises(FileNotFoundError):
        load_table(context.config)


def test_prepare_command_blocks_without_a_cohort(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del capsys
    status = prepare_main(
        [
            "--experiment",
            str(CONFIG_ROOT / "experiment" / "smoke.yaml"),
            "--root",
            str(CONFIG_ROOT),
            "--run-dir",
            str(tmp_path / "run"),
            f"cohort.root={tmp_path / 'absent'}",
        ]
    )
    assert status == EXIT_BLOCKED


def test_prepare_command_reads_a_sample(sample_root: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "sample_run"
    status = prepare_main(
        [
            "--experiment",
            str(CONFIG_ROOT / "experiment" / "smoke.yaml"),
            "--root",
            str(CONFIG_ROOT),
            "--run-dir",
            str(run_dir),
            f"cohort.root={sample_root}",
        ]
    )
    assert status == EXIT_OK
    from fluoroscopy_alert_takeover.utils.io import read_json

    record = read_json(run_dir / "prepare.json")
    assert record["status"] == "PASS"
    assert record["rows"] > 0
