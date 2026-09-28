"""Minimal training loop on the smoke configuration.

The smoke config exists for this file only: it shrinks the analysis guard rails so a full
training pass fits in a test session. Nothing here is a reported result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fluoroscopy_alert_takeover.cli.common import (
    analysis_config_from,
    prepare_context,
    window_spec_from,
)
from fluoroscopy_alert_takeover.cohort.builder import build_person_windows
from fluoroscopy_alert_takeover.cohort.layout import CohortLayout
from fluoroscopy_alert_takeover.perception.data import (
    FluoroscopyWindowDataset,
    FrameStore,
    site_order,
)
from fluoroscopy_alert_takeover.perception.spatial_predicate import (
    AlertPolicyConfig,
    AlertPolicyNet,
)
from fluoroscopy_alert_takeover.perception.training import (
    AlertTrainingConfig,
    score_windows,
    train_alert_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"


def test_smoke_config_trains_and_decreases_the_loss(sample_root: Path, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    context = prepare_context(
        "train_alert",
        "smoke training",
        [
            "--experiment",
            str(CONFIG_ROOT / "experiment" / "smoke.yaml"),
            "--root",
            str(CONFIG_ROOT),
            "--run-dir",
            str(tmp_path / "smoke"),
            f"cohort.root={sample_root}",
        ],
    )
    spec = window_spec_from(context.config)
    windows = build_person_windows(CohortLayout(root=sample_root), spec)
    store = FrameStore(root=sample_root, image_size=(32, 32))
    dataset = FluoroscopyWindowDataset(
        table=windows, store=store, window_frames=4, sites=site_order(windows)
    )
    model = AlertPolicyNet(AlertPolicyConfig(width=8, depth=3))
    config = AlertTrainingConfig(epochs=2, batch_size=16, learning_rate=5e-3, warmup_epochs=1)
    trace = train_alert_policy(
        model, dataset, config, device="cpu", checkpoint_path=tmp_path / "policy.pt"
    )
    assert trace.steps >= 2
    assert trace.decreased
    assert trace.total[-1] < trace.total[0]
    assert trace.as_dict()["decreased"] is True

    scores, _states, rows, dice = score_windows(model, dataset, batch_size=32)
    assert scores.shape[0] == windows.size
    assert rows.tolist() == sorted(rows.tolist())
    assert bool(((scores >= 0.0) & (scores <= 1.0)).all())
    assert np.isfinite(dice).all()


def test_smoke_analysis_config_is_light() -> None:
    context = prepare_context(
        "prepare",
        "smoke config",
        [
            "--experiment",
            str(CONFIG_ROOT / "experiment" / "smoke.yaml"),
            "--root",
            str(CONFIG_ROOT),
            "--run-dir",
            "/tmp/fat-smoke-config",
        ],
    )
    config = analysis_config_from(context.config)
    assert config.n_folds == 3
    assert config.forest.trees == 8
    assert config.n_bootstrap == 50
    assert config.thresholds == (0.30, 0.40, 0.50)


def test_smoke_pipeline_runs(sample_root: Path) -> None:
    from fluoroscopy_alert_takeover.analysis.pipeline import run_analysis

    context = prepare_context(
        "prepare",
        "smoke pipeline",
        [
            "--experiment",
            str(CONFIG_ROOT / "experiment" / "smoke.yaml"),
            "--root",
            str(CONFIG_ROOT),
            "--run-dir",
            "/tmp/fat-smoke-pipeline",
        ],
    )
    windows = build_person_windows(CohortLayout(root=sample_root), window_spec_from(context.config))
    result = run_analysis(windows, analysis_config_from(context.config))
    assert np.isfinite(result.primary.point)
    assert result.probes[0].verdict in {"PASS", "FAIL"}
