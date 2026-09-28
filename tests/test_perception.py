"""Alert-policy perception: shapes, losses, the training loop and checkpointing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fluoroscopy_alert_takeover.perception.calibration import (
    applied_recalibration,
    calibration_curve,
    logit,
    recalibrate,
)
from fluoroscopy_alert_takeover.perception.checkpoint import load_checkpoint, save_checkpoint
from fluoroscopy_alert_takeover.perception.data import (
    FluoroscopyWindowDataset,
    FrameStore,
    WindowSample,
    collate,
    site_order,
)
from fluoroscopy_alert_takeover.perception.dice import (
    dice_coefficient,
    per_site_dice,
    soft_dice_loss,
)
from fluoroscopy_alert_takeover.perception.encoder import (
    FluoroscopyEncoder,
    StateSpaceEncoder,
    build_encoder,
)
from fluoroscopy_alert_takeover.perception.losses import policy_loss
from fluoroscopy_alert_takeover.perception.spatial_predicate import (
    AlertPolicyConfig,
    AlertPolicyNet,
    window_alert_stream,
)
from fluoroscopy_alert_takeover.perception.training import (
    AlertTrainingConfig,
    build_optimizer,
    build_scheduler,
    score_windows,
    train_alert_policy,
)
from fluoroscopy_alert_takeover.protocol.schema import PersonWindowTable


def test_encoder_shapes() -> None:
    encoder = FluoroscopyEncoder(width=8, depth=3)
    out = encoder(torch.randn(2, 1, 32, 32))
    assert out.shape[:2] == (2, encoder.out_channels)
    assert out.shape[0] == 2
    alternative = StateSpaceEncoder(width=6)
    assert alternative(torch.randn(2, 1, 32, 32)).shape[:2] == (2, 6)
    with pytest.raises(ValueError):
        build_encoder("unknown")
    with pytest.raises(ValueError):
        FluoroscopyEncoder(depth=1)


def test_policy_forward_keeps_the_frame_grid() -> None:
    model = AlertPolicyNet(AlertPolicyConfig(width=8, depth=3))
    frames = torch.randn(2, 3, 1, 24, 40)
    output = model(frames)
    assert output.predicate_logits.shape == (2, 3, 1, 24, 40)
    assert output.alert_logits.shape == (2, 3)
    assert output.window_scores().shape == (2,)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 24, 40))


def test_window_alert_stream_withholds_the_band() -> None:
    scores = torch.tensor([0.1, 0.35, 0.5, 0.9])
    assert window_alert_stream(scores, 0.3).tolist() == [False, True, True, True]
    banded = window_alert_stream(scores, 0.3, 0.4, 0.6)
    assert banded.tolist() == [False, True, False, True]


def test_dice_closed_forms() -> None:
    perfect = torch.tensor([[[[9.0, 9.0]]]])
    reference = torch.tensor([[[[1.0, 1.0]]]])
    assert dice_coefficient(perfect, reference) == pytest.approx(1.0)
    disjoint = torch.tensor([[[[-9.0, -9.0]]]])
    assert dice_coefficient(disjoint, reference) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        dice_coefficient(perfect, torch.zeros(1, 1, 3, 3))
    loss = soft_dice_loss(perfect, reference)
    assert float(loss) < 0.01


def test_per_site_dice_by_hand() -> None:
    prediction = torch.tensor([[[[9.0]]], [[[9.0]]], [[[-9.0]]]])
    reference = torch.ones_like(prediction)
    site_index = torch.tensor([0, 1, 0])
    values = per_site_dice(prediction, reference, site_index, 2)
    # Site 0 holds rows 0 and 2: one positive hit and one negative missed, so
    # 2 * 1 / (1 + 2) = 2/3.
    assert values[0] == pytest.approx(2.0 / 3.0)
    assert values[1] == pytest.approx(1.0)
    assert np.isnan(per_site_dice(prediction, reference, site_index, 3)[2])


def test_policy_loss_terms_are_finite() -> None:
    model = AlertPolicyNet(AlertPolicyConfig(width=8, depth=3))
    frames = torch.randn(2, 3, 1, 16, 16)
    mask = (torch.rand(2, 3, 1, 16, 16) > 0.5).float()
    output = model(frames)
    breakdown = policy_loss(
        output,
        mask,
        torch.rand(2, 3),
        torch.rand(2, 1),
        1.0,
        1.0,
        0.5,
        1.0,
        mask_observed=torch.ones(2, 3, dtype=torch.bool),
    )
    values = breakdown.as_dict()
    assert all(np.isfinite(list(values.values())))
    with pytest.raises(ValueError):
        policy_loss(
            output,
            mask,
            torch.rand(2, 3),
            torch.rand(2, 1),
            1.0,
            1.0,
            0.5,
            1.0,
            mask_observed=torch.zeros_like(mask, dtype=torch.bool),
        )


def test_optimizer_skips_decay_on_flat_parameters() -> None:
    model = AlertPolicyNet(AlertPolicyConfig(width=8, depth=3))
    config = AlertTrainingConfig(epochs=1, weight_decay=0.1)
    optimizer = build_optimizer(model, config)
    decays = [group["weight_decay"] for group in optimizer.param_groups]
    assert 0.0 in decays and 0.1 in decays


def test_scheduler_decays() -> None:
    torch.set_num_threads(1)
    model = AlertPolicyNet(AlertPolicyConfig(width=8, depth=3))
    optimizer = build_optimizer(model, AlertTrainingConfig(epochs=2, warmup_epochs=1))
    scheduler = build_scheduler(optimizer, AlertTrainingConfig(epochs=2, warmup_epochs=1), 4)
    rates = []
    for _ in range(8):
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert rates[-1] < rates[0]


def test_dataset_yields_aligned_targets(sample_root: Path, windows: PersonWindowTable) -> None:
    store = FrameStore(root=sample_root, image_size=(16, 16))
    dataset = FluoroscopyWindowDataset(
        table=windows, store=store, window_frames=4, sites=site_order(windows)
    )
    sample = dataset[0]
    assert sample["frames"].shape == (4, 1, 16, 16)
    assert sample["mask"].shape == (4, 1, 16, 16)
    assert sample["frame_alert"].shape == (4,)
    assert sample["window_alert"].shape == (1,)
    assert sample["mask_observed"].shape == (4,)
    batch = collate([dataset[0], dataset[1]])
    assert batch["frames"].shape[0] == 2
    assert all(isinstance(batch[key], torch.Tensor) for key in batch)


def test_dataset_points_at_a_missing_procedure(
    sample_root: Path, windows: PersonWindowTable
) -> None:
    store = FrameStore(root=sample_root / "absent", image_size=(16, 16))
    dataset = FluoroscopyWindowDataset(
        table=windows, store=store, window_frames=2, sites=site_order(windows)
    )
    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_training_reduces_the_loss(sample_root: Path, windows: PersonWindowTable) -> None:
    torch.set_num_threads(1)
    store = FrameStore(root=sample_root, image_size=(16, 16))
    dataset = FluoroscopyWindowDataset(
        table=windows, store=store, window_frames=3, sites=site_order(windows)
    )
    model = AlertPolicyNet(AlertPolicyConfig(width=8, depth=3))
    trace = train_alert_policy(
        model,
        dataset,
        AlertTrainingConfig(epochs=3, batch_size=16, learning_rate=5e-3, warmup_epochs=1),
        device="cpu",
    )
    assert trace.steps > 0
    assert trace.decreased
    assert trace.total[-1] < trace.total[0]


def test_checkpoint_round_trip_is_exact(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    model = AlertPolicyNet(AlertPolicyConfig(width=8, depth=3))
    frames = torch.randn(1, 2, 1, 16, 16)
    path = save_checkpoint(tmp_path / "policy.pt", model, None, 7, 99, extra={"tag": "round trip"})
    restored = AlertPolicyNet(AlertPolicyConfig(width=8, depth=3))
    epoch, seed, extra = load_checkpoint(path, restored, None)
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.equal(model(frames).window_scores(), restored(frames).window_scores())
    assert (epoch, seed, extra) == (7, 99, {"tag": "round trip"})
    assert not list(tmp_path.glob("*.tmp"))


def test_score_windows_reports_states_and_dice(
    sample_root: Path, windows: PersonWindowTable
) -> None:
    torch.set_num_threads(1)
    store = FrameStore(root=sample_root, image_size=(16, 16))
    dataset = FluoroscopyWindowDataset(
        table=windows, store=store, window_frames=3, sites=site_order(windows)
    )
    model = AlertPolicyNet(AlertPolicyConfig(width=8, depth=3))
    scores, states, rows, dice = score_windows(model, dataset, batch_size=32)
    assert scores.shape == states.shape == rows.shape
    assert scores.shape[0] == dataset.__len__()
    assert bool(((scores >= 0.0) & (scores <= 1.0)).all())
    assert dice.size >= 1


def test_calibration_recovers_a_shifted_score() -> None:
    rng = np.random.default_rng(4)
    latent = rng.normal(size=600)
    labels = (rng.uniform(size=600) < 1.0 / (1.0 + np.exp(-latent))).astype(np.bool_)
    scores = 1.0 / (1.0 + np.exp(-(0.5 * latent + 0.8)))
    curve = calibration_curve(scores, labels, 5)
    assert curve.counts.sum() == 600
    assert 0.0 <= curve.expected_error <= 0.5
    groups = np.asarray(["A"] * 300 + ["B"] * 300)
    fits = recalibrate(scores, labels, groups)
    assert [fit.site for fit in fits] == ["A", "B"]
    assert all(np.isfinite(fit.slope) for fit in fits)
    raised = applied_recalibration(scores, fits[0])
    assert raised.shape == scores.shape
    assert float(logit(np.asarray([0.5]))[0]) == pytest.approx(0.0)


def test_calibration_handles_a_single_class_group() -> None:
    scores = np.linspace(0.05, 0.95, 40)
    labels = np.zeros(40, dtype=np.bool_)
    fits = recalibrate(scores, labels, np.asarray(["A"] * 40))
    assert not np.isfinite(fits[0].slope)


def test_window_sample_is_a_mapping(sample_root: Path, windows: PersonWindowTable) -> None:
    store = FrameStore(root=sample_root, image_size=(8, 8))
    dataset = FluoroscopyWindowDataset(
        table=windows, store=store, window_frames=2, sites=site_order(windows)
    )
    sample: WindowSample = dataset[0]
    assert set(sample) == {
        "frames",
        "mask",
        "frame_alert",
        "window_alert",
        "mask_observed",
        "site",
        "row",
    }
