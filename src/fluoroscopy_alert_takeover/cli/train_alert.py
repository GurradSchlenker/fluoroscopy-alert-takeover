"""Train the alert policy on the fluoroscopic stream.

Ref: Sec. 2.2, p. 3 - the spatial predicate and its Dice; Table A1, p. 25 - the deployment
footprint. The manuscript prints no optimiser, schedule or epoch count, so the training
settings come from the alert config and are engineering defaults.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..cohort.builder import build_person_windows
from ..cohort.layout import CohortLayout
from ..perception.data import FluoroscopyWindowDataset, FrameStore, site_order
from ..perception.spatial_predicate import AlertPolicyConfig, AlertPolicyNet
from ..perception.training import AlertTrainingConfig, score_windows, train_alert_policy
from ..protocol.windowing import WindowSpec
from ..utils.config import dig
from ..utils.io import write_json
from ..utils.logging import get_logger
from .common import EXIT_BLOCKED, EXIT_OK, cohort_root_from, prepare_context

_LOG = get_logger("cli.train_alert")


def main(argv: list[str] | None = None) -> int:
    """Train the policy, record the loss history and the scored window stream."""
    context = prepare_context("train_alert", "Train the navigation alert policy", argv)
    root = cohort_root_from(context.config)
    layout = CohortLayout(root=root)
    if layout.missing():
        context.record({"status": "BLOCKED", "reason": "cohort root not staged", "root": str(root)})
        return EXIT_BLOCKED

    spec = WindowSpec(
        length_s=float(dig(context.config, "protocol.window_length_s", 30.0)),
        stride_s=float(dig(context.config, "protocol.window_stride_s", 15.0)),
        response_s=float(dig(context.config, "protocol.response_window_s", 30.0)),
    )
    table = build_person_windows(layout, spec)
    train = AlertTrainingConfig(
        epochs=int(dig(context.config, "alert.epochs", 30)),
        batch_size=int(dig(context.config, "alert.batch_size", 8)),
        learning_rate=float(dig(context.config, "alert.learning_rate", 3e-4)),
        weight_decay=float(dig(context.config, "alert.weight_decay", 1e-4)),
        grad_clip=float(dig(context.config, "alert.grad_clip", 1.0)),
        warmup_epochs=int(dig(context.config, "alert.warmup_epochs", 2)),
        precision=str(dig(context.config, "alert.precision", "fp32")),
        seed=context.seed,
        checkpoint_every=int(dig(context.config, "alert.checkpoint_every", 10)),
    )
    policy = AlertPolicyConfig(
        route=str(dig(context.config, "alert.route", "spatial_predicate")),
        width=int(dig(context.config, "alert.width", 32)),
        depth=int(dig(context.config, "alert.depth", 4)),
    )
    store = FrameStore(
        root=root,
        image_size=(
            int(dig(context.config, "alert.frame_height", 64)),
            int(dig(context.config, "alert.frame_width", 64)),
        ),
    )
    dataset = FluoroscopyWindowDataset(
        table=table,
        store=store,
        window_frames=int(dig(context.config, "alert.window_frames", 8)),
        sites=site_order(table),
    )
    model = AlertPolicyNet(policy)
    checkpoint = context.run_dir / "alert_policy.pt"
    trace = train_alert_policy(
        model,
        dataset,
        train,
        device=str(dig(context.config, "alert.device", "cpu")),
        checkpoint_path=checkpoint,
    )
    scores, states, rows, dice = score_windows(
        model, dataset, device=str(dig(context.config, "alert.device", "cpu"))
    )
    record = {
        "status": "PASS" if trace.decreased else "FAIL",
        "training": trace.as_dict(),
        "checkpoint": str(checkpoint),
        "scored_windows": float(scores.shape[0]),
        "predicate_dice": float(dice.mean()) if dice.size else float("nan"),
        # The regenerated stream at the edition's own threshold must reproduce the edition's
        # exposure state, which is the consistency the exposure model is allowed to assume.
        "policy_edition_state_agreement": float(
            np.mean(scores >= float(dig(context.config, "alert.frozen_threshold", 0.4))) == states
        ),
        "row_index_sha256": _digest(rows),
    }
    write_json(
        context.run_dir / "alert_scores.json",
        {"rows": [int(value) for value in rows], "scores": [float(v) for v in scores]},
    )
    context.record(record)
    _LOG.info("training finished with %d steps", trace.steps)
    return EXIT_OK


def _digest(values: object) -> str:
    array = np.asarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


__all__ = ["main"]
