"""Training and scoring of the alert policy.

Ref: Table A1, p. 25 - the deployment configuration is a single mid-range accelerator per
site at batch size 1; the manuscript prints the policy's latency, throughput and memory,
not its optimiser, schedule or epoch count, so every value in
:class:`AlertTrainingConfig` is a release engineering default.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..utils.logging import get_logger
from ..utils.types import BoolArray, FloatArray, IntArray
from .checkpoint import save_checkpoint
from .data import FluoroscopyWindowDataset, WindowSample, collate
from .dice import dice_coefficient
from .losses import LossBreakdown, policy_loss
from .spatial_predicate import AlertPolicyNet

_LOG = get_logger("perception.training")

_PRECISIONS = ("fp32", "fp16", "bf16")


@dataclass(frozen=True)
class AlertTrainingConfig:
    """Optimisation settings of the alert policy."""

    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    precision: str = "fp32"
    seed: int = 20260101
    mask_weight: float = 1.0
    alert_weight: float = 1.0
    window_weight: float = 0.5
    positive_weight: float = 1.0
    checkpoint_every: int = 10

    def __post_init__(self) -> None:
        if self.precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {_PRECISIONS}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")


@dataclass
class TrainingTrace:
    """Per-epoch loss history and the run's step count."""

    epoch: list[int] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    mask: list[float] = field(default_factory=list)
    frame_alert: list[float] = field(default_factory=list)
    window_alert: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0

    @property
    def decreased(self) -> bool:
        """Whether the final epoch loss is below the first."""
        if len(self.total) < 2:
            return False
        return self.total[-1] < self.total[0]

    def as_dict(self) -> dict[str, object]:
        return {
            "epoch": list(self.epoch),
            "total": list(self.total),
            "mask": list(self.mask),
            "frame_alert": list(self.frame_alert),
            "window_alert": list(self.window_alert),
            "learning_rate": list(self.learning_rate),
            "steps": self.steps,
            "seconds": self.seconds,
            "decreased": self.decreased,
        }


def _autocast(device: torch.device, precision: str) -> torch.autocast:
    dtype = torch.float32
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    enabled = precision != "fp32" and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def build_optimizer(model: nn.Module, config: AlertTrainingConfig) -> torch.optim.Optimizer:
    """AdamW with the configured decay; normalisation and bias parameters are undecayed."""
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: AlertTrainingConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up into a cosine decay, on the step counter."""
    warmup_steps = max(1, config.warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, config.epochs * steps_per_epoch)

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _step_loss(
    model: AlertPolicyNet, batch: WindowSample, config: AlertTrainingConfig
) -> LossBreakdown:
    output = model(batch["frames"])
    return policy_loss(
        output=output,
        mask_reference=batch["mask"],
        frame_alert_target=batch["frame_alert"],
        window_alert_target=batch["window_alert"],
        mask_weight=config.mask_weight,
        alert_weight=config.alert_weight,
        window_weight=config.window_weight,
        positive_weight=config.positive_weight,
        mask_observed=batch["mask_observed"] > 0.5,
    )


def train_alert_policy(
    model: AlertPolicyNet,
    dataset: FluoroscopyWindowDataset,
    config: AlertTrainingConfig,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
) -> TrainingTrace:
    """Train the policy for ``config.epochs`` and return the loss history."""
    target = torch.device(device)
    model.to(target)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, max(1, len(loader)))
    trace = TrainingTrace()
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        running: dict[str, float] = dict.fromkeys(
            ("total", "mask", "frame_alert", "window_alert"), 0.0
        )
        for batch in loader:
            prepared = WindowSample(
                frames=batch["frames"].to(target),
                mask=batch["mask"].to(target),
                frame_alert=batch["frame_alert"].to(target),
                window_alert=batch["window_alert"].to(target),
                mask_observed=batch["mask_observed"].to(target),
                site=batch["site"].to(target),
                row=batch["row"].to(target),
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(target, config.precision):
                breakdown = _step_loss(model, prepared, config)
            breakdown.total.backward()  # type: ignore[no-untyped-call]
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            trace.steps += 1
            for key, value in breakdown.as_dict().items():
                running[key] += value
        divisor = float(max(1, len(loader)))
        trace.epoch.append(epoch)
        trace.total.append(running["total"] / divisor)
        trace.mask.append(running["mask"] / divisor)
        trace.frame_alert.append(running["frame_alert"] / divisor)
        trace.window_alert.append(running["window_alert"] / divisor)
        trace.learning_rate.append(float(optimizer.param_groups[0]["lr"]))
        _LOG.info("epoch %d/%d loss %.4f", epoch + 1, config.epochs, trace.total[-1])
        if checkpoint_path is not None and (epoch + 1) % config.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                config.seed,
                extra={
                    "config": {item.name: getattr(config, item.name) for item in fields(config)}
                },
            )
    trace.seconds = time.perf_counter() - started
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config.epochs - 1,
            config.seed,
            extra={"config": {item.name: getattr(config, item.name) for item in fields(config)}},
        )
    return trace


@torch.no_grad()
def score_windows(
    model: AlertPolicyNet,
    dataset: FluoroscopyWindowDataset,
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[FloatArray, BoolArray, IntArray, FloatArray]:
    """Score every window: policy score, frozen-edition state, row index and predicate Dice.

    The returned score is the window score that Algorithm 5 thresholds; the state is the
    frozen edition's exposure, so a caller can check that regenerating the stream at the
    edition's own threshold reproduces it (Sec. 4.3, p. 15).
    """
    target = torch.device(device)
    model.to(target)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    scores: list[FloatArray] = []
    states: list[BoolArray] = []
    rows: list[IntArray] = []
    dice: list[float] = []
    for batch in loader:
        prepared = WindowSample(
            frames=batch["frames"].to(target),
            mask=batch["mask"].to(target),
            frame_alert=batch["frame_alert"].to(target),
            window_alert=batch["window_alert"].to(target),
            mask_observed=batch["mask_observed"].to(target),
            site=batch["site"].to(target),
            row=batch["row"].to(target),
        )
        output = model(prepared["frames"])
        scores.append(output.window_scores().detach().cpu().numpy().astype(np.float64))
        states.append(
            (prepared["window_alert"].reshape(-1) > 0.5).detach().cpu().numpy().astype(np.bool_)
        )
        rows.append(prepared["row"].reshape(-1).detach().cpu().numpy().astype(np.int64))
        observed = (prepared["mask_observed"] > 0.5).reshape(-1)
        if bool(observed.any()):
            logits = output.predicate_logits.reshape(-1, *output.predicate_logits.shape[2:])
            targets = prepared["mask"].reshape(-1, *prepared["mask"].shape[2:])
            dice.append(dice_coefficient(logits[observed], targets[observed]))
    return (
        np.concatenate(scores),
        np.concatenate(states),
        np.concatenate(rows),
        np.asarray(dice, dtype=np.float64),
    )


__all__ = [
    "AlertTrainingConfig",
    "TrainingTrace",
    "build_optimizer",
    "build_scheduler",
    "score_windows",
    "train_alert_policy",
]
