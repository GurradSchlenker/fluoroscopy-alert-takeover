"""The alert policy: a spatial predicate over fluoroscopic frames plus a window score.

Ref: Sec. 2.2, p. 3 (the spatial predicate and its Dice); Sec. 4.6, p. 19 - the detection
policy is built from the navigation record and the fluoroscopic stream; Sec. 4.3, p. 15 -
the alert state used for the contrast is read from the frozen policy edition, and the
scored stream of Algorithm 5 is regenerated from the same edition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .encoder import build_encoder


@dataclass(frozen=True)
class AlertPolicyConfig:
    """Architecture and weighting of the alert policy.

    The manuscript prints the policy's Dice, AUROC, latency and memory, not its
    architecture or loss weights; every field here is a release engineering default.
    """

    route: str = "spatial_predicate"
    width: int = 32
    depth: int = 4
    in_channels: int = 1
    mask_weight: float = 1.0
    alert_weight: float = 1.0
    window_weight: float = 0.5
    positive_weight: float = 1.0


@dataclass(frozen=True)
class PolicyOutput:
    """Per-frame predicate logits and per-frame alert logits."""

    predicate_logits: Tensor
    alert_logits: Tensor

    def window_scores(self) -> Tensor:
        """Window score of each item: the largest frame-level alert probability.

        The window score is what Algorithm 5 thresholds, so it is the maximum over the
        frames of the analysis period.
        """
        return torch.sigmoid(self.alert_logits).amax(dim=1)


class AlertPolicyNet(nn.Module):
    """Encoder plus a dense predicate head and a pooled alert head."""

    def __init__(self, config: AlertPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or AlertPolicyConfig()
        self.encoder, channels = build_encoder(
            self.config.route,
            width=self.config.width,
            depth=self.config.depth,
            in_channels=self.config.in_channels,
        )
        self.predicate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.alert_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )

    def forward(self, frames: Tensor) -> PolicyOutput:
        """Score a batch of frame sequences.

        ``frames`` has shape ``(batch, time, channels, height, width)``. The predicate is
        upsampled back to the frame grid so the Dice of Sec. 2.2 is computed at frame
        resolution.
        """
        if frames.dim() != 5:
            raise ValueError("frames must be (batch, time, channels, height, width)")
        batch, time = frames.shape[0], frames.shape[1]
        flat = frames.reshape(batch * time, *frames.shape[2:])
        features = self.encoder(flat)
        predicate = self.predicate(features)
        upsampled = nn.functional.interpolate(
            predicate, size=frames.shape[-2:], mode="bilinear", align_corners=False
        )
        pooled = features.mean(dim=(2, 3))
        alert = self.alert_head(pooled)
        return PolicyOutput(
            predicate_logits=upsampled.reshape(batch, time, 1, *frames.shape[-2:]),
            alert_logits=alert.reshape(batch, time),
        )


def window_alert_stream(
    scores: Tensor, threshold: float, abstention_low: float = 0.0, abstention_high: float = 0.0
) -> Tensor:
    """Regenerate the alert stream from a frozen edition at one operating threshold.

    Ref: Algorithm 5, step 2, p. 20. Windows inside the abstention band are withheld
    regardless of the threshold, which is what makes the band policy a constraint rather
    than a re-calibration.
    """
    above = scores >= threshold
    if abstention_high > abstention_low:
        inside_band = (scores > abstention_low) & (scores < abstention_high)
        above = above & ~inside_band
    return above


__all__ = [
    "AlertPolicyConfig",
    "AlertPolicyNet",
    "PolicyOutput",
    "window_alert_stream",
]
