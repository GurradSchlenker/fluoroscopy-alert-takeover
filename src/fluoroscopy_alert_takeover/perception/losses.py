"""Training objective of the alert policy.

Ref: Sec. 2.2, p. 3 - the policy is scored by the Dice of its spatial predicate against
the reference, and by the discrimination of the alert it emits; Table 1, p. 7 - AUROC,
AUPRC and the Brier score at the frozen threshold are the reported alert-model
performance quantities.

The three terms below are the release's construction of that objective; the manuscript
prints no loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .dice import soft_dice_loss
from .spatial_predicate import PolicyOutput


@dataclass(frozen=True)
class LossBreakdown:
    """Scalar values of each objective term for logging."""

    total: Tensor
    mask: Tensor
    frame_alert: Tensor
    window_alert: Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().item()),
            "mask": float(self.mask.detach().item()),
            "frame_alert": float(self.frame_alert.detach().item()),
            "window_alert": float(self.window_alert.detach().item()),
        }


def _weighted_bce(logits: Tensor, targets: Tensor, positive_weight: float) -> Tensor:
    weight = torch.where(targets > 0.5, positive_weight, 1.0)
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=weight, reduction="none"
    )
    return elementwise.mean()


def policy_loss(
    output: PolicyOutput,
    mask_reference: Tensor,
    frame_alert_target: Tensor,
    window_alert_target: Tensor,
    mask_weight: float,
    alert_weight: float,
    window_weight: float,
    positive_weight: float,
    mask_observed: Tensor | None = None,
) -> LossBreakdown:
    """Combine the predicate Dice, the frame-level alert BCE and the window-level BCE.

    ``mask_observed`` marks the frames that carry a reference mask: only annotated frames
    enter the Dice term, which is what makes the predicated Dice comparable across sites
    with different annotation density (Table A2, p. 26).
    """
    if mask_observed is None:
        mask_term = soft_dice_loss(output.predicate_logits, mask_reference)
    else:
        if not bool(mask_observed.any()):
            raise ValueError("mask_observed selected no frames")
        # Annotated frames are selected on the flattened batch, because the Dice is a
        # set overlap over frames and a per-frame boolean cannot index a spatial axis.
        flattened_logits = output.predicate_logits.reshape(-1, *output.predicate_logits.shape[2:])
        flattened_masks = mask_reference.reshape(-1, *mask_reference.shape[2:])
        selected = mask_observed.reshape(-1) > 0.5
        mask_term = soft_dice_loss(flattened_logits[selected], flattened_masks[selected])
    frame_term = _weighted_bce(output.alert_logits, frame_alert_target, positive_weight)
    # The window target may arrive as (batch,) or (batch, 1); the pooled logits are (batch,).
    window_term = _weighted_bce(
        output.alert_logits.amax(dim=1) if output.alert_logits.dim() > 1 else output.alert_logits,
        window_alert_target.reshape(output.alert_logits.shape[0]),
        positive_weight,
    )
    total = mask_weight * mask_term + alert_weight * frame_term + window_weight * window_term
    return LossBreakdown(
        total=total, mask=mask_term, frame_alert=frame_term, window_alert=window_term
    )


__all__ = ["LossBreakdown", "policy_loss"]
