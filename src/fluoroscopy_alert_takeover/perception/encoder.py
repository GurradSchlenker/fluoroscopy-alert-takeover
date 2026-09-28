"""Fluoroscopic encoder and its linear-time state-space alternative.

Ref: Sec. 4.6, p. 19 (the detection policy reads the navigation record and the
fluoroscopic stream); Table A1, p. 25 - two perception routes are reported, the primary
spatial-predicate route and a state-space alternative "retained for its linear-time
scaling".
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Strided convolution, group normalisation and a gated activation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, inputs: Tensor) -> Tensor:
        convolved: Tensor = self.conv(inputs)
        normalised: Tensor = self.norm(convolved)
        activated: Tensor = self.act(normalised)
        return activated


class FluoroscopyEncoder(nn.Module):
    """Small convolutional backbone over single-channel fluoroscopic frames.

    The manuscript reports the primary route's deployment footprint (Table A1, p. 25) and
    its Dice and AUROC, not its architecture. Width, depth and downsampling factor are
    release engineering choices.
    """

    def __init__(self, width: int = 32, depth: int = 4, in_channels: int = 1) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("encoder depth must be at least 2")
        layers: list[nn.Module] = []
        channels = in_channels
        for level in range(depth):
            out_channels = width * (2 ** min(level, 2))
            layers.append(ConvBlock(channels, out_channels, stride=2 if level < 2 else 1))
            channels = out_channels
        self.blocks = nn.ModuleList(layers)
        self.out_channels = channels
        self.stride = 4

    def forward(self, frames: Tensor) -> Tensor:
        features = frames
        for block in self.blocks:
            features = block(features)
        return features


class StateSpaceBlock(nn.Module):
    """Selective scan over the flattened spatial axis, evaluated in linear time.

    Input-dependent decay, write and read gates make the block a selective state-space
    mixer; the recurrence is stepped explicitly, which is what the alternative route's
    latency in Table A1 measures.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.decay = nn.Linear(channels, channels)
        self.write = nn.Linear(channels, channels)
        self.read = nn.Linear(channels, channels)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        sequence = features.flatten(2).transpose(1, 2)
        decay = torch.sigmoid(self.decay(sequence))
        written = torch.tanh(self.write(sequence))
        read = torch.sigmoid(self.read(sequence))
        state = torch.zeros(batch, channels, dtype=features.dtype, device=features.device)
        outputs: list[Tensor] = []
        for step in range(sequence.shape[1]):
            state = decay[:, step] * state + (1.0 - decay[:, step]) * written[:, step]
            outputs.append(read[:, step] * state)
        stacked = torch.stack(outputs, dim=1)
        mixed = stacked.transpose(1, 2).reshape(batch, channels, height, width)
        combined: Tensor = self.norm(mixed + features)
        return combined


class StateSpaceEncoder(nn.Module):
    """The alternative perception route: a shallow stem plus one selective scan."""

    def __init__(self, width: int = 32, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, width, stride=2)
        self.scan = StateSpaceBlock(width)
        self.out_channels = width
        self.stride = 2

    def forward(self, frames: Tensor) -> Tensor:
        stemmed = self.stem(frames)
        scanned: Tensor = self.scan(stemmed)
        return scanned


def build_encoder(
    route: str, width: int = 32, depth: int = 4, in_channels: int = 1
) -> tuple[nn.Module, int]:
    """Instantiate one of the two reported perception routes, with its output width."""
    if route == "spatial_predicate":
        encoder = FluoroscopyEncoder(width=width, depth=depth, in_channels=in_channels)
        return encoder, encoder.out_channels
    if route == "state_space":
        alternative = StateSpaceEncoder(width=width, in_channels=in_channels)
        return alternative, alternative.out_channels
    raise ValueError(f"unknown perception route: {route}")


__all__ = [
    "ConvBlock",
    "FluoroscopyEncoder",
    "StateSpaceBlock",
    "StateSpaceEncoder",
    "build_encoder",
]
