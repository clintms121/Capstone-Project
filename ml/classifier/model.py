"""CNN classifier over spectrograms: glitch classes + real-signal class.

Small ResNet-style stack for single-channel 140x170 Q-scans: a strided stem,
four residual stages (each halving resolution and doubling width), global
average pooling, and a linear head. ~2.8M parameters -- sized for ~9k
training images; expand capacity only if validation shows underfitting.
"""

from __future__ import annotations

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class SpectrogramClassifier(nn.Module):
    def __init__(self, num_classes: int, widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], 5, 2, 2, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        in_ch = widths[0]
        for i, out_ch in enumerate(widths):
            stages.append(_ResidualBlock(in_ch, out_ch, stride=1 if i == 0 else 2))
            stages.append(_ResidualBlock(out_ch, out_ch, stride=1))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, H, W) in [0, 1]. Returns (batch, num_classes) logits."""
        return self.head(self.stages(self.stem(x)))


def build_model(num_classes: int) -> SpectrogramClassifier:
    return SpectrogramClassifier(num_classes)
