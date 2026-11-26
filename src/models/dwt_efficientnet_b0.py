"""Two-branch spatial/frequency CT deepfake classifier.

Based on the *architecture* in Pradeepan & Gladston Raj (CICN 2024),
Fig. 3 / section III.B. Unspecified paper details are explicit choices:
one-level orthonormal Haar DWT, all four subbands, conv 3x3, max-pool 2,
and adaptive global average pooling of the frequency branch.

Input: 3-channel CT tensor, produced by prepare_imagenet_input() from
one grayscale .npy already windowed and min-max scaled to [0, 1].
Spatial branch receives ImageNet-normalized channels unchanged.
Frequency branch reconstructs the original [0,1] grayscale CT from the
first channel (inverse ImageNet normalization), then applies the Haar DWT.
No per-image normalization or precomputed wavelet files are introduced.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class FixedHaarDWT2D(nn.Module):
    """Single-level orthonormal 2D Haar DWT, (B,1,H,W)->(B,4,H/2,W/2).

    Filters are fixed buffers (not learnable). Channel order: LL, LH, HL, HH.
    For even H,W, stride-2 valid convolution is a critically sampled transform.
    """

    def __init__(self) -> None:
        super().__init__()
        low = torch.tensor([1.0, 1.0], dtype=torch.float32) / math.sqrt(2.0)
        high = torch.tensor([1.0, -1.0], dtype=torch.float32) / math.sqrt(2.0)
        filters = torch.stack([
            torch.outer(low, low),
            torch.outer(low, high),
            torch.outer(high, low),
            torch.outer(high, high),
        ]).unsqueeze(1)
        self.register_buffer("filters", filters, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 1:
            raise ValueError(f"DWT expects (B, 1, H, W), got {tuple(x.shape)}")
        if x.size(-2) % 2 or x.size(-1) % 2:
            raise ValueError("Haar DWT requires even height and width")
        return F.conv2d(x, self.filters.to(dtype=x.dtype), stride=2)


class DWTEfficientNetB0(nn.Module):
    """EfficientNet-B0 spatial features + DWT/2-conv frequency features.

    Paper: spatial EfficientNet-B0 classifier replaced with identity;
    frequency branch has conv(32), conv(64), BN, ReLU, pooling;
    concatenate both vectors, Linear(512)->BN->ReLU->Dropout(.5)->Linear(2).
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.spatial_branch = models.efficientnet_b0(weights=weights)
        spatial_dim = self.spatial_branch.classifier[1].in_features
        self.spatial_branch.classifier = nn.Identity()

        self.dwt = FixedHaarDWT2D()
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError(f"Expected ImageNet-normalized (B,3,H,W), got {tuple(images.shape)}")
        spatial = self.spatial_branch(images)
        # The original CT grayscale was copied into RGB and then normalized
        # separately per ImageNet channel; invert channel 0 (R).
        grayscale_01 = images[:, :1] * 0.229 + 0.485
        wavelet_bands = self.dwt(grayscale_01)
        frequency = self.frequency_branch(wavelet_bands)
        fused = torch.cat((spatial, frequency), dim=1)
        return self.classifier(fused)
