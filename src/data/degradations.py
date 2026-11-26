"""Controlled degradations for preprocessed grayscale CT images.

Inputs: float32 tensors shaped (B, 1, H, W), normalized to [0, 1].
The functions do not modify their input tensors.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F


def _check_images(images: torch.Tensor) -> None:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Expected images with shape (B, 1, H, W)")
    if images.shape[0] == 0 or min(images.shape[-2:]) < 2:
        raise ValueError("Batch must be nonempty and image dimensions >= 2")
    if images.dtype != torch.float32:
        raise ValueError("Expected float32 preprocessed images")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("Images contain non-finite values")
    if bool((images < -1e-6).any()) or bool((images > 1 + 1e-6).any()):
        raise ValueError("Images must be normalized to [0, 1]")


def _check_sigma(sigma: float) -> float:
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite number")
    return sigma


def stable_image_seed(sample_id: str, global_seed: int = 42) -> int:
    """Derive a stable 63-bit seed from a sample ID, independent of Python hash()."""
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty string")
    if not isinstance(global_seed, int) or global_seed < 0:
        raise ValueError("global_seed must be a nonnegative integer")
    payload = f"{global_seed}:{len(sample_id)}:{sample_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little") & ((1 << 63) - 1)


def gaussian_noise(
    images: torch.Tensor,
    sigma: float,
    sample_ids: Sequence[str],
    global_seed: int = 42,
) -> torch.Tensor:
    """Add per-image repeatable N(0,sigma^2) noise and clip to [0, 1].

    A sample ID always receives the same standard-normal noise field for a
    given global seed, regardless of batch composition, model, or sigma.
    Sampling uses a Generator on the same device as images; cross-device
    bit-for-bit equality is not guaranteed.
    """
    _check_images(images)
    sigma = _check_sigma(sigma)
    if len(sample_ids) != images.shape[0]:
        raise ValueError("sample_ids length must match batch size")

    noise = torch.empty_like(images)
    for i, sample_id in enumerate(sample_ids):
        rng = torch.Generator(device=images.device)
        rng.manual_seed(stable_image_seed(sample_id, global_seed))
        noise[i] = torch.randn(
            images[i].shape,
            generator=rng,
            device=images.device,
            dtype=images.dtype,
        )
    return (images + sigma * noise).clamp(0.0, 1.0)


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur with radius ceil(3*sigma) and reflect padding."""
    _check_images(images)
    sigma = _check_sigma(sigma)
    radius = math.ceil(3 * sigma)
    height, width = images.shape[-2:]
    if radius >= min(height, width):
        raise ValueError("Image dimensions are too small for reflect padding")

    coords = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
    weights = torch.exp(-0.5 * (coords / sigma).square())
    weights = weights / weights.sum()
    horizontal = weights.view(1, 1, 1, -1)
    vertical = weights.view(1, 1, -1, 1)

    padded = F.pad(images, (radius, radius, 0, 0), mode="reflect")
    blurred = F.conv2d(padded, horizontal)
    padded = F.pad(blurred, (0, 0, radius, radius), mode="reflect")
    blurred = F.conv2d(padded, vertical)
    return blurred.clamp(0.0, 1.0)


def apply_degradation(
    images: torch.Tensor,
    degradation: str,
    sigma: float,
    sample_ids: Sequence[str] | None = None,
    global_seed: int = 42,
) -> torch.Tensor:
    """Apply one degradation to a batch of preprocessed grayscale images."""
    if degradation == "noise":
        if sample_ids is None:
            raise ValueError("sample_ids are required for reproducible Gaussian noise")
        return gaussian_noise(images, sigma, sample_ids, global_seed)
    if degradation == "blur":
        return gaussian_blur(images, sigma)
    raise ValueError("degradation must be 'noise' or 'blur'")
