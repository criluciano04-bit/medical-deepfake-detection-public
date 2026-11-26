
"""Inspect spatial HU residuals for HU-027 and HU-030."""

from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.data.adapters.lidc import load_lidc_hu
from src.data.adapters.m3dsynth import (
    M3DSynthIntensityMapping,
    load_m3dsynth_image,
)

SELECTED = REPO / "results/hu_validation/m3dsynth/select_hu_validation_slices.csv"
COMPARISON = REPO / "results/hu_validation/m3dsynth/validate_m3dsynth_hu.csv"
MAPPING = (
    REPO
    / "metadata/m3dsynth/audits/m3dsynth_volume_intensity_mapping.csv"
)
OUTPUT_DIR = REPO / "results/hu_validation/m3dsynth/residual_inspection"


def load_env():
    env_path = REPO / ".env"

    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )


def inspect(validation_id, selected, comparison, mapping, lidc_root, m3d_root):
    row = selected.loc[selected.validation_id == validation_id].iloc[0]
    result = comparison.loc[
        comparison.validation_id == validation_id
    ].iloc[0]

    if result.status != "measured":
        raise ValueError(f"{validation_id} was not successfully validated")

    dicom_path = (
        lidc_root
        / row.patient_id
        / row.SeriesInstanceUID
        / result.matched_dicom_filename
    )

    image_path = m3d_root / row.image_relative_path
    mask_path = m3d_root / row.mask_relative_path

    lidc_hu = load_lidc_hu(dicom_path)
    m3d_hu = load_m3dsynth_image(
        image_path,
        row.patient_id,
        row.sdir_id,
        mapping,
    )

    mask = tifffile.imread(mask_path) != 0

    if lidc_hu.shape != m3d_hu.shape or mask.shape != lidc_hu.shape:
        raise ValueError("Image/mask shape mismatch")

    residual = np.abs(
        m3d_hu.astype(np.float64)
        - lidc_hu.astype(np.float64)
    )

    outside = ~mask
    changed = outside & (residual > 0)

    # For pixels outside the mask, distance to nearest masked pixel.
    distance = distance_transform_edt(outside)

    print(f"\n{validation_id} | {row.patient_id}")
    print(f"Generator: {row.generator}")
    print(f"Different pixels outside mask: {changed.sum():,}")
    print(f"Maximum outside-mask error: {residual[outside].max():.3f} HU")

    if changed.any():
        for radius in [1, 3, 5, 10]:
            near = changed & (distance <= radius)
            percentage = 100 * near.sum() / changed.sum()

            print(
                f"Within {radius:2d} px of mask: "
                f"{near.sum():,} / {changed.sum():,} "
                f"({percentage:.2f}%)"
            )

        print(
            "Median distance of changed pixels: "
            f"{np.median(distance[changed]):.2f} px"
        )

    # Display CT in a lung window.
    vmin, vmax = -1000, 400

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(
        lidc_hu,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Original LIDC (HU)")

    axes[1].imshow(
        lidc_hu,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].contour(
        mask.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=1,
    )
    axes[1].set_title("Manipulation mask (red outline)")

    error_map = np.where(outside, residual, np.nan)

    im = axes[2].imshow(
        error_map,
        cmap="inferno",
        vmin=0,
        vmax=max(1, np.percentile(residual[outside], 99.9)),
    )
    axes[2].contour(
        mask.astype(float),
        levels=[0.5],
        colors="cyan",
        linewidths=1,
    )
    axes[2].set_title("Absolute HU difference outside mask")

    fig.colorbar(im, ax=axes[2], label="Absolute error (HU)")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{validation_id} â€” {row.patient_id} â€” {row.generator}",
        fontsize=13,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{validation_id}_residuals.png"

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    load_env()

    lidc_root = Path(os.environ["LIDC_ROOT"])
    m3d_root = Path(os.environ["M3D_SELECTED_ROOT"])

    selected = pd.read_csv(SELECTED, dtype={"sdir_id": str})
    comparison = pd.read_csv(COMPARISON)

    mapping = M3DSynthIntensityMapping(MAPPING)

    for validation_id in ["HU-027", "HU-030"]:
        inspect(
            validation_id,
            selected,
            comparison,
            mapping,
            lidc_root,
            m3d_root,
        )


if __name__ == "__main__":
    main()


