from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.data.adapters.lidc import load_lidc_hu
from src.data.adapters.m3dsynth import (
    M3DSynthIntensityMapping,
    load_m3dsynth_image,
)
from src.data.matching.m3dsynth_lidc import find_matching_lidc_slice
from src.data.preprocessing import apply_hu_window
from src.data.paths import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")

LIDC_ROOT = Path(os.environ["LIDC_ROOT"])
M3D_ROOT = Path(os.environ["M3D_SELECTED_ROOT"])

OUTPUT_DIR = PROJECT_ROOT / "results" / "preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = {
    "Lung window": (-1350, 150),
    "Intermediate window": (-1100, 400),
    "Wide window": (-1000, 1000),
}


def clipped_percentages(image, lower, upper):
    """Compute percentages of pixels outside the HU window."""
    below = 100 * np.mean(image < lower)
    above = 100 * np.mean(image > upper)
    return below, above


def main():
    selected = pd.read_csv(
        PROJECT_ROOT
        / "results/hu_validation/select_hu_validation_slices.csv"
    )

    # Start with one validated example.
    row = selected.iloc[0]

    series_dir = (
        LIDC_ROOT
        / row["patient_id"]
        / row["SeriesInstanceUID"]
    )

    dicom_path = find_matching_lidc_slice(
        series_dir,
        int(row["slice_idx"]),
    )

    tiff_path = M3D_ROOT / row["image_relative_path"]

    mapping = M3DSynthIntensityMapping(
        PROJECT_ROOT
        / "metadata/m3dsynth/audits/"
        / "m3dsynth_volume_intensity_mapping.csv"
    )

    real_hu = load_lidc_hu(dicom_path)

    fake_hu = load_m3dsynth_image(
        tiff_path,
        patient_id=row["patient_id"],
        sdir_id=row["sdir_id"],
        mapping=mapping,
    )

    if real_hu.shape != fake_hu.shape:
        raise ValueError(
            f"Shape mismatch: {real_hu.shape} vs {fake_hu.shape}"
        )

    images = {
        "LIDC real": real_hu,
        "M3Dsynth fake": fake_hu,
    }

    fig, axes = plt.subplots(
        2,
        len(WINDOWS) + 1,
        figsize=(17, 9),
        constrained_layout=True,
    )

    original_min = min(real_hu.min(), fake_hu.min())
    original_max = max(real_hu.max(), fake_hu.max())

    stats = []

    for i, (image_name, image) in enumerate(images.items()):
        axes[i, 0].imshow(
            image,
            cmap="gray",
            vmin=original_min,
            vmax=original_max,
        )
        axes[i, 0].set_title(f"{image_name} - Original HU")
        axes[i, 0].axis("off")

        for j, (window_name, (lower, upper)) in enumerate(
            WINDOWS.items(), start=1
        ):
            windowed = apply_hu_window(image, lower, upper)

            axes[i, j].imshow(
                windowed,
                cmap="gray",
                vmin=lower,
                vmax=upper,
            )
            axes[i, j].set_title(
                f"{image_name} - {window_name}"
            )
            axes[i, j].axis("off")

            below, above = clipped_percentages(
                image, lower, upper
            )

            stats.append({
                "validation_id": row["validation_id"],
                "image": image_name,
                "window": window_name,
                "lower_hu": lower,
                "upper_hu": upper,
                "below_percent": below,
                "above_percent": above,
                "total_clipped_percent": below + above,
            })

    figure_path = (
        OUTPUT_DIR
        / f"{row['validation_id']}_window_comparison.png"
    )

    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    stats_path = (
        OUTPUT_DIR
        / f"{row['validation_id']}_window_statistics.csv"
    )

    pd.DataFrame(stats).to_csv(stats_path, index=False)

    print(f"Figure saved to: {figure_path}")
    print(f"Statistics saved to: {stats_path}")
    print(pd.DataFrame(stats).to_string(index=False))


if __name__ == "__main__":
    main()
