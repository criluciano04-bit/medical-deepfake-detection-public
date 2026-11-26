import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "preprocessing" / "masked_windows"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = {
    "Lung": (-1350, 150),
    "Intermediate": (-1100, 400),
    "Wide": (-1000, 1000),
}

N_SAMPLES = 5


def normalize_window(image, lower, upper):
    """Window and scale HU values to [0, 1]."""
    windowed = apply_hu_window(image, lower, upper)
    return (windowed - lower) / (upper - lower)


def analyze_pair(row, mapping):
    """Analyze one matched LIDC/M3Dsynth pair."""

    series_dir = (
        LIDC_ROOT
        / row["patient_id"]
        / row["SeriesInstanceUID"]
    )

    dicom_path = find_matching_lidc_slice(
        series_dir, int(row["slice_idx"])
    )

    real = load_lidc_hu(dicom_path)

    fake = load_m3dsynth_image(
        M3D_ROOT / row["image_relative_path"],
        patient_id=row["patient_id"],
        sdir_id=row["sdir_id"],
        mapping=mapping,
    )

    mask = tifffile.imread(
        M3D_ROOT / row["mask_relative_path"]
    )

    if mask.dtype != np.bool_:
        raise TypeError("Expected a boolean manipulation mask.")

    if real.shape != fake.shape or real.shape != mask.shape:
        raise ValueError("Image and mask shapes do not match.")

    if not mask.any():
        raise ValueError("Empty manipulation mask.")

    stats = []

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 8),
        constrained_layout=True,
    )

    for j, (name, (lower, upper)) in enumerate(WINDOWS.items()):
        real_norm = normalize_window(real, lower, upper)
        fake_norm = normalize_window(fake, lower, upper)

        axes[0, j].imshow(
            real_norm, cmap="gray", vmin=0, vmax=1
        )
        axes[1, j].imshow(
            fake_norm, cmap="gray", vmin=0, vmax=1
        )

        for i in range(2):
            axes[i, j].contour(
                mask.astype(np.uint8),
                levels=[0.5],
                colors="red",
                linewidths=0.8,
            )
            axes[i, j].axis("off")

        axes[0, j].set_title(f"Real - {name}")
        axes[1, j].set_title(f"Fake - {name}")

        real_below = 100 * np.mean(real[mask] < lower)
        real_above = 100 * np.mean(real[mask] > upper)

        fake_below = 100 * np.mean(fake[mask] < lower)
        fake_above = 100 * np.mean(fake[mask] > upper)

        mean_abs_diff = np.mean(
            np.abs(real_norm[mask] - fake_norm[mask])
        )

        stats.append({
            "validation_id": row["validation_id"],
            "generator": row["generator"],
            "manipulation_type": row["manipulation_type"],
            "window": name,
            "mask_pixels": int(mask.sum()),
            "real_clipped_percent": real_below + real_above,
            "fake_clipped_percent": fake_below + fake_above,
            "mean_abs_diff_normalized": mean_abs_diff,
        })

    fig.suptitle(
        f"{row['validation_id']} | "
        f"{row['generator']} | "
        f"{row['manipulation_type']}"
    )

    fig.savefig(
        OUTPUT_DIR / f"{row['validation_id']}_masked.png",
        dpi=200,
    )
    plt.close(fig)

    return stats


def main():
    selected = pd.read_csv(
        PROJECT_ROOT
        / "results/hu_validation/select_hu_validation_slices.csv"
    )

    mapping = M3DSynthIntensityMapping(
        PROJECT_ROOT
        / "metadata/m3dsynth/audits/"
        / "m3dsynth_volume_intensity_mapping.csv"
    )

    results = []

    for _, row in selected.head(N_SAMPLES).iterrows():
        print(f"Processing {row['validation_id']}...")

        results.extend(analyze_pair(row, mapping))

    df = pd.DataFrame(results)

    output_csv = OUTPUT_DIR / "masked_window_statistics.csv"
    df.to_csv(output_csv, index=False)

    print("\nResults:")
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_csv}")


if __name__ == "__main__":
    main()