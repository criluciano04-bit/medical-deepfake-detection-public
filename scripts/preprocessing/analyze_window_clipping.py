from pathlib import Path
import os

import numpy as np
import pandas as pd
import tifffile

from src.data.adapters.lidc import load_lidc_hu
from src.data.adapters.m3dsynth import (
    M3DSynthIntensityMapping,
    load_m3dsynth_image,
)
from src.data.matching.m3dsynth_lidc import find_matching_lidc_slice
from src.data.preprocessing import apply_hu_window
from src.data.paths import PROJECT_ROOT


WINDOWS = {
    "Lung": (-1350, 150),
    "Intermediate": (-1100, 400),
    "Wide": (-1000, 1000),
}

SELECTION_CSV = (
    PROJECT_ROOT
    / "results/hu_validation/select_hu_validation_slices.csv"
)

MAPPING_CSV = (
    PROJECT_ROOT
    / "metadata/m3dsynth/audits/m3dsynth_volume_intensity_mapping.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results/preprocessing/window_clipping"


def clipping_percentages(image, mask, lower, upper):
    pixels = image[mask]
    if pixels.size == 0:
        raise ValueError("Empty analysis region.")

    below = np.mean(pixels < lower) * 100
    above = np.mean(pixels > upper) * 100

    return below, above, below + above


def difference_statistics(real, fake, mask, lower, upper):
    real_pixels = real[mask]
    fake_pixels = fake[mask]

    original_diff = np.abs(real_pixels - fake_pixels)

    real_windowed = apply_hu_window(real_pixels, lower, upper)
    fake_windowed = apply_hu_window(fake_pixels, lower, upper)

    windowed_diff = np.abs(real_windowed - fake_windowed)

    # Loss of absolute real/fake difference caused by clipping.
    lost_diff = original_diff - windowed_diff

    return {
        "original_mad_hu": float(original_diff.mean()),
        "windowed_mad_hu": float(windowed_diff.mean()),
        "lost_mad_hu": float(lost_diff.mean()),
        "lost_mad_percent": (
            float(lost_diff.sum() / original_diff.sum() * 100)
            if original_diff.sum() > 0
            else 0.0
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(SELECTION_CSV)

    if len(selection) != 30:
        raise ValueError(
            f"Expected 30 validated pairs, found {len(selection)}."
        )

    mapping = M3DSynthIntensityMapping(MAPPING_CSV)

    lidc_root = Path(os.environ["LIDC_ROOT"])
    m3d_root = Path(os.environ["M3D_SELECTED_ROOT"])

    rows = []

    for _, item in selection.iterrows():
        patient_id = str(item["patient_id"])
        sdir_id = int(item["sdir_id"])
        slice_idx = int(item["slice_idx"])

        series_dir = (
            lidc_root / patient_id / str(item["SeriesInstanceUID"])
        )

        real_path = find_matching_lidc_slice(series_dir, slice_idx)

        fake_path = m3d_root / str(item["image_relative_path"])
        mask_path = m3d_root / str(item["mask_relative_path"])

        real = load_lidc_hu(real_path)
        fake = load_m3dsynth_image(
            fake_path,
            patient_id,
            sdir_id,
            mapping,
        )

        manipulation_mask = tifffile.imread(mask_path).astype(bool)

        if real.shape != fake.shape or real.shape != manipulation_mask.shape:
            raise ValueError(
                f"Shape mismatch for {item['validation_id']}: "
                f"real={real.shape}, fake={fake.shape}, "
                f"mask={manipulation_mask.shape}"
            )

        regions = {
            "Full": np.ones(real.shape, dtype=bool),
            "Mask": manipulation_mask,
        }

        for window_name, (lower, upper) in WINDOWS.items():
            for region_name, region_mask in regions.items():
                real_below, real_above, real_total = clipping_percentages(
                    real, region_mask, lower, upper
                )
                fake_below, fake_above, fake_total = clipping_percentages(
                    fake, region_mask, lower, upper
                )

                differences = difference_statistics(
                    real, fake, region_mask, lower, upper
                )

                rows.append({
                    "validation_id": item["validation_id"],
                    "patient_id": patient_id,
                    "generator": item["generator"],
                    "manipulation_type": item["manipulation_type"],
                    "window": window_name,
                    "lower_hu": lower,
                    "upper_hu": upper,
                    "region": region_name,
                    "region_pixels": int(region_mask.sum()),
                    "real_below_percent": real_below,
                    "real_above_percent": real_above,
                    "real_total_clipped_percent": real_total,
                    "fake_below_percent": fake_below,
                    "fake_above_percent": fake_above,
                    "fake_total_clipped_percent": fake_total,
                    **differences,
                })

        print(f"Processed {item['validation_id']}")

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_DIR / "window_clipping_per_pair.csv",
        index=False,
    )

    metrics = [
        "real_below_percent",
        "real_above_percent",
        "real_total_clipped_percent",
        "fake_below_percent",
        "fake_above_percent",
        "fake_total_clipped_percent",
        "original_mad_hu",
        "windowed_mad_hu",
        "lost_mad_hu",
        "lost_mad_percent",
    ]

    summary = (
        results.groupby(["window", "region"])[metrics]
        .agg([
            "mean",
            "median",
            "std",
            "min",
            "max",
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
        ])
    )

    summary.columns = [
        f"{metric}_{stat if isinstance(stat, str) else 'quantile'}"
        for metric, stat in summary.columns
    ]

    summary.to_csv(
        OUTPUT_DIR / "window_clipping_summary.csv"
    )

    print("\nAnalysis completed.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()