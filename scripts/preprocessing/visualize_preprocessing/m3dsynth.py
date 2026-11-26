from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.paths import get_data_root
from src.data.adapters.m3dsynth import (
    M3DSynthIntensityMapping,
    load_m3dsynth_image,
)
from src.data.preprocessing import (
    apply_hu_window,
    resize_image,
    normalize_image,
)


LOWER_HU = -1000
UPPER_HU = 1000
TARGET_SIZE = (512, 512)


def preprocess(image: np.ndarray):
    """Apply the shared preprocessing pipeline."""

    windowed = apply_hu_window(
        image, LOWER_HU, UPPER_HU
    )

    resized = resize_image(
        windowed, TARGET_SIZE
    )

    normalized = normalize_image(
        resized, LOWER_HU, UPPER_HU
    )

    # Post-preprocessing checks
    assert normalized.shape == TARGET_SIZE
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0

    return windowed, normalized


def main():

    # Project metadata (relative paths)
    selection_csv = Path(
        "metadata/m3dsynth/m3dsynth_selected_slices.csv"
    )

    mapping_csv = Path(
        "metadata/m3dsynth/audits/"
        "m3dsynth_volume_intensity_mapping.csv"
    )

    # Dataset root from .env
    data_root = get_data_root() / "m3d_selected"

    # Output directory
    output_dir = Path(
        "results/preprocessing/visual_checks/m3dsynth"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    selection = pd.read_csv(
        selection_csv,
        dtype={
            "patient_id": "string",
            "sdir_id": "string",
        },
    )

    mapping = M3DSynthIntensityMapping(mapping_csv)

    # Only training samples
    train = selection.loc[
        selection["split"] == "train"
    ].copy()

    if train.empty:
        raise ValueError("No training samples found.")

    # Select one visibly manipulated slice per group.
    # Highest manipulated_fraction, with a stable tie-break.
    samples = (
    train.groupby(
        ["generator", "manipulation_type"],
        group_keys=False,
    )
    .sample(n=1, random_state=42)
    .sort_values(
        ["generator", "manipulation_type"]
    )
    .reset_index(drop=True)
)

    if samples.empty:
        raise ValueError("No samples selected.")

    print(f"Selected {len(samples)} training samples.")

    # Create visualization
    fig, axes = plt.subplots(
        len(samples),
        3,
        figsize=(14, 4 * len(samples)),
        squeeze=False,
    )

    for row_idx, sample in samples.iterrows():

        # Reconstruct image path from metadata
        relative_path = PurePosixPath(
            str(sample["image_relative_path"])
        )

        # Prevent access outside the training directory
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.parts[:2] != ("images", "train")
        ):
            raise ValueError(
                f"Invalid training image path: {relative_path}"
            )

        image_path = data_root.joinpath(
            *relative_path.parts
        )

        # Load TIFF and reconstruct HU
        original = load_m3dsynth_image(
            image_path=image_path,
            patient_id=str(sample["patient_id"]),
            sdir_id=str(sample["sdir_id"]),
            mapping=mapping,
        )

        # Apply shared preprocessing
        windowed, processed = preprocess(original)

        # Three stages
        images = [
            original,
            windowed,
            processed,
        ]

        titles = [
            "Original HU",
            "Windowed HU",
            "Final preprocessing",
        ]

        display_ranges = [
            (-1350, 2000),
            (LOWER_HU, UPPER_HU),
            (0, 1),
        ]

        for col_idx, (image, title, value_range) in enumerate(
            zip(images, titles, display_ranges)
        ):

            ax = axes[row_idx, col_idx]

            ax.imshow(
                image,
                cmap="gray",
                vmin=value_range[0],
                vmax=value_range[1],
            )

            ax.set_title(
                f"{sample['generator']} / "
                f"{sample['manipulation_type']}\n"
                f"{title}"
            )

            ax.axis("off")

        print(
            f"{sample['generator']} / "
            f"{sample['manipulation_type']} | "
            f"patient={sample['patient_id']} | "
            f"shape={original.shape} -> {processed.shape} | "
            f"range=[{processed.min():.3f}, "
            f"{processed.max():.3f}]"
        )

    plt.tight_layout()

    output_path = (
        output_dir / "m3dsynth_train_visual_check.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
