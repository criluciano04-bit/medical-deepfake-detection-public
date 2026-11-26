from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.paths import get_data_root
from src.data.adapters.lidc import load_lidc_hu
from src.data.preprocessing import (
    apply_hu_window,
    resize_image,
    normalize_image,
)

LOWER_HU = -1000
UPPER_HU = 1000
TARGET_SIZE = (512, 512)


def preprocess(image: np.ndarray) -> np.ndarray:
    windowed = apply_hu_window(image, LOWER_HU, UPPER_HU)
    resized = resize_image(windowed, TARGET_SIZE)
    normalized = normalize_image(resized, LOWER_HU, UPPER_HU)

    assert normalized.shape == TARGET_SIZE
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0

    return normalized


def main():
    train_root = (
        get_data_root() / "lidc_selected" / "images" / "train"
    )

    output_dir = Path("results/preprocessing/visual_checks/lidc")
    output_dir.mkdir(parents=True, exist_ok=True)

    # One representative slice from each of four training patients.
    patients = sorted(
        path for path in train_root.iterdir()
        if path.is_dir()
)

    if len(patients) < 4:
        raise ValueError("Fewer than four training patients found.")

    indices = np.linspace(0, len(patients) - 1, 4, dtype=int)
    patients = [patients[i] for i in indices]

    fig, axes = plt.subplots(4, 3, figsize=(14, 18))

    for row, patient_dir in enumerate(patients):
        files = sorted(patient_dir.rglob("*.dcm"))

        if not files:
            raise FileNotFoundError(
                f"No DICOM images found in {patient_dir}"
            )

        path = files[len(files) // 2]
        original = load_lidc_hu(path)

        windowed = apply_hu_window(
            original, LOWER_HU, UPPER_HU
        )
        processed = preprocess(original)

        axes[row, 0].imshow(
            original, cmap="gray", vmin=-1350, vmax=2000
        )
        axes[row, 0].set_title(
            f"{patient_dir.name} — Original HU"
        )

        axes[row, 1].imshow(
            windowed, cmap="gray",
            vmin=LOWER_HU, vmax=UPPER_HU
        )
        axes[row, 1].set_title("Windowed HU")

        axes[row, 2].imshow(
            processed, cmap="gray", vmin=0, vmax=1
        )
        axes[row, 2].set_title("Final preprocessing")

        for ax in axes[row]:
            ax.axis("off")

        print(
            f"{patient_dir.name}: "
            f"original={original.shape}, "
            f"processed={processed.shape}, "
            f"range=[{processed.min():.3f}, "
            f"{processed.max():.3f}]"
        )

    plt.tight_layout()

    output_path = output_dir / "lidc_train_visual_check.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
