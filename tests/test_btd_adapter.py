from pathlib import Path

import numpy as np

from src.data.paths import get_data_root
from src.data.adapters.btd import load_btd_image


def test_btd_known_values(tmp_path: Path) -> None:
    """Verify clipping using a synthetic BTD image."""

    pixels = np.array(
        [
            [-1000, -700, -500],
            [0, 1000, 2500],
        ],
        dtype=np.float64,
    )

    image_path = tmp_path / "synthetic.npy"
    np.save(image_path, pixels)

    result = load_btd_image(image_path)

    expected = np.array(
        [
            [-700, -700, -500],
            [0, 1000, 2000],
        ],
        dtype=np.float32,
    )

    assert result.ndim == 2
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)

    print("Known-values test passed!")


def test_btd_image(image_path: Path) -> None:
    """Verify the adapter output against an original BTD file."""

    original = np.load(image_path, allow_pickle=False)

    result = load_btd_image(image_path)

    expected = np.clip(
        original.astype(np.float32),
        -700,
        2000,
    )

    assert result.ndim == 2
    assert result.dtype == np.float32
    assert np.isfinite(result).all()

    np.testing.assert_array_equal(result, expected)

    print("BTD image test passed!")
    print("Shape:", result.shape)
    print("Dtype:", result.dtype)
    print("Original range:", float(original.min()), float(original.max()))
    print("Clipped range:", float(result.min()), float(result.max()))


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_btd_known_values(Path(directory))

    images_root = get_data_root() / "btd_selected" / "images"

    for group in ["real", "fake_ctgan", "fake_sd"]:
        group_path = images_root / group

        npy_files = sorted(group_path.rglob("*.npy"))

        if not npy_files:
            raise FileNotFoundError(
                f"No NPY files found in {group_path}"
            )

        print(f"\nTesting group: {group}")
        test_btd_image(npy_files[0])