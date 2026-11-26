
import numpy as np

from src.data.preprocessing import (
    apply_hu_window,
    resize_image,
    normalize_image,
)


def test_preprocessing_known_values():
    image = np.array(
        [
            [-1500, -1000, 0],
            [500, 1000, 1500],
        ],
        dtype=np.float32,
    )

    windowed = apply_hu_window(image, -1000, 1000)

    expected_windowed = np.array(
        [
            [-1000, -1000, 0],
            [500, 1000, 1000],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(windowed, expected_windowed)

    normalized = normalize_image(windowed, -1000, 1000)

    expected_normalized = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.75, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(normalized, expected_normalized)

    resized = resize_image(normalized, target_size=(512, 512))

    assert resized.shape == (512, 512)
    assert resized.dtype == np.float32
    assert np.isfinite(resized).all()
    assert 0.0 <= resized.min() <= resized.max() <= 1.0

    print("Preprocessing known-values test passed!")


if __name__ == "__main__":
    test_preprocessing_known_values()
