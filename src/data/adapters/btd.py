from pathlib import Path

import numpy as np


def load_btd_image(path: str | Path) -> np.ndarray:
    """Load a BTD CT slice and apply the dataset-specific intensity clipping."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"BTD image not found: {path}"
        )

    image = np.load(path, allow_pickle=False)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D CT slice, got shape {image.shape}"
        )

    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(
            f"Expected numeric image data, got {image.dtype}"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            f"Non-finite intensity values in {path}"
        )

    image = image.astype(np.float32)

    image = np.clip(image, -700, 2000)

    return image