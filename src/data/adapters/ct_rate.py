from pathlib import Path

import numpy as np


def load_ct_rate_image(path: str | Path) -> np.ndarray:
    """Load a selected CT-RATE Real or Generated slice, already expressed in HU.

    No HU conversion, clipping, windowing, resizing, or normalization is applied.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"CT-RATE image not found: {path}")

    image = np.load(path, allow_pickle=False)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D CT slice, got shape {image.shape}")

    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"Expected numeric image data, got {image.dtype}")

    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite intensity values in {path}")

    return image.astype(np.float32, copy=False)
