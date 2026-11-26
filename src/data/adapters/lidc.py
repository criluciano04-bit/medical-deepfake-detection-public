from pathlib import Path

import numpy as np
import pydicom


def load_lidc_hu(path: str | Path) -> np.ndarray:
    """Read a LIDC-IDRI DICOM slice and return a 2D float32 HU array."""

    dicom = pydicom.dcmread(path)

    pixels = dicom.pixel_array.astype(np.float32)

    if pixels.ndim != 2:
        raise ValueError(
            f"Expected a 2D CT slice, got shape {pixels.shape}"
        )

    slope = float(dicom.RescaleSlope)
    intercept = float(dicom.RescaleIntercept)

    hu = pixels * slope + intercept

    if not np.isfinite(hu).all():
        raise ValueError(f"Non-finite HU values in {path}")

    return hu.astype(np.float32)
