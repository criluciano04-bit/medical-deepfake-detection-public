
from pathlib import Path

import numpy as np
import pydicom

from src.data.paths import get_data_root
from src.data.adapters.lidc import load_lidc_hu


def test_lidc_hu(dicom_path: Path) -> None:
    dicom = pydicom.dcmread(dicom_path)

    hu = load_lidc_hu(dicom_path)

    expected = (
        dicom.pixel_array.astype(np.float32)
        * float(dicom.RescaleSlope)
        + float(dicom.RescaleIntercept)
    )

    assert hu.ndim == 2
    assert hu.dtype == np.float32
    assert np.isfinite(hu).all()
    np.testing.assert_allclose(hu, expected)

    print("Test passed!")
    print("Shape:", hu.shape)
    print("Dtype:", hu.dtype)
    print("HU range:", float(hu.min()), float(hu.max()))


from unittest.mock import patch
from pydicom.dataset import Dataset


def test_lidc_known_values() -> None:
    dicom = Dataset()
    dicom.RescaleSlope = 1
    dicom.RescaleIntercept = -1024

    pixels = np.array(
        [[0, 1024], [1124, 2024]],
        dtype=np.uint16,
    )

    with patch.object(
        type(dicom),
        "pixel_array",
        new_callable=lambda: property(lambda self: pixels),
    ):
        with patch(
            "src.data.adapters.lidc.pydicom.dcmread",
            return_value=dicom,
        ):
            result = load_lidc_hu("synthetic.dcm")

    expected = np.array(
        [[-1024, 0], [100, 1000]],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(result, expected)
    print("Known-values test passed!")



if __name__ == "__main__":
    test_lidc_known_values()

    path = get_data_root() / "lidc_selected" / "images" / "train"

    dicom_files = list(path.rglob("*.dcm"))

    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {path}")

    test_lidc_hu(dicom_files[40])
