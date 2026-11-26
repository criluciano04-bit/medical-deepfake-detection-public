"""Match BTD real CT slices to their original LIDC-IDRI DICOM slices."""

from pathlib import Path

import numpy as np
import pydicom


def load_dicom_hu(dicom_path: Path) -> np.ndarray:
    """Read a DICOM image and convert its pixel values to HU."""

    ds = pydicom.dcmread(dicom_path)

    pixels = ds.pixel_array.astype(np.float64)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    return pixels * slope + intercept


def find_lidc_series(
    lidc_root: Path,
    series_uid: str,
) -> Path:
    """Find the LIDC-IDRI directory corresponding to a series UID."""

    matches = [
        path
        for path in lidc_root.glob(f"LIDC-IDRI-*/{series_uid}")
        if path.is_dir()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one LIDC series for {series_uid}, "
            f"found {len(matches)}"
        )

    return matches[0]


def get_sorted_dicom_slices(series_dir: Path) -> list[Path]:
    """Return DICOM slices sorted by increasing patient Z coordinate."""

    slices = []

    for dicom_path in series_dir.glob("*.dcm"):
        ds = pydicom.dcmread(
            dicom_path,
            stop_before_pixels=True,
            specific_tags=["ImagePositionPatient"],
        )

        z = float(ds.ImagePositionPatient[2])

        slices.append((z, dicom_path))

    slices.sort(key=lambda item: item[0])

    return [path for _, path in slices]


def match_btd_to_lidc(
    btd_path: str | Path,
    lidc_root: str | Path,
) -> Path | None:
    """
    Find the original LIDC DICOM corresponding to a BTD real slice.

    The BTD slice number is interpreted as a zero-based index
    into the LIDC series ordered by increasing Z coordinate.

    Returns:
        Path to the matching DICOM, if exact pixel equality is confirmed.
        None otherwise.
    """

    btd_path = Path(btd_path)
    lidc_root = Path(lidc_root)

    # BTD filename: <series_uid>__<slice_number>.npy
    series_uid, slice_str = btd_path.stem.rsplit("__", 1)
    slice_number = int(slice_str)

    # Locate the original LIDC series.
    series_dir = find_lidc_series(lidc_root, series_uid)

    # Order its slices anatomically.
    dicom_slices = get_sorted_dicom_slices(series_dir)

    # Check that the requested index exists.
    if not 0 <= slice_number < len(dicom_slices):
        return None

    # Select the corresponding DICOM.
    dicom_path = dicom_slices[slice_number]

    # Compare original pixel values in HU.
    btd_image = np.load(btd_path)
    lidc_image = load_dicom_hu(dicom_path)

    if np.array_equal(btd_image, lidc_image):
        return dicom_path

    return None
