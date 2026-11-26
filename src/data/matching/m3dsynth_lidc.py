from pathlib import Path
import pydicom

def find_matching_lidc_slice(
    series_dir: str | Path,
    slice_idx: int,
) -> Path:
    """
    Find the original LIDC-IDRI DICOM slice corresponding
    to an M3Dsynth slice index.

    Assumes M3Dsynth uses descending anatomical Z order.
    """

    series_dir = Path(series_dir)

    if not series_dir.is_dir():
        raise FileNotFoundError(
            f"DICOM series directory not found: {series_dir}"
        )

    slices = []

    for path in series_dir.glob("*.dcm"):
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=["ImagePositionPatient"],
        )

        slices.append(
            (float(ds.ImagePositionPatient[2]), path)
        )

    if not slices:
        raise ValueError(
            f"No DICOM slices found in {series_dir}"
        )

    slices.sort(key=lambda item: item[0], reverse=True)

    if not 0 <= slice_idx < len(slices):
        raise IndexError(
            f"slice_idx={slice_idx} outside range "
            f"[0, {len(slices) - 1}]"
        )

    return slices[slice_idx][1]