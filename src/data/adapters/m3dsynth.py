
from pathlib import Path
import pandas as pd
import numpy as np
import tifffile

class M3DSynthIntensityMapping:
    """Load and retrieve intensity metadata for M3Dsynth series."""

    REQUIRED_COLUMNS = {
        "PatientID",
        "M3Dsynth_sdir_id",
        "offset",
        "method",
        "dtype",
        "slope_values",
        "intercept_values",
        "volume_min",
        "volume_max",
        "clipping_occurred",
        "clipped_pixel_count",
    }

    def __init__(self, csv_path: str | Path):
        csv_path = Path(csv_path)

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"M3Dsynth intensity mapping not found: {csv_path}"
            )

        df = pd.read_csv(
            csv_path,
            dtype={
                "PatientID": "string",
                "M3Dsynth_sdir_id": "string",
            },
        )

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if df[list(self.REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError(
                "Intensity mapping contains missing required values."
            )

        keys = ["PatientID", "M3Dsynth_sdir_id"]

        if df.duplicated(subset=keys).any():
            raise ValueError(
                "Duplicate patient-series pairs in intensity mapping."
            )

        numeric_columns = [
            "offset",
            "slope_values",
            "intercept_values",
            "volume_min",
            "volume_max",
            "clipped_pixel_count",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="raise")

        valid_clipping = {"true": True, "false": False}
        clipping = (
            df["clipping_occurred"]
            .astype(str)
            .str.lower()
            .map(valid_clipping)
        )

        if clipping.isna().any():
            raise ValueError("Invalid clipping_occurred value.")

        df["clipping_occurred"] = clipping.astype(bool)

        self._mapping = {}

        for row in df.itertuples(index=False):
            key = (
                str(row.PatientID),
                str(row.M3Dsynth_sdir_id),
            )

            self._mapping[key] = {
                "offset": float(row.offset),
                "method": str(row.method),
                "dtype": str(row.dtype),
                "slope": float(row.slope_values),
                "intercept": float(row.intercept_values),
                "volume_min": float(row.volume_min),
                "volume_max": float(row.volume_max),
                "clipping_occurred": bool(row.clipping_occurred),
                "clipped_pixel_count": int(row.clipped_pixel_count),
            }

    def get(self, patient_id: str, sdir_id: str | int) -> dict:
        """Return intensity metadata for a patient-series pair."""

        key = (str(patient_id), str(sdir_id))

        if key not in self._mapping:
            raise KeyError(
                f"No intensity mapping for patient={patient_id}, "
                f"series={sdir_id}"
            )

        return self._mapping[key].copy()



def load_m3dsynth_image(
    image_path: str | Path,
    patient_id: str,
    sdir_id: str | int,
    mapping: M3DSynthIntensityMapping,
) -> np.ndarray:
    """Load an M3Dsynth TIFF image and reconstruct HU intensities."""

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"M3Dsynth image not found: {image_path}"
        )

    image = tifffile.imread(image_path)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D TIFF image, got shape {image.shape}"
        )

    if image.dtype != np.uint16:
        raise ValueError(
            f"Expected uint16 TIFF, got {image.dtype}"
        )

    metadata = mapping.get(patient_id, sdir_id)

    pixels = image.astype(np.float32)

    original_pixels = pixels - metadata["offset"]

    hu = (
        original_pixels * metadata["slope"]
        + metadata["intercept"]
    )

    if not np.isfinite(hu).all():
        raise ValueError(
            f"Non-finite intensity values in {image_path}"
        )

    return hu.astype(np.float32)

