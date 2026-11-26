
from pathlib import Path
import pandas as pd

from src.data.matching.m3dsynth_lidc import (
    find_matching_lidc_slice,
)
from src.data.paths import PROJECT_ROOT

from dotenv import load_dotenv
import os


def test_matching_against_validated_pairs():
    """
    Verify that the matching function reproduces
    the 30 previously validated LIDC-M3Dsynth pairs.
    """

    load_dotenv(PROJECT_ROOT / ".env")

    lidc_root = Path(os.environ["LIDC_ROOT"])

    selected = pd.read_csv(
        PROJECT_ROOT
        / "results/hu_validation/select_hu_validation_slices.csv"
    )

    validated = pd.read_csv(
        PROJECT_ROOT
        / "results/hu_validation/check_slice_matching.csv"
    )

    pairs = selected.merge(
        validated[
            ["validation_id", "dicom_filename", "status"]
        ],
        on="validation_id",
        validate="one_to_one",
    )

    assert len(pairs) == 30
    assert (pairs["status_y"] == "OK").all()

    for row in pairs.itertuples(index=False):
        series_dir = (
            lidc_root
            / row.patient_id
            / row.SeriesInstanceUID
        )

        actual_path = find_matching_lidc_slice(
            series_dir,
            int(row.slice_idx),
        )

        assert actual_path.name == row.dicom_filename, (
            f"Mismatch for {row.validation_id}: "
            f"expected {row.dicom_filename}, "
            f"got {actual_path.name}"
        )