
from pathlib import Path
import pytest
from src.data.adapters.m3dsynth import M3DSynthIntensityMapping


MAPPING_PATH = (
    Path(__file__).resolve().parents[1]
    / "metadata"
    / "m3dsynth"
    / "audits"
    / "m3dsynth_volume_intensity_mapping.csv"
)


def test_mapping_loads_and_retrieves_metadata():
    mapping = M3DSynthIntensityMapping(MAPPING_PATH)

    result = mapping.get("LIDC-IDRI-0884", "2")

    assert result["offset"] == 2000.0
    assert result["method"] == "volume_min"
    assert result["dtype"] == "int16"
    assert result["slope"] == 1.0
    assert result["intercept"] == -1024.0
    assert result["clipping_occurred"] is False
    assert result["clipped_pixel_count"] == 0


def test_mapping_rejects_unknown_series():
    mapping = M3DSynthIntensityMapping(MAPPING_PATH)

    with pytest.raises(KeyError):
        mapping.get("LIDC-IDRI-0884", "nonexistent")



import numpy as np
import tifffile

from src.data.adapters.m3dsynth import load_m3dsynth_image


def test_load_m3dsynth_image_reconstructs_hu(tmp_path):
    image_path = tmp_path / "test.tiff"

    image = np.array(
        [[0, 1000], [2000, 3000]],
        dtype=np.uint16,
    )

    tifffile.imwrite(image_path, image)

    class FakeMapping:
        def get(self, patient_id, sdir_id):
            return {
                "offset": 2000.0,
                "slope": 1.0,
                "intercept": -1024.0,
            }

    result = load_m3dsynth_image(
        image_path,
        patient_id="test_patient",
        sdir_id="test_series",
        mapping=FakeMapping(),
    )

    expected = np.array(
        [[-3024, -2024], [-1024, -24]],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(result, expected)

    assert result.dtype == np.float32


def test_load_m3dsynth_image_rejects_missing_file(tmp_path):
    class FakeMapping:
        def get(self, patient_id, sdir_id):
            return {
                "offset": 0.0,
                "slope": 1.0,
                "intercept": 0.0,
            }

    with pytest.raises(FileNotFoundError):
        load_m3dsynth_image(
            tmp_path / "missing.tiff",
            patient_id="test_patient",
            sdir_id="test_series",
            mapping=FakeMapping(),
        )


def test_load_m3dsynth_image_rejects_wrong_dtype(tmp_path):
    image_path = tmp_path / "wrong_dtype.tiff"

    image = np.array(
        [[1, 2], [3, 4]],
        dtype=np.uint8,
    )

    tifffile.imwrite(image_path, image)

    class FakeMapping:
        def get(self, patient_id, sdir_id):
            return {
                "offset": 0.0,
                "slope": 1.0,
                "intercept": 0.0,
            }

    with pytest.raises(ValueError, match="Expected uint16"):
        load_m3dsynth_image(
            image_path,
            patient_id="test_patient",
            sdir_id="test_series",
            mapping=FakeMapping(),
        )



import pandas as pd

from src.data.paths import get_data_root


def test_load_real_m3dsynth_image():
    csv_path = (
        Path(__file__).resolve().parents[1]
        / "metadata"
        / "m3dsynth"
        / "m3dsynth_selected_slices.csv"
    )

    df = pd.read_csv(csv_path)

    row = df.iloc[0]

    image_path = (
        get_data_root()
        / "m3d_selected"
        / row["image_relative_path"]
    )

    mapping = M3DSynthIntensityMapping(MAPPING_PATH)

    result = load_m3dsynth_image(
        image_path=image_path,
        patient_id=str(row["patient_id"]),
        sdir_id=str(row["sdir_id"]),
        mapping=mapping,
    )

    assert result.ndim == 2
    assert result.shape == (512, 512)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()

    print(f"\nImage: {image_path}")
    print(f"HU range: {result.min()} to {result.max()}")
    metadata = mapping.get(
    str(row["patient_id"]),
    str(row["sdir_id"]),
    )

    print("Mapping metadata:", metadata)
    print("Raw TIFF range:", tifffile.imread(image_path).min(),
      "to", tifffile.imread(image_path).max())  