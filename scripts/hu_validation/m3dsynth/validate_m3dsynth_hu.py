
"""Validate M3Dsynth HU reconstruction against matching LIDC-IDRI DICOMs.

Matching:
1. Read DICOM headers and sort by InstanceNumber.
2. Determine Z direction using first, middle and last slices.
3. Verify monotonic Z ordering across the series.
4. Select the DICOM at slice_idx in descending Z order.
5. Verify raw-pixel correlation outside the manipulation mask.
6. Compute HU reconstruction metrics only after successful matching.

Run:
    python scripts/validate_m3dsynth_hu.py

Required environment:
    LIDC_ROOT
    M3D_SELECTED_ROOT
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import tifffile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.data.adapters.lidc import load_lidc_hu
from src.data.adapters.m3dsynth import (
    M3DSynthIntensityMapping,
    load_m3dsynth_image,
)

SELECTED = REPO / "results/hu_validation/m3dsynth/select_hu_validation_slices.csv"
MAPPING = (
    REPO
    / "metadata/m3dsynth/audits/m3dsynth_volume_intensity_mapping.csv"
)
OUTPUT = REPO / "results/hu_validation/m3dsynth/validate_m3dsynth_hu.csv"

MIN_MATCH_CORRELATION = 0.999


def load_local_env(path: Path) -> None:
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)

        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()

    if a.size < 100:
        return float("nan")

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def build_series_index(series_dir: Path, expected_count: int) -> dict:
    """Build a metadata-only index for one DICOM series."""

    records = []

    for path in series_dir.glob("*.dcm"):
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=[
                "InstanceNumber",
                "ImagePositionPatient",
            ],
        )

        records.append({
            "path": path,
            "instance": int(ds.InstanceNumber),
            "z": float(ds.ImagePositionPatient[2]),
        })

    if len(records) != expected_count:
        raise ValueError(
            f"DICOM count mismatch: expected {expected_count}, "
            f"found {len(records)}"
        )

    if len(records) < 3:
        raise ValueError("At least three DICOM slices required")

    records.sort(key=lambda item: item["instance"])

    instances = np.array(
        [item["instance"] for item in records],
        dtype=np.int64,
    )

    if len(np.unique(instances)) != len(instances):
        raise ValueError("Duplicate InstanceNumber values")

    z_values = np.array(
        [item["z"] for item in records],
        dtype=np.float64,
    )

    # Inspect three positions to determine the acquisition direction.
    first_z = z_values[0]
    middle_z = z_values[len(z_values) // 2]
    last_z = z_values[-1]

    if first_z > middle_z > last_z:
        direction = "decreasing"
    elif first_z < middle_z < last_z:
        direction = "increasing"
    else:
        raise ValueError(
            "Cannot determine consistent Z direction "
            "from first/middle/last slices"
        )

    # Verify the direction over the entire series.
    differences = np.diff(z_values)

    if direction == "decreasing":
        if not np.all(differences < 0):
            raise ValueError("Non-monotonic decreasing Z series")
        descending = records
    else:
        if not np.all(differences > 0):
            raise ValueError("Non-monotonic increasing Z series")
        descending = list(reversed(records))

    return {
        "direction": direction,
        "slices": descending,
        "count": len(records),
        "z_first": float(first_z),
        "z_middle": float(middle_z),
        "z_last": float(last_z),
    }


def hu_metrics(
    lidc: np.ndarray,
    m3d: np.ndarray,
    outside: np.ndarray,
) -> dict:
    x = lidc[outside].astype(np.float64)
    y = m3d[outside].astype(np.float64)

    difference = y - x
    absolute = np.abs(difference)

    return {
        "outside_pixels": int(difference.size),
        "mae_hu": float(np.mean(absolute)),
        "rmse_hu": float(np.sqrt(np.mean(difference ** 2))),
        "mean_diff_hu": float(np.mean(difference)),
        "median_diff_hu": float(np.median(difference)),
        "max_abs_diff_hu": float(np.max(absolute)),
        "correlation_hu": correlation(x, y),
        "exact_pct": float(100 * np.mean(difference == 0)),
        "within_1_hu_pct": float(100 * np.mean(absolute <= 1)),
        "within_5_hu_pct": float(100 * np.mean(absolute <= 5)),
        "within_10_hu_pct": float(100 * np.mean(absolute <= 10)),
    }


def process(
    row,
    *,
    lidc_root: Path,
    m3d_root: Path,
    mapping,
    counts: dict,
    series_cache: dict,
) -> dict:

    result = {
        "validation_id": row.validation_id,
        "patient_id": row.patient_id,
        "generator": row.generator,
        "manipulation_type": row.manipulation_type,
        "method": row.method,
        "offset": row.offset,
        "slice_idx": int(row.slice_idx),
        "status": "error",
        "error": "",
    }

    try:
        key = (str(row.patient_id), str(row.sdir_id))

        if key not in counts:
            raise ValueError(f"No volume count for {key}")

        expected_count = counts[key]
        slice_idx = int(row.slice_idx)

        if not 0 <= slice_idx < expected_count:
            raise ValueError(
                f"slice_idx={slice_idx} outside volume range "
                f"[0, {expected_count})"
            )

        relative = os.environ.get(
            "LIDC_SERIES_LAYOUT",
            "{patient_id}/{series_uid}",
        ).format(
            patient_id=row.patient_id,
            series_uid=row.SeriesInstanceUID,
        )

        series_dir = lidc_root / relative

        # Cache by physical DICOM series.
        series_key = (str(row.patient_id), str(row.SeriesInstanceUID))

        if series_key not in series_cache:
            series_cache[series_key] = build_series_index(
                series_dir,
                expected_count,
            )

        series = series_cache[series_key]

        if series["count"] != expected_count:
            raise ValueError("Cached series count mismatch")

        match = series["slices"][slice_idx]
        dicom_path = match["path"]

        result.update({
            "expected_image_count": expected_count,
            "z_direction": series["direction"],
            "matched_dicom_filename": dicom_path.name,
            "matched_instance_number": match["instance"],
            "matched_z": match["z"],
            "z_first": series["z_first"],
            "z_middle": series["z_middle"],
            "z_last": series["z_last"],
        })

        header = pydicom.dcmread(
            dicom_path,
            stop_before_pixels=True,
        )

        if str(header.SeriesInstanceUID) != str(row.SeriesInstanceUID):
            raise ValueError("SeriesInstanceUID mismatch")

        image_path = m3d_root / str(row.image_relative_path)
        mask_path = m3d_root / str(row.mask_relative_path)

        raw_tiff = tifffile.imread(image_path)
        raw_dicom = pydicom.dcmread(dicom_path).pixel_array
        mask = tifffile.imread(mask_path)

        if (
            raw_tiff.shape != raw_dicom.shape
            or mask.shape != raw_tiff.shape
        ):
            raise ValueError(
                f"Shape mismatch: TIFF={raw_tiff.shape}, "
                f"DICOM={raw_dicom.shape}, mask={mask.shape}"
            )

        outside = mask == 0

        if np.count_nonzero(outside) < 100:
            raise ValueError("Too few pixels outside manipulation mask")

        # Verify anatomical matching independently of HU conversion.
        match_corr = correlation(
            raw_dicom[outside],
            raw_tiff[outside],
        )

        result["matching_correlation"] = match_corr

        if not np.isfinite(match_corr):
            raise ValueError("Invalid matching correlation")

        if match_corr < MIN_MATCH_CORRELATION:
            result["status"] = "matching_failed"
            result["error"] = (
                f"Matching correlation {match_corr:.9f} "
                f"below {MIN_MATCH_CORRELATION}"
            )
            return result

        # Validate DICOM rescaling metadata.
        meta = mapping.get(row.patient_id, row.sdir_id)

        dicom_slope = float(header.RescaleSlope)
        dicom_intercept = float(header.RescaleIntercept)

        result.update({
            "dicom_slope": dicom_slope,
            "dicom_intercept": dicom_intercept,
            "mapping_slope": float(meta["slope"]),
            "mapping_intercept": float(meta["intercept"]),
            "clipping_occurred": meta["clipping_occurred"],
        })

        if not (
            np.isclose(dicom_slope, meta["slope"])
            and np.isclose(dicom_intercept, meta["intercept"])
        ):
            raise ValueError("Slope/intercept mismatch")

        # Existing adapters: no changes to conversion logic.
        m3d_hu = load_m3dsynth_image(
            image_path,
            row.patient_id,
            row.sdir_id,
            mapping,
        )

        lidc_hu = load_lidc_hu(dicom_path)

        result.update(
            hu_metrics(lidc_hu, m3d_hu, outside)
        )

        result["status"] = "measured"

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def main() -> None:
    load_local_env(REPO / ".env")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lidc-root",
        default=os.getenv("LIDC_ROOT"),
    )

    parser.add_argument(
        "--m3d-root",
        default=os.getenv("M3D_SELECTED_ROOT"),
    )

    args = parser.parse_args()

    if not args.lidc_root or not args.m3d_root:
        parser.error(
            "Set LIDC_ROOT and M3D_SELECTED_ROOT "
            "in .env or via arguments"
        )

    lidc_root = Path(args.lidc_root)
    m3d_root = Path(args.m3d_root)

    for root in (lidc_root, m3d_root):
        if not root.is_dir():
            parser.error(f"Directory not found: {root}")

    selected = pd.read_csv(
        SELECTED,
        dtype={"sdir_id": str},
    )

    if len(selected) != 30 or selected.validation_id.nunique() != 30:
        parser.error("Expected 30 distinct validation IDs")

    mapping_df = pd.read_csv(
        MAPPING,
        dtype={
            "PatientID": str,
            "M3Dsynth_sdir_id": str,
        },
    )

    counts = {}

    for row in mapping_df.itertuples(index=False):
        count = int(row.expected_image_count)

        if count < 1:
            raise ValueError(
                f"Invalid expected_image_count: {count}"
            )

        counts[
            (str(row.PatientID), str(row.M3Dsynth_sdir_id))
        ] = count

    mapping = M3DSynthIntensityMapping(MAPPING)

    series_cache = {}
    rows = []

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for row in selected.itertuples(index=False):

        result = process(
            row,
            lidc_root=lidc_root,
            m3d_root=m3d_root,
            mapping=mapping,
            counts=counts,
            series_cache=series_cache,
        )

        rows.append(result)

        # Save progressively in case the run is interrupted.
        pd.DataFrame(rows).to_csv(
            OUTPUT,
            index=False,
        )

        match_corr = result.get(
            "matching_correlation",
            float("nan"),
        )

        mae = result.get("mae_hu", float("nan"))

        print(
            f"{result['validation_id']} | "
            f"{result['patient_id']} | "
            f"Z={result.get('z_direction', 'N/A')} | "
            f"Instance={result.get('matched_instance_number', 'N/A')} | "
            f"match_corr={match_corr:.9f} | "
            f"MAE={mae:.6f} HU | "
            f"{result['status']} "
            f"{result['error']}",
            flush=True,
        )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print(df["status"].value_counts().to_string())

    measured = df[df["status"] == "measured"]

    if not measured.empty:
        print("\nHU METRICS (measured samples)")
        print(
            measured[
                [
                    "validation_id",
                    "method",
                    "mae_hu",
                    "median_diff_hu",
                    "exact_pct",
                    "within_1_hu_pct",
                ]
            ].to_string(index=False)
        )

        print("\nMAE BY MAPPING METHOD")
        print(
            measured.groupby("method")["mae_hu"]
            .agg(["count", "mean", "median", "max"])
            .to_string()
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()


