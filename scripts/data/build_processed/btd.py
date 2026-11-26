"""Materialize BTD Test 1 as normalized float32 .npy slices, without a Dataset class.

Run from the repository root:
    python -m scripts.data.build_processed.btd

DATA_ROOT must point to selected/ (configured in repository .env). All file paths
saved in the output manifest are relative; no machine-specific paths are stored.

Output (sibling of selected/):
    processed/test1/images/btd_000001.npy ...
    processed/test1/processed_manifest.csv
    processed/test1/preprocessing_config.json

Input manifest and selected images are never modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from src.data.paths import get_data_root
from src.data.adapters.btd import load_btd_image
from src.data.preprocessing import apply_hu_window, resize_image, normalize_image

EXPECTED_COUNT = 1981
SOURCE_COLUMNS = [
    "group", "label", "generator", "manipulations", "patient_id",
    "series_uid", "slice_number", "source_path", "selected_path",
]
# 'label' is numeric in all processed datasets (0 real, 1 fake).
# The original three-valued BTD label is preserved as 'original_label'.
MANIFEST_COLUMNS = [
    "sample_id", "dataset", "label", "split", "patient_id", "series_id",
    "slice_id", "generator", "manipulation_type", "source_row",
    "original_path", "processed_path", "original_label", "group",
    "manipulations", "series_uid", "slice_number", "source_path",
    "selected_path",
]

CONFIG = {
    "format": "npy", "dtype": "float32", "channels": 1, "shape": [512, 512],
    "source_dataset": "BTD selected", "split": "test1", "samples": EXPECTED_COUNT,
    "pre_window_clipping": [-700, 2000], "hu_window": [-1000, 1000],
    "resize": "bilinear, antialias=True, align_corners=False (only if needed)",
    "normalization": "fixed HU min-max to [0,1]", "imagenet_normalization": False,
    "channel_replication": False,
    "effective_intensity_range_before_normalization": [-700, 1000],
    "intensity_assumption": "BTD intensities treated as HU-consistent per prior audit; no slope/intercept conversion",
    "overlap": "all images retained; group preserved as metadata only",
    "source_manifest": "btd_selected/metadata/btd_selected_manifest.csv (relative to DATA_ROOT)",
    "processed_path": "relative to processed/test1/",
}


def write_atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def check_image(image: np.ndarray, path: Path) -> None:
    if image.shape != (512, 512) or image.dtype != np.float32:
        raise ValueError(f"Unexpected shape/dtype at {path}: {image.shape}/{image.dtype}")
    if not np.isfinite(image).all() or image.min() < -1e-6 or image.max() > 1.0 + 1e-6:
        raise ValueError(f"Invalid normalized values at {path}")


def preprocess(source: Path) -> tuple[np.ndarray, bool]:
    """BTD-specific adapter, followed by shared project preprocessing."""
    image = load_btd_image(source)  # Includes the agreed [-700, 2000] clip.
    image = apply_hu_window(image, -1000.0, 1000.0)
    resized = image.shape != (512, 512)
    image = resize_image(image, (512, 512))
    image = normalize_image(image, -1000.0, 1000.0)
    image = image.astype(np.float32, copy=False)
    check_image(image, source)
    return image, resized


def validate_relative_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError(f"Expected POSIX relative path, got: {value}")
    rel = PurePosixPath(value)
    if (not value or rel.is_absolute() or ".." in rel.parts or ":" in value
            or rel.suffix.lower() != ".npy"):
        raise ValueError(f"Unsafe input path: {value}")
    if rel.parts[0] not in {"real", "fake_ctgan", "fake_sd"}:
        raise ValueError(f"Unexpected BTD image directory: {value}")
    return rel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Optional processed/ location; default sibling of DATA_ROOT")
    args = parser.parse_args()
    selected_root = get_data_root()
    selected = selected_root / "btd_selected"
    input_manifest = selected / "metadata" / "btd_selected_manifest.csv"
    processed_root = (args.output_root if args.output_root is not None
                      else selected_root.parent / "processed").expanduser()
    output = processed_root / "test1"
    images_dir = output / "images"
    manifest_out = output / "processed_manifest.csv"
    config_out = output / "preprocessing_config.json"

    source = pd.read_csv(input_manifest, dtype=str, keep_default_na=False)
    if source.columns.tolist() != SOURCE_COLUMNS:
        raise ValueError(f"Unexpected BTD manifest columns: {source.columns.tolist()}")
    if len(source) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} BTD images, got {len(source)}")
    if source.selected_path.duplicated().any():
        raise ValueError("Duplicate selected_path entries")
    expected_classes = {"real": 387, "fake_ctgan": 797, "fake_sd": 797}
    if source.label.value_counts().to_dict() != expected_classes:
        raise ValueError(f"Unexpected BTD class counts: {source.label.value_counts().to_dict()}")

    rows = []
    for index, item in source.iterrows():
        rel = validate_relative_path(item.selected_path)
        if rel.parts[0] != item.label:
            raise ValueError(f"Inconsistent label/path in row {index + 1}")
        sample_id = f"btd_{index + 1:06d}"
        rows.append({
            "sample_id": sample_id, "dataset": "btd",
            "label": 0 if item.label == "real" else 1, "split": "test1",
            "patient_id": item.patient_id, "series_id": item.series_uid,
            "slice_id": item.slice_number, "generator": item.generator,
            "manipulation_type": item.manipulations, "source_row": index,
            "original_path": f"btd_selected/images/{rel.as_posix()}",
            "processed_path": f"images/{sample_id}.npy", "original_label": item.label,
            "group": item.group, "manipulations": item.manipulations,
            "series_uid": item.series_uid, "slice_number": item.slice_number,
            "source_path": item.source_path, "selected_path": item.selected_path,
        })
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    if manifest_out.exists():
        old = pd.read_csv(manifest_out, dtype=str, keep_default_na=False)
        if not old.equals(frame.astype(str)):
            raise RuntimeError("Existing processed manifest differs. Aborting to protect ID mapping")
    if config_out.exists():
        previous = json.loads(config_out.read_text(encoding="utf-8"))
        previous_comparable = dict(previous)
        # Earlier builds recorded the former directory name in this descriptive field.
        if previous_comparable.get("processed_path") == "relative to processed/test1_btd/":
            previous_comparable["processed_path"] = "relative to processed/test1/"
        if previous_comparable != CONFIG:
            raise RuntimeError("Existing preprocessing config differs. Aborting")

    images_dir.mkdir(parents=True, exist_ok=True)
    created = reused = resized_count = 0
    for count, row in enumerate(rows, start=1):
        src_rel = validate_relative_path(row["selected_path"])
        src = selected / "images" / Path(*src_rel.parts)
        dst = output / row["processed_path"]
        if dst.exists():
            image = np.load(dst, mmap_mode="r", allow_pickle=False)
            check_image(image, dst)
            reused += 1
        else:
            if not src.is_file():
                raise FileNotFoundError(src)
            image, resized = preprocess(src)
            resized_count += int(resized)
            tmp = dst.with_name(dst.name + ".tmp")
            with tmp.open("wb") as f:
                np.save(f, image, allow_pickle=False)
            os.replace(tmp, dst)
            created += 1
        if count % 100 == 0 or count == len(rows):
            print(f"BTD {count}/{len(rows)} | new={created} reused={reused} resized={resized_count}", flush=True)

    write_atomic_text(manifest_out, frame.to_csv(index=False))
    write_atomic_text(config_out, json.dumps(CONFIG, indent=2, ensure_ascii=False) + "\n")
    print(f"COMPLETED: {len(rows)} BTD images; output: {output}")
    print("Original selected/ data unchanged. No per-class or overlap split.")


if __name__ == "__main__":
    main()
