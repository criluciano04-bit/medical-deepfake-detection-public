"""Materialize the selected CT-RATE Real + Generated slices for evaluation Test 2.

Run from the repository root (with DATA_ROOT pointing to selected/):

    python -m scripts.data.build_processed.ct_rate --manifest PATH_TO_GIT_MANIFEST

PATH_TO_GIT_MANIFEST is the ct_rate_selected_manifest.csv committed in your repo.
Paths inside the source manifest remain relative to their documented roots:
- selected_path: relative to DATA_ROOT/CT-RATE_selected/
- original_path: relative to DATA_ROOT.parent/ (medical-deepfake-data/)

Output: processed/test2/images/<sample_id>.npy,
        processed/test2/processed_manifest.csv,
        processed/test2/preprocessing_config.json.

No training, ImageNet normalization, extra HU conversion, or class-specific processing.
Input files are never changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from src.data.adapters.ct_rate import load_ct_rate_image
from src.data.paths import get_data_root
from src.data.preprocessing import apply_hu_window, normalize_image, resize_image

EXPECTED_REAL = 2250
EXPECTED_FAKE = 2250
EXPECTED_TOTAL = EXPECTED_REAL + EXPECTED_FAKE
TARGET_SIZE = (512, 512)
HU_LOWER = -1000.0
HU_UPPER = 1000.0

SOURCE_COLUMNS = [
    "sample_id", "dataset", "label", "split", "patient_id", "series_id",
    "slice_id", "generator", "manipulation_type", "source_row",
    "original_path", "selected_path", "volume_id", "rotation",
    "sampling_method", "shape", "dtype",
]
PROCESSED_COLUMNS = [
    "sample_id", "dataset", "label", "split", "patient_id", "series_id",
    "slice_id", "generator", "manipulation_type", "source_row",
    "original_path", "processed_path", "selected_path", "volume_id",
    "rotation", "sampling_method", "shape", "dtype",
]
CONFIG = {
    "format": "npy",
    "dtype": "float32",
    "channels": 1,
    "shape": [512, 512],
    "source_dataset": "CT-RATE Real + Generated selected",
    "split": "test2",
    "samples": EXPECTED_TOTAL,
    "real_samples": EXPECTED_REAL,
    "fake_samples": EXPECTED_FAKE,
    "source_intensity": "HU (real valid_fixed; generated values already clipped to [-1000,1000])",
    "additional_hu_conversion": False,
    "hu_window": [-1000, 1000],
    "resize": "bilinear, antialias=True, align_corners=False (only if needed)",
    "normalization": "fixed HU min-max to [0,1]",
    "imagenet_normalization": False,
    "channel_replication": False,
    "source_manifest": "ct_rate_selected_manifest.csv (versioned in repository)",
    "selected_path": "relative to DATA_ROOT/CT-RATE_selected/",
    "original_path": "relative to DATA_ROOT.parent/",
    "processed_path": "relative to processed/test2/",
}


def validate_relative(value: str, *, suffix: str | None = None) -> PurePosixPath:
    """Reject absolute, traversing, drive-letter, and backslash paths."""
    if not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts or "." in rel.parts:
        raise ValueError(f"Invalid relative path: {value!r}")
    if suffix is not None and rel.suffix != suffix:
        raise ValueError(f"Unexpected extension: {value!r}")
    return rel


def write_atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_processed(image: np.ndarray, path: Path) -> None:
    if image.shape != TARGET_SIZE or image.dtype != np.float32:
        raise ValueError(f"Invalid output shape/dtype at {path}: {image.shape}/{image.dtype}")
    if not np.isfinite(image).all() or image.min() < -1e-6 or image.max() > 1 + 1e-6:
        raise ValueError(f"Invalid normalized intensities at {path}")


def preprocess_image(path: Path) -> tuple[np.ndarray, bool]:
    image = load_ct_rate_image(path)  # in HU, no conversion or clipping here
    resized = image.shape != TARGET_SIZE
    image = apply_hu_window(image, HU_LOWER, HU_UPPER)
    image = resize_image(image, TARGET_SIZE)
    image = normalize_image(image, HU_LOWER, HU_UPPER)
    image = np.ascontiguousarray(image, dtype=np.float32)
    validate_processed(image, path)
    return image, resized


def build_rows(source: pd.DataFrame) -> list[dict[str, str]]:
    if source.columns.tolist() != SOURCE_COLUMNS:
        raise ValueError(f"Unexpected source manifest columns: {source.columns.tolist()}")
    if len(source) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} rows, got {len(source)}")
    if source["sample_id"].duplicated().any() or source["selected_path"].duplicated().any():
        raise ValueError("Duplicate sample_id or selected_path")
    if not source["dataset"].eq("ct_rate").all() or not source["split"].eq("test").all():
        raise ValueError("Source manifest must have dataset='ct_rate' and split='test'")
    if source["label"].value_counts().to_dict() != {"0": EXPECTED_REAL, "1": EXPECTED_FAKE}:
        raise ValueError(f"Unexpected class counts: {source['label'].value_counts().to_dict()}")

    output_rows = []
    for row in source.to_dict(orient="records"):
        sample_id = row["sample_id"]
        label = row["label"]
        if not sample_id or any(ch in sample_id for ch in "/\\:") or sample_id in {".", ".."}:
            raise ValueError(f"Unsafe sample_id: {sample_id!r}")
        expected_prefix = "ct_rate_real_" if label == "0" else "ct_rate_fake_"
        if not sample_id.startswith(expected_prefix):
            raise ValueError(f"Inconsistent sample_id and label: {sample_id}/{label}")
        selected = validate_relative(row["selected_path"], suffix=".npy")
        expected_group = "real" if label == "0" else "fake"
        if selected.parts[:2] != ("images", expected_group) or len(selected.parts) != 3:
            raise ValueError(f"Wrong selected image group: {selected}")
        original = validate_relative(row["original_path"], suffix=".gz")
        if not original.parts or original.parts[0] != "raw":
            raise ValueError(f"Invalid original_path root: {original}")
        output_rows.append({
            **row,
            "split": "test2",
            "processed_path": f"images/{sample_id}.npy",
        })
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Path to the Git-versioned ct_rate_selected_manifest.csv (may be relative to current directory)",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Optional processed/ directory; defaults to sibling of DATA_ROOT",
    )
    args = parser.parse_args()
    selected_root = get_data_root()
    selected_dir = selected_root / "CT-RATE_selected"
    manifest_input = args.manifest.expanduser()
    if not manifest_input.is_file():
        raise FileNotFoundError(f"Source manifest not found: {manifest_input}")

    processed_root = (
        args.output_root.expanduser() if args.output_root is not None
        else selected_root.parent / "processed"
    )
    output_dir = processed_root / "test2"
    images_dir = output_dir / "images"
    manifest_out = output_dir / "processed_manifest.csv"
    config_out = output_dir / "preprocessing_config.json"

    source = pd.read_csv(manifest_input, dtype=str, keep_default_na=False)
    rows = build_rows(source)
    frame = pd.DataFrame(rows, columns=PROCESSED_COLUMNS)

    # Protect established sample IDs and preprocessing choices on reruns.
    if manifest_out.exists():
        old = pd.read_csv(manifest_out, dtype=str, keep_default_na=False)
        if not old.equals(frame):
            raise RuntimeError("Existing Test 2 manifest differs; aborting to protect sample-ID mapping")
    if config_out.exists():
        if json.loads(config_out.read_text(encoding="utf-8")) != CONFIG:
            raise RuntimeError("Existing Test 2 preprocessing config differs; aborting")

    images_dir.mkdir(parents=True, exist_ok=True)
    created = reused = resized_count = 0
    for count, row in enumerate(rows, start=1):
        relative = validate_relative(row["selected_path"], suffix=".npy")
        source_path = selected_dir.joinpath(*relative.parts)
        dest = output_dir / row["processed_path"]
        if dest.is_file():
            image = np.load(dest, mmap_mode="r", allow_pickle=False)
            validate_processed(image, dest)
            reused += 1
            del image
        else:
            image, resized = preprocess_image(source_path)
            resized_count += int(resized)
            tmp = dest.with_name(dest.name + ".tmp")
            try:
                with tmp.open("wb") as f:
                    np.save(f, image, allow_pickle=False)
                os.replace(tmp, dest)
            finally:
                tmp.unlink(missing_ok=True)
            created += 1
        if count % 100 == 0 or count == EXPECTED_TOTAL:
            print(
                f"CT-RATE {count}/{EXPECTED_TOTAL} | new={created} "
                f"reused={reused} resized={resized_count}",
                flush=True,
            )

    write_atomic_text(manifest_out, frame.to_csv(index=False))
    write_atomic_text(config_out, json.dumps(CONFIG, indent=2, ensure_ascii=False) + "\n")
    print(f"COMPLETED: {len(rows)} images in {output_dir}")
    print("Selected input images unchanged; no additional HU conversion applied.")


if __name__ == "__main__":
    main()
