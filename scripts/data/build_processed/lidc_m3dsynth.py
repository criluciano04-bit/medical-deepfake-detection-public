"""Materialize the existing LIDC/M3Dsynth preprocessing once as .npy arrays.

Run from the repository root. DATA_ROOT points to selected/ (via .env or
an environment variable). No absolute machine paths are stored in manifests.

Output layout (next to selected/ by default):
    processed/training/train/*.npy
    processed/training/valid/*.npy
    processed/training/processed_manifest.csv
    processed/training/preprocessing_config.json
    processed/test0/images/*.npy
    processed/test0/processed_manifest.csv
    processed/test0/preprocessing_config.json

Both config files have identical content. Each manifest's processed_path is
relative to the directory containing that manifest (training/ or test0/).
The original Dataset performs the existing DICOM/TIFF loading, HU conversion,
windowing, conditional resize, min-max normalization, and validation.

For a small check:
    python -m scripts.data.build_processed.lidc_m3dsynth --split train --limit-per-class 2
For all 22,500 selected images:
    python -m scripts.data.build_processed.lidc_m3dsynth --split all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from src.data.datasets.lidc_m3dsynth import LIDC_M3DSynth_Dataset
from src.data.paths import get_data_root

CONFIG = {
    "format": "npy",
    "dtype": "float32",
    "channels": 1,
    "shape": [512, 512],
    "hu_window": [-1000, 1000],
    "resize": "bilinear, antialias=True, align_corners=False (only if needed)",
    "normalization": "fixed HU min-max to [0,1]",
    "imagenet_normalization": False,
    "channel_replication": False,
    "source_dataset": "LIDC_M3DSynth_Dataset",
}

MANIFEST_COLUMNS = [
    "sample_id", "dataset", "label", "split", "patient_id", "series_id",
    "slice_id", "generator", "manipulation_type", "source_row",
    "original_path", "processed_path",
]


def write_atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def target_location(split: str, sample_id: str) -> tuple[str, PurePosixPath]:
    """Return partition and path relative to its own manifest directory."""
    if split == "test":
        return "test0", PurePosixPath("images", f"{sample_id}.npy")
    if split in ("train", "valid"):
        return "training", PurePosixPath(split) / f"{sample_id}.npy"
    raise ValueError(f"Unsupported split: {split}")


def source_details(row: pd.Series, split: str) -> tuple[str, str, str]:
    """Return path relative to selected/, original series, original slice ID."""
    original_rel = PurePosixPath(str(row["image_path"]))
    if (original_rel.is_absolute() or ".." in original_rel.parts
            or original_rel.parts[:2] != ("images", split)):
        raise ValueError(f"Unexpected original path: {original_rel}")

    name = str(row["dataset"])
    if name == "lidc":
        if len(original_rel.parts) < 5:
            raise ValueError(f"Unexpected LIDC path: {original_rel}")
        prefix = "lidc_selected"
        series_id = original_rel.parts[-2]
    elif name == "m3dsynth":
        prefix = "m3d_selected"
        if pd.isna(row["sdir_id"]):
            raise ValueError(f"Missing M3Dsynth series ID for {row['sample_id']}")
        series_id = str(row["sdir_id"])
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    original_path = PurePosixPath(prefix, *original_rel.parts).as_posix()
    return original_path, series_id, original_rel.stem


def validate_array(image: np.ndarray, path: Path) -> None:
    if image.shape != (512, 512) or image.dtype != np.float32:
        raise ValueError(f"Unexpected shape/dtype: {path}")
    if (not np.isfinite(image).all() or image.min() < -1e-6
            or image.max() > 1.0 + 1e-6):
        raise ValueError(f"Invalid normalized pixel values: {path}")


def save_array(path: Path, image: np.ndarray) -> None:
    validate_array(image, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, image, allow_pickle=False)
    os.replace(temp, path)


def read_records(manifest: Path) -> dict[str, dict]:
    if not manifest.exists():
        return {}
    frame = pd.read_csv(
        manifest,
        dtype={"sample_id": "string", "patient_id": "string",
               "series_id": "string", "slice_id": "string", "source_row": "Int64"},
        keep_default_na=False,
    )
    if set(frame.columns) != set(MANIFEST_COLUMNS):
        raise ValueError(f"Existing manifest columns do not match: {manifest}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in {manifest}")
    return {str(row["sample_id"]): row.to_dict() for _, row in frame.iterrows()}


def save_manifest(path: Path, records: dict[str, dict]) -> None:
    ordered = sorted(records.values(), key=lambda r: (r["split"], r["dataset"], r["sample_id"]))
    data = pd.DataFrame(ordered, columns=MANIFEST_COLUMNS)
    write_atomic_text(path, data.to_csv(index=False))


def ensure_config(partition_root: Path) -> None:
    path = partition_root / "preprocessing_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != CONFIG:
            raise ValueError(f"Different preprocessing configuration at: {path}")
    else:
        write_atomic_text(path, json.dumps(CONFIG, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Optional processed/ destination (default: sibling of selected/)")
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Use only first N LIDC and first N M3Dsynth samples per split")
    args = parser.parse_args()
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be positive")

    selected_root = get_data_root()
    root = (args.output_root if args.output_root is not None
            else selected_root.parent / "processed").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "processed_manifest.csv").exists() or (root / "preprocessing_config.json").exists():
        raise RuntimeError(
            "Old-format files found directly in processed/. "
            "Inspect/migrate them before running the new layout."
        )

    splits = ("train", "valid", "test") if args.split == "all" else (args.split,)
    for split in splits:
        partition, _ = target_location(split, "example")
        partition_root = root / partition
        ensure_config(partition_root)
        manifest = partition_root / "processed_manifest.csv"
        records = read_records(manifest)
        dataset = LIDC_M3DSynth_Dataset(split=split)

        indices = list(range(len(dataset)))
        if args.limit_per_class is not None:
            indices = []
            for source in ("lidc", "m3dsynth"):
                matches = dataset.samples.index[dataset.samples["dataset"] == source].tolist()
                indices.extend(matches[:args.limit_per_class])

        new_count = reused_count = 0
        for position, index in enumerate(indices, start=1):
            row = dataset.samples.iloc[index]
            sample_id = str(row["sample_id"])
            if (not sample_id or sample_id in (".", "..")
                    or "/" in sample_id or "\\" in sample_id):
                raise ValueError(f"Unsafe sample ID: {sample_id}")
            actual_partition, relative = target_location(split, sample_id)
            if actual_partition != partition:
                raise AssertionError("Inconsistent partition selection")
            destination = partition_root.joinpath(*relative.parts)
            original, series, slice_id = source_details(row, split)
            record = {
                "sample_id": sample_id,
                "dataset": str(row["dataset"]),
                "label": int(row["label"]),
                "split": split,
                "patient_id": str(row["patient_id"]),
                "series_id": series,
                "slice_id": slice_id,
                "generator": str(row["generator"]),
                "manipulation_type": str(row["manipulation_type"]),
                "source_row": int(row["source_row"]),
                "original_path": original,
                "processed_path": relative.as_posix(),
            }
            if sample_id in records and any(
                str(records[sample_id][key]) != str(value) for key, value in record.items()
            ):
                raise ValueError(f"Conflicting manifest record for {sample_id}")

            if destination.exists():
                existing = np.load(destination, mmap_mode="r", allow_pickle=False)
                validate_array(existing, destination)
                reused_count += 1
            else:
                tensor, target = dataset[index]
                if int(target.item()) != record["label"]:
                    raise ValueError(f"Label mismatch: {sample_id}")
                save_array(destination, tensor.squeeze(0).numpy())
                new_count += 1
            records[sample_id] = record

            if position % 100 == 0 or position == len(indices):
                print(f"{split}: {position}/{len(indices)} | new={new_count} reused={reused_count}", flush=True)

        # The manifest is updated only after successfully finishing this split.
        # In limited test runs this is deliberately a *partial* manifest.
        save_manifest(manifest, records)
        print(f"Completed {split}: manifest={manifest} rows={len(records)}", flush=True)

    print("Original selected/ files have not been modified.", flush=True)
    if args.limit_per_class is not None:
        print("NOTE: --limit-per-class generated partial manifests, not full datasets.", flush=True)


if __name__ == "__main__":
    main()
