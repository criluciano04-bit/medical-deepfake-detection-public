"""Generic PyTorch Dataset for already materialized evaluation sets.

Expected layout under processed_root (default: sibling of DATA_ROOT/selected):
    test1/images/<sample_id>.npy
    test1/processed_manifest.csv
    test2/images/<sample_id>.npy
    test2/processed_manifest.csv

No preprocessing, class balancing, or class-specific filtering is performed.
Runpod and local machines can pass --processed-root to the calling evaluation script.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.paths import get_data_root


class Test_Processed_Dataset(Dataset):
    """Read (1, 512, 512) float32 images and 0/1 labels from testN/ manifests."""

    def __init__(self, test: str = "test1", processed_root: str | Path | None = None):
        # Test 0 can be read here as well, although the existing LIDC/M3Dsynth
        # processed Dataset already supports it.
        if not isinstance(test, str) or not re.fullmatch(r"test(?:0|[1-9][0-9]*)", test):
            raise ValueError("test must be 'test0', 'test1', 'test2', ...")

        self.test = test
        self.processed_root = (
            Path(processed_root).expanduser()
            if processed_root is not None
            else get_data_root().parent / "processed"
        )
        self.partition_root = self.processed_root / test
        manifest_path = self.partition_root / "processed_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")

        manifest = pd.read_csv(
            manifest_path,
            dtype={"sample_id": "string", "patient_id": "string", "processed_path": "string"},
            keep_default_na=False,
        )
        required = {"sample_id", "dataset", "label", "split", "processed_path"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Missing columns in {manifest_path}: {sorted(missing)}")
        if manifest.empty:
            raise ValueError(f"Processed manifest is empty: {manifest_path}")
        if manifest["sample_id"].astype(str).eq("").any() or manifest["processed_path"].astype(str).eq("").any():
            raise ValueError("Empty sample_id or processed_path")
        if manifest["sample_id"].duplicated().any():
            raise ValueError("Duplicate sample_id in processed manifest")

        # The Test 0 manifest was created with split='test'; external tests
        # use split='test1', 'test2', etc. Preserve both conventions.
        expected_split = "test" if test == "test0" else test
        if not manifest["split"].eq(expected_split).all():
            raise ValueError(f"Unexpected split: expected {expected_split!r} in {manifest_path}")
        labels = pd.to_numeric(manifest["label"], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("Expected binary numeric labels 0=real and 1=fake")
        manifest["label"] = labels.astype(np.int64)
        self.samples = manifest.reset_index(drop=True)

        self.image_paths: list[Path] = []
        for row in self.samples.itertuples(index=False):
            sample_id = str(row.sample_id)
            if not sample_id or sample_id in {".", ".."} or any(c in sample_id for c in "/\\:"):
                raise ValueError(f"Unsafe sample_id: {sample_id}")
            raw_relative = str(row.processed_path)
            relative = PurePosixPath(raw_relative)
            if ("\\" in raw_relative or ":" in raw_relative or relative.is_absolute()
                    or ".." in relative.parts or relative != PurePosixPath("images", f"{sample_id}.npy")):
                raise ValueError(f"Unexpected processed_path for {sample_id}: {raw_relative}")
            self.image_paths.append(self.partition_root.joinpath(*relative.parts))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[index]
        image = np.load(path, allow_pickle=False)
        if image.shape != (512, 512) or image.dtype != np.float32:
            raise ValueError(f"Invalid image shape/dtype: {path}: {image.shape}/{image.dtype}")
        if not np.isfinite(image).all() or image.min() < -1e-6 or image.max() > 1.0 + 1e-6:
            raise ValueError(f"Invalid normalized pixel range: {path}")
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        label_tensor = torch.tensor(int(self.samples.iloc[index]["label"]), dtype=torch.long)
        return image_tensor, label_tensor
