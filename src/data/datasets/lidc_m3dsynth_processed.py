"""PyTorch Dataset for preprocessed LIDC-IDRI / M3Dsynth CT slices.

Preprocessing is performed once by scripts.data.build_processed.
The training manifest contains train and valid; the Test 0 manifest contains test.
All manifest paths are relative to the directory containing the manifest.
"""

from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.paths import get_data_root


class LIDC_M3DSynth_Processed_Dataset(Dataset):
    """Return (1, 512, 512) float32 CT tensors and integer class labels."""

    def __init__(self, split: str, processed_root: str | Path | None = None):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.split = split
        # Default: sibling processed/ of the selected/ folder configured via DATA_ROOT.
        self.processed_root = (
            Path(processed_root).expanduser()
            if processed_root is not None
            else get_data_root().parent / "processed"
        )
        self.partition_root = self.processed_root / (
            "test0" if split == "test" else "training"
        )
        manifest_path = self.partition_root / "processed_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")

        manifest = pd.read_csv(
            manifest_path,
            dtype={"sample_id": "string", "patient_id": "string"},
        )
        required = {"sample_id", "dataset", "label", "split", "processed_path"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Missing columns in {manifest_path}: {sorted(missing)}")
        if manifest[list(required)].isna().any().any():
            raise ValueError("Processed manifest contains missing required values")
        if manifest["sample_id"].duplicated().any():
            raise ValueError("Duplicate sample_id in processed manifest")

        self.samples = manifest.loc[manifest["split"] == split].copy().reset_index(drop=True)
        if self.samples.empty:
            raise ValueError(f"No processed samples for split: {split}")

        expected_labels = self.samples["dataset"].map({"lidc": 0, "m3dsynth": 1})
        if expected_labels.isna().any() or not self.samples["label"].eq(expected_labels).all():
            raise ValueError("Unexpected dataset or inconsistent labels")

        # Validate the paths once, not for every training epoch.
        self.image_paths: list[Path] = []
        for row in self.samples.itertuples(index=False):
            relative = PurePosixPath(str(row.processed_path))
            if (relative.is_absolute() or ".." in relative.parts
                    or "\\" in str(row.processed_path)
                    or relative.suffix != ".npy"):
                raise ValueError(f"Invalid processed_path: {row.processed_path}")

            if split == "test":
                expected = PurePosixPath("images", f"{row.sample_id}.npy")
            else:
                expected = PurePosixPath(split, f"{row.sample_id}.npy")
            if relative != expected:
                raise ValueError(
                    f"Unexpected processed_path for {row.sample_id}: {relative}"
                )
            self.image_paths.append(self.partition_root.joinpath(*relative.parts))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.load(self.image_paths[index], allow_pickle=False)
        if image.shape != (512, 512) or image.dtype != np.float32:
            raise ValueError(f"Invalid processed image: {self.image_paths[index]}")
        # Per-pixel validity was checked during build_processed.py.
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        label_tensor = torch.tensor(int(self.samples.iloc[index]["label"]), dtype=torch.long)
        return image_tensor, label_tensor
