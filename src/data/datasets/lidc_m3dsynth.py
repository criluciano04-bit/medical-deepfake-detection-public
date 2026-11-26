from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.paths import get_data_root
from src.data.adapters.lidc import load_lidc_hu
from src.data.adapters.m3dsynth import (
    M3DSynthIntensityMapping,
    load_m3dsynth_image,
)
from src.data.preprocessing import (
    apply_hu_window,
    resize_image,
    normalize_image,
)


class LIDC_M3DSynth_Dataset(Dataset):
    """LIDC + M3Dsynth binary classification dataset."""

    def __init__(
        self,
        split: str,
        lower_hu: float = -1000,
        upper_hu: float = 1000,
        target_size: tuple[int, int] = (512, 512),
    ):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split: {split}")

        if lower_hu >= upper_hu:
            raise ValueError("Invalid HU window bounds.")

        self.split = split
        self.lower_hu = lower_hu
        self.upper_hu = upper_hu
        self.target_size = target_size
        self.data_root = get_data_root()

        # Main manifest: labels, dataset and splits
        manifest = pd.read_csv(
            "metadata/lidc_m3d_manifest.csv",
            dtype={"patient_id": "string"},
        )

        self.samples = manifest.loc[
            manifest["split"] == split
        ].copy()

        if self.samples.empty:
            raise ValueError(f"No samples for split: {split}")

        if not self.samples["dataset"].isin(
            ["lidc", "m3dsynth"]
        ).all():
            raise ValueError("Unexpected dataset in manifest.")

        expected_labels = self.samples["dataset"].map(
            {"lidc": 0, "m3dsynth": 1}
        )

        if not self.samples["label"].eq(
            expected_labels
        ).all():
            raise ValueError("Inconsistent dataset labels.")

        # M3Dsynth metadata: retrieve the series ID
        m3d_selection = pd.read_csv(
            "metadata/m3dsynth/m3dsynth_selected_slices.csv",
            dtype={
                "patient_id": "string",
                "sdir_id": "string",
            },
        )

        # Match the two CSVs using their shared image path
        m3d_selection = m3d_selection.rename(
            columns={
                "image_relative_path": "image_path",
                "split": "metadata_split",
            }
        )

        if m3d_selection["image_path"].duplicated().any():
            raise ValueError(
                "Duplicate M3Dsynth image paths in metadata."
            )

        self.samples = self.samples.merge(
            m3d_selection[
                [
                    "image_path",
                    "patient_id",
                    "sdir_id",
                    "metadata_split",
                ]
            ],
            on=["image_path", "patient_id"],
            how="left",
            validate="many_to_one",
        )

        fake = self.samples["dataset"] == "m3dsynth"

        if self.samples.loc[fake, "sdir_id"].isna().any():
            raise ValueError(
                "Missing M3Dsynth series identifiers."
            )

        if not self.samples.loc[
            fake, "metadata_split"
        ].eq(split).all():
            raise ValueError(
                "M3Dsynth split mismatch between metadata files."
            )

        self.samples = self.samples.reset_index(drop=True)

        # Shared intensity mapping for all M3Dsynth images
        self.m3d_mapping = M3DSynthIntensityMapping(
            "metadata/m3dsynth/audits/"
            "m3dsynth_volume_intensity_mapping.csv"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples.iloc[index]

        dataset_name = sample["dataset"]
        label = int(sample["label"])

        relative_path = PurePosixPath(
            str(sample["image_path"])
        )

        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.parts[:2] != ("images", self.split)
        ):
            raise ValueError(
                f"Invalid image path: {relative_path}"
            )

        if dataset_name == "lidc":
            dataset_root = self.data_root / "lidc_selected"
            image_path = dataset_root.joinpath(
                *relative_path.parts
            )
            image = load_lidc_hu(image_path)

        elif dataset_name == "m3dsynth":
            dataset_root = self.data_root / "m3d_selected"
            image_path = dataset_root.joinpath(
                *relative_path.parts
            )
            image = load_m3dsynth_image(
                image_path=image_path,
                patient_id=str(sample["patient_id"]),
                sdir_id=str(sample["sdir_id"]),
                mapping=self.m3d_mapping,
            )

        else:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}"
            )

        # Shared preprocessing
        image = apply_hu_window(
            image, self.lower_hu, self.upper_hu
        )
        image = resize_image(
            image, self.target_size
        )
        image = normalize_image(
            image, self.lower_hu, self.upper_hu
        )

        # Per-image automatic checks
        if image.shape != self.target_size:
            raise ValueError("Unexpected preprocessed shape.")

        if image.dtype != np.float32:
            raise ValueError("Unexpected preprocessed dtype.")

        if not np.isfinite(image).all():
            raise ValueError("Non-finite preprocessed values.")

        if image.min() < -1e-6 or image.max() > 1.0 + 1e-6:
            raise ValueError(
                "Preprocessed values outside [0, 1]."
            )

        # Convert the preprocessed CT image to a single-channel tensor (1, H, W)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image)
        ).unsqueeze(0)

        label_tensor = torch.tensor(
            label, dtype=torch.long
        )

        return image_tensor, label_tensor
