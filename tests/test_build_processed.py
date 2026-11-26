import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from src.data.paths import get_data_root
from src.data.datasets.lidc_m3dsynth import LIDC_M3DSynth_Dataset


processed_root = get_data_root().parent / "processed"

training_root = processed_root / "training"
test0_root = processed_root / "test0"

# 1. Verifica delle configurazioni
training_config = json.loads(
    (training_root / "preprocessing_config.json").read_text()
)
test0_config = json.loads(
    (test0_root / "preprocessing_config.json").read_text()
)

assert training_config == test0_config
print("OK - Configurazioni identiche")

# 2. Verifica dei manifest e delle immagini
total_checked = 0

for split in ("train", "valid", "test"):
    folder = training_root if split != "test" else test0_root

    manifest = pd.read_csv(
        folder / "processed_manifest.csv",
        dtype={"sample_id": "string"},
    )
    rows = manifest[manifest["split"] == split]

    assert len(rows) == 4, f"Numero errato: {split}"
    assert rows["sample_id"].is_unique

    original_dataset = LIDC_M3DSynth_Dataset(split=split)

    original_indices = {
        str(row["sample_id"]): index
        for index, row in original_dataset.samples.iterrows()
    }

    for _, row in rows.iterrows():
        sample_id = str(row["sample_id"])
        relative = PurePosixPath(str(row["processed_path"]))

        assert not relative.is_absolute()
        assert ".." not in relative.parts

        image_path = folder.joinpath(*relative.parts)
        assert image_path.is_file(), f"File mancante: {image_path}"

        processed = np.load(image_path, allow_pickle=False)

        assert processed.shape == (512, 512)
        assert processed.dtype == np.float32
        assert np.isfinite(processed).all()
        assert 0 <= processed.min() <= processed.max() <= 1

        original_index = original_indices[sample_id]
        original_tensor, original_label = original_dataset[original_index]

        original = original_tensor.squeeze(0).numpy()

        assert int(original_label.item()) == int(row["label"])

        assert np.array_equal(original, processed), (
            f"Immagine differente: {sample_id}"
        )

        total_checked += 1

    print(f"OK - {split}: {len(rows)} immagini identiche")

assert total_checked == 12

print(f"\nVERIFICA COMPLETATA: {total_checked} immagini corrette")
