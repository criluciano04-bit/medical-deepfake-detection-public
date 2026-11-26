from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import tifffile

ROOT = Path(__file__).resolve().parents[3]


def load_env(path):
    values = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        key, sep, value = line.partition("=")

        if sep:
            values[key.strip()] = value.strip().strip('"').strip("'")

    return values


env = load_env(ROOT / ".env")

lidc_root = Path(env["LIDC_ROOT"])
m3d_root = Path(env["M3D_SELECTED_ROOT"])

pairs = pd.read_csv(
    ROOT / "results/hu_validation/m3dsynth/select_hu_validation_slices.csv"
)

results = []

for _, row in pairs.iterrows():
    validation_id = row["validation_id"]
    patient = row["patient_id"]
    series_uid = row["SeriesInstanceUID"]
    slice_idx = int(row["slice_idx"])

    series_dir = lidc_root / patient / series_uid
    tiff_path = m3d_root / row["image_relative_path"]

    try:
        # Leggiamo solo gli header per costruire l'ordine anatomico.
        slices = []

        for path in series_dir.glob("*.dcm"):
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=[
                    "ImagePositionPatient",
                    "InstanceNumber",
                ],
            )

            slices.append({
                "path": path,
                "z": float(ds.ImagePositionPatient[2]),
                "instance": int(ds.InstanceNumber),
            })

        # M3Dsynth: ipotesi di ordinamento Z decrescente.
        slices.sort(key=lambda s: s["z"], reverse=True)

        if not 0 <= slice_idx < len(slices):
            raise ValueError(
                f"slice_idx={slice_idx}, "
                f"numero DICOM={len(slices)}"
            )

        match = slices[slice_idx]

        dicom = pydicom.dcmread(
            match["path"]
        ).pixel_array.astype(np.float64)

        tiff = tifffile.imread(
            tiff_path
        ).astype(np.float64)

        if dicom.shape != tiff.shape:
            raise ValueError(
                f"Shape diverse: {dicom.shape} vs {tiff.shape}"
            )

        correlation = np.corrcoef(
            dicom.ravel(),
            tiff.ravel()
        )[0, 1]

        results.append({
            "validation_id": validation_id,
            "patient_id": patient,
            "slice_idx": slice_idx,
            "dicom_instance": match["instance"],
            "dicom_filename": match["path"].name,
            "correlation": correlation,
            "status": "OK" if correlation > 0.99 else "CHECK",
        })

        print(
            f"{validation_id} | {patient} | "
            f"slice_idx={slice_idx} | "
            f"InstanceNumber={match['instance']} | "
            f"corr={correlation:.9f} | "
            f"{results[-1]['status']}"
        )

    except Exception as exc:
        results.append({
            "validation_id": validation_id,
            "patient_id": patient,
            "slice_idx": slice_idx,
            "dicom_instance": None,
            "dicom_filename": None,
            "correlation": None,
            "status": f"ERROR: {exc}",
        })

        print(f"{validation_id} | ERROR: {exc}")


df = pd.DataFrame(results)

output_path = ROOT / "results/hu_validation/m3dsynth/check_slice_matching.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("SUMMARY")
print(df["status"].value_counts().to_string())
print(f"\nResults saved to: {output_path}")

