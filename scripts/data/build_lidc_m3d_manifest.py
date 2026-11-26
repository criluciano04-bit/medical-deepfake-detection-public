import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LIDC_CSV = ROOT / "metadata/lidc/lidc_selected_slices.csv"
M3D_CSV = ROOT / "metadata/m3dsynth/m3dsynth_selected_slices.csv"
OUTPUT = ROOT / "metadata/lidc_m3d_manifest.csv"

COLUMNS = [
    "sample_id",
    "dataset",
    "label",
    "split",
    "patient_id",
    "image_path",
    "generator",
    "manipulation_type",
    "source_row",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    lidc = read_csv(LIDC_CSV)
    m3d = read_csv(M3D_CSV)

    samples = []

    for index, row in enumerate(lidc, start=1):
        samples.append({
            "sample_id": f"lidc_{index:06d}",
            "dataset": "lidc",
            "label": 0,
            "split": row["split"],
            "patient_id": row["PatientID"],
            "image_path": row["relative_path"].replace("\\", "/"),
            "generator": "none",
            "manipulation_type": "none",
            "source_row": index,
        })

    for index, row in enumerate(m3d, start=1):
        samples.append({
            "sample_id": f"m3d_{index:06d}",
            "dataset": "m3dsynth",
            "label": 1,
            "split": row["split"],
            "patient_id": row["patient_id"],
            "image_path": row["image_relative_path"].replace("\\", "/"),
            "generator": row["generator"],
            "manipulation_type": row["manipulation_type"],
            "source_row": index,
        })

    # Verifica preventiva degli split a livello di paziente
    patient_splits = defaultdict(set)
    for sample in samples:
        patient_splits[sample["patient_id"]].add(sample["split"])

    leakage = {
        patient: splits
        for patient, splits in patient_splits.items()
        if len(splits) > 1
    }

    if leakage:
        raise ValueError(
            f"Patient-level leakage rilevato: {len(leakage)} pazienti"
        )

    if len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Sample ID duplicati")

    if any(s["split"] not in {"train", "valid", "test"} for s in samples):
        raise ValueError("Split non riconosciuto")

    # Scrive il file soltanto dopo i controlli
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(samples)

    print(f"Manifest creato: {OUTPUT}")
    print(f"Campioni: {len(samples)}")
    print(f"Pazienti: {len(patient_splits)}")
    print(f"Distribuzione: {dict(Counter((s['split'], s['label']) for s in samples))}")
    print("Patient-level leakage: 0")


if __name__ == "__main__":
    main()