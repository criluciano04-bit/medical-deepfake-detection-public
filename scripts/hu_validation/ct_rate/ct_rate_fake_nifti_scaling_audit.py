"""Audit dello scaling NIfTI dei 150 volumi CT-RATE Generated selezionati.

Eseguire dalla radice del repository:
    python ct_rate_fake_nifti_scaling_audit.py

Legge soltanto gli header .nii.gz, non modifica i file su Drive.
Salva results/hu_validation/ct_rate/ct_rate_fake_nifti_scaling_audit.csv.
"""

import csv
from collections import Counter
from pathlib import Path

import nibabel as nib

from src.data.paths import get_data_root


def main() -> None:
    # get_data_root() punta alla cartella selected/ configurata nel progetto.
    selected_root = get_data_root()
    ct_rate_root = selected_root / "CT-RATE_selected"
    raw_volumes = selected_root.parent / "raw" / "CT-RATE_generated" / "volumes"

    manifests = list(ct_rate_root.rglob("ct_rate_fake_extraction_manifest.csv"))
    if len(manifests) != 1:
        raise RuntimeError(f"Atteso 1 manifest di estrazione fake, trovati {len(manifests)}")

    with manifests[0].open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 2250:
        raise ValueError(f"Attese 2250 slice nel manifest, trovate {len(rows)}")

    volume_ids = sorted({row["synthetic_volume_id"].strip() for row in rows}, key=int)
    if len(volume_ids) != 150:
        raise ValueError(f"Attesi 150 volumi selezionati, trovati {len(volume_ids)}")

    results = []
    for volume_id in volume_ids:
        path = raw_volumes / f"{volume_id}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Volume originale mancante: {path}")

        # nib.load legge l'header senza caricare l'intero volume in RAM.
        image = nib.load(str(path))
        slope = float(image.dataobj.slope)
        intercept = float(image.dataobj.inter)
        results.append({
            "synthetic_volume_id": volume_id,
            "slope": slope,
            "intercept": intercept,
        })

    counts = Counter((r["slope"], r["intercept"]) for r in results)
    anomalies = [r for r in results if r["slope"] != 1.0 or r["intercept"] != 0.0]

    print("Volumi controllati:", len(results))
    print("\nCombinazioni slope/intercept:")
    for (slope, intercept), count in sorted(counts.items()):
        print(f"  slope={slope}, intercept={intercept}: {count} volumi")
    print("\nVolumi con scaling NIfTI non neutro:", len(anomalies))
    for row in anomalies:
        print(row)

    output = (
        Path.cwd() / "results" / "hu_validation" / "ct_rate"
        / "ct_rate_fake_nifti_scaling_audit.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["synthetic_volume_id", "slope", "intercept"])
        writer.writeheader()
        writer.writerows(results)
    print("\nAudit salvato in:", output)


if __name__ == "__main__":
    main()
