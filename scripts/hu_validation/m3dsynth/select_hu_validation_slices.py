
"""Select 30 reproducible M3Dsynth slices for HU validation."""

from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]

SLICES_CSV = REPO / "metadata/m3dsynth/m3dsynth_selected_slices.csv"
MAPPING_CSV = REPO / "metadata/m3dsynth/audits/m3dsynth_volume_intensity_mapping.csv"
OUTPUT_CSV = REPO / "results/hu_validation/m3dsynth/select_hu_validation_slices.csv"

SEED = 42

QUOTAS = {
    ("uint16_identity", 0): 6,
    ("volume_min", 1024): 6,
    ("volume_min", 2000): 6,
    ("volume_min", 2048): 6,
    ("corner_median", 1024): 3,
    ("corner_median", 2048): 3,
}

GENERATORS = ("cycle", "diffusion", "pix2pix")
TARGET_PER_GENERATOR = 10


def select(slices, mapping):
    s = slices.copy()
    m = mapping.copy()

    s["sdir_id"] = s["sdir_id"].astype("int64")
    m["M3Dsynth_sdir_id"] = m["M3Dsynth_sdir_id"].astype("int64")

    df = s.merge(
        m,
        left_on=["patient_id", "sdir_id"],
        right_on=["PatientID", "M3Dsynth_sdir_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if (df["_merge"] != "both").any():
        raise ValueError("Some slices have no volume mapping")

    df = df.drop(columns="_merge")
    df["offset"] = df["offset"].astype(int)
    df["stratum"] = list(zip(df["method"], df["offset"]))
    df = df[df["stratum"].isin(QUOTAS)].copy()

    rng = np.random.default_rng(SEED)

    strata_order = sorted(
        QUOTAS,
        key=lambda st: (sum(df["stratum"] == st), st),
    )

    groups = {
        (st, g): df[
            (df["stratum"] == st) & (df["generator"] == g)
        ].index.to_numpy()
        for st in QUOTAS
        for g in GENERATORS
    }

    best = None
    best_score = float("inf")

    for _ in range(3000):
        selected = []
        used_patients = set()
        counts_g = {g: 0 for g in GENERATORS}
        counts_type = {"inj": 0, "rem": 0}
        valid = True

        for st in strata_order:
            for _slot in range(QUOTAS[st]):
                options = []

                for g in GENERATORS:
                    if counts_g[g] >= TARGET_PER_GENERATOR:
                        continue

                    for i in groups[(st, g)]:
                        i = int(i)
                        patient = df.at[i, "patient_id"]

                        if patient in used_patients:
                            continue

                        t = df.at[i, "manipulation_type"]

                        penalty = (
                            8 * counts_g[g]
                            + 2 * counts_type.get(t, 0)
                            + rng.uniform(0, 6)
                        )

                        options.append((penalty, i, g, t))

                if not options:
                    valid = False
                    break

                _, i, g, t = min(options)

                selected.append(i)
                used_patients.add(df.at[i, "patient_id"])
                counts_g[g] += 1
                counts_type[t] = counts_type.get(t, 0) + 1

            if not valid:
                break

        if not valid:
            continue

        if any(counts_g[g] != TARGET_PER_GENERATOR for g in GENERATORS):
            continue

        score = abs(counts_type["inj"] - counts_type["rem"])

        if score < best_score:
            best = selected
            best_score = score

        if score == 0:
            break

    if best is None:
        raise RuntimeError("Could not find a valid sample")

    result = df.loc[best].copy()

    result = result.sort_values(
        ["method", "offset", "generator", "patient_id", "slice_idx"]
    ).reset_index(drop=True)

    result.insert(
        0,
        "validation_id",
        [f"HU-{i:03d}" for i in range(1, len(result) + 1)],
    )

    assert len(result) == 30
    assert result["patient_id"].nunique() == 30
    assert result.groupby(["method", "offset"]).size().to_dict() == QUOTAS
    assert all(
        result["generator"].value_counts().get(g, 0) == 10
        for g in GENERATORS
    )

    columns = [
        "validation_id",
        "split",
        "patient_id",
        "generator",
        "manipulation_type",
        "img_id",
        "sdir_id",
        "slice_idx",
        "method",
        "offset",
        "SeriesInstanceUID",
        "status",
        "slope_values",
        "intercept_values",
        "clipping_occurred",
        "clipped_pixel_count",
        "image_relative_path",
        "mask_relative_path",
    ]

    return result[[c for c in columns if c in result.columns]]


def main():
    slices = pd.read_csv(SLICES_CSV)
    mapping = pd.read_csv(MAPPING_CSV)

    result = select(slices, mapping)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(result)} slices to {OUTPUT_CSV}")

    print("\nMETHOD x OFFSET:")
    print(result.groupby(["method", "offset"]).size())

    print("\nGENERATORS:")
    print(result["generator"].value_counts())

    print("\nMANIPULATIONS:")
    print(result["manipulation_type"].value_counts())

    print("\nCLIPPING:")
    print(result["clipping_occurred"].value_counts(dropna=False))

    print("\nDISTINCT PATIENTS:")
    print(result["patient_id"].nunique())


if __name__ == "__main__":
    main()

