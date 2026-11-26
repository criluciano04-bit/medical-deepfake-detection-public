
"""Analyze intensity relationships in BTD CTGAN Removal pairs."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# --------------------------------------------------
# Paths
# --------------------------------------------------

REPO = Path(__file__).resolve().parents[3]

load_dotenv(REPO / ".env")

DATA_ROOT = Path(os.environ["DATA_ROOT"])
BTD_ROOT = DATA_ROOT / "btd_selected"

RESULTS = REPO / "results" / "hu_validation" / "btd"
SAMPLES = RESULTS / "sample_pairs.csv"
OUTPUT = RESULTS / "ctgan_removal_diagnostics.csv"


# --------------------------------------------------
# Resolve image paths
# --------------------------------------------------

def resolve_image(value):
    path = Path(str(value).replace("\\", "/"))

    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    if ".." in path.parts:
        raise ValueError(f"Unsafe path: {value}")

    candidates = [
        BTD_ROOT / path,
        BTD_ROOT / "images" / path,
        DATA_ROOT / path,
        REPO / path,
    ]

    matches = list(dict.fromkeys(
        p.resolve() for p in candidates if p.is_file()
    ))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Cannot uniquely resolve {value}: {matches}"
        )

    return matches[0]


# --------------------------------------------------
# Intensity analysis
# --------------------------------------------------

def analyze_pair(real, fake):

    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)

    if real.shape != fake.shape:
        raise ValueError("Real and Fake shapes differ")

    if not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise ValueError("Non-finite pixel values found")

    x = real.ravel()
    y = fake.ravel()

    # Fit: Fake = a * Real + b
    x_mean = x.mean()
    y_mean = y.mean()

    variance = np.mean((x - x_mean) ** 2)

    if variance == 0:
        raise ValueError("Real image has zero variance")

    covariance = np.mean(
        (x - x_mean) * (y - y_mean)
    )

    slope = covariance / variance
    intercept = y_mean - slope * x_mean

    predicted_fake = slope * real + intercept

    # Correlation
    pearson_r = np.corrcoef(x, y)[0, 1]

    # Errors
    mae_original = np.mean(np.abs(fake - real))
    mae_after_fit = np.mean(
        np.abs(fake - predicted_fake)
    )

    reduction = (
        100 * (mae_original - mae_after_fit) / mae_original
        if mae_original > 0
        else 0.0
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "pearson_r": pearson_r,
        "mae_original": mae_original,
        "mae_after_linear_fit": mae_after_fit,
        "linear_fit_mae_reduction_pct": reduction,
        "real_mean": x_mean,
        "fake_mean": y_mean,
        "real_std": x.std(),
        "fake_std": y.std(),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    df = pd.read_csv(
        SAMPLES,
        dtype={"series_uid": str}
    )

    required = {
        "attack",
        "real_path",
        "ctgan_path",
        "series_uid",
        "slice_number",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.loc[
        df["attack"].str.lower() == "removal"
    ].copy()

    if df.empty:
        raise ValueError("No CTGAN Removal samples found")

    records = []

    for _, row in df.iterrows():

        real_path = resolve_image(row["real_path"])
        fake_path = resolve_image(row["ctgan_path"])

        real = np.load(real_path, allow_pickle=False)
        fake = np.load(fake_path, allow_pickle=False)

        metrics = analyze_pair(real, fake)

        records.append({
            "series_uid": row["series_uid"],
            "slice_number": row["slice_number"],
            **metrics,
        })

        print(
            f"slice {row['slice_number']}: "
            f"a={metrics['slope']:.4f}, "
            f"b={metrics['intercept']:.2f}, "
            f"r={metrics['pearson_r']:.4f}, "
            f"MAE={metrics['mae_original']:.2f}, "
            f"MAE after fit={metrics['mae_after_linear_fit']:.2f}, "
            f"reduction={metrics['linear_fit_mae_reduction_pct']:.2f}%"
        )

    result = pd.DataFrame(records)

    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    columns = [
        "slope",
        "intercept",
        "pearson_r",
        "mae_original",
        "mae_after_linear_fit",
        "linear_fit_mae_reduction_pct",
    ]

    print("\nMedian across CTGAN Removal samples:")
    print(
        result[columns]
        .median()
        .round(4)
        .to_string()
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
