"""Compare matched BTD Real and Fake slices without modifying the originals.

Uses DATA_ROOT from the repository .env; selected files are located through
btd_selected_manifest.csv, not inferred solely from filenames.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[3]
load_dotenv(REPO / ".env")

DATA_ROOT = Path(os.environ["DATA_ROOT"])
BTD_ROOT = DATA_ROOT / "btd_selected"
MANIFEST = BTD_ROOT / "metadata" / "btd_selected_manifest.csv"
if not MANIFEST.is_file():
    MANIFEST = REPO / "metadata" / "btd" / "btd_selected_manifest.csv"

RESULT_DIR = REPO / "results" / "hu_validation" / "btd"
FIGURE_DIR = RESULT_DIR / "difference_maps"
SELECTION_CSV = RESULT_DIR / "sample_pairs.csv"
METRICS_CSV = RESULT_DIR / "pixel_comparison.csv"

SAMPLE_SIZE = {"injection": 4, "removal": 8}
TOLERANCE = 1e-4  # Tolerance in stored intensity units; HU only when validated.
CLIP_MIN, CLIP_MAX = -700.0, 2000.0
KEY = ["group", "series_uid", "slice_number"]


def locate_image(relative_path: str) -> Path:
    """Support both known BTD layouts; never silently select another image."""
    relative = Path(str(relative_path).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe selected_path: {relative_path}")
    paths = [BTD_ROOT / relative, BTD_ROOT / "images" / relative]
    existing = [p for p in paths if p.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file for {relative_path}; found {existing}. "
            f"Checked: {paths}"
        )
    return existing[0]


def build_triples(manifest: pd.DataFrame, attack: str) -> pd.DataFrame:
    real = manifest[manifest["label"] == "real"][KEY + ["selected_path"]]
    gan = manifest[(manifest["label"] == "fake_ctgan") &
                   (manifest["manipulations"] == attack)][KEY + ["selected_path"]]
    sd = manifest[(manifest["label"] == "fake_sd") &
                  (manifest["manipulations"] == attack)][KEY + ["selected_path"]]
    for label, frame in [("real", real), ("CTGAN", gan), ("SD", sd)]:
        if frame.duplicated(KEY).any():
            raise ValueError(f"Duplicate {label} join keys: {attack}")
    pairs = (
        real.rename(columns={"selected_path": "real_path"})
        .merge(gan.rename(columns={"selected_path": "ctgan_path"}), on=KEY,
               validate="one_to_one")
        .merge(sd.rename(columns={"selected_path": "sd_path"}), on=KEY,
               validate="one_to_one")
    )
    pairs.insert(0, "attack", attack)
    return pairs


def stable_rank(row: pd.Series) -> str:
    text = f"BTD-validation-v1|{row['attack']}|{row['group']}|{row['series_uid']}|{row['slice_number']}"
    return hashlib.sha256(text.encode()).hexdigest()


def choose_sample(pairs: pd.DataFrame, count: int) -> pd.DataFrame:
    """Deterministically favor different series, then different groups."""
    pairs = pairs.copy()
    pairs["rank"] = pairs.apply(stable_rank, axis=1)
    pairs = pairs.sort_values("rank")
    selected = []
    selected_keys = set()
    used_series = set()
    used_groups = set()
    for _ in range(min(count, len(pairs))):
        remaining = pairs.loc[~pairs.index.isin(selected_keys)]
        if remaining.empty:
            break
        # Prefer an unseen series; if tied, an unseen group.
        pool = remaining.loc[~remaining["series_uid"].isin(used_series)]
        if pool.empty:
            pool = remaining
        other_group = pool.loc[~pool["group"].isin(used_groups)]
        if not other_group.empty:
            pool = other_group
        idx = pool.index[0]
        row = pairs.loc[idx]
        selected.append(row)
        selected_keys.add(idx)
        used_series.add(row["series_uid"])
        used_groups.add(row["group"])
    return pd.DataFrame(selected).drop(columns=["rank"], errors="ignore")


def comparison(reference: np.ndarray, fake: np.ndarray) -> dict:
    """Compute metrics for two images in the same coordinate/pixel grid."""
    if reference.shape != fake.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {fake.shape}")
    real = reference.astype(np.float64)
    altered = fake.astype(np.float64)
    if not np.isfinite(real).all() or not np.isfinite(altered).all():
        raise ValueError("Encountered non-finite pixel values")
    difference = np.abs(real - altered)
    unchanged = difference <= TOLERANCE
    return {
        "exact_equal": bool(np.array_equal(reference, fake)),
        "equal_fraction_tolerance": float(unchanged.mean()),
        "mae": float(difference.mean()),
        "p99_absolute_difference": float(np.percentile(difference, 99)),
        "changed_fraction": float((~unchanged).mean()),
        "difference": difference,
    }


def save_visualization(real: np.ndarray, fake: np.ndarray, stem: str) -> None:
    """Display originals under identical windowing and direct/clipped difference maps."""
    real64 = real.astype(np.float64)
    fake64 = fake.astype(np.float64)
    clipped = np.clip(real64, CLIP_MIN, CLIP_MAX)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    axes[0].imshow(real64, cmap="gray", vmin=CLIP_MIN, vmax=200)
    axes[0].set_title("Real (shared display window)")
    axes[1].imshow(fake64, cmap="gray", vmin=CLIP_MIN, vmax=200)
    axes[1].set_title("Fake (shared display window)")
    axes[2].imshow(np.abs(real64 - fake64), cmap="magma", vmin=0)
    axes[2].set_title("|Real − Fake|")
    axes[3].imshow(np.abs(clipped - fake64), cmap="magma", vmin=0)
    axes[3].set_title("|clip(Real) − Fake|")
    for ax in axes:
        ax.axis("off")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"BTD manifest not found at {MANIFEST}")
    manifest = pd.read_csv(MANIFEST, dtype={"series_uid": str, "group": str,
                                           "selected_path": str, "label": str})
    needed = set(KEY + ["label", "manipulations", "selected_path"])
    if not needed.issubset(manifest.columns):
        raise ValueError(f"Manifest missing columns: {needed - set(manifest.columns)}")
    if manifest[KEY + ["selected_path"]].isna().any().any():
        raise ValueError("Missing pairing keys or selected_path in manifest")

    chosen = []
    for attack, count in SAMPLE_SIZE.items():
        available = build_triples(manifest, attack)
        print(f"{attack}: {len(available)} eligible Real–CTGAN–SD triples "
              f"({available['series_uid'].nunique()} series)")
        if len(available) < count:
            raise RuntimeError(f"Cannot select {count} {attack} triples")
        chosen.append(choose_sample(available, count))
    sample = pd.concat(chosen, ignore_index=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SELECTION_CSV, index=False)

    metrics_rows = []
    for i, row in sample.iterrows():
        real_path = locate_image(row["real_path"])
        real = np.load(real_path, allow_pickle=False)
        for generator, column in [("ctgan", "ctgan_path"), ("sd", "sd_path")]:
            fake_path = locate_image(row[column])
            fake = np.load(fake_path, allow_pickle=False)
            raw = comparison(real, fake)
            clipped = comparison(np.clip(real.astype(np.float64), CLIP_MIN, CLIP_MAX), fake)
            stem = f"{row['attack']}_{i:02d}_{generator}"
            save_visualization(real, fake, stem)
            metrics_rows.append({
                "attack": row["attack"], "group": row["group"],
                "series_uid": row["series_uid"],
                "slice_number": int(row["slice_number"]),
                "generator": generator,
                "real_path": row["real_path"], "fake_path": row[column],
                "real_dtype": str(real.dtype), "fake_dtype": str(fake.dtype),
                "real_min": float(np.min(real)), "real_max": float(np.max(real)),
                "fake_min": float(np.min(fake)), "fake_max": float(np.max(fake)),
                "exact_equal_direct": raw["exact_equal"],
                "equal_fraction_direct": raw["equal_fraction_tolerance"],
                "mae_direct": raw["mae"],
                "p99_diff_direct": raw["p99_absolute_difference"],
                "changed_fraction_direct": raw["changed_fraction"],
                "exact_equal_after_real_clip": clipped["exact_equal"],
                "equal_fraction_after_real_clip": clipped["equal_fraction_tolerance"],
                "mae_after_real_clip": clipped["mae"],
                "p99_diff_after_real_clip": clipped["p99_absolute_difference"],
                "changed_fraction_after_real_clip": clipped["changed_fraction"],
                "figure": str((FIGURE_DIR / f"{stem}.png").relative_to(REPO)),
            })
            print(f"[{len(metrics_rows):02d}/24] {row['attack']} {generator} "
                  f"slice={row['slice_number']} | "
                  f"equal direct={raw['equal_fraction_tolerance']:.2%} "
                  f"clipped={clipped['equal_fraction_tolerance']:.2%}")

    results = pd.DataFrame(metrics_rows)
    results.to_csv(METRICS_CSV, index=False)
    print("\nSummary by attack/generator (median across pairs):")
    print(results.groupby(["attack", "generator"])[
        ["equal_fraction_direct", "equal_fraction_after_real_clip",
         "mae_direct", "mae_after_real_clip"]].median().round(4).to_string())
    print(f"\nSelection: {SELECTION_CSV}\nMetrics: {METRICS_CSV}")
    print(f"Figures: {FIGURE_DIR}")
    print("Note: global exact equality alone cannot establish where a tumor was edited.")


if __name__ == "__main__":
    main()
