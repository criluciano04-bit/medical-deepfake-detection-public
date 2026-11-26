
"""Test BTD-to-LIDC matching on 10 different series."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.matching.btd_lidc import match_btd_to_lidc


load_dotenv(PROJECT_ROOT / ".env")

DATA_ROOT = Path(os.environ["DATA_ROOT"])
LIDC_ROOT = Path(os.environ["LIDC_ROOT"])

BTD_DIR = DATA_ROOT / "btd_selected" / "images" / "real"

NUM_SAMPLES = 30


def select_samples():
    """Select one BTD image from each of 10 distinct series."""

    selected = []
    used_series = set()

    for path in sorted(BTD_DIR.glob("*.npy")):
        series_uid, _ = path.stem.rsplit("__", 1)

        if series_uid in used_series:
            continue

        selected.append(path)
        used_series.add(series_uid)

        if len(selected) == NUM_SAMPLES:
            break

    return selected


def main():

    samples = select_samples()

    if len(samples) != NUM_SAMPLES:
        raise RuntimeError(
            f"Expected {NUM_SAMPLES} distinct series, "
            f"found {len(samples)}"
        )

    passed = 0

    for i, btd_path in enumerate(samples, 1):

        print(f"\n[{i}/{len(samples)}] {btd_path.name}")

        dicom_path = match_btd_to_lidc(
            btd_path=btd_path,
            lidc_root=LIDC_ROOT,
        )

        if dicom_path is not None:
            print(f"LIDC: {dicom_path.name}")
            print(f"Patient: {dicom_path.parent.parent.name}")
            print("PASS")
            passed += 1
        else:
            print("FAIL: no exact match")

    print("\n" + "=" * 40)
    print(f"RESULT: {passed}/{len(samples)} exact matches")


if __name__ == "__main__":
    main()
