"""Evaluate one checkpoint on one degraded version of Test 0.

Run as a module from the repository root, e.g.:
    python -m scripts.evaluation.evaluate_robustness \
        --model efficientnet_b0 --checkpoint CHECKPOINT.pt \
        --degradation noise --sigma 0.01 \
        --processed-root PROCESSED_ROOT --device cuda

Degradations are performed on [0, 1] preprocessed grayscale images, before
ImageNet normalization. No undegraded Test 0 evaluation is performed here.
Run six separate conditions per model. No training or checkpoint modifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

from src.data.dataloader import create_dataloader
from src.data.datasets.test_processed import Test_Processed_Dataset
from src.data.model_transforms import prepare_imagenet_input
from src.data.paths import get_data_root
from src.data.degradations import apply_degradation
from src.evaluation.metrics import compute_metrics
from scripts.evaluation.evaluate import file_sha256, load_model, select_device, write_json

MODEL_NAMES = ("efficientnet_b0", "convnext_tiny", "dwt_efficientnet_b0")
CONDITIONS = {"noise": (0.01, 0.03, 0.05), "blur": (0.5, 1.0, 2.0)}
SUMMARY_COLUMNS = [
    "test", "model", "degradation", "sigma", "seed", "n_samples",
    "accuracy", "precision_fake", "recall_fake", "specificity", "f1_fake",
    "roc_auc", "tp", "tn", "fp", "fn", "checkpoint_sha256",
]
OUTPUT_FILENAMES = (
    "predictions.csv", "metrics.json", "confusion_matrix.csv", "evaluation_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Test 0 degradation on one trained checkpoint")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=None,
                        help="Root containing test0/processed_manifest.csv and images/")
    parser.add_argument("--output-root", type=Path, default=Path("results/robustness"))
    parser.add_argument("--degradation", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Smoke test on first N samples; outputs isolated under smoke/")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace outputs for this exact model and condition")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max-samples must be positive")
    # Freeze the approved experimental conditions rather than silently running an
    # unintended severity (math.isclose is deliberately not used here).
    if args.sigma not in CONDITIONS[args.degradation]:
        parser.error(f"sigma for {args.degradation} must be one of {CONDITIONS[args.degradation]}")
    return args


def condition_name(degradation: str, sigma: float) -> str:
    return f"{degradation}_sigma_{sigma:g}"


def write_summary(summary_path: Path, record: dict, overwrite: bool) -> None:
    key_columns = ("test", "model", "degradation", "sigma", "seed")
    new_key = tuple(str(record[name]) for name in key_columns)
    old_rows: list[dict] = []
    if summary_path.is_file():
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != SUMMARY_COLUMNS:
                raise ValueError("Existing robustness summary.csv has an incompatible schema")
            for row in reader:
                if tuple(str(row[name]) for name in key_columns) == new_key:
                    if not overwrite:
                        raise FileExistsError("This robustness evaluation is already in summary.csv")
                else:
                    old_rows.append(row)
    old_rows.append({name: record.get(name) for name in SUMMARY_COLUMNS})
    old_rows.sort(key=lambda row: (row["test"], row["model"], row["degradation"], float(row["sigma"]), int(row["seed"])))
    temp_path = summary_path.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(old_rows)
    os.replace(temp_path, summary_path)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    if args.processed_root is not None:
        processed_root = args.processed_root.expanduser()
    elif os.environ.get("PROCESSED_DATA_ROOT"):
        processed_root = Path(os.environ["PROCESSED_DATA_ROOT"]).expanduser()
    else:
        processed_root = get_data_root().parent / "processed"
    processed_root = processed_root.resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    dataset = Test_Processed_Dataset(test="test0", processed_root=processed_root)
    full_count = len(dataset)
    sample_count = min(args.max_samples, full_count) if args.max_samples is not None else full_count
    test_dataset = Subset(dataset, range(sample_count)) if args.max_samples is not None else dataset
    condition = condition_name(args.degradation, args.sigma)
    output_dir = output_root / ("smoke/test0" if args.max_samples is not None else "test0") / args.model / condition
    expected_ids = dataset.samples.iloc[:sample_count]["sample_id"].astype(str).tolist()
    expected_labels = dataset.samples.iloc[:sample_count]["label"].astype(int).tolist()
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate sample IDs in evaluation set")

    if not args.overwrite and any((output_dir / name).exists() for name in OUTPUT_FILENAMES):
        raise FileExistsError(f"Outputs already exist under {output_dir}; pass --overwrite to replace")
    summary_path = output_root / "summary.csv"
    if args.max_samples is None and summary_path.is_file() and not args.overwrite:
        summary = pd.read_csv(summary_path)
        match = (
            summary["test"].eq("test0") & summary["model"].eq(args.model)
            & summary["degradation"].eq(args.degradation)
            & summary["sigma"].eq(args.sigma) & summary["seed"].eq(args.seed)
        )
        if match.any():
            raise FileExistsError("This condition already exists in summary.csv; pass --overwrite")

    print(f"Test 0 | {args.model} | {condition} | device={device} | samples={sample_count}", flush=True)
    model, checkpoint = load_model(args.model, checkpoint_path, device)
    loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    labels: list[int] = []
    probabilities: list[float] = []
    cursor = 0
    with torch.inference_mode():
        for images, batch_labels in tqdm(loader, desc=f"test0 / {args.model} / {condition}", unit="batch"):
            batch_count = int(images.shape[0])
            batch_ids = expected_ids[cursor:cursor + batch_count]
            if len(batch_ids) != batch_count:
                raise RuntimeError("DataLoader returned more images than expected")
            images = images.to(device, non_blocking=(device.type == "cuda"))
            images = apply_degradation(
                images, degradation=args.degradation, sigma=args.sigma,
                sample_ids=batch_ids, global_seed=args.seed,
            )
            images = prepare_imagenet_input(images)
            logits = model(images)
            if logits.ndim != 2 or logits.shape != (batch_count, 2):
                raise ValueError(f"Expected logits of shape ({batch_count}, 2), got {tuple(logits.shape)}")
            fake_probs = torch.softmax(logits.float(), dim=1)[:, 1]
            probabilities.extend(fake_probs.cpu().tolist())
            labels.extend(batch_labels.tolist())
            cursor += batch_count

    if cursor != sample_count or labels != expected_labels:
        raise RuntimeError("Prediction count/order/labels mismatch; refusing to save")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities")

    metrics = compute_metrics(labels, probabilities)
    predicted = (np.asarray(probabilities) > 0.5).astype(int)
    outcomes = ["TP" if y == 1 and p == 1 else "TN" if y == 0 and p == 0
                else "FP" if y == 0 else "FN" for y, p in zip(labels, predicted)]
    for key, category in (("tp", "TP"), ("tn", "TN"), ("fp", "FP"), ("fn", "FN")):
        if metrics[key] != outcomes.count(category):
            raise RuntimeError(f"Mismatch between metric {key} and individual outcomes")

    predictions = pd.DataFrame({
        "sample_id": expected_ids, "true_label": labels,
        "fake_probability": probabilities, "predicted_label": predicted, "outcome": outcomes,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, float_format="%.17g")
    write_json(output_dir / "metrics.json", metrics)
    pd.DataFrame([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
                 index=["true_real_0", "true_fake_1"],
                 columns=["pred_real_0", "pred_fake_1"]).to_csv(output_dir / "confusion_matrix.csv")
    configuration = {
        "model": args.model, "test": "test0", "degradation": args.degradation,
        "sigma": args.sigma, "seed": args.seed,
        "noise_seed_method": "SHA256(global_seed, sample_id), same standard Gaussian field across sigma",
        "blur_kernel": "separable Gaussian, radius=ceil(3*sigma), reflection padding" if args.degradation == "blur" else None,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("completed_epoch"),
        "checkpoint_best_valid_loss": checkpoint.get("best_valid_loss"),
        "checkpoint_training_config": checkpoint.get("config"),
        "processed_root": str(processed_root),
        "processed_manifest_sha256": file_sha256(processed_root / "test0" / "processed_manifest.csv"),
        "device": str(device), "batch_size": args.batch_size,
        "num_workers": args.num_workers, "n_samples": sample_count,
        "full_test_size": full_count, "smoke_test": args.max_samples is not None,
        "threshold": 0.5, "positive_class": "fake (1)",
        "transform": "degradation on grayscale [0,1], then prepare_imagenet_input",
    }
    write_json(output_dir / "evaluation_config.json", configuration)
    if args.max_samples is None:
        write_summary(summary_path, {
            "test": "test0", "model": args.model, "degradation": args.degradation,
            "sigma": args.sigma, "seed": args.seed, **metrics,
            "checkpoint_sha256": configuration["checkpoint_sha256"],
        }, overwrite=args.overwrite)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
