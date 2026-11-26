"""Evaluate one trained model checkpoint on one materialized test set.

Run from repository root:
  python -m scripts.evaluation.evaluate --model efficientnet_b0 \
      --test test0 --checkpoint /path/to/efficientnet_b0_best.pt \
      --processed-root /path/to/processed --device cuda

Use --max-samples 32 for a smoke test; smoke outputs are isolated from
full-test outputs. No training, augmentation, tuning, or dataset balancing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data.dataloader import create_dataloader
from src.data.datasets.test_processed import Test_Processed_Dataset
from src.data.model_transforms import prepare_imagenet_input
from src.data.paths import get_data_root
from src.evaluation.metrics import compute_metrics
from src.models.model_factory import create_model

MODEL_NAMES = ("efficientnet_b0", "convnext_tiny", "dwt_efficientnet_b0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint on one processed test set")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--test", required=True, choices=("test0", "test1", "test2"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to selected *_best.pt checkpoint")
    parser.add_argument("--processed-root", type=Path, default=None,
                        help="Root directory containing test0/, test1/, test2/")
    parser.add_argument("--output-root", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Smoke test: evaluate only first N examples and save under smoke/")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing outputs")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max-samples must be positive")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_device(requested: str) -> torch.device:
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but no CUDA GPU is available")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu")


def load_model(model_name: str, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected training checkpoint with model_state_dict")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict) or config.get("model") != model_name:
        raise ValueError(f"Checkpoint model mismatch: expected {model_name}, got {config.get('model') if isinstance(config, dict) else config}")
    model = create_model(model_name, num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_summary(summary_path: Path, record: dict) -> None:
    """Full-test summary only; replace model/test row if --overwrite was requested."""
    columns = ["test", "model", "n_samples", "accuracy", "precision_fake", "recall_fake",
               "specificity", "f1_fake", "roc_auc", "tp", "tn", "fp", "fn", "checkpoint_sha256"]
    old_rows = []
    if summary_path.is_file():
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns:
                raise ValueError("Existing summary.csv has an incompatible schema")
            old_rows = [row for row in reader if (row["test"], row["model"]) != (record["test"], record["model"])]
    old_rows.append({col: record.get(col) for col in columns})
    old_rows.sort(key=lambda row: (row["test"], row["model"]))
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(old_rows)
    os.replace(temporary, summary_path)


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

    dataset = Test_Processed_Dataset(test=args.test, processed_root=processed_root)
    full_count = len(dataset)
    sample_count = min(args.max_samples, full_count) if args.max_samples is not None else full_count
    if args.max_samples is not None:
        from torch.utils.data import Subset
        test_dataset = Subset(dataset, list(range(sample_count)))
        output_dir = output_root / "smoke" / args.test / args.model
    else:
        test_dataset = dataset
        output_dir = output_root / args.test / args.model

    expected_ids = dataset.samples.iloc[:sample_count]["sample_id"].astype(str).tolist()
    expected_labels = dataset.samples.iloc[:sample_count]["label"].astype(int).tolist()
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Duplicated sample_id in evaluation subset")

    output_files = [output_dir / name for name in
                    ("predictions.csv", "metrics.json", "confusion_matrix.csv", "evaluation_config.json")]
    if not args.overwrite and any(path.exists() for path in output_files):
        raise FileExistsError(f"Results already present in {output_dir}; pass --overwrite to replace")
    if args.max_samples is None and not args.overwrite and (output_root / "summary.csv").is_file():
        summary = pd.read_csv(output_root / "summary.csv")
        if "test" in summary and "model" in summary and ((summary["test"] == args.test) & (summary["model"] == args.model)).any():
            raise FileExistsError("Evaluation already present in summary.csv; pass --overwrite to replace")

    print(f"Test: {args.test} | model: {args.model} | device: {device} | samples: {sample_count}", flush=True)
    model, checkpoint = load_model(args.model, checkpoint_path, device)
    loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    labels: list[int] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        for images, batch_labels in tqdm(loader, desc=f"{args.test} / {args.model}", unit="batch"):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            images = prepare_imagenet_input(images)  # identical to training
            logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"Expected (B, 2) logits, got {tuple(logits.shape)}")
            fake_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
            probabilities.extend(fake_probabilities.cpu().tolist())
            labels.extend(batch_labels.tolist())

    if len(labels) != sample_count or labels != expected_labels:
        raise RuntimeError("Predictions and manifest order/labels mismatch; refusing to save results")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities detected")
    metrics = compute_metrics(labels, probabilities)
    predicted = (np.asarray(probabilities) > 0.5).astype(int)
    outcomes = ["TP" if y == 1 and p == 1 else "TN" if y == 0 and p == 0
                else "FP" if y == 0 else "FN" for y, p in zip(labels, predicted)]
    predictions = pd.DataFrame({
        "sample_id": expected_ids, "true_label": labels, "fake_probability": probabilities,
        "predicted_label": predicted, "outcome": outcomes,
    })
    if sum(outcomes.count(x) for x in ("TP", "TN", "FP", "FN")) != sample_count:
        raise RuntimeError("Invalid outcome records")
    for key, category in (("tp", "TP"), ("tn", "TN"), ("fp", "FP"), ("fn", "FN")):
        if metrics[key] != outcomes.count(category):
            raise RuntimeError(f"Metrics/outcomes mismatch: {key}")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, float_format="%.17g")
    write_json(output_dir / "metrics.json", metrics)
    pd.DataFrame([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
                 index=["true_real_0", "true_fake_1"],
                 columns=["pred_real_0", "pred_fake_1"]).to_csv(output_dir / "confusion_matrix.csv")
    config = {
        "model": args.model, "test": args.test, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("completed_epoch"),
        "checkpoint_best_valid_loss": checkpoint.get("best_valid_loss"),
        "checkpoint_training_config": checkpoint.get("config"),
        "processed_root": str(processed_root),
        "processed_manifest_sha256": file_sha256(processed_root / args.test / "processed_manifest.csv"),
        "device": str(device), "batch_size": args.batch_size, "num_workers": args.num_workers,
        "n_samples": sample_count, "full_test_size": full_count,
        "smoke_test": args.max_samples is not None,
        "threshold": 0.5, "positive_class": "fake (1)",
        "transform": "prepare_imagenet_input (repeat 1->3, ImageNet normalization)",
    }
    write_json(output_dir / "evaluation_config.json", config)
    if args.max_samples is None:
        write_summary(output_root / "summary.csv", {
            "test": args.test, "model": args.model,
            **metrics, "checkpoint_sha256": config["checkpoint_sha256"],
        })
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
