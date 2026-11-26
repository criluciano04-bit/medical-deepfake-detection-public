"""Balanced EfficientNet-B0 GPU pilot with timing, metrics, checkpoints and resume.

Run from repository root after DATA_ROOT is configured and data is accessible.
Example (Colab; Drive must be mounted beforehand):
    python -m scripts.training.train_efficientnet_pilot \
      --checkpoint-dir /content/drive/MyDrive/medical-deepfake-data/pilot_checkpoints

The script is NOT automatically executed by importing it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from src.data.dataloader import create_dataloader
from src.data.datasets.lidc_m3dsynth import LIDC_M3DSynth_Dataset
from src.models.model_factory import create_model
from src.data.model_transforms import prepare_imagenet_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balanced EfficientNet-B0 GPU pilot")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Persistent directory, preferably mounted Google Drive")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-images-per-class", type=int, default=100)
    parser.add_argument("--valid-images-per-class", type=int, default=40)
    parser.add_argument("--timing-warmup-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "train_images_per_class", "valid_images_per_class"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be >= 1")
    if args.num_workers < 0 or args.lr <= 0 or args.timing_warmup_batches < 0:
        parser.error("num-workers and timing-warmup-batches must be >= 0; lr must be > 0")
    if args.batch_size > 2 * min(args.train_images_per_class, args.valid_images_per_class):
        parser.error("batch-size must not exceed the smaller pilot split size")
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def balanced_subset(dataset: LIDC_M3DSynth_Dataset, per_class: int, seed: int) -> Subset:
    """Select exactly per_class Real (0) and per_class Fake (1) samples.

    Select from manifest metadata only: no images are loaded here.
    """
    labels = dataset.samples["label"].to_numpy()
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(f"Split {dataset.split}: {len(candidates)} samples of label {label}, need {per_class}")
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return Subset(dataset, selected)


def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision_fake": precision, "recall_fake": recall, "f1_fake": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool,
    warmup_batches: int,
) -> dict:
    """Time entire pipeline and separate data waiting from GPU processing.

    Time results are synchronized on CUDA for measurement accuracy. The timing
    sample excludes the first warmup_batches steps (already included in training).
    """
    model.train(training)
    total_loss, total_samples = 0.0, 0
    tp = tn = fp = fn = 0
    measured_images = 0
    measured_data_seconds = 0.0
    measured_compute_seconds = 0.0
    batches = 0
    synchronize(device)
    epoch_start = time.perf_counter()
    iterator = iter(loader)
    with torch.set_grad_enabled(training):
        while True:
            # This captures dataloader wait/decode/preprocessing time. If workers
            # prefetch in parallel, compute and data timings overlap in reality.
            synchronize(device)
            data_start = time.perf_counter()
            try:
                images, labels = next(iterator)
            except StopIteration:
                break
            data_seconds = time.perf_counter() - data_start

            synchronize(device)
            compute_start = time.perf_counter()
            images = prepare_imagenet_input(images.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            tp += int(((predictions == 1) & (labels == 1)).sum().item())
            tn += int(((predictions == 0) & (labels == 0)).sum().item())
            fp += int(((predictions == 1) & (labels == 0)).sum().item())
            fn += int(((predictions == 0) & (labels == 1)).sum().item())
            n = labels.size(0)
            total_loss += loss.item() * n
            total_samples += n
            synchronize(device)
            compute_seconds = time.perf_counter() - compute_start
            if batches >= warmup_batches:
                measured_images += n
                measured_data_seconds += data_seconds
                measured_compute_seconds += compute_seconds
            batches += 1
    synchronize(device)
    elapsed = time.perf_counter() - epoch_start
    if not batches:
        raise RuntimeError("Empty dataloader")
    measured_seconds = measured_data_seconds + measured_compute_seconds
    return {
        "loss": total_loss / total_samples,
        "seconds_total": elapsed,
        "seconds_per_batch_total": elapsed / batches,
        "images_per_second_total": total_samples / elapsed,
        "images": total_samples,
        "batches": batches,
        "warmup_batches_excluded_from_timing": min(warmup_batches, batches),
        "timing_images": measured_images,
        "data_seconds_measured": measured_data_seconds,
        "compute_seconds_measured": measured_compute_seconds,
        "seconds_per_image_after_warmup": measured_seconds / measured_images if measured_images else None,
        "compute_seconds_per_image_after_warmup": measured_compute_seconds / measured_images if measured_images else None,
        **compute_metrics(tp, tn, fp, fn),
    }


def atomic_save(checkpoint: dict, destination: Path) -> None:
    tmp = destination.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, destination)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Il pilota richiede CUDA: nessun training sulla CPU.")
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(device))

    # The directory must point to mounted Drive to survive Colab restarts.
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = args.checkpoint_dir / "efficientnet_b0_pilot_latest.pt"
    best = args.checkpoint_dir / "efficientnet_b0_pilot_best.pt"
    history_file = args.checkpoint_dir / "efficientnet_b0_pilot_history.json"

    train_dataset = LIDC_M3DSynth_Dataset(split="train")
    valid_dataset = LIDC_M3DSynth_Dataset(split="valid")
    train_subset = balanced_subset(train_dataset, args.train_images_per_class, args.seed)
    valid_subset = balanced_subset(valid_dataset, args.valid_images_per_class, args.seed + 1)
    train_loader = create_dataloader(train_subset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = create_dataloader(valid_subset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Training: {len(train_subset)} CT (Real={args.train_images_per_class}, Fake={args.train_images_per_class})")
    print(f"Validation: {len(valid_subset)} CT (Real={args.valid_images_per_class}, Fake={args.valid_images_per_class})")

    # Even on resume this creates the same architecture; saved weights overwrite it.
    model = create_model("efficientnet_b0", num_classes=2,
                         pretrained=(not args.no_pretrained and not args.resume)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {
        "batch_size": args.batch_size, "lr": args.lr,
        "train_images_per_class": args.train_images_per_class,
        "valid_images_per_class": args.valid_images_per_class,
        "seed": args.seed,
        "pretrained": not args.no_pretrained,
    }
    start_epoch, best_valid_loss = 0, float("inf")
    history: list[dict] = []
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(f"Checkpoint non trovato: {latest}")
        checkpoint = torch.load(latest, map_location=device, weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("Configurazione incompatibile con il checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["completed_epoch"])
        best_valid_loss = float(checkpoint["best_valid_loss"])
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([
            state.cpu() for state in checkpoint["cuda_rng_state"]
            ])
        print(f"Resume dall'epoca {start_epoch + 1}")

    for epoch in range(start_epoch, args.epochs):
        torch.cuda.reset_peak_memory_stats(device)
        train = run_epoch(model, train_loader, criterion, optimizer, device,
                          training=True, warmup_batches=args.timing_warmup_batches)
        valid = run_epoch(model, valid_loader, criterion, optimizer, device,
                          training=False, warmup_batches=args.timing_warmup_batches)
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        # Extrapolation includes measured data + model time; estimates, not guarantees.
        train_estimate = (train["seconds_per_image_after_warmup"] * len(train_dataset)
                          if train["seconds_per_image_after_warmup"] is not None else None)
        valid_estimate = (valid["seconds_per_image_after_warmup"] * len(valid_dataset)
                          if valid["seconds_per_image_after_warmup"] is not None else None)
        entry = {
            "epoch": epoch + 1,
            "train": train, "valid": valid,
            "peak_gpu_memory_allocated_gib": peak_gib,
            "estimated_full_train_seconds": train_estimate,
            "estimated_full_valid_seconds": valid_estimate,
            "estimated_full_epoch_seconds": (train_estimate + valid_estimate
                                             if train_estimate is not None and valid_estimate is not None else None),
        }
        history.append(entry)
        print(json.dumps(entry, indent=2))
        if valid["loss"] < best_valid_loss:
            best_valid_loss = valid["loss"]
            improved = True
        else:
            improved = False
        checkpoint = {
            "completed_epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_valid_loss": best_valid_loss,
            "config": config,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }
        atomic_save(checkpoint, latest)
        if improved:
            atomic_save(checkpoint, best)
        history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print("Checkpoint salvato:", latest)

    print("Pilota concluso. Stime indicative: ripetere la misura sul dataset completo per conferma.")


if __name__ == "__main__":
    main()
