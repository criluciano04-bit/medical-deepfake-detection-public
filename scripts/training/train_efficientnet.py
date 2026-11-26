"""Full EfficientNet-B0 training on LIDC-IDRI / M3Dsynth.

Run from the repository root. DATA_ROOT must point to the selected dataset.
Outputs default to a sibling training_checkpoints directory next to DATA_ROOT.
Does not run any test-set evaluations. Uses only train and valid splits.

Example:
    python -m scripts.training.train_efficientnet \
      --epochs 20 --batch-size 16 --lr 3e-5

Resume by repeating the same command with --resume.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

from src.data.dataloader import create_dataloader
from src.data.datasets.lidc_m3dsynth import LIDC_M3DSynth_Dataset
from src.data.model_transforms import prepare_imagenet_input
from src.models.model_factory import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full EfficientNet-B0 training")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Optional run output directory; by default uses DATA_ROOT parent / "
                             "training_checkpoints / efficientnet_b0 / lr_<value>")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum total epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5,
                        help="Stop if validation loss does not improve for this many epochs; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0 or args.patience < 0:
        parser.error("epochs, batch-size and lr must be positive; num-workers and patience must be nonnegative")
    return args


def compute_metrics(tp: int, tn: int, fp: int, fn: int,
                    labels: list[int], fake_probabilities: list[float]) -> dict:
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    roc_auc = (float(roc_auc_score(labels, fake_probabilities))
               if len(set(labels)) == 2 else None)
    return {
        "roc_auc": roc_auc,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_epoch(model: nn.Module, loader: torch.utils.data.DataLoader,
              criterion: nn.Module, optimizer: torch.optim.Optimizer,
              device: torch.device, training: bool, epoch_number: int) -> dict:
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0
    epoch_labels: list[int] = []
    epoch_fake_probabilities: list[float] = []
    started = time.perf_counter()

    phase = "Train" if training else "Valid"
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch_number} | {phase}",
        unit="batch",
        disable=True
    )

    print(
        f"Epoch {epoch_number} | {phase} | "
        f"Inizio elaborazione di {len(loader)} batch",
        flush=True
    )

    with torch.set_grad_enabled(training):
        for batch_idx, (images, labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            images = prepare_imagenet_input(images)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            fake_probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
            epoch_labels.extend(labels.detach().cpu().tolist())
            epoch_fake_probabilities.extend(fake_probabilities.cpu().tolist())
            tp += int(((predictions == 1) & (labels == 1)).sum().item())
            tn += int(((predictions == 0) & (labels == 0)).sum().item())
            fp += int(((predictions == 1) & (labels == 0)).sum().item())
            fn += int(((predictions == 0) & (labels == 1)).sum().item())
            n = labels.size(0)
            total_loss += loss.item() * n
            total_samples += n
            if batch_idx % 50 == 0 or batch_idx == len(loader):
                elapsed = time.perf_counter() - started

                print(
                    f"Epoch {epoch_number} | {phase} | "
                    f"Batch {batch_idx}/{len(loader)} | "
                    f"Loss: {total_loss / total_samples:.4f} | "
                    f"Tempo: {elapsed:.1f}s",
                    flush=True
                    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if total_samples == 0:
        raise RuntimeError("Empty DataLoader")
    return {
        "loss": total_loss / total_samples,
        "seconds_total": elapsed,
        "images": total_samples,
        **compute_metrics(tp, tn, fp, fn, epoch_labels, epoch_fake_probabilities),
    }


def atomic_save(checkpoint: dict, destination: Path) -> None:
    temporary = destination.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    if args.checkpoint_dir is None:
        data_root = os.environ.get("DATA_ROOT")
        if not data_root:
            raise RuntimeError("DATA_ROOT is required to derive the checkpoint directory")
        lr_label = format(args.lr, ".0e").replace("e-0", "e-").replace("e+0", "e+")
        args.checkpoint_dir = (Path(data_root).expanduser().resolve().parent /
                               "training_checkpoints" / "efficientnet_b0" / f"lr_{lr_label}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Full training requires a CUDA GPU")
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(device), flush=True)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = args.checkpoint_dir / "efficientnet_b0_latest.pt"
    best = args.checkpoint_dir / "efficientnet_b0_best.pt"
    history_file = args.checkpoint_dir / "efficientnet_b0_history.json"
    config_file = args.checkpoint_dir / "efficientnet_b0_config.json"
    if not args.resume and any(p.exists() for p in (latest, best, history_file, config_file)):
        raise FileExistsError("Existing training outputs found; use --resume or a new checkpoint directory")

    train_dataset = LIDC_M3DSynth_Dataset(split="train")
    valid_dataset = LIDC_M3DSynth_Dataset(split="valid")
    for ds, expected in ((train_dataset, {0: 7500, 1: 7500}),
                         (valid_dataset, {0: 1500, 1: 1500})):
        counts = ds.samples["label"].value_counts().to_dict()
        counts = {int(k): int(v) for k, v in counts.items()}
        if counts != expected:
            raise ValueError(f"Unexpected {ds.split} class counts: {counts}; expected {expected}")
        print(f"{ds.split}: {len(ds):,} images; label counts: {counts}", flush=True)

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = create_dataloader(valid_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = create_model("efficientnet_b0", num_classes=2, pretrained=not args.resume).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {
        "model": "efficientnet_b0", "batch_size": args.batch_size,
        "lr": args.lr, "seed": args.seed, "train_images": len(train_dataset),
        "valid_images": len(valid_dataset), "patience": args.patience,
        "input": "CT HU [-1000,1000] -> [0,1] -> repeat x3 -> ImageNet normalized",
        "augmentation": False,
        "optimizer": "Adam", "loss": "CrossEntropyLoss",
        "fine_tuning": "full", "lr_scheduler": None,
        "max_epochs": args.epochs, "num_workers": args.num_workers,
    }

    if not args.resume:
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    start_epoch = 0
    best_valid_loss = float("inf")
    bad_epochs = 0
    history: list[dict] = []
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(f"No latest checkpoint at {latest}")
        checkpoint = torch.load(latest, map_location=device, weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("Configuration incompatible with checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["completed_epoch"])
        best_valid_loss = float(checkpoint["best_valid_loss"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history = checkpoint["history"]
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np_state = checkpoint["numpy_rng_state"]
            np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32),
                                 np_state[2], np_state[3], np_state[4]))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        print(f"Resuming from completed epoch {start_epoch}; next is {start_epoch + 1}", flush=True)

    if args.patience and bad_epochs >= args.patience:
        print("Early stopping had already been reached in checkpoint", flush=True)
        return

    for epoch in range(start_epoch, args.epochs):
        torch.cuda.reset_peak_memory_stats(device)
        train = run_epoch(model, train_loader, criterion, optimizer, device, training=True, epoch_number=epoch + 1)
        valid = run_epoch(model, valid_loader, criterion, optimizer, device, training=False, epoch_number=epoch + 1)
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        entry = {
            "epoch": epoch + 1,
            "train": train,
            "valid": valid,
            "peak_gpu_memory_allocated_gib": peak_gib,
        }
        history.append(entry)
        improved = valid["loss"] < best_valid_loss
        if improved:
            best_valid_loss = valid["loss"]
            bad_epochs = 0
        else:
            bad_epochs += 1

        checkpoint = {
            "completed_epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_valid_loss": best_valid_loss,
            "bad_epochs": bad_epochs,
            "config": config,
            "history": history,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": (lambda state: (state[0], state[1].tolist(),
                                                 state[2], state[3], state[4]))(np.random.get_state()),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }
        atomic_save(checkpoint, latest)
        if improved:
            atomic_save(checkpoint, best)
        history_temporary = history_file.with_suffix(".json.tmp")
        history_temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
        os.replace(history_temporary, history_file)
        print(json.dumps(entry, indent=2), flush=True)
        print(f"Checkpoint saved: {latest}", flush=True)
        if args.patience and bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without validation loss improvement", flush=True)
            break

    print("Training finished. Do not tune hyperparameters using test sets.", flush=True)


if __name__ == "__main__":
    main()
