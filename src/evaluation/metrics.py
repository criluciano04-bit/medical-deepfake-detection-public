"""Binary classification metrics for medical deepfake evaluation.

Labels: 0=real, 1=fake. `fake_probabilities` contains P(fake).
This module is independent of the Dataset, model, and GPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(
    labels: Sequence[int] | np.ndarray,
    fake_probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    """Compute binary metrics, preserving the definitions used in training.

    Uses P(fake) > 0.5 to match torch.argmax(logits, dim=1) for a
    two-class softmax, including ties (ties predict class 0).
    The optional threshold can be used for *prespecified* analyses, but must
    never be tuned on a test set.
    ROC-AUC is None if only one true class is present.
    """
    y = np.asarray(labels)
    p = np.asarray(fake_probabilities, dtype=np.float64)

    if y.ndim != 1 or p.ndim != 1 or y.size != p.size or y.size == 0:
        raise ValueError("labels and fake_probabilities must be nonempty 1D arrays of equal length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must contain only 0 (real) and 1 (fake)")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("fake_probabilities must be finite and between 0 and 1")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    y = y.astype(np.int64)
    pred = (p > threshold).astype(np.int64)

    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None

    return {
        "n_samples": int(y.size),
        "roc_auc": auc,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / y.size,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
