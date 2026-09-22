"""Per-class decision threshold calibration for the multi-label model.

The training loss uses per-class pos_weight to correct for severe class
imbalance (Hernia's weight was ~478x), which has a known side effect: it
inflates predicted probabilities across the board, so a single flat 0.5
threshold doesn't mean the same thing for every class anymore. This shows
up exactly as reported: genuinely healthy images getting several
conditions "moderately" flagged (50-60%) even though none are present.

This finds, per class, the probability threshold that maximizes F1 on the
validation set (never the test set - that stays untouched for an honest
final check), then re-evaluates on the test set with those thresholds to
show the real effect - especially the false-positive rate on genuinely
healthy (No Finding) test images, which is the actual symptom reported.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from train_multilabel import (
    CONDITIONS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_PATH,
    NIH14Dataset, build_rows, load_manifest, patient_level_split,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CALIBRATION_REPORT_PATH = MODEL_DIR / "nih14_calibration_report.json"

MIN_POSITIVE_VAL_EXAMPLES = 15  # below this, a calibrated threshold is unstable - keep the 0.5 default
THRESHOLD_MIN, THRESHOLD_MAX = 0.5, 0.85  # never go below 0.5 - only raise the threshold, never lower it
RECALL_RETENTION = 0.90  # only raise the threshold where recall stays >= 90% of its value at 0.5


def get_probs_and_targets(model, loader, device):
    all_targets, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
    return np.concatenate(all_targets), np.concatenate(all_probs)


def find_best_threshold(targets_col: np.ndarray, probs_col: np.ndarray) -> tuple[float, bool]:
    """Returns (threshold, was_calibrated). Falls back to 0.5 (uncalibrated
    baseline) when there aren't enough positive examples to trust the
    estimate, or when no threshold above 0.5 can reduce false positives
    without giving up too much recall.

    Deliberately conservative after two rejected approaches: pure
    F1-maximization let some classes' recall collapse below 10% (e.g.
    Pneumonia 50%->6%) chasing precision; a flat 75% recall floor forced
    thresholds for hard-to-separate classes *below* 0.5, making the
    healthy-image false-positive rate worse, not better (62%->80%). Both
    confirmed the model's raw probabilities aren't well-separated enough
    for many of these 14 classes to freely trade recall for precision - so
    this only ever raises the threshold from the 0.5 baseline, and only
    where doing so keeps recall within RECALL_RETENTION of where it
    started. Where no such point exists, the class stays at 0.5: a
    genuine limit of post-hoc thresholding, not something to force past."""
    n_positive = int(targets_col.sum())
    if n_positive < MIN_POSITIVE_VAL_EXAMPLES:
        return 0.5, False

    precision, recall, thresholds = precision_recall_curve(targets_col, probs_col)
    precision, recall = precision[:-1], recall[:-1]  # drop the (1,0) endpoint with no threshold
    if len(thresholds) == 0:
        return 0.5, False

    baseline_idx = int(np.argmin(np.abs(thresholds - 0.5)))
    baseline_recall = recall[baseline_idx]
    if baseline_recall <= 0:
        return 0.5, False

    at_or_above_half = thresholds >= 0.5
    retains_recall = recall >= RECALL_RETENTION * baseline_recall
    candidates = at_or_above_half & retains_recall

    if not candidates.any():
        return 0.5, False  # no safe way to raise the threshold for this class

    candidate_precisions = np.where(candidates, precision, -1)
    best_idx = int(np.argmax(candidate_precisions))

    best_threshold = float(thresholds[best_idx])
    return float(np.clip(best_threshold, THRESHOLD_MIN, THRESHOLD_MAX)), True


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    df = load_manifest()
    _, val_df, test_df = patient_level_split(df)

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_loader = DataLoader(NIH14Dataset(build_rows(val_df), eval_transform), batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(NIH14Dataset(build_rows(test_df), eval_transform), batch_size=32, shuffle=False, num_workers=2)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CONDITIONS))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    print("Running inference on validation set...", flush=True)
    val_targets, val_probs = get_probs_and_targets(model, val_loader, device)

    thresholds, was_calibrated = {}, {}
    for i, cond in enumerate(CONDITIONS):
        t, calibrated = find_best_threshold(val_targets[:, i], val_probs[:, i])
        thresholds[cond] = t
        was_calibrated[cond] = calibrated

    print("\nCalibrated per-class thresholds:")
    for cond, t in thresholds.items():
        flag = "" if was_calibrated[cond] else "  (too few positive val examples - kept default)"
        print(f"  {cond}: {t:.3f}{flag}")

    print("\nRunning inference on test set for verification...", flush=True)
    test_targets, test_probs = get_probs_and_targets(model, test_loader, device)

    healthy_mask = test_targets.sum(axis=1) == 0
    print(f"\n{int(healthy_mask.sum())} genuinely healthy (No Finding) images in test set")

    def false_positive_rate_on_healthy(preds):
        if healthy_mask.sum() == 0:
            return None
        return float((preds[healthy_mask].sum(axis=1) > 0).mean())

    flat_preds = (test_probs >= 0.5).astype(int)
    calibrated_thresh_array = np.array([thresholds[c] for c in CONDITIONS])
    calibrated_preds = (test_probs >= calibrated_thresh_array[None, :]).astype(int)

    flat_fpr = false_positive_rate_on_healthy(flat_preds)
    calibrated_fpr = false_positive_rate_on_healthy(calibrated_preds)

    print(f"\nFraction of healthy test images with >=1 false-positive finding:")
    print(f"  Flat 0.5 threshold:    {flat_fpr:.1%}")
    print(f"  Calibrated thresholds: {calibrated_fpr:.1%}")

    precision, recall, f1, support = precision_recall_fscore_support(
        test_targets, calibrated_preds, average=None, zero_division=0
    )
    flat_precision, flat_recall, flat_f1, _ = precision_recall_fscore_support(
        test_targets, flat_preds, average=None, zero_division=0
    )

    print("\nPer-class test performance, flat 0.5 vs calibrated:")
    for i, cond in enumerate(CONDITIONS):
        print(f"  {cond}: threshold={thresholds[cond]:.3f}  "
              f"recall {flat_recall[i]:.3f}->{recall[i]:.3f}  "
              f"precision {flat_precision[i]:.3f}->{precision[i]:.3f}  "
              f"support={support[i]}")

    report = {
        "thresholds": thresholds,
        "was_calibrated": was_calibrated,
        "healthy_false_positive_rate": {"flat_0.5": flat_fpr, "calibrated": calibrated_fpr},
        "per_class_test_performance": {
            CONDITIONS[i]: {
                "threshold": thresholds[CONDITIONS[i]],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(CONDITIONS))
        },
    }
    CALIBRATION_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved calibration report to {CALIBRATION_REPORT_PATH}")

    checkpoint["decision_thresholds"] = thresholds
    torch.save(checkpoint, MODEL_PATH)
    print(f"Updated {MODEL_PATH} with calibrated per-class thresholds")


if __name__ == "__main__":
    main()
