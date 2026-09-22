"""Per-class temperature scaling (Guo et al. 2017) for the multi-label model.

Threshold calibration (calibrate_thresholds.py) fixes the *operating point*
given the model's probabilities. This fixes the probabilities themselves:
pos_weight in the training loss inflates predicted probabilities broadly,
so "70% confidence" doesn't actually mean "true positive 70% of the time."
Temperature scaling divides each class's logit by a learned scalar T > 1
before the sigmoid, which softens overconfident probabilities without
changing the model's ranking of examples (AUROC is unchanged - only the
probability *values* move, which is exactly what we want to verify below).

After rescaling, the threshold search from calibrate_thresholds.py is
re-run on the new, better-calibrated probabilities, since the old
thresholds were fit to the old (miscalibrated) probability scale.
"""

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from calibrate_thresholds import find_best_threshold
from train_multilabel import (
    CONDITIONS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_DIR, MODEL_PATH,
    NIH14Dataset, build_rows, load_manifest, patient_level_split,
)

REPORT_PATH = MODEL_DIR / "nih14_temperature_report.json"
TEMPERATURE_GRID = np.geomspace(0.1, 10.0, 200)  # log-spaced search, T=1 means "no change"


def get_logits_and_targets(model, loader, device):
    all_targets, all_logits = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images).cpu().numpy()
            all_logits.append(logits)
            all_targets.append(targets.numpy())
    return np.concatenate(all_targets), np.concatenate(all_logits)


def binary_nll(targets: np.ndarray, probs: np.ndarray) -> float:
    eps = 1e-7
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(targets * np.log(probs) + (1 - targets) * np.log(1 - probs)))


def find_best_temperature(targets_col: np.ndarray, logits_col: np.ndarray) -> float:
    best_t, best_nll = 1.0, binary_nll(targets_col, 1 / (1 + np.exp(-logits_col)))
    for t in TEMPERATURE_GRID:
        probs = 1 / (1 + np.exp(-logits_col / t))
        nll = binary_nll(targets_col, probs)
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def expected_calibration_error(targets: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted probability and actual positive
    rate, within probability bins - the standard calibration quality metric."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = probs[mask].mean()
        bin_accuracy = targets[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_confidence - bin_accuracy)
    return float(ece)


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

    print("Running inference on validation set (logits)...", flush=True)
    val_targets, val_logits = get_logits_and_targets(model, val_loader, device)

    temperatures = {}
    for i, cond in enumerate(CONDITIONS):
        temperatures[cond] = find_best_temperature(val_targets[:, i], val_logits[:, i])

    print("\nLearned per-class temperatures (>1 = was overconfident):")
    for cond, t in temperatures.items():
        print(f"  {cond}: {t:.3f}")

    print("\nRunning inference on test set for verification...", flush=True)
    test_targets, test_logits = get_logits_and_targets(model, test_loader, device)

    raw_probs = 1 / (1 + np.exp(-test_logits))
    temp_array = np.array([temperatures[c] for c in CONDITIONS])
    scaled_probs = 1 / (1 + np.exp(-test_logits / temp_array[None, :]))

    print("\nSanity check - AUROC must be unchanged by temperature scaling (it's monotonic):")
    for i, cond in enumerate(CONDITIONS):
        raw_auroc = roc_auc_score(test_targets[:, i], raw_probs[:, i])
        scaled_auroc = roc_auc_score(test_targets[:, i], scaled_probs[:, i])
        print(f"  {cond}: raw={raw_auroc:.4f} scaled={scaled_auroc:.4f}")

    print("\nExpected Calibration Error (lower is better), raw vs temperature-scaled:")
    ece_report = {}
    for i, cond in enumerate(CONDITIONS):
        raw_ece = expected_calibration_error(test_targets[:, i], raw_probs[:, i])
        scaled_ece = expected_calibration_error(test_targets[:, i], scaled_probs[:, i])
        ece_report[cond] = {"raw": raw_ece, "scaled": scaled_ece}
        print(f"  {cond}: {raw_ece:.4f} -> {scaled_ece:.4f}")

    # Re-run the same conservative threshold search from calibrate_thresholds.py,
    # but now on temperature-scaled probabilities instead of raw ones.
    print("\nRe-calibrating thresholds on temperature-scaled probabilities...", flush=True)
    val_scaled_probs = 1 / (1 + np.exp(-val_logits / temp_array[None, :]))
    thresholds = {}
    for i, cond in enumerate(CONDITIONS):
        t, _ = find_best_threshold(val_targets[:, i], val_scaled_probs[:, i])
        thresholds[cond] = t

    healthy_mask = test_targets.sum(axis=1) == 0
    print(f"\n{int(healthy_mask.sum())} genuinely healthy (No Finding) images in test set")

    def fpr_on_healthy(preds):
        return float((preds[healthy_mask].sum(axis=1) > 0).mean())

    flat_05_preds = (raw_probs >= 0.5).astype(int)
    prior_calibrated_preds = (raw_probs >= np.array([checkpoint["decision_thresholds"][c] for c in CONDITIONS])[None, :]).astype(int)
    temp_scaled_preds = (scaled_probs >= np.array([thresholds[c] for c in CONDITIONS])[None, :]).astype(int)

    print("\nFraction of healthy test images with >=1 false-positive finding:")
    print(f"  Flat 0.5 (no calibration at all):        {fpr_on_healthy(flat_05_preds):.1%}")
    print(f"  Threshold-only calibration (prior fix):  {fpr_on_healthy(prior_calibrated_preds):.1%}")
    print(f"  Temperature scaling + threshold search:  {fpr_on_healthy(temp_scaled_preds):.1%}")

    report = {
        "temperatures": temperatures,
        "expected_calibration_error": ece_report,
        "thresholds_after_temperature_scaling": thresholds,
        "healthy_false_positive_rate": {
            "flat_0.5": fpr_on_healthy(flat_05_preds),
            "threshold_only": fpr_on_healthy(prior_calibrated_preds),
            "temperature_plus_threshold": fpr_on_healthy(temp_scaled_preds),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {REPORT_PATH}")

    checkpoint["temperatures"] = temperatures
    checkpoint["decision_thresholds"] = thresholds
    torch.save(checkpoint, MODEL_PATH)
    print(f"Updated {MODEL_PATH} with temperatures + re-calibrated thresholds")


if __name__ == "__main__":
    main()
