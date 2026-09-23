"""End-to-end evaluation of the two-stage system: normal/abnormal gate, then
the 14-condition ensemble only if the gate says "abnormal".

The gate only ever vetoes findings, so it can reduce false alarms but can
also cost sensitivity: a real finding is only shown if the gate passes the
scan AND the ensemble flags it. This measures both sides at several gate
sensitivity targets (the gate threshold is chosen on NIH validation):

- Healthy false-alarm rate: NIH test "No Finding" scans, and Kermany
  pediatric NORMAL (external - neither model trained on it)
- Per-class recall on the NIH test split
- External sensitivity: share of Kermany pediatric PNEUMONIA scans that get
  at least one finding (external, real sick scans)
"""

import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from evaluate_ensemble import (
    KERMANY_DIR, DualInputDataset, load_models, nih_loader, per_class_recall,
)
from train_gate import GATE_MODEL_PATH
from train_multilabel import CONDITIONS, MODEL_DIR, load_manifest, patient_level_split

REPORT_PATH = MODEL_DIR / "gated_system_report.json"
ENSEMBLE_CONFIG_PATH = MODEL_DIR / "ensemble_config.json"
GATE_RECALL_TARGETS = [0.90, 0.95, 0.975, 0.99]


def load_gate(device):
    gate = models.resnet50(weights=None)
    gate.fc = nn.Linear(gate.fc.in_features, 1)
    gate.load_state_dict(torch.load(GATE_MODEL_PATH, map_location=device)["model_state"])
    return gate.to(device).eval()


def predict_all(ours, txv, txv_indices, gate, loader, device):
    """Returns (ensemble probs [N,14], gate abnormal probs [N], targets [N,14])."""
    ens, gate_probs, targets = [], [], []
    with torch.no_grad():
        for ours_in, txv_in, t in loader:
            ours_in = ours_in.to(device)
            p_ours = torch.sigmoid(ours(ours_in)).cpu().numpy()
            p_txv = txv(txv_in.to(device))[:, txv_indices].cpu().numpy()
            ens.append((p_ours + p_txv) / 2)
            gate_probs.append(torch.sigmoid(gate(ours_in)).cpu().numpy().ravel())
            targets.append(t.numpy())
    return np.concatenate(ens), np.concatenate(gate_probs), np.concatenate(targets)


def kermany_loader(label: str):
    paths = []
    for split in ["train", "validation", "test"]:
        labels = pd.read_csv(KERMANY_DIR / split / "labels.csv")
        paths += [str(KERMANY_DIR / split / f) for f in labels[labels["label"] == label]["filename"]]
    empty = torch.zeros(len(CONDITIONS))
    return DataLoader(DualInputDataset(paths, [empty] * len(paths)), batch_size=32, shuffle=False, num_workers=2)


def gate_threshold_at_recall(abnormal, gate_probs, target_recall: float) -> float:
    positives = np.sort(gate_probs[abnormal])[::-1]
    k = min(max(1, int(np.ceil(target_recall * len(positives)))), len(positives))
    return float(positives[k - 1])


def flags(ens_probs, gate_probs, thresholds: dict, gate_threshold: float | None):
    """Boolean [N,14]: findings shown to the user. gate_threshold=None means
    no gate (the currently deployed behavior)."""
    t = np.array([thresholds[c] for c in CONDITIONS])[None, :]
    shown = ens_probs >= t
    if gate_threshold is not None:
        shown &= (gate_probs >= gate_threshold)[:, None]
    return shown


def any_flag_rate(shown) -> float:
    return float(shown.any(axis=1).mean())


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ours, txv, txv_indices, _ = load_models(device)
    gate = load_gate(device)
    thresholds = json.loads(ENSEMBLE_CONFIG_PATH.read_text())["thresholds"]

    _, val_df, test_df = patient_level_split(load_manifest())

    print("Predicting on NIH validation...", flush=True)
    _, val_gate, val_targets = predict_all(ours, txv, txv_indices, gate, nih_loader(val_df), device)
    val_abnormal = val_targets.sum(axis=1) > 0

    print("Predicting on NIH test...", flush=True)
    test_ens, test_gate, test_targets = predict_all(ours, txv, txv_indices, gate, nih_loader(test_df), device)
    test_healthy = test_targets.sum(axis=1) == 0

    print("Predicting on Kermany NORMAL and PNEUMONIA (external)...", flush=True)
    kn_ens, kn_gate, _ = predict_all(ours, txv, txv_indices, gate, kermany_loader("NORMAL"), device)
    kp_ens, kp_gate, _ = predict_all(ours, txv, txv_indices, gate, kermany_loader("PNEUMONIA"), device)

    variants = {"no gate (deployed now)": None}
    for r in GATE_RECALL_TARGETS:
        variants[f"gate @ {r:.1%} val sensitivity"] = gate_threshold_at_recall(val_abnormal, val_gate, r)

    report = {}
    for name, gt in variants.items():
        test_shown = flags(test_ens, test_gate, thresholds, gt)
        report[name] = {
            "gate_threshold": gt,
            "nih_healthy_false_alarm": any_flag_rate(test_shown[test_healthy]),
            "kermany_normal_false_alarm": any_flag_rate(flags(kn_ens, kn_gate, thresholds, gt)),
            "kermany_pneumonia_flagged": any_flag_rate(flags(kp_ens, kp_gate, thresholds, gt)),
            "nih_per_class_recall": per_class_recall(
                test_targets, test_shown.astype(float), {c: 0.5 for c in CONDITIONS}
            ),
        }

    print("\n=== Healthy scans flagged with >=1 finding (lower is better) / external sick scans caught (higher is better) ===")
    print(f"{'variant':34s} {'NIH healthy':>12s} {'ext. healthy':>13s} {'ext. pneumonia':>15s}")
    for name, r in report.items():
        print(f"{name:34s} {r['nih_healthy_false_alarm']:>12.1%} {r['kermany_normal_false_alarm']:>13.1%} "
              f"{r['kermany_pneumonia_flagged']:>15.1%}")

    print("\n=== NIH test per-class recall ===")
    names = list(report)
    print(f"{'condition':20s}" + "".join(f"{n[:14]:>16s}" for n in names))
    for c in CONDITIONS:
        print(f"{c:20s}" + "".join(f"{report[n]['nih_per_class_recall'][c]:>16.3f}" for n in names))

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
