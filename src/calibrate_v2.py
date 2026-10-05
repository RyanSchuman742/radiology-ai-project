"""Choose v2's decision thresholds from validation data only.

Rules are the ones the deployed system used, and were fixed before v2 was
scored on the scoreboard's test split (so they can't be tuned toward it):
- each condition: the threshold that gives v2 the same recall on NIH
  validation as the deployed system has there - same sensitivity, so the
  comparison comes down to false alarms (matched recall, as in
  evaluate_ensemble.py)
- abnormal output: 90% sensitivity on NIH validation abnormal scans, the
  rule used for the deployed gate

Writes models/v2_config.json, which scoreboard.py's "v2" system reads.
"""

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_ensemble import load_models, matched_recall_threshold, nih_loader, per_class_recall
from evaluate_gated_system import gate_threshold_at_recall, load_gate, predict_all
from train_multilabel import CONDITIONS, MODEL_DIR, load_manifest, patient_level_split
from train_v2 import ABNORMAL, MODEL_PATH, OUTPUTS, XrayDataset, XrayNet

CONFIG_PATH = MODEL_DIR / "v2_config.json"
DEPLOYED_CONFIG_PATH = MODEL_DIR / "ensemble_config.json"
GATE_SENSITIVITY = 0.90


def load_v2(device) -> XrayNet:
    model = XrayNet(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)["model_state"])
    return model.to(device).eval()


def predict_v2(model, paths: list[str], device) -> tuple[np.ndarray, np.ndarray]:
    """(condition probs [N,14], abnormal prob [N])."""
    rows = [{"path": p, "target": [0.0] * len(OUTPUTS), "mask": [1.0] * len(OUTPUTS)} for p in paths]
    loader = DataLoader(XrayDataset(rows, train=False), batch_size=32, shuffle=False, num_workers=8)
    out = []
    with torch.no_grad():
        for x, _, _ in loader:
            out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    out = np.concatenate(out)
    return out[:, :ABNORMAL], out[:, ABNORMAL]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, val_df, _ = patient_level_split(load_manifest())
    deployed_config = json.loads(DEPLOYED_CONFIG_PATH.read_text())

    print(f"Deployed system on NIH validation ({len(val_df)} images)...", flush=True)
    ours, txv, txv_indices, _ = load_models(device)
    ens, gate_probs, targets = predict_all(ours, txv, txv_indices, load_gate(device), nih_loader(val_df), device)
    deployed_recall = per_class_recall(targets, ens, deployed_config["thresholds"])
    abnormal = targets.sum(axis=1) > 0
    deployed_gate_recall = float((gate_probs[abnormal] >= deployed_config["gate_threshold"]).mean())
    del ours, txv

    print("v2 on NIH validation...", flush=True)
    probs, abnormal_prob = predict_v2(load_v2(device), val_df["path"].tolist(), device)

    thresholds = {c: matched_recall_threshold(targets[:, i], probs[:, i], deployed_recall[c])
                  for i, c in enumerate(CONDITIONS)}
    gate_threshold = gate_threshold_at_recall(abnormal, abnormal_prob, GATE_SENSITIVITY)

    config = {"model": MODEL_PATH.name, "thresholds": thresholds, "gate_threshold": gate_threshold,
              "calibration": "matched recall to deployed on NIH val; abnormal at 90% NIH val sensitivity",
              "deployed_val_recall": deployed_recall, "deployed_val_gate_recall": deployed_gate_recall}
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    v2_recall = per_class_recall(targets, probs, thresholds)
    print("\nCondition: deployed recall -> v2 recall at its threshold (NIH val)")
    for c in CONDITIONS:
        print(f"  {c:20s} {deployed_recall[c]:.3f} -> {v2_recall[c]:.3f}  (threshold {thresholds[c]:.3f})")
    print(f"Abnormal: deployed gate recall {deployed_gate_recall:.3f}; v2 threshold {gate_threshold:.3f} "
          f"for {GATE_SENSITIVITY:.0%}")
    print(f"\nSaved {CONFIG_PATH}")


if __name__ == "__main__":
    main()
