"""Deployed vs. v2 away from VinDr, where v2 has no home advantage.

v2 trained on VinDr's train split, so the scoreboard (VinDr test) shares
hospitals, scanners and labelers with part of its training data while the
deployed models never saw VinDr. This checks whether v2's gains hold on:
- NIH ChestX-ray14 test split (same patients held out for both systems;
  both carry TorchXRayVision's NIH contamination, the deployed ensemble via
  its DenseNet member and v2 via its starting weights - so comparable)
- Kermany pediatric X-rays (neither system trained on them): NORMAL for
  false alarms, PNEUMONIA for whether sick scans still get flagged.
  Children's X-rays are out of distribution for both - a robustness check,
  not a clinical number.

"Flagged" = the page leads with a finding: a condition over its threshold
and the abnormality output over its threshold (the deployed gate's rule).

Writes models/external_comparison_report.json.
"""

import json

import numpy as np
import pandas as pd
import torch

from calibrate_v2 import CONFIG_PATH as V2_CONFIG_PATH, load_v2, predict_v2
from evaluate_ensemble import KERMANY_DIR, load_models, nih_loader, per_class_auroc, per_class_recall
from evaluate_gated_system import kermany_loader, load_gate, predict_all
from train_multilabel import CONDITIONS, MODEL_DIR, load_manifest, patient_level_split

REPORT_PATH = MODEL_DIR / "external_comparison_report.json"
DEPLOYED_CONFIG_PATH = MODEL_DIR / "ensemble_config.json"


def kermany_paths(label: str) -> list[str]:
    paths = []
    for split in ["train", "validation", "test"]:
        labels = pd.read_csv(KERMANY_DIR / split / "labels.csv")
        paths += [str(KERMANY_DIR / split / f) for f in labels[labels["label"] == label]["filename"]]
    return paths


def flagged(probs, abnormal_prob, config) -> np.ndarray:
    above = probs >= np.array([config["thresholds"][c] for c in CONDITIONS])[None, :]
    return above.any(axis=1) & (abnormal_prob >= config["gate_threshold"])


def summarize(nih_probs, nih_abn, nih_targets, kn, kp, config) -> dict:
    healthy = nih_targets.sum(axis=1) == 0
    auroc = per_class_auroc(nih_targets, nih_probs)
    return {
        "nih_test_macro_auroc": float(np.mean([v for v in auroc.values() if v is not None])),
        "nih_test_per_class_auroc": auroc,
        "nih_test_per_class_recall": per_class_recall(nih_targets, nih_probs, config["thresholds"]),
        "nih_healthy_false_alarm": float(flagged(nih_probs, nih_abn, config)[healthy].mean()),
        "nih_sick_flagged": float(flagged(nih_probs, nih_abn, config)[~healthy].mean()),
        "kermany_normal_false_alarm": float(flagged(*kn, config).mean()),
        "kermany_pneumonia_flagged": float(flagged(*kp, config).mean()),
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, _, test_df = patient_level_split(load_manifest())
    results = {}

    print(f"Deployed: NIH test ({len(test_df)}) + Kermany...", flush=True)
    ours, txv, idx, _ = load_models(device)
    gate = load_gate(device)
    nih_probs, nih_abn, targets = predict_all(ours, txv, idx, gate, nih_loader(test_df), device)
    kn = predict_all(ours, txv, idx, gate, kermany_loader("NORMAL"), device)[:2]
    kp = predict_all(ours, txv, idx, gate, kermany_loader("PNEUMONIA"), device)[:2]
    results["deployed"] = summarize(nih_probs, nih_abn, targets, kn, kp, json.loads(DEPLOYED_CONFIG_PATH.read_text()))
    del ours, txv, gate

    print("v2: NIH test + Kermany...", flush=True)
    v2 = load_v2(device)
    nih_probs, nih_abn = predict_v2(v2, test_df["path"].tolist(), device)
    kn = predict_v2(v2, kermany_paths("NORMAL"), device)
    kp = predict_v2(v2, kermany_paths("PNEUMONIA"), device)
    results["v2"] = summarize(nih_probs, nih_abn, targets, kn, kp, json.loads(V2_CONFIG_PATH.read_text()))

    REPORT_PATH.write_text(json.dumps(results, indent=2))
    keys = ["nih_test_macro_auroc", "nih_healthy_false_alarm", "nih_sick_flagged",
            "kermany_normal_false_alarm", "kermany_pneumonia_flagged"]
    print(f"\n{'':28s} {'deployed':>9s} {'v2':>9s}")
    for k in keys:
        print(f"{k:28s} {results['deployed'][k]:9.3f} {results['v2'][k]:9.3f}")
    print("\nNIH test per-class AUROC (deployed -> v2):")
    for c in CONDITIONS:
        print(f"  {c:20s} {results['deployed']['nih_test_per_class_auroc'][c]:.3f} -> "
              f"{results['v2']['nih_test_per_class_auroc'][c]:.3f}")
    print(f"\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
