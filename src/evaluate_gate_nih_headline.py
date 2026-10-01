"""Page-level check on NIH for the retrained (VinDr) gate vs. the current one.

The retrained gate ranks NIH scans slightly worse (AUROC 0.752 vs 0.772), so
before deploying it: what share of NIH test scans would the page lead with a
finding for, healthy and abnormal, at each gate's chosen threshold?
Thresholds come from gate_vindr_eval_report.json (setups 1 and 3).
"""

import json

import numpy as np
import torch

from evaluate_ensemble import load_models, nih_loader
from evaluate_gate_vindr import load_gate_from, predict
from evaluate_gated_system import load_gate
from train_gate_vindr import REPORT_PATH as TRAIN_REPORT_PATH, variant_path
from train_multilabel import CONDITIONS, MODEL_DIR, load_manifest, patient_level_split

CONFIG_PATH = MODEL_DIR / "ensemble_config.json"
EVAL_REPORT_PATH = MODEL_DIR / "gate_vindr_eval_report.json"
REPORT_PATH = MODEL_DIR / "gate_nih_headline_report.json"


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ours, txv, txv_indices, _ = load_models(device)
    config = json.loads(CONFIG_PATH.read_text())
    thresholds = np.array([config["thresholds"][c] for c in CONDITIONS])[None, :]
    winner = json.loads(TRAIN_REPORT_PATH.read_text())["winner_by_val"]
    setups = json.loads(EVAL_REPORT_PATH.read_text())["setups"]
    gate_thresholds = {
        "current": next(v["gate_threshold"] for k, v in setups.items() if k.startswith("1.")),
        "retrained": next(v["gate_threshold"] for k, v in setups.items() if k.startswith("3.")),
    }
    gates = {"current": load_gate(device), "retrained": load_gate_from(variant_path(winner), device)}

    _, _, test_df = patient_level_split(load_manifest())
    print(f"Predicting on NIH test ({len(test_df)} images)...", flush=True)
    ens, gate_probs = predict(ours, txv, txv_indices, gates, nih_loader(test_df), device)
    abnormal = (test_df["labels"] != "No Finding").to_numpy()
    any_finding = (ens >= thresholds).any(axis=1)

    report = {}
    for name, t in gate_thresholds.items():
        headline = any_finding & (gate_probs[name] >= t)
        report[name] = {
            "gate_threshold": t,
            "healthy_headline_finding": float(headline[~abnormal].mean()),
            "abnormal_headline_finding": float(headline[abnormal].mean()),
        }
        print(f"{name:10s} healthy led with a finding: {report[name]['healthy_headline_finding']:.1%}   "
              f"abnormal led with a finding: {report[name]['abnormal_headline_finding']:.1%}")

    REPORT_PATH.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
