"""Which NIH-labeled abnormal scans does the VinDr-retrained gate stop leading
with a finding for, compared to the current gate?

On NIH test, the retrained gate's page-level sensitivity drops from 80.6% to
71.0%, while it improves on VinDr and on external pediatric X-rays. If the
dropped cases concentrate in findings whose NIH report-mined labels are
unreliable, or in scans with a single finding (more likely to be a
labeling artifact than multi-finding scans), that points to NIH label noise
rather than real misses. Per-condition headline sensitivity for both gates,
plus the breakdown of newly dropped cases.
"""

import json

import numpy as np
import pandas as pd
import torch

from evaluate_ensemble import load_models, nih_loader
from evaluate_gate_vindr import load_gate_from, predict
from evaluate_gated_system import load_gate
from train_gate_vindr import REPORT_PATH as TRAIN_REPORT_PATH, variant_path
from train_multilabel import CONDITIONS, MODEL_DIR, load_manifest, patient_level_split

CONFIG_PATH = MODEL_DIR / "ensemble_config.json"
EVAL_REPORT_PATH = MODEL_DIR / "gate_vindr_eval_report.json"
REPORT_PATH = MODEL_DIR / "gate_nih_drops_report.json"
PREDICTIONS_PATH = MODEL_DIR / "nih_test_gate_predictions.csv"


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ours, txv, txv_indices, _ = load_models(device)
    config = json.loads(CONFIG_PATH.read_text())
    finding_t = np.array([config["thresholds"][c] for c in CONDITIONS])[None, :]
    setups = json.loads(EVAL_REPORT_PATH.read_text())["setups"]
    gate_t = {
        "current": next(v["gate_threshold"] for k, v in setups.items() if k.startswith("1.")),
        "retrained": next(v["gate_threshold"] for k, v in setups.items() if k.startswith("3.")),
    }
    winner = json.loads(TRAIN_REPORT_PATH.read_text())["winner_by_val"]
    gates = {"current": load_gate(device), "retrained": load_gate_from(variant_path(winner), device)}

    _, _, test = patient_level_split(load_manifest())
    test = test.reset_index(drop=True)
    print(f"Predicting on NIH test ({len(test)} images)...", flush=True)
    ens, g = predict(ours, txv, txv_indices, gates, nih_loader(test), device)
    any_finding = (ens >= finding_t).any(axis=1)

    test["current_headline"] = any_finding & (g["current"] >= gate_t["current"])
    test["retrained_headline"] = any_finding & (g["retrained"] >= gate_t["retrained"])
    test["current_gate_prob"], test["retrained_gate_prob"] = g["current"], g["retrained"]
    test["n_labels"] = test["labels"].apply(lambda s: 0 if s == "No Finding" else len(s.split("|")))
    test[["filename", "patient_id", "labels", "n_labels", "current_gate_prob", "retrained_gate_prob",
          "current_headline", "retrained_headline"]].to_csv(PREDICTIONS_PATH, index=False)

    abnormal = test[test["n_labels"] > 0]
    dropped = abnormal[abnormal["current_headline"] & ~abnormal["retrained_headline"]]
    gained = abnormal[~abnormal["current_headline"] & abnormal["retrained_headline"]]
    print(f"\nNIH abnormal test scans: {len(abnormal)}")
    print(f"  led with a finding by current but not retrained (dropped): {len(dropped)}")
    print(f"  led with a finding by retrained but not current (gained):  {len(gained)}")

    per_condition = {}
    print(f"\n{'condition':20s}{'n':>6s}{'current':>9s}{'retrained':>10s}{'change':>8s}")
    for c in CONDITIONS:
        has_c = abnormal[abnormal["labels"].str.split("|").apply(lambda l: c in l)]
        cur, ret = has_c["current_headline"].mean(), has_c["retrained_headline"].mean()
        per_condition[c] = {"n": len(has_c), "current": float(cur), "retrained": float(ret)}
        print(f"{c:20s}{len(has_c):>6d}{cur:>9.1%}{ret:>10.1%}{(ret - cur) * 100:>+7.1f}")

    by_count = {}
    print("\nBy number of NIH labels on the scan:")
    for label, mask in [("1 finding", abnormal["n_labels"] == 1), ("2+ findings", abnormal["n_labels"] >= 2)]:
        part = abnormal[mask]
        by_count[label] = {
            "n": len(part),
            "current": float(part["current_headline"].mean()),
            "retrained": float(part["retrained_headline"].mean()),
        }
        print(f"  {label:12s} n={len(part):5d}  current {part['current_headline'].mean():.1%}  "
              f"retrained {part['retrained_headline'].mean():.1%}")

    share_single = float((dropped["n_labels"] == 1).mean())
    print(f"\nOf the dropped scans, {share_single:.0%} had only a single NIH label "
          f"(vs {(abnormal['n_labels'] == 1).mean():.0%} of all abnormal scans)")

    REPORT_PATH.write_text(json.dumps({
        "gate_thresholds": gate_t,
        "n_abnormal": len(abnormal), "n_dropped": len(dropped), "n_gained": len(gained),
        "per_condition_headline_sensitivity": per_condition,
        "by_label_count": by_count,
        "dropped_share_single_label": share_single,
    }, indent=2))
    print(f"\nSaved report to {REPORT_PATH} and per-image predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
