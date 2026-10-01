"""Combined gate: average the NIH-trained and VinDr-retrained gates' scores.

The VinDr-retrained gate cut false alarms on VinDr and on external pediatric
X-rays, but on NIH it led with "no significant abnormality" for ~10 points
more sick scans than the current gate. The two gates learned "normal" from
different labels and different hospitals, so they likely make different
mistakes - the same reason averaging our model with TorchXRayVision helped.

Threshold rule for the combined score: 90% sensitivity on NIH validation AND
VinDr validation at once (the lower of the two thresholds), so neither
dataset's sick scans are traded away. Thresholds tuned on each validation set
alone are reported alongside to show the tradeoff.
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from evaluate_ensemble import load_models, nih_loader
from evaluate_gate_vindr import load_gate_from, predict, vindr_loader
from evaluate_gated_system import gate_threshold_at_recall, kermany_loader, load_gate
from train_gate_vindr import REPORT_PATH as TRAIN_REPORT_PATH, variant_path
from train_multilabel import CONDITIONS, MODEL_DIR, PROJECT_ROOT, load_manifest, patient_level_split

VINDR_DIR = PROJECT_ROOT / "data" / "vindr"
CONFIG_PATH = MODEL_DIR / "ensemble_config.json"
PRIOR_EVAL_PATH = MODEL_DIR / "gate_vindr_eval_report.json"
REPORT_PATH = MODEL_DIR / "combined_gate_report.json"
SENSITIVITY = 0.90


def gate_scores(gates, loader, device):
    """Gate abnormal probabilities only (validation sets don't need the ensemble)."""
    out = {n: [] for n in gates}
    with torch.no_grad():
        for ours_in, _, _ in loader:
            ours_in = ours_in.to(device)
            for n, g in gates.items():
                out[n].append(torch.sigmoid(g(ours_in)).cpu().numpy().ravel())
    return {n: np.concatenate(v) for n, v in out.items()}


def with_combined(scores: dict) -> dict:
    return {**scores, "combined": (scores["current"] + scores["retrained"]) / 2}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ours, txv, txv_indices, _ = load_models(device)
    config = json.loads(CONFIG_PATH.read_text())
    finding_thresholds = np.array([config["thresholds"][c] for c in CONDITIONS])[None, :]
    winner = json.loads(TRAIN_REPORT_PATH.read_text())["winner_by_val"]
    prior = json.loads(PRIOR_EVAL_PATH.read_text())["setups"]
    retrained_threshold = next(v["gate_threshold"] for k, v in prior.items() if k.startswith("3."))
    gates = {"current": load_gate(device), "retrained": load_gate_from(variant_path(winner), device)}

    _, nih_val, nih_test = patient_level_split(load_manifest())
    vindr = pd.read_csv(VINDR_DIR / "labels.csv").merge(pd.read_csv(VINDR_DIR / "split.csv")[["image_id", "split"]])
    vindr_val, vindr_test = vindr[vindr["split"] == "val"], vindr[vindr["split"] == "test"]

    print("Scoring validation sets (gates only)...", flush=True)
    nv = with_combined(gate_scores(gates, nih_loader(nih_val), device))
    vv = with_combined(gate_scores(gates, vindr_loader(vindr_val), device))
    nih_val_abn = (nih_val["labels"] != "No Finding").to_numpy()
    vindr_val_abn = vindr_val["abnormal"].to_numpy().astype(bool)

    t_nih = gate_threshold_at_recall(nih_val_abn, nv["combined"], SENSITIVITY)
    t_vindr = gate_threshold_at_recall(vindr_val_abn, vv["combined"], SENSITIVITY)
    setups = {
        "current gate (live now)": ("current", config["gate_threshold"]),
        "retrained gate": ("retrained", retrained_threshold),
        "combined, 90% on both (proposed)": ("combined", min(t_nih, t_vindr)),
        "combined, tuned on NIH only": ("combined", t_nih),
        "combined, tuned on VinDr only": ("combined", t_vindr),
    }

    print("Scoring test sets (ensemble + gates)...", flush=True)
    test_sets = {}
    for name, ldr, abnormal in [
        ("NIH", nih_loader(nih_test), (nih_test["labels"] != "No Finding").to_numpy()),
        ("VinDr", vindr_loader(vindr_test), vindr_test["abnormal"].to_numpy().astype(bool)),
        ("ext. healthy", kermany_loader("NORMAL"), None),
        ("ext. pneumonia", kermany_loader("PNEUMONIA"), None),
    ]:
        ens, g = predict(ours, txv, txv_indices, gates, ldr, device)
        test_sets[name] = (ens, with_combined(g), abnormal)
        print(f"  {name} done", flush=True)

    report = {"thresholds": {k: v[1] for k, v in setups.items()}, "gate_auroc": {}, "setups": {}}
    for ds in ["NIH", "VinDr"]:
        _, g, abnormal = test_sets[ds]
        report["gate_auroc"][ds] = {n: float(roc_auc_score(abnormal, s)) for n, s in g.items()}

    for setup, (gate_name, t) in setups.items():
        r = {}
        for ds, (ens, g, abnormal) in test_sets.items():
            headline = (ens >= finding_thresholds).any(axis=1) & (g[gate_name] >= t)
            if abnormal is None:
                r[ds] = float(headline.mean())
            else:
                r[f"{ds} healthy"] = float(headline[~abnormal].mean())
                r[f"{ds} sick"] = float(headline[abnormal].mean())
        report["setups"][setup] = r

    print("\nGate AUROC:")
    for ds, a in report["gate_auroc"].items():
        print(f"  {ds}: " + ", ".join(f"{n}={v:.3f}" for n, v in a.items()))

    cols = ["NIH healthy", "NIH sick", "VinDr healthy", "VinDr sick", "ext. healthy", "ext. pneumonia"]
    print("\nPage leads with a finding (healthy: lower is better; sick/pneumonia: higher is better):")
    print(f"{'setup':36s}" + "".join(f"{c:>15s}" for c in cols))
    for setup, r in report["setups"].items():
        print(f"{setup:36s}" + "".join(f"{r[c]:>15.1%}" for c in cols))

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
