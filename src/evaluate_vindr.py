"""Measure the deployed system (ensemble + gate) against radiologist labels.

VinDr-CXR (Kaggle release, 15,000 images, 3 radiologists each) is the first
evaluation set with trustworthy "normal" labels - NIH's "No Finding" is
report-mined and noisy, and Kermany is pediatric. None of our models
trained on it.

In this release every image is unanimous on normal vs. abnormal (all 3
radiologists said "No finding", or all 3 marked something), so borderline
cases are likely underrepresented - real-world false-alarm rates may be
somewhat worse than measured here.

VinDr tracks some findings we don't (e.g. aortic enlargement, calcification),
so sensitivity is reported both over all abnormal scans and over the subset
where a majority of radiologists marked something that maps to one of our
14 conditions - the fair measure of what the app is supposed to catch.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from evaluate_ensemble import DualInputDataset, load_models
from evaluate_gated_system import load_gate
from train_multilabel import CONDITIONS, MODEL_DIR, PROJECT_ROOT

VINDR_DIR = PROJECT_ROOT / "data" / "vindr"
REPORT_PATH = MODEL_DIR / "vindr_eval_report.json"
CONFIG_PATH = MODEL_DIR / "ensemble_config.json"

# VinDr class -> our conditions it corresponds to
VINDR_TO_OURS = {
    "Atelectasis": ["Atelectasis"],
    "Cardiomegaly": ["Cardiomegaly"],
    "Consolidation": ["Consolidation"],
    "Infiltration": ["Infiltration"],
    "Pleural effusion": ["Effusion"],
    "Pleural thickening": ["Pleural_Thickening"],
    "Pneumothorax": ["Pneumothorax"],
    "Pulmonary fibrosis": ["Fibrosis"],
    "Nodule/Mass": ["Nodule", "Mass"],
}


def predict(ours, txv, txv_indices, gate, loader, device):
    ens, gate_probs = [], []
    with torch.no_grad():
        for ours_in, txv_in, _ in loader:
            ours_in = ours_in.to(device)
            p_ours = torch.sigmoid(ours(ours_in)).cpu().numpy()
            p_txv = txv(txv_in.to(device))[:, txv_indices].cpu().numpy()
            ens.append((p_ours + p_txv) / 2)
            gate_probs.append(torch.sigmoid(gate(ours_in)).cpu().numpy().ravel())
    return np.concatenate(ens), np.concatenate(gate_probs)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ours, txv, txv_indices, _ = load_models(device)
    gate = load_gate(device)
    config = json.loads(CONFIG_PATH.read_text())
    thresholds = np.array([config["thresholds"][c] for c in CONDITIONS])[None, :]

    labels = pd.read_csv(VINDR_DIR / "labels.csv")
    paths = [str(VINDR_DIR / "images" / f"{i}.png") for i in labels["image_id"]]
    empty = torch.zeros(len(CONDITIONS))
    loader = DataLoader(DualInputDataset(paths, [empty] * len(paths)), batch_size=32, shuffle=False, num_workers=4)

    print(f"Predicting on {len(paths)} VinDr images...", flush=True)
    ens, gate_probs = predict(ours, txv, txv_indices, gate, loader, device)

    abnormal = labels["abnormal"].to_numpy().astype(bool)
    majority = labels["n_rads"].to_numpy() / 2
    trackable = np.zeros(len(labels), dtype=bool)
    for vindr_class in VINDR_TO_OURS:
        if vindr_class in labels:
            trackable |= labels[vindr_class].to_numpy() > majority

    any_finding = (ens >= thresholds).any(axis=1)
    gate_abnormal = gate_probs >= config["gate_threshold"]
    headline_finding = any_finding & gate_abnormal  # what the page leads with

    normal = ~abnormal
    report = {
        "n_images": len(labels),
        "n_normal": int(normal.sum()),
        "n_abnormal": int(abnormal.sum()),
        "n_abnormal_trackable": int(trackable.sum()),
        "gate_auroc_normal_vs_abnormal": float(roc_auc_score(abnormal, gate_probs)),
        "healthy_with_any_finding_no_gate": float(any_finding[normal].mean()),
        "healthy_with_headline_finding": float(headline_finding[normal].mean()),
        "abnormal_with_any_finding": float(any_finding[abnormal].mean()),
        "abnormal_with_headline_finding": float(headline_finding[abnormal].mean()),
        "trackable_with_any_finding": float(any_finding[trackable].mean()),
        "trackable_with_headline_finding": float(headline_finding[trackable].mean()),
    }

    print(f"\nVinDr: {report['n_normal']} normal, {report['n_abnormal']} abnormal "
          f"({report['n_abnormal_trackable']} with a majority-agreed finding we track)")
    print(f"Gate AUROC (normal vs abnormal): {report['gate_auroc_normal_vs_abnormal']:.3f}")
    print("\nHealthy scans (lower is better):")
    print(f"  flagged with >=1 finding, no gate:          {report['healthy_with_any_finding_no_gate']:.1%}")
    print(f"  page leads with a finding (deployed):       {report['healthy_with_headline_finding']:.1%}")
    print("\nAbnormal scans (higher is better):")
    print(f"  all abnormal - any finding shown anywhere:  {report['abnormal_with_any_finding']:.1%}")
    print(f"  all abnormal - page leads with a finding:   {report['abnormal_with_headline_finding']:.1%}")
    print(f"  trackable    - any finding shown anywhere:  {report['trackable_with_any_finding']:.1%}")
    print(f"  trackable    - page leads with a finding:   {report['trackable_with_headline_finding']:.1%}")

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
