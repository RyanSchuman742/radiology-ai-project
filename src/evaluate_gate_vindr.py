"""Compare gates end to end on the held-out VinDr test split, plus the
external Kermany pediatric set.

Three setups, so a gain can be attributed to the right cause:
1. current gate, deployed threshold (chosen on NIH validation)
2. current gate, threshold re-chosen on VinDr validation - isolates how much
   comes from a better cutoff alone
3. retrained gate (train_gate_vindr.py's winner), threshold chosen the same way

Thresholds for (2) and (3) are set for 90% sensitivity on VinDr validation
abnormal scans, the same rule used for the deployed threshold on NIH.

Shortcut checks - a retrained gate scoring far higher on VinDr could be
learning which scanner took the image rather than pathology (in VinDr,
images stored as MONOCHROME1 are 57% abnormal vs. 23% for MONOCHROME2):
- gate AUROC within each photometric group separately (a scanner shortcut
  can't help within a single group)
- gate AUROC on the NIH test split (different scanners entirely; real
  pathology knowledge should transfer, VinDr-specific cues shouldn't)
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from evaluate_ensemble import DualInputDataset, load_models, nih_loader
from evaluate_gated_system import gate_threshold_at_recall, kermany_loader, load_gate
from evaluate_vindr import VINDR_TO_OURS
from train_gate_vindr import REPORT_PATH as TRAIN_REPORT_PATH, variant_path
from train_multilabel import CONDITIONS, MODEL_DIR, PROJECT_ROOT, load_manifest, patient_level_split

VINDR_DIR = PROJECT_ROOT / "data" / "vindr"
CONFIG_PATH = MODEL_DIR / "ensemble_config.json"
REPORT_PATH = MODEL_DIR / "gate_vindr_eval_report.json"
GATE_SENSITIVITY = 0.90


def load_gate_from(path, device):
    gate = load_gate(device)  # current gate architecture; swap weights
    gate.load_state_dict(torch.load(path, map_location=device)["model_state"])
    return gate.eval()


def predict(ours, txv, txv_indices, gates: dict, loader, device):
    """Returns ensemble probs [N,14] and {gate name: abnormal probs [N]}."""
    ens, gate_probs = [], {n: [] for n in gates}
    with torch.no_grad():
        for ours_in, txv_in, _ in loader:
            ours_in = ours_in.to(device)
            p_ours = torch.sigmoid(ours(ours_in)).cpu().numpy()
            p_txv = txv(txv_in.to(device))[:, txv_indices].cpu().numpy()
            ens.append((p_ours + p_txv) / 2)
            for n, g in gates.items():
                gate_probs[n].append(torch.sigmoid(g(ours_in)).cpu().numpy().ravel())
    return np.concatenate(ens), {n: np.concatenate(v) for n, v in gate_probs.items()}


def vindr_loader(df):
    paths = [str(VINDR_DIR / "images" / f"{i}.png") for i in df["image_id"]]
    empty = torch.zeros(len(CONDITIONS))
    return DataLoader(DualInputDataset(paths, [empty] * len(paths)), batch_size=32, shuffle=False, num_workers=4)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ours, txv, txv_indices, _ = load_models(device)
    config = json.loads(CONFIG_PATH.read_text())
    thresholds = np.array([config["thresholds"][c] for c in CONDITIONS])[None, :]

    winner = json.loads(TRAIN_REPORT_PATH.read_text())["winner_by_val"]
    gates = {"current": load_gate(device), "retrained": load_gate_from(variant_path(winner), device)}

    labels = pd.read_csv(VINDR_DIR / "labels.csv")
    split = pd.read_csv(VINDR_DIR / "split.csv")[["image_id", "split"]]
    meta = pd.read_csv(VINDR_DIR / "dicom_meta.csv")
    labels = labels.merge(split, on="image_id").merge(meta, on="image_id")
    val, test = labels[labels["split"] == "val"], labels[labels["split"] == "test"]

    print("Predicting on VinDr val, VinDr test, Kermany normal, Kermany pneumonia...", flush=True)
    _, val_g = predict(ours, txv, txv_indices, gates, vindr_loader(val), device)
    test_ens, test_g = predict(ours, txv, txv_indices, gates, vindr_loader(test), device)
    kn_ens, kn_g = predict(ours, txv, txv_indices, gates, kermany_loader("NORMAL"), device)
    kp_ens, kp_g = predict(ours, txv, txv_indices, gates, kermany_loader("PNEUMONIA"), device)
    print("Predicting on NIH test...", flush=True)
    _, _, nih_test = patient_level_split(load_manifest())
    _, nih_g = predict(ours, txv, txv_indices, gates, nih_loader(nih_test), device)
    nih_abnormal = (nih_test["labels"] != "No Finding").to_numpy()

    val_abnormal = val["abnormal"].to_numpy().astype(bool)
    test_abnormal = test["abnormal"].to_numpy().astype(bool)
    majority = test["n_rads"].to_numpy() / 2
    trackable = np.zeros(len(test), dtype=bool)
    for c in VINDR_TO_OURS:
        trackable |= test[c].to_numpy() > majority

    setups = {
        "1. current gate, deployed threshold": ("current", config["gate_threshold"]),
        "2. current gate, VinDr-tuned threshold": ("current", gate_threshold_at_recall(val_abnormal, val_g["current"], GATE_SENSITIVITY)),
        f"3. retrained gate ({winner}), VinDr-tuned threshold": ("retrained", gate_threshold_at_recall(val_abnormal, val_g["retrained"], GATE_SENSITIVITY)),
    }

    def headline(ens, gate_probs, gate_threshold):
        return (ens >= thresholds).any(axis=1) & (gate_probs >= gate_threshold)

    photometric = test["photometric"].to_numpy()
    report = {
        "gate_test_auroc": {n: float(roc_auc_score(test_abnormal, g)) for n, g in test_g.items()},
        "gate_test_auroc_within_photometric_group": {
            n: {
                grp: float(roc_auc_score(test_abnormal[photometric == grp], g[photometric == grp]))
                for grp in sorted(set(photometric))
            }
            for n, g in test_g.items()
        },
        "gate_nih_test_auroc": {n: float(roc_auc_score(nih_abnormal, g)) for n, g in nih_g.items()},
        "setups": {},
    }
    for name, (gate_name, t) in setups.items():
        h_test = headline(test_ens, test_g[gate_name], t)
        report["setups"][name] = {
            "gate_threshold": t,
            "vindr_healthy_false_alarm": float(h_test[~test_abnormal].mean()),
            "vindr_abnormal_caught": float(h_test[test_abnormal].mean()),
            "vindr_trackable_caught": float(h_test[trackable].mean()),
            "kermany_healthy_false_alarm": float(headline(kn_ens, kn_g[gate_name], t).mean()),
            "kermany_pneumonia_caught": float(headline(kp_ens, kp_g[gate_name], t).mean()),
        }

    print(f"\nGate AUROC on VinDr test: " + ", ".join(f"{n}={v:.3f}" for n, v in report["gate_test_auroc"].items()))
    for n, groups in report["gate_test_auroc_within_photometric_group"].items():
        print(f"  {n}, within scanner group: " + ", ".join(f"{g}={v:.3f}" for g, v in groups.items()))
    print(f"Gate AUROC on NIH test (other scanners): " + ", ".join(f"{n}={v:.3f}" for n, v in report["gate_nih_test_auroc"].items()))
    print("\nPage leads with a finding (VinDr test = held out; Kermany = external pediatric):")
    print(f"{'setup':52s} {'healthy':>8s} {'abnormal':>9s} {'tracked':>8s} {'ext.healthy':>12s} {'ext.pneum.':>11s}")
    for name, r in report["setups"].items():
        print(f"{name:52s} {r['vindr_healthy_false_alarm']:>8.1%} {r['vindr_abnormal_caught']:>9.1%} "
              f"{r['vindr_trackable_caught']:>8.1%} {r['kermany_healthy_false_alarm']:>12.1%} {r['kermany_pneumonia_caught']:>11.1%}")

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
