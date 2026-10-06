"""Second, independent scoreboard: the CheXpert validation set.

202 frontal X-rays from 200 Stanford patients, labeled by board-certified
radiologists. No model in this project trained on Stanford data except
through TorchXRayVision's starting weights, which used CheXpert's *train*
split (different patients). Unlike the VinDr scoreboard, v2 has no home
advantage here.

Labels come from CheXlocalize (data/chexpert/chexlocalize): radiologists
outlined every finding present per the ground-truth labels, so a condition
is positive exactly when it has an outline - checked against 30 of
CheXlocalize's own label-carrying files (30/30 agree). Images come from
CheXpert Plus's validation PNGs (data/chexpert/valid).

Abnormal = any of CheXlocalize's 9 pathology labels (Support Devices
excluded). Scans with none of the 9 count as healthy, though they may carry
findings CheXlocalize doesn't track - so "healthy false alarm" is approximate.

Small set: 95% intervals are wide. Conditions with < MIN_POSITIVES
positives are shown but left out of the macro average.

Usage: python src/evaluate_chexpert.py   (scores "deployed" and "v2")
"""

import json

import numpy as np

from scoreboard import SYSTEMS, auroc, bootstrap, rate, with_intervals
from train_multilabel import CONDITIONS, MODEL_DIR, PROJECT_ROOT

CHEXPERT_DIR = PROJECT_ROOT / "data" / "chexpert"
REPORT_PATH = MODEL_DIR / "chexpert_val_report.json"
MIN_POSITIVES = 15
HEADLINE = ["macro_auroc", "abnormal_auroc", "healthy_false_alarm", "sick_flagged"]
LOWER_IS_BETTER = {"healthy_false_alarm"}

# CheXpert label -> our conditions (Lung Lesion scored as max of Nodule/Mass)
CHEXPERT_TO_OURS = {
    "Atelectasis": ["Atelectasis"],
    "Cardiomegaly": ["Cardiomegaly"],
    "Consolidation": ["Consolidation"],
    "Edema": ["Edema"],
    "Pleural Effusion": ["Effusion"],
    "Pneumothorax": ["Pneumothorax"],
    "Lung Lesion": ["Nodule", "Mass"],
}


def load_set():
    ann = json.loads((CHEXPERT_DIR / "chexlocalize" / "gt_annotations_val.json").read_text())
    tasks = json.loads((CHEXPERT_DIR / "chexlocalize" / "chexlocalize_tasks.json").read_text())["chexplanation_tasks"]
    pathologies = [t for t in tasks if t != "Support Devices"]
    paths = sorted((CHEXPERT_DIR / "valid").glob("*/*/*frontal*.png"))
    ids = ["_".join(p.relative_to(CHEXPERT_DIR / "valid").with_suffix("").parts) for p in paths]
    truth = {f"pos:{c}": np.array([c in ann.get(i, {}) for i in ids]) for c in CHEXPERT_TO_OURS}
    truth["abnormal"] = np.array([any(t in ann.get(i, {}) for t in pathologies) for i in ids])
    return [str(p) for p in paths], truth


def decisions(probs, abnormal_prob, config) -> dict:
    above = probs >= np.array([config["thresholds"][c] for c in CONDITIONS])[None, :]
    pred = {"abnormal_prob": abnormal_prob,
            "headline": above.any(axis=1) & (abnormal_prob >= config["gate_threshold"])}
    for c, ours in CHEXPERT_TO_OURS.items():
        pred[f"score:{c}"] = probs[:, [CONDITIONS.index(o) for o in ours]].max(axis=1)
    return pred


def metrics(truth, pred, classes, idx) -> dict:
    t = {k: v[idx] for k, v in truth.items()}
    p = {k: v[idx] for k, v in pred.items()}
    m = {"abnormal_auroc": auroc(t["abnormal"], p["abnormal_prob"]),
         "healthy_false_alarm": rate(p["headline"][~t["abnormal"]]),
         "sick_flagged": rate(p["headline"][t["abnormal"]])}
    for c in CHEXPERT_TO_OURS:
        m[f"auroc:{c}"] = auroc(t[f"pos:{c}"], p[f"score:{c}"])
    m["macro_auroc"] = float(np.mean([m[f"auroc:{c}"] for c in classes]))
    return m


def main():
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    paths, truth = load_set()
    classes = [c for c in CHEXPERT_TO_OURS if truth[f"pos:{c}"].sum() >= MIN_POSITIVES]
    n = len(paths)
    print(f"{n} frontal CheXpert validation images; {int((~truth['abnormal']).sum())} with no tracked pathology", flush=True)

    preds, report = {}, {"n_images": n, "n_healthy": int((~truth["abnormal"]).sum()), "headline_classes": classes,
                         "positives": {c: int(truth[f"pos:{c}"].sum()) for c in CHEXPERT_TO_OURS}, "systems": {}}
    for name in ["deployed", "v2"]:
        print(f"Predicting with {name}...", flush=True)
        probs, abnormal_prob, config = SYSTEMS[name](paths, device)
        preds[name] = decisions(probs, abnormal_prob, config)
        point = metrics(truth, preds[name], classes, np.arange(n))
        report["systems"][name] = with_intervals(point, bootstrap(lambda i: metrics(truth, preds[name], classes, i), n))

    diff = lambda i: {k: metrics(truth, preds["v2"], classes, i)[k] - metrics(truth, preds["deployed"], classes, i)[k]
                      for k in HEADLINE}
    report["v2_minus_deployed"] = with_intervals(diff(np.arange(n)), bootstrap(diff, n))
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    fmt = lambda m, pct: (f"{m['value']:.1%} ({m['ci95'][0]:.1%}-{m['ci95'][1]:.1%})" if pct
                          else f"{m['value']:.3f} ({m['ci95'][0]:.3f}-{m['ci95'][1]:.3f})")
    print(f"\n{'':24s} {'deployed':>24s} {'v2':>24s}   v2 - deployed")
    for k in HEADLINE + [f"auroc:{c}" for c in CHEXPERT_TO_OURS]:
        pct = k in ("healthy_false_alarm", "sick_flagged")
        d = report["v2_minus_deployed"].get(k)
        verdict = ""
        if d:
            lo, hi = d["ci95"]
            better = d["value"] < 0 if k in LOWER_IS_BETTER else d["value"] > 0
            verdict = f"{d['value']:+.3f} [{lo:+.3f}, {hi:+.3f}] " + (
                "no clear difference" if lo <= 0 <= hi else ("better" if better else "worse"))
        label = k if not k.startswith("auroc:") else f"  {k[6:]} (n+={report['positives'][k[6:]]})"
        print(f"{label:24s} {fmt(report['systems']['deployed'][k], pct):>24s} "
              f"{fmt(report['systems']['v2'][k], pct):>24s}   {verdict}")
    print(f"\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
