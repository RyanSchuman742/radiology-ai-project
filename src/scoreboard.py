"""VinDr scoreboard: one fixed benchmark that every model change is scored on.

The test set is the 2,250-image held-out split in data/vindr/split.csv
(15% of VinDr, stratified normal/abnormal, made by train_gate_vindr.py).
No model may train or tune on it - future training uses only the "train"
split, and thresholds come from "val". Labels are radiologist reads (3 per
image), so unlike NIH's report-mined labels the scores here can be trusted.

Every number comes with a 95% bootstrap confidence interval, and `compare`
runs a paired bootstrap on the difference between two systems, so a change
only counts as an improvement when its interval excludes zero.

Headline numbers:
- macro_auroc: mean per-condition AUROC over the conditions with at least
  MIN_POSITIVES majority-agreed positives in the test set (threshold-free;
  the main ranking number)
- abnormal_auroc: normal vs. abnormal from the system's abnormality score
- healthy_false_alarm: share of healthy scans where the page leads with a
  finding (lower is better)
- trackable_sensitivity: share of scans with a majority-agreed finding we
  track where the page leads with a finding (higher is better)

Per condition, a VinDr image is positive when at least 2 of 3 radiologists
marked it and negative when none did; 1-vote images are left out.

Caveat: VinDr has no patient IDs, so the split is by image. A future model
trained on VinDr's train split may have seen other images of a test patient,
which would inflate its scores; the deployed models never saw VinDr.

Usage:
    python src/scoreboard.py score deployed
    python src/scoreboard.py compare deployed <other system>
"""

import argparse
import json
import subprocess
from datetime import date

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from evaluate_ensemble import DualInputDataset, load_models
from evaluate_gated_system import load_gate
from evaluate_vindr import VINDR_TO_OURS, predict as predict_deployed
from train_multilabel import CONDITIONS, MODEL_DIR, PROJECT_ROOT

VINDR_DIR = PROJECT_ROOT / "data" / "vindr"
SCOREBOARD_DIR = MODEL_DIR / "scoreboard"
CONFIG_PATH = MODEL_DIR / "ensemble_config.json"
MIN_POSITIVES = 30
N_BOOT = 2000
SEED = 42

HEADLINE = ["macro_auroc", "abnormal_auroc", "healthy_false_alarm", "trackable_sensitivity"]
LOWER_IS_BETTER = {"healthy_false_alarm"}


# --- systems ----------------------------------------------------------------
# Each takes image paths and returns (probs [N,14] in CONDITIONS order,
# abnormality score [N], decision config with "thresholds" and
# "gate_threshold"). Register new models here to put them on the board.

def deployed_system(paths, device):
    ours, txv, txv_indices, _ = load_models(device)
    gate = load_gate(device)
    empty = torch.zeros(len(CONDITIONS))
    loader = DataLoader(DualInputDataset(paths, [empty] * len(paths)), batch_size=32, shuffle=False, num_workers=4)
    probs, abnormal = predict_deployed(ours, txv, txv_indices, gate, loader, device)
    return probs, abnormal, json.loads(CONFIG_PATH.read_text())


SYSTEMS = {"deployed": deployed_system}


# --- ground truth and decisions ---------------------------------------------

def load_test_set() -> pd.DataFrame:
    split = pd.read_csv(VINDR_DIR / "split.csv")
    labels = pd.read_csv(VINDR_DIR / "labels.csv")
    meta = pd.read_csv(VINDR_DIR / "dicom_meta.csv")
    test = split.loc[split["split"] == "test", ["image_id"]]
    return test.merge(labels, on="image_id").merge(meta, on="image_id").reset_index(drop=True)


def ground_truth(test: pd.DataFrame) -> dict:
    """Flat dict of boolean arrays, so a bootstrap resample is one index op."""
    truth = {
        "abnormal": test["abnormal"].to_numpy().astype(bool),
        "mono1": (test["photometric"] == "MONOCHROME1").to_numpy(),
        "trackable": np.zeros(len(test), dtype=bool),
    }
    for c in VINDR_TO_OURS:
        votes = test[c].to_numpy()
        truth[f"pos:{c}"] = votes >= 2
        truth[f"neg:{c}"] = votes == 0
        truth["trackable"] |= truth[f"pos:{c}"]
    return truth


def headline_classes(truth: dict) -> list[str]:
    return [c for c in VINDR_TO_OURS if truth[f"pos:{c}"].sum() >= MIN_POSITIVES]


def decisions(probs, abnormal_prob, config) -> dict:
    """What the app would show, per image. Nodule/Mass scores as the higher
    of our Nodule and Mass outputs."""
    above = probs >= np.array([config["thresholds"][c] for c in CONDITIONS])[None, :]
    pred = {
        "abnormal_prob": abnormal_prob,
        "any_finding": above.any(axis=1),
        "gate_ok": abnormal_prob >= config["gate_threshold"],
    }
    for c, ours in VINDR_TO_OURS.items():
        cols = [CONDITIONS.index(o) for o in ours]
        pred[f"score:{c}"] = probs[:, cols].max(axis=1)
        pred[f"flag:{c}"] = above[:, cols].any(axis=1)
    return pred


# --- metrics ----------------------------------------------------------------

def auroc(y, s) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def rate(flags) -> float:
    return float(flags.mean()) if len(flags) else np.nan


def metrics(truth: dict, pred: dict, classes: list[str], idx) -> dict:
    t = {k: v[idx] for k, v in truth.items()}
    p = {k: v[idx] for k, v in pred.items()}
    normal = ~t["abnormal"]
    headline = p["any_finding"] & p["gate_ok"]  # the page leads with a finding

    m = {
        "healthy_false_alarm": rate(headline[normal]),
        "healthy_any_finding_no_gate": rate(p["any_finding"][normal]),
        "trackable_sensitivity": rate(headline[t["trackable"]]),
        "abnormal_auroc": auroc(t["abnormal"], p["abnormal_prob"]),
        "abnormal_auroc_monochrome1": auroc(t["abnormal"][t["mono1"]], p["abnormal_prob"][t["mono1"]]),
        "abnormal_auroc_monochrome2": auroc(t["abnormal"][~t["mono1"]], p["abnormal_prob"][~t["mono1"]]),
    }
    for c in VINDR_TO_OURS:
        pos, neg = t[f"pos:{c}"], t[f"neg:{c}"]
        keep = pos | neg
        m[f"auroc:{c}"] = auroc(pos[keep], p[f"score:{c}"][keep])
        m[f"sensitivity:{c}"] = rate(p[f"flag:{c}"][pos])
        m[f"healthy_flag_rate:{c}"] = rate(p[f"flag:{c}"][normal])
    m["macro_auroc"] = float(np.mean([m[f"auroc:{c}"] for c in classes]))
    return m


def bootstrap(fn, n: int) -> list[dict]:
    rng = np.random.default_rng(SEED)
    return [fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]


def with_intervals(point: dict, samples: list[dict]) -> dict:
    out = {}
    for k, v in point.items():
        lo, hi = np.nanpercentile([s[k] for s in samples], [2.5, 97.5])
        out[k] = {"value": v, "ci95": [float(lo), float(hi)]}
    return out


# --- commands ---------------------------------------------------------------

def git_commit() -> str:
    run = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD") + ("+dirty" if run("status", "--porcelain", "src") else "")


def load_system(name: str):
    saved = np.load(SCOREBOARD_DIR / f"{name}.npz")
    config = json.loads((SCOREBOARD_DIR / f"{name}.json").read_text())["config"]
    return decisions(saved["probs"], saved["abnormal_prob"], config)


def score(name: str):
    test = load_test_set()
    truth = ground_truth(test)
    classes = headline_classes(truth)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    paths = [str(VINDR_DIR / "images" / f"{i}.png") for i in test["image_id"]]
    print(f"Predicting {len(paths)} held-out VinDr images with '{name}'...", flush=True)
    probs, abnormal_prob, config = SYSTEMS[name](paths, device)

    SCOREBOARD_DIR.mkdir(exist_ok=True)
    np.savez_compressed(SCOREBOARD_DIR / f"{name}.npz", image_id=test["image_id"].to_numpy(dtype=str),
                        probs=probs, abnormal_prob=abnormal_prob)

    pred = decisions(probs, abnormal_prob, config)
    point = metrics(truth, pred, classes, np.arange(len(test)))
    print(f"Bootstrapping {N_BOOT} resamples...", flush=True)
    samples = bootstrap(lambda idx: metrics(truth, pred, classes, idx), len(test))

    result = {
        "system": name,
        "date": date.today().isoformat(),
        "commit": git_commit(),
        "n_images": len(test),
        "n_normal": int((~truth["abnormal"]).sum()),
        "n_trackable": int(truth["trackable"].sum()),
        "headline_classes": classes,
        "positives": {c: int(truth[f"pos:{c}"].sum()) for c in VINDR_TO_OURS},
        "config": config,
        "metrics": with_intervals(point, samples),
    }
    (SCOREBOARD_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))
    print_result(result)
    write_board()


def compare(base: str, other: str):
    test = load_test_set()
    truth = ground_truth(test)
    classes = headline_classes(truth)
    a, b = load_system(base), load_system(other)
    diff = lambda idx: {k: metrics(truth, b, classes, idx)[k] - metrics(truth, a, classes, idx)[k] for k in HEADLINE}
    point = diff(np.arange(len(test)))
    samples = bootstrap(diff, len(test))  # paired: both systems see the same resample

    print(f"\n{other} minus {base} (95% paired bootstrap CI)")
    for k in HEADLINE:
        lo, hi = np.nanpercentile([s[k] for s in samples], [2.5, 97.5])
        better = point[k] < 0 if k in LOWER_IS_BETTER else point[k] > 0
        verdict = "no clear difference" if lo <= 0 <= hi else ("better" if better else "worse")
        print(f"  {k:24s} {point[k]:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")


# --- output -----------------------------------------------------------------

def fmt(metric: dict, pct: bool) -> str:
    v, (lo, hi) = metric["value"], metric["ci95"]
    if np.isnan(v):
        return "n/a"
    return f"{v:.1%} ({lo:.1%}–{hi:.1%})" if pct else f"{v:.3f} ({lo:.3f}–{hi:.3f})"


def print_result(r: dict):
    m = r["metrics"]
    print(f"\n{r['system']}  ({r['n_images']} images: {r['n_normal']} healthy, {r['n_trackable']} with a tracked finding)")
    print(f"  macro AUROC ({len(r['headline_classes'])} conditions)  {fmt(m['macro_auroc'], False)}")
    print(f"  abnormal AUROC                  {fmt(m['abnormal_auroc'], False)}")
    print(f"  healthy false alarm (lower)     {fmt(m['healthy_false_alarm'], True)}")
    print(f"  trackable sensitivity (higher)  {fmt(m['trackable_sensitivity'], True)}")
    print("\n  condition            n+   AUROC                  sensitivity            healthy flag rate")
    for c, n in r["positives"].items():
        print(f"  {c:20s} {n:4d}  {fmt(m[f'auroc:{c}'], False):22s} "
              f"{fmt(m[f'sensitivity:{c}'], True):22s} {fmt(m[f'healthy_flag_rate:{c}'], True)}")


def write_board():
    results = [json.loads(p.read_text()) for p in sorted(SCOREBOARD_DIR.glob("*.json"))]
    lines = [
        "# VinDr scoreboard",
        "",
        "Generated by `src/scoreboard.py` - do not edit by hand. Held-out VinDr-CXR",
        "test split (2,250 images, radiologist labels). 95% bootstrap intervals in",
        "parentheses; use `scoreboard.py compare` to test whether a difference is real.",
        "",
        "| System | Date | Commit | Macro AUROC | Abnormal AUROC | Healthy false alarm ↓ | Trackable sensitivity ↑ |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r["metrics"]
        lines.append(f"| {r['system']} | {r['date']} | {r['commit']} | {fmt(m['macro_auroc'], False)} | "
                     f"{fmt(m['abnormal_auroc'], False)} | {fmt(m['healthy_false_alarm'], True)} | "
                     f"{fmt(m['trackable_sensitivity'], True)} |")

    classes = list(VINDR_TO_OURS)
    lines += ["", "## Per-condition AUROC", "",
              "Macro AUROC averages the conditions marked *; the rest have too few test positives to rank on.", "",
              "| System | " + " | ".join(classes) + " |",
              "|---|" + "---|" * len(classes)]
    if results:
        r0 = results[0]
        lines.append("| *n positives* | " + " | ".join(
            f"{r0['positives'][c]}{'*' if c in r0['headline_classes'] else ''}" for c in classes) + " |")
    for r in results:
        lines.append(f"| {r['system']} | " + " | ".join(
            f"{r['metrics'][f'auroc:{c}']['value']:.3f}" for c in classes) + " |")

    (SCOREBOARD_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("score").add_argument("system", choices=SYSTEMS)
    cmp = sub.add_parser("compare")
    cmp.add_argument("base")
    cmp.add_argument("other")
    args = parser.parse_args()
    score(args.system) if args.command == "score" else compare(args.base, args.other)


if __name__ == "__main__":
    main()
