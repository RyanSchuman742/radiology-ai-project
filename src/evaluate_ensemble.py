"""Does ensembling our ResNet50 with TorchXRayVision's multi-dataset DenseNet
actually help? Measured before wiring anything into the app.

Two evaluations, because the first one is contaminated:

1. NIH-14 held-out test split - per-class AUROC and healthy-image false
   positive rate for ours, TXV alone, and the ensemble. CAVEAT: TXV's
   "all" weights were trained on NIH ChestX-ray14 among other datasets
   (the weights file is literally named nih-pc-chex-mimic_ch-...), so TXV
   has likely seen some of these "held-out" test images. Its numbers here
   are optimistic.

2. External healthy set - Kermany pediatric NORMAL X-rays, which neither
   model trained on. Only measures the false-positive rate on healthy scans
   (no 14-condition labels exist there), but it's the clean, uncontaminated
   check, and the closest match to the original complaint: healthy X-rays
   from outside the training data getting flagged.

Ensemble thresholds are calibrated on the NIH validation split with the same
conservative search as calibrate_thresholds.py.
"""

import json

import numpy as np
import pandas as pd
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from calibrate_thresholds import find_best_threshold
from train_multilabel import (
    CONDITIONS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_DIR, MODEL_PATH,
    PROJECT_ROOT, labels_to_multihot, load_manifest, patient_level_split,
)

REPORT_PATH = MODEL_DIR / "nih14_ensemble_report.json"
KERMANY_DIR = PROJECT_ROOT / "data" / "pneumonia_dataset"
TXV_WEIGHTS = "densenet121-res224-all"

OUR_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
TXV_TRANSFORM = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224),
])


class DualInputDataset(Dataset):
    """Yields each image preprocessed two ways - ours (RGB, ImageNet
    normalization) and TXV's (grayscale, [-1024, 1024], center crop)."""

    def __init__(self, paths: list[str], targets: list[torch.Tensor]):
        self.paths = paths
        self.targets = targets

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx])
        ours = OUR_TRANSFORM(image.convert("RGB"))

        gray = np.array(image.convert("L")).astype(np.float32)
        gray = xrv.utils.normalize(gray, 255)[None, ...]
        txv = torch.from_numpy(TXV_TRANSFORM(gray)).float()

        return ours, txv, self.targets[idx]


def load_models(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    ours = models.resnet50(weights=None)
    ours.fc = nn.Linear(ours.fc.in_features, len(CONDITIONS))
    ours.load_state_dict(checkpoint["model_state"])
    ours.to(device).eval()

    txv = xrv.models.DenseNet(weights=TXV_WEIGHTS).to(device).eval()
    txv_indices = [txv.pathologies.index(c) for c in CONDITIONS]
    return ours, txv, txv_indices, checkpoint


def predict(ours, txv, txv_indices, loader, device):
    all_ours, all_txv, all_targets = [], [], []
    with torch.no_grad():
        for ours_in, txv_in, targets in loader:
            all_ours.append(torch.sigmoid(ours(ours_in.to(device))).cpu().numpy())
            all_txv.append(txv(txv_in.to(device))[:, txv_indices].cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_ours), np.concatenate(all_txv), np.concatenate(all_targets)


def per_class_auroc(targets, probs):
    return {
        cond: roc_auc_score(targets[:, i], probs[:, i]) if len(set(targets[:, i])) > 1 else None
        for i, cond in enumerate(CONDITIONS)
    }


def fpr_on_healthy(probs, thresholds: dict, healthy_mask):
    thresh = np.array([thresholds[c] for c in CONDITIONS])[None, :]
    flagged = (probs >= thresh).sum(axis=1) > 0
    return float(flagged[healthy_mask].mean())


def per_class_recall(targets, probs, thresholds: dict):
    thresh = np.array([thresholds[c] for c in CONDITIONS])[None, :]
    preds = probs >= thresh
    return {
        cond: float(preds[targets[:, i] == 1, i].mean()) if targets[:, i].sum() > 0 else None
        for i, cond in enumerate(CONDITIONS)
    }


def matched_recall_threshold(targets_col, probs_col, target_recall) -> float:
    """Highest threshold that still catches at least `target_recall` of the
    positive cases - i.e. match the sensitivity we already ship, and take
    whatever false-positive reduction the ensemble gives at that level."""
    positives = np.sort(probs_col[targets_col == 1])[::-1]
    if len(positives) == 0 or target_recall is None:
        return 0.5
    k = min(max(1, int(np.ceil(target_recall * len(positives)))), len(positives))
    return float(positives[k - 1])


def nih_loader(df):
    paths = df["path"].tolist()
    targets = [labels_to_multihot(l) for l in df["labels"]]
    return DataLoader(DualInputDataset(paths, targets), batch_size=32, shuffle=False, num_workers=2)


def kermany_normal_loader():
    rows = []
    for split in ["train", "validation", "test"]:
        labels = pd.read_csv(KERMANY_DIR / split / "labels.csv")
        normals = labels[labels["label"] == "NORMAL"]
        rows += [str(KERMANY_DIR / split / f) for f in normals["filename"]]
    empty = torch.zeros(len(CONDITIONS))
    return DataLoader(DualInputDataset(rows, [empty] * len(rows)), batch_size=32, shuffle=False, num_workers=2)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ours, txv, txv_indices, checkpoint = load_models(device)
    our_thresholds = checkpoint["decision_thresholds"]
    txv_thresholds = {c: 0.5 for c in CONDITIONS}  # TXV outputs are already op-normalized: 0.5 = its own cutoff

    df = load_manifest()
    _, val_df, test_df = patient_level_split(df)

    print("Predicting on NIH validation split...", flush=True)
    val_ours, val_txv, val_targets = predict(ours, txv, txv_indices, nih_loader(val_df), device)
    val_ens = (val_ours + val_txv) / 2

    ensemble_thresholds = {
        cond: find_best_threshold(val_targets[:, i], val_ens[:, i])[0]
        for i, cond in enumerate(CONDITIONS)
    }
    our_val_recall = per_class_recall(val_targets, val_ours, our_thresholds)
    matched_thresholds = {
        cond: matched_recall_threshold(val_targets[:, i], val_ens[:, i], our_val_recall[cond])
        for i, cond in enumerate(CONDITIONS)
    }

    print("Predicting on NIH test split...", flush=True)
    test_ours, test_txv, test_targets = predict(ours, txv, txv_indices, nih_loader(test_df), device)
    test_ens = (test_ours + test_txv) / 2
    nih_healthy = test_targets.sum(axis=1) == 0

    auroc = {
        "ours": per_class_auroc(test_targets, test_ours),
        "txv": per_class_auroc(test_targets, test_txv),
        "ensemble": per_class_auroc(test_targets, test_ens),
    }
    macro = {k: float(np.mean([v for v in a.values() if v is not None])) for k, a in auroc.items()}

    recall = {
        "ours": per_class_recall(test_targets, test_ours, our_thresholds),
        "ensemble": per_class_recall(test_targets, test_ens, ensemble_thresholds),
        "ensemble_matched": per_class_recall(test_targets, test_ens, matched_thresholds),
    }

    nih_fpr = {
        "ours": fpr_on_healthy(test_ours, our_thresholds, nih_healthy),
        "txv": fpr_on_healthy(test_txv, txv_thresholds, nih_healthy),
        "ensemble": fpr_on_healthy(test_ens, ensemble_thresholds, nih_healthy),
        "ensemble_matched": fpr_on_healthy(test_ens, matched_thresholds, nih_healthy),
    }

    print("Predicting on external Kermany NORMAL set...", flush=True)
    k_ours, k_txv, _ = predict(ours, txv, txv_indices, kermany_normal_loader(), device)
    k_ens = (k_ours + k_txv) / 2
    all_healthy = np.ones(len(k_ours), dtype=bool)
    kermany_fpr = {
        "ours": fpr_on_healthy(k_ours, our_thresholds, all_healthy),
        "txv": fpr_on_healthy(k_txv, txv_thresholds, all_healthy),
        "ensemble": fpr_on_healthy(k_ens, ensemble_thresholds, all_healthy),
        "ensemble_matched": fpr_on_healthy(k_ens, matched_thresholds, all_healthy),
    }

    print("\n=== NIH-14 test split (TXV numbers optimistic - it likely trained on some of these) ===")
    print(f"Macro AUROC:  ours={macro['ours']:.4f}  txv={macro['txv']:.4f}  ensemble={macro['ensemble']:.4f}")
    print("Per-class AUROC (ours / txv / ensemble):")
    for c in CONDITIONS:
        print(f"  {c}: {auroc['ours'][c]:.3f} / {auroc['txv'][c]:.3f} / {auroc['ensemble'][c]:.3f}")
    print("\nPer-class recall (ours / ensemble / ensemble_matched):")
    for c in CONDITIONS:
        print(f"  {c}: {recall['ours'][c]:.3f} / {recall['ensemble'][c]:.3f} / {recall['ensemble_matched'][c]:.3f}")
    print(f"\nHealthy NIH test images with >=1 false positive ({int(nih_healthy.sum())} images):")
    for k, v in nih_fpr.items():
        print(f"  {k}: {v:.1%}")

    print(f"\n=== External: Kermany pediatric NORMAL ({len(k_ours)} images, neither model trained on these) ===")
    print("Healthy images with >=1 false positive:")
    for k, v in kermany_fpr.items():
        print(f"  {k}: {v:.1%}")

    print("\nEnsemble thresholds, conservative / matched-recall (both calibrated on NIH validation):")
    for c in CONDITIONS:
        print(f"  {c}: {ensemble_thresholds[c]:.3f} / {matched_thresholds[c]:.3f}")

    REPORT_PATH.write_text(json.dumps({
        "caveat": "TXV 'all' weights were trained on NIH ChestX-ray14 among other datasets; NIH test numbers for txv/ensemble are optimistic. Kermany NORMAL is the uncontaminated check.",
        "nih_test_macro_auroc": macro,
        "nih_test_per_class_auroc": auroc,
        "nih_test_per_class_recall": recall,
        "nih_test_healthy_false_positive_rate": nih_fpr,
        "kermany_normal_false_positive_rate": kermany_fpr,
        "ensemble_thresholds": ensemble_thresholds,
        "ensemble_matched_recall_thresholds": matched_thresholds,
    }, indent=2))
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
