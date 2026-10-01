"""Retrain the normal/abnormal gate on VinDr-CXR's radiologist labels.

The NIH-trained gate (train_gate.py) scored AUROC 0.772 against NIH's
report-mined labels but 0.878 against VinDr's radiologist labels - NIH label
noise was limiting both its training and how well we could measure it.

Two variants, chosen between on VinDr validation:
- finetune: the NIH gate, fine-tuned on VinDr at a lower learning rate
  (big noisy dataset first, small clean one second)
- imagenet: ImageNet ResNet50 trained on VinDr only, for comparison

VinDr is split 70/15/15 by image, stratified by normal/abnormal, and the split
is saved so the evaluation uses exactly the same held-out test images. The
Kaggle release has no patient IDs, so the split can't be patient-level; if a
patient has multiple images in the release, they could land on both sides.

Same 224px input and preprocessing as the deployed gate, so a winner can drop
into the app without changes.
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import models, transforms

from train_gate import GATE_MODEL_PATH, GateDataset, run_epoch
from train_multilabel import BATCH_SIZE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_DIR, PROJECT_ROOT

VINDR_DIR = PROJECT_ROOT / "data" / "vindr"
SPLIT_PATH = VINDR_DIR / "split.csv"
REPORT_PATH = MODEL_DIR / "gate_vindr_train_report.json"
SEED = 42

VARIANTS = {
    # name: (initial weights, learning rate, epochs)
    "finetune": ("nih_gate", 3e-5, 8),
    "imagenet": ("imagenet", 1e-4, 12),
}


def variant_path(name: str):
    return MODEL_DIR / f"resnet_gate_vindr_{name}.pt"


def make_split() -> pd.DataFrame:
    labels = pd.read_csv(VINDR_DIR / "labels.csv")
    train, rest = train_test_split(labels, test_size=0.30, stratify=labels["abnormal"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["abnormal"], random_state=SEED)
    split = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    split[["image_id", "abnormal", "split"]].to_csv(SPLIT_PATH, index=False)
    return split


def loader(df, transform, shuffle, workers):
    paths = [str(VINDR_DIR / "images" / f"{i}.png") for i in df["image_id"]]
    ds = GateDataset(paths, df["abnormal"].astype(float).tolist(), transform)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=workers)


def build_gate(init: str, device):
    if init == "imagenet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(GATE_MODEL_PATH, map_location=device)["model_state"])
    return model.to(device)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = make_split()
    for s in ["train", "val", "test"]:
        part = split[split["split"] == s]
        print(f"{s}: {len(part)} images, {part['abnormal'].mean():.1%} abnormal", flush=True)

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_loader = loader(split[split["split"] == "train"], train_tf, True, 4)
    val_loader = loader(split[split["split"] == "val"], eval_tf, False, 2)
    test_loader = loader(split[split["split"] == "test"], eval_tf, False, 2)
    criterion = nn.BCEWithLogitsLoss()

    report = {}

    baseline = build_gate("nih_gate", device)
    _, base_val, _, _ = run_epoch(baseline, val_loader, criterion, None, device, train=False)
    _, base_test, _, _ = run_epoch(baseline, test_loader, criterion, None, device, train=False)
    report["current_nih_gate"] = {"val_auroc": base_val, "test_auroc": base_test}
    print(f"\nCurrent (NIH-trained) gate on VinDr: val AUROC {base_val:.4f}, test AUROC {base_test:.4f}", flush=True)

    for name, (init, lr, epochs) in VARIANTS.items():
        print(f"\n=== {name} (init={init}, lr={lr}, {epochs} epochs) ===", flush=True)
        model = build_gate(init, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val, best_epoch = 0.0, 0
        for epoch in range(1, epochs + 1):
            train_loss, train_auc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            _, val_auc, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} train_auroc={train_auc:.4f} val_auroc={val_auc:.4f}", flush=True)
            if val_auc > best_val:
                best_val, best_epoch = val_auc, epoch
                torch.save({"model_state": model.state_dict()}, variant_path(name))

        model.load_state_dict(torch.load(variant_path(name), map_location=device)["model_state"])
        _, test_auc, _, _ = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        report[name] = {"best_epoch": best_epoch, "val_auroc": best_val, "test_auroc": test_auc}
        print(f"  best epoch {best_epoch}: val AUROC {best_val:.4f}, test AUROC {test_auc:.4f}", flush=True)

    winner = max(VARIANTS, key=lambda n: report[n]["val_auroc"])
    report["winner_by_val"] = winner
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWinner (by validation AUROC): {winner}")
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
