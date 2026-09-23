"""Normal-vs-abnormal gate: stage one of a two-stage system.

The deployed app runs 14 independent per-condition tests, and a healthy scan
counts as a false alarm if *any* of them fires. Even at a 5% false-alarm
rate per condition, the chance that at least one of 14 fires is ~51% - so
per-condition tuning alone can't get the healthy false-alarm rate low. A
single model answering only "is anything abnormal here?" (trained on NIH's
"No Finding" vs. everything else) can veto the 14-condition step when it's
confident the scan is normal. Same approach as commercial triage tools.

Deliberately reuses train_multilabel.py's recipe (ImageNet ResNet50, 224px,
same augmentation, same patient-level split) so the only new variable is
the task itself. Resumable from the last completed epoch, like the
14-condition trainer.
"""

import json

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from train_multilabel import (
    BATCH_SIZE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LEARNING_RATE, MODEL_DIR,
    load_manifest, patient_level_split,
)

GATE_MODEL_PATH = MODEL_DIR / "resnet_gate.pt"
GATE_LATEST_PATH = MODEL_DIR / "resnet_gate_latest.pt"  # gitignored (*_latest.pt)
GATE_REPORT_PATH = MODEL_DIR / "gate_eval_report.json"
NUM_EPOCHS = 12  # the 14-condition model peaked at epoch 11 of 20; this task is easier


def is_abnormal(labels_str: str) -> float:
    return 0.0 if labels_str == "No Finding" else 1.0


class GateDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[float], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), torch.tensor([self.labels[idx]], dtype=torch.float32)


def make_loader(df, transform, shuffle: bool, workers: int) -> DataLoader:
    labels = [is_abnormal(l) for l in df["labels"]]
    ds = GateDataset(df["path"].tolist(), labels, transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=workers)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total = 0.0, 0
    all_targets, all_probs = [], []

    with torch.set_grad_enabled(train):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            all_targets.append(targets.cpu())
            all_probs.append(torch.sigmoid(logits).detach().cpu())

    targets = torch.cat(all_targets).numpy().ravel()
    probs = torch.cat(all_probs).numpy().ravel()
    return total_loss / total, roc_auc_score(targets, probs), targets, probs


def specificity_at_recall(targets, probs, target_recall: float) -> tuple[float, float]:
    """(threshold, specificity) at the highest threshold still catching
    target_recall of abnormal scans."""
    positives = np.sort(probs[targets == 1])[::-1]
    k = min(max(1, int(np.ceil(target_recall * len(positives)))), len(positives))
    threshold = float(positives[k - 1])
    specificity = float((probs[targets == 0] < threshold).mean())
    return threshold, specificity


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    train_df, val_df, test_df = patient_level_split(load_manifest())
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        abnormal = sum(is_abnormal(l) for l in df["labels"])
        print(f"{name}: {len(df)} images, {abnormal / len(df):.1%} abnormal", flush=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_loader = make_loader(train_df, train_transform, shuffle=True, workers=4)
    val_loader = make_loader(val_df, eval_transform, shuffle=False, workers=2)
    test_loader = make_loader(test_df, eval_transform, shuffle=False, workers=2)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    # ~54% normal / 46% abnormal - balanced enough that no pos_weight is needed.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_auroc, start_epoch = 0.0, 1
    if GATE_LATEST_PATH.exists():
        ckpt = torch.load(GATE_LATEST_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_auroc = ckpt["best_val_auroc"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} (best_val_auroc so far: {best_val_auroc:.4f})", flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss, train_auroc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_auroc, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss={train_loss:.4f} train_auroc={train_auroc:.4f} | "
              f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f}", flush=True)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({"model_state": model.state_dict()}, GATE_MODEL_PATH)
            print(f"  Saved new best gate (val_auroc={val_auroc:.4f})", flush=True)

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_auroc": best_val_auroc,
        }, GATE_LATEST_PATH)

    model.load_state_dict(torch.load(GATE_MODEL_PATH, map_location=device)["model_state"])
    _, test_auroc, test_targets, test_probs = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

    operating_points = {}
    print(f"\nTest AUROC: {test_auroc:.4f}")
    print("Specificity (share of normal scans correctly passed as normal) at fixed sensitivity:")
    for recall in [0.90, 0.95, 0.975]:
        threshold, spec = specificity_at_recall(test_targets, test_probs, recall)
        operating_points[str(recall)] = {"threshold": threshold, "specificity": spec}
        print(f"  catch {recall:.1%} of abnormal -> {spec:.1%} of normals cleared (threshold {threshold:.3f})")

    GATE_REPORT_PATH.write_text(json.dumps({
        "test_auroc": test_auroc,
        "best_val_auroc": best_val_auroc,
        "test_operating_points": operating_points,
    }, indent=2))
    print(f"\nSaved report to {GATE_REPORT_PATH}")


if __name__ == "__main__":
    main()
