"""Phase 2: multi-label ResNet50 fine-tuning on the NIH ChestX-ray14 subset.

Differs from train_classifier.py (Phase 1, binary pneumonia/normal) in the
ways that actually matter for multi-label: BCEWithLogitsLoss instead of
CrossEntropyLoss, independent per-class sigmoid instead of softmax, a
multi-hot target vector instead of a single class index, and a patient-level
train/val/test split (NIH-14 has multiple images per patient - splitting by
image instead of patient leaks the same patient's anatomy across splits and
inflates validation/test metrics).

"No Finding" is not one of the 14 target classes - a scan with all 14
findings absent is implicitly "no finding", matching the convention used in
the original ChestX-ray14 paper and most published benchmarks on it.
"""

import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nih14_subset"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "resnet_nih14.pt"
LATEST_CHECKPOINT_PATH = MODEL_DIR / "resnet_nih14_latest.pt"  # every epoch, for resuming a crashed/interrupted run
REPORT_PATH = MODEL_DIR / "nih14_eval_report.json"

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
SEED = 42
DECISION_THRESHOLD = 0.5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NIH14Dataset(Dataset):
    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), row["target"]


def labels_to_multihot(labels_str: str) -> torch.Tensor:
    present = set(labels_str.split("|"))
    return torch.tensor([1.0 if c in present else 0.0 for c in CONDITIONS])


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "labels.csv")
    df["path"] = df["filename"].apply(lambda f: str(DATA_DIR / "images" / f))
    return df


def patient_level_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = df["patient_id"].unique().tolist()
    random.Random(SEED).shuffle(patients)

    n_val = int(len(patients) * VAL_FRACTION)
    n_test = int(len(patients) * TEST_FRACTION)
    val_patients = set(patients[:n_val])
    test_patients = set(patients[n_val:n_val + n_test])
    train_patients = set(patients[n_val + n_test:])

    train_df = df[df["patient_id"].isin(train_patients)]
    val_df = df[df["patient_id"].isin(val_patients)]
    test_df = df[df["patient_id"].isin(test_patients)]
    return train_df, val_df, test_df


def build_rows(df: pd.DataFrame) -> list[dict]:
    return [
        {"path": row["path"], "target": labels_to_multihot(row["labels"])}
        for _, row in df.iterrows()
    ]


def build_model() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(CONDITIONS))
    return model


def compute_pos_weight(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weight per class for BCEWithLogitsLoss's pos_weight,
    same idea as Phase 1's class weighting but per-class instead of binary."""
    pos_counts = torch.zeros(len(CONDITIONS))
    for labels_str in train_df["labels"]:
        present = set(labels_str.split("|"))
        for i, cond in enumerate(CONDITIONS):
            if cond in present:
                pos_counts[i] += 1

    total = len(train_df)
    neg_counts = total - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).to(device)
    return pos_weight


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

    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    per_class_auroc = []
    for i in range(len(CONDITIONS)):
        if len(set(all_targets[:, i])) < 2:
            continue  # AUROC undefined with only one class present in this split
        per_class_auroc.append(roc_auc_score(all_targets[:, i], all_probs[:, i]))
    macro_auroc = sum(per_class_auroc) / len(per_class_auroc) if per_class_auroc else 0.0

    return total_loss / total, macro_auroc, all_targets, all_probs


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_manifest()
    train_df, val_df, test_df = patient_level_split(df)
    print(f"Patients -> train: {train_df['patient_id'].nunique()}  "
          f"val: {val_df['patient_id'].nunique()}  test: {test_df['patient_id'].nunique()}")
    print(f"Images -> train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

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

    train_ds = NIH14Dataset(build_rows(train_df), train_transform)
    val_ds = NIH14Dataset(build_rows(val_df), eval_transform)
    test_ds = NIH14Dataset(build_rows(test_df), eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model().to(device)
    pos_weight = compute_pos_weight(train_df, device)
    print(f"Pos weights: {dict(zip(CONDITIONS, pos_weight.tolist()))}", flush=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_auroc = 0.0
    start_epoch = 1
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from the last completed epoch if this run was interrupted
    # (crash, machine slept, etc.) - important for an unattended overnight
    # run where nobody's there to notice and manually restart it.
    if LATEST_CHECKPOINT_PATH.exists():
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_val_auroc = checkpoint["best_val_auroc"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} (best_val_auroc so far: {best_val_auroc:.4f})", flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss, train_auroc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_auroc, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_macro_auroc={train_auroc:.4f} | "
              f"val_loss={val_loss:.4f} val_macro_auroc={val_auroc:.4f}", flush=True)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": CONDITIONS,
                "decision_threshold": DECISION_THRESHOLD,
            }, MODEL_PATH)
            print(f"  Saved new best model (val_macro_auroc={val_auroc:.4f}) to {MODEL_PATH}", flush=True)

        # Every-epoch checkpoint for resuming, separate from the "best" model
        # above (which is what the app actually loads for inference).
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_auroc": best_val_auroc,
        }, LATEST_CHECKPOINT_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_auroc, test_targets, test_probs = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )

    test_preds = (test_probs >= DECISION_THRESHOLD).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_targets, test_preds, average=None, zero_division=0
    )

    per_class_report = {}
    for i, cond in enumerate(CONDITIONS):
        auroc = roc_auc_score(test_targets[:, i], test_probs[:, i]) if len(set(test_targets[:, i])) > 1 else None
        per_class_report[cond] = {
            "auroc": auroc,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    report = {
        "test_set_size": len(test_df),
        "test_patients": test_df["patient_id"].nunique(),
        "macro_auroc": test_auroc,
        "decision_threshold": DECISION_THRESHOLD,
        "per_class": per_class_report,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\nFinal test set: loss={test_loss:.4f} macro_auroc={test_auroc:.4f}")
    print("\nPer-class test AUROC:")
    for cond in CONDITIONS:
        auroc = per_class_report[cond]["auroc"]
        support = per_class_report[cond]["support"]
        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A (no positive examples in test split)"
        print(f"  {cond}: {auroc_str}  (support={support})")

    print(f"\nFull report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
