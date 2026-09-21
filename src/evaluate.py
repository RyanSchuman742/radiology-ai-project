"""Evaluate the trained classifier on the held-out test set with metrics
that actually matter for a medical screening tool, not just accuracy.

Sensitivity (recall on the positive class) is the critical number here:
a missed pneumonia case (false negative) is far worse than a false alarm.
"""

import json
from pathlib import Path

import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn

from train_classifier import XrayDataset, read_split, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
from torchvision import transforms

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "resnet_pneumonia.pt"
REPORT_PATH = Path(__file__).resolve().parent.parent / "models" / "eval_report.json"

POSITIVE_CLASS = "PNEUMONIA"


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    positive_idx = class_names.index(POSITIVE_CLASS)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_samples, _ = read_split("test")
    test_ds = XrayDataset(test_samples, eval_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, positive_idx].cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names)))
    )
    auroc = roc_auc_score(
        [1 if l == positive_idx else 0 for l in all_labels], all_probs
    )

    tn, fp, fn, tp = cm[1 - positive_idx, 1 - positive_idx], cm[1 - positive_idx, positive_idx], \
        cm[positive_idx, 1 - positive_idx], cm[positive_idx, positive_idx]
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    report = {
        "test_set_size": len(all_labels),
        "accuracy": accuracy,
        "sensitivity_pneumonia_recall": sensitivity,
        "specificity_normal_recall": specificity,
        "auroc": auroc,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.tolist(),
        },
        "per_class": {
            class_names[i]: {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }
            for i in range(len(class_names))
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"Test set: {len(all_labels)} images")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"AUROC:       {auroc:.4f}")
    print(f"Sensitivity (recall on PNEUMONIA — catches true cases): {sensitivity:.4f}")
    print(f"Specificity (recall on NORMAL — avoids false alarms):   {specificity:.4f}")
    print(f"\nConfusion matrix ({class_names}):")
    print(cm)
    print(f"\nFalse negatives (missed pneumonia): {fn}")
    print(f"False positives (false alarms):     {fp}")
    print(f"\nFull report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
