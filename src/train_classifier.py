"""Step 2b: fine-tune a ResNet on the pneumonia dataset saved by
download_pneumonia_dataset.py.

Structure is deliberately generic (reads class names from labels.csv rather
than hardcoding "Normal"/"Pneumonia") so this can later be pointed at a
multi-class dataset (e.g. NIH ChestX-ray14 conditions) without a rewrite.
"""

import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "pneumonia_dataset"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "resnet_pneumonia.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
VAL_FRACTION = 0.1  # carved out of the official "train" split; official "val" is only 16 images
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class XrayDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def read_split(split_name: str) -> tuple[list[tuple[Path, int]], list[str]]:
    split_dir = DATA_ROOT / split_name
    labels_csv = split_dir / "labels.csv"

    with open(labels_csv) as f:
        rows = list(csv.DictReader(f))

    class_names = sorted({row["label"] for row in rows})
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = [(split_dir / row["filename"], class_to_idx[row["label"]]) for row in rows]
    return samples, class_names


def build_datasets():
    train_samples, class_names = read_split("train")
    val_official_samples, _ = read_split("validation")
    test_samples, _ = read_split("test")

    random.Random(SEED).shuffle(train_samples)
    n_val = int(len(train_samples) * VAL_FRACTION)
    val_samples = train_samples[:n_val] + val_official_samples
    train_samples = train_samples[n_val:]

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

    return (
        XrayDataset(train_samples, train_transform),
        XrayDataset(val_samples, eval_transform),
        XrayDataset(test_samples, eval_transform),
        class_names,
    )


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds, class_names = build_datasets()
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "class_names": class_names}, MODEL_PATH)
            print(f"  Saved new best model (val_acc={val_acc:.4f}) to {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nFinal test set: loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
