"""Step 2a: download the Kermany Chest X-Ray Pneumonia dataset (Normal vs Pneumonia)
from Hugging Face (hf-vision/chest-xray-pneumonia, CC-BY-4.0) and save it to disk
as plain image files organized by split/label, plus a labels.csv per split.

Source: https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia
Original data: Kermany, D.; Zhang, K.; Goldbaum, M. (2018), "Labeled Optical
Coherence Tomography (OCT) and Chest X-Ray Images for Classification."
"""

import csv
from pathlib import Path

from datasets import load_dataset

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "pneumonia_dataset"


def save_split(dataset, split_name: str) -> None:
    split_dir = OUTPUT_ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    labels_path = split_dir / "labels.csv"
    label_names = dataset.features["label"].names

    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        for i, example in enumerate(dataset):
            label_name = label_names[example["label"]]
            filename = f"{split_name}_{i:05d}.png"

            image = example["image"].convert("L")
            image.save(split_dir / filename)

            writer.writerow([filename, label_name])

    print(f"Saved {len(dataset)} images to {split_dir} (labels: {labels_path})")


if __name__ == "__main__":
    print("Downloading hf-vision/chest-xray-pneumonia ...")
    ds = load_dataset("hf-vision/chest-xray-pneumonia")
    print(ds)

    for split in ds.keys():
        save_split(ds[split], split)

    print("Done.")
