"""Download a subset of the NIH ChestX-ray14 dataset for Phase 2 prototyping.

Pulls the first N zip archives (not the full 12) from the official NIH
release mirrored on Hugging Face, extracts them, and builds a manifest CSV
joining each extracted image to its multi-label findings from the official
Data_Entry_2017_v2020.csv metadata.

This is a prototyping subset, not the final training set - class balance
across the 14 conditions is whatever naturally falls out of these zips
(NIH's official chunking is roughly patient-ID order, not randomized).
"""

import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "alkzar90/NIH-Chest-X-ray-dataset"
ZIP_FILES = ["data/images/images_001.zip", "data/images/images_002.zip"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "nih14_subset"
IMAGES_DIR = OUTPUT_DIR / "images"


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading label metadata...")
    labels_csv_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename="data/Data_Entry_2017_v2020.csv"
    )
    labels_df = pd.read_csv(labels_csv_path).set_index("Image Index")

    extracted_filenames = []
    for zip_filename in ZIP_FILES:
        print(f"Downloading {zip_filename} ...")
        zip_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=zip_filename)

        print(f"Extracting {zip_filename} ...")
        with zipfile.ZipFile(zip_path) as zf:
            # Skip macOS AppleDouble resource-fork artifacts (._*.png) bundled
            # into the zip - not real images, just metadata sidecar files.
            members = [
                m for m in zf.namelist()
                if m.lower().endswith(".png") and not Path(m).name.startswith("._")
            ]
            for member in members:
                target_name = Path(member).name
                target_path = IMAGES_DIR / target_name
                if not target_path.exists():
                    with zf.open(member) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())
                extracted_filenames.append(target_name)
        print(f"  {len(members)} images extracted from {zip_filename}")

    print(f"\nTotal extracted images: {len(extracted_filenames)}")

    manifest_rows = []
    missing_labels = 0
    for filename in extracted_filenames:
        if filename not in labels_df.index:
            missing_labels += 1
            continue
        row = labels_df.loc[filename]
        manifest_rows.append({
            "filename": filename,
            "patient_id": row["Patient ID"],
            "labels": row["Finding Labels"],
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUT_DIR / "labels.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"Manifest saved to {manifest_path} ({len(manifest_df)} labeled images, {missing_labels} missing labels)")

    from collections import Counter
    condition_counts = Counter()
    for labels in manifest_df["labels"]:
        for cond in labels.split("|"):
            condition_counts[cond] += 1

    print("\nPer-condition counts in this subset:")
    all_conditions = [
        "No Finding", "Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass",
        "Pneumothorax", "Consolidation", "Pleural_Thickening", "Cardiomegaly",
        "Emphysema", "Edema", "Fibrosis", "Pneumonia", "Hernia",
    ]
    for cond in all_conditions:
        count = condition_counts.get(cond, 0)
        flag = "  <-- MISSING" if count == 0 else ""
        print(f"  {cond}: {count}{flag}")


if __name__ == "__main__":
    main()
