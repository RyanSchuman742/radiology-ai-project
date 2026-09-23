"""Convert the VinBigData (VinDr-CXR, Kaggle release) training DICOMs to PNG
and build image-level labels from the radiologist annotations.

Reads each DICOM straight out of the downloaded competition zip, so the
~192 GB of raw DICOMs never has to be unpacked to disk.

DICOM pitfalls handled here rather than trusted to a third-party
conversion (the reason we used the official files at all):
- VOI LUT: each file stores the display window radiologists read it with.
  Raw pixel values without it look washed out or crushed.
- MONOCHROME1: some X-rays are stored as negatives (bone dark). Flipped so
  every output image has bone white, like NIH-14.

Labels: each of the 15,000 training images was read independently by 3
radiologists. train.csv has one row per (image, radiologist, finding box),
with "No finding" rows for radiologists who saw nothing. An image is
abnormal if a majority of its radiologists marked any finding. In practice
every image in this release is unanimous on normal vs. abnormal (10,606
all-"No finding", 4,394 all-abnormal - checked against the raw
train.csv), so `unanimous` is always True here; radiologists still
disagree on *which* findings, which the per-class vote counts capture.

The Kaggle test set (3,000 images) has no public labels and isn't used.
"""

import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixels import apply_voi_lut

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VINDR_DIR = PROJECT_ROOT / "data" / "vindr"
ZIP_PATH = VINDR_DIR / "vinbigdata-chest-xray-abnormalities-detection.zip"
IMAGES_DIR = VINDR_DIR / "images"
LABELS_PATH = VINDR_DIR / "labels.csv"
LONG_SIDE = 1024  # 4x our 224px training size, 2x the planned 512px upgrade
NO_FINDING = "No finding"

_zip = None


def _open_zip():
    global _zip
    _zip = zipfile.ZipFile(ZIP_PATH)


def dicom_to_uint8(ds: pydicom.Dataset) -> np.ndarray:
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        arr = arr.max() - arr
    arr -= arr.min()
    arr /= max(float(arr.max()), 1e-6)
    return (arr * 255).astype(np.uint8)


def convert(member: str) -> tuple[str, str | None]:
    """Returns (image_id, PhotometricInterpretation), or (image_id, None) if
    already converted by an earlier run."""
    image_id = Path(member).stem
    out_path = IMAGES_DIR / f"{image_id}.png"
    if out_path.exists():
        return image_id, None

    with _zip.open(member) as f:
        ds = pydicom.dcmread(io.BytesIO(f.read()))
    image = Image.fromarray(dicom_to_uint8(ds))
    image.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
    image.save(out_path)
    return image_id, str(ds.PhotometricInterpretation)


def build_labels(annotations: pd.DataFrame) -> pd.DataFrame:
    per_rad_abnormal = (
        annotations.groupby(["image_id", "rad_id"])["class_name"]
        .apply(lambda names: (names != NO_FINDING).any())
    )
    labels = per_rad_abnormal.groupby("image_id").agg(n_rads="size", n_abnormal_votes="sum")
    labels["abnormal"] = labels["n_abnormal_votes"] * 2 > labels["n_rads"]
    labels["unanimous"] = (labels["n_abnormal_votes"] == 0) | (labels["n_abnormal_votes"] == labels["n_rads"])

    findings = annotations[annotations["class_name"] != NO_FINDING].drop_duplicates(
        ["image_id", "rad_id", "class_name"]
    )
    class_votes = findings.pivot_table(
        index="image_id", columns="class_name", values="rad_id", aggfunc="count", fill_value=0
    )
    labels = labels.join(class_votes, how="left").fillna(0)
    return labels.reset_index()


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = [m for m in zf.namelist() if m.startswith("train/") and m.endswith(".dicom")]
        annotations = pd.read_csv(zf.open("train.csv"))
    print(f"{len(members)} training DICOMs in zip, {annotations['image_id'].nunique()} images in train.csv", flush=True)

    photometric_counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=8, initializer=_open_zip) as pool:
        for i, (_, photometric) in enumerate(pool.map(convert, members, chunksize=16), 1):
            if photometric:
                photometric_counts[photometric] = photometric_counts.get(photometric, 0) + 1
            if i % 1000 == 0:
                print(f"  converted {i}/{len(members)}", flush=True)

    labels = build_labels(annotations)
    converted = {p.stem for p in IMAGES_DIR.glob("*.png")}
    labels = labels[labels["image_id"].isin(converted)]
    labels.to_csv(LABELS_PATH, index=False)

    print(f"\nPhotometric interpretation (this run): {photometric_counts}")
    print(f"Labeled images: {len(labels)}")
    print(f"Abnormal (majority of radiologists): {labels['abnormal'].mean():.1%}")
    print(f"All radiologists agreed: {labels['unanimous'].mean():.1%}")
    print(f"Radiologists per image: {labels['n_rads'].value_counts().to_dict()}")
    print(f"Saved labels to {LABELS_PATH}")


if __name__ == "__main__":
    main()
