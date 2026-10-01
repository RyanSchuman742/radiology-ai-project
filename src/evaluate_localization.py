"""Do the heatmaps point where radiologists marked the finding?

VinDr-CXR radiologists drew a bounding box around every finding they
marked, so for each held-out scoreboard image (VinDr test split) and each
finding a majority of them marked (>=2 of 3, one of the conditions we
track), this compares the live app's Grad-CAM for that condition against
the union of all radiologists' boxes for it.

Two measures per (image, finding):
- pointing: is the heatmap's single hottest point inside a box?
- heat in box: share of the heatmap as displayed (after the app's soft
  threshold) that falls inside the boxes.
Both are compared with chance - the fraction of the image the boxes cover,
which is what a heatmap with no idea where the finding is would score on
average. "Lift" = heat in box / box area; 1.0 means no better than chance.

Coordinates: the model sees the scan stretched to a 224px square and the
app stretches the heatmap back, so boxes (in original DICOM pixels) and
heatmaps are compared in the same normalized square grid.

Writes models/localization_report.json and an example figure.
"""

import json

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from evaluate_vindr import VINDR_TO_OURS
from gradcam_multilabel import CONDITION_COLORS, composite_multicolor_heatmap, load_models, normalize_heatmap, preprocess
from scoreboard import SCOREBOARD_DIR, VINDR_DIR, load_test_set
from train_multilabel import MODEL_DIR

GRID = 448
REPORT_PATH = MODEL_DIR / "localization_report.json"
FIGURE_PATH = MODEL_DIR / "localization_examples.png"


def finding_masks(test: pd.DataFrame) -> list[tuple[str, str, np.ndarray, list]]:
    """(image_id, VinDr class, union-of-boxes mask on GRID, normalized boxes)
    for every majority-agreed finding we track."""
    ann = pd.read_csv(VINDR_DIR / "annotations.csv")
    ann = ann[ann["class_name"].isin(VINDR_TO_OURS)].merge(
        pd.read_csv(VINDR_DIR / "original_sizes.csv"), on="image_id")
    ann = ann.set_index(["image_id", "class_name"]).sort_index()

    out = []
    for _, row in test.iterrows():
        for c in VINDR_TO_OURS:
            if row[c] < 2:
                continue
            boxes = ann.loc[[(row["image_id"], c)]]  # list key: always a DataFrame
            mask = np.zeros((GRID, GRID), dtype=bool)
            norm = []
            for b in boxes.itertuples():
                x0, x1 = b.x_min / b.orig_width, b.x_max / b.orig_width
                y0, y1 = b.y_min / b.orig_height, b.y_max / b.orig_height
                mask[int(y0 * GRID):int(np.ceil(y1 * GRID)), int(x0 * GRID):int(np.ceil(x1 * GRID))] = True
                norm.append((x0, y0, x1, y1))
            out.append((row["image_id"], c, mask, norm))
    return out


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _, _, _, class_names, config = load_models(str(device))
    cam_engine = GradCAM(model=model, target_layers=[model.layer4[-1]])

    test = load_test_set()
    saved = np.load(SCOREBOARD_DIR / "deployed.npz")  # ensemble probs, to know what the app showed
    probs = dict(zip(saved["image_id"], saved["probs"]))
    thresholds = np.array([config["thresholds"][c] for c in class_names])

    pairs = finding_masks(test)
    print(f"{len(pairs)} radiologist-marked findings on {len({p[0] for p in pairs})} held-out images", flush=True)

    rows, examples = [], []
    current_id, input_tensor = None, None
    for n, (image_id, vindr_class, mask, boxes) in enumerate(pairs):
        if image_id != current_id:
            image = Image.open(VINDR_DIR / "images" / f"{image_id}.png")
            input_tensor = preprocess(image)[0].to(device)
            current_id = image_id
        p = probs[image_id]
        # Nodule/Mass: use whichever of our two the ensemble rated higher
        condition = max(VINDR_TO_OURS[vindr_class], key=lambda c: p[class_names.index(c)])
        idx = class_names.index(condition)

        cam = cam_engine(input_tensor=input_tensor, targets=[ClassifierOutputTarget(idx)])[0]
        cam = cv2.resize(cam.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_LINEAR)
        shown = normalize_heatmap(cam)
        peak = np.unravel_index(cam.argmax(), cam.shape)

        rows.append({
            "image_id": image_id, "vindr_class": vindr_class, "condition": condition,
            "flagged": bool(p[idx] >= thresholds[idx]),
            "pointing_hit": bool(mask[peak]),
            "heat_in_box": float((shown * mask).sum() / max(shown.sum(), 1e-6)),
            "box_area": float(mask.mean()),
        })
        if rows[-1]["flagged"]:
            examples.append((image_id, condition, cam, boxes, rows[-1]))
        if n % 100 == 0:
            print(f"  {n}/{len(pairs)}", flush=True)

    df = pd.DataFrame(rows)
    report = {"n_findings": len(df), "n_images": int(df["image_id"].nunique()),
              "overall": summarize(df), "overall_flagged_only": summarize(df[df["flagged"]]),
              "per_condition": {c: summarize(g) for c, g in df.groupby("vindr_class")}}
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    df.to_csv(MODEL_DIR / "localization_per_finding.csv", index=False)

    print(f"\n{'':20s} {'n':>4s}  {'pointing':>8s} {'chance':>7s}  {'heat in box':>11s}  {'lift':>5s}")
    for name, s in [("ALL", report["overall"]), ("all, flagged only", report["overall_flagged_only"])] + \
                   sorted(report["per_condition"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"{name:20s} {s['n']:4d}  {s['pointing']:8.1%} {s['chance']:7.1%}  {s['heat_in_box']:11.1%}  {s['lift']:5.2f}")
    draw_examples(examples)
    print(f"\nSaved {REPORT_PATH.name}, localization_per_finding.csv and {FIGURE_PATH.name}")


def summarize(df: pd.DataFrame) -> dict:
    return {"n": int(len(df)), "pointing": float(df["pointing_hit"].mean()),
            "chance": float(df["box_area"].mean()), "heat_in_box": float(df["heat_in_box"].mean()),
            "lift": float(df["heat_in_box"].mean() / df["box_area"].mean())}


def draw_examples(examples, size: int = 360):
    """One flagged example per condition: the app's heatmap with the
    radiologists' boxes drawn on top, so the numbers can be eyeballed."""
    chosen, seen = [], set()
    rng = np.random.default_rng(0)
    for i in rng.permutation(len(examples)):
        ex = examples[i]
        if ex[1] not in seen:
            seen.add(ex[1])
            chosen.append(ex)
    tiles = []
    for image_id, condition, cam, boxes, row in chosen[:8]:
        image = Image.open(VINDR_DIR / "images" / f"{image_id}.png")
        _, _, rgb = preprocess(image)
        tile = Image.fromarray(composite_multicolor_heatmap(rgb, {condition: cam}))
        draw = ImageDraw.Draw(tile)
        w, h = tile.size
        for x0, y0, x1, y1 in boxes:
            draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(255, 255, 255), width=max(2, w // 200))
        tile.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size + 28), (18, 21, 26))
        canvas.paste(tile, ((size - tile.width) // 2, 28 + (size - tile.height) // 2))
        verdict = "hit" if row["pointing_hit"] else "miss"
        ImageDraw.Draw(canvas).text((8, 8), f"{condition} - peak {verdict}, {row['heat_in_box']:.0%} heat in box",
                                    fill=CONDITION_COLORS[condition])
        tiles.append(canvas)
    cols = 4
    grid = Image.new("RGB", (cols * size, ((len(tiles) + cols - 1) // cols) * (size + 28)), (18, 21, 26))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * size, (i // cols) * (size + 28)))
    grid.save(FIGURE_PATH)


if __name__ == "__main__":
    main()
