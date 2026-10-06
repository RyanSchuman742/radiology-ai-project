"""Which heatmap method points best at what radiologists marked?

Compares Grad-CAM variants for the deployed models on two independent
localization benchmarks, without changing any prediction:
- VinDr test split: radiologist bounding boxes (783 findings, 540 images)
- CheXpert validation: radiologist outlines from CheXlocalize (frontal
  images, conditions mapped to ours)

Variants (all on the deployed models):
- gradcam_l4      ResNet50 last block - what the app ships today (baseline)
- hirescam_l4     HiResCAM: keeps per-location gradients instead of averaging
- gradcampp_l4    Grad-CAM++: weights positive gradients, often better for
                  multiple instances of a finding
- gradcam_l3      ResNet50 second-to-last block: 14x14 instead of 7x7
- gradcam_l3l4    both blocks averaged
- layercam_l3l4   LayerCAM, designed for combining finer layers
- txv_l           Grad-CAM on the ensemble's TorchXRayVision DenseNet
- ensemble        average of gradcam_l4 and txv_l (both models' attention)

Metrics per finding (same as evaluate_localization.py): pointing = the
heatmap's peak lies in the marked region; heat in box = share of the heatmap
as displayed (after the app's soft threshold) inside it. Differences vs. the
baseline get 95% intervals from a paired bootstrap that resamples images.

Writes models/heatmap_comparison_report.json.
"""

import json

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from evaluate_chexpert import CHEXPERT_DIR, CHEXPERT_TO_OURS
from evaluate_localization import GRID, finding_masks
from evaluate_vindr import VINDR_TO_OURS
from gradcam_multilabel import load_models, normalize_heatmap, preprocess
from scoreboard import VINDR_DIR, load_test_set
from train_multilabel import CONDITIONS, MODEL_DIR

REPORT_PATH = MODEL_DIR / "heatmap_comparison_report.json"
BASELINE = "gradcam_l4"
N_BOOT = 1000
SEED = 42


def chexpert_findings():
    """(image path, CheXpert class, outline mask on GRID) for every positive."""
    seg = json.loads((CHEXPERT_DIR / "chexlocalize" / "gt_segmentations_val.json").read_text())
    ann = json.loads((CHEXPERT_DIR / "chexlocalize" / "gt_annotations_val.json").read_text())
    out = []
    for path in sorted((CHEXPERT_DIR / "valid").glob("*/*/*frontal*.png")):
        image_id = "_".join(path.relative_to(CHEXPERT_DIR / "valid").with_suffix("").parts)
        for c in CHEXPERT_TO_OURS:
            if c in ann.get(image_id, {}):
                m = mask_utils.decode(seg[image_id][c]).astype(np.uint8)
                out.append((str(path), c, cv2.resize(m, (GRID, GRID), interpolation=cv2.INTER_NEAREST) > 0))
    return out


def vindr_findings():
    return [(str(VINDR_DIR / "images" / f"{i}.png"), c, m) for i, c, m, _ in finding_masks(load_test_set())]


class Heatmaps:
    def __init__(self, device):
        self.device = device
        self.ours, self.txv, self.txv_idx, _, _, _ = load_models(str(device))
        l3, l4 = self.ours.layer3[-1], self.ours.layer4[-1]
        self.engines = {
            "gradcam_l4": GradCAM(self.ours, [l4]),
            "hirescam_l4": HiResCAM(self.ours, [l4]),
            "gradcampp_l4": GradCAMPlusPlus(self.ours, [l4]),
            "gradcam_l3": GradCAM(self.ours, [l3]),
            "gradcam_l3l4": GradCAM(self.ours, [l3, l4]),
            "layercam_l3l4": LayerCAM(self.ours, [l3, l4]),
        }
        self.txv_engine = GradCAM(self.txv, [self.txv.features[-1]])

    def prepare(self, path):
        image = Image.open(path)
        ours_x, txv_x, _ = preprocess(image)
        ours_x, txv_x = ours_x.to(self.device), txv_x.to(self.device)
        with torch.no_grad():
            probs = (torch.sigmoid(self.ours(ours_x))[0] + self.txv(txv_x)[0, self.txv_idx]).cpu().numpy() / 2
        return {"ours": ours_x, "txv": txv_x, "probs": probs, "size": image.size}

    def all_variants(self, prep, idx: int) -> dict:
        """Every variant's heatmap for one condition, on the normalized GRID."""
        target = [ClassifierOutputTarget(idx)]
        cams = {}
        for name, engine in self.engines.items():
            cam = engine(input_tensor=prep["ours"], targets=target)[0]
            cams[name] = cv2.resize(cam.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_LINEAR)
        txv_cam = self.txv_engine(input_tensor=prep["txv"], targets=[ClassifierOutputTarget(self.txv_idx[idx])])[0]
        cams["txv_l"] = self.uncrop(txv_cam, prep["size"])
        cams["ensemble"] = (cams["gradcam_l4"] + cams["txv_l"]) / 2
        return cams

    @staticmethod
    def uncrop(cam, size):
        """TXV center-crops to a square; put its heatmap back where that
        square sits in the full image (outside it: no attention)."""
        w, h = size
        side = min(w, h)
        x0, x1 = round((w - side) / 2 / w * GRID), round((w + side) / 2 / w * GRID)
        y0, y1 = round((h - side) / 2 / h * GRID), round((h + side) / 2 / h * GRID)
        canvas = np.zeros((GRID, GRID), dtype=np.float32)
        canvas[y0:y1, x0:x1] = cv2.resize(cam.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        return canvas


def score(cam, mask) -> tuple[bool, float]:
    shown = normalize_heatmap(cam)
    peak = np.unravel_index(cam.argmax(), cam.shape)
    return bool(mask[peak]), float((shown * mask).sum() / max(shown.sum(), 1e-6))


def run(benchmark, findings, mapping, heatmaps) -> pd.DataFrame:
    rows, cache_path, prep = [], None, None
    for n, (path, cls, mask) in enumerate(findings):
        if path != cache_path:
            prep, cache_path = heatmaps.prepare(path), path
        condition = max(mapping[cls], key=lambda c: prep["probs"][CONDITIONS.index(c)])
        for variant, cam in heatmaps.all_variants(prep, CONDITIONS.index(condition)).items():
            hit, heat = score(cam, mask)
            rows.append({"benchmark": benchmark, "image": path, "class": cls, "variant": variant,
                         "pointing": hit, "heat_in_box": heat, "box_area": float(mask.mean())})
        if n % 100 == 0:
            print(f"  {benchmark} {n}/{len(findings)}", flush=True)
    return pd.DataFrame(rows)


def paired_ci(df, metric, variant):
    """Bootstrap over images of (variant - baseline) mean metric."""
    wide = df.pivot_table(index=["image", "class"], columns="variant", values=metric).reset_index()
    diff = (wide[variant] - wide[BASELINE]).to_numpy()
    images = wide["image"].to_numpy()
    groups = [np.flatnonzero(images == i) for i in np.unique(images)]
    rng = np.random.default_rng(SEED)
    boots = [np.mean(diff[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])])
             for _ in range(N_BOOT)]
    return float(diff.mean()), [float(x) for x in np.percentile(boots, [2.5, 97.5])]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    heatmaps = Heatmaps(device)
    vindr, chexpert = vindr_findings(), chexpert_findings()
    print(f"{len(vindr)} VinDr findings, {len(chexpert)} CheXpert findings", flush=True)
    df = pd.concat([run("vindr", vindr, VINDR_TO_OURS, heatmaps),
                    run("chexpert", chexpert, CHEXPERT_TO_OURS, heatmaps)])
    df.to_csv(MODEL_DIR / "heatmap_comparison_per_finding.csv", index=False)

    report = {}
    for bench, g in df.groupby("benchmark"):
        report[bench] = {"n_findings": int(len(g) / g["variant"].nunique()), "chance": float(g["box_area"].mean()),
                         "variants": {}, "per_class_pointing": {}}
        print(f"\n=== {bench}: {report[bench]['n_findings']} findings, chance {report[bench]['chance']:.1%} ===")
        print(f"{'variant':15s} {'pointing':>8s} {'heat in box':>11s}   {'Δ pointing vs baseline':>30s}   {'Δ heat vs baseline':>28s}")
        for variant, v in g.groupby("variant"):
            entry = {"pointing": float(v["pointing"].mean()), "heat_in_box": float(v["heat_in_box"].mean())}
            line = f"{variant:15s} {entry['pointing']:8.1%} {entry['heat_in_box']:11.1%}"
            if variant != BASELINE:
                for m in ["pointing", "heat_in_box"]:
                    d, (lo, hi) = paired_ci(g, m, variant)
                    entry[f"delta_{m}"] = {"value": d, "ci95": [lo, hi]}
                    tag = "same" if lo <= 0 <= hi else ("BETTER" if d > 0 else "worse")
                    line += f"   {d:+.3f} [{lo:+.3f},{hi:+.3f}] {tag:6s}"
            report[bench]["variants"][variant] = entry
            report[bench]["per_class_pointing"][variant] = v.groupby("class")["pointing"].mean().round(3).to_dict()
            print(line)
        pc = pd.DataFrame(report[bench]["per_class_pointing"])
        print("\npointing by condition:\n" + pc.to_string(float_format=lambda x: f"{x:.0%}"))
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
