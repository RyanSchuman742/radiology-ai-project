"""Phase 2: multi-label prediction + Grad-CAM.

Predictions are an ensemble: the average of our fine-tuned ResNet50 and
TorchXRayVision's DenseNet121 trained on several combined public datasets
(see src/evaluate_ensemble.py for why - it cut false positives on outside
healthy X-rays from 79% to 53% at unchanged sensitivity). Grad-CAM comes
from our ResNet50 only.

Instead of one heatmap for "the" prediction, computes a separate Grad-CAM
pass per condition flagged as present, then composites them into one image
where each condition gets its own fixed color - so a scan with multiple
findings shows multiple color-coded regions at once, with a legend.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms
from torch import nn

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "resnet_nih14.pt"
ENSEMBLE_CONFIG_PATH = MODELS_DIR / "ensemble_config.json"
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_BLEND_ALPHA = 0.65  # blend strength at an individual heatmap's peak attention
HEATMAP_SOFT_THRESHOLD = 0.35  # suppress low-attention noise below this (post per-heatmap normalization)

# A fixed, visually distinct color per condition (kept stable across runs so
# the same condition always renders the same color in the legend and image).
CONDITION_COLORS = {
    "Atelectasis": (60, 180, 75),
    "Cardiomegaly": (60, 180, 220),
    "Consolidation": (255, 225, 25),
    "Edema": (170, 110, 40),
    "Effusion": (230, 25, 75),
    "Emphysema": (250, 190, 212),
    "Fibrosis": (0, 128, 128),
    "Hernia": (128, 128, 0),
    "Infiltration": (245, 130, 48),
    "Mass": (145, 30, 180),
    "Nodule": (70, 240, 240),
    "Pleural_Thickening": (220, 190, 255),
    "Pneumonia": (128, 0, 0),
    "Pneumothorax": (240, 50, 230),
}


OUR_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
TXV_TRANSFORM = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224),
])


@lru_cache(maxsize=1)
def load_models(device_type: str):
    """Loaded once per process, not per request."""
    device = torch.device(device_type)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    ours = models.resnet50(weights=None)
    ours.fc = nn.Linear(ours.fc.in_features, len(class_names))
    ours.load_state_dict(checkpoint["model_state"])
    ours.to(device).eval()

    # Thresholds are calibrated for the ensemble's averaged output, not for
    # either model alone - see src/evaluate_ensemble.py.
    config = json.loads(ENSEMBLE_CONFIG_PATH.read_text())
    txv = xrv.models.DenseNet(weights=config["txv_weights"]).to(device).eval()
    txv_indices = [txv.pathologies.index(c) for c in class_names]

    # Normal/abnormal gate (src/train_gate.py). It only demotes findings in
    # the UI, never removes them - see evaluate_gated_system.py for why.
    gate = models.resnet50(weights=None)
    gate.fc = nn.Linear(gate.fc.in_features, 1)
    gate.load_state_dict(torch.load(MODELS_DIR / config["gate_weights"], map_location=device)["model_state"])
    gate.to(device).eval()

    return ours, txv, txv_indices, gate, class_names, config


def preprocess(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Returns (our model's input, TXV's input, resized RGB float array [0,1]
    for the overlay). Both model inputs match evaluate_ensemble.py exactly,
    since the thresholds were calibrated through that pipeline."""
    rgb = image.convert("RGB")
    ours_tensor = OUR_TRANSFORM(rgb).unsqueeze(0)

    gray = np.array(image.convert("L")).astype(np.float32)
    gray = xrv.utils.normalize(gray, 255)[None, ...]
    txv_tensor = torch.from_numpy(TXV_TRANSFORM(gray)).float().unsqueeze(0)

    rgb_float = np.array(rgb.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    return ours_tensor, txv_tensor, rgb_float


def normalize_heatmap(cam: np.ndarray) -> np.ndarray:
    """Scale to this heatmap's own [0,1] range, then suppress low-attention
    noise so only the meaningfully "hot" region renders, not a faint tint
    over the whole image."""
    cam = cam - cam.min()
    peak = cam.max()
    if peak > 1e-6:
        cam = cam / peak
    cam = (cam - HEATMAP_SOFT_THRESHOLD) / (1 - HEATMAP_SOFT_THRESHOLD)
    return np.clip(cam, 0, 1)


def composite_multicolor_heatmap(
    rgb_float: np.ndarray, cams_by_condition: dict[str, np.ndarray]
) -> np.ndarray:
    """Blend each condition's heatmap onto the base image in its own color.
    Where two conditions' hot regions overlap, the pixel shows a weighted
    mix of both colors rather than one hiding the other."""
    h, w, _ = rgb_float.shape
    total_weight = np.zeros((h, w), dtype=np.float32)
    weighted_color = np.zeros((h, w, 3), dtype=np.float32)

    for condition, cam in cams_by_condition.items():
        color = np.array(CONDITION_COLORS[condition], dtype=np.float32) / 255.0
        weight = normalize_heatmap(cam) * MAX_BLEND_ALPHA
        weighted_color += weight[..., None] * color[None, None, :]
        total_weight += weight

    total_weight_clipped = np.clip(total_weight, 0, 1)
    safe_weight = np.clip(total_weight, 1e-6, None)
    mixed_color = weighted_color / safe_weight[..., None]

    composite = rgb_float * (1 - total_weight_clipped[..., None]) + mixed_color * total_weight_clipped[..., None]
    return np.clip(composite * 255, 0, 255).astype(np.uint8)


def diagnose_with_heatmap(image_path: str, device: torch.device | None = None, max_findings: int = 5):
    """Run multi-label inference + per-condition Grad-CAM on a single X-ray.

    Returns a dict with: findings (list of {condition, confidence} above the
    decision threshold, sorted by confidence, capped at max_findings for
    legibility), all_probabilities (all 14 conditions), heatmap_overlay
    (multi-color composite, uint8 RGB), legend (condition -> "#rrggbb"),
    and likely_normal (the gate's verdict - findings are still returned
    either way, the UI just ranks them lower when it's True).
    """
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, txv, txv_indices, gate, class_names, config = load_models(str(device))
    decision_thresholds = config["thresholds"]

    image = Image.open(image_path)
    input_tensor, txv_tensor, rgb_float = preprocess(image)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        ours_probs = torch.sigmoid(model(input_tensor))[0].cpu().numpy()
        txv_probs = txv(txv_tensor.to(device))[0, txv_indices].cpu().numpy()
        gate_abnormal_prob = float(torch.sigmoid(gate(input_tensor)).item())
    probs = (ours_probs + txv_probs) / 2

    all_probabilities = dict(zip(class_names, probs.tolist()))
    findings = sorted(
        [(name, prob) for name, prob in all_probabilities.items() if prob >= decision_thresholds[name]],
        key=lambda x: x[1],
        reverse=True,
    )[:max_findings]

    target_layer = model.layer4[-1]
    cam_engine = GradCAM(model=model, target_layers=[target_layer])

    cams_by_condition = {}
    for condition, _ in findings:
        class_idx = class_names.index(condition)
        grayscale_cam = cam_engine(
            input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)]
        )[0]
        cams_by_condition[condition] = grayscale_cam

    if cams_by_condition:
        overlay = composite_multicolor_heatmap(rgb_float, cams_by_condition)
    else:
        # No condition crossed the threshold - show the plain X-ray, no tint.
        overlay = (rgb_float * 255).astype(np.uint8)

    legend = {
        condition: "#{:02x}{:02x}{:02x}".format(*CONDITION_COLORS[condition])
        for condition, _ in findings
    }

    # Individual single-condition overlays, for when the composite's colors
    # overlap and blend (common - many conditions concentrate in the same
    # anatomical region) and someone needs to see one condition in isolation.
    individual_heatmaps = {
        condition: composite_multicolor_heatmap(rgb_float, {condition: cam})
        for condition, cam in cams_by_condition.items()
    }

    return {
        "findings": [{"condition": c, "confidence": p} for c, p in findings],
        "all_probabilities": all_probabilities,
        "heatmap_overlay": overlay,
        "individual_heatmaps": individual_heatmaps,
        "legend": legend,
        "decision_thresholds": decision_thresholds,
        "gate_abnormal_probability": gate_abnormal_prob,
        "likely_normal": gate_abnormal_prob < config["gate_threshold"],
    }


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        project_root = Path(__file__).resolve().parent.parent
        image_path = project_root / "data" / "nih14_subset" / "images" / "00000001_000.png"
    else:
        image_path = sys.argv[1]

    result = diagnose_with_heatmap(str(image_path))

    print("Findings (above threshold):")
    for f in result["findings"]:
        print(f"  {f['condition']}: {f['confidence']:.1%}  ({result['legend'][f['condition']]})")
    if not result["findings"]:
        print("  (none - model predicts no finding)")

    print("\nAll probabilities:")
    for cond, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {cond}: {prob:.1%}")

    out_path = Path(image_path).parent / f"{Path(image_path).stem}_multicolor_gradcam.png"
    plt.imsave(out_path, result["heatmap_overlay"])
    print(f"\nSaved multi-color heatmap to {out_path}")
