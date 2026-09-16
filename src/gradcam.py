"""Step 3: Grad-CAM heatmap generation for the fine-tuned ResNet classifier.

Given an X-ray image and a trained checkpoint (from train_classifier.py),
this produces the predicted class, confidence, and a heatmap overlay showing
which regions of the scan drove that prediction.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms
from torch import nn

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "resnet_pneumonia.pt"
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(device: torch.device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    return model, class_names


def preprocess(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Returns (normalized tensor for the model, resized RGB float array [0,1] for overlay)."""
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_float = np.array(resized).astype(np.float32) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(resized).unsqueeze(0)

    return tensor, rgb_float


def diagnose_with_heatmap(image_path: str, device: torch.device | None = None):
    """Run inference + Grad-CAM on a single X-ray.

    Returns a dict with: predicted_class, confidence, class_probabilities,
    and heatmap_overlay (uint8 RGB array ready to save/display).
    """
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, class_names = load_model(device)

    image = Image.open(image_path)
    input_tensor, rgb_float = preprocess(image)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_idx = int(probs.argmax())
    predicted_class = class_names[predicted_idx]
    confidence = float(probs[predicted_idx])

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": dict(zip(class_names, probs.tolist())),
        "heatmap_overlay": overlay,
    }


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        project_root = Path(__file__).resolve().parent.parent
        image_path = project_root / "data" / "sample_xrays" / "00000001_000.png"
    else:
        image_path = sys.argv[1]

    result = diagnose_with_heatmap(str(image_path))

    print(f"Prediction: {result['predicted_class']} (confidence={result['confidence']:.3f})")
    print(f"Class probabilities: {result['class_probabilities']}")

    out_path = Path(image_path).parent / f"{Path(image_path).stem}_gradcam.png"
    plt.imsave(out_path, result["heatmap_overlay"])
    print(f"Saved heatmap overlay to {out_path}")
