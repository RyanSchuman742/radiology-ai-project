"""Step 1: load a chest X-ray from disk and display it."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_xray_image(path: str) -> np.ndarray:
    """Load an X-ray image file as a single-channel grayscale uint8 array."""
    img = Image.open(path).convert("L")
    return np.array(img)


def display_xray(image: np.ndarray, title: str | None = None, save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    ax.set_title(title or "Chest X-ray")
    ax.axis("off")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved preview to {save_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sample_path = project_root / "data" / "sample_xrays" / "00000001_000.png"

    image = load_xray_image(str(sample_path))
    print(f"Loaded {sample_path.name}: shape={image.shape}, dtype={image.dtype}, "
          f"min={image.min()}, max={image.max()}")

    preview_path = project_root / "data" / "sample_xrays" / "00000001_000_preview.png"
    display_xray(image, title=sample_path.name, save_path=str(preview_path))
