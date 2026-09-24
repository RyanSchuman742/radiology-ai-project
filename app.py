"""Flask web app: upload a chest X-ray, get back a Grad-CAM heatmap,
predicted diagnosis, and a plain-English explanation."""

import base64
import io
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, render_template, request
from PIL import Image

from src.explain_multilabel import get_explanation
from src.gradcam_multilabel import diagnose_with_heatmap

load_dotenv()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
MIN_DIMENSION = 100  # px - reject anything too small to plausibly be an X-ray
GRAYSCALE_CHANNEL_TOLERANCE = 12  # avg per-pixel RGB deviation above this looks like a color photo

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

UPLOAD_DIR = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def looks_like_color_photo(image: Image.Image) -> bool:
    """X-rays are grayscale even when saved as RGB (R==G==B per pixel).
    A real color photo has much higher channel divergence - used as a soft
    heuristic to warn the user, not to hard-block uploads."""
    rgb = image.convert("RGB").resize((64, 64))
    pixels = list(rgb.getdata())
    total_deviation = sum(max(r, g, b) - min(r, g, b) for r, g, b in pixels)
    avg_deviation = total_deviation / len(pixels)
    return avg_deviation > GRAYSCALE_CHANNEL_TOLERANCE


def image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/diagnose", methods=["POST"])
def diagnose():
    file = request.files.get("xray")
    if not file or file.filename == "":
        return render_template("index.html", error="Please choose an X-ray image file.")

    if not allowed_file(file.filename):
        return render_template("index.html", error="Please upload a PNG or JPEG image.")

    try:
        uploaded_image = Image.open(file.stream)
        uploaded_image.load()
    except Exception:
        return render_template("index.html", error="That file couldn't be read as an image.")

    if uploaded_image.width < MIN_DIMENSION or uploaded_image.height < MIN_DIMENSION:
        return render_template(
            "index.html",
            error=f"Image is too small ({uploaded_image.width}x{uploaded_image.height}px) to be a usable X-ray.",
        )

    warning = None
    if looks_like_color_photo(uploaded_image):
        warning = "This doesn't look like a grayscale X-ray. Results below may not be meaningful."

    original_image = uploaded_image.convert("L")
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.png")
    original_image.save(temp_path)

    try:
        result = diagnose_with_heatmap(temp_path)
    except FileNotFoundError:
        return render_template(
            "index.html",
            error="No trained model found yet. Train the classifier first (src/train_multilabel.py).",
        )
    finally:
        os.remove(temp_path)

    heatmap_image = Image.fromarray(result["heatmap_overlay"])
    explanation = get_explanation(
        result["findings"], result["legend"], heatmap_image, likely_normal=result["likely_normal"]
    )

    views = {
        "original": image_to_data_uri(Image.fromarray(result["display_image"])),
        "overlay": image_to_data_uri(heatmap_image),
        "findings": {
            condition: image_to_data_uri(Image.fromarray(overlay))
            for condition, overlay in result["individual_heatmaps"].items()
        },
    }

    findings = [
        {"condition": f["condition"], "confidence": f["confidence"], "color": result["legend"][f["condition"]]}
        for f in result["findings"]
    ]

    thresholds = result["decision_thresholds"]
    all_conditions = sorted(
        (
            {"name": name, "prob": prob, "above": prob >= thresholds[name]}
            for name, prob in result["all_probabilities"].items()
        ),
        key=lambda c: c["prob"],
        reverse=True,
    )

    if findings and not result["likely_normal"]:
        status = "abnormal"
    elif findings:
        status = "low_confidence"
    else:
        status = "normal"

    return render_template(
        "result.html",
        views=views,
        findings=findings,
        all_conditions=all_conditions,
        status=status,
        abnormality_score=result["gate_abnormal_probability"],
        explanation=explanation,
        warning=warning,
        filename=file.filename,
        dimensions=f"{uploaded_image.width} × {uploaded_image.height}",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
