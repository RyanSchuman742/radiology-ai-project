"""Step 4: plain-English explanation layer using the Anthropic API.

Sends Claude the classifier's output *and* the actual Grad-CAM heatmap
image, so the explanation describes what's actually highlighted on this
specific scan rather than generic disease boilerplate. Requires
ANTHROPIC_API_KEY to be set in the environment.
"""

import base64
import io
import os

import anthropic
from PIL import Image

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "You are assisting a prototype chest X-ray screening tool. You are shown "
    "a Grad-CAM heatmap overlaid on a chest X-ray - warmer colors (red/orange/"
    "yellow) mark the regions that most influenced a CNN classifier's "
    "prediction - along with the classifier's predicted finding and "
    "confidence. Look at THIS specific image and describe, in plain English "
    "(3-5 sentences), where the highlighted region actually is (e.g. upper/"
    "mid/lower lung zone, left or right side, central, near the heart "
    "border, near the edge of the image), roughly how large or concentrated "
    "it looks, and whether that location is plausible for the predicted "
    "finding. Be specific to what you observe in this image - do not give a "
    "generic description of the disease that would apply to any scan. If "
    "the highlighted region looks like it falls outside the lungs (on ribs, "
    "the diaphragm edge, image borders, or text/markers on the film), say "
    "so explicitly as a red flag that the model may be keying off an "
    "irrelevant artifact. Do not claim certainty, and end with one sentence "
    "noting this is an educational prototype, not a medical diagnosis and "
    "not a substitute for a radiologist."
)


def image_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def get_explanation(
    predicted_class: str,
    confidence: float,
    class_probabilities: dict,
    heatmap_image: Image.Image,
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "(No ANTHROPIC_API_KEY set - skipping explanation. "
            f"Model predicted '{predicted_class}' with {confidence:.1%} confidence.)"
        )

    client = anthropic.Anthropic(api_key=api_key)

    probs_str = ", ".join(f"{name}: {p:.1%}" for name, p in class_probabilities.items())
    user_text = (
        f"Predicted finding: {predicted_class}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Full class probabilities: {probs_str}\n\n"
        "Here is the Grad-CAM heatmap for this specific X-ray. Describe what "
        "you actually see in the highlighted region and whether it supports "
        "the prediction."
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_to_base64_png(heatmap_image),
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }],
        )
    except anthropic.APIError as e:
        return (
            "(Explanation unavailable - Anthropic API error: "
            f"{e.message if hasattr(e, 'message') else e}. "
            f"Model predicted '{predicted_class}' with {confidence:.1%} confidence.)"
        )

    return response.content[0].text
