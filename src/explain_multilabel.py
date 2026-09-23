"""Phase 2: explanation layer for the multi-label, multi-color Grad-CAM.

Same vision-based approach as Phase 1's explain.py, but adapted for
multiple simultaneous findings: tells Claude the color legend so it can
describe each color-coded region, and explicitly flags that overlapping
colors mean overlapping conditions rather than a single mixed finding.
"""

import base64
import io
import os

import anthropic
from PIL import Image

MODEL = "claude-sonnet-5"


LIKELY_NORMAL_CONTEXT = (
    "\n\nImportant context: a separate screening model that judges whether "
    "the scan is abnormal at all rated this scan as likely normal. The "
    "findings below still crossed their individual thresholds, so treat them "
    "as low-confidence possibilities, not established findings. Say so "
    "plainly at the start, and for each region, give an honest read on "
    "whether it looks like a plausible genuine concern or more like noise."
)


def build_system_prompt(legend: dict, likely_normal: bool) -> str:
    legend_lines = "\n".join(f"- {color}: {condition}" for condition, color in legend.items())
    return (
        "You are assisting a prototype chest X-ray screening tool. You are "
        "shown a composite Grad-CAM heatmap where each predicted condition "
        "has its own fixed color, blended onto the X-ray with intensity "
        "proportional to how strongly that region drove that condition's "
        "prediction. The color legend for this specific image is:\n"
        f"{legend_lines}\n\n"
        "Where two conditions' colors overlap in the same region, the pixel "
        "shows a blended mix of both colors, not a new condition - call "
        "this out explicitly if you see it, since it means the model is "
        "drawing on the same image region for multiple findings. Write a "
        "plain-English explanation (4-7 sentences) a non-radiologist could "
        "use to sanity-check the results: describe where each color's "
        "region actually is (upper/mid/lower lung zone, left/right, "
        "central, etc.), whether that location is plausible for its "
        "condition, and flag anything that looks like it falls outside the "
        "lungs (ribs, spine, diaphragm edge, image borders, text markers) "
        "as a possible red flag. Do not claim certainty, and end with one "
        "sentence noting this is an educational prototype, not a medical "
        "diagnosis and not a substitute for a radiologist. Write plain prose "
        "only - the text is shown as-is on a web page, so don't use markdown "
        "(no asterisks, bullet points, or headings)."
    ) + (LIKELY_NORMAL_CONTEXT if likely_normal else "")


def image_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def get_explanation(
    findings: list[dict], legend: dict, heatmap_image: Image.Image, likely_normal: bool = False
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        names = ", ".join(f"{f['condition']} ({f['confidence']:.1%})" for f in findings) or "No Finding"
        return f"(No ANTHROPIC_API_KEY set - skipping explanation. Model predicted: {names}.)"

    if not findings:
        return (
            "The model did not flag any of the 14 tracked conditions above "
            "its decision threshold for this scan, i.e. a 'No Finding' "
            "result. This is an educational prototype, not a medical "
            "diagnosis and not a substitute for a radiologist."
        )

    client = anthropic.Anthropic(api_key=api_key)

    findings_str = ", ".join(f"{f['condition']}: {f['confidence']:.1%}" for f in findings)
    user_text = (
        f"Predicted findings (above decision threshold): {findings_str}\n\n"
        "Here is the composite color-coded Grad-CAM heatmap for this "
        "specific X-ray. Describe what you actually see for each colored "
        "region and whether it supports its corresponding prediction."
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=800,
            system=build_system_prompt(legend, likely_normal),
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
        names = ", ".join(f"{f['condition']} ({f['confidence']:.1%})" for f in findings)
        return (
            "(Explanation unavailable - Anthropic API error: "
            f"{e.message if hasattr(e, 'message') else e}. "
            f"Model predicted: {names}.)"
        )

    return response.content[0].text
