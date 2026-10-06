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
    "\n\nA separate screening model rated this study as likely normal, so "
    "the findings are low-confidence. Begin with the line: \"Screening model "
    "rates this study as likely normal; findings below are low confidence.\" "
    "For each finding, state whether the localization looks like a plausible "
    "concern or more like noise."
)


def build_system_prompt(legend: dict, likely_normal: bool) -> str:
    legend_lines = "\n".join(f"- {color}: {condition}" for condition, color in legend.items())
    return (
        "You write the summary section of a chest X-ray decision-support "
        "tool. You are shown the X-ray with a heatmap overlay: each flagged "
        "condition has its own color, with intensity showing which regions "
        "drove that condition's score. Color key (for your reading only - "
        "never mention colors or color codes in the output):\n"
        f"{legend_lines}\n\n"
        "Write one line per condition, in the order given, formatted as "
        "\"<Condition>: <where the heatmap concentrates> - <whether that "
        "location is consistent with the condition>.\" Use standard anatomic "
        "terms (right/left upper, mid, lower zone; perihilar; cardiac "
        "silhouette; costophrenic angle; apex). Keep each line under 25 "
        "words. If a condition's heatmap falls mainly outside the lungs "
        "(ribs, spine, diaphragm, image edge, text markers) or two "
        "conditions rely on the same region, add a final line starting "
        "\"Note:\". Terse, neutral, clinical register: no preamble, no first "
        "person, no addressing the reader, no hedging adjectives like "
        "\"interestingly\", no disclaimers (the page shows one), no markdown."
    ) + (LIKELY_NORMAL_CONTEXT if likely_normal else "")


def image_to_base64_png(image: Image.Image, max_side: int = 768) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side))  # plenty for describing regions; fewer tokens than full size
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
        return "No tracked condition exceeded its decision threshold."

    client = anthropic.Anthropic(api_key=api_key)

    findings_str = ", ".join(f"{f['condition']}: {f['confidence']:.1%}" for f in findings)
    user_text = (
        f"Predicted findings (above decision threshold): {findings_str}\n\n"
        "Summarize where each condition's heatmap concentrates on this "
        "X-ray and whether that location supports it."
    )

    try:
        response = client.messages.create(
            model=MODEL,
            # Thinking is on by default for this model and its tokens count
            # against max_tokens, so leave room; low effort suits a short summary.
            max_tokens=4000,
            output_config={"effort": "low"},
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

    # The response can start with a thinking block - keep only the text.
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        names = ", ".join(f"{f['condition']} ({f['confidence']:.1%})" for f in findings)
        return f"(Summary unavailable. Model predicted: {names}.)"
    return text
