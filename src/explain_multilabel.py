"""Structured preliminary interpretation of a chest radiograph.

Claude sees the radiograph twice - as uploaded, so it can assess the actual
radiographic signs, and with the model's heatmap overlay, so it can judge
whether each flagged condition's attention is anatomically plausible - plus
the model's scores. It drafts a report in the conventional structure
(technique, findings, model correlation, impression, recommendations) with
standard terminology, for a radiologist to review. The prompt keeps it
grounded: describe only what is visible, state limitations, hedge as
radiology reports do, and keep image observations separate from model output.
"""

import base64
import io
import os

import anthropic
from PIL import Image

MODEL = "claude-opus-5-5"
EFFORT = "medium"  # careful read without a long wait; thinking is always on for this model
HEADINGS = ["TECHNIQUE", "FINDINGS", "MODEL CORRELATION", "IMPRESSION", "RECOMMENDATIONS"]

SYSTEM_PROMPT = """You draft the preliminary interpretation for a chest radiograph \
decision-support research tool. A radiologist reviews every draft, so write as a \
careful radiologist would: precise, systematic, and appropriately hedged.

You receive Image 1, the frontal radiograph as uploaded, and (when the model flagged \
anything) Image 2, the same radiograph with the model's heatmap overlay, where each \
flagged condition has its own color showing which regions drove its score. You also \
receive the model's output. The model is an ensemble classifier trained mainly on NIH \
ChestX-ray14; its labels follow that dataset's conventions: "Infiltration" means a \
nonspecific parenchymal opacity, "Mass" is a lesion over 3 cm and "Nodule" one of \
3 cm or less, "Hernia" is usually a hiatal hernia, and "Pleural_Thickening" includes \
apical capping and pleural plaques.

Write these sections, each heading alone on its own line, in this order:

TECHNIQUE
Projection (PA or AP, if inferable from scapular position, clavicle orientation, or \
markers; otherwise say it cannot be determined), patient rotation (medial clavicular \
heads relative to the spinous processes), inspiratory effort (approximate number of \
posterior ribs above the hemidiaphragm), penetration, and any limitation such as a \
cropped field, overlying artifact, or reduced resolution.

FINDINGS
One line per system, starting with its label: Lines and devices, Airway, Lungs, \
Pleura, Heart and mediastinum, Hila, Bones and soft tissues, Upper abdomen. Report \
pertinent positives and negatives in standard Fleischner Society terminology - for \
example silhouette sign, air bronchograms, volume loss, blunting of the costophrenic \
angle, meniscus sign, visceral pleural line, cephalization, Kerley B lines, \
peribronchial cuffing, approximate cardiothoracic ratio (and that AP projection \
magnifies the heart). If something cannot be assessed at this resolution or \
projection, say so instead of guessing.

MODEL CORRELATION
For each flagged condition, one short paragraph: its score, where its heatmap \
concentrates in anatomic terms, the radiographic signs that support or argue against \
it, the main alternative explanation, and whether the heatmap location is \
anatomically plausible or falls on non-diagnostic structures (ribs, spine, \
diaphragm, image edge, labels). If two conditions rely on the same region, say so. \
If nothing was flagged, state that and note any visible abnormality the model's 14 \
conditions would not capture.

IMPRESSION
Numbered statements, most important first, in conventional hedged language ("likely", \
"may represent", "cannot be excluded"), each with a brief differential where useful.

RECOMMENDATIONS
Only what is warranted - for example a lateral view, comparison with prior imaging, \
CT, or clinical correlation - or "None."

Rules: describe only what is visible in these images; never invent clinical history, \
symptoms, prior studies, or measurements you cannot make. Keep image observations \
distinct from model output. Refer to heatmaps by condition name, never by color or \
color code. No treatment advice. No disclaimers (the page shows one). Plain text \
only - no markdown, asterisks, or bullet or hyphen lists (write the model correlation \
as prose); numbered lines in the impression are fine. About 300 to 500 words."""

LIKELY_NORMAL_NOTE = (
    "Screening model verdict: likely normal. The flagged conditions are therefore "
    "low confidence; weigh each one critically, and make the impression reflect the "
    "overall likely-normal assessment unless the image clearly shows otherwise."
)


def image_to_base64_png(image: Image.Image, max_side: int) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def image_block(image: Image.Image, max_side: int) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_to_base64_png(image, max_side)}}


def describe_model_output(findings, legend, likely_normal, abnormality_score) -> str:
    lines = ["Model output:"]
    if abnormality_score is not None:
        lines.append(f"Abnormality score: {abnormality_score * 100:.0f}/100")
    if findings:
        lines.append("Conditions above their decision thresholds (heatmap color in Image 2):")
        lines += [f"- {f['condition']}: {f['confidence']:.0%} (color {legend[f['condition']]})" for f in findings]
    else:
        lines.append("No condition exceeded its decision threshold; no heatmap was produced.")
    if likely_normal:
        lines.append(LIKELY_NORMAL_NOTE)
    return "\n".join(lines)


def fallback_text(findings, reason: str) -> str:
    names = ", ".join(f"{f['condition']} ({f['confidence']:.0%})" for f in findings) or "no findings"
    return f"(Interpretation unavailable: {reason}. Model output: {names}.)"


def get_explanation(
    findings: list[dict],
    legend: dict,
    original_image: Image.Image,
    heatmap_image: Image.Image | None = None,
    likely_normal: bool = False,
    abnormality_score: float | None = None,
) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return fallback_text(findings, "no ANTHROPIC_API_KEY set")

    content = [{"type": "text", "text": "Image 1: the radiograph as uploaded."}, image_block(original_image, 1024)]
    if findings and heatmap_image is not None:
        content += [{"type": "text", "text": "Image 2: the same radiograph with the model's heatmap overlay."},
                    image_block(heatmap_image, 1024)]
    content.append({"type": "text", "text": describe_model_output(findings, legend, likely_normal, abnormality_score)})

    try:
        response = anthropic.Anthropic().beta.messages.create(
            model=MODEL,
            max_tokens=16000,  # thinking tokens count against this
            output_config={"effort": EFFORT},
            # On a safety-classifier decline, the API reruns the request on a
            # fallback model within the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.APIError as e:
        return fallback_text(findings, f"API error ({getattr(e, 'message', e)})")

    if response.stop_reason == "refusal":
        return fallback_text(findings, "the request was declined")
    # Thinking (and any fallback marker) blocks come before the text - keep only text.
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or fallback_text(findings, "empty response")
