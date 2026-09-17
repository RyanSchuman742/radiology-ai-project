"""Step 4: plain-English explanation layer using the Anthropic API.

Takes the classifier's output (predicted class, confidence, per-class
probabilities) and returns a short explanation a non-expert can use to
sanity-check the model's reasoning. Requires ANTHROPIC_API_KEY to be set
in the environment.
"""

import os

import anthropic

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "You are assisting a prototype chest X-ray screening tool. You are given "
    "a CNN classifier's prediction and confidence for a single chest X-ray, "
    "along with the fact that a Grad-CAM heatmap has been overlaid on the "
    "image showing which regions most influenced the prediction. Write a "
    "short, plain-English explanation (3-5 sentences) a non-radiologist "
    "could use to sanity-check whether the model's reasoning seems "
    "plausible. Mention what the predicted finding typically looks like on "
    "an X-ray and what a highlighted region for that finding would suggest. "
    "Do not claim certainty, and end with one sentence noting this is an "
    "educational prototype, not a medical diagnosis and not a substitute "
    "for a radiologist."
)


def get_explanation(predicted_class: str, confidence: float, class_probabilities: dict) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "(No ANTHROPIC_API_KEY set - skipping explanation. "
            f"Model predicted '{predicted_class}' with {confidence:.1%} confidence.)"
        )

    client = anthropic.Anthropic(api_key=api_key)

    probs_str = ", ".join(f"{name}: {p:.1%}" for name, p in class_probabilities.items())
    user_prompt = (
        f"Predicted finding: {predicted_class}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Full class probabilities: {probs_str}\n\n"
        "Explain this result."
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.APIError as e:
        return (
            "(Explanation unavailable - Anthropic API error: "
            f"{e.message if hasattr(e, 'message') else e}. "
            f"Model predicted '{predicted_class}' with {confidence:.1%} confidence.)"
        )

    return response.content[0].text
