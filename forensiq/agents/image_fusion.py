
"""
forensiq/agents/image_fusion.py

Two-detector, LLM-orchestrated fusion for the Image Agent.

FIX: calibration was validated during evaluation
(temperature=1.5998 on the original dataset, independently refit to
1.6051 on OpenFake -- consistent within 0.5%, suggesting this reflects
a genuine, stable property of UnivFD's own overconfidence rather than
a dataset-specific artifact) but was never actually wired into this
function -- evaluation notebooks applied it manually, while the
deployed app called the uncalibrated version. get_univfd_score now
applies calibration by default, using the original dataset's fitted
value as the production default given it was fit on a dedicated,
class-balanced calibration set.
"""
import torch
from PIL import Image

# Default calibration temperature: fit on the original dataset's
# 20+20 balanced calibration sample. OpenFake's independent refit
# (1.6051) landed within 0.5% of this value, supporting its use as a
# stable, non-dataset-specific default rather than requiring a fresh
# fit per deployment.
DEFAULT_CALIBRATION_TEMPERATURE = 1.5998


def get_univfd_score(univfd_model, image_path, device, transform, temperature=DEFAULT_CALIBRATION_TEMPERATURE):
    """
    Returns UnivFD's CALIBRATED fake probability. Previously returned
    the raw, uncalibrated sigmoid output -- this function is what the
    live app actually calls, so the app was silently using an
    uncalibrated score even after calibration was validated to improve
    results in evaluation.
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    univfd_model.eval()
    with torch.no_grad():
        logit = univfd_model(img_tensor)
        fake_prob = torch.sigmoid(logit / temperature).item()
    return fake_prob


FUSION_SYSTEM_PROMPT = """You are a forensic image analyst. You will examine an image in two stages.

STAGE 1 - Independent visual assessment: Before seeing any other evidence, examine the image for visual cues consistent with AI generation: anatomical inconsistencies (hands, eyes, teeth, ears), unnatural texture or skin smoothing, inconsistent lighting/shadow direction, warped or nonsensical background elements, distorted text, or unnaturally perfect symmetry. Form your own independent judgment first.

STAGE 2 - Integration with a second detector's evidence: You will then be given a separate, calibrated score from UnivFD, a trained classifier (0.0 = confidently real, 1.0 = confidently fake). Compare this to your own Stage 1 judgment. If they agree, state this and give a confident combined verdict. If they DISAGREE, explicitly reason through why -- do not simply average the two scores. Consider: your visual judgment may catch generation artifacts the classifier wasn't trained on; the classifier may catch subtle statistical patterns invisible to human/visual inspection. State which signal you weight more heavily for THIS specific image, and why.

Respond in exactly this format:
STAGE1_VERDICT: [real/fake]
STAGE1_CONFIDENCE: [0.0-1.0]
STAGE1_REASONING: [1-2 sentences on visual cues observed]
FINAL_VERDICT: [real/fake]
FINAL_CONFIDENCE: [0.0-1.0]
FINAL_REASONING: [2-3 sentences explaining how Stage 1 and UnivFD's score were reconciled]"""


def parse_fusion_response(text):
    result = {}
    for line in text.strip().split("\n"):
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        value = value.strip()
        if key in ["STAGE1_VERDICT", "FINAL_VERDICT"]:
            result[key.lower()] = "fake" if "fake" in value.lower() else "real"
        elif key in ["STAGE1_CONFIDENCE", "FINAL_CONFIDENCE"]:
            try:
                result[key.lower()] = float(value)
            except ValueError:
                result[key.lower()] = 0.5
        elif key in ["STAGE1_REASONING", "FINAL_REASONING"]:
            result[key.lower()] = value
    return result


def run_fused_image_agent(univfd_model, image_path, device, transform, gemini_api_key,
                            model="gemini-flash-latest", temperature=DEFAULT_CALIBRATION_TEMPERATURE):
    from google import genai
    from google.genai import types

    univfd_score = get_univfd_score(univfd_model, image_path, device, transform, temperature=temperature)

    client = genai.Client(api_key=gemini_api_key)
    img = Image.open(image_path)

    prompt = (
        f"Examine this image and complete Stage 1 as instructed. "
        f"Then, for Stage 2: UnivFD (a calibrated, CLIP-based trained classifier) scored this image "
        f"{univfd_score:.4f} on a 0.0 (confidently real) to 1.0 (confidently fake) scale. "
        f"Integrate this with your Stage 1 judgment as instructed."
    )

    response = client.models.generate_content(
        model=model,
        contents=[img, prompt],
        config=types.GenerateContentConfig(system_instruction=FUSION_SYSTEM_PROMPT),
    )

    parsed = parse_fusion_response(response.text)

    final_verdict = parsed.get("final_verdict", "real")
    final_confidence = parsed.get("final_confidence", 0.5)

    return {
        "agent_name": "image",
        "verdict": final_verdict,
        "confidence": final_confidence,
        "raw_score": univfd_score,
        "explanation_data": {
            "univfd_score": univfd_score,
            "calibration_temperature": temperature,
            "stage1_verdict": parsed.get("stage1_verdict"),
            "stage1_confidence": parsed.get("stage1_confidence"),
            "stage1_reasoning": parsed.get("stage1_reasoning"),
            "final_reasoning": parsed.get("final_reasoning"),
            "raw_llm_response": response.text,
        },
        "error": None,
    }
