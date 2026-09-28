
"""
forensiq/agents/video_fusion.py

Applies the SAME proven fusion technique validated on the Image Agent
(+17.15 point improvement) to the Video Agent's face-swap/reenactment
detection: temperature-calibrated EfficientNet score + independent
Gemini visual reasoning, with explicit disagreement resolution rather
than blind averaging.

This is a genuine re-test, distinct from the earlier "dual-pathway"
DigiFakeAV experiment (which combined EfficientNet + UnivFD via simple
confidence-weighted averaging, with no calibration and no LLM
reasoning over disagreement -- a cruder combination method than what
is used here, and not the same technique that produced the Image
Agent's validated improvement).
"""
import torch
from PIL import Image


def get_video_frame_logit(video_model, frame, device, transform):
    """
    frame: a PIL Image (a single extracted video frame), not a file
    path -- callers extract the frame themselves so this function
    works identically whether the source is a real video file or a
    pre-extracted frame image.
    """
    img_tensor = transform(frame).unsqueeze(0).to(device)
    video_model.eval()
    with torch.no_grad():
        logits = video_model(img_tensor)
    # Video Agent's class_to_idx: class 0 = fake. Return the raw
    # class-0 logit for calibration, consistent with how UnivFD's
    # single-logit calibration was structured.
    return logits[0, 0].item()


VIDEO_FUSION_SYSTEM_PROMPT = """You are a forensic video analyst examining a single frame extracted from a video, for signs of face-swap or face-reenactment deepfake manipulation.

STAGE 1 - Independent visual assessment: Before seeing any other evidence, examine the face for cues specific to face-swap/reenactment manipulation: blending artifacts at the face boundary (jawline, hairline), inconsistent skin texture between face and neck/ears, unnatural or asymmetric eye reflections, lighting direction mismatch between the face and the rest of the scene, unnatural mouth/teeth rendering, or an unnaturally smooth/plastic skin appearance. Form your own independent judgment first.

STAGE 2 - Integration with a second detector's evidence: You will then be given a separate calibrated score from an EfficientNet-based classifier trained specifically on face-swap/reenactment techniques (0.0 = confidently real, 1.0 = confidently fake). Compare this to your own Stage 1 judgment. If they agree, give a confident combined verdict. If they disagree, reason through why explicitly -- do not simply average. State which signal you weight more heavily for THIS specific frame, and why.

Respond in exactly this format:
STAGE1_VERDICT: [real/fake]
STAGE1_CONFIDENCE: [0.0-1.0]
STAGE1_REASONING: [1-2 sentences on visual cues observed]
FINAL_VERDICT: [real/fake]
FINAL_CONFIDENCE: [0.0-1.0]
FINAL_REASONING: [2-3 sentences explaining how Stage 1 and the classifier score were reconciled]"""


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


def run_fused_video_agent(video_model, frame, device, transform, temperature, gemini_api_key, model="gemini-3.5-flash-lite"):
    """
    frame: a PIL Image (single extracted video frame).
    temperature: the fitted calibration temperature for the video
        model's logits (fit separately, same method as the Image
        Agent's calibration).
    """
    from google import genai
    from google.genai import types

    raw_logit = get_video_frame_logit(video_model, frame, device, transform)
    calibrated_score = torch.sigmoid(torch.tensor(raw_logit / temperature)).item()

    client = genai.Client(api_key=gemini_api_key)

    prompt = (
        f"Examine this video frame and complete Stage 1 as instructed. "
        f"Then, for Stage 2: a calibrated EfficientNet classifier (trained on "
        f"face-swap/reenactment techniques) scored this frame {calibrated_score:.4f} "
        f"on a 0.0 (confidently real) to 1.0 (confidently fake) scale. "
        f"Integrate this with your Stage 1 judgment as instructed."
    )

    response = client.models.generate_content(
        model=model,
        contents=[frame, prompt],
        config=types.GenerateContentConfig(system_instruction=VIDEO_FUSION_SYSTEM_PROMPT),
    )

    parsed = parse_fusion_response(response.text)

    return {
        "agent_name": "video",
        "verdict": parsed.get("final_verdict", "real"),
        "confidence": parsed.get("final_confidence", 0.5),
        "raw_score": calibrated_score,
        "explanation_data": {
            "efficientnet_calibrated_score": calibrated_score,
            "stage1_verdict": parsed.get("stage1_verdict"),
            "stage1_confidence": parsed.get("stage1_confidence"),
            "stage1_reasoning": parsed.get("stage1_reasoning"),
            "final_reasoning": parsed.get("final_reasoning"),
            "raw_llm_response": response.text,
        },
        "error": None,
    }
