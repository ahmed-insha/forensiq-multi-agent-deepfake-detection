"""
forensiq/agents/narration.py
"""
import PIL.Image

NARRATION_SYSTEM_PROMPT = """You are a forensic media analysis assistant. You will be given:
1. Optionally, a still frame from a video under analysis.
2. Structured, computer-generated evidence, including each agent's verdict, confidence, the CATEGORY OF MANIPULATION each agent's detection is consistent with, and any cross-modal disagreement or timestamp localization already computed.

Write a short report with exactly three parts:

PART 1 - Content description (1-2 sentences): if an image was provided, briefly describe what is visually depicted. Purely descriptive, no forensic judgment. Skip if no image provided.

PART 2 - Forensic reasoning (3-5 sentences): explain the overall verdict using ONLY the structured evidence provided. Reference specific agents, confidence scores, and timestamps where relevant. If agents disagree, explain what that disagreement suggests. Do not invent evidence or claims the data does not support.

PART 3 - Type of manipulation (1-2 sentences): state which CATEGORY of manipulation the evidence is consistent with (e.g. "consistent with a generative face-swap or reenactment deepfake" or "consistent with synthetic/AI-generated speech" or "consistent with a duplicated video segment"). Be clear this is a category, not identification of a specific tool or software, since the underlying detectors were not trained to distinguish between specific generation techniques.

Write for a forensic investigator audience: precise, factual, non-dramatic. Total response under 180 words."""

MANIPULATION_TYPE_DESCRIPTIONS = {
    "audio": "synthetic/AI-generated speech",
    "video": "generative face-forgery (deepfake face-swap or reenactment)",
    "structural": "spliced/cloned/inpainted content, or a duplicated video segment",
    "image": "fully AI-generated image content",
}


def build_evidence_summary(orchestrator_result, timeline_summary=None, clear_summary=None):
    lines = [f"Overall verdict: {orchestrator_result['final_verdict'].upper()} "
             f"(confidence: {orchestrator_result['final_confidence']*100:.1f}%)"]

    lines.append("\nPer-agent results (with manipulation category if flagged fake):")
    for agent_name, result in orchestrator_result.get("agent_summary", {}).items():
        category = MANIPULATION_TYPE_DESCRIPTIONS.get(agent_name, "unknown")
        note = f" -- category if fake: {category}" if result["verdict"] == "fake" else ""
        lines.append(f"- {agent_name.capitalize()} Agent: {result['verdict']} "
                      f"(confidence: {result['confidence']*100:.1f}%){note}")

    if orchestrator_result.get("disagreement_detected"):
        lines.append(f"\nCross-modal disagreement: {orchestrator_result['disagreement_details']}")

    if timeline_summary:
        lines.append("\nTimestamp localization:")
        for status, ranges in timeline_summary.items():
            if ranges:
                range_strs = [f"{r[0]:.1f}s-{r[1]:.1f}s" for r in ranges]
                lines.append(f"- {status}: {', '.join(range_strs)}")

    if clear_summary and clear_summary.get("structural_tampering_detected"):
        lines.append("\nStructural findings (note: may reflect synthetic test construction, not genuine tampering):")
        for f in clear_summary.get("structural_findings", []):
            lines.append(f"- {f['type']}: content at {f['original_time']:.1f}s duplicated at {f['duplicate_time']:.1f}s")

    return "\n".join(lines)


def generate_narration(orchestrator_result, api_key, timeline_summary=None, clear_summary=None, sample_frame_path=None, model="gemini-3.6-flash"):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    evidence_text = build_evidence_summary(orchestrator_result, timeline_summary, clear_summary)
    prompt_text = f"Structured evidence from detection agents:\n\n{evidence_text}\n\nWrite the three-part report as instructed."

    contents = []
    if sample_frame_path:
        contents.append(PIL.Image.open(sample_frame_path))
    contents.append(prompt_text)

    response = client.models.generate_content(
        model=model, contents=contents,
        config=types.GenerateContentConfig(system_instruction=NARRATION_SYSTEM_PROMPT),
    )
    return response.text
