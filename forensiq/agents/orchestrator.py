
"""
forensiq/agents/orchestrator.py

LangGraph-based orchestrator. image_node now passes the Gemini API key
(if available in models dict) so the Image Agent uses the validated
fusion improvement. Fusion includes inconclusive-verdict logic (Boato
et al., "Don't Guess, Escalate") for narrow-margin disagreements.
"""
import os
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from forensiq.agents.agents import (
    run_audio_agent, run_video_agent, run_structural_agent, run_image_agent,
)


class OrchestratorState(TypedDict, total=False):
    input_path: str
    input_type: str
    audio_path: Optional[str]
    frame_paths: Optional[list]
    sample_frame_path: Optional[str]
    audio_result: Optional[dict]
    video_result: Optional[dict]
    structural_result: Optional[dict]
    image_result: Optional[dict]
    final_verdict: str
    final_confidence: float
    disagreement_detected: bool
    disagreement_details: str
    agent_summary: dict


def detect_and_prepare_node(state: OrchestratorState, models: dict, device) -> dict:
    import cv2
    import tempfile
    path = state["input_path"]
    ext = os.path.splitext(path)[1].lower()
    update = {}

    if ext in [".mp4", ".mov", ".avi"]:
        update["input_type"] = "video"
        audio_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        os.system(f'ffmpeg -y -i "{path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_out}" -loglevel error')
        update["audio_path"] = audio_out if os.path.exists(audio_out) else None

        cap = cv2.VideoCapture(path)
        frame_paths = []
        frame_dir = tempfile.mkdtemp()
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or idx >= 16:
                break
            frame_path = os.path.join(frame_dir, f"frame_{idx:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            idx += 1
        cap.release()
        update["frame_paths"] = frame_paths
        update["sample_frame_path"] = frame_paths[0] if frame_paths else None

    elif ext in [".flac", ".wav", ".mp3"]:
        update["input_type"] = "audio"
        update["audio_path"] = path
    elif ext in [".jpg", ".jpeg", ".png"]:
        update["input_type"] = "image"
    else:
        update["input_type"] = "unknown"

    return update

def audio_node(state: OrchestratorState, models: dict, device) -> dict:
    if state.get("audio_path"):
        result = run_audio_agent(models["audio"], state["audio_path"], device)
        return {"audio_result": result.__dict__}
    return {}


def video_node(state: OrchestratorState, models: dict, device) -> dict:
    if state.get("frame_paths"):
        result = run_video_agent(
            models["video"], state["frame_paths"], device, models["video_transform"],
            full_video_path=state["input_path"],
        )
        return {"video_result": result.__dict__}
    return {}


def structural_node(state: OrchestratorState, models: dict, device) -> dict:
    if state.get("sample_frame_path"):
        result = run_structural_agent(
            models["structural_mvssnet"], models["structural_interframe_funcs"],
            state["sample_frame_path"], state["input_path"], device,
        )
        return {"structural_result": result.__dict__}
    return {}


def image_node(state: OrchestratorState, models: dict, device) -> dict:
    if state["input_type"] == "image":
        gemini_api_key = models.get("gemini_api_key")
        result = run_image_agent(
            models["image"], state["input_path"], device, models["image_transform"],
            gemini_api_key=gemini_api_key,
        )
        return {"image_result": result.__dict__}
    return {}


def fusion_node(state: OrchestratorState) -> dict:
    results = {}
    for key in ["audio_result", "video_result", "structural_result", "image_result"]:
        if state.get(key) and state[key].get("verdict") != "unknown":
            results[key.replace("_result", "")] = state[key]

    if not results:
        return {
            "agent_summary": results, "final_verdict": "unknown", "final_confidence": 0.0,
            "disagreement_detected": False, "disagreement_details": "No agent produced a valid result.",
        }

    verdicts = {name: r["verdict"] for name, r in results.items()}
    fake_votes = [name for name, v in verdicts.items() if v == "fake"]
    real_votes = [name for name, v in verdicts.items() if v == "real"]
    disagreement = len(fake_votes) > 0 and len(real_votes) > 0

    fake_weight = sum(results[name]["confidence"] for name in fake_votes)
    real_weight = sum(results[name]["confidence"] for name in real_votes)
    total_weight = fake_weight + real_weight
    weight_gap = abs(fake_weight - real_weight)
    CONFIDENCE_GAP_THRESHOLD = 0.15

    if total_weight > 0 and (weight_gap / total_weight) < CONFIDENCE_GAP_THRESHOLD and disagreement:
        final_verdict = "inconclusive"
        final_confidence = 0.5
        disagreement_details = (
            f"Agents disagree with a narrow confidence margin ({', '.join(fake_votes)} weight "
            f"{fake_weight:.2f} vs. {', '.join(real_votes)} weight {real_weight:.2f}). "
            f"This result is inconclusive and should be escalated for human review."
        )
    elif fake_weight > real_weight:
        final_verdict = "fake"
        final_confidence = fake_weight / total_weight if total_weight > 0 else 0.5
        disagreement_details = (
            f"Agents disagree: {', '.join(fake_votes)} flagged 'fake' (weight {fake_weight:.2f}), "
            f"{', '.join(real_votes)} flagged 'real' (weight {real_weight:.2f})."
        ) if disagreement else "All agents agree."
    else:
        final_verdict = "real"
        final_confidence = real_weight / total_weight if total_weight > 0 else 0.5
        disagreement_details = (
            f"Agents disagree: {', '.join(fake_votes)} flagged 'fake' (weight {fake_weight:.2f}), "
            f"{', '.join(real_votes)} flagged 'real' (weight {real_weight:.2f})."
        ) if disagreement else "All agents agree."

    return {
        "agent_summary": results, "final_verdict": final_verdict, "final_confidence": final_confidence,
        "disagreement_detected": disagreement, "disagreement_details": disagreement_details,
    }


def route_after_detect(state: OrchestratorState) -> list:
    if state["input_type"] == "video":
        return ["audio_node", "video_node", "structural_node"]
    elif state["input_type"] == "audio":
        return ["audio_node"]
    elif state["input_type"] == "image":
        return ["image_node"]
    else:
        return ["fusion_node"]


def build_orchestrator_graph(models: dict, device):
    graph = StateGraph(OrchestratorState)
    graph.add_node("detect_and_prepare", lambda s: detect_and_prepare_node(s, models, device))
    graph.add_node("audio_node", lambda s: audio_node(s, models, device))
    graph.add_node("video_node", lambda s: video_node(s, models, device))
    graph.add_node("structural_node", lambda s: structural_node(s, models, device))
    graph.add_node("image_node", lambda s: image_node(s, models, device))
    graph.add_node("fusion_node", fusion_node)
    graph.set_entry_point("detect_and_prepare")
    graph.add_conditional_edges("detect_and_prepare", route_after_detect)
    for node_name in ["audio_node", "video_node", "structural_node", "image_node"]:
        graph.add_edge(node_name, "fusion_node")
    graph.add_edge("fusion_node", END)
    return graph.compile()


def run_orchestrator(file_path, models, device):
    app = build_orchestrator_graph(models, device)
    result = app.invoke({"input_path": file_path})
    return result
