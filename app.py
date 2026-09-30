"""
app.py

ForensiQ Streamlit application -- deployment-ready version. Model
checkpoints are downloaded from Hugging Face Hub on first run (and
cached locally afterward), rather than read from a personal Google
Drive path, so this runs on Streamlit Community Cloud, Hugging Face
Spaces, or any other host, not only inside the author's own Colab
session.
"""
import os
import sys
import tempfile
import shutil
import subprocess
import types as pytypes
import importlib.util

import streamlit as st
import torch
import cv2
from huggingface_hub import hf_hub_download

import packaging as _packaging
if "pkg_resources" not in sys.modules:
    fake_pkg_resources = pytypes.ModuleType("pkg_resources")
    fake_pkg_resources.packaging = _packaging
    sys.modules["pkg_resources"] = fake_pkg_resources

# ============================================================
# Hugging Face Hub checkpoint repo -- set once, used everywhere below
# ============================================================
HF_CHECKPOINT_REPO = "insha142/forensiq-checkpoints"

def get_checkpoint(filename):
    """Downloads a checkpoint from Hugging Face Hub, caching it locally
    after the first download so app restarts don't re-download."""
    return hf_hub_download(repo_id=HF_CHECKPOINT_REPO, filename=filename)


def _load_mvssnet_module():
    repo_path = "/tmp/mvssnet_repo"
    models_dir = f"{repo_path}/models"
    mvssnet_file = f"{models_dir}/mvssnet.py"

    if not os.path.exists(mvssnet_file):
        shutil.rmtree(repo_path, ignore_errors=True)
        subprocess.run(["git", "clone", "https://github.com/dong03/MVSS-Net.git", repo_path])
    if not os.path.exists(mvssnet_file):
        raise RuntimeError("MVSS-Net clone failed - mvssnet.py missing")

    package = pytypes.ModuleType("mvssnet_models_unique")
    package.__path__ = [models_dir]
    sys.modules["mvssnet_models_unique"] = package
    spec = importlib.util.spec_from_file_location(
        "mvssnet_models_unique.mvssnet", mvssnet_file, submodule_search_locations=[models_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mvssnet_models_unique.mvssnet"] = mod
    spec.loader.exec_module(mod)

    if repo_path not in sys.path:
        sys.path.append(repo_path)

    return mod


@st.cache_resource
def load_all_agents():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Audio Agent ---
    from forensiq.models.audio_model import Wav2Vec2SpoofClassifier
    audio_model = Wav2Vec2SpoofClassifier(freeze_feature_extractor=True)
    audio_ckpt = get_checkpoint("audio_phase_c_best.pt")
    audio_model.load_state_dict(torch.load(audio_ckpt, map_location="cpu"))
    audio_model.to(device); audio_model.eval()

    # --- Video Agent ---
    from forensiq.models.video_model import EfficientNetFrameClassifier
    video_model = EfficientNetFrameClassifier(freeze_backbone=False)
    video_ckpt = get_checkpoint("video_phase_a_best.pt")
    video_model.load_state_dict(torch.load(video_ckpt, map_location="cpu"))
    video_model.to(device); video_model.eval()

    # --- Structural Agent (MVSS-Net) ---

    if "albumentations" not in sys.modules:
        fake_albumentations = pytypes.ModuleType("albumentations")
        fake_albumentations.pytorch = pytypes.ModuleType("albumentations.pytorch")
        fake_albumentations.pytorch.functional = pytypes.ModuleType("albumentations.pytorch.functional")
        sys.modules["albumentations"] = fake_albumentations
        sys.modules["albumentations.pytorch"] = fake_albumentations.pytorch
        sys.modules["albumentations.pytorch.functional"] = fake_albumentations.pytorch.functional
    
    mvssnet_module = _load_mvssnet_module()
    get_mvss = mvssnet_module.get_mvss

    import numpy as np
    if not hasattr(np, 'sctypes'):
        np.sctypes = {
            'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128],
            'others': [bool, object, bytes, str, np.void],
        }

    mvssnet_model = get_mvss(backbone='resnet50', pretrained_base=True, nclass=1, sobel=True, constrain=True, n_input=3)
    mvssnet_ckpt = get_checkpoint("mvssnet_pretrained/mvssnet_casia.pt")
    mvssnet_model.load_state_dict(torch.load(mvssnet_ckpt, map_location='cpu'), strict=True)
    mvssnet_model.to(device); mvssnet_model.eval()

    from forensiq.models.interframe_detector import (
        compute_frame_features, compute_self_similarity_matrix, find_duplicated_blocks_adaptive,
    )

    # --- Image Agent (UnivFD) -- weights come from UnivFD's own repo, not our checkpoint store ---
    shutil.rmtree("/tmp/univfd_repo", ignore_errors=True)
    subprocess.run(["git", "clone", "https://github.com/WisconsinAIVision/UniversalFakeDetect.git", "/tmp/univfd_repo"])
    try:
        import ftfy, regex
    except ImportError:
        subprocess.run(["pip", "install", "ftfy", "regex", "--quiet"])

    if "/tmp/univfd_repo" not in sys.path:
        sys.path.append("/tmp/univfd_repo")


    import forensiq.models.image_model as image_model_module
    image_model_module.CLIP_CACHE_DIR = "/tmp/clip_cache"
    os.makedirs("/tmp/clip_cache", exist_ok=True)
    
    from forensiq.models.image_model import load_univfd_model
    univfd_model = load_univfd_model("/tmp/univfd_repo/pretrained_weights/fc_weights.pth", device=device)

    from torchvision import transforms
    from torchvision.models import EfficientNet_B0_Weights

    models = {
        "audio": audio_model,
        "video": video_model,
        "video_transform": EfficientNet_B0_Weights.DEFAULT.transforms(),
        "structural_mvssnet": mvssnet_model,
        "structural_interframe_funcs": {
            "compute_frame_features": compute_frame_features,
            "compute_self_similarity_matrix": compute_self_similarity_matrix,
            "find_duplicated_blocks_adaptive": find_duplicated_blocks_adaptive,
        },
        "image": univfd_model,
        "image_transform": transforms.Compose([
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]),
    }
    return models, device


st.set_page_config(page_title="ForensiQ", layout="wide")
st.title("ForensiQ — Multi-Agent Forensic Media Analysis")
st.caption("Upload a video, audio, or image file for multi-modal deepfake/tampering analysis.")

with st.sidebar:
    st.subheader("Settings")
    gemini_api_key = st.text_input("Gemini API key (for Image Agent fusion + narration)", type="password",
                                     help="Get a free key at aistudio.google.com/apikey. Without a key, "
                                          "the Image Agent falls back to a weaker, uncalibrated baseline.")
    enable_narration = st.checkbox("Generate written forensic narration", value=bool(gemini_api_key))
    if gemini_api_key:
        st.success("Image Agent: using validated fusion (calibrated UnivFD + Gemini reasoning)")
    else:
        st.warning("Image Agent: using plain UnivFD only (add a key above for the validated +17-point improvement)")

with st.spinner("Loading specialist agents (first run only downloads checkpoints, may take a few minutes)..."):
    models, device = load_all_agents()
    models["gemini_api_key"] = gemini_api_key if gemini_api_key else None

from forensiq.agents.orchestrator import run_orchestrator
from forensiq.agents.report_summary import build_clear_summary
from forensiq.agents.localization import (
    sliding_window_audio_scores, score_video_frames_over_time, build_combined_timeline,
    summarize_timeline, extract_audio_evidence_clip, extract_video_evidence_clip,
)

uploaded_file = st.file_uploader("Upload a file", type=["mp4", "mov", "flac", "wav", "mp3", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.divider()

    if suffix.lower() in [".mp4", ".mov"]:
        st.video(temp_path)
    elif suffix.lower() in [".flac", ".wav", ".mp3"]:
        st.audio(temp_path)
    else:
        st.image(temp_path)

    with st.spinner("Running analysis..."):
        result = run_orchestrator(temp_path, models, device)

    verdict = result["final_verdict"]
    confidence = result["final_confidence"]

    if verdict == "fake":
        st.error(f"### Verdict: LIKELY MANIPULATED ({confidence*100:.1f}% confidence)")
    elif verdict == "real":
        st.success(f"### Verdict: LIKELY AUTHENTIC ({confidence*100:.1f}% confidence)")
    elif verdict == "inconclusive":
        st.warning("### Verdict: INCONCLUSIVE — recommend human review")
    else:
        st.warning("### Verdict: UNABLE TO ANALYZE")

    if result.get("disagreement_detected"):
        st.info(f"ℹ️ {result['disagreement_details']}")

    st.subheader("Per-Agent Breakdown")
    if result.get("agent_summary"):
        cols = st.columns(len(result["agent_summary"]))
        for i, (agent_name, agent_result) in enumerate(result["agent_summary"].items()):
            with cols[i]:
                st.metric(label=agent_name.capitalize(), value=agent_result["verdict"].upper(),
                          delta=f"{agent_result['confidence']*100:.1f}% confidence")
                if agent_name == "image" and agent_result.get("explanation_data", {}).get("final_reasoning"):
                    with st.expander("Reasoning"):
                        st.write(agent_result["explanation_data"]["final_reasoning"])

    timeline_summary = None
    clear_summary = None
    sample_frame_path = None

    if result["input_type"] == "video":
        st.subheader("Timeline Analysis")

        audio_out = temp_path + "_audio.wav"
        os.system(f'ffmpeg -y -i "{temp_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_out}" -loglevel error')

        audio_scores = sliding_window_audio_scores(models["audio"], audio_out, device) if os.path.exists(audio_out) else []
        video_scores = score_video_frames_over_time(models["video"], temp_path, device, models["video_transform"], num_samples=20)

        cap = cv2.VideoCapture(temp_path)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()

        timeline = build_combined_timeline(audio_scores, video_scores, duration)
        timeline_summary = summarize_timeline(timeline)
        clear_summary = build_clear_summary(result, timeline, temp_path, duration=duration)

        for line in clear_summary.get("summary_text", []):
            st.write(f"**{line}**")

        if "audio_evidence_window" in clear_summary:
            w = clear_summary["audio_evidence_window"]
            st.write(f"Audio evidence (transition detected at {w['start']:.1f}s):")
            audio_evidence_path = temp_path + "_audio_evidence.wav"
            extract_audio_evidence_clip(temp_path, w["start"], w["end"], audio_evidence_path)
            st.audio(audio_evidence_path)

        if "video_evidence_window" in clear_summary:
            w = clear_summary["video_evidence_window"]
            st.write(f"Video evidence (transition detected at {w['start']:.1f}s):")
            video_evidence_path = temp_path + "_video_evidence.mp4"
            extract_video_evidence_clip(temp_path, w["start"], w["end"], video_evidence_path)
            st.video(video_evidence_path)

        if clear_summary.get("structural_tampering_detected"):
            st.write("**Structural findings:**")
            for line in clear_summary.get("structural_findings_text", []):
                st.write(f"- {line}")

        cap = cv2.VideoCapture(temp_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            sample_frame_path = temp_path + "_frame.jpg"
            cv2.imwrite(sample_frame_path, frame)

    if enable_narration and gemini_api_key:
        from forensiq.agents.narration import generate_narration
        with st.spinner("Generating forensic narration..."):
            try:
                narration = generate_narration(
                    result, api_key=gemini_api_key, timeline_summary=timeline_summary,
                    clear_summary=clear_summary, sample_frame_path=sample_frame_path,
                )
                st.subheader("Forensic Narration")
                st.write(narration)
            except Exception as e:
                st.error(f"Narration generation failed: {e}")
