"""
app.py

ForensiQ Streamlit application -- lazy-loading version. Models are
loaded ONLY when a file of the matching type is actually uploaded,
rather than all four eagerly at startup. This substantially reduces
peak memory: an image-only upload never loads the Audio, Video, or
Structural agents at all. A video upload still needs Audio + Video +
Structural together (per the orchestrator's design), so peak memory
for video is unchanged -- but audio-only and image-only analysis,
likely the most common cases in a demo/viva setting, now run far
lighter.
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

HF_CHECKPOINT_REPO = "insha142/forensiq-checkpoints"

def get_checkpoint(filename):
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

    if "albumentations" not in sys.modules:
        fake_albumentations = pytypes.ModuleType("albumentations")
        fake_albumentations.pytorch = pytypes.ModuleType("albumentations.pytorch")
        fake_albumentations.pytorch.functional = pytypes.ModuleType("albumentations.pytorch.functional")
        sys.modules["albumentations"] = fake_albumentations
        sys.modules["albumentations.pytorch"] = fake_albumentations.pytorch
        sys.modules["albumentations.pytorch.functional"] = fake_albumentations.pytorch.functional

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


# ============================================================
# Per-agent lazy loaders -- each cached independently, so loading
# one doesn't force the others to load too.
# ============================================================

@st.cache_resource
def load_audio_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from forensiq.models.audio_model import Wav2Vec2SpoofClassifier
    model = Wav2Vec2SpoofClassifier(freeze_feature_extractor=True)
    ckpt = get_checkpoint("audio_phase_c_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device); model.eval()
    return model, device


@st.cache_resource
def load_video_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from forensiq.models.video_model import EfficientNetFrameClassifier
    model = EfficientNetFrameClassifier(freeze_backbone=False)
    ckpt = get_checkpoint("video_phase_a_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device); model.eval()
    from torchvision.models import EfficientNet_B0_Weights
    transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    return model, transform, device


@st.cache_resource
def load_structural_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvssnet_module = _load_mvssnet_module()
    get_mvss = mvssnet_module.get_mvss

    import numpy as np
    if not hasattr(np, 'sctypes'):
        np.sctypes = {
            'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128],
            'others': [bool, object, bytes, str, np.void],
        }

    model = get_mvss(backbone='resnet50', pretrained_base=True, nclass=1, sobel=True, constrain=True, n_input=3)
    ckpt = get_checkpoint("mvssnet_pretrained/mvssnet_casia.pt")
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
    model.to(device); model.eval()

    from forensiq.models.interframe_detector import (
        compute_frame_features, compute_self_similarity_matrix, find_duplicated_blocks_adaptive,
    )
    interframe_funcs = {
        "compute_frame_features": compute_frame_features,
        "compute_self_similarity_matrix": compute_self_similarity_matrix,
        "find_duplicated_blocks_adaptive": find_duplicated_blocks_adaptive,
    }
    return model, interframe_funcs, device


@st.cache_resource
def load_image_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shutil.rmtree("/tmp/univfd_repo", ignore_errors=True)
    subprocess.run(["git", "clone", "https://github.com/WisconsinAIVision/UniversalFakeDetect.git", "/tmp/univfd_repo"])
    try:
        import ftfy, regex
    except ImportError:
        subprocess.run(["pip", "install", "ftfy", "regex", "--quiet"])
    if "/tmp/univfd_repo" not in sys.path:
        sys.path.append("/tmp/univfd_repo")

    if "pkg_resources" not in sys.modules:
        import packaging as _packaging
        fake_pkg_resources = pytypes.ModuleType("pkg_resources")
        fake_pkg_resources.packaging = _packaging
        sys.modules["pkg_resources"] = fake_pkg_resources

    import forensiq.models.image_model as image_model_module
    image_model_module.CLIP_CACHE_DIR = "/tmp/clip_cache"
    os.makedirs("/tmp/clip_cache", exist_ok=True)

    from forensiq.models.image_model import load_univfd_model
    model = load_univfd_model("/tmp/univfd_repo/pretrained_weights/fc_weights.pth", device=device)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return model, transform, device


st.set_page_config(page_title="ForensiQ", layout="wide")
st.title("ForensiQ — Multi-Agent Forensic Media Analysis")
st.caption("Upload a video, audio, or image file for multi-modal deepfake/tampering analysis.")
st.caption("Specialist agents load on demand, based on your file type — only what's needed for your upload is loaded into memory.")

with st.sidebar:
    st.subheader("Settings")
    gemini_api_key = st.text_input("Gemini API key (for Image Agent fusion + narration)", type="password",
                                     help="Get a free key at aistudio.google.com/apikey. Without a key, "
                                          "the Image Agent falls back to a weaker, uncalibrated baseline.")
    enable_narration = st.checkbox("Generate written forensic narration", value=bool(gemini_api_key))

from forensiq.agents.agents import run_audio_agent, run_video_agent, run_structural_agent, run_image_agent
from forensiq.agents.localization import (
    sliding_window_audio_scores, score_video_frames_over_time, build_combined_timeline,
    summarize_timeline, extract_audio_evidence_clip, extract_video_evidence_clip,
)
from forensiq.agents.report_summary import build_clear_summary

uploaded_file = st.file_uploader("Upload a file", type=["mp4", "mov", "flac", "wav", "mp3", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.divider()
    is_video = suffix.lower() in [".mp4", ".mov"]
    is_audio = suffix.lower() in [".flac", ".wav", ".mp3"]
    is_image = suffix.lower() in [".jpg", ".jpeg", ".png"]

    if is_video:
        st.video(temp_path)
    elif is_audio:
        st.audio(temp_path)
    else:
        st.image(temp_path)

    agent_results = {}
    timeline_summary = None
    clear_summary = None
    sample_frame_path = None

    if is_image:
        with st.spinner("Loading Image Agent..."):
            univfd_model, image_transform, device = load_image_agent()
        with st.spinner("Analyzing..."):
            result = run_image_agent(univfd_model, temp_path, device, image_transform, gemini_api_key=gemini_api_key)
        agent_results["image"] = result.__dict__

    elif is_audio:
        with st.spinner("Loading Audio Agent..."):
            audio_model, device = load_audio_agent()
        with st.spinner("Analyzing..."):
            result = run_audio_agent(audio_model, temp_path, device)
        agent_results["audio"] = result.__dict__

    elif is_video:
        with st.spinner("Loading Audio, Video, and Structural Agents (video needs all three)..."):
            audio_model, audio_device = load_audio_agent()
            video_model, video_transform, video_device = load_video_agent()
            mvssnet_model, interframe_funcs, struct_device = load_structural_agent()

        audio_out = temp_path + "_audio.wav"
        os.system(f'ffmpeg -y -i "{temp_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_out}" -loglevel error')

        with st.spinner("Analyzing..."):
            if os.path.exists(audio_out):
                audio_result = run_audio_agent(audio_model, audio_out, audio_device)
                agent_results["audio"] = audio_result.__dict__

            video_result = run_video_agent(video_model, [], video_device, video_transform, full_video_path=temp_path)
            agent_results["video"] = video_result.__dict__

            cap = cv2.VideoCapture(temp_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                sample_frame_path = temp_path + "_frame.jpg"
                cv2.imwrite(sample_frame_path, frame)
                struct_result = run_structural_agent(mvssnet_model, interframe_funcs, sample_frame_path, temp_path, struct_device)
                agent_results["structural"] = struct_result.__dict__

            audio_scores = sliding_window_audio_scores(audio_model, audio_out, audio_device) if os.path.exists(audio_out) else []
            video_scores = score_video_frames_over_time(video_model, temp_path, video_device, video_transform, num_samples=20)
            cap = cv2.VideoCapture(temp_path)
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
            cap.release()
            timeline = build_combined_timeline(audio_scores, video_scores, duration)
            timeline_summary = summarize_timeline(timeline)

    # --- Simple confidence-weighted fusion across whatever agents actually ran ---
    fake_votes = [n for n, r in agent_results.items() if r["verdict"] == "fake"]
    real_votes = [n for n, r in agent_results.items() if r["verdict"] == "real"]
    fake_weight = sum(agent_results[n]["confidence"] for n in fake_votes)
    real_weight = sum(agent_results[n]["confidence"] for n in real_votes)
    total_weight = fake_weight + real_weight
    disagreement = len(fake_votes) > 0 and len(real_votes) > 0

    if total_weight == 0:
        final_verdict, final_confidence = "unknown", 0.0
    elif disagreement and total_weight > 0 and abs(fake_weight - real_weight) / total_weight < 0.15:
        final_verdict, final_confidence = "inconclusive", 0.5
    elif fake_weight > real_weight:
        final_verdict, final_confidence = "fake", fake_weight / total_weight
    else:
        final_verdict, final_confidence = "real", real_weight / total_weight

    if final_verdict == "fake":
        st.error(f"### Verdict: LIKELY MANIPULATED ({final_confidence*100:.1f}% confidence)")
    elif final_verdict == "real":
        st.success(f"### Verdict: LIKELY AUTHENTIC ({final_confidence*100:.1f}% confidence)")
    elif final_verdict == "inconclusive":
        st.warning("### Verdict: INCONCLUSIVE — recommend human review")
    else:
        st.warning("### Verdict: UNABLE TO ANALYZE")

    if disagreement:
        st.info(f"ℹ️ Agents disagree: {', '.join(fake_votes)} flagged fake, {', '.join(real_votes)} flagged real.")

    st.subheader("Per-Agent Breakdown")
    if agent_results:
        cols = st.columns(len(agent_results))
        for i, (agent_name, r) in enumerate(agent_results.items()):
            with cols[i]:
                st.metric(label=agent_name.capitalize(), value=r["verdict"].upper(),
                          delta=f"{r['confidence']*100:.1f}% confidence")
                if agent_name == "image" and r.get("explanation_data", {}).get("final_reasoning"):
                    with st.expander("Reasoning"):
                        st.write(r["explanation_data"]["final_reasoning"])

    if is_video and timeline_summary:
        st.subheader("Timeline Analysis")
        clear_summary = build_clear_summary({"agent_summary": agent_results, "final_verdict": final_verdict,
                                              "final_confidence": final_confidence}, timeline, temp_path, duration=duration)
        for line in clear_summary.get("summary_text", []):
            st.write(f"**{line}**")
        if "audio_evidence_window" in clear_summary:
            w = clear_summary["audio_evidence_window"]
            audio_evidence_path = temp_path + "_audio_evidence.wav"
            extract_audio_evidence_clip(temp_path, w["start"], w["end"], audio_evidence_path)
            st.audio(audio_evidence_path)
        if "video_evidence_window" in clear_summary:
            w = clear_summary["video_evidence_window"]
            video_evidence_path = temp_path + "_video_evidence.mp4"
            extract_video_evidence_clip(temp_path, w["start"], w["end"], video_evidence_path)
            st.video(video_evidence_path)

    if enable_narration and gemini_api_key:
        from forensiq.agents.narration import generate_narration
        with st.spinner("Generating forensic narration..."):
            try:
                narration = generate_narration(
                    {"agent_summary": agent_results, "final_verdict": final_verdict, "final_confidence": final_confidence},
                    api_key=gemini_api_key, timeline_summary=timeline_summary,
                    clear_summary=clear_summary, sample_frame_path=sample_frame_path,
                )
                st.subheader("Forensic Narration")
                st.write(narration)
            except Exception as e:
                st.error(f"Narration generation failed: {e}")
