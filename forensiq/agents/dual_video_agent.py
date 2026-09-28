"""
forensiq/agents/dual_video_agent.py

Dual-pathway video scoring: runs the existing Video Agent (EfficientNet,
trained on FaceForensics++'s classic face-swap/reenactment techniques)
AND the existing Image Agent (UnivFD/CLIP, trained for cross-generator
generalization on AI-generated imagery) on the same video frames,
combining both via confidence-weighted fusion.

Motivation: zero-shot testing showed the EfficientNet-based Video Agent
detected 0 of 8 diffusion-based fakes from DigiFakeAV (a modern,
diffusion-based digital-human dataset), while correctly identifying
real videos -- a clean demonstration that a detector trained on older
generation techniques does not transfer to newer ones. Since a video
frame from a diffusion-generated video is, from a detection standpoint,
a diffusion-generated image, the already-trained Image Agent (UnivFD)
is reused here as a second pathway, rather than integrating or
training a new video-specific model under time constraints.
"""
import cv2
import torch
from PIL import Image


def score_frame_dual(frame_rgb, video_model, video_transform, image_model, image_transform, device):
    """
    Scores a single frame with BOTH models, returning each pathway's
    fake probability separately, plus a combined confidence-weighted
    score (same fusion principle as the orchestrator's agent voting).
    """
    img = Image.fromarray(frame_rgb)

    # Pathway 1: EfficientNet, tuned for classic face-swap/reenactment
    video_tensor = video_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        video_logits = video_model(video_tensor)
        video_fake_prob = torch.softmax(video_logits, dim=1)[0, 0].item()  # class 0 = fake

    # Pathway 2: UnivFD/CLIP, tuned for cross-generator generative artifacts
    image_tensor = image_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_logit = image_model(image_tensor)
        image_fake_prob = torch.sigmoid(image_logit).item()

    # Confidence-weighted combination: whichever pathway is more
    # confident contributes more to the combined score, rather than a
    # flat average that could dilute a strong detection from either side.
    video_conf = abs(video_fake_prob - 0.5)
    image_conf = abs(image_fake_prob - 0.5)
    total_conf = video_conf + image_conf
    if total_conf > 0:
        combined_fake_prob = (video_fake_prob * video_conf + image_fake_prob * image_conf) / total_conf
    else:
        combined_fake_prob = (video_fake_prob + image_fake_prob) / 2

    return {
        "video_pathway_fake_prob": video_fake_prob,
        "image_pathway_fake_prob": image_fake_prob,
        "combined_fake_prob": combined_fake_prob,
    }


def score_video_dual_pathway(video_path, video_model, video_transform, image_model, image_transform, device, num_samples=10):
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    frame_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
    results = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scores = score_frame_dual(frame_rgb, video_model, video_transform, image_model, image_transform, device)
        results.append(scores)

    cap.release()
    if not results:
        return None

    return {
        "avg_video_pathway": sum(r["video_pathway_fake_prob"] for r in results) / len(results),
        "avg_image_pathway": sum(r["image_pathway_fake_prob"] for r in results) / len(results),
        "avg_combined": sum(r["combined_fake_prob"] for r in results) / len(results),
    }
