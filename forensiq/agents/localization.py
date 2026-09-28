
"""
forensiq/agents/localization.py

Temporal localization for the Audio and Video agents, and a combined
timeline that synchronizes both onto shared time bins -- directly
supporting the project's core requirement: identifying WHICH part of a
file is manipulated, and whether it is the audio, the video, or both,
rather than only a single whole-file verdict.
"""
import os
import subprocess

import cv2
import numpy as np
import soundfile as sf
import torch
from PIL import Image

from forensiq.data.datasets.asvspoof import pad


# ------------------------------------------------------------------
# Audio localization (unchanged from earlier work)
# ------------------------------------------------------------------

def sliding_window_audio_scores(model, audio_path, device, window_sec=4.0, hop_sec=2.0, sr=16000):
    X, file_sr = sf.read(audio_path)
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    scores = []
    total_samples = len(X)

    if total_samples <= window_samples:
        x_padded = pad(X, window_samples)
        x_inp = torch.tensor(x_padded, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x_inp), dim=1)[0]
        scores.append({"start_time": 0.0, "end_time": total_samples / sr, "fake_prob": probs[0].item()})
        return scores

    for start in range(0, total_samples - window_samples + 1, hop_samples):
        window = X[start:start + window_samples]
        x_inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x_inp), dim=1)[0]
        scores.append({
            "start_time": start / sr, "end_time": (start + window_samples) / sr,
            "fake_prob": probs[0].item(),
        })

    return scores


def extract_audio_evidence_clip(source_path, start_time, end_time, output_path):
    duration = end_time - start_time
    cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", source_path, "-t", str(duration), "-vn", output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def extract_video_evidence_clip(source_path, start_time, end_time, output_path):
    duration = end_time - start_time
    cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", source_path, "-t", str(duration), "-c", "copy", output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def get_most_suspicious_segment(scores, threshold=0.5):
    fake_segments = [s for s in scores if s["fake_prob"] > threshold]
    if not fake_segments:
        return None
    return max(fake_segments, key=lambda s: s["fake_prob"])


# ------------------------------------------------------------------
# Video localization -- NEW: samples across the FULL video duration,
# not just the first N frames, so real timestamps can be assigned.
# ------------------------------------------------------------------

def score_video_frames_over_time(model, video_path, device, transform, num_samples=20):
    """
    Samples num_samples frames evenly spread across the ENTIRE video
    duration (not just the beginning), scores each with the Video
    Agent, and returns a list of {timestamp, fake_prob} -- the video
    equivalent of sliding_window_audio_scores.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    frame_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)

    model.eval()
    scores = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            fake_prob = probs[0].item()  # class 0 = fake, per Video Agent's class_to_idx

        timestamp = idx / fps
        scores.append({"timestamp": timestamp, "fake_prob": fake_prob})

    cap.release()
    return scores


# ------------------------------------------------------------------
# Combined timeline -- resamples both agents' scores onto shared time
# bins and classifies each bin into one of four states, directly
# supporting the project's core "which modality, which part" question.
# ------------------------------------------------------------------

def build_combined_timeline(audio_scores, video_scores, duration, bin_size=1.0, threshold=0.5):
    """
    Resamples audio (time-range scores) and video (point-in-time
    scores) onto common time bins of bin_size seconds, and classifies
    each bin as one of:
      "both_real"             - neither agent flags this segment
      "both_fake"             - both agents flag this segment
      "video_fake_audio_real" - visual manipulation only
      "audio_fake_video_real" - audio manipulation only

    This is the direct implementation of the project's stated goal:
    localizing which SPECIFIC part of a file is manipulated, and in
    which modality, rather than a single whole-file verdict.
    """
    num_bins = max(1, int(np.ceil(duration / bin_size)))
    timeline = []

    for i in range(num_bins):
        bin_start = i * bin_size
        bin_end = min((i + 1) * bin_size, duration)
        bin_mid = (bin_start + bin_end) / 2

        # Audio: average all audio window scores that overlap this bin.
        overlapping_audio = [
            s["fake_prob"] for s in audio_scores
            if s["start_time"] < bin_end and s["end_time"] > bin_start
        ]
        audio_prob = float(np.mean(overlapping_audio)) if overlapping_audio else None

        # Video: average all sampled frame scores that fall within this bin.
        overlapping_video = [
            s["fake_prob"] for s in video_scores
            if bin_start <= s["timestamp"] < bin_end
        ]
        video_prob = float(np.mean(overlapping_video)) if overlapping_video else None

        audio_fake = audio_prob is not None and audio_prob > threshold
        video_fake = video_prob is not None and video_prob > threshold

        if video_fake and audio_fake:
            status = "both_fake"
        elif video_fake and not audio_fake:
            status = "video_fake_audio_real"
        elif audio_fake and not video_fake:
            status = "audio_fake_video_real"
        else:
            status = "both_real"

        timeline.append({
            "bin_start": bin_start, "bin_end": bin_end,
            "audio_fake_prob": audio_prob, "video_fake_prob": video_prob,
            "status": status,
        })

    return timeline


def summarize_timeline(timeline):
    """Returns the time ranges (merged consecutive bins) for each disagreement type."""
    summary = {"both_fake": [], "video_fake_audio_real": [], "audio_fake_video_real": []}

    current_status = None
    current_start = None

    for bin_data in timeline:
        status = bin_data["status"]
        if status in summary:
            if status != current_status:
                if current_status in summary and current_start is not None:
                    summary[current_status].append((current_start, bin_data["bin_start"]))
                current_status = status
                current_start = bin_data["bin_start"]
        else:
            if current_status in summary and current_start is not None:
                summary[current_status].append((current_start, bin_data["bin_start"]))
            current_status = None
            current_start = None

    if current_status in summary and current_start is not None:
        summary[current_status].append((current_start, timeline[-1]["bin_end"]))

    return summary
