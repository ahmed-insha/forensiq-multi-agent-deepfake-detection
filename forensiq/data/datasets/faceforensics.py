"""
Frame extraction for FaceForensics++.

Samples a fixed number of evenly spaced frames from each video and saves
them as JPEGs organized into real/fake subfolders, compatible with
torchvision.datasets.ImageFolder.

Output filenames are prefixed with the source/method name to prevent
collisions, since FaceForensics++ reuses identical source/target video
filenames across all six manipulation methods.
"""
import os
from pathlib import Path

import cv2
from tqdm import tqdm

MANIPULATED_METHODS = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]
ORIGINAL_SOURCES = ["actors", "youtube"]


def extract_frames_from_video(video_path, output_dir, filename_prefix, num_frames=32):
    """
    Extracts num_frames evenly spaced frames from a single video and saves
    them as JPEGs in output_dir. filename_prefix disambiguates output files
    across categories that may share the same underlying video filename.
    Returns the number of frames written.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return 0

    frame_indices = set(
        int(i * total_frames / num_frames) for i in range(num_frames)
    )

    written = 0
    current_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if current_idx in frame_indices:
            ret, frame = cap.retrieve()
            if ret:
                out_name = f"{filename_prefix}_{video_name}_{current_idx:05d}.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, frame)
                written += 1
        current_idx += 1

    cap.release()
    return written


def build_frame_dataset(faceforensics_dir, output_dir, num_frames=32, compression="c23"):
    """
    Walks the standard FaceForensics++ directory layout and extracts frames
    for every video into:

        output_dir/
            real/
            fake/

    Returns a list of per-video extraction records for logging/verification.
    """
    real_out = os.path.join(output_dir, "real")
    fake_out = os.path.join(output_dir, "fake")

    log = []

    for source in ORIGINAL_SOURCES:
        video_dir = os.path.join(
            faceforensics_dir, "original_sequences", source, compression, "videos"
        )
        if not os.path.isdir(video_dir):
            continue
        videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        for video_file in tqdm(videos, desc=f"real/{source}"):
            video_path = os.path.join(video_dir, video_file)
            n = extract_frames_from_video(video_path, real_out, source, num_frames)
            log.append({"video": video_file, "label": "real", "source": source, "frames_written": n})

    for method in MANIPULATED_METHODS:
        video_dir = os.path.join(
            faceforensics_dir, "manipulated_sequences", method, compression, "videos"
        )
        if not os.path.isdir(video_dir):
            continue
        videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        for video_file in tqdm(videos, desc=f"fake/{method}"):
            video_path = os.path.join(video_dir, video_file)
            n = extract_frames_from_video(video_path, fake_out, method, num_frames)
            log.append({"video": video_file, "label": "fake", "source": method, "frames_written": n})

    return log
