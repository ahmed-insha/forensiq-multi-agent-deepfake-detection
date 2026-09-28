"""
Builds the Phase B fine-tuning dataset for the Video Agent.

Each video contributes frames for TWO conditions: clean, and one randomly
selected degradation preset. This guarantees clean data remains 50% of
the training mix, addressing underrepresentation identified after the
first Phase B attempt, where clean data made up only ~20% of training
data and clean-condition performance regressed as a result.
"""
import os
import random

from forensiq.data.degradation import apply_preset
from forensiq.data.datasets.faceforensics import (
    extract_frames_from_video,
    MANIPULATED_METHODS,
    ORIGINAL_SOURCES,
)


def select_finetune_videos(faceforensics_dir, exclude_paths, n_per_category=50, compression="c23", seed=456):
    random.seed(seed)
    exclude_set = set(exclude_paths)
    selected = []

    for source in ORIGINAL_SOURCES:
        video_dir = os.path.join(faceforensics_dir, "original_sequences", source, compression, "videos")
        videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        candidates = [v for v in videos if os.path.join(video_dir, v) not in exclude_set]
        sampled = random.sample(candidates, min(n_per_category, len(candidates)))
        for v in sampled:
            selected.append({"path": os.path.join(video_dir, v), "label": "real", "source": source})

    for method in MANIPULATED_METHODS:
        video_dir = os.path.join(faceforensics_dir, "manipulated_sequences", method, compression, "videos")
        videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        candidates = [v for v in videos if os.path.join(video_dir, v) not in exclude_set]
        sampled = random.sample(candidates, min(n_per_category, len(candidates)))
        for v in sampled:
            selected.append({"path": os.path.join(video_dir, v), "label": "fake", "source": method})

    return selected


def build_phase_b_dataset(
    videos,
    output_dir,
    degraded_presets=("mild_compress", "heavy_compress", "social_media", "noisy_blurry", "compound_severe"),
    num_frames=16,
    seed=456,
):
    random.seed(seed)
    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    tmp_dir = "/content/tmp_phase_b_degraded"
    os.makedirs(tmp_dir, exist_ok=True)

    log = []

    for i, video_info in enumerate(videos):
        video_path = video_info["path"]
        label = video_info["label"]
        source = video_info["source"]
        target_dir = real_dir if label == "real" else fake_dir

        # always extract a clean set of frames
        n_clean = extract_frames_from_video(video_path, target_dir, f"{source}_clean", num_frames)

        # plus one randomly chosen degraded condition
        condition = random.choice(degraded_presets)
        degraded_path = apply_preset(video_path, tmp_dir, condition, is_audio=False)
        n_degraded = extract_frames_from_video(degraded_path, target_dir, f"{source}_{condition}", num_frames)
        os.remove(degraded_path)

        log.append({
            "video": video_path, "label": label,
            "clean_frames": n_clean, "degraded_condition": condition, "degraded_frames": n_degraded,
        })

        if (i + 1) % 50 == 0:
            print(f"processed {i + 1}/{len(videos)} videos")

    return log
