"""
Clean vs. degraded evaluation for the Video Agent.

Selects a held-out set of source videos, applies each degradation preset
to them using forensiq.data.degradation, extracts frames from both the
clean and degraded versions, and evaluates a trained model on each
condition. Produces the comparison table showing how much performance
drops under simulated real-world degradation.
"""
import os
import random
import shutil

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights

from forensiq.data.degradation import apply_preset, PRESETS
from forensiq.data.datasets.faceforensics import (
    extract_frames_from_video,
    MANIPULATED_METHODS,
    ORIGINAL_SOURCES,
)
from forensiq.evaluation.metrics import evaluate_model


def select_holdout_videos(faceforensics_dir, n_per_category=10, compression="c23", seed=123):
    """
    Selects a fixed random sample of videos per category, held out
    specifically for degraded-vs-clean evaluation, separate from the
    frame-level train/validation split used in Phase A training.
    """
    random.seed(seed)
    holdout = []

    for source in ORIGINAL_SOURCES:
        video_dir = os.path.join(faceforensics_dir, "original_sequences", source, compression, "videos")
        videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        sampled = random.sample(videos, min(n_per_category, len(videos)))
        for v in sampled:
            holdout.append({"path": os.path.join(video_dir, v), "label": "real", "source": source})

    for method in MANIPULATED_METHODS:
        video_dir = os.path.join(faceforensics_dir, "manipulated_sequences", method, compression, "videos")
        videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        sampled = random.sample(videos, min(n_per_category, len(videos)))
        for v in sampled:
            holdout.append({"path": os.path.join(video_dir, v), "label": "fake", "source": method})

    return holdout


def build_condition_frames(holdout_videos, output_root, preset_name=None, num_frames=16):
    """
    Extracts frames for one evaluation condition (clean, or a named
    degradation preset). If preset_name is given, each video is degraded
    first, then frames are extracted from the degraded copy.
    """
    real_dir = os.path.join(output_root, "real")
    fake_dir = os.path.join(output_root, "fake")

    for video_info in holdout_videos:
        video_path = video_info["path"]
        label = video_info["label"]
        source = video_info["source"]
        target_dir = real_dir if label == "real" else fake_dir

        if preset_name and preset_name != "clean":
            tmp_dir = "/content/tmp_degraded"
            os.makedirs(tmp_dir, exist_ok=True)
            degraded_path = apply_preset(video_path, tmp_dir, preset_name, is_audio=False)
            extract_frames_from_video(degraded_path, target_dir, f"{source}_{preset_name}", num_frames)
            os.remove(degraded_path)
        else:
            extract_frames_from_video(video_path, target_dir, source, num_frames)


def run_degradation_evaluation(
    model,
    holdout_videos,
    presets_to_test,
    device,
    tmp_root="/content/degradation_eval",
    batch_size=32,
):
    """
    Runs the full clean-vs-degraded comparison and returns a results
    DataFrame, one row per condition, with the same metrics used for
    Phase A (accuracy, precision, recall, F1, AUC-ROC).
    """
    transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    results = []

    conditions = ["clean"] + list(presets_to_test)

    for condition in conditions:
        condition_dir = os.path.join(tmp_root, condition)
        shutil.rmtree(condition_dir, ignore_errors=True)
        os.makedirs(os.path.join(condition_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(condition_dir, "fake"), exist_ok=True)

        print(f"building frames for condition: {condition}")
        build_condition_frames(holdout_videos, condition_dir, preset_name=condition)

        dataset = ImageFolder(condition_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        metrics = evaluate_model(model, loader, device)
        metrics["condition"] = condition
        results.append(metrics)
        print(f"  {condition}: {metrics}")

    return pd.DataFrame(results)
