"""
HTVD Intraframe dataset extraction: Splice, Clone, Inpaint.

Unlike FaceForensics/ASVspoof, HTVD ships each sample with multiple
pre-encoded variants (GOP size, frame type, CRF), matching the columns
of the dataset's own description spreadsheet. This is used deliberately
here to both increase usable training data and provide a dataset-native
compression-severity dimension, distinct from the degradation.py module
used for the Video Agent.

Each sample's mask/ folder contains one mask per frame, frame-index
aligned with the corresponding tampered video, supporting pixel-level
manipulation localization rather than only binary classification.
"""
import os
import re

import cv2
import numpy as np
from tqdm import tqdm

CATEGORIES = ["Splice", "Clone", "Inpaint"]
PREFIX_MAP = {"Splice": "S", "Clone": "C", "Inpaint": "I"}


def list_encoding_variants(sample_dir):
    """Returns all .mp4 files in a sample's Tamp/Auth folder."""
    return sorted(f for f in os.listdir(sample_dir) if f.endswith(".mp4"))


def extract_tampered_frames_with_masks(
    tamp_video_path, mask_dir, output_frames_dir, output_masks_dir, variant_tag
):
    """
    Extracts every frame from a tampered video and copies its matching
    per-frame mask, preserving frame-index alignment.
    """
    os.makedirs(output_frames_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    mask_files = sorted(os.listdir(mask_dir))
    cap = cv2.VideoCapture(str(tamp_video_path))

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(mask_files):
            break

        frame_name = f"{variant_tag}_{frame_idx:05d}.jpg"
        cv2.imwrite(os.path.join(output_frames_dir, frame_name), frame)

        mask_path = os.path.join(mask_dir, mask_files[frame_idx])
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(output_masks_dir, frame_name), mask_img)

        frame_idx += 1
        written += 1

    cap.release()
    return written


def extract_authentic_frames(auth_video_path, output_frames_dir, output_masks_dir, variant_tag):
    """
    Extracts frames from an authentic video, paired with all-zero
    (fully authentic, no tampered pixels) masks.
    """
    os.makedirs(output_frames_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(auth_video_path))
    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"{variant_tag}_{frame_idx:05d}.jpg"
        cv2.imwrite(os.path.join(output_frames_dir, frame_name), frame)

        blank_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.imwrite(os.path.join(output_masks_dir, frame_name), blank_mask)

        frame_idx += 1
        written += 1

    cap.release()
    return written


def build_intraframe_dataset(
    htvd_intraframe_dir, output_dir, n_samples_per_category=10, variants_per_sample=2
):
    """
    variants_per_sample: how many of the available encoding variants to use
    per source sample (not all 8-9, to keep total data volume manageable
    while still capturing some compression-severity diversity).
    """
    frames_dir = os.path.join(output_dir, "frames")
    masks_dir = os.path.join(output_dir, "masks")
    log = []

    for category in CATEGORIES:
        prefix = PREFIX_MAP[category]
        category_dir = os.path.join(htvd_intraframe_dir, category)

        for i in range(1, n_samples_per_category + 1):
            sample_dir = os.path.join(category_dir, f"{category}{i}")
            tamp_dir = os.path.join(sample_dir, f"{prefix}_Tamp{i}")
            auth_dir = os.path.join(sample_dir, f"{prefix}_Auth{i}")
            mask_dir = os.path.join(sample_dir, "mask")

            if not (os.path.isdir(tamp_dir) and os.path.isdir(auth_dir) and os.path.isdir(mask_dir)):
                print(f"skipping {category}{i}: expected folders not found")
                continue

            tamp_variants = list_encoding_variants(tamp_dir)[:variants_per_sample]
            auth_variants = list_encoding_variants(auth_dir)[:variants_per_sample]

            for variant_file in tamp_variants:
                variant_tag = f"{category}{i}_{os.path.splitext(variant_file)[0]}"
                n = extract_tampered_frames_with_masks(
                    os.path.join(tamp_dir, variant_file), mask_dir,
                    os.path.join(frames_dir, "tampered"), os.path.join(masks_dir, "tampered"),
                    variant_tag,
                )
                log.append({"sample": variant_tag, "type": "tampered", "category": category, "frames": n})

            for variant_file in auth_variants:
                variant_tag = f"{category}{i}_auth_{os.path.splitext(variant_file)[0]}"
                n = extract_authentic_frames(
                    os.path.join(auth_dir, variant_file),
                    os.path.join(frames_dir, "authentic"), os.path.join(masks_dir, "authentic"),
                    variant_tag,
                )
                log.append({"sample": variant_tag, "type": "authentic", "category": category, "frames": n})

            print(f"processed {category}{i}")

    return log
