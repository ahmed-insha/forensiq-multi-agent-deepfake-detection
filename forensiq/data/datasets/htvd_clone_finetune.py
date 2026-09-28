"""
forensiq/data/datasets/htvd_clone_finetune.py

Dataset loader for fine-tuning MVSS-Net specifically on HTVD's Clone
(copy-move) category. See methodology log for details on mask threshold
and folder-naming fixes applied here.
"""
import os
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
MASK_THRESHOLD = 100
MASK_FOLDER_CANDIDATES = ["mask", "mask 2"]


def preprocess_frame(frame_bgr, resize=512):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (resize, resize))
    frame_float = frame_resized.astype(np.float32) / 255.0
    frame_normalized = (frame_float - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(frame_normalized.transpose(2, 0, 1)).float()
    return tensor


def preprocess_mask(mask_gray, resize=512):
    mask_binary = (mask_gray > MASK_THRESHOLD).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_binary, (resize, resize), interpolation=cv2.INTER_NEAREST)
    mask_normalized = (mask_resized / 255.0).astype(np.float32)
    return torch.from_numpy(mask_normalized).unsqueeze(0)


def find_mask_dir(sample_dir):
    for candidate in MASK_FOLDER_CANDIDATES:
        path = os.path.join(sample_dir, candidate)
        if os.path.isdir(path):
            return path
    return None


def read_frame_robust(video_path, frame_idx, max_backtrack=5):
    """
    Reads a specific frame by index, falling back to nearby earlier
    frames if the exact index fails to read -- OpenCV's frame-seeking
    (CAP_PROP_POS_FRAMES) is not always reliable, particularly near the
    end of a video.
    """
    for offset in range(max_backtrack + 1):
        target_idx = max(0, frame_idx - offset)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return frame
    return None


class HTVDCloneFineTuneDataset(Dataset):
    def __init__(self, intraframe_dir, sample_names, frames_per_video=15, resize=512):
        self.resize = resize
        self.items = []

        for sample_name in sample_names:
            match = re.search(r"\d+", sample_name)
            sample_number = match.group()

            sample_dir = os.path.join(intraframe_dir, "Clone", sample_name)
            tamp_dir = os.path.join(sample_dir, f"C_Tamp{sample_number}")
            mask_dir = find_mask_dir(sample_dir)

            if not (os.path.isdir(tamp_dir) and mask_dir):
                print(f"skipping {sample_name}: expected folders not found "
                      f"(tamp_dir={tamp_dir}, mask_dir={mask_dir})")
                continue

            variant_files = sorted(f for f in os.listdir(tamp_dir) if f.endswith(".mp4"))
            if not variant_files:
                continue
            video_path = os.path.join(tamp_dir, variant_files[0])

            mask_files = sorted(os.listdir(mask_dir))
            # Sample from the first 90% of available frames, avoiding the
            # very last few frames where OpenCV's exact-index seeking was
            # found to sometimes fail.
            usable_range = max(1, int(len(mask_files) * 0.9))
            num_frames = min(usable_range, frames_per_video)
            frame_indices = np.linspace(0, usable_range - 1, num_frames, dtype=int)

            for idx in frame_indices:
                self.items.append((video_path, mask_dir, int(idx), mask_files[idx]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        video_path, mask_dir, frame_idx, mask_filename = self.items[index]

        frame = read_frame_robust(video_path, frame_idx)
        if frame is None:
            # As a last resort, skip to the next item entirely rather
            # than crash the whole training run over one bad frame read.
            return self.__getitem__((index + 1) % len(self.items))

        mask = cv2.imread(os.path.join(mask_dir, mask_filename), cv2.IMREAD_GRAYSCALE)

        frame_tensor = preprocess_frame(frame, self.resize)
        mask_tensor = preprocess_mask(mask, self.resize)

        return frame_tensor, mask_tensor
