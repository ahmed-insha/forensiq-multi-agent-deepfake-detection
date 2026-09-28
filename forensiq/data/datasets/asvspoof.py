"""
forensiq/data/datasets/asvspoof.py

ASVspoof2021 DF eval dataset loader.
"""
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset


def list_available_keys(flac_dir, cache_path=None, force_refresh=False):
    if cache_path and os.path.exists(cache_path) and not force_refresh:
        with open(cache_path) as f:
            return set(json.load(f))

    keys = set(f.replace(".flac", "") for f in os.listdir(flac_dir) if f.endswith(".flac"))

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(sorted(keys), f)

    return keys


def load_df_keys_labels(trial_metadata_path, flac_dir, cache_path=None):
    available_keys = list_available_keys(flac_dir, cache_path=cache_path)

    d_meta = {}
    file_list = []
    with open(trial_metadata_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            key = parts[1]
            if key not in available_keys:
                continue
            label = parts[5]
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0

    return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


class ASVspoofDataset(Dataset):
    """
    list_IDs : list of utterance keys (from load_df_keys_labels)
    labels   : dict {key: 0/1}
    base_dir : Path to folder containing flac/ subfolder
    train    : if True, random crop augmentation; if False, deterministic crop
    max_retries : maximum number of alternate indices to try before raising
        an error, preventing unbounded retries if many files are unreadable.
    """

    def __init__(self, list_IDs, labels, base_dir, train=True, max_retries=10):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = Path(base_dir)
        self.train = train
        self.cut = 64600
        self.max_retries = max_retries

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        for attempt in range(self.max_retries):
            key = self.list_IDs[index]
            flac_path = self.base_dir / "flac" / f"{key}.flac"
            try:
                X, _ = sf.read(str(flac_path))
                X_pad = pad_random(X, self.cut) if self.train else pad(X, self.cut)
                x_inp = Tensor(X_pad)
                y = self.labels[key]
                return x_inp, y
            except Exception as e:
                print(f"[WARN] Could not read {flac_path}: {e}")
                index = (index + 1) % len(self.list_IDs)

        raise RuntimeError(
            f"Failed to read a valid audio file after {self.max_retries} attempts. "
            "This likely indicates a systemic problem (e.g. a corrupted local copy) "
            "rather than a few isolated bad files."
        )
