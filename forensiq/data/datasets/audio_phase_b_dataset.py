"""
forensiq/data/datasets/audio_phase_b_dataset.py

Combined clean + degraded dataset for Audio Agent Phase B fine-tuning.

Each fine-tuning file contributes BOTH its clean version and its
degraded version, guaranteeing exactly 50% clean representation in the
training mix -- directly applying the lesson learned from Video Agent
Phase B v1, where assigning only one condition per file at random
caused clean representation to drop to roughly 20% and measurably
regressed clean-condition performance.
"""
import os

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

from forensiq.data.datasets.asvspoof import pad, pad_random


class AudioPhaseBDataset(Dataset):
    """
    keys       : list of utterance keys used for Phase B fine-tuning
    labels     : dict {key: 0/1}, same labels as Phase A (degradation
                 does not change whether audio is bonafide or spoof)
    clean_dir  : folder containing original {key}.flac files
    degraded_dir : folder containing degraded {key}__{preset}.flac files
    """

    def __init__(self, keys, labels, clean_dir, degraded_dir, train=True):
        self.labels = labels
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.train = train
        self.cut = 64600

        # Build one entry per (key, is_degraded) pair, so each source
        # file contributes exactly two dataset items: one clean, one
        # degraded.
        self.items = []
        degraded_files = {f for f in os.listdir(degraded_dir)}

        for key in keys:
            self.items.append((key, False))  # clean version
            matches = [f for f in degraded_files if f.startswith(key + "__")]
            if matches:
                self.items.append((key, matches[0]))  # degraded version

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        for attempt in range(10):
            key, degraded_filename = self.items[index]

            if degraded_filename is False:
                path = os.path.join(self.clean_dir, f"{key}.flac")
            else:
                path = os.path.join(self.degraded_dir, degraded_filename)

            try:
                X, _ = sf.read(path)
                X_pad = pad_random(X, self.cut) if self.train else pad(X, self.cut)
                x_inp = Tensor(X_pad)
                y = self.labels[key]
                return x_inp, y
            except Exception as e:
                print(f"[WARN] Could not read {path}: {e}")
                index = (index + 1) % len(self.items)

        raise RuntimeError(f"Failed to read a valid file after 10 attempts.")
