"""
Phase A training for the Audio Agent: Wav2Vec2SpoofClassifier trained on
clean (non-degraded) ASVspoof2021 DF eval audio.
"""
import os
import random

import torch
from torch.utils.data import DataLoader, random_split

from forensiq.data.datasets.asvspoof import load_df_keys_labels, ASVspoofDataset
from forensiq.models.audio_model import Wav2Vec2SpoofClassifier
from forensiq.training.trainer import TrainConfig, run_training


def train_phase_a(
    trial_metadata_path,
    flac_dir,
    base_dir,
    checkpoint_path,
    valid_keys,
    max_train_samples=15000,
    val_fraction=0.15,
    batch_size=16,
    seed=42,
    max_epochs=6,
    patience=2,
    learning_rate=1e-5,
):
    labels, keys = load_df_keys_labels(trial_metadata_path, flac_dir)
    print(f"after load_df_keys_labels: {len(keys)} keys, {len(labels)} labels")

    keys = [k for k in keys if k in valid_keys]
    print(f"after valid_keys filter: {len(keys)} keys")

    random.seed(seed)
    if len(keys) > max_train_samples:
        keys = random.sample(keys, max_train_samples)
    print(f"after sampling: {len(keys)} keys")

    labels = {k: labels[k] for k in keys}
    print(f"after rebuilding labels dict: {len(labels)} labels")

    full_dataset = ASVspoofDataset(keys, labels, base_dir, train=True)
    print(f"dataset length: {len(full_dataset)}")

    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"train samples: {train_size}, val samples: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = Wav2Vec2SpoofClassifier(freeze_feature_extractor=True)

    config = TrainConfig(
        epochs=max_epochs,
        learning_rate=learning_rate,
        patience=patience,
    )

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    result = run_training(
        model, train_loader, val_loader, config,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )

    return result
