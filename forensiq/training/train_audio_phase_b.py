"""
forensiq/training/train_audio_phase_b.py

Phase B fine-tuning for the Audio Agent: starts from the Phase A
(clean-trained) checkpoint and continues training on a 50/50 mix of
clean and degraded audio, to improve robustness against real-world
quality loss.

Uses a lower learning rate than Phase A, standard practice when
fine-tuning an already-trained model, and the same frozen-feature-
extractor configuration that performed best during the original
hyperparameter search.
"""
import os

import torch
from torch.utils.data import DataLoader, random_split

from forensiq.models.audio_model import Wav2Vec2SpoofClassifier
from forensiq.training.trainer import TrainConfig, run_training


def train_phase_b(
    phase_b_dataset,
    phase_a_checkpoint_path,
    phase_b_checkpoint_path,
    val_fraction=0.15,
    batch_size=16,
    seed=42,
    max_epochs=4,
    patience=2,
    learning_rate=1e-6,
):
    val_size = int(len(phase_b_dataset) * val_fraction)
    train_size = len(phase_b_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(phase_b_dataset, [train_size, val_size], generator=generator)

    print(f"train samples: {train_size}, val samples: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = Wav2Vec2SpoofClassifier(freeze_feature_extractor=True)
    model.load_state_dict(torch.load(phase_a_checkpoint_path, map_location="cpu"))
    print(f"loaded Phase A weights from: {phase_a_checkpoint_path}")

    config = TrainConfig(
        epochs=max_epochs,
        learning_rate=learning_rate,
        patience=patience,
    )

    os.makedirs(os.path.dirname(phase_b_checkpoint_path), exist_ok=True)

    result = run_training(
        model, train_loader, val_loader, config,
        checkpoint_path=phase_b_checkpoint_path,
        verbose=True,
    )

    return result
