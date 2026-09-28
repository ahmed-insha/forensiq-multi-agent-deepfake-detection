
"""
forensiq/training/train_audio_phase_c.py

Phase C corrective fine-tune for the Audio Agent: continues from the
Phase B checkpoint, this time applying class-weighted loss to correct
for ASVspoof's true dataset-wide class imbalance (3.70% bonafide,
96.30% spoof), which was identified but deliberately deferred during
original training (see methodology log, Section 2.8) and subsequently
confirmed as a real, contributing issue across three independent
tests: FakeAVCeleb cross-modal testing, an ASVspoof-domain sanity
check, and the synthetic timeline localization test.

This is a short corrective fine-tune, not a full retrain -- it reuses
the existing Phase B checkpoint and training data, only adding the
missing class-weighting term to the loss.
"""
import os

import torch
from torch.utils.data import DataLoader, random_split

from forensiq.models.audio_model import Wav2Vec2SpoofClassifier
from forensiq.training.trainer import TrainConfig, run_training


def train_phase_c(
    phase_b_dataset,
    phase_b_checkpoint_path,
    phase_c_checkpoint_path,
    val_fraction=0.15,
    batch_size=16,
    seed=42,
    max_epochs=3,
    patience=2,
    learning_rate=1e-6,
):
    """
    Class weights are computed from the TRUE dataset-wide ASVspoof2021
    DF ratio (confirmed earlier against the full label file), not from
    any local sample, since local samples were found to have highly
    variable class balance depending on which dataset part they came
    from.
    """
    # weight[class] = 1 / (num_classes * class_frequency)
    # class 0 = spoof (96.30%), class 1 = bonafide (3.70%)
    class_weights = torch.tensor([
        1 / (2 * 0.9630),  # spoof: ~0.519
        1 / (2 * 0.0370),  # bonafide: ~13.514
    ], dtype=torch.float32)
    print(f"Class weights: spoof={class_weights[0]:.4f}, bonafide={class_weights[1]:.4f}")

    val_size = int(len(phase_b_dataset) * val_fraction)
    train_size = len(phase_b_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(phase_b_dataset, [train_size, val_size], generator=generator)

    print(f"train samples: {train_size}, val samples: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = Wav2Vec2SpoofClassifier(freeze_feature_extractor=True)
    model.load_state_dict(torch.load(phase_b_checkpoint_path, map_location="cpu"))
    print(f"loaded Phase B weights from: {phase_b_checkpoint_path}")

    config = TrainConfig(
        epochs=max_epochs,
        learning_rate=learning_rate,
        patience=patience,
    )

    os.makedirs(os.path.dirname(phase_c_checkpoint_path), exist_ok=True)

    result = run_training(
        model, train_loader, val_loader, config,
        checkpoint_path=phase_c_checkpoint_path,
        class_weights=class_weights,
        verbose=True,
    )

    return result
