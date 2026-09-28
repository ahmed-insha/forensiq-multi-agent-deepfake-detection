"""
Phase B fine-tuning for the Video Agent: starts from the Phase A
(clean-trained) checkpoint and continues training on a mix of clean and
degraded frames, to improve robustness against real-world quality loss.

Uses a lower learning rate than Phase A, standard practice when
fine-tuning an already-trained model to avoid overwriting previously
learned features with large updates.
"""
import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights

from forensiq.models.video_model import EfficientNetFrameClassifier
from forensiq.training.trainer import TrainConfig, run_training
from forensiq.training.search_video import compute_class_weights


def train_phase_b(
    phase_b_frames_dir,
    phase_a_checkpoint_path,
    phase_b_checkpoint_path,
    val_fraction=0.15,
    batch_size=32,
    seed=42,
    max_epochs=5,
    patience=2,
    learning_rate=1e-5,
):
    transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    full_dataset = ImageFolder(phase_b_frames_dir, transform=transform)

    class_weights = compute_class_weights(full_dataset)
    print(f"class_to_idx: {full_dataset.class_to_idx}")
    print(f"class_weights: {class_weights}")

    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"train samples: {train_size}, val samples: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EfficientNetFrameClassifier(freeze_backbone=False)
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
        class_weights=class_weights,
        verbose=True,
    )

    return result
