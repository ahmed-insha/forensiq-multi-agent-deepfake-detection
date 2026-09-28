"""
Phase A training for the Video Agent: EfficientNetFrameClassifier trained
on clean (non-degraded) FaceForensics++ frames.

Uses the configuration selected by forensiq.training.search_video
(learning_rate=0.0001, freeze_backbone=False).
"""
import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights

from forensiq.models.video_model import EfficientNetFrameClassifier
from forensiq.training.trainer import TrainConfig, run_training
from forensiq.training.search_video import compute_class_weights


def train_phase_a(frames_dir, checkpoint_path, val_fraction=0.15, batch_size=32, seed=42):
    transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    full_dataset = ImageFolder(frames_dir, transform=transform)

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

    config = TrainConfig(
        epochs=10,
        learning_rate=0.0001,
        patience=3,
    )

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    result = run_training(
        model, train_loader, val_loader, config,
        checkpoint_path=checkpoint_path,
        class_weights=class_weights,
        verbose=True,
    )

    return result
