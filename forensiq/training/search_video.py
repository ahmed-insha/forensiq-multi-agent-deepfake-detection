"""
Small-scale hyperparameter search for the Video Agent (EfficientNetFrameClassifier).

Search is run on a small, fixed random subset of the extracted frame data
for a small, fixed number of epochs, to compare configurations quickly.
This does not measure final model performance -- it only ranks
configurations against each other under matched, reduced conditions. The
winning configuration is then trained for real (full data, full epoch
budget, early stopping) using forensiq.training.trainer.run_training.

Class weighting is applied throughout, including during the search, since
the extracted frame dataset has a known class imbalance (approximately
9600 real frames vs 28800 fake frames).
"""
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights

from forensiq.models.video_model import EfficientNetFrameClassifier
from forensiq.training.trainer import TrainConfig, run_training


def build_search_subset(dataset, n_train=2000, n_val=400, seed=42):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def compute_class_weights(dataset):
    """
    Computes inverse-frequency class weights from the full dataset's
    class distribution (not the search subset), so weighting reflects
    the true imbalance regardless of what happened to land in the subset.
    """
    counts = [0, 0]
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    num_classes = len(counts)
    weights = [total / (num_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def search(
    frames_dir,
    learning_rates=(1e-5, 5e-5, 1e-4),
    freeze_options=(True, False),
    search_epochs=3,
    batch_size=32,
):
    transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    full_dataset = ImageFolder(frames_dir, transform=transform)

    class_weights = compute_class_weights(full_dataset)
    print(f"class_to_idx: {full_dataset.class_to_idx}")
    print(f"class_weights: {class_weights}")

    train_subset, val_subset = build_search_subset(full_dataset)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    results = []
    import itertools
    configs = list(itertools.product(learning_rates, freeze_options))

    for i, (lr, freeze) in enumerate(configs):
        print(f"config {i + 1}/{len(configs)}: lr={lr}, freeze_backbone={freeze}")

        model = EfficientNetFrameClassifier(freeze_backbone=freeze)
        train_config = TrainConfig(epochs=search_epochs, learning_rate=lr, patience=search_epochs)

        result = run_training(
            model, train_loader, val_loader, train_config,
            class_weights=class_weights, verbose=True,
        )

        results.append({
            "learning_rate": lr,
            "freeze_backbone": freeze,
            "best_val_accuracy": result.best_val_accuracy,
            "best_val_loss": result.best_val_loss,
            "epochs_run": result.epochs_run,
        })

        del model
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(results).sort_values("best_val_accuracy", ascending=False)
    return results_df
