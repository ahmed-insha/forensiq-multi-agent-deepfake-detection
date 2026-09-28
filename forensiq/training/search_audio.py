"""
forensiq/training/search_audio.py

Small-scale hyperparameter search for the Audio Agent (Wav2Vec2SpoofClassifier).
Uses only files confirmed readable by the validation scan in validate_flac.py,
since approximately 43% of the source dataset was found to be corrupted at
the source (confirmed via direct comparison of Drive-hosted files against
local copies, ruling out a copy-process issue).
"""
import itertools
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from forensiq.data.datasets.asvspoof import load_df_keys_labels, ASVspoofDataset
from forensiq.models.audio_model import Wav2Vec2SpoofClassifier
from forensiq.training.trainer import TrainConfig, run_training


def build_search_subset(dataset, n_train=2000, n_val=400, seed=42):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def search(
    trial_metadata_path,
    flac_dir,
    base_dir,
    valid_keys=None,
    learning_rates=(1e-5, 5e-5, 1e-4),
    freeze_options=(True, False),
    search_epochs=3,
    batch_size=16,
):
    """
    valid_keys : optional set of keys known to be readable, from the
        validation scan. If given, the dataset is filtered to only these
        keys before building the search subset.
    """
    labels, keys = load_df_keys_labels(trial_metadata_path, flac_dir)

    if valid_keys is not None:
        keys = [k for k in keys if k in valid_keys]
        labels = {k: v for k, v in labels.items() if k in valid_keys}
        print(f"filtered to {len(keys)} validated keys")

    full_dataset = ASVspoofDataset(keys, labels, base_dir, train=True)

    train_subset, val_subset = build_search_subset(full_dataset)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = []
    configs = list(itertools.product(learning_rates, freeze_options))

    for i, (lr, freeze) in enumerate(configs):
        print(f"config {i + 1}/{len(configs)}: lr={lr}, freeze_feature_extractor={freeze}")

        model = Wav2Vec2SpoofClassifier(freeze_feature_extractor=freeze)
        train_config = TrainConfig(epochs=search_epochs, learning_rate=lr, patience=search_epochs)

        result = run_training(model, train_loader, val_loader, train_config, verbose=True)

        results.append({
            "learning_rate": lr,
            "freeze_feature_extractor": freeze,
            "best_val_accuracy": result.best_val_accuracy,
            "best_val_loss": result.best_val_loss,
            "epochs_run": result.epochs_run,
        })

        del model
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(results).sort_values("best_val_accuracy", ascending=False)
    return results_df
