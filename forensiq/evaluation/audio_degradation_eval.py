"""
forensiq/evaluation/audio_degradation_eval.py
"""
import os

import torch
from torch.utils.data import DataLoader

from forensiq.data.degradation import apply_preset
from forensiq.data.datasets.asvspoof import ASVspoofDataset
from forensiq.evaluation.metrics import evaluate_model

FLAC_SOURCE_DIR = "/content/asvspoof_flac"  # actual location of the raw
                                              # flac files, distinct from
                                              # clean_dir (which is "/content",
                                              # used only for ASVspoofDataset's
                                              # base_dir/flac/ convention)


def build_degraded_holdout_set(holdout_keys, output_dir, preset_name):
    os.makedirs(output_dir, exist_ok=True)
    degraded_paths = {}

    for key in holdout_keys:
        clean_path = os.path.join(FLAC_SOURCE_DIR, f"{key}.flac")
        if not os.path.exists(clean_path):
            continue
        try:
            degraded_path = apply_preset(clean_path, output_dir, preset_name, is_audio=True)
            degraded_paths[key] = degraded_path
        except Exception as e:
            print(f"[WARN] Failed to degrade {key}: {e}")

    return degraded_paths


def evaluate_condition(model, holdout_keys, labels, audio_dir, device, is_degraded_flat_dir=False):
    if is_degraded_flat_dir:
        import soundfile as sf
        from torch import Tensor
        from forensiq.data.datasets.asvspoof import pad

        class FlatDegradedDataset(torch.utils.data.Dataset):
            def __init__(self, keys, labels, audio_dir):
                self.items = []
                for key in keys:
                    matches = [f for f in os.listdir(audio_dir) if f.startswith(key + "__")]
                    if matches:
                        self.items.append((key, os.path.join(audio_dir, matches[0])))

            def __len__(self):
                return len(self.items)

            def __getitem__(self, index):
                key, path = self.items[index]
                X, _ = sf.read(path)
                X_pad = pad(X, 64600)
                return Tensor(X_pad), labels[key]

        dataset = FlatDegradedDataset(holdout_keys, labels, audio_dir)
    else:
        dataset = ASVspoofDataset(list(holdout_keys), labels, audio_dir, train=False)

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    return evaluate_model(model, loader, device)


def run_full_comparison(
    model_a,
    model_b,
    holdout_keys,
    labels,
    clean_dir,
    device,
    presets_to_test=("mild_compress", "heavy_compress", "phone_call_audio"),
    tmp_root="/content/audio_degradation_eval",
):
    import pandas as pd

    results = []

    for model_name, model in [("Phase A (clean-trained)", model_a), ("Phase B (degraded fine-tuned)", model_b)]:
        metrics = evaluate_condition(model, holdout_keys, labels, clean_dir, device, is_degraded_flat_dir=False)
        metrics["phase"] = model_name
        metrics["condition"] = "clean"
        results.append(metrics)
        print(f"{model_name} | clean: {metrics}")

    for preset in presets_to_test:
        preset_dir = os.path.join(tmp_root, preset)
        print(f"\nbuilding degraded files for: {preset}")
        # FIXED: no longer passes clean_dir ("/content") into the
        # degradation step — uses the real flac source location directly.
        degraded = build_degraded_holdout_set(holdout_keys, preset_dir, preset)
        print(f"  degraded files created: {len(degraded)}")

        for model_name, model in [("Phase A (clean-trained)", model_a), ("Phase B (degraded fine-tuned)", model_b)]:
            metrics = evaluate_condition(model, holdout_keys, labels, preset_dir, device, is_degraded_flat_dir=True)
            metrics["phase"] = model_name
            metrics["condition"] = preset
            results.append(metrics)
            print(f"{model_name} | {preset}: {metrics}")

    return pd.DataFrame(results)
