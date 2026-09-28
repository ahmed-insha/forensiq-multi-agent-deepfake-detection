"""
forensiq/data/datasets/build_audio_phase_b.py

Builds the Phase B fine-tuning dataset for the Audio Agent: a mix of
clean and degraded audio, used to fine-tune the Phase A checkpoint
toward robustness against real-world audio quality loss.

Applies two lessons learned from the Video Agent's Phase B experiments:
1. Each source file contributes BOTH a clean version and one randomly
   degraded version, guaranteeing exactly 50% clean representation in
   the training mix -- Video Phase B v1 assigned only ONE condition per
   file at random, causing clean representation to drop to ~20% and
   measurably regressing clean-condition performance.
2. The held-out evaluation set (used for the final clean-vs-degraded
   comparison) is excluded from this training set at the file level,
   preventing the leakage that made Video Phase B v2's internal
   validation metric misleadingly optimistic.
"""
import os
import random

from forensiq.data.degradation import apply_preset

DEGRADED_PRESETS = ["mild_compress", "heavy_compress", "phone_call_audio", "social_media"]


def select_finetune_files(all_keys, exclude_keys, n_files=3000, seed=456):
    """
    Selects a random sample of audio files for Phase B fine-tuning,
    excluding any file reserved for held-out evaluation.
    """
    random.seed(seed)
    candidates = [k for k in all_keys if k not in exclude_keys]
    selected = random.sample(candidates, min(n_files, len(candidates)))
    return selected


def build_phase_b_audio_dataset(
    selected_keys,
    flac_source_dir,
    output_dir,
    seed=456,
):
    """
    For each selected audio file, copies the clean version and creates
    one randomly degraded version, both into class-matching output
    folders mirroring the original bonafide/spoof structure.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    log = []

    for i, key in enumerate(selected_keys):
        clean_path = os.path.join(flac_source_dir, f"{key}.flac")
        if not os.path.exists(clean_path):
            continue

        condition = random.choice(DEGRADED_PRESETS)

        try:
            degraded_path = apply_preset(clean_path, output_dir, condition, is_audio=True)
            log.append({"key": key, "condition": condition, "degraded_path": degraded_path})
        except Exception as e:
            print(f"[WARN] Failed to degrade {key}: {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"processed {i + 1}/{len(selected_keys)} files")

    return log
