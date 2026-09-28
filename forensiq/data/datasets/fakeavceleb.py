"""
forensiq/data/datasets/fakeavceleb.py

Loader for the FakeAVCeleb dataset (Kaggle re-upload by shreyaty08).
Uses the dataset's own meta_data.csv directly.

Column mapping confirmed by direct inspection:
  - 'type' holds the four category names (RealVideo-RealAudio, etc.)
  - 'Unnamed: 9' holds the relative path to the identity FOLDER, not
    the file itself
  - 'path' holds the actual video filename within that folder
  - 'category' holds an unrelated short code, NOT the category name
"""
import os

import pandas as pd


def load_fakeavceleb_samples(base_dir, meta_csv_path, n_per_category=10, seed=42):
    meta_df = pd.read_csv(meta_csv_path)

    categories = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
    samples = {}

    for category in categories:
        category_rows = meta_df[meta_df["type"] == category]
        sampled_rows = category_rows.sample(n=min(n_per_category, len(category_rows)), random_state=seed)

        paths = []
        for _, row in sampled_rows.iterrows():
            rel_folder = row["Unnamed: 9"].replace("FakeAVCeleb/", "", 1)
            full_path = os.path.join(base_dir, rel_folder, row["path"])
            if os.path.exists(full_path):
                paths.append(full_path)

        samples[category] = paths
        print(f"{category}: {len(paths)}/{len(sampled_rows)} files found on disk ({len(category_rows)} total available)")

    return samples
