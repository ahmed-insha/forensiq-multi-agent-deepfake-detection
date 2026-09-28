"""
One-time validation scan for the ASVspoof2021 flac dataset.

Reads every available file once to determine which are genuinely decodable,
and saves the result to a cache file. This avoids repeatedly re-discovering
the same unreadable files on every future run, and avoids relying on
re-copying to fix files that are broken at the source.
"""
import json
import os

import soundfile as sf
from tqdm import tqdm


def validate_all_files(flac_dir, keys, cache_path, force_refresh=False):
    if os.path.exists(cache_path) and not force_refresh:
        with open(cache_path) as f:
            result = json.load(f)
        return set(result["good"]), set(result["bad"])

    good, bad = [], []
    for key in tqdm(keys, desc="validating"):
        path = os.path.join(flac_dir, f"{key}.flac")
        try:
            sf.read(path)
            good.append(key)
        except Exception:
            bad.append(key)

    with open(cache_path, "w") as f:
        json.dump({"good": good, "bad": bad}, f)

    return set(good), set(bad)
