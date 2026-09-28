"""
forensiq/models/interframe_detector.py

Frame-consistency-based detection for Interframe tampering. See
docstrings below and methodology log for the two-technique design
(discontinuity detection for deletion/insertion, self-similarity matrix
for block duplication).
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_frame_similarity_series(video_path, resize=256):
    cap = cv2.VideoCapture(video_path)
    similarities = []
    prev_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (resize, resize))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            similarities.append(ssim(prev_gray, gray))
        prev_gray = gray
    cap.release()
    return similarities


def detect_discontinuities(similarities, disc_std_multiplier=2.5):
    similarities = np.array(similarities)
    mean_sim = similarities.mean()
    std_sim = similarities.std()
    threshold = mean_sim - disc_std_multiplier * std_sim
    flags = np.where(similarities < threshold)[0]
    return flags.tolist(), float(threshold)


def compute_frame_features(video_path, resize=32):
    cap = cv2.VideoCapture(video_path)
    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (resize, resize))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        features.append(gray.flatten())
    cap.release()
    features = np.array(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return features / norms


def compute_self_similarity_matrix(features):
    return features @ features.T


def find_duplicated_blocks_adaptive(
    similarity_matrix,
    min_lag=20,
    # Lowered from 50: with the full lag range now searched (see fix
    # below), a smaller min_lag no longer risks false positives from
    # ordinary short-range smoothness, since genuine duplication runs
    # are distinguished by their LENGTH and adaptive threshold, not by
    # excluding short lags entirely.
    min_run_length=15,
    percentile_threshold=99.5,
):
    """
    Searches lags from min_lag up to (n - min_run_length) -- the full
    range in which a valid duplicated block of at least min_run_length
    frames could exist. An earlier version capped the search at n // 2,
    which failed to detect duplications where the copy is inserted far
    from its original occurrence (see methodology log: F_Dup2's
    duplicate is placed ~295 frames from its original, farther than
    half of a ~400-450 frame video).
    """
    n = similarity_matrix.shape[0]
    max_lag = n - min_run_length  # full valid range, not n // 2

    off_diagonal_values = []
    for lag in range(min_lag, max_lag):
        off_diagonal_values.extend(similarity_matrix[i, i + lag] for i in range(n - lag))
    adaptive_threshold = np.percentile(off_diagonal_values, percentile_threshold)

    candidates = []
    for lag in range(min_lag, max_lag):
        diagonal = np.array([similarity_matrix[i, i + lag] for i in range(n - lag)])
        is_match = diagonal > adaptive_threshold

        run_start = None
        for i, matched in enumerate(is_match):
            if matched and run_start is None:
                run_start = i
            elif not matched and run_start is not None:
                run_length = i - run_start
                if run_length >= min_run_length:
                    candidates.append({
                        "original_start": run_start, "duplicate_start": run_start + lag,
                        "length": run_length, "lag": lag,
                        "mean_similarity": float(diagonal[run_start:i].mean()),
                    })
                run_start = None
        if run_start is not None and len(is_match) - run_start >= min_run_length:
            candidates.append({
                "original_start": run_start, "duplicate_start": run_start + lag,
                "length": len(is_match) - run_start, "lag": lag,
                "mean_similarity": float(diagonal[run_start:].mean()),
            })

    candidates.sort(key=lambda c: c["length"], reverse=True)
    return candidates, adaptive_threshold
