
"""
forensiq/models/temporal_consistency.py

A lightweight temporal-consistency signal for the Video Agent,
addressing a real, named limitation: EfficientNet scores every frame
completely independently, with no sense of motion or frame-to-frame
continuity. This is a plausible explanation for the DigiFakeAV
zero-shot failure (0/8) -- the discriminating signal for that specific
generation technique may live in HOW content moves between frames,
not in any single frame's static appearance.

This is NOT a full spatiotemporal architecture (that would require
AltFreezing/FTCN-scale integration, documented separately as future
work). It is a cheap, signal-processing-only heuristic: measuring how
SMOOTH optical flow is between consecutive frames. Genuine human
motion tends to be smooth and continuous; some generation/reenactment
techniques have been observed to introduce subtle frame-to-frame
discontinuities not perceptible in any single frame.

Note: this project's earlier optical-flow work (frame-deletion
detection, Structural Agent) found dense optical flow UNDERESTIMATES
motion at genuine discontinuities on some footage -- this is
documented, known behavior of the technique, not assumed to be
reliable a priori. This module is explicitly framed as an EXPERIMENT
to be tested, not a presumed improvement, consistent with how every
other technique in this project has been validated before being
trusted.
"""
import cv2
import numpy as np


def compute_frame_to_frame_flow_magnitude(frame_a, frame_b):
    """
    Computes dense optical flow between two consecutive frames and
    returns the mean flow magnitude -- how much apparent motion
    occurred between them.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(magnitude.mean())


def compute_temporal_consistency_score(video_path, num_frames=20):
    """
    Samples num_frames evenly across a video, computes frame-to-frame
    optical flow magnitude between each consecutive pair, and returns
    a "jitter" score: the coefficient of variation (std/mean) of the
    flow-magnitude sequence.

    Rationale: genuine motion, even fast motion, tends to change
    smoothly from one frame-pair to the next. A HIGH coefficient of
    variation means motion magnitude is jumping around erratically
    between frame-pairs -- a candidate signal for temporal
    manipulation artifacts, distinct from anything a single-frame
    classifier can see.

    Returns None if fewer than 3 frames could be read (insufficient
    for a meaningful sequence).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if len(frames) < 3:
        return None

    flow_magnitudes = []
    for i in range(len(frames) - 1):
        mag = compute_frame_to_frame_flow_magnitude(frames[i], frames[i + 1])
        flow_magnitudes.append(mag)

    flow_magnitudes = np.array(flow_magnitudes)
    mean_flow = flow_magnitudes.mean()
    std_flow = flow_magnitudes.std()

    if mean_flow < 1e-6:
        # Essentially no motion detected at all across the whole clip
        # -- coefficient of variation is undefined/meaningless here.
        return {
            "jitter_score": None, "mean_flow": float(mean_flow),
            "std_flow": float(std_flow), "per_pair_flow": flow_magnitudes.tolist(),
            "note": "Near-zero motion detected; jitter score not meaningful for this clip.",
        }

    jitter_score = std_flow / mean_flow

    return {
        "jitter_score": float(jitter_score),
        "mean_flow": float(mean_flow),
        "std_flow": float(std_flow),
        "per_pair_flow": flow_magnitudes.tolist(),
    }
