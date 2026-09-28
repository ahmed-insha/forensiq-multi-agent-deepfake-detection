"""
forensiq/models/interframe_motion_detector.py

Optical-flow-based deletion/insertion detector for Interframe tampering.

Frame-to-frame SSIM similarity was found not to reliably detect clean
mid-video deletion (see methodology log): removing a segment can leave
visually similar content on both sides of the cut, producing no
detectable similarity drop. This approach instead measures the
MAGNITUDE of implied motion between consecutive frames using optical
flow. The hypothesis: even if content looks visually similar across a
deletion point, anything that was moving during the deleted segment
must appear to "jump" an anomalously large distance in a single frame
step, since 150 frames' worth of accumulated motion is compressed into
one frame transition.
"""
import cv2
import numpy as np


def compute_optical_flow_magnitude_series(video_path, resize=256):
    """
    Returns the mean optical flow magnitude for every consecutive frame
    pair across the video, using dense Farneback optical flow.
    """
    cap = cv2.VideoCapture(video_path)
    magnitudes = []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_small = cv2.resize(frame, (resize, resize))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            magnitudes.append(float(magnitude.mean()))

        prev_gray = gray

    cap.release()
    return magnitudes


def detect_motion_discontinuities(magnitudes, std_multiplier=2.5):
    """
    Flags frame pairs with anomalously HIGH motion magnitude relative to
    the video's own typical motion level -- the expected signature of a
    deletion/insertion point, as opposed to the LOW-similarity signature
    that frame-similarity-based detection looks for.
    """
    magnitudes = np.array(magnitudes)
    mean_mag = magnitudes.mean()
    std_mag = magnitudes.std()
    threshold = mean_mag + std_multiplier * std_mag
    flags = np.where(magnitudes > threshold)[0]
    return flags.tolist(), float(threshold)
