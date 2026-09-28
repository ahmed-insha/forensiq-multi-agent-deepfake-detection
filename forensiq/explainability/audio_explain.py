"""
forensiq/explainability/audio_explain.py

Occlusion-based saliency for the Audio Agent, the raw-waveform
equivalent of Grad-CAM for images. Since audio is a 1D time signal
rather than a spatial image, this masks short consecutive time
segments one at a time and measures how much the model's predicted
probability changes -- segments that cause a large change when removed
are the ones the model most relies on for its decision.
"""
import numpy as np
import torch


def compute_occlusion_saliency(model, waveform, device, segment_size=1600, stride=800):
    """
    waveform: 1D numpy array or tensor, length 64600 (matching the
        Audio Agent's fixed input length).
    segment_size: number of samples masked at once (1600 samples =
        100ms at 16kHz -- a duration a listener could reasonably
        associate with the model's decision).
    stride: how far the occlusion window moves each step (overlapping
        windows give a smoother saliency curve than non-overlapping).

    Returns: (saliency array aligned to time, baseline probability)
    """
    model.eval()
    if not torch.is_tensor(waveform):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    waveform = waveform.to(device)

    with torch.no_grad():
        baseline_logits = model(waveform.unsqueeze(0))
        baseline_prob = torch.softmax(baseline_logits, dim=1)[0, 1].item()  # prob of "fake"/class 1

    length = waveform.shape[0]
    saliency = np.zeros(length)
    counts = np.zeros(length)

    for start in range(0, length - segment_size, stride):
        occluded = waveform.clone()
        occluded[start:start + segment_size] = 0.0  # mute this segment

        with torch.no_grad():
            occluded_logits = model(occluded.unsqueeze(0))
            occluded_prob = torch.softmax(occluded_logits, dim=1)[0, 1].item()

        # Large drop in "fake" probability when this segment is muted
        # means the model was relying heavily on it.
        importance = baseline_prob - occluded_prob
        saliency[start:start + segment_size] += importance
        counts[start:start + segment_size] += 1

    counts[counts == 0] = 1  # avoid division by zero at the very edges
    saliency = saliency / counts

    return saliency, baseline_prob
