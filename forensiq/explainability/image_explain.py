"""
forensiq/explainability/image_explain.py

Occlusion-based saliency for the Image Agent (UnivFD/CLIP).

Grad-CAM requires convolutional feature maps and does not directly
apply to CLIP's Vision Transformer backbone (patch-based, no conv
layers). This uses the same occlusion principle applied to the Audio
Agent: mask a region of the image, measure the resulting change in
predicted probability, repeated in a sliding grid to build a full
saliency map.
"""
import numpy as np
import torch


def compute_image_occlusion_saliency(model, image_tensor, device, patch_size=32, stride=16):
    """
    image_tensor: preprocessed CLIP input, shape (3, 224, 224).
    patch_size: size of the square region masked at each step.
    stride: how far the occlusion window moves each step (smaller
        stride gives a smoother map but costs more forward passes).
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    _, h, w = image_tensor.shape

    with torch.no_grad():
        baseline_logit = model(image_tensor.unsqueeze(0))
        baseline_prob = torch.sigmoid(baseline_logit).item()

    saliency = np.zeros((h, w))
    counts = np.zeros((h, w))

    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            occluded = image_tensor.clone()
            occluded[:, y:y + patch_size, x:x + patch_size] = 0.0

            with torch.no_grad():
                occluded_logit = model(occluded.unsqueeze(0))
                occluded_prob = torch.sigmoid(occluded_logit).item()

            importance = baseline_prob - occluded_prob
            saliency[y:y + patch_size, x:x + patch_size] += importance
            counts[y:y + patch_size, x:x + patch_size] += 1

    counts[counts == 0] = 1
    saliency = saliency / counts

    return saliency, baseline_prob
