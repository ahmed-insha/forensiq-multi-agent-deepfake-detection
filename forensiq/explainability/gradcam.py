"""
Grad-CAM implementation for the Video Agent.

Produces a heatmap over an input frame showing which spatial regions most
influenced the model's prediction, using gradients flowing into the final
convolutional layer of the backbone.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Computes Grad-CAM heatmaps for a given model and target layer.

    Parameters
    ----------
    model : torch.nn.Module
        The trained classification model.
    target_layer : torch.nn.Module
        The convolutional layer to compute gradients with respect to.
        For EfficientNetFrameClassifier, this is model.backbone.features[-1].
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Runs a forward and backward pass and returns a normalized heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            A single image, shape (1, 3, H, W).
        target_class : int, optional
            Class index to explain. If None, uses the model's predicted
            class for this input.

        Returns
        -------
        numpy.ndarray
            Heatmap of shape (H, W), values normalized to [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        input_size = input_tensor.shape[-2:]
        cam = cv2.resize(cam, (input_size[1], input_size[0]))

        return cam, target_class


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """
    Overlays a Grad-CAM heatmap on top of the original image for
    visualization.

    Parameters
    ----------
    original_image : numpy.ndarray
        RGB image, shape (H, W, 3), values in [0, 255], dtype uint8.
    heatmap : numpy.ndarray
        Normalized heatmap, shape (H, W), values in [0, 1].
    alpha : float
        Blend strength of the heatmap over the original image.

    Returns
    -------
    numpy.ndarray
        RGB image with heatmap overlay, shape (H, W, 3), dtype uint8.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_color + (1 - alpha) * original_image).astype(np.uint8)
    return overlay
