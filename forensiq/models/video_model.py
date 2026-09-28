"""
Video Agent: EfficientNet-B0 based real/fake frame classifier.

Input: RGB image tensor of shape (batch, 3, 224, 224), normalized with
ImageNet statistics. Matches the output of a standard torchvision
ImageFolder + transform pipeline pointed at extracted FaceForensics++
frames (see forensiq.data.datasets.faceforensics).
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetFrameClassifier(nn.Module):
    """
    Binary real/fake classifier built on a pretrained EfficientNet-B0
    backbone with a replaced classification head.

    Parameters
    ----------
    pretrained : bool
        If True, initializes the backbone with ImageNet-pretrained weights.
    freeze_backbone : bool
        If True, freezes all backbone layers and only trains the
        replaced classification head.
    num_classes : int
        Number of output classes (2: real, fake).
    dropout : float
        Dropout probability in the classification head.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor
            Batch of images, shape (batch, 3, 224, 224).

        Returns
        -------
        torch.Tensor
            Class logits, shape (batch, num_classes).
        """
        return self.backbone(x)
