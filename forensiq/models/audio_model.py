

"""
forensiq/models/audio_model.py

Audio Agent: Wav2Vec 2.0 based spoof/deepfake classifier.

Input: raw waveform tensor of shape (batch, 64600), sampled at 16kHz.
Matches the output of forensiq.data.datasets.asvspoof.ASVspoofDataset.
No spectrogram conversion is required.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2SpoofClassifier(nn.Module):
    """
    Binary spoof/bonafide classifier built on a pretrained Wav2Vec 2.0
    backbone with a lightweight classification head.

    Parameters
    ----------
    pretrained_name : str
        HuggingFace checkpoint name. Default is the English-only "base"
        model (95M params). The multilingual XLSR-53 checkpoint (300M+
        params) is a drop-in alternative for stronger cross-lingual
        robustness at the cost of training speed.
    freeze_feature_extractor : bool
        If True, freezes the convolutional feature-extractor layers and
        only fine-tunes the transformer layers and classification head.
    num_classes : int
        Number of output classes (2: bonafide, spoof).
    dropout : float
        Dropout probability in the classification head.
    """

    def __init__(
        self,
        pretrained_name: str = "facebook/wav2vec2-base",
        freeze_feature_extractor: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(pretrained_name)

        if freeze_feature_extractor:
            self.backbone.feature_extractor._freeze_parameters()

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """Zero-mean, unit-variance normalization, as expected by Wav2Vec2."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-7
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor
            Raw waveform, shape (batch, samples), sampled at 16kHz.

        Returns
        -------
        torch.Tensor
            Class logits, shape (batch, num_classes).
        """
        x = self._normalize(x)
        outputs = self.backbone(x)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

