"""
forensiq/models/image_model.py

Image Agent: CLIP ViT-L/14 based AI-generated image detector (UnivFD).

Note: UnivFD's repo and MVSS-Net's repo both define a top-level
package named 'models', which collide in sys.modules when both repos
are cloned into the same session. This module purges any cached
'models' package before importing UnivFD's version, and ensures
UnivFD's repo path takes priority, so this works correctly regardless
of which repo was imported first in the current session.
"""
import sys

import torch

UNIVFD_REPO_PATH = "/content/univfd_repo"

# Purge any stale 'models' package cached from a different repo
# (e.g. MVSS-Net) before importing UnivFD's own 'models' package.
for mod_name in list(sys.modules.keys()):
    if mod_name == "models" or mod_name.startswith("models."):
        del sys.modules[mod_name]

if UNIVFD_REPO_PATH in sys.path:
    sys.path.remove(UNIVFD_REPO_PATH)
sys.path.insert(0, UNIVFD_REPO_PATH)  # ensure it's checked first

from models.clip import clip

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CLIP_CACHE_DIR = "/content/drive/MyDrive/MDX Data Science & AI/THESIS Project/ForensiQ/Forensiq_Code/checkpoints/clip_cache"


class UnivFDModel(torch.nn.Module):
    def __init__(self, clip_name="ViT-L/14", num_classes=1, device="cuda"):
        super().__init__()
        self.model, _ = clip.load(clip_name, device=device, download_root=CLIP_CACHE_DIR)
        self.fc = torch.nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.model.encode_image(x)
        return self.fc(features)


def load_univfd_model(checkpoint_path, clip_name="ViT-L/14", device="cuda"):
    model = UnivFDModel(clip_name=clip_name, device=device)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.fc.load_state_dict(state_dict)
    model.eval()
    return model
