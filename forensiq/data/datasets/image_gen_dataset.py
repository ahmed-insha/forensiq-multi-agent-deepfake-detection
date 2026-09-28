"""
forensiq/data/datasets/image_gen_dataset.py

Dataset adapter for julienlucas/midjourney-dalle-sd-dataset.

Label convention CORRECTED: this dataset's own Hugging Face metadata
documents label 0 = fake, 1 = real -- the opposite of the convention
initially assumed (based on UnivFD's paper/validate.py, without
checking this specific dataset's own documentation). Remapped here so
downstream code consistently uses 0 = real, 1 = fake, matching UnivFD's
expected output convention.
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class HFImageGenDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        example = self.hf_dataset[index]
        img = example["image"].convert("RGB")
        img_tensor = self.transform(img)
        # Remap: dataset's 0=fake/1=real -> standard 0=real/1=fake
        label = 1 - example["label"]
        return img_tensor, label
