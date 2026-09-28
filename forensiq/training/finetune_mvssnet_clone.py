"""
forensiq/training/finetune_mvssnet_clone.py

Fine-tunes the CASIAv2-pretrained MVSS-Net on HTVD's Clone (copy-move)
category specifically, targeting the architectural weakness identified
during zero-shot evaluation (pixel_f1 = 0.0231 on Clone, versus 0.9507
on Splice) -- MVSS-Net's noise-sensitive branch has no signal to detect
same-image copy-move tampering, since a cloned region shares identical
noise/compression history with its surroundings.

Only the pixel-level segmentation output is fine-tuned (via
BCEWithLogitsLoss against ground truth masks). The image-level
classification output is left untouched, since this fine-tuning set
contains only tampered frames (no authentic/real images), and training
the classification head on exclusively positive-labeled data would
bias it rather than improve it.

Given the very small fine-tuning set (105 training frames from 7 source
videos), a low learning rate and few epochs are used deliberately to
limit overfitting risk, with checkpointing based on validation pixel
F1 rather than training loss.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.utils import calculate_pixel_f1


def run_validation(model, val_loader, device):
    """
    Computes average pixel F1 across the validation set. Returns to
    training mode before exiting, since this is called mid-training.
    """
    model.eval()
    f1_scores = []

    with torch.no_grad():
        for frames, masks in val_loader:
            frames = frames.to(device)
            _, seg_logits = model(frames)

            # Ground truth masks were built at 512x512 (see
            # htvd_clone_finetune.py); if the model's raw output
            # resolution differs, align them before scoring rather than
            # assuming an exact match.
            if seg_logits.shape[-2:] != masks.shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            probs = torch.sigmoid(seg_logits).cpu().numpy()
            gt = masks.numpy()

            for i in range(probs.shape[0]):
                f1, _, _ = calculate_pixel_f1(
                    (probs[i, 0] > 0.5).astype(float).flatten(),
                    gt[i, 0].flatten(),
                )
                f1_scores.append(f1)

    model.train()
    return sum(f1_scores) / len(f1_scores)


def finetune_clone(
    model,
    train_dataset,
    val_dataset,
    checkpoint_path,
    max_epochs=8,
    batch_size=4,
    learning_rate=1e-6,  # deliberately low: fine-tuning pretrained
                          # weights on a small dataset, not training
                          # from scratch
    patience=3,
    device="cuda",
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        total_loss = 0.0
        num_batches = 0

        for frames, masks in train_loader:
            frames = frames.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            _, seg_logits = model(frames)

            if seg_logits.shape[-2:] != masks.shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            loss = criterion(seg_logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        val_f1 = run_validation(model, val_loader, device)

        print(f"epoch {epoch + 1}/{max_epochs} train_loss={avg_train_loss:.4f} val_pixel_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  saved checkpoint (val_pixel_f1 improved to {val_f1:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"early stopping at epoch {epoch + 1}")
            break

    return best_val_f1
