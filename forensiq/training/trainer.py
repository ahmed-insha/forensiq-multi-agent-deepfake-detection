"""
Generic, model-agnostic training and evaluation loop, shared across all
ForensiQ specialist agents (audio, video, image, structural). Each model's
training script builds a model and dataloaders, then calls run_training()
here rather than reimplementing the loop.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainResult:
    best_val_accuracy: float = 0.0
    best_val_loss: float = float("inf")
    history: list = field(default_factory=list)
    stopped_early: bool = False
    epochs_run: int = 0


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def run_training(
    model,
    train_loader,
    val_loader,
    config: TrainConfig,
    checkpoint_path: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> TrainResult:
    """
    class_weights: optional per-class weight tensor passed to
    CrossEntropyLoss, used to correct for class imbalance in the training
    data (for example, when one class has substantially more samples than
    another).
    """
    device = config.device
    model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    result = TrainResult()
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        elapsed = time.time() - start
        result.history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": elapsed,
        })

        if verbose:
            print(
                f"epoch {epoch + 1}/{config.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"({elapsed:.1f}s)"
            )

        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_val_accuracy = val_acc
            epochs_without_improvement = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        result.epochs_run = epoch + 1

        if epochs_without_improvement >= config.patience:
            result.stopped_early = True
            if verbose:
                print(f"early stopping at epoch {epoch + 1}")
            break

    return result
