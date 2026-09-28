"""
forensiq/evaluation/image_metrics.py

Evaluation for UnivFD-style models, which output a single logit per
sample (scored via sigmoid), unlike the two-logit softmax convention
used by the Audio and Video agents. Kept separate from metrics.py to
avoid changing the already-working evaluation path for those agents.

Label convention confirmed from image_gen_dataset.py / the dataset
itself: 0 = real, 1 = fake.
"""
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_univfd_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).flatten().cpu()
            preds = (probs > 0.5).long()

            all_labels.extend(y.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs),
    }
