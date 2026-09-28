"""
Evaluation metrics for ForensiQ models.

Computes a full metrics report (accuracy, precision, recall, F1, AUC-ROC)
from model predictions, rather than relying on accuracy alone, since
class-imbalanced datasets can produce misleading accuracy scores.
"""
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(model, data_loader, device):
    """
    Runs the model over data_loader and returns a dictionary of metrics.
    Assumes binary classification with class index 1 as the positive class.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs),
    }


def compute_eer(labels, scores):
    """
    Computes Equal Error Rate (EER), the standard metric used in the
    ASVspoof audio deepfake detection literature, for direct comparability
    with published baselines.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold_idx = (fpr - fnr).__abs__().argmin()
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    return eer
