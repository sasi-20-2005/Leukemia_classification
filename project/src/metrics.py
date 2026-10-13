"""
metrics.py
----------
Proposal Section 19: Evaluation Metrics.
Accuracy, Precision, Recall, F1-Score, ROC-AUC (one-vs-rest), Confusion Matrix,
and a stratified k-fold cross-validation helper.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                         class_names: list[str]):
    n_classes = len(class_names)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2:
            metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
            )
    except Exception:
        metrics["roc_auc_macro"] = None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    report = classification_report(
        y_true, y_pred, labels=list(range(n_classes)), target_names=class_names,
        zero_division=0, output_dict=True,
    )
    metrics["confusion_matrix"] = cm.tolist()
    metrics["per_class_report"] = report
    return metrics


def cross_validate_summary(build_and_train_fn, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                            random_state: int = 42):
    """
    Generic stratified k-fold CV driver.
    `build_and_train_fn(X_train, y_train, X_val, y_val)` must return (y_val, y_pred, y_proba).
    Returns per-fold accuracy/F1 plus mean +/- std (Section 19: "Cross-validation should
    also be considered because of the relatively small number of samples.")
    """
    n_splits = min(n_splits, np.min(np.bincount(y)))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_acc, fold_f1 = [], []
    for train_idx, val_idx in skf.split(X, y):
        y_val, y_pred, _ = build_and_train_fn(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        fold_acc.append(accuracy_score(y_val, y_pred))
        fold_f1.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
    return {
        "n_splits": n_splits,
        "fold_accuracy": fold_acc,
        "fold_f1_macro": fold_f1,
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "mean_f1_macro": float(np.mean(fold_f1)),
        "std_f1_macro": float(np.std(fold_f1)),
    }
