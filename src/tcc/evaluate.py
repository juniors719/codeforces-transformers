"""Cálculo de métricas para tags (multirrótulo) e dificuldade (ordinal 5 classes)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)


def tune_per_class_thresholds(
    y_val_true: np.ndarray,
    y_val_probs: np.ndarray,
    step: float = 0.05,
    min_thr: float = 0.1,
    max_thr: float = 0.9,
) -> np.ndarray:
    """Ajusta o limiar ótimo por tag usando EXCLUSIVAMENTE o conjunto de validação."""
    grid = np.arange(min_thr, max_thr + 1e-9, step)
    n_tags = y_val_true.shape[1]
    best_thresholds = np.full(n_tags, 0.5, dtype=np.float32)

    for j in range(n_tags):
        y_col = y_val_true[:, j]
        if y_col.sum() == 0:
            best_thresholds[j] = 0.5
            continue
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            preds = (y_val_probs[:, j] >= t).astype(int)
            score = f1_score(y_col, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = float(t)
        best_thresholds[j] = best_t

    return best_thresholds


def compute_tag_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
    per_class_thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula Precision, Recall, F1 (micro e macro), Hamming Loss e Subset Accuracy."""
    y_true_bin = (y_true >= 0.5).astype(int)
    if per_class_thresholds is not None:
        thr = np.asarray(per_class_thresholds, dtype=np.float32)[None, :]
        y_pred_bin = (y_probs >= thr).astype(int)
    else:
        y_pred_bin = (y_probs >= threshold).astype(int)

    return {
        "precision_micro": float(precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true_bin, y_pred_bin)),
        "subset_accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
    }


def compute_difficulty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> dict[str, Any]:
    """Calcula Acurácia, F1-macro, MAE, Quadratic Weighted Kappa, Acc@±1 e matriz de confusão."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    # Evita NaN do cohen_kappa_score caso todas as predições e rótulos sejam idênticos em 1 classe
    if len(np.unique(y_t)) < 2 and len(np.unique(y_p)) < 2:
        qwk = 1.0 if np.array_equal(y_t, y_p) else 0.0
    else:
        qwk = float(cohen_kappa_score(y_t, y_p, weights="quadratic"))
        if np.isnan(qwk):
            qwk = 0.0

    cm = confusion_matrix(y_t, y_p, labels=list(range(num_classes)))

    return {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "f1_macro": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_t, y_p)),
        "qwk": float(qwk),
        "acc_within_1": float(np.mean(np.abs(y_t - y_p) <= 1)),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_predictions(
    y_tags_true: np.ndarray,
    y_tags_probs: np.ndarray,
    y_dif_true: np.ndarray,
    y_dif_pred: np.ndarray,
    tag_threshold: float = 0.5,
    per_class_thresholds: np.ndarray | None = None,
    num_classes: int = 5,
) -> dict[str, Any]:
    """Consolida métricas de tags (limiar 0.5 e opcional tunado na validação) e dificuldade."""
    result: dict[str, Any] = {
        "tags": compute_tag_metrics(y_tags_true, y_tags_probs, threshold=tag_threshold),
        "difficulty": compute_difficulty_metrics(y_dif_true, y_dif_pred, num_classes=num_classes),
    }
    if per_class_thresholds is not None:
        result["tags_tuned_val"] = compute_tag_metrics(
            y_tags_true,
            y_tags_probs,
            per_class_thresholds=per_class_thresholds,
        )
    return result
