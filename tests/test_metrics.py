"""Testes unitários para métricas de tags (multirrótulo) e dificuldade (incluindo QWK)."""

import numpy as np
import pytest

from tcc.evaluate import (
    compute_difficulty_metrics,
    compute_tag_metrics,
    evaluate_predictions,
    tune_per_class_thresholds,
)


def test_compute_tag_metrics_exact_values():
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    y_probs = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.1], [0.8, 0.9, 0.2]], dtype=np.float32)

    m = compute_tag_metrics(y_true, y_probs, threshold=0.5)
    assert m["precision_micro"] == pytest.approx(1.0)
    assert m["recall_micro"] == pytest.approx(1.0)
    assert m["f1_micro"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["hamming_loss"] == pytest.approx(0.0)
    assert m["subset_accuracy"] == pytest.approx(1.0)


def test_tune_per_class_thresholds_improves_or_matches():
    y_val = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    probs_val = np.array([[0.35, 0.1], [0.40, 0.2], [0.1, 0.8], [0.2, 0.9]], dtype=np.float32)

    thrs = tune_per_class_thresholds(y_val, probs_val, step=0.05)
    assert thrs.shape == (2,)
    assert thrs[0] <= 0.35
    m_tuned = compute_tag_metrics(y_val, probs_val, per_class_thresholds=thrs)
    assert m_tuned["f1_macro"] == pytest.approx(1.0)


def test_compute_difficulty_metrics_qwk_and_within_1():
    y_true = np.array([0, 1, 2, 3, 4])
    m_perfect = compute_difficulty_metrics(y_true, y_true, num_classes=5)
    assert m_perfect["accuracy"] == pytest.approx(1.0)
    assert m_perfect["f1_macro"] == pytest.approx(1.0)
    assert m_perfect["mae"] == pytest.approx(0.0)
    assert m_perfect["qwk"] == pytest.approx(1.0)
    assert m_perfect["acc_within_1"] == pytest.approx(1.0)
    assert len(m_perfect["confusion_matrix"]) == 5
    assert len(m_perfect["confusion_matrix"][0]) == 5

    # Erro leve (±1) deve ter QWK maior que erro grave (±4)
    y_near = np.array([1, 1, 2, 3, 3])
    y_far = np.array([4, 1, 2, 3, 0])
    m_near = compute_difficulty_metrics(y_true, y_near, num_classes=5)
    m_far = compute_difficulty_metrics(y_true, y_far, num_classes=5)
    assert m_near["acc_within_1"] == pytest.approx(1.0)
    assert m_far["acc_within_1"] == pytest.approx(0.6)
    assert m_near["qwk"] > m_far["qwk"]
    assert m_near["mae"] < m_far["mae"]


def test_evaluate_predictions_combines_both_tasks():
    y_tags = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y_dif = np.array([0, 4])
    out = evaluate_predictions(y_tags, y_tags, y_dif, y_dif)
    assert "tags" in out and "difficulty" in out
    assert out["tags"]["f1_micro"] == pytest.approx(1.0)
    assert out["difficulty"]["qwk"] == pytest.approx(1.0)
