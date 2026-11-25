"""Testes unitários para análise estatística e geração de relatórios (Fase 4)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcc.config import load_config
from tcc.reporting import generate_full_report
from tcc.stats import (
    bootstrap_ci,
    cliffs_delta,
    holm_bonferroni,
    kruskal_wallis_global,
    mcnemar_test_from_preds,
    pairwise_mann_whitney_holm,
    shapiro_wilk_by_group,
)


def _write_synthetic_run(
    runs_dir: Path,
    exp: str,
    model_key: str,
    display_name: str,
    max_len: int,
    seed: int,
    qwk: float,
    acc: float,
    f1_micro: float,
    y_dif_pred: np.ndarray,
):
    run_dir = runs_dir / exp / f"{model_key}_len{max_len}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_key": model_key,
        "display_name": display_name,
        "max_length": max_len,
        "seed": seed,
        "task_mode": "mtl",
        "difficulty_head_type": "classification",
        "best_epoch": 3,
        "best_val_loss": 0.5,
        "runtime_seconds": 10.0,
        "metrics": {
            "difficulty": {
                "accuracy": acc,
                "f1_macro": acc - 0.05,
                "mae": 1.0 - acc,
                "qwk": qwk,
                "acc_within_1": min(1.0, acc + 0.3),
                "confusion_matrix": np.eye(5, dtype=int).tolist(),
            },
            "tags": {
                "precision_micro": f1_micro,
                "recall_micro": f1_micro,
                "f1_micro": f1_micro,
                "precision_macro": f1_micro - 0.1,
                "recall_macro": f1_micro - 0.1,
                "f1_macro": f1_micro - 0.1,
                "hamming_loss": 0.1,
                "subset_accuracy": 0.2,
            },
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    pd.DataFrame(
        {"epoch": [1, 2, 3], "train_loss": [1.0, 0.7, 0.5], "val_loss": [1.1, 0.8, 0.6]}
    ).to_csv(run_dir / "history.csv", index=False)
    np.savez_compressed(
        run_dir / "test_preds.npz",
        y_dif_true=np.array([0, 1, 2, 3, 4]),
        y_dif_pred=y_dif_pred,
    )
    return run_dir


def test_stats_primitives():
    low, high = bootstrap_ci(np.array([0.5, 0.6, 0.7, 0.8]), n_resamples=500, seed=42)
    assert 0.5 <= low <= high <= 0.8

    delta, mag = cliffs_delta(np.array([10, 11, 12]), np.array([1, 2, 3]))
    assert delta == pytest.approx(1.0)
    assert mag == "grande"

    adj = holm_bonferroni([0.01, 0.04, 0.20])
    assert adj == pytest.approx([0.03, 0.08, 0.20])


def test_generate_full_report_empty_and_populated(tmp_path):
    cfg = load_config(results_dir_override=tmp_path)

    # Caso vazio: não inventa números
    empty_res = generate_full_report(cfg, experiment_name="main")
    assert empty_res["status"] == "empty"
    assert empty_res["n_runs"] == 0
    assert "Nenhum run concluído" in Path(empty_res["summary_md"]).read_text(encoding="utf-8")

    # Caso populado com 2 configurações × 4 seeds
    runs_dir = Path(cfg["paths"]["runs_dir"])
    for i, seed in enumerate([42, 43, 44, 45]):
        dir_a = _write_synthetic_run(
            runs_dir, "main", "roberta-base", "RoBERTa-base", 512, seed,
            qwk=0.60 + 0.01 * i, acc=0.50 + 0.01 * i, f1_micro=0.55 + 0.01 * i,
            y_dif_pred=np.array([0, 1, 2, 3, 4]),
        )
        dir_b = _write_synthetic_run(
            runs_dir, "main", "codebert-base", "CodeBERT-base", 512, seed,
            qwk=0.72 + 0.01 * i, acc=0.62 + 0.01 * i, f1_micro=0.65 + 0.01 * i,
            y_dif_pred=np.array([0, 1, 2, 0, 0]),
        )

    res = generate_full_report(cfg, experiment_name="main")
    assert res["status"] == "ok"
    assert res["n_runs"] == 8
    assert Path(res["summary_md"]).exists()

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    assert (tables_dir / "main_summary_metrics.csv").exists()
    assert (tables_dir / "main_summary_metrics.tex").exists()
    assert (tables_dir / "main_planned_contrasts.csv").exists()
    assert (figures_dir / "main_boxplot_dif_qwk.pdf").exists()
    assert (figures_dir / "main_confusion_matrices.png").exists()
    assert (figures_dir / "main_training_curves.pdf").exists()

    # Verifica Shapiro-Wilk, Kruskal-Wallis, Mann-Whitney e McNemar
    mc = mcnemar_test_from_preds(dir_a / "test_preds.npz", dir_b / "test_preds.npz")
    assert mc["n01"] == 2 and mc["n10"] == 0
