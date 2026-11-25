"""Análise estatística dos experimentos sobre as 10 seeds por modelo/configuração.

Metodologia:
  - Resumo por grupo: média, desvio padrão e IC 95% por bootstrap sobre as seeds.
  - Normalidade: Shapiro-Wilk por grupo (reportado com ressalva do baixo poder para n=10).
  - Teste global: Kruskal-Wallis entre grupos.
  - Comparações par a par: Mann-Whitney U (amostras independentes), correção de Holm
    e tamanho de efeito Cliff's delta.
  - Contrastes planejados destacados conforme base.yaml.
  - Opcional: teste de McNemar entre pares de modelos usando as predições salvas (test_preds.npz).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PRIMARY_METRICS: list[tuple[str, str, str]] = [
    # (chave_plana, grupo_em_metrics_json, nome_exibicao)
    ("dif_accuracy", "difficulty", "Dificuldade: Acurácia"),
    ("dif_f1_macro", "difficulty", "Dificuldade: F1-macro"),
    ("dif_mae", "difficulty", "Dificuldade: MAE"),
    ("dif_qwk", "difficulty", "Dificuldade: QWK"),
    ("dif_acc_within_1", "difficulty", "Dificuldade: Acc@±1"),
    ("tags_precision_micro", "tags", "Tags: Precision-micro"),
    ("tags_recall_micro", "tags", "Tags: Recall-micro"),
    ("tags_f1_micro", "tags", "Tags: F1-micro"),
    ("tags_precision_macro", "tags", "Tags: Precision-macro"),
    ("tags_recall_macro", "tags", "Tags: Recall-macro"),
    ("tags_f1_macro", "tags", "Tags: F1-macro"),
    ("tags_hamming_loss", "tags", "Tags: Hamming Loss"),
    ("tags_subset_accuracy", "tags", "Tags: Subset Accuracy"),
]


def load_run_records(runs_dir: str | Path, experiment_name: str = "main") -> pd.DataFrame:
    """Lê todos os metrics.json concluídos em results/runs/<experiment_name>/."""
    exp_dir = Path(runs_dir) / experiment_name
    if not exp_dir.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(exp_dir.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model_key = data["model_key"]
        max_len = int(data["max_length"])
        task_mode = data.get("task_mode", "mtl")
        suffix = f"_{task_mode}" if task_mode and task_mode != "mtl" else ""
        config_id = f"{model_key}_len{max_len}{suffix}"

        row: dict[str, Any] = {
            "run_dir": str(metrics_path.parent),
            "config_id": config_id,
            "display_label": f"{data.get('display_name', model_key)} ({max_len})",
            "model_key": model_key,
            "max_length": max_len,
            "seed": int(data["seed"]),
            "task_mode": task_mode,
            "difficulty_head_type": data.get("difficulty_head_type", "classification"),
            "best_epoch": data.get("best_epoch"),
            "best_val_loss": data.get("best_val_loss"),
            "runtime_seconds": data.get("runtime_seconds"),
        }

        dif_metrics = data.get("metrics", {}).get("difficulty", {})
        tag_metrics = data.get("metrics", {}).get("tags", {})

        for flat_key, group_name, _ in PRIMARY_METRICS:
            src_dict = dif_metrics if group_name == "difficulty" else tag_metrics
            sub_key = flat_key.replace("dif_", "", 1) if group_name == "difficulty" else flat_key.replace("tags_", "", 1)
            row[flat_key] = src_dict.get(sub_key, np.nan)

        row["confusion_matrix"] = dif_metrics.get("confusion_matrix")
        rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Calcula intervalo de confiança (1 - alpha) por bootstrap percentil sobre a média."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    low = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Calcula o tamanho de efeito não-paramétrico Cliff's delta e sua magnitude (Romano et al.)."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), "indefinido"

    diff = a[:, None] - b[None, :]
    delta = float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(a) * len(b)))
    abs_d = abs(delta)
    if abs_d < 0.147:
        mag = "desprezível"
    elif abs_d < 0.33:
        mag = "pequeno"
    elif abs_d < 0.474:
        mag = "médio"
    else:
        mag = "grande"
    return delta, mag


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Aplica correção de Holm-Bonferroni para múltiplas comparações."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, (m - rank) * float(p))
        running_max = max(running_max, val)
        adjusted[orig_idx] = running_max
    return adjusted


def summarize_by_model(
    runs_df: pd.DataFrame,
    metrics: list[str] | None = None,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Calcula média, desvio padrão e IC 95% por bootstrap para cada modelo/configuração."""
    if runs_df.empty:
        return pd.DataFrame()

    target_metrics = metrics or [m[0] for m in PRIMARY_METRICS]
    rows: list[dict[str, Any]] = []

    for config_id, grp in runs_df.groupby("config_id", sort=False):
        label = grp["display_label"].iloc[0]
        n_seeds = int(len(grp))
        for metric in target_metrics:
            vals = grp[metric].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                continue
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_low, ci_high = bootstrap_ci(vals, n_resamples=n_resamples, alpha=alpha, seed=seed)
            rows.append(
                {
                    "config_id": config_id,
                    "modelo": label,
                    "n_seeds": n_seeds,
                    "metrica": metric,
                    "media": mean_v,
                    "desvio_padrao": std_v,
                    "ic95_inf": ci_low,
                    "ic95_sup": ci_high,
                }
            )
    return pd.DataFrame(rows)


def shapiro_wilk_by_group(runs_df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Executa Shapiro-Wilk por modelo/configuração (com n=10 o poder estatístico é limitado)."""
    if runs_df.empty:
        return pd.DataFrame()

    target_metrics = metrics or [m[0] for m in PRIMARY_METRICS]
    rows: list[dict[str, Any]] = []
    for config_id, grp in runs_df.groupby("config_id", sort=False):
        label = grp["display_label"].iloc[0]
        for metric in target_metrics:
            vals = grp[metric].dropna().to_numpy(dtype=np.float64)
            if len(vals) < 3 or np.ptp(vals) == 0:
                stat, p_val = float("nan"), float("nan")
            else:
                res = scipy_stats.shapiro(vals)
                stat, p_val = float(res.statistic), float(res.pvalue)
            rows.append(
                {
                    "config_id": config_id,
                    "modelo": label,
                    "metrica": metric,
                    "n": int(len(vals)),
                    "shapiro_W": stat,
                    "p_valor": p_val,
                }
            )
    return pd.DataFrame(rows)


def kruskal_wallis_global(
    runs_df: pd.DataFrame,
    metrics: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Executa teste global não-paramétrico de Kruskal-Wallis entre os modelos para cada métrica."""
    if runs_df.empty:
        return pd.DataFrame()

    target_metrics = metrics or [m[0] for m in PRIMARY_METRICS]
    rows: list[dict[str, Any]] = []
    for metric in target_metrics:
        groups = [
            grp[metric].dropna().to_numpy(dtype=np.float64)
            for _, grp in runs_df.groupby("config_id", sort=False)
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        try:
            stat, p_val = scipy_stats.kruskal(*groups)
            h_stat, p_v = float(stat), float(p_val)
        except ValueError:
            h_stat, p_v = 0.0, 1.0
        rows.append(
            {
                "metrica": metric,
                "n_grupos": len(groups),
                "H_kruskal": h_stat,
                "p_valor": p_v,
                "significativo": bool(p_v < alpha),
            }
        )
    return pd.DataFrame(rows)


def pairwise_mann_whitney_holm(
    runs_df: pd.DataFrame,
    metrics: list[str] | None = None,
    planned_contrasts: list[dict[str, str]] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compara pares de modelos com Mann-Whitney U, correção de Holm (por métrica) e Cliff's delta."""
    if runs_df.empty:
        return pd.DataFrame()

    target_metrics = metrics or [m[0] for m in PRIMARY_METRICS]
    planned_pairs: dict[tuple[str, str], str] = {}
    for c in planned_contrasts or []:
        a, b = c["model_a"], c["model_b"]
        planned_pairs[(a, b)] = c["name"]
        planned_pairs[(b, a)] = c["name"]

    configs = list(runs_df["config_id"].drop_duplicates())
    if len(configs) < 2:
        return pd.DataFrame()

    all_rows: list[dict[str, Any]] = []
    for metric in target_metrics:
        metric_rows: list[dict[str, Any]] = []
        raw_pvals: list[float] = []

        for cfg_a, cfg_b in combinations(configs, 2):
            vals_a = runs_df.loc[runs_df["config_id"] == cfg_a, metric].dropna().to_numpy(dtype=np.float64)
            vals_b = runs_df.loc[runs_df["config_id"] == cfg_b, metric].dropna().to_numpy(dtype=np.float64)
            if len(vals_a) == 0 or len(vals_b) == 0:
                continue

            u_res = scipy_stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            delta, mag = cliffs_delta(vals_a, vals_b)
            contrast_name = planned_pairs.get((cfg_a, cfg_b), "")

            raw_pvals.append(float(u_res.pvalue))
            metric_rows.append(
                {
                    "metrica": metric,
                    "modelo_a": cfg_a,
                    "modelo_b": cfg_b,
                    "contraste_planejado": contrast_name,
                    "media_a": float(np.mean(vals_a)),
                    "media_b": float(np.mean(vals_b)),
                    "U_stat": float(u_res.statistic),
                    "p_valor_bruto": float(u_res.pvalue),
                    "cliffs_delta": delta,
                    "magnitude_efeito": mag,
                }
            )

        adj_pvals = holm_bonferroni(raw_pvals)
        for row, p_adj in zip(metric_rows, adj_pvals):
            row["p_valor_holm"] = p_adj
            row["significativo_holm"] = bool(p_adj < alpha)
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def mcnemar_test_from_preds(npz_a: str | Path, npz_b: str | Path) -> dict[str, Any]:
    """Teste de McNemar pareado no conjunto de teste para acurácia da dificuldade entre dois runs."""
    da = np.load(npz_a)
    db = np.load(npz_b)
    y_true = da["y_dif_true"]
    correct_a = da["y_dif_pred"] == y_true
    correct_b = db["y_dif_pred"] == y_true

    # b: A acerta, B erra; c: A erra, B acerta
    n01 = int(np.sum(correct_a & (~correct_b)))
    n10 = int(np.sum((~correct_a) & correct_b))
    total_disc = n01 + n10
    if total_disc == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_valor": 1.0}

    chi2 = ((abs(n01 - n10) - 1) ** 2) / total_disc
    p_val = float(scipy_stats.chi2.sf(chi2, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(chi2), "p_valor": p_val}
