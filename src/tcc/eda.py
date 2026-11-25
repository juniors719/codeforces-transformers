"""Funções reutilizáveis de Análise Exploratória de Dados (EDA) e exportação .csv/.tex/.pdf/.png."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)

from tcc.labels import DEFAULT_DIFFICULTY_NAMES, binarize_tags
from tcc.tokenization import compute_token_length_stats


def _escape_latex(val: Any) -> str:
    text = str(val)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
    ]:
        text = text.replace(old, new)
    return text


def save_table_csv_and_tex(
    df: pd.DataFrame,
    tables_dir: str | Path,
    stem: str,
    float_fmt: str = "{:.2f}",
) -> tuple[Path, Path]:
    """Salva uma tabela em .csv e em .tex (tabular simples, sem dependência de template)."""
    out_dir = Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    tex_path = out_dir / f"{stem}.tex"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    col_spec = "l" + "r" * max(len(df.columns) - 1, 0)
    header = " & ".join(_escape_latex(c) for c in df.columns) + r" \\"
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        header,
        r"\hline",
    ]
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                cells.append(_escape_latex(float_fmt.format(val)))
            else:
                cells.append(_escape_latex(val))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])

    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, tex_path


def compute_difficulty_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    difficulty_names: list[str] | None = None,
) -> pd.DataFrame:
    """Calcula tabela de distribuição das 5 faixas de dificuldade em treino, validação e teste."""
    names = difficulty_names or DEFAULT_DIFFICULTY_NAMES
    rows = []
    for cls_idx, label in enumerate(names):
        tr_n = int((train_df["difficulty_class"] == cls_idx).sum())
        va_n = int((val_df["difficulty_class"] == cls_idx).sum())
        te_n = int((test_df["difficulty_class"] == cls_idx).sum())
        rows.append(
            {
                "classe": cls_idx,
                "faixa_rating": label,
                "treino_n": tr_n,
                "treino_pct": 100.0 * tr_n / max(len(train_df), 1),
                "val_n": va_n,
                "val_pct": 100.0 * va_n / max(len(val_df), 1),
                "teste_n": te_n,
                "teste_pct": 100.0 * te_n / max(len(test_df), 1),
            }
        )
    return pd.DataFrame(rows)


def compute_tag_cooccurrence(
    df: pd.DataFrame,
    tag_vocab: list[str],
    exclude_prefix: str = "*",
) -> pd.DataFrame:
    """Calcula a matriz de co-ocorrência (simétrica) entre tags selecionadas."""
    mat = binarize_tags(df["clean_tags"], tag_vocab, exclude_prefix=exclude_prefix)
    cooc = (mat.T @ mat).astype(int)
    return pd.DataFrame(cooc, index=tag_vocab, columns=tag_vocab)


def compute_tags_per_problem_stats(
    df: pd.DataFrame,
    tag_vocab: list[str],
    exclude_prefix: str = "*",
) -> dict[str, float]:
    """Estatísticas do número de tags selecionadas por problema."""
    mat = binarize_tags(df["clean_tags"], tag_vocab, exclude_prefix=exclude_prefix)
    counts = mat.sum(axis=1)
    return {
        "media": float(np.mean(counts)),
        "mediana": float(np.median(counts)),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "pct_zero_tags": float(100.0 * np.mean(counts == 0)),
    }


def compute_trivial_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tag_vocab: list[str],
    exclude_prefix: str = "*",
) -> dict[str, Any]:
    """Estima baselines triviais (classe majoritária) para dificuldade e tags no conjunto de teste."""
    # Dificuldade: sempre prever a classe mais frequente no treino
    maj_class = int(train_df["difficulty_class"].mode().iloc[0])
    y_dif_true = test_df["difficulty_class"].to_numpy(dtype=int)
    y_dif_pred = np.full_like(y_dif_true, fill_value=maj_class)

    dif_baseline = {
        "majority_class": maj_class,
        "accuracy": float(accuracy_score(y_dif_true, y_dif_pred)),
        "f1_macro": float(f1_score(y_dif_true, y_dif_pred, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_dif_true, y_dif_pred)),
        "qwk": float(cohen_kappa_score(y_dif_true, y_dif_pred, weights="quadratic")),
        "acc_within_1": float(np.mean(np.abs(y_dif_true - y_dif_pred) <= 1)),
    }

    # Tags: prever 1 para cada tag cuja prevalência no treino é >= 0.5, senão 0
    y_tags_train = binarize_tags(train_df["clean_tags"], tag_vocab, exclude_prefix=exclude_prefix)
    y_tags_test = binarize_tags(test_df["clean_tags"], tag_vocab, exclude_prefix=exclude_prefix)
    tag_prevalence = y_tags_train.mean(axis=0)
    maj_tag_vec = (tag_prevalence >= 0.5).astype(np.float32)
    y_tags_pred = np.tile(maj_tag_vec, (len(y_tags_test), 1))

    tags_baseline = {
        "n_tags_predicted_positive": int(maj_tag_vec.sum()),
        "precision_micro": float(precision_score(y_tags_test, y_tags_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_tags_test, y_tags_pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_tags_test, y_tags_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_tags_test, y_tags_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_tags_test, y_tags_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_tags_test, y_tags_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_tags_test, y_tags_pred)),
        "subset_accuracy": float(accuracy_score(y_tags_test, y_tags_pred)),
    }

    return {"difficulty": dif_baseline, "tags": tags_baseline}


def analyze_token_lengths_by_tokenizer(
    df: pd.DataFrame,
    model_specs: dict[str, dict[str, Any]],
    fields: list[str],
    difficulty_names: list[str] | None = None,
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Calcula estatísticas de comprimento em tokens por tokenizador (geral e por faixa de dificuldade)."""
    from transformers import AutoTokenizer

    names = difficulty_names or DEFAULT_DIFFICULTY_NAMES
    overall_rows: list[dict[str, Any]] = []
    by_class_rows: list[dict[str, Any]] = []
    raw_lengths: dict[str, np.ndarray] = {}

    for model_key, spec in model_specs.items():
        if model_key == "bert-tiny":
            continue
        hf_name = spec["hf_name"]
        display_name = spec.get("display_name", model_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir=cache_dir)

        overall = compute_token_length_stats(df, tokenizer, fields=fields, threshold=512)
        overall_rows.append(
            {
                "modelo": display_name,
                "media": overall["mean"],
                "mediana": overall["median"],
                "p90": overall["p90"],
                "p95": overall["p95"],
                "p99": overall["p99"],
                "max": overall["max"],
                "pct_gt_512": overall["pct_gt_512"],
                "pct_gt_2048": overall["pct_gt_2048"],
                "input_fmt_p95": overall.get("input_format_p95", np.nan),
                "input_fmt_p99": overall.get("input_format_p99", np.nan),
            }
        )

        # Guarda comprimentos para histogramas
        from tcc.tokenization import tokenize_example

        lengths = [
            tokenize_example(row, tokenizer, max_length=100_000, fields=fields, strategy="right")[
                "original_token_count"
            ]
            for row in df.to_dict(orient="records")
        ]
        raw_lengths[display_name] = np.asarray(lengths, dtype=int)

        # Estatísticas por faixa de dificuldade
        for cls_idx, cls_name in enumerate(names):
            sub_df = df[df["difficulty_class"] == cls_idx]
            if len(sub_df) == 0:
                continue
            cls_stats = compute_token_length_stats(sub_df, tokenizer, fields=fields, threshold=512)
            by_class_rows.append(
                {
                    "modelo": display_name,
                    "faixa_rating": cls_name,
                    "n": cls_stats["n_samples"],
                    "mediana": cls_stats["median"],
                    "p95": cls_stats["p95"],
                    "pct_gt_512": cls_stats["pct_gt_512"],
                    "pct_gt_2048": cls_stats["pct_gt_2048"],
                }
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(by_class_rows), raw_lengths


def propose_input_format_budget(overall_length_df: pd.DataFrame, current_budget: int = 128) -> str:
    """Analisa o p95 de tokens do input_format entre os tokenizadores e gera recomendação objetiva."""
    if "input_fmt_p95" not in overall_length_df.columns or overall_length_df["input_fmt_p95"].isna().all():
        return "Campo input_format não analisado."
    max_p95 = float(overall_length_df["input_fmt_p95"].max())
    min_p95 = float(overall_length_df["input_fmt_p95"].min())
    if max_p95 <= current_budget:
        return (
            f"O p95 de input_format varia entre {min_p95:.1f} e {max_p95:.1f} tokens "
            f"(abaixo da reserva atual de {current_budget} tokens). Manter {current_budget} tokens é adequado."
        )
    suggested = int(np.ceil(max_p95 / 16.0) * 16)
    return (
        f"O p95 de input_format chega a {max_p95:.1f} tokens (acima da reserva atual de {current_budget}). "
        f"Considere ajustar reserve_input_format_tokens para {suggested} em configs/base.yaml."
    )


def plot_difficulty_distribution(
    dist_df: pd.DataFrame,
    figures_dir: str | Path,
    stem: str = "eda_difficulty_distribution",
) -> tuple[Path, Path]:
    """Gera gráfico de barras da distribuição das 5 faixas em treino, validação e teste."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(dist_df))
    width = 0.25
    ax.bar(x - width, dist_df["treino_pct"], width, label="Treino")
    ax.bar(x, dist_df["val_pct"], width, label="Validação")
    ax.bar(x + width, dist_df["teste_pct"], width, label="Teste")

    ax.set_xticks(x)
    ax.set_xticklabels(dist_df["faixa_rating"])
    ax.set_ylabel("Proporção (%)")
    ax.set_xlabel("Faixa de dificuldade (rating)")
    ax.set_title("Distribuição das faixas de dificuldade por partição")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return pdf_path, png_path


def plot_token_length_histograms(
    raw_lengths: dict[str, np.ndarray],
    figures_dir: str | Path,
    stem: str = "eda_token_lengths",
) -> tuple[Path, Path]:
    """Gera histogramas de comprimento em tokens para cada tokenizador com linhas em 512 e 2048."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_models = len(raw_lengths)
    fig, axes = plt.subplots(1, max(n_models, 1), figsize=(4.2 * max(n_models, 1), 3.8), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, lengths) in zip(axes, raw_lengths.items()):
        clipped = np.clip(lengths, 0, 2500)
        ax.hist(clipped, bins=40, alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.axvline(512, color="red", linestyle="--", linewidth=1.2, label="512")
        ax.axvline(2048, color="green", linestyle=":", linewidth=1.2, label="2048")
        ax.set_title(model_name)
        ax.set_xlabel("Tokens (limitado a 2500)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Número de problemas")
    fig.suptitle("Distribuição do comprimento dos enunciados por tokenizador", y=1.02)
    fig.tight_layout()

    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path
