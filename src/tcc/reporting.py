"""Geração de tabelas (.csv/.tex), figuras (.pdf/.png) e relatório summary.md com dados reais."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcc.eda import save_table_csv_and_tex
from tcc.labels import DEFAULT_DIFFICULTY_NAMES
from tcc.stats import (
    PRIMARY_METRICS,
    kruskal_wallis_global,
    load_run_records,
    pairwise_mann_whitney_holm,
    shapiro_wilk_by_group,
    summarize_by_model,
)


def plot_metric_boxplots(
    runs_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    figures_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    """Gera boxplot por modelo/configuração sobre as seeds em .pdf e .png."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    data = []
    for _, grp in runs_df.groupby("config_id", sort=False):
        vals = grp[metric].dropna().to_numpy(dtype=np.float64)
        if len(vals) > 0:
            labels.append(grp["display_label"].iloc[0])
            data.append(vals)

    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(labels)), 4.6))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return pdf_path, png_path


def plot_mean_confusion_matrices(
    runs_df: pd.DataFrame,
    figures_dir: str | Path,
    difficulty_names: list[str] | None = None,
    stem: str = "confusion_matrices_mean",
) -> tuple[Path, Path] | None:
    """Plota as matrizes de confusão médias (normalizadas por linha) para cada configuração."""
    names = difficulty_names or DEFAULT_DIFFICULTY_NAMES
    configs = [
        (cid, grp)
        for cid, grp in runs_df.groupby("config_id", sort=False)
        if grp["confusion_matrix"].notna().any()
    ]
    if not configs:
        return None

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cfg = len(configs)
    fig, axes = plt.subplots(1, n_cfg, figsize=(4.0 * n_cfg, 3.8))
    if n_cfg == 1:
        axes = [axes]

    for ax, (_, grp) in zip(axes, configs):
        mats = [np.asarray(m, dtype=np.float64) for m in grp["confusion_matrix"].dropna()]
        mean_mat = np.mean(mats, axis=0)
        row_sums = mean_mat.sum(axis=1, keepdims=True)
        norm_mat = np.divide(mean_mat, np.maximum(row_sums, 1e-8))

        im = ax.imshow(norm_mat, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(grp["display_label"].iloc[0], fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Predito", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Real", fontsize=8)

        for i in range(norm_mat.shape[0]):
            for j in range(norm_mat.shape[1]):
                val = norm_mat[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                    fontsize=7,
                )

    fig.colorbar(im, ax=axes, shrink=0.8, label="Proporção por classe real")
    fig.suptitle("Matrizes de confusão médias na tarefa de dificuldade", y=1.03)
    fig.tight_layout()

    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_training_curves(
    runs_df: pd.DataFrame,
    figures_dir: str | Path,
    stem: str = "training_curves",
) -> tuple[Path, Path] | None:
    """Plota curvas médias (± DP) de train_loss e val_loss por época para cada configuração."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = list(runs_df.groupby("config_id", sort=False))
    if not configs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    has_data = False

    for _, grp in configs:
        label = grp["display_label"].iloc[0]
        hist_frames = []
        for rdir in grp["run_dir"]:
            h_path = Path(rdir) / "history.csv"
            if h_path.exists():
                hist_frames.append(pd.read_csv(h_path))
        if not hist_frames:
            continue
        has_data = True
        cat = pd.concat(hist_frames, ignore_index=True)
        agg = cat.groupby("epoch")[["train_loss", "val_loss"]].agg(["mean", "std"]).reset_index()
        epochs = agg["epoch"].to_numpy()
        axes[0].plot(epochs, agg[("train_loss", "mean")], label=label)
        axes[1].plot(epochs, agg[("val_loss", "mean")], label=label)

    if not has_data:
        plt.close(fig)
        return None

    axes[0].set_title("Perda de treino média por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Perda de validação média por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return pdf_path, png_path


def generate_full_report(
    cfg: dict[str, Any],
    experiment_name: str = "main",
) -> dict[str, Any]:
    """Executa análise estatística completa e gera tabelas, figuras e results/summary.md."""
    runs_dir = Path(cfg["paths"]["runs_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    stats_cfg = cfg.get("stats", {})

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    runs_df = load_run_records(runs_dir, experiment_name=experiment_name)
    summary_md_path = results_dir / "summary.md"

    if runs_df.empty:
        summary_md_path.write_text(
            "# Relatório Experimental — TCC2 Transformers\n\n"
            f"Nenhum run concluído (`metrics.json`) encontrado para o experimento `{experiment_name}` em `{runs_dir}`.\n"
            "Execute `python -m tcc.runner --config configs/experiments/main.yaml` antes de gerar a análise.\n",
            encoding="utf-8",
        )
        return {"status": "empty", "summary_md": str(summary_md_path), "n_runs": 0}

    alpha = float(stats_cfg.get("alpha", 0.05))
    n_resamples = int(stats_cfg.get("bootstrap_resamples", 10_000))
    boot_seed = int(stats_cfg.get("bootstrap_seed", 42))
    planned_contrasts = stats_cfg.get("planned_contrasts", [])

    # 1. Tabelas estatísticas
    summary_df = summarize_by_model(runs_df, n_resamples=n_resamples, alpha=alpha, seed=boot_seed)
    shapiro_df = shapiro_wilk_by_group(runs_df)
    kruskal_df = kruskal_wallis_global(runs_df, alpha=alpha)
    pairwise_df = pairwise_mann_whitney_holm(
        runs_df,
        planned_contrasts=planned_contrasts,
        alpha=alpha,
    )

    save_table_csv_and_tex(summary_df, tables_dir, f"{experiment_name}_summary_metrics", float_fmt="{:.4f}")
    save_table_csv_and_tex(shapiro_df, tables_dir, f"{experiment_name}_shapiro_wilk", float_fmt="{:.4f}")
    if not kruskal_df.empty:
        save_table_csv_and_tex(kruskal_df, tables_dir, f"{experiment_name}_kruskal_wallis", float_fmt="{:.4f}")
    if not pairwise_df.empty:
        save_table_csv_and_tex(pairwise_df, tables_dir, f"{experiment_name}_pairwise_mann_whitney", float_fmt="{:.4f}")
        contrasts_only = pairwise_df[pairwise_df["contraste_planejado"] != ""]
        if not contrasts_only.empty:
            save_table_csv_and_tex(
                contrasts_only, tables_dir, f"{experiment_name}_planned_contrasts", float_fmt="{:.4f}"
            )

    # 2. Figuras principais
    key_plots = [
        ("dif_qwk", "Dificuldade: Quadratic Weighted Kappa (QWK)", "QWK", f"{experiment_name}_boxplot_dif_qwk"),
        ("dif_accuracy", "Dificuldade: Acurácia no Teste", "Acurácia", f"{experiment_name}_boxplot_dif_accuracy"),
        ("tags_f1_micro", "Tags Algorítmicas: F1-micro no Teste", "F1-micro", f"{experiment_name}_boxplot_tags_f1_micro"),
        ("tags_f1_macro", "Tags Algorítmicas: F1-macro no Teste", "F1-macro", f"{experiment_name}_boxplot_tags_f1_macro"),
    ]
    for metric, title, ylabel, stem in key_plots:
        plot_metric_boxplots(runs_df, metric, title, ylabel, figures_dir, stem)

    plot_mean_confusion_matrices(
        runs_df,
        figures_dir,
        difficulty_names=cfg.get("labels", {}).get("difficulty_names"),
        stem=f"{experiment_name}_confusion_matrices",
    )
    plot_training_curves(runs_df, figures_dir, stem=f"{experiment_name}_training_curves")

    # 3. Relatório summary.md com números reais observados
    lines = [
        f"# Relatório Experimental — `{experiment_name}`",
        "",
        f"- Total de runs concluídos carregados: **{len(runs_df)}**",
        f"- Configurações avaliadas: **{runs_df['config_id'].nunique()}**",
        f"- Nível de significância: `alpha = {alpha}` | Bootstrap: `{n_resamples}` reamostragens",
        "",
        "## 1. Resumo por modelo (média ± DP e IC 95% bootstrap)",
        "",
        "| Modelo | n | Dificuldade QWK | Dificuldade Acc | Dificuldade MAE | Tags F1-micro | Tags F1-macro |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def _fmt_cell(cid: str, metric: str) -> str:
        sub = summary_df[(summary_df["config_id"] == cid) & (summary_df["metrica"] == metric)]
        if sub.empty:
            return "—"
        r = sub.iloc[0]
        return f"{r['media']:.4f} ± {r['desvio_padrao']:.4f} [{r['ic95_inf']:.4f}, {r['ic95_sup']:.4f}]"

    for cid, grp in runs_df.groupby("config_id", sort=False):
        label = grp["display_label"].iloc[0]
        n_s = len(grp)
        lines.append(
            f"| {label} | {n_s} | "
            f"{_fmt_cell(cid, 'dif_qwk')} | "
            f"{_fmt_cell(cid, 'dif_accuracy')} | "
            f"{_fmt_cell(cid, 'dif_mae')} | "
            f"{_fmt_cell(cid, 'tags_f1_micro')} | "
            f"{_fmt_cell(cid, 'tags_f1_macro')} |"
        )

    if not kruskal_df.empty:
        lines.extend(
            [
                "",
                "## 2. Teste global de Kruskal-Wallis",
                "",
                "| Métrica | H | valor-p | Significativo (alpha=0.05) |",
                "|---|---:|---:|:---:|",
            ]
        )
        for _, r in kruskal_df.iterrows():
            lines.append(
                f"| `{r['metrica']}` | {r['H_kruskal']:.4f} | {r['p_valor']:.4e} | {'Sim' if r['significativo'] else 'Não'} |"
            )

    if not pairwise_df.empty:
        contrasts_only = pairwise_df[pairwise_df["contraste_planejado"] != ""]
        if not contrasts_only.empty:
            lines.extend(
                [
                    "",
                    "## 3. Contrastes planejados (Mann-Whitney U + Holm + Cliff's delta)",
                    "",
                    "| Contraste | Métrica | Média A | Média B | p-valor bruto | p-valor Holm | Cliff's delta | Magnitude |",
                    "|---|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            key_set = {"dif_qwk", "dif_accuracy", "dif_f1_macro", "tags_f1_micro", "tags_f1_macro"}
            for _, r in contrasts_only[contrasts_only["metrica"].isin(key_set)].iterrows():
                lines.append(
                    f"| {r['contraste_planejado']} | `{r['metrica']}` | {r['media_a']:.4f} | {r['media_b']:.4f} | "
                    f"{r['p_valor_bruto']:.4e} | {r['p_valor_holm']:.4e} | {r['cliffs_delta']:.4f} | {r['magnitude_efeito']} |"
                )

    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "summary_md": str(summary_md_path),
        "n_runs": int(len(runs_df)),
        "summary_df": summary_df,
        "kruskal_df": kruskal_df,
        "pairwise_df": pairwise_df,
    }
