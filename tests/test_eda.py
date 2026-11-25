"""Testes unitários para funções reutilizáveis de EDA e exportação de tabelas."""

import pandas as pd

from tcc.eda import (
    compute_difficulty_distribution,
    compute_trivial_baselines,
    propose_input_format_budget,
    save_table_csv_and_tex,
)


def test_eda_distribution_baselines_and_table_export(tmp_path):
    train_df = pd.DataFrame(
        {
            "difficulty_class": [0, 0, 1, 2],
            "clean_tags": [["dp", "math"], ["dp"], ["dp", "greedy"], ["math"]],
        }
    )
    val_df = pd.DataFrame(
        {
            "difficulty_class": [0, 1],
            "clean_tags": [["dp"], ["math"]],
        }
    )
    test_df = pd.DataFrame(
        {
            "difficulty_class": [0, 1, 2],
            "clean_tags": [["dp"], ["dp", "math"], ["greedy"]],
        }
    )
    vocab = ["dp", "greedy", "math"]

    dist_df = compute_difficulty_distribution(train_df, val_df, test_df)
    assert len(dist_df) == 5
    assert dist_df["treino_n"].sum() == 4
    assert dist_df["teste_n"].sum() == 3

    csv_path, tex_path = save_table_csv_and_tex(dist_df, tmp_path, "dist_test")
    assert csv_path.exists()
    assert tex_path.exists()
    tex_content = tex_path.read_text(encoding="utf-8")
    assert r"\begin{tabular}" in tex_content
    assert r"\end{tabular}" in tex_content

    baselines = compute_trivial_baselines(train_df, test_df, vocab)
    assert baselines["difficulty"]["majority_class"] == 0
    assert 0.0 <= baselines["difficulty"]["accuracy"] <= 1.0
    assert baselines["tags"]["n_tags_predicted_positive"] == 2  # dp (75%) e math (50%)

    rec_ok = propose_input_format_budget(pd.DataFrame({"input_fmt_p95": [80.0, 110.0]}), current_budget=128)
    assert "Manter 128" in rec_ok
    rec_up = propose_input_format_budget(pd.DataFrame({"input_fmt_p95": [140.0, 175.0]}), current_budget=128)
    assert "176" in rec_up
