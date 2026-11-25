"""Testes unitários para filtragem, deduplicação por aliases e ausência de vazamento."""

import pandas as pd

from tcc.config import load_config
from tcc.data import prepare_splits_from_dataframes, verify_no_leakage


def _make_problem(pid, rating=1200, tags=None, desc="Statement text", aliases=None, year=2023):
    return {
        "id": pid,
        "aliases": aliases if aliases is not None else [],
        "contest_start_year": year,
        "title": f"Problem {pid}",
        "description": desc,
        "input_format": "1 <= n <= 10^5",
        "rating": rating,
        "tags": tags if tags is not None else ["dp", "greedy"],
    }


def test_prepare_splits_alias_dedup_conflict_reporting_and_no_leakage():
    cfg = load_config()
    cfg["labels"]["min_tag_freq"] = 1
    cfg["data"]["val_ratio"] = 0.2

    train_rows = [
        # Linhas inválidas que devem ser filtradas antes da deduplicação
        _make_problem("bad_rating", rating=None),
        _make_problem("bad_tags", tags=["*special"]),
        _make_problem("bad_desc", desc="   "),
        # Grupo de aliases no treino com conflito de rating e tags -> deve manter 100A (menor id)
        _make_problem("100B", rating=1500, tags=["math"], aliases=["100A"]),
        _make_problem("100A", rating=1200, tags=["dp"], aliases=["100B"]),
        # Grupo cujo membro também aparece no teste -> deve ser REMOVIDO do treino/val
        _make_problem("200A", rating=1800, tags=["graphs"], aliases=["200C"]),
        # Demais problemas independentes no treino para permitir split 80/20
        *[_make_problem(f"300{i}", rating=800 + 200 * (i % 5), tags=["dp"]) for i in range(10)],
    ]
    test_rows = [
        _make_problem("200C", rating=1800, tags=["graphs"], aliases=["200A"], year=2025),
        _make_problem("200D", rating=1800, tags=["graphs"], aliases=["200C"], year=2025),
        _make_problem("900A", rating=2400, tags=["dp"], year=2024),
    ]

    bundle = prepare_splits_from_dataframes(pd.DataFrame(train_rows), pd.DataFrame(test_rows), cfg)

    train_df = bundle["train"]
    val_df = bundle["val"]
    test_df = bundle["test"]

    # Verifica ausência total de vazamento
    verify_no_leakage(train_df, val_df, test_df)

    # 200A não pode existir em treino nem em validação (pois 200C está no teste)
    train_val_ids = set(train_df["id"]).union(set(val_df["id"]))
    assert "200A" not in train_val_ids
    assert "200C" in set(test_df["id"])
    assert "200D" not in set(test_df["id"])  # deduplicado no teste mantendo 200C (menor id)

    # Grupo 100A/100B deduplicado mantendo 100A (primeira por id)
    assert "100A" in train_val_ids
    assert "100B" not in train_val_ids

    # Relatório de conflitos de aliases
    report = bundle["alias_report"]
    assert report["groups_with_conflicting_rating"] == 1
    assert report["groups_with_conflicting_tags"] == 1
    assert report["groups_spanning_train_and_test"] == 1
    assert report["rows_removed_train_aliases_in_test"] == 1
    assert report["rows_removed_intra_split_dedup"] == 2
