"""Carregamento, filtragem, deduplicação por aliases e particionamento do open-r1/codeforces."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from tcc.cleaning import clean_dataframe_fields
from tcc.labels import normalize_tag_list, rating_to_class, select_tag_vocabulary

RELEVANT_COLUMNS = [
    "id",
    "aliases",
    "contest_start_year",
    "title",
    "description",
    "input_format",
    "rating",
    "tags",
]


class _UnionFind:
    """Estrutura Union-Find para agrupar problemas ligados por aliases (ex.: Div1/Div2)."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Mantém sempre o menor id lexicográfico como raiz canônica do grupo
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def _parse_aliases(raw_aliases: Any) -> list[str]:
    """Normaliza o campo aliases em uma lista de strings."""
    if raw_aliases is None:
        return []
    if isinstance(raw_aliases, float) and np.isnan(raw_aliases):
        return []
    if isinstance(raw_aliases, str):
        text = raw_aliases.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [text]
    if isinstance(raw_aliases, (list, tuple, set, np.ndarray)):
        return [str(x).strip() for x in raw_aliases if x is not None and str(x).strip()]
    return []


def assign_alias_groups(df: pd.DataFrame) -> pd.Series:
    """Atribui um identificador canônico de grupo para cada linha via componentes conexas de id/aliases."""
    uf = _UnionFind()
    ids = df["id"].astype(str).tolist()
    aliases_list = df["aliases"].tolist() if "aliases" in df.columns else [[] for _ in ids]

    for pid, raw_aliases in zip(ids, aliases_list):
        uf.find(pid)
        for alias in _parse_aliases(raw_aliases):
            uf.union(pid, alias)

    return pd.Series([uf.find(pid) for pid in ids], index=df.index, name="alias_group_id")


def _has_nonempty_description(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return bool(str(value).strip())


def prepare_splits_from_dataframes(
    raw_train_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Executa filtragem, deduplicação por aliases, split 90/10 e limpeza em memória.

    Ordem das etapas:
      1. Filtrar linhas sem rating, sem tags válidas (excluindo '*') ou com description vazia.
      2. Construir grupos de aliases (Union-Find) sobre os problemas válidos.
      3. Reportar conflitos de rating e tags dentro de grupos de aliases.
      4. Se um grupo tem membro no teste, remover os demais membros de treino/validação.
      5. Deduplicar cada grupo mantendo a primeira ocorrência ordenada por id.
      6. Dividir treino restante em treino (90%) e validação (10%) respeitando grupos de aliases.
      7. Verificar ausência total de vazamento e aplicar limpeza textual configurada.
    """
    data_cfg = cfg.get("data", {})
    labels_cfg = cfg.get("labels", {})
    cleaning_cfg = cfg.get("cleaning", {})
    tok_cfg = cfg.get("tokenization", {})

    exclude_prefix = labels_cfg.get("exclude_tag_prefix", "*")
    rating_bounds = labels_cfg.get("rating_upper_bounds", [1100, 1500, 1900, 2300])
    min_tag_freq = int(labels_cfg.get("min_tag_freq", 50))
    val_ratio = float(data_cfg.get("val_ratio", 0.10))
    split_seed = int(data_cfg.get("split_seed", 42))
    drop_if_in_test = bool(data_cfg.get("drop_train_val_if_alias_in_test", True))

    train_part = raw_train_df.copy()
    test_part = raw_test_df.copy()
    train_part["hf_split"] = "train"
    test_part["hf_split"] = "test"
    df = pd.concat([train_part, test_part], ignore_index=True)

    # Garante colunas esperadas
    for col in RELEVANT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    filter_steps: list[dict[str, Any]] = []

    def _record_step(step_name: str, frame: pd.DataFrame) -> None:
        n_tr = int((frame["hf_split"] == "train").sum())
        n_te = int((frame["hf_split"] == "test").sum())
        filter_steps.append(
            {
                "step": step_name,
                "total": int(len(frame)),
                "hf_train": n_tr,
                "hf_test": n_te,
            }
        )

    _record_step("0_raw_dataset", df)

    # 1. Remover sem rating
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating_num"].notna()].copy()
    _record_step("1_valid_rating", df)

    # 2. Normalizar tags e remover linhas sem tags válidas (desconsiderando prefixo '*')
    df["clean_tags"] = df["tags"].map(lambda x: normalize_tag_list(x, exclude_prefix=exclude_prefix))
    df = df[df["clean_tags"].map(len) > 0].copy()
    _record_step("2_valid_tags", df)

    # 3. Remover linhas com description vazia
    df = df[df["description"].map(_has_nonempty_description)].copy()
    _record_step("3_nonempty_description", df)

    # 4. Agrupamento por aliases (Union-Find) após filtros de validade
    df["id"] = df["id"].astype(str).str.strip()
    df["alias_group_id"] = assign_alias_groups(df)

    # Relatório de grupos e conflitos antes da deduplicação
    group_sizes = df.groupby("alias_group_id").size()
    multi_groups = group_sizes[group_sizes > 1].index.tolist()

    rating_conflict_groups = 0
    tags_conflict_groups = 0
    cross_split_groups = 0

    grouped = df[df["alias_group_id"].isin(multi_groups)].groupby("alias_group_id")
    for _, grp in grouped:
        if grp["rating_num"].nunique() > 1:
            rating_conflict_groups += 1
        tag_tuples = {tuple(t) for t in grp["clean_tags"]}
        if len(tag_tuples) > 1:
            tags_conflict_groups += 1
        if grp["hf_split"].nunique() > 1:
            cross_split_groups += 1

    # 5. Se um grupo tiver membro no teste, remover os demais membros de treino/validação
    test_groups = set(df.loc[df["hf_split"] == "test", "alias_group_id"])
    removed_cross_split = 0
    if drop_if_in_test and test_groups:
        leak_mask = (df["hf_split"] != "test") & (df["alias_group_id"].isin(test_groups))
        removed_cross_split = int(leak_mask.sum())
        df = df[~leak_mask].copy()
    _record_step("4_remove_train_aliases_in_test", df)

    # 6. Deduplicar por grupo de aliases mantendo a primeira ocorrência ordenada por id
    df = df.sort_values(["hf_split", "id"], kind="mergesort")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["hf_split", "alias_group_id"], keep="first").copy()
    removed_intra_dedup = before_dedup - len(df)
    _record_step("5_dedup_aliases_first_by_id", df)

    # 7. Discretização do rating em 5 classes ordinais (0..4)
    df["difficulty_class"] = df["rating_num"].map(lambda r: rating_to_class(r, rating_bounds)).astype(int)

    # 8. Limpeza textual nos campos de entrada
    fields_to_clean = list(dict.fromkeys(["title", "description", "input_format", *tok_cfg.get("fields", [])]))
    df = clean_dataframe_fields(df, fields_to_clean, cleaning_cfg)

    # Garante que nenhuma description ficou vazia após limpeza HTML/LaTeX
    df = df[df["description"].str.len() > 0].copy()
    _record_step("6_nonempty_after_cleaning", df)

    # 9. Particionamento cronológico (test = hf_split == "test") + treino/val 90%/10% por grupo
    test_df = df[df["hf_split"] == "test"].reset_index(drop=True)
    train_pool = df[df["hf_split"] == "train"].reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=split_seed)
    train_idx, val_idx = next(splitter.split(train_pool, groups=train_pool["alias_group_id"]))
    train_df = train_pool.iloc[train_idx].reset_index(drop=True)
    val_df = train_pool.iloc[val_idx].reset_index(drop=True)

    # 10. Verificação rigorosa de ausência de vazamento entre partições
    verify_no_leakage(train_df, val_df, test_df)

    # 11. Seleção do vocabulário de tags usando EXCLUSIVAMENTE o treino
    tag_vocab, train_tag_freq_df = select_tag_vocabulary(
        train_df["clean_tags"],
        min_tag_freq=min_tag_freq,
        exclude_prefix=exclude_prefix,
    )

    # Resumo cronológico (contest_start_year)
    year_summary = {}
    for split_name, split_frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        years = pd.to_numeric(split_frame["contest_start_year"], errors="coerce").dropna()
        year_summary[split_name] = {
            "count": int(len(split_frame)),
            "min_year": int(years.min()) if len(years) else None,
            "max_year": int(years.max()) if len(years) else None,
            "year_counts": {str(int(k)): int(v) for k, v in years.value_counts().sort_index().items()},
        }

    alias_report = {
        "total_multi_member_groups": int(len(multi_groups)),
        "groups_with_conflicting_rating": int(rating_conflict_groups),
        "groups_with_conflicting_tags": int(tags_conflict_groups),
        "groups_spanning_train_and_test": int(cross_split_groups),
        "rows_removed_train_aliases_in_test": int(removed_cross_split),
        "rows_removed_intra_split_dedup": int(removed_intra_dedup),
    }

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "tag_vocab": tag_vocab,
        "train_tag_freq": train_tag_freq_df,
        "filter_steps": filter_steps,
        "alias_report": alias_report,
        "year_summary": year_summary,
    }


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Verifica que não há interseção de id, alias_group_id ou aliases brutos entre partições."""
    splits = {"train": train_df, "val": val_df, "test": test_df}
    names = list(splits.keys())

    def _collect_all_ids(frame: pd.DataFrame) -> set[str]:
        all_ids = set(frame["id"].astype(str))
        if "aliases" in frame.columns:
            for raw in frame["aliases"]:
                all_ids.update(_parse_aliases(raw))
        return all_ids

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            df_a, df_b = splits[a], splits[b]

            group_overlap = set(df_a["alias_group_id"]).intersection(set(df_b["alias_group_id"]))
            if group_overlap:
                raise AssertionError(
                    f"Vazamento detectado entre '{a}' e '{b}': {len(group_overlap)} grupos de aliases em comum."
                )

            id_overlap = _collect_all_ids(df_a).intersection(_collect_all_ids(df_b))
            if id_overlap:
                raise AssertionError(
                    f"Vazamento de id/alias detectado entre '{a}' e '{b}': {len(id_overlap)} ids em comum."
                )


_SAVED_SPLIT_COLUMNS = [
    "id",
    "aliases",
    "contest_start_year",
    "title",
    "description",
    "input_format",
    "rating",
    "rating_num",
    "difficulty_class",
    "tags",
    "clean_tags",
    "alias_group_id",
    "hf_split",
]

_JSON_LIST_COLUMNS = ("aliases", "tags", "clean_tags")


def _serialize_split_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona apenas colunas necessárias e serializa listas como JSON para compatibilidade PyArrow."""
    cols = [c for c in _SAVED_SPLIT_COLUMNS if c in df.columns]
    out = df[cols].copy()
    for col in _JSON_LIST_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: json.dumps(list(v) if isinstance(v, (list, tuple, set, np.ndarray)) else ([] if v is None or (isinstance(v, float) and np.isnan(v)) else [v]), ensure_ascii=False)
            )
    return out


def _read_split_parquet(path: Path) -> pd.DataFrame:
    """Lê um Parquet salvo e restaura colunas de lista serializadas em JSON."""
    df = pd.read_parquet(path)
    for col in _JSON_LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: json.loads(v) if isinstance(v, str) else (list(v) if isinstance(v, (list, tuple, np.ndarray)) else [])
            )
    return df


def load_or_prepare_splits(cfg: dict[str, Any], force_rebuild: bool = False) -> dict[str, Any]:
    """Carrega partições salvas em splits_dir ou baixa open-r1/codeforces e gera tudo de forma determinística."""
    splits_dir = Path(cfg["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "val.parquet"
    test_path = splits_dir / "test.parquet"
    meta_path = splits_dir / "splits_metadata.json"
    tag_freq_path = splits_dir / "train_tag_frequencies.csv"

    if not force_rebuild and all(p.exists() for p in (train_path, val_path, test_path, meta_path, tag_freq_path)):
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "train": _read_split_parquet(train_path),
                "val": _read_split_parquet(val_path),
                "test": _read_split_parquet(test_path),
                "tag_vocab": meta["tag_vocab"],
                "train_tag_freq": pd.read_csv(tag_freq_path),
                "filter_steps": meta["filter_steps"],
                "alias_report": meta["alias_report"],
                "year_summary": meta["year_summary"],
            }
        except Exception as exc:
            print(f"[AVISO] Falha ao ler splits em cache ({type(exc).__name__}: {exc}). Reconstruindo...")

    from datasets import load_dataset

    data_cfg = cfg["data"]
    ds = load_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("subset", "default"),
        cache_dir=cfg["paths"].get("cache_dir"),
    )
    raw_train_df = ds["train"].to_pandas()
    raw_test_df = ds["test"].to_pandas()

    bundle = prepare_splits_from_dataframes(raw_train_df, raw_test_df, cfg)

    # Persiste partições fixas para reutilização por todos os modelos e seeds
    _serialize_split_for_parquet(bundle["train"]).to_parquet(train_path, index=False)
    _serialize_split_for_parquet(bundle["val"]).to_parquet(val_path, index=False)
    _serialize_split_for_parquet(bundle["test"]).to_parquet(test_path, index=False)
    bundle["train_tag_freq"].to_csv(tag_freq_path, index=False)

    meta = {
        "tag_vocab": bundle["tag_vocab"],
        "filter_steps": bundle["filter_steps"],
        "alias_report": bundle["alias_report"],
        "year_summary": bundle["year_summary"],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return bundle
