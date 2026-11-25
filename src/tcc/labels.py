"""Discretização de dificuldade (rating) e processamento das tags algorítmicas."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np
import pandas as pd

DEFAULT_RATING_UPPER_BOUNDS: list[int] = [1100, 1500, 1900, 2300]
DEFAULT_DIFFICULTY_NAMES: list[str] = [
    "<=1100",
    "1200-1500",
    "1600-1900",
    "2000-2300",
    ">=2400",
]


def rating_to_class(
    rating: int | float,
    upper_bounds: list[int] | None = None,
) -> int:
    """Mapeia o rating numérico do Codeforces para uma classe ordinal de 0 a 4.

    Faixas padrão:
      0: <= 1100
      1: 1200-1500 (1101..1500)
      2: 1600-1900 (1501..1900)
      3: 2000-2300 (1901..2300)
      4: >= 2400   (> 2300)
    """
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        raise ValueError("Rating inválido (None ou NaN).")

    bounds = upper_bounds if upper_bounds is not None else DEFAULT_RATING_UPPER_BOUNDS
    value = float(rating)
    for class_idx, upper in enumerate(bounds):
        if value <= upper:
            return class_idx
    return len(bounds)


def normalize_tag_list(raw_tags: Any, exclude_prefix: str = "*") -> list[str]:
    """Normaliza uma lista de tags, removendo vazias e tags iniciadas pelo prefixo excluído."""
    if raw_tags is None:
        return []
    if isinstance(raw_tags, float) and np.isnan(raw_tags):
        return []
    if isinstance(raw_tags, str):
        items = [raw_tags]
    elif isinstance(raw_tags, (list, tuple, set, np.ndarray)):
        items = list(raw_tags)
    else:
        return []

    cleaned: set[str] = set()
    for tag in items:
        if tag is None:
            continue
        t = str(tag).strip()
        if not t:
            continue
        if exclude_prefix and t.startswith(exclude_prefix):
            continue
        cleaned.add(t)
    return sorted(cleaned)


def compute_tag_frequencies(
    tags_series: Iterable[Iterable[str]],
    exclude_prefix: str = "*",
) -> pd.DataFrame:
    """Calcula tabela completa de frequência de tags (ordenada por contagem decrescente)."""
    counter: Counter[str] = Counter()
    total_problems = 0
    for raw_tags in tags_series:
        total_problems += 1
        for tag in normalize_tag_list(raw_tags, exclude_prefix=exclude_prefix):
            counter[tag] += 1

    rows = [
        {
            "tag": tag,
            "count": count,
            "pct": (100.0 * count / total_problems) if total_problems > 0 else 0.0,
        }
        for tag, count in counter.most_common()
    ]
    return pd.DataFrame(rows, columns=["tag", "count", "pct"])


def select_tag_vocabulary(
    train_tags: Iterable[Iterable[str]],
    min_tag_freq: int = 50,
    exclude_prefix: str = "*",
) -> tuple[list[str], pd.DataFrame]:
    """Seleciona o vocabulário final de tags usando exclusivamente o conjunto de treino."""
    freq_df = compute_tag_frequencies(train_tags, exclude_prefix=exclude_prefix)
    freq_df["selected"] = freq_df["count"] >= min_tag_freq
    selected_tags = sorted(freq_df.loc[freq_df["selected"], "tag"].tolist())
    return selected_tags, freq_df


def binarize_tags(
    tags_series: Iterable[Iterable[str]],
    tag_vocab: list[str],
    exclude_prefix: str = "*",
) -> np.ndarray:
    """Converte sequências de tags em matriz binária float32 de formato (N, len(tag_vocab))."""
    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_vocab)}
    rows = list(tags_series)
    matrix = np.zeros((len(rows), len(tag_vocab)), dtype=np.float32)

    for row_idx, raw_tags in enumerate(rows):
        for tag in normalize_tag_list(raw_tags, exclude_prefix=exclude_prefix):
            col_idx = tag_to_idx.get(tag)
            if col_idx is not None:
                matrix[row_idx, col_idx] = 1.0
    return matrix


def compute_pos_weight(y_tags: np.ndarray) -> np.ndarray:
    """Calcula pos_weight = (N - pos) / max(pos, 1) por tag para BCEWithLogitsLoss opcional."""
    n_samples = y_tags.shape[0]
    positives = y_tags.sum(axis=0)
    negatives = n_samples - positives
    return (negatives / np.maximum(positives, 1.0)).astype(np.float32)
