"""Testes unitários para discretização de rating e filtragem de tags."""

import numpy as np
import pytest

from tcc.labels import (
    binarize_tags,
    compute_pos_weight,
    compute_tag_frequencies,
    normalize_tag_list,
    rating_to_class,
    select_tag_vocabulary,
)


@pytest.mark.parametrize(
    "rating,expected",
    [
        (800, 0),
        (1100, 0),
        (1200, 1),
        (1500, 1),
        (1600, 2),
        (1900, 2),
        (2000, 3),
        (2300, 3),
        (2400, 4),
        (3500, 4),
    ],
)
def test_rating_to_class_boundaries(rating, expected):
    assert rating_to_class(rating) == expected


def test_rating_to_class_invalid_raises():
    with pytest.raises(ValueError):
        rating_to_class(float("nan"))


def test_normalize_tag_list_excludes_star_tags():
    raw = ["dp", "*special", "greedy", "dp", " *interactive "]
    assert normalize_tag_list(raw) == ["dp", "greedy"]


def test_select_tag_vocabulary_and_binarize():
    train_tags = [
        ["dp", "greedy", "*special"],
        ["dp", "math"],
        ["dp", "greedy"],
    ]
    freq_df = compute_tag_frequencies(train_tags)
    assert list(freq_df["tag"]) == ["dp", "greedy", "math"]

    vocab, full_df = select_tag_vocabulary(train_tags, min_tag_freq=2)
    assert vocab == ["dp", "greedy"]
    assert len(full_df) == 3

    mat = binarize_tags(train_tags, vocab)
    assert mat.shape == (3, 2)
    assert np.array_equal(mat, np.array([[1, 1], [1, 0], [1, 1]], dtype=np.float32))

    pw = compute_pos_weight(mat)
    assert np.allclose(pw, np.array([0.0, 0.5], dtype=np.float32))
