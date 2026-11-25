"""Testes unitários para limpeza textual de enunciados do Codeforces."""

from tcc.cleaning import clean_text


def test_clean_text_removes_triple_dollars_and_converts_latex():
    raw = (
        "Given $$$n$$$ integers ($$$1 \\le n \\le 2 \\cdot 10^5$$$), "
        "find $$$a_1, a_2, \\ldots, a_n$$$ such that $$$a_i \\ge 0$$$ and $$$a_i \\neq a_j$$$."
    )
    cleaned = clean_text(raw)
    assert "$$$" not in cleaned
    assert "\\le" not in cleaned
    assert "\\cdot" not in cleaned
    assert "\\ldots" not in cleaned
    assert "\\ge" not in cleaned
    assert "\\neq" not in cleaned
    assert "≤" in cleaned
    assert "·" in cleaned
    assert "..." in cleaned
    assert "≥" in cleaned
    assert "≠" in cleaned


def test_clean_text_strips_html_and_preserves_case_and_stopwords():
    raw = "<p>The <b>Quick</b> Fox &amp; Dog</p>\n\n  <div>with $$$\\frac{a}{b}$$$ and \\text{YES}</div>"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "&amp;" not in cleaned
    # Preserva capitalização (lowercasing fica a cargo do tokenizador) e stopwords
    assert "The Quick Fox & Dog" in cleaned
    assert "(a)/(b)" in cleaned
    assert "YES" in cleaned
    assert "  " not in cleaned


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("   \n\t  ") == ""
