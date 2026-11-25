"""Testes unitários para estratégias de truncamento na tokenização."""

from tcc.tokenization import tokenize_example


class _DummyTokenizer:
    """Tokenizador simples por espaço para testes determinísticos sem rede."""

    cls_token_id = 101
    sep_token_id = 102

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return []
        return [1000 + (abs(hash(tok)) % 50000) for tok in text.split()]


def test_tokenize_no_truncation():
    tok = _DummyTokenizer()
    ex = {"title": "Sum", "description": "Find the sum", "input_format": "1 <= n <= 10"}
    res = tokenize_example(ex, tok, max_length=32, strategy="reserve_input_format")
    assert res["was_truncated"] is False
    assert res["input_ids"][0] == tok.cls_token_id
    assert res["input_ids"][-1] == tok.sep_token_id
    assert len(res["input_ids"]) == len(res["attention_mask"]) == res["original_token_count"]


def test_tokenize_reserve_input_format_preserves_constraints():
    tok = _DummyTokenizer()
    ex = {
        "title": "T1 T2",
        "description": " ".join([f"d{i}" for i in range(50)]),
        "input_format": " ".join([f"c{i}" for i in range(10)]),
    }
    res = tokenize_example(
        ex,
        tok,
        max_length=25,
        strategy="reserve_input_format",
        reserve_input_format_tokens=10,
    )
    assert res["was_truncated"] is True
    assert len(res["input_ids"]) == 25

    expected_input_fmt_ids = tok.encode(ex["input_format"], add_special_tokens=False)
    # Todos os 10 tokens de input_format devem estar preservados antes do SEP final
    assert res["input_ids"][-11:-1] == expected_input_fmt_ids


def test_tokenize_right_and_head_tail():
    tok = _DummyTokenizer()
    ex = {
        "title": "Title",
        "description": " ".join([f"w{i}" for i in range(40)]),
        "input_format": "n <= 100",
    }
    res_right = tokenize_example(ex, tok, max_length=20, strategy="right")
    res_ht = tokenize_example(ex, tok, max_length=20, strategy="head_tail", head_tail_head_tokens=8)

    assert res_right["was_truncated"] is True
    assert res_ht["was_truncated"] is True
    assert len(res_right["input_ids"]) == 20
    assert len(res_ht["input_ids"]) == 20
    # head_tail preserva o final (input_format) enquanto right descarta o final
    assert res_ht["input_ids"][-2] == tok.encode("100")[0]
    assert res_right["input_ids"][-2] != tok.encode("100")[0]
