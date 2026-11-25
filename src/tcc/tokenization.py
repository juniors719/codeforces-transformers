"""Tokenização e estratégias configuráveis de truncamento para enunciados do Codeforces.

Estratégias suportadas:
  - "right": concatena os campos com sep_token nativo e trunca pela direita.
  - "head_tail": mantém head_tokens iniciais + cauda final até atingir max_length.
  - "reserve_input_format": reserva até N tokens (padrão 128) para input_format
    (onde ficam as restrições numéricas) e trunca description no espaço restante.
"""

from __future__ import annotations

from typing import Any, Mapping


def _get_special_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Obtém os ids de início (CLS/BOS) e separador (SEP/EOS) do tokenizador."""
    cls_id = tokenizer.cls_token_id
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizador não define cls/bos_token_id ou sep/eos_token_id.")
    return int(cls_id), int(sep_id)


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def tokenize_example(
    example: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
    fields: list[str] | None = None,
    strategy: str = "reserve_input_format",
    reserve_input_format_tokens: int = 128,
    head_tail_head_tokens: int = 128,
) -> dict[str, Any]:
    """Tokeniza um problema aplicando a estratégia de truncamento configurada.

    Retorna:
      - input_ids: lista de ids com tamanho <= max_length
      - attention_mask: lista de 1s do mesmo tamanho
      - was_truncated: booleano indicando se algum token foi descartado
      - original_token_count: comprimento total sem truncamento (incluindo tokens especiais)
    """
    active_fields = fields or ["title", "description", "input_format"]
    cls_id, sep_id = _get_special_token_ids(tokenizer)

    field_tokens: dict[str, list[int]] = {
        field: _encode_no_special(tokenizer, str(example.get(field, "") or ""))
        for field in active_fields
    }

    # Monta sequência completa sem truncamento: [CLS] f1 [SEP] f2 [SEP] ... [SEP]
    full_body: list[int] = []
    for idx, field in enumerate(active_fields):
        if idx > 0:
            full_body.append(sep_id)
        full_body.extend(field_tokens[field])

    original_token_count = len(full_body) + 2  # + CLS + SEP final
    body_budget = max_length - 2
    if body_budget <= 0:
        raise ValueError(f"max_length={max_length} é pequeno demais para tokens especiais.")

    if len(full_body) <= body_budget:
        input_ids = [cls_id, *full_body, sep_id]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "was_truncated": False,
            "original_token_count": original_token_count,
        }

    # A partir daqui, há truncamento
    if strategy == "right":
        truncated_body = full_body[:body_budget]

    elif strategy == "head_tail":
        head_len = min(head_tail_head_tokens, body_budget)
        tail_len = body_budget - head_len
        if tail_len > 0:
            truncated_body = full_body[:head_len] + full_body[-tail_len:]
        else:
            truncated_body = full_body[:head_len]

    elif strategy == "reserve_input_format":
        if "input_format" not in active_fields:
            truncated_body = full_body[:body_budget]
        else:
            n_seps = len(active_fields) - 1
            content_budget = max(body_budget - n_seps, 0)

            # Reserva até reserve_input_format_tokens para input_format (sem desperdiçar sobra)
            in_fmt_raw = field_tokens.get("input_format", [])
            in_fmt_budget = min(len(in_fmt_raw), reserve_input_format_tokens, content_budget)
            in_fmt_kept = in_fmt_raw[:in_fmt_budget]
            remaining = content_budget - len(in_fmt_kept)

            # Preserva campos auxiliares (ex.: title) antes de alocar o restante para description
            kept_by_field: dict[str, list[int]] = {"input_format": in_fmt_kept}
            other_fields = [f for f in active_fields if f not in ("description", "input_format")]
            for field in other_fields:
                toks = field_tokens.get(field, [])
                take = min(len(toks), remaining)
                kept_by_field[field] = toks[:take]
                remaining -= take

            if "description" in active_fields:
                desc_toks = field_tokens.get("description", [])
                kept_by_field["description"] = desc_toks[:remaining]

            truncated_body = []
            for idx, field in enumerate(active_fields):
                if idx > 0:
                    truncated_body.append(sep_id)
                truncated_body.extend(kept_by_field.get(field, []))
            truncated_body = truncated_body[:body_budget]
    else:
        raise ValueError(
            f"Estratégia de truncamento desconhecida: '{strategy}'. "
            "Use 'right', 'head_tail' ou 'reserve_input_format'."
        )

    input_ids = [cls_id, *truncated_body, sep_id]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "was_truncated": True,
        "original_token_count": original_token_count,
    }


def compute_token_length_stats(
    df: Any,
    tokenizer: Any,
    fields: list[str] | None = None,
    threshold: int = 512,
) -> dict[str, Any]:
    """Calcula estatísticas de comprimento em tokens (média, mediana, p90, p95, p99, max e %>512)."""
    import numpy as np

    active_fields = fields or ["title", "description", "input_format"]
    lengths: list[int] = []
    input_fmt_lengths: list[int] = []

    for row in df.to_dict(orient="records"):
        res = tokenize_example(
            row,
            tokenizer=tokenizer,
            max_length=100_000,
            fields=active_fields,
            strategy="right",
        )
        lengths.append(res["original_token_count"])
        if "input_format" in active_fields:
            input_fmt_lengths.append(len(_encode_no_special(tokenizer, str(row.get("input_format", "") or ""))))

    arr = np.asarray(lengths, dtype=np.float64)
    stats: dict[str, Any] = {
        "n_samples": int(len(arr)),
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
        "max": int(np.max(arr)) if len(arr) else 0,
        f"pct_gt_{threshold}": float(100.0 * np.mean(arr > threshold)) if len(arr) else 0.0,
        "pct_gt_2048": float(100.0 * np.mean(arr > 2048)) if len(arr) else 0.0,
    }

    if input_fmt_lengths:
        if_arr = np.asarray(input_fmt_lengths, dtype=np.float64)
        stats["input_format_median"] = float(np.median(if_arr))
        stats["input_format_p90"] = float(np.percentile(if_arr, 90))
        stats["input_format_p95"] = float(np.percentile(if_arr, 95))
        stats["input_format_p99"] = float(np.percentile(if_arr, 99))
        stats["input_format_max"] = int(np.max(if_arr))

    return stats
