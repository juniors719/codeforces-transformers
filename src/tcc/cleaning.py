"""Limpeza textual de enunciados do Codeforces (LaTeX, HTML e espaços).

Regras metodológicas do TCC:
- Remove delimitadores "$$$" usados pelo Codeforces.
- Converte comandos LaTeX comuns para texto ou Unicode legível.
- Remove resíduos de tags/entidades HTML.
- Normaliza espaços em branco.
- NÃO remove stopwords, NÃO aplica stemming/lematização e NÃO faz lowercasing
  (o tratamento de maiúsculas/minúsculas fica a cargo do tokenizador de cada modelo).
"""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd

# Substituições de comandos LaTeX mais frequentes em enunciados de programação competitiva.
# Comandos que são prefixos de outros (ex.: \leq antes de \le) vêm primeiro.
_LATEX_COMMAND_MAP: list[tuple[str, str]] = [
    (r"\\leqslant\b", "≤"),
    (r"\\leq\b", "≤"),
    (r"\\le\b", "≤"),
    (r"\\geqslant\b", "≥"),
    (r"\\geq\b", "≥"),
    (r"\\ge\b", "≥"),
    (r"\\neq\b", "≠"),
    (r"\\ne\b", "≠"),
    (r"\\lt\b", "<"),
    (r"\\gt\b", ">"),
    (r"\\cdot\b", "·"),
    (r"\\times\b", "×"),
    (r"\\ldots\b", "..."),
    (r"\\cdots\b", "..."),
    (r"\\vdots\b", "..."),
    (r"\\dots\b", "..."),
    (r"\\infty\b", "∞"),
    (r"\\pm\b", "±"),
    (r"\\mp\b", "∓"),
    (r"\\approx\b", "≈"),
    (r"\\equiv\b", "≡"),
    (r"\\sim\b", "~"),
    (r"\\bmod\b", " mod "),
    (r"\\pmod\b", " mod "),
    (r"\\mod\b", " mod "),
    (r"\\notin\b", "∉"),
    (r"\\in\b", "∈"),
    (r"\\subseteq\b", "⊆"),
    (r"\\subset\b", "⊂"),
    (r"\\supseteq\b", "⊇"),
    (r"\\supset\b", "⊃"),
    (r"\\cup\b", "∪"),
    (r"\\cap\b", "∩"),
    (r"\\emptyset\b", "∅"),
    (r"\\leftrightarrow\b", "↔"),
    (r"\\Leftrightarrow\b", "↔"),
    (r"\\rightarrow\b", "→"),
    (r"\\Rightarrow\b", "→"),
    (r"\\to\b", "→"),
    (r"\\leftarrow\b", "←"),
    (r"\\Leftarrow\b", "←"),
    (r"\\forall\b", "∀"),
    (r"\\exists\b", "∃"),
    (r"\\sum\b", "sum"),
    (r"\\prod\b", "prod"),
    (r"\\sqrt\b", "sqrt"),
    (r"\\min\b", "min"),
    (r"\\max\b", "max"),
    (r"\\log\b", "log"),
    (r"\\gcd\b", "gcd"),
    (r"\\lcm\b", "lcm"),
    (r"\\oplus\b", "⊕"),
    (r"\\land\b", " and "),
    (r"\\lor\b", " or "),
    (r"\\lfloor\b", "⌊"),
    (r"\\rfloor\b", "⌋"),
    (r"\\lceil\b", "⌈"),
    (r"\\rceil\b", "⌉"),
    (r"\\left\b", ""),
    (r"\\right\b", ""),
    (r"\\qquad\b", " "),
    (r"\\quad\b", " "),
    (r"\\[,;:!]", " "),
]

_COMPILED_LATEX: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern), repl) for pattern, repl in _LATEX_COMMAND_MAP
]

# Wrappers de formatação LaTeX como \text{abc}, \mathrm{n}, \mathbf{A}, etc.
_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:text|textrm|textbf|textit|texttt|mathrm|mathbf|mathit|mathtt|mathcal|mathbb|operatorname)\{([^{}]*)\}"
)

# Frações simples \frac{a}{b} -> (a)/(b)
_LATEX_FRAC_RE = re.compile(r"\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}")

# Delimitadores $$$ (ou sequências maiores de $) do Codeforces
_TRIPLE_DOLLAR_RE = re.compile(r"\${3,}")

# Tags HTML residuais
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Espaços múltiplos e quebras de linha
_WHITESPACE_RE = re.compile(r"\s+")


def convert_latex_to_text(text: str) -> str:
    """Converte comandos LaTeX comuns em texto/Unicode preservando o conteúdo matemático."""
    # Desembrulha frações e wrappers de texto (até 3 passadas para aninhamento raso)
    for _ in range(3):
        prev = text
        text = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", text)
        text = _LATEX_WRAPPER_RE.sub(r"\1", text)
        if text == prev:
            break

    for pattern, repl in _COMPILED_LATEX:
        text = pattern.sub(repl, text)

    # Remove barras invertidas isoladas antes de chaves literais (\{ e \})
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    return text


def clean_text(text: str | None, cleaning_cfg: dict[str, Any] | None = None) -> str:
    """Limpa um campo textual de enunciado segundo a configuração informada."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    cfg = cleaning_cfg or {}
    out = str(text)

    if cfg.get("strip_html", True):
        out = html.unescape(out)
        out = _HTML_TAG_RE.sub(" ", out)

    if cfg.get("remove_triple_dollar", True):
        out = _TRIPLE_DOLLAR_RE.sub("", out)

    if cfg.get("convert_latex", True):
        out = convert_latex_to_text(out)

    if cfg.get("normalize_whitespace", True):
        out = _WHITESPACE_RE.sub(" ", out).strip()

    return out


def clean_dataframe_fields(
    df: pd.DataFrame,
    fields: list[str],
    cleaning_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Aplica clean_text às colunas especificadas, retornando uma cópia do DataFrame."""
    cleaned = df.copy()
    for col in fields:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].map(lambda x: clean_text(x, cleaning_cfg))
    return cleaned
