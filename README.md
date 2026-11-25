# Análise Comparativa de Arquiteturas baseadas em Transformers para Classificação de Dificuldade em Problemas de Programação Competitiva

Repositório experimental do Trabalho de Conclusão de Curso (Ciência da Computação — UFC Quixadá).

## 1. Visão geral

O objetivo é predizer, a partir **apenas do texto do enunciado** (cenário *cold start*, sem código-fonte), simultaneamente:
- **Tags algorítmicas** (classificação multirrótulo);
- **Faixa de dificuldade** (5 classes ordinais derivadas do *rating* do Codeforces: `<=1100`, `1200-1500`, `1600-1900`, `2000-2300`, `>=2400`).

A arquitetura utiliza **aprendizado multitarefa (MTL)** com compartilhamento rígido de parâmetros (*hard parameter sharing*): encoder Transformer compartilhado, extração do vetor `h[CLS]` (índice 0 do último *hidden state*), dropout de `0.1` e duas cabeças lineares independentes.

---

## 2. Estrutura do projeto

```text
tcc2-transformers/
├── configs/                 # Todas as configurações em YAML (base, catálogo de modelos, experimentos)
├── notebooks/               # Notebooks de setup Colab, EDA, execução da grade e análise estatística
├── results/                 # Saídas reprodutíveis: splits/, runs/, tables/, figures/, summary.md
├── src/tcc/                 # Pacote Python com toda a lógica do experimento
├── tests/                   # Testes unitários (pytest)
├── pyproject.toml
└── requirements.txt
```

---

## 3. Como executar no Google Colab

### Caminho A — Notebook local conectado a um runtime do Colab (extensão da IDE)
1. Abra a pasta `tcc2-transformers` como workspace ativo na sua IDE.
2. Conecte o kernel do notebook a um runtime GPU do Colab (T4 ou L4) pela extensão do Colab da IDE.
3. Execute `notebooks/00_setup_colab.ipynb` com `USE_DRIVE = False` (ou `True` se quiser espelhar no Drive).
4. Execute `notebooks/01_eda.ipynb` para gerar a análise exploratória e fixar as partições em `results/splits/`.

### Caminho B (Fallback) — Google Colab no navegador + Google Drive sincronizado
1. Envie o repositório para o Git (recomendado) ou compacte a pasta `tcc2-transformers` em `tcc2-transformers.zip` e coloque-a em `Meu Drive/tcc2/`.
2. No Google Colab, escolha **Ambiente de execução → Alterar tipo de ambiente de execução → GPU (T4 ou L4)**.
3. Abra `notebooks/00_setup_colab.ipynb`, mantenha `USE_DRIVE = True` e execute todas as células:
   - O Drive será montado em `/content/drive/MyDrive/tcc2/`;
   - A variável `TCC_RESULTS_DIR` apontará para `/content/drive/MyDrive/tcc2/results`, sincronizando automaticamente via Google Drive para computador;
   - As dependências fixadas e o pacote `tcc` serão instalados.
4. Execute `notebooks/01_eda.ipynb` e, nas fases seguintes, `notebooks/02_run_experiments.ipynb` e `notebooks/03_analysis.ipynb`.
5. Para baixar todos os resultados como `.zip` a qualquer momento:
   ```python
   from tcc.utils import zip_results_dir
   zip_results_dir("/content/drive/MyDrive/tcc2/results", "/content/results_tcc.zip")
   ```

---

## 4. Decisões metodológicas e divergências em relação ao texto inicial do TCC

| Aspecto | Decisão implementada | Justificativa |
|---|---|---|
| **Warmup da taxa de aprendizado** | `warmup_ratio = 0.1` com decaimento linear | O texto inicial do TCC não especifica *warmup*; 10% é prática consolidada para estabilizar o *fine-tuning* de Transformers (a registrar na Tabela 3). |
| **Campo `input_format` na entrada** | Entrada padrão: `title [SEP] description [SEP] input_format` | As restrições numéricas em `input_format` são sinal forte de complexidade algorítmica. A configuração `configs/experiments/ablation_input.yaml` mantém apenas `title + description` para ablação. |
| **Separador entre campos** | Usa `tokenizer.sep_token` nativo de cada modelo (`[SEP]` ou `</s>`) | Evita fragmentação artificial do separador literal `[SEP]` em modelos da família RoBERTa/CodeBERT/BigBird. |
| **Truncamento padrão** | `reserve_input_format` (reserva até 128 tokens para `input_format`) | Preserva as restrições numéricas no fim do enunciado mesmo quando `description` excede `max_len`. O valor de 128 é verificado pelo p95 na EDA. |
| **Deduplicação de `aliases`** | Union-Find sobre `id`/`aliases` após exigir `rating`, `tags` e `description` válidos; remove de treino/val grupos presentes no teste; mantém 1ª ocorrência por `id` | Elimina vazamento entre cópias Div1/Div2 e evita inflação artificial de desempenho no teste. Conflitos de rating/tags nos grupos são reportados na EDA. |
| **BigBird em `max_len = 512`** | Usa `attention_type="original_full"` (padrão do Hugging Face para sequências curtas) | Com `block_size=64`, a atenção esparsa por blocos do BigBird exige sequências maiores (~704+ tokens); em 512 tokens o HF recorre à atenção completa. Em `max_len = 2048` usa `block_sparse`. |
| **ModernBERT no Colab** | `reference_compile=False` e `transformers==4.49.0` | Evita falhas/atrasos de compilação na GPU T4 mantendo compatibilidade total com `answerdotai/ModernBERT-base`. |

---

## 5. Testes unitários

Para rodar a suíte de testes localmente:
```bash
pytest -q
```
