"""Pacote principal do TCC: classificação multitarefa de problemas de programação competitiva."""

import os

# Evita que transformers tente importar TensorFlow pré-instalado no Colab (conflito de protobuf)
# e evita warnings de fork do tokenizers no DataLoader.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
