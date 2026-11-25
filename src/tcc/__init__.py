"""Pacote principal do TCC: classificação multitarefa de problemas de programação competitiva."""

import os

# Evita que transformers tente importar TensorFlow pré-instalado no Colab (conflito de protobuf)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

__version__ = "0.1.0"
