"""Arquitetura multitarefa (MTL) com compartilhamento rígido de parâmetros sobre h[CLS].

Suporta:
  - Modo MTL (tags + dificuldade), single-task tags_only e single-task difficulty_only.
  - Cabeça de dificuldade padrão (CrossEntropy, 5 classes) ou ordinal CORAL (opcional).
  - Configurações específicas de ModernBERT (reference_compile=False) e BigBird (attention_type).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


def coral_targets(y_dif: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Converte rótulos inteiros (0..K-1) em alvos ordinais acumulados I(y > k) para k=0..K-2."""
    thresholds = torch.arange(num_classes - 1, device=y_dif.device).unsqueeze(0)
    return (y_dif.unsqueeze(1) > thresholds).float()


def coral_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Converte logits acumulados CORAL (B, K-1) em distribuição de probabilidades (B, K)."""
    cum_probs = torch.sigmoid(logits)
    # P(y = 0) = 1 - P(y > 0)
    # P(y = k) = P(y > k-1) - P(y > k)
    # P(y = K-1) = P(y > K-2)
    ones = torch.ones((cum_probs.size(0), 1), device=cum_probs.device, dtype=cum_probs.dtype)
    zeros = torch.zeros((cum_probs.size(0), 1), device=cum_probs.device, dtype=cum_probs.dtype)
    extended = torch.cat([ones, cum_probs, zeros], dim=1)
    probs = torch.clamp(extended[:, :-1] - extended[:, 1:], min=0.0)
    denom = probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return probs / denom


def coral_logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
    """Prediz a classe ordinal somando quantos limiares acumulados têm P(y > k) > 0.5."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


class CoralHead(nn.Module):
    """Cabeça ordinal CORAL: peso linear compartilhado + K-1 vieses independentes."""

    def __init__(self, hidden_size: int, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h) + self.biases


class MultiTaskTransformer(nn.Module):
    """Encoder Transformer compartilhado + h[CLS] + dropout + duas cabeças lineares."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_tags: int,
        num_difficulty_classes: int = 5,
        dropout: float = 0.1,
        task_mode: str = "mtl",
        difficulty_head_type: str = "classification",
        alpha_tags: float = 1.0,
        beta_dif: float = 1.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if task_mode not in ("mtl", "tags_only", "difficulty_only"):
            raise ValueError(f"task_mode inválido: {task_mode}")
        if difficulty_head_type not in ("classification", "coral"):
            raise ValueError(f"difficulty_head_type inválido: {difficulty_head_type}")

        self.encoder = encoder
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.num_difficulty_classes = num_difficulty_classes
        self.task_mode = task_mode
        self.difficulty_head_type = difficulty_head_type
        self.alpha_tags = float(alpha_tags)
        self.beta_dif = float(beta_dif)

        self.dropout = nn.Dropout(dropout)
        self.head_tags = nn.Linear(hidden_size, num_tags)
        if difficulty_head_type == "coral":
            self.head_dif: nn.Module = CoralHead(hidden_size, num_difficulty_classes)
        else:
            self.head_dif = nn.Linear(hidden_size, num_difficulty_classes)

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    @classmethod
    def from_pretrained_spec(
        cls,
        model_spec: dict[str, Any],
        max_length: int,
        num_tags: int,
        num_difficulty_classes: int = 5,
        dropout: float = 0.1,
        task_mode: str = "mtl",
        difficulty_head_type: str = "classification",
        alpha_tags: float = 1.0,
        beta_dif: float = 1.0,
        pos_weight: torch.Tensor | None = None,
        cache_dir: str | None = None,
    ) -> tuple["MultiTaskTransformer", dict[str, Any]]:
        """Instancia o encoder a partir do catálogo models.yaml e retorna (modelo, metadados)."""
        hf_name = model_spec["hf_name"]
        extra_kwargs: dict[str, Any] = dict(model_spec.get("extra_model_kwargs", {}) or {})
        meta: dict[str, Any] = {"hf_name": hf_name, "attention_type": None}

        # Trata atenção do BigBird conforme max_length (512 -> original_full, 2048 -> block_sparse)
        attn_map = model_spec.get("attention_type_by_length", {}) or {}
        attn_type = attn_map.get(max_length) or attn_map.get(str(max_length))
        if attn_type:
            extra_kwargs["attention_type"] = attn_type
            meta["attention_type"] = attn_type

        hf_config = AutoConfig.from_pretrained(hf_name, cache_dir=cache_dir)
        if attn_type and hasattr(hf_config, "attention_type"):
            hf_config.attention_type = attn_type
        if "reference_compile" in extra_kwargs and hasattr(hf_config, "reference_compile"):
            hf_config.reference_compile = extra_kwargs.pop("reference_compile")

        encoder = AutoModel.from_pretrained(
            hf_name,
            config=hf_config,
            cache_dir=cache_dir,
            **extra_kwargs,
        )

        # Gradient checkpointing configurável por max_length
        gc_map = model_spec.get("gradient_checkpointing", {}) or {}
        use_gc = bool(gc_map.get(max_length, gc_map.get(str(max_length), False)))
        if use_gc and hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
        meta["gradient_checkpointing"] = use_gc

        hidden_size = int(getattr(hf_config, "hidden_size"))
        model = cls(
            encoder=encoder,
            hidden_size=hidden_size,
            num_tags=num_tags,
            num_difficulty_classes=num_difficulty_classes,
            dropout=dropout,
            task_mode=task_mode,
            difficulty_head_type=difficulty_head_type,
            alpha_tags=alpha_tags,
            beta_dif=beta_dif,
            pos_weight=pos_weight,
        )
        return model, meta

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_tags: torch.Tensor | None = None,
        labels_dif: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Vetor h[CLS]: índice 0 do último hidden state
        h_cls = self.dropout(outputs.last_hidden_state[:, 0, :])

        logits_tags = self.head_tags(h_cls)
        logits_dif = self.head_dif(h_cls)

        loss_tags: torch.Tensor | None = None
        loss_dif: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None

        if labels_tags is not None and self.task_mode in ("mtl", "tags_only"):
            loss_tags = F.binary_cross_entropy_with_logits(
                logits_tags,
                labels_tags.float(),
                pos_weight=self.pos_weight,
            )

        if labels_dif is not None and self.task_mode in ("mtl", "difficulty_only"):
            if self.difficulty_head_type == "coral":
                targets = coral_targets(labels_dif.long(), self.num_difficulty_classes)
                loss_dif = F.binary_cross_entropy_with_logits(logits_dif, targets)
            else:
                loss_dif = F.cross_entropy(logits_dif, labels_dif.long())

        if self.task_mode == "mtl" and loss_tags is not None and loss_dif is not None:
            total_loss = self.alpha_tags * loss_tags + self.beta_dif * loss_dif
        elif self.task_mode == "tags_only" and loss_tags is not None:
            total_loss = loss_tags
        elif self.task_mode == "difficulty_only" and loss_dif is not None:
            total_loss = loss_dif

        return {
            "loss": total_loss,
            "loss_tags": loss_tags,
            "loss_dif": loss_dif,
            "logits_tags": logits_tags,
            "logits_dif": logits_dif,
        }
