"""Testes unitários para o modelo MTL, cabeça CORAL, modos STL e grade do runner."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tcc.config import load_config
from tcc.model import MultiTaskTransformer, coral_logits_to_pred, coral_logits_to_probs
from tcc.runner import build_run_plan


class _DummyEncoder(nn.Module):
    """Encoder leve determinístico para testar as cabeças sem baixar pesos da rede."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def test_mtl_model_shapes_and_losses():
    encoder = _DummyEncoder(hidden_size=16)
    model = MultiTaskTransformer(
        encoder=encoder,
        hidden_size=16,
        num_tags=12,
        num_difficulty_classes=5,
        task_mode="mtl",
        difficulty_head_type="classification",
    )

    input_ids = torch.randint(0, 100, (4, 10))
    attention_mask = torch.ones((4, 10), dtype=torch.long)
    labels_tags = torch.randint(0, 2, (4, 12), dtype=torch.float32)
    labels_dif = torch.randint(0, 5, (4,), dtype=torch.long)

    out = model(input_ids, attention_mask, labels_tags=labels_tags, labels_dif=labels_dif)
    assert out["logits_tags"].shape == (4, 12)
    assert out["logits_dif"].shape == (4, 5)
    assert out["loss"] is not None
    assert out["loss_tags"] is not None
    assert out["loss_dif"] is not None
    assert torch.isclose(out["loss"], out["loss_tags"] + out["loss_dif"])


def test_coral_head_shapes_and_probabilities():
    encoder = _DummyEncoder(hidden_size=16)
    model = MultiTaskTransformer(
        encoder=encoder,
        hidden_size=16,
        num_tags=6,
        num_difficulty_classes=5,
        task_mode="mtl",
        difficulty_head_type="coral",
    )

    input_ids = torch.randint(0, 100, (3, 8))
    attention_mask = torch.ones((3, 8), dtype=torch.long)
    labels_tags = torch.zeros((3, 6), dtype=torch.float32)
    labels_dif = torch.tensor([0, 2, 4], dtype=torch.long)

    out = model(input_ids, attention_mask, labels_tags=labels_tags, labels_dif=labels_dif)
    assert out["logits_dif"].shape == (3, 4)  # K - 1 limiares acumulados
    probs = coral_logits_to_probs(out["logits_dif"])
    assert probs.shape == (3, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)
    preds = coral_logits_to_pred(out["logits_dif"])
    assert preds.shape == (3,)
    assert ((preds >= 0) & (preds <= 4)).all()


@pytest.mark.parametrize("mode", ["tags_only", "difficulty_only"])
def test_single_task_modes(mode):
    encoder = _DummyEncoder(hidden_size=16)
    model = MultiTaskTransformer(
        encoder=encoder,
        hidden_size=16,
        num_tags=5,
        num_difficulty_classes=5,
        task_mode=mode,
    )
    input_ids = torch.randint(0, 100, (2, 6))
    attention_mask = torch.ones((2, 6), dtype=torch.long)
    labels_tags = torch.zeros((2, 5), dtype=torch.float32)
    labels_dif = torch.tensor([1, 3], dtype=torch.long)

    out = model(input_ids, attention_mask, labels_tags=labels_tags, labels_dif=labels_dif)
    assert out["loss"] is not None
    if mode == "tags_only":
        assert out["loss_tags"] is not None
        assert out["loss_dif"] is None
    else:
        assert out["loss_dif"] is not None
        assert out["loss_tags"] is None


def test_build_run_plan_main_grid_has_70_runs():
    cfg = load_config(experiment_path="configs/experiments/main.yaml")
    plan = build_run_plan(cfg)
    assert len(plan) == 70
    unique_ids = {item["run_id"] for item in plan}
    assert len(unique_ids) == 70
