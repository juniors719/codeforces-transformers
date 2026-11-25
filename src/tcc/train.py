"""Loop de treino e avaliação para um run individual (modelo × max_len × seed).

Recursos implementados:
  - Mixed precision automática (bf16 se suportado pela GPU, senão fp16, ou fp32 em CPU).
  - Gradient accumulation para manter batch efetivo configurável (padrão 32).
  - AdamW + decaimento linear com warmup_ratio configurável (padrão 0.1).
  - Early stopping monitorando exclusivamente a validation loss total.
  - Checkpoint da melhor época durante o run (removido ao final se keep_checkpoint=False).
  - Contagem exata de exemplos truncados por partição registrada nos metadados do run.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from tcc.evaluate import evaluate_predictions, tune_per_class_thresholds
from tcc.labels import binarize_tags, compute_pos_weight
from tcc.model import MultiTaskTransformer, coral_logits_to_pred, coral_logits_to_probs
from tcc.tokenization import tokenize_example
from tcc.utils import get_environment_metadata, set_seed


class ProblemDataset(Dataset):
    """Dataset pré-tokenizado que contabiliza quantos exemplos sofreram truncamento."""

    def __init__(
        self,
        df: pd.DataFrame,
        y_tags: np.ndarray,
        tokenizer: Any,
        max_length: int,
        fields: list[str],
        strategy: str,
        reserve_input_format_tokens: int,
        head_tail_head_tokens: int,
    ) -> None:
        self.samples: list[dict[str, Any]] = []
        self.n_truncated = 0

        records = df.to_dict(orient="records")
        y_dif = df["difficulty_class"].to_numpy(dtype=np.int64)

        for idx, row in enumerate(records):
            tok = tokenize_example(
                row,
                tokenizer=tokenizer,
                max_length=max_length,
                fields=fields,
                strategy=strategy,
                reserve_input_format_tokens=reserve_input_format_tokens,
                head_tail_head_tokens=head_tail_head_tokens,
            )
            if tok["was_truncated"]:
                self.n_truncated += 1
            self.samples.append(
                {
                    "input_ids": tok["input_ids"],
                    "attention_mask": tok["attention_mask"],
                    "labels_tags": y_tags[idx],
                    "labels_dif": int(y_dif[idx]),
                }
            )

        self.pct_truncated = 100.0 * self.n_truncated / max(len(self.samples), 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def make_collate_fn(pad_token_id: int):
    """Padding dinâmico até o maior exemplo do batch."""

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(item["attention_mask"], dtype=torch.long)

        labels_tags = torch.tensor(np.stack([item["labels_tags"] for item in batch]), dtype=torch.float32)
        labels_dif = torch.tensor([item["labels_dif"] for item in batch], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_tags": labels_tags,
            "labels_dif": labels_dif,
        }

    return _collate


def resolve_amp_config(mixed_precision: str, device: torch.device) -> tuple[str, torch.dtype | None, bool]:
    """Define precisão mista (bf16/fp16/fp32) e se GradScaler deve ser ativado."""
    if device.type != "cuda":
        return "fp32", None, False

    mode = (mixed_precision or "auto").lower()
    if mode == "auto":
        if torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16, False
        return "fp16", torch.float16, True
    if mode == "bf16":
        return "bf16", torch.bfloat16, False
    if mode == "fp16":
        return "fp16", torch.float16, True
    return "fp32", None, False


@torch.no_grad()
def predict_loader(
    model: MultiTaskTransformer,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    """Executa inferência e calcula perda média e probabilidades em um DataLoader."""
    model.eval()
    total_loss = 0.0
    total_loss_tags = 0.0
    total_loss_dif = 0.0
    n_examples = 0

    all_tags_true: list[np.ndarray] = []
    all_tags_probs: list[np.ndarray] = []
    all_dif_true: list[np.ndarray] = []
    all_dif_probs: list[np.ndarray] = []
    all_dif_pred: list[np.ndarray] = []

    use_amp = amp_dtype is not None and device.type == "cuda"

    for batch in loader:
        bsz = batch["input_ids"].size(0)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels_tags = batch["labels_tags"].to(device, non_blocking=True)
        labels_dif = batch["labels_dif"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_tags=labels_tags,
                labels_dif=labels_dif,
            )

        if out["loss"] is not None:
            total_loss += float(out["loss"].item()) * bsz
        if out["loss_tags"] is not None:
            total_loss_tags += float(out["loss_tags"].item()) * bsz
        if out["loss_dif"] is not None:
            total_loss_dif += float(out["loss_dif"].item()) * bsz
        n_examples += bsz

        tags_probs = torch.sigmoid(out["logits_tags"].float()).cpu().numpy()
        if model.difficulty_head_type == "coral":
            dif_probs = coral_logits_to_probs(out["logits_dif"].float()).cpu().numpy()
            dif_pred = coral_logits_to_pred(out["logits_dif"].float()).cpu().numpy()
        else:
            dif_probs = torch.softmax(out["logits_dif"].float(), dim=-1).cpu().numpy()
            dif_pred = np.argmax(dif_probs, axis=-1)

        all_tags_true.append(batch["labels_tags"].numpy())
        all_tags_probs.append(tags_probs)
        all_dif_true.append(batch["labels_dif"].numpy())
        all_dif_probs.append(dif_probs)
        all_dif_pred.append(dif_pred)

    denom = max(n_examples, 1)
    return {
        "loss": total_loss / denom,
        "loss_tags": total_loss_tags / denom,
        "loss_dif": total_loss_dif / denom,
        "y_tags_true": np.concatenate(all_tags_true, axis=0),
        "y_tags_probs": np.concatenate(all_tags_probs, axis=0),
        "y_dif_true": np.concatenate(all_dif_true, axis=0),
        "y_dif_probs": np.concatenate(all_dif_probs, axis=0),
        "y_dif_pred": np.concatenate(all_dif_pred, axis=0),
    }


def run_single_experiment(
    cfg: dict[str, Any],
    bundle: dict[str, Any],
    model_key: str,
    max_length: int,
    seed: int,
    run_dir: Path,
    task_mode_override: str | None = None,
) -> dict[str, Any]:
    """Executa treino completo + avaliação no teste para uma combinação (modelo, max_len, seed)."""
    run_start = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    model_spec = cfg["models"][model_key]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    tok_cfg = cfg["tokenization"]
    eval_cfg = cfg["evaluation"]
    labels_cfg = cfg["labels"]

    task_mode = task_mode_override or model_cfg.get("task_mode", "mtl")
    difficulty_head_type = model_cfg.get("difficulty_head_type", "classification")
    cache_dir = cfg["paths"].get("cache_dir")

    # Subamostragem opcional para dry-run
    exp_cfg = cfg.get("experiment", {})
    train_df = bundle["train"]
    val_df = bundle["val"]
    test_df = bundle["test"]
    if exp_cfg.get("max_train_samples"):
        train_df = train_df.head(int(exp_cfg["max_train_samples"])).reset_index(drop=True)
    if exp_cfg.get("max_val_samples"):
        val_df = val_df.head(int(exp_cfg["max_val_samples"])).reset_index(drop=True)
    if exp_cfg.get("max_test_samples"):
        test_df = test_df.head(int(exp_cfg["max_test_samples"])).reset_index(drop=True)

    tag_vocab = bundle["tag_vocab"]
    exclude_prefix = labels_cfg.get("exclude_tag_prefix", "*")
    y_tags_train = binarize_tags(train_df["clean_tags"], tag_vocab, exclude_prefix=exclude_prefix)
    y_tags_val = binarize_tags(val_df["clean_tags"], tag_vocab, exclude_prefix=exclude_prefix)
    y_tags_test = binarize_tags(test_df["clean_tags"], tag_vocab, exclude_prefix=exclude_prefix)

    pos_weight_tensor = None
    if model_cfg.get("use_pos_weight", False):
        pos_weight_tensor = torch.tensor(compute_pos_weight(y_tags_train), dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_spec["hf_name"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    fields = list(tok_cfg.get("fields", ["title", "description", "input_format"]))
    strategy = tok_cfg.get("truncation_strategy", "reserve_input_format")
    reserve_tok = int(tok_cfg.get("reserve_input_format_tokens", 128))
    head_tok = int(tok_cfg.get("head_tail_head_tokens", 128))

    train_ds = ProblemDataset(train_df, y_tags_train, tokenizer, max_length, fields, strategy, reserve_tok, head_tok)
    val_ds = ProblemDataset(val_df, y_tags_val, tokenizer, max_length, fields, strategy, reserve_tok, head_tok)
    test_ds = ProblemDataset(test_df, y_tags_test, tokenizer, max_length, fields, strategy, reserve_tok, head_tok)

    truncation_stats = {
        "train": {"n_truncated": train_ds.n_truncated, "total": len(train_ds), "pct": round(train_ds.pct_truncated, 2)},
        "val": {"n_truncated": val_ds.n_truncated, "total": len(val_ds), "pct": round(val_ds.pct_truncated, 2)},
        "test": {"n_truncated": test_ds.n_truncated, "total": len(test_ds), "pct": round(test_ds.pct_truncated, 2)},
    }

    # Batch size por dispositivo e gradient accumulation para atingir batch efetivo configurado
    eff_bs = int(train_cfg.get("effective_batch_size", 32))
    per_dev_map = model_spec.get("per_device_batch_size", {}) or {}
    per_device_bs = int(per_dev_map.get(max_length, per_dev_map.get(str(max_length), 8)))
    per_device_bs = min(per_device_bs, eff_bs)
    grad_accum = max(1, math.ceil(eff_bs / per_device_bs))

    collate_fn = make_collate_fn(int(tokenizer.pad_token_id))
    num_workers = int(train_cfg.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_bs,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=per_device_bs,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=per_device_bs,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_mode, amp_dtype, use_scaler = resolve_amp_config(train_cfg.get("mixed_precision", "auto"), device)
    use_amp = amp_dtype is not None and device.type == "cuda"

    model, model_meta = MultiTaskTransformer.from_pretrained_spec(
        model_spec=model_spec,
        max_length=max_length,
        num_tags=len(tag_vocab),
        num_difficulty_classes=len(labels_cfg.get("difficulty_names", [])) or 5,
        dropout=float(model_cfg.get("dropout", 0.1)),
        task_mode=task_mode,
        difficulty_head_type=difficulty_head_type,
        alpha_tags=float(train_cfg.get("alpha_tags", 1.0)),
        beta_dif=float(train_cfg.get("beta_dif", 1.0)),
        pos_weight=pos_weight_tensor,
        cache_dir=cache_dir,
    )
    model.to(device)

    lr = float(model_spec.get("learning_rate", train_cfg.get("learning_rate", 2e-5)))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    max_epochs = int(train_cfg.get("max_epochs", 20))
    patience = int(train_cfg.get("early_stopping_patience", 3))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_ckpt_path = run_dir / "best_checkpoint.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss_sum = 0.0
        train_count = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            bsz = batch["input_ids"].size(0)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels_tags = batch["labels_tags"].to(device, non_blocking=True)
            labels_dif = batch["labels_dif"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels_tags=labels_tags,
                    labels_dif=labels_dif,
                )
                loss = out["loss"] / grad_accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss_sum += float(out["loss"].item()) * bsz
            train_count += bsz

            if step_idx % grad_accum == 0 or step_idx == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_out = predict_loader(model, val_loader, device, amp_dtype)
        val_metrics = evaluate_predictions(
            val_out["y_tags_true"],
            val_out["y_tags_probs"],
            val_out["y_dif_true"],
            val_out["y_dif_pred"],
            tag_threshold=float(eval_cfg.get("tag_threshold", 0.5)),
        )

        ep_time = time.perf_counter() - ep_start
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_out["loss"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_loss_tags": val_out["loss_tags"],
                "val_loss_dif": val_out["loss_dif"],
                "val_tags_f1_micro": val_metrics["tags"]["f1_micro"],
                "val_tags_f1_macro": val_metrics["tags"]["f1_macro"],
                "val_dif_accuracy": val_metrics["difficulty"]["accuracy"],
                "val_dif_qwk": val_metrics["difficulty"]["qwk"],
                "epoch_seconds": round(ep_time, 2),
            }
        )

        print(
            f"  [{run_dir.name}] Época {epoch}/{max_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_qwk={val_metrics['difficulty']['qwk']:.4f} | "
            f"val_f1_micro={val_metrics['tags']['f1_micro']:.4f} | {ep_time:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Recarrega o melhor checkpoint (escolhido exclusivamente pela val_loss)
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))

    # Ajuste opcional de limiares por classe SOMENTE na validação
    per_class_thresholds = None
    if eval_cfg.get("tune_thresholds_on_val", False):
        best_val_out = predict_loader(model, val_loader, device, amp_dtype)
        per_class_thresholds = tune_per_class_thresholds(
            best_val_out["y_tags_true"],
            best_val_out["y_tags_probs"],
            step=float(eval_cfg.get("threshold_grid_step", 0.05)),
        )

    # Avaliação final no conjunto de teste
    test_out = predict_loader(model, test_loader, device, amp_dtype)
    test_metrics = evaluate_predictions(
        test_out["y_tags_true"],
        test_out["y_tags_probs"],
        test_out["y_dif_true"],
        test_out["y_dif_pred"],
        tag_threshold=float(eval_cfg.get("tag_threshold", 0.5)),
        per_class_thresholds=per_class_thresholds,
    )

    elapsed_sec = time.perf_counter() - run_start
    env_meta = get_environment_metadata()
    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated(device) / (1024**3), 3) if device.type == "cuda" else None
    )

    # Salva histórico de treino por época (.csv)
    pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

    # Salva predições/probabilidades do teste (.npz)
    np.savez_compressed(
        run_dir / "test_preds.npz",
        problem_ids=test_df["id"].astype(str).to_numpy(),
        y_tags_true=test_out["y_tags_true"],
        y_tags_probs=test_out["y_tags_probs"],
        y_dif_true=test_out["y_dif_true"],
        y_dif_probs=test_out["y_dif_probs"],
        y_dif_pred=test_out["y_dif_pred"],
    )

    # Salva configuração exata usada neste run
    run_config_snapshot = {
        "model_key": model_key,
        "hf_name": model_spec["hf_name"],
        "max_length": max_length,
        "seed": seed,
        "task_mode": task_mode,
        "difficulty_head_type": difficulty_head_type,
        "learning_rate": lr,
        "per_device_batch_size": per_device_bs,
        "grad_accum": grad_accum,
        "effective_batch_size_actual": per_device_bs * grad_accum,
        "amp_mode": amp_mode,
        "model_meta": model_meta,
        "tokenization": tok_cfg,
        "training": train_cfg,
        "evaluation": eval_cfg,
    }
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_config_snapshot, f, sort_keys=False, allow_unicode=True)

    # Remove checkpoint temporário ao finalizar (a menos que keep_checkpoint=True)
    if not train_cfg.get("keep_checkpoint", False) and best_ckpt_path.exists():
        best_ckpt_path.unlink()

    # Grava metrics.json por último (marcador atômico de conclusão do run)
    payload = {
        "model_key": model_key,
        "display_name": model_spec.get("display_name", model_key),
        "hf_name": model_spec["hf_name"],
        "max_length": max_length,
        "seed": seed,
        "task_mode": task_mode,
        "difficulty_head_type": difficulty_head_type,
        "best_epoch": best_epoch,
        "epochs_ran": len(history_rows),
        "best_val_loss": best_val_loss,
        "test_loss": test_out["loss"],
        "test_loss_tags": test_out["loss_tags"],
        "test_loss_dif": test_out["loss_dif"],
        "metrics": test_metrics,
        "truncation_stats": truncation_stats,
        "runtime_seconds": round(elapsed_sec, 2),
        "amp_mode": amp_mode,
        "peak_vram_gb": peak_vram_gb,
        "model_meta": model_meta,
        "environment": env_meta,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload
