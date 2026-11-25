"""Smoke test por arquitetura real (1 passo com batch 2) para validar memória e compatibilidade."""

from __future__ import annotations

import gc
import time
from typing import Any

import torch

from tcc.config import load_config
from tcc.model import MultiTaskTransformer
from tcc.train import resolve_amp_config


def run_architecture_smoke_tests(
    cfg: dict[str, Any],
    batch_size: int = 2,
    num_tags: int = 25,
    only_models: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Executa 1 passo (forward + backward + optimizer.step) com batch 2 para cada (modelo, max_len)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_mode, amp_dtype, use_scaler = resolve_amp_config(cfg["training"].get("mixed_precision", "auto"), device)
    use_amp = amp_dtype is not None and device.type == "cuda"
    cache_dir = cfg["paths"].get("cache_dir")

    results: list[dict[str, Any]] = []
    for model_key, spec in cfg["models"].items():
        if model_key == "bert-tiny":
            continue
        if only_models and model_key not in only_models:
            continue

        for max_len in spec.get("max_lengths", [512]):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            t0 = time.perf_counter()
            status = "PASS"
            error_msg = None
            peak_vram_gb = None
            attn_type = None
            use_gc = None

            try:
                model, meta = MultiTaskTransformer.from_pretrained_spec(
                    model_spec=spec,
                    max_length=int(max_len),
                    num_tags=num_tags,
                    num_difficulty_classes=5,
                    cache_dir=cache_dir,
                )
                attn_type = meta.get("attention_type")
                use_gc = meta.get("gradient_checkpointing")
                model.to(device)
                model.train()

                vocab_size = int(getattr(model.encoder.config, "vocab_size", 30000))
                input_ids = torch.randint(10, min(vocab_size, 10000), (batch_size, int(max_len)), device=device)
                attention_mask = torch.ones((batch_size, int(max_len)), dtype=torch.long, device=device)
                labels_tags = torch.randint(0, 2, (batch_size, num_tags), dtype=torch.float32, device=device)
                labels_dif = torch.randint(0, 5, (batch_size,), dtype=torch.long, device=device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
                scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels_tags=labels_tags,
                        labels_dif=labels_dif,
                    )
                    loss = out["loss"]

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)

                del model, optimizer, out, loss
            except Exception as exc:
                status = "FAIL"
                error_msg = f"{type(exc).__name__}: {exc}"

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            row = {
                "model": model_key,
                "max_len": int(max_len),
                "device": str(device),
                "amp_mode": amp_mode,
                "attention_type": attn_type,
                "grad_checkpointing": use_gc,
                "peak_vram_gb": peak_vram_gb,
                "seconds": round(time.perf_counter() - t0, 2),
                "status": status,
                "error": error_msg,
            }
            results.append(row)
            print(
                f"[SMOKE] {model_key} @ {max_len} -> {status} "
                f"(VRAM={peak_vram_gb} GB, amp={amp_mode}, attn={attn_type}, {row['seconds']}s)"
                + (f" | {error_msg}" if error_msg else "")
            )

    return results


if __name__ == "__main__":
    cfg = load_config()
    run_architecture_smoke_tests(cfg)
