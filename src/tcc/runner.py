"""Orquestrador CLI de experimentos com retomada automática e estimativa de tempo restante (ETA)."""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

import torch

from tcc.config import load_config
from tcc.data import load_or_prepare_splits
from tcc.smoke_test import run_architecture_smoke_tests
from tcc.train import run_single_experiment


def _format_duration(seconds: float) -> str:
    sec = int(max(seconds, 0))
    hours, rem = divmod(sec, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def build_run_plan(
    cfg: dict[str, Any],
    filter_models: list[str] | None = None,
    filter_seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Expande a grade (modelo × max_len × seed) definida no YAML de experimento."""
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "default")
    seeds = filter_seeds if filter_seeds is not None else [int(s) for s in cfg["project"]["seeds"]]

    plan: list[dict[str, Any]] = []
    for entry in exp_cfg.get("runs", []):
        model_key = entry["model"]
        if filter_models and model_key not in filter_models:
            continue
        task_mode = entry.get("task_mode")
        for max_len in entry.get("max_lengths", [512]):
            for seed in seeds:
                suffix = f"_{task_mode}" if task_mode else ""
                run_id = f"{model_key}_len{int(max_len)}{suffix}_seed{seed}"
                plan.append(
                    {
                        "experiment": exp_name,
                        "model_key": model_key,
                        "max_length": int(max_len),
                        "seed": int(seed),
                        "task_mode": task_mode,
                        "run_id": run_id,
                    }
                )
    return plan


def run_experiment_grid(
    experiment_config: str | Path,
    results_dir: str | Path | None = None,
    filter_models: list[str] | None = None,
    filter_seeds: list[int] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Executa a grade completa pulando runs já finalizados (com metrics.json) e exibindo ETA."""
    cfg = load_config(experiment_path=experiment_config, results_dir_override=results_dir)
    exp_cfg = cfg.get("experiment", {})
    if exp_cfg.get("enabled", True) is False and not force:
        print(
            f"[AVISO] O experimento '{exp_cfg.get('name')}' está desligado por padrão (enabled: false). "
            "Use --force para executá-lo."
        )
        return []

    bundle = load_or_prepare_splits(cfg, force_rebuild=False)
    plan = build_run_plan(cfg, filter_models=filter_models, filter_seeds=filter_seeds)
    runs_root = Path(cfg["paths"]["runs_dir"]) / exp_cfg.get("name", "default")
    runs_root.mkdir(parents=True, exist_ok=True)

    total_runs = len(plan)
    pending = [
        item for item in plan if force or not (runs_root / item["run_id"] / "metrics.json").exists()
    ]
    skipped = total_runs - len(pending)

    print(
        f"Experimento: {exp_cfg.get('name')} | Total na grade: {total_runs} | "
        f"Já concluídos (pulados): {skipped} | Pendentes: {len(pending)}"
    )

    measured_times: list[float] = []
    completed_payloads: list[dict[str, Any]] = []

    for idx, item in enumerate(pending, start=1):
        run_dir = runs_root / item["run_id"]
        print(
            f"\n[{idx}/{len(pending)}] Iniciando {item['run_id']} "
            f"(modelo={item['model_key']}, max_len={item['max_length']}, seed={item['seed']})"
        )

        t0 = time.perf_counter()
        payload = run_single_experiment(
            cfg=cfg,
            bundle=bundle,
            model_key=item["model_key"],
            max_length=item["max_length"],
            seed=item["seed"],
            run_dir=run_dir,
            task_mode_override=item["task_mode"],
        )
        dt = time.perf_counter() - t0
        measured_times.append(dt)
        completed_payloads.append(payload)

        # Limpeza de memória GPU entre runs independentes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        remaining = len(pending) - idx
        avg_sec = sum(measured_times) / len(measured_times)
        eta_sec = avg_sec * remaining

        dif = payload["metrics"]["difficulty"]
        tags = payload["metrics"]["tags"]
        tags_tuned = payload["metrics"].get("tags_tuned_val")
        trunc_tr = payload["truncation_stats"]["train"]
        tuned_str = (
            f" | Tags(val-thr) F1-micro={tags_tuned['f1_micro']:.4f}, F1-macro={tags_tuned['f1_macro']:.4f}"
            if tags_tuned
            else ""
        )
        print(
            f"[OK] {item['run_id']} concluído em {_format_duration(dt)} | "
            f"val_loss={payload['best_val_loss']:.4f} (época {payload['best_epoch']}) | "
            f"Teste: Dif Acc={dif['accuracy']:.4f}, QWK={dif['qwk']:.4f}, "
            f"Tags(0.5) F1-micro={tags['f1_micro']:.4f}, F1-macro={tags['f1_macro']:.4f}"
            f"{tuned_str} | "
            f"Truncados treino: {trunc_tr['n_truncated']}/{trunc_tr['total']} ({trunc_tr['pct']}%)"
        )
        if remaining > 0:
            print(
                f"     Estimativa restante da grade ({remaining} runs, média {_format_duration(avg_sec)}/run): "
                f"~{_format_duration(eta_sec)}"
            )

    return completed_payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner de experimentos do TCC (com retomada e ETA).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/main.yaml",
        help="Caminho para o YAML do experimento (ex.: configs/experiments/dry_run.yaml).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Sobrescreve o diretório results/ (ex.: /content/drive/MyDrive/tcc2/results).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filtra modelos específicos (ex.: --models bert-base-uncased roberta-base).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Filtra seeds específicas (ex.: --seeds 42 43).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reexecuta runs mesmo que metrics.json já exista (ou ativa ablações desligadas).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Executa smoke test de 1 passo (batch=2) para cada arquitetura real em models.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        cfg = load_config(experiment_path=args.config, results_dir_override=args.results_dir)
        run_architecture_smoke_tests(cfg, batch_size=2, only_models=args.models)
        return

    run_experiment_grid(
        experiment_config=args.config,
        results_dir=args.results_dir,
        filter_models=args.models,
        filter_seeds=args.seeds,
        force=args.force,
    )


if __name__ == "__main__":
    main()
