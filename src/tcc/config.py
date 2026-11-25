"""Carregamento e mesclagem de arquivos YAML de configuração."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Mescla dois dicionários recursivamente sem mutar as entradas."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def find_project_root(start: Path | None = None) -> Path:
    """Localiza a raiz do repositório procurando por pyproject.toml ou configs/base.yaml."""
    candidates = [start or Path.cwd(), Path(__file__).resolve().parent]
    for origin in candidates:
        current = origin.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / "pyproject.toml").exists() or (candidate / "configs" / "base.yaml").exists():
                return candidate
    return (start or Path.cwd()).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Carrega um arquivo YAML como dicionário."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_paths(cfg: dict[str, Any], project_root: Path, results_dir_override: str | Path | None = None) -> dict[str, Any]:
    """Resolve os diretórios configurados, permitindo redirecionar results/ (ex.: Google Drive)."""
    resolved = copy.deepcopy(cfg)
    env_results = os.environ.get("TCC_RESULTS_DIR")
    results_override = results_dir_override or env_results

    paths_cfg = resolved.setdefault("paths", {})
    if results_override:
        base_results = Path(results_override).expanduser().resolve()
        paths_cfg["results_dir"] = str(base_results)
        paths_cfg["cache_dir"] = str(base_results / "cache")
        paths_cfg["splits_dir"] = str(base_results / "splits")
        paths_cfg["runs_dir"] = str(base_results / "runs")
        paths_cfg["tables_dir"] = str(base_results / "tables")
        paths_cfg["figures_dir"] = str(base_results / "figures")
    else:
        for key in ("results_dir", "cache_dir", "splits_dir", "runs_dir", "tables_dir", "figures_dir"):
            if key in paths_cfg:
                raw_path = Path(paths_cfg[key])
                paths_cfg[key] = str(raw_path if raw_path.is_absolute() else (project_root / raw_path).resolve())

    resolved["_project_root"] = str(project_root)
    return resolved


def load_config(
    base_path: str | Path | None = None,
    models_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    results_dir_override: str | Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Carrega base.yaml, models.yaml e opcionalmente um YAML de experimento."""
    root = project_root or find_project_root()
    base_file = Path(base_path) if base_path else root / "configs" / "base.yaml"
    models_file = Path(models_path) if models_path else root / "configs" / "models.yaml"

    cfg = load_yaml(base_file)
    if models_file.exists():
        cfg = deep_merge(cfg, load_yaml(models_file))
    if experiment_path is not None:
        exp_file = Path(experiment_path)
        if not exp_file.is_absolute() and not exp_file.exists():
            exp_file = root / exp_file
        cfg = deep_merge(cfg, load_yaml(exp_file))

    return resolve_paths(cfg, root, results_dir_override=results_dir_override)
