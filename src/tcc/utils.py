"""Utilitários gerais: seeds determinísticas, metadados de ambiente e exportação .zip."""

from __future__ import annotations

import os
import platform
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Fixa seeds de Python, NumPy e PyTorch (quando instalado)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_environment_metadata() -> dict[str, Any]:
    """Coleta informações de hardware (GPU), Python e versões das bibliotecas."""
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "bf16_supported": False,
        "versions": {},
    }

    for pkg in ("torch", "transformers", "datasets", "sklearn", "scipy", "pandas", "numpy"):
        try:
            mod = __import__(pkg)
            info["versions"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info["versions"][pkg] = None

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_available"] = True
            info["gpu_name"] = props.name
            info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except ImportError:
        pass

    return info


def zip_results_dir(results_dir: str | Path, output_zip: str | Path) -> Path:
    """Empacota a pasta results/ em um .zip ignorando checkpoints pesados e cache."""
    src = Path(results_dir).resolve()
    dst = Path(output_zip).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    skip_names = {"best_checkpoint.pt", "last_state.pt"}
    skip_dirs = {"cache"}

    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(src)
            if any(part in skip_dirs for part in rel.parts):
                continue
            if path.name in skip_names:
                continue
            zf.write(path, arcname=str(rel))

    return dst
