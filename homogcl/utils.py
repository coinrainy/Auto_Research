from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def git_state(cwd: str | os.PathLike[str]) -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "UNAVAILABLE"

    status = run(["git", "status", "--porcelain"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def dependency_state() -> dict[str, str]:
    deps = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    try:
        import torch_geometric

        deps["torch_geometric"] = torch_geometric.__version__
    except Exception as exc:  # pragma: no cover - diagnostic only.
        deps["torch_geometric"] = f"UNAVAILABLE: {exc}"
    return deps
