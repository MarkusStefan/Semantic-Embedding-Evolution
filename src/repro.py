from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch


PathLike = Union[str, os.PathLike, Path]


def repo_root(start: PathLike | None = None) -> Path:
    """Resolve repository root robustly.

    Heuristic: walk upward until a directory containing both `src/` and `README.md`.
    """

    base = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for cur in (base, *base.parents):
        if (cur / "src").is_dir() and (cur / "README.md").is_file():
            return cur
    return base


def seed_everything(seed: int = 0, deterministic: bool = True) -> None:
    """Seed Python/NumPy/PyTorch for reproducible experiments.

    Notes:
    - Full determinism on CUDA may reduce performance.
    - Some ops may remain nondeterministic depending on PyTorch/CUDA version.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
