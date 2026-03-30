"""File helpers for checkpoints and JSON results."""

from __future__ import annotations

from pathlib import Path
import json

import torch


def ensure_parent(path: str | Path) -> Path:
    """Create parent directory if needed and return the final Path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_checkpoint(path: str | Path, payload: dict) -> Path:
    """Save a PyTorch checkpoint."""

    target = ensure_parent(path)
    torch.save(payload, target)
    return target


def load_checkpoint(path: str | Path) -> dict:
    """Load a local checkpoint.

    这里显式设置 `weights_only=False`，是因为我们保存的不只是权重，
    还包含训练指标和配置。
    """

    return torch.load(path, map_location='cpu', weights_only=False)


def save_json(path: str | Path, payload: dict | list) -> Path:
    """Save a dict/list as UTF-8 JSON."""

    target = ensure_parent(path)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return target
