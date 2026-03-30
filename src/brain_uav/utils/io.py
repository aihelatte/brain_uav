from __future__ import annotations

from pathlib import Path
import json

import torch


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_checkpoint(path: str | Path, payload: dict) -> Path:
    target = ensure_parent(path)
    torch.save(payload, target)
    return target


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def save_json(path: str | Path, payload: dict | list) -> Path:
    target = ensure_parent(path)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return target
