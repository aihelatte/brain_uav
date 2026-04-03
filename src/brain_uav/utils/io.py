"""File helpers for checkpoints and structured result files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import json

import torch


def ensure_parent(path: str | Path) -> Path:
    """Create parent directory if needed and return the final Path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as Path."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
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


def save_csv_rows(path: str | Path, rows: list[dict]) -> Path:
    """Save a list of dict rows as CSV.

    适合把按 episode 分段的统计结果导出成表格，方便 Excel 和 AI 一起读。
    """

    target = ensure_parent(path)
    if not rows:
        target.write_text('', encoding='utf-8')
        return target
    fieldnames = list(rows[0].keys())
    with target.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def now_timestamp() -> str:
    """Return a compact local timestamp for output file names."""

    return datetime.now().strftime('%Y%m%d_%H%M%S')


def build_log_paths(
    base_output: str | Path,
    base_metrics: str | Path,
    timestamp: str,
    *,
    log_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Keep model path fixed, but place logs under a timestamped directory.

    返回：
    - log_dir
    - fixed model output path
    - timestamped metrics path inside log_dir
    """

    base_output = Path(base_output)
    base_metrics = Path(base_metrics)
    if log_root is None:
        log_dir = ensure_dir(base_output.parent / 'logs' / timestamp)
    else:
        log_dir = ensure_dir(Path(log_root) / timestamp)
    output = ensure_parent(base_output)
    metrics = log_dir / base_metrics.name
    return log_dir, output, metrics


def model_output_path(stage: str, *, model: str, level: str | None = None) -> Path:
    """Return the default reusable model path under outputs/models/."""

    root = Path('outputs/models')
    if stage == 'bc':
        return root / 'bc' / f'bc_{model}_latest.pt'
    if stage == 'td3':
        if level is None:
            raise ValueError('level is required for TD3 model outputs')
        return root / level / f'td3_{model}_{level}_latest.pt'
    if stage == 'bootstrap':
        return root / 'bootstrap' / f'{model}_bootstrap_latest.pt'
    raise ValueError(f'Unsupported stage: {stage}')


def log_root_path(stage: str, *, level: str | None = None) -> Path:
    """Return the default log root under outputs/logs/."""

    root = Path('outputs/logs')
    if stage == 'bc':
        return root / 'bc'
    if stage == 'td3':
        if level is None:
            raise ValueError('level is required for TD3 log roots')
        return root / level
    if stage == 'benchmark':
        return root / 'benchmark'
    raise ValueError(f'Unsupported stage: {stage}')
