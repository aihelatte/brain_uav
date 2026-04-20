"""Train the behavior cloning initialization model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import ExperimentConfig
from ..scripts.common import make_actor
from ..utils.io import (
    build_log_paths,
    load_checkpoint,
    log_root_path,
    model_output_path,
    now_timestamp,
    save_checkpoint,
    save_json,
)


def _resolve_device(requested: str) -> str:
    if requested == 'cpu':
        return 'cpu'
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('--device cuda was requested, but torch.cuda.is_available() is False.')
        return 'cuda'
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    raise ValueError(f'Unsupported device: {requested}')


def _device_log_line(prefix: str, device: str) -> str:
    cuda_version = torch.version.cuda or 'unavailable'
    gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'n/a'
    return f'[{prefix}] device={device}, torch_cuda={cuda_version}, gpu={gpu_name}'


def _checkpoint_payload(
    *,
    actor,
    model: str,
    history: list[float],
    cfg: ExperimentConfig,
    finished_at: str,
    log_dir: Path,
    dataset_path: Path,
    dataset_version: str,
    dataset_config: dict | None,
    curriculum_level: str,
    curriculum_mix: dict,
    init_checkpoint: Path | None,
    best_loss: float | None = None,
    best_epoch: int | None = None,
) -> dict:
    return {
        'model_type': model,
        'state_dict': actor.state_dict(),
        'loss_history': history,
        'config': cfg.to_dict(),
        'finished_at': finished_at,
        'log_dir': str(log_dir),
        'dataset_path': str(dataset_path),
        'dataset_version': dataset_version,
        'dataset_config': dataset_config,
        'curriculum_level': curriculum_level,
        'curriculum_mix': curriculum_mix,
        'init_checkpoint': str(init_checkpoint) if init_checkpoint else None,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
    }


def train_behavior_cloning_with_best(
    *,
    actor,
    dataset_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    best_output: Path | None,
    checkpoint_context: dict,
) -> tuple[list[float], float, int]:
    payload = np.load(dataset_path)
    obs = torch.tensor(payload['observations'], dtype=torch.float32)
    actions = torch.tensor(payload['actions'], dtype=torch.float32)
    loader = DataLoader(TensorDataset(obs, actions), batch_size=batch_size, shuffle=True)
    actor.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history: list[float] = []
    best_loss = math.inf
    best_epoch = 0
    print(f"[BC] dataset={dataset_path} samples={len(obs)} batch_size={batch_size} epochs={epochs}")
    for epoch_idx in range(epochs):
        running = 0.0
        count = 0
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            pred = actor(batch_obs)
            loss = criterion(pred, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(batch_obs)
            count += len(batch_obs)
        epoch_loss = running / max(count, 1)
        history.append(epoch_loss)
        print(f"[BC] epoch {epoch_idx + 1}/{epochs} loss={epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch_idx + 1
            if best_output is not None:
                actor.to('cpu')
                save_checkpoint(
                    best_output,
                    _checkpoint_payload(
                        actor=actor,
                        history=history.copy(),
                        best_loss=float(best_loss),
                        best_epoch=int(best_epoch),
                        **checkpoint_context,
                    ),
                )
                actor.to(device)
                print(f"[BC] saved best checkpoint epoch={best_epoch} loss={best_loss:.6f} to {best_output}")
    actor.to('cpu')
    return history, float(best_loss), int(best_epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train behavior cloning initialization.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', choices=['snn', 'ann'], default='snn')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--init-checkpoint', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--best-output', type=Path, default=None)
    parser.add_argument('--metrics-out', type=Path, default=None)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.training.device = _resolve_device(args.device)
    print(_device_log_line('train_bc', cfg.training.device))
    data = np.load(args.dataset)
    dataset_version = str(data['dataset_version']) if 'dataset_version' in data else 'unknown'
    dataset_config = json.loads(str(data['config_json'])) if 'config_json' in data else None
    curriculum_level = str(data['curriculum_level']) if 'curriculum_level' in data else 'easy'
    curriculum_mix = json.loads(str(data['curriculum_mix'])) if 'curriculum_mix' in data else {curriculum_level: 1.0}

    actor = make_actor(cfg, args.model, data['observations'].shape[1], data['actions'].shape[1])
    if args.init_checkpoint is not None:
        actor.load_state_dict(load_checkpoint(args.init_checkpoint)['state_dict'])

    finished_at = now_timestamp()
    base_output = args.output or model_output_path('bc', model=args.model)
    base_metrics = args.metrics_out or Path(f'bc_{args.model}_metrics.json')
    log_dir, output, metrics_out = build_log_paths(
        base_output,
        base_metrics,
        finished_at,
        log_root=log_root_path('bc'),
    )
    best_output = Path(args.best_output) if args.best_output is not None else None
    checkpoint_context = {
        'model': args.model,
        'cfg': cfg,
        'finished_at': finished_at,
        'log_dir': log_dir,
        'dataset_path': args.dataset,
        'dataset_version': dataset_version,
        'dataset_config': dataset_config,
        'curriculum_level': curriculum_level,
        'curriculum_mix': curriculum_mix,
        'init_checkpoint': args.init_checkpoint,
    }
    history, best_loss, best_epoch = train_behavior_cloning_with_best(
        actor=actor,
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=cfg.training.actor_lr,
        device=cfg.training.device,
        best_output=best_output,
        checkpoint_context=checkpoint_context,
    )

    save_checkpoint(
        output,
        _checkpoint_payload(
            actor=actor,
            history=history,
            best_loss=float(best_loss),
            best_epoch=int(best_epoch),
            **checkpoint_context,
        ),
    )
    save_json(
        metrics_out,
        {
            'model': args.model,
            'loss_history': history,
            'final_loss': history[-1],
            'best_loss': float(best_loss),
            'best_epoch': int(best_epoch),
            'best_checkpoint': str(best_output) if best_output is not None else None,
            'final_checkpoint': str(output),
            'finished_at': finished_at,
            'log_dir': str(log_dir),
            'dataset_path': str(args.dataset),
            'dataset_version': dataset_version,
            'device': cfg.training.device,
            'curriculum_level': curriculum_level,
            'curriculum_mix': curriculum_mix,
            'init_checkpoint': str(args.init_checkpoint) if args.init_checkpoint else None,
        },
    )
    print(f'Saved BC checkpoint to {output}')
    if best_output is not None:
        print(f'Saved best BC checkpoint to {best_output}')
    print(f'Saved BC metrics to {metrics_out}')
    print(f'BC dataset version: {dataset_version}, curriculum={curriculum_level}')
    print(f'Final BC loss: {history[-1]:.6f}')


if __name__ == '__main__':
    main()
