"""Train the behavior cloning initialization model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..scripts.common import make_actor
from ..trainers import train_behavior_cloning
from ..utils.io import build_log_paths, now_timestamp, save_checkpoint, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Train behavior cloning initialization.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', choices=['snn', 'ann'], default='snn')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--metrics-out', type=Path, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    data = np.load(args.dataset)
    dataset_version = str(data['dataset_version']) if 'dataset_version' in data else 'unknown'
    dataset_config = None
    if 'config_json' in data:
        dataset_config = json.loads(str(data['config_json']))
    else:
        print('[BC] warning: dataset has no config_json metadata; please prefer a v5 dataset regenerated under current rules')

    actor = make_actor(cfg, args.model, data['observations'].shape[1], data['actions'].shape[1])
    history = train_behavior_cloning(
        actor,
        args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=cfg.training.actor_lr,
        device=cfg.training.device,
    )

    finished_at = now_timestamp()
    base_output = args.output or Path(f'outputs/formal_v1/bc_{args.model}_formal_v5.pt')
    base_metrics = args.metrics_out or Path(f'outputs/formal_v1/bc_{args.model}_formal_v5_metrics.json')
    log_dir, output, metrics_out = build_log_paths(base_output, base_metrics, finished_at)

    save_checkpoint(
        output,
        {
            'model_type': args.model,
            'state_dict': actor.state_dict(),
            'loss_history': history,
            'config': cfg.to_dict(),
            'finished_at': finished_at,
            'log_dir': str(log_dir),
            'dataset_path': str(args.dataset),
            'dataset_version': dataset_version,
            'dataset_config': dataset_config,
        },
    )
    save_json(
        metrics_out,
        {
            'model': args.model,
            'loss_history': history,
            'final_loss': history[-1],
            'finished_at': finished_at,
            'log_dir': str(log_dir),
            'dataset_path': str(args.dataset),
            'dataset_version': dataset_version,
        },
    )
    print(f'Saved BC checkpoint to {output}')
    print(f'Saved BC metrics to {metrics_out}')
    print(f'BC dataset version: {dataset_version}')
    print(f'Final BC loss: {history[-1]:.6f}')


if __name__ == '__main__':
    main()
