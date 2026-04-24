"""Train the behavior cloning initialization model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..scripts.common import (
    DEVICE_CHOICES,
    SNN_BACKEND_CHOICES,
    build_log_prefix,
    configure_training_runtime,
    make_actor,
)
from ..trainers import train_behavior_cloning
from ..utils.io import (
    build_log_paths,
    load_checkpoint,
    log_root_path,
    model_output_path,
    now_timestamp,
    save_checkpoint,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train behavior cloning initialization.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', choices=['snn', 'ann'], default='snn')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--init-checkpoint', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--best-output', type=Path, default=None)
    parser.add_argument('--metrics-out', type=Path, default=None)
    parser.add_argument('--log-root', type=Path, default=None)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--snn-backend', choices=SNN_BACKEND_CHOICES, default='torch')
    return parser


def build_bc_checkpoint_payload(
    *,
    model: str,
    actor,
    history: list[float],
    cfg: ExperimentConfig,
    finished_at: str,
    log_dir: Path,
    dataset_path: Path,
    dataset_version: str,
    dataset_config: dict | None,
    curriculum_level: str,
    curriculum_mix: dict[str, float],
    init_checkpoint: Path | None,
    best_loss: float | None,
    best_epoch: int | None,
    checkpoint_kind: str,
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
        'checkpoint_kind': checkpoint_kind,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = ExperimentConfig()
    resolved_device = configure_training_runtime(
        cfg,
        model_type=args.model,
        device=args.device,
        snn_backend=args.snn_backend,
    )
    log_prefix = build_log_prefix(args.model, 'bc')
    data = np.load(args.dataset)
    dataset_version = str(data['dataset_version']) if 'dataset_version' in data else 'unknown'
    dataset_config = json.loads(str(data['config_json'])) if 'config_json' in data else None
    curriculum_level = str(data['curriculum_level']) if 'curriculum_level' in data else 'easy'
    curriculum_mix = json.loads(str(data['curriculum_mix'])) if 'curriculum_mix' in data else {curriculum_level: 1.0}
    finished_at = now_timestamp()
    base_output = args.output or model_output_path('bc', model=args.model)
    base_metrics = args.metrics_out or Path(f'bc_{args.model}_metrics.json')
    log_dir, output, metrics_out = build_log_paths(
        base_output,
        base_metrics,
        finished_at,
        log_root=args.log_root or log_root_path('bc'),
    )
    if args.best_output is None:
        best_output = output.with_name(f'{output.stem}_best{output.suffix}')
    else:
        best_output = args.best_output

    actor = make_actor(cfg, args.model, data['observations'].shape[1], data['actions'].shape[1])
    if args.init_checkpoint is not None:
        actor.load_state_dict(load_checkpoint(args.init_checkpoint)['state_dict'])

    best_loss = math.inf
    best_epoch = -1

    def save_best_checkpoint(epoch: int, epoch_loss: float, history: list[float], current_actor) -> None:
        nonlocal best_loss, best_epoch
        if epoch_loss >= best_loss:
            return
        best_loss = epoch_loss
        best_epoch = epoch
        save_checkpoint(
            best_output,
            build_bc_checkpoint_payload(
                model=args.model,
                actor=current_actor,
                history=history,
                cfg=cfg,
                finished_at=finished_at,
                log_dir=log_dir,
                dataset_path=args.dataset,
                dataset_version=dataset_version,
                dataset_config=dataset_config,
                curriculum_level=curriculum_level,
                curriculum_mix=curriculum_mix,
                init_checkpoint=args.init_checkpoint,
                best_loss=best_loss,
                best_epoch=best_epoch,
                checkpoint_kind='best',
            ),
        )

    history = train_behavior_cloning(
        actor,
        args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=cfg.training.actor_lr,
        device=cfg.training.device,
        epoch_end_callback=save_best_checkpoint,
        log_prefix=log_prefix,
    )

    recorded_best_loss = None if best_epoch < 0 else best_loss
    recorded_best_epoch = None if best_epoch < 0 else best_epoch
    save_checkpoint(
        output,
        build_bc_checkpoint_payload(
            model=args.model,
            actor=actor,
            history=history,
            cfg=cfg,
            finished_at=finished_at,
            log_dir=log_dir,
            dataset_path=args.dataset,
            dataset_version=dataset_version,
            dataset_config=dataset_config,
            curriculum_level=curriculum_level,
            curriculum_mix=curriculum_mix,
            init_checkpoint=args.init_checkpoint,
            best_loss=recorded_best_loss,
            best_epoch=recorded_best_epoch,
            checkpoint_kind='final',
        ),
    )
    save_json(
        metrics_out,
        {
            'model': args.model,
            'loss_history': history,
            'final_loss': history[-1],
            'best_loss': recorded_best_loss,
            'best_epoch': recorded_best_epoch,
            'best_output': str(best_output),
            'finished_at': finished_at,
            'log_dir': str(log_dir),
            'device': resolved_device,
            'snn_backend': cfg.training.snn_backend if args.model == 'snn' else None,
            'dataset_path': str(args.dataset),
            'dataset_version': dataset_version,
            'curriculum_level': curriculum_level,
            'curriculum_mix': curriculum_mix,
            'init_checkpoint': str(args.init_checkpoint) if args.init_checkpoint else None,
        },
    )
    print(f'{log_prefix} saved checkpoint to {output}')
    print(f'{log_prefix} saved best checkpoint to {best_output}')
    print(f'{log_prefix} saved metrics to {metrics_out}')
    print(f'{log_prefix} dataset version={dataset_version} curriculum={curriculum_level} device={resolved_device}')
    print(f'{log_prefix} final loss={history[-1]:.6f}')


if __name__ == '__main__':
    main()
