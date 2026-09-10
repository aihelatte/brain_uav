"""Train ANN or strict SpikingJelly SNN actors from V2 trajectory clusters."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from brain_uav.config import TrainingConfig
from brain_uav.models import (
    V2ANNPolicyActor,
    V2SNNPolicyActor,
    require_v2_spikingjelly,
)
from brain_uav.models.zone_set_encoder import ZoneSetEncoderConfig
from brain_uav.observations import V2ObservationScales
from brain_uav.scripts.common import DEVICE_CHOICES, resolve_training_device
from brain_uav.trainers.v2_bc import (
    V2BCTrainingConfig,
    V2BCTrajectoryCluster,
    build_v2_bc_checkpoint_payload,
    build_v2_snn_bc_checkpoint_payload,
    load_v2_bc_trajectory_cluster,
    split_v2_bc_scenarios,
    train_v2_bc_actor,
)
from brain_uav.trainers.v2_reporting import V2BCTrainingReporter
from brain_uav.utils.seeding import set_global_seed


def _resolve_v2_bc_device(
    device: str | torch.device,
) -> tuple[str, str]:
    """Resolve one requested BC device without allowing CUDA fallback."""

    if isinstance(device, torch.device):
        requested = str(device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError(f'Unsupported device: {requested}')
        resolved_type = resolve_training_device(device.type)
        resolved = requested if device.type == 'cuda' and device.index is not None else resolved_type
        return requested, resolved
    if not isinstance(device, str):
        raise TypeError('device must be a string or torch.device.')
    if device in DEVICE_CHOICES:
        return device, resolve_training_device(device)
    try:
        parsed = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f'Unsupported device: {device}') from exc
    if parsed.type not in ('cpu', 'cuda'):
        raise ValueError(f'Unsupported device: {device}')
    resolved_type = resolve_training_device(parsed.type)
    resolved = str(parsed) if parsed.type == 'cuda' and parsed.index is not None else resolved_type
    return device, resolved


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f'Output path exists and is not a directory: {path}')
        if any(path.iterdir()):
            raise FileExistsError(f'Refusing to overwrite non-empty output directory: {path}')
    else:
        path.mkdir(parents=True)


def _write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {path}')
    text = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(text + '\n', encoding='utf-8')


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing checkpoint: {path}')
    torch.save(payload, path)


def build_v2_bc_actor(
    cluster: V2BCTrajectoryCluster,
    *,
    actor_hidden_dim: int,
    encoder_config: ZoneSetEncoderConfig | None = None,
    model: str = 'ann',
    snn_time_window: int = 4,
    snn_tau: float = 2.0,
) -> V2ANNPolicyActor | V2SNNPolicyActor:
    """Construct one strict V2 actor from bound trajectory provenance."""

    if not isinstance(cluster, V2BCTrajectoryCluster):
        raise TypeError('cluster must be a V2BCTrajectoryCluster.')
    if type(actor_hidden_dim) is not int or actor_hidden_dim <= 0:
        raise ValueError('actor_hidden_dim must be a positive integer.')
    if encoder_config is None:
        encoder_config = ZoneSetEncoderConfig()
    if not isinstance(encoder_config, ZoneSetEncoderConfig):
        raise TypeError('encoder_config must be a ZoneSetEncoderConfig.')
    if model not in ('ann', 'snn'):
        raise ValueError('model must be "ann" or "snn".')
    scenario = cluster.scenario_config
    scales = V2ObservationScales(
        world_xy=float(scenario.world_xy),
        world_z_min=float(scenario.world_z_min),
        world_z_max=float(scenario.world_z_max),
        gamma_max=float(scenario.gamma_max),
    )
    action_limit = torch.tensor(
        [scenario.delta_gamma_max, scenario.delta_psi_max],
        dtype=torch.float32,
    )
    common = dict(
        scales=scales,
        action_dim=2,
        hidden_dim=actor_hidden_dim,
        action_limit=action_limit,
        uav_radius=cluster.uav_collision_radius,
        encoder_config=encoder_config,
    )
    if model == 'ann':
        return V2ANNPolicyActor(**common)
    return V2SNNPolicyActor(
        **common,
        time_window=snn_time_window,
        tau=snn_tau,
    )


def train_v2_behavior_cloning(
    *,
    trajectory_cluster: str | Path,
    output_dir: str | Path,
    seed: int | None = None,
    validation_fraction: float = 0.2,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device: str | torch.device | None = None,
    training_config: TrainingConfig | None = None,
    model: str = 'ann',
    snn_time_window: int = 4,
    shard_cache_mb: float = 256.0,
    deduplicate_identical_trajectories: bool = False,
) -> dict[str, Any]:
    """Run independent V2 ANN or SNN BC and write strict artifacts."""

    defaults = training_config or TrainingConfig()
    if not isinstance(defaults, TrainingConfig):
        raise TypeError('training_config must be a TrainingConfig.')
    effective_seed = defaults.seed if seed is None else seed
    effective_epochs = defaults.bc_epochs if epochs is None else epochs
    effective_batch_size = defaults.batch_size if batch_size is None else batch_size
    effective_learning_rate = defaults.actor_lr if learning_rate is None else learning_rate
    requested_device, resolved_device = _resolve_v2_bc_device(
        defaults.device if device is None else device
    )

    run_config = V2BCTrainingConfig(
        epochs=effective_epochs,
        batch_size=effective_batch_size,
        learning_rate=effective_learning_rate,
        seed=effective_seed,
        validation_fraction=validation_fraction,
        device=resolved_device,
    )
    if model not in ('ann', 'snn'):
        raise ValueError('model must be "ann" or "snn".')
    if type(snn_time_window) is not int or snn_time_window <= 0:
        raise ValueError('snn_time_window must be a positive integer.')
    if model == 'snn':
        require_v2_spikingjelly()
    print(json.dumps({
        'event': 'v2_bc_start',
        'model': model,
        'batch_size': run_config.batch_size,
        'shard_cache_mb': shard_cache_mb,
        'deduplicate_identical_trajectories': deduplicate_identical_trajectories,
        'requested_device': requested_device,
        'resolved_device': resolved_device,
    }, allow_nan=False, ensure_ascii=False), flush=True)
    cluster = load_v2_bc_trajectory_cluster(
        trajectory_cluster,
        shard_cache_mb=shard_cache_mb,
        deduplicate_identical_trajectories=deduplicate_identical_trajectories,
    )
    split = split_v2_bc_scenarios(
        cluster,
        validation_fraction=run_config.validation_fraction,
        seed=run_config.seed,
    )
    output = Path(output_dir)
    _prepare_output_directory(output)

    # This must precede Actor construction so PyTorch parameter initialization
    # is governed by the same explicit run seed as the split and epoch ordering.
    set_global_seed(run_config.seed)
    actor = build_v2_bc_actor(
        cluster,
        actor_hidden_dim=defaults.hidden_dim,
        encoder_config=ZoneSetEncoderConfig(),
        model=model,
        snn_time_window=snn_time_window,
    )
    reporter = V2BCTrainingReporter(output)
    try:
        result = train_v2_bc_actor(
            actor,
            cluster,
            split,
            run_config,
            epoch_callback=reporter.record_epoch,
        )
        finished_at = datetime.now(timezone.utc).isoformat()
        payload_builder = (
            build_v2_bc_checkpoint_payload
            if model == 'ann'
            else build_v2_snn_bc_checkpoint_payload
        )
        best_payload = payload_builder(
            checkpoint_kind='best',
            actor=actor,
            actor_state_dict=result.best_state_dict,
            cluster=cluster,
            split=split,
            config=run_config,
            result=result,
            finished_at=finished_at,
        )
        final_payload = payload_builder(
            checkpoint_kind='final',
            actor=actor,
            actor_state_dict=result.final_state_dict,
            cluster=cluster,
            split=split,
            config=run_config,
            result=result,
            finished_at=finished_at,
        )
        best_path = output / f'bc_v2_{model}_best.pt'
        final_path = output / f'bc_v2_{model}_final.pt'
        metrics_path = output / 'metrics.json'
        split_path = output / 'split.json'
        _save_checkpoint(best_path, best_payload)
        _save_checkpoint(final_path, final_payload)
        split_payload = split.to_dict()
        _write_strict_json(split_path, split_payload)
        metrics = {
            'format': (
                'v2_bc_training_metrics'
                if model == 'ann'
                else 'v2_snn_bc_training_metrics'
            ),
            'format_version': 1,
            'model': f'v2_{model}',
            'seed': run_config.seed,
            'epochs': run_config.epochs,
            'batch_size': run_config.batch_size,
            'learning_rate': run_config.learning_rate,
            'validation_fraction': run_config.validation_fraction,
            'requested_device': requested_device,
            'resolved_device': resolved_device,
            'shard_cache_mb': float(shard_cache_mb),
            'deduplicate_identical_trajectories': (
                deduplicate_identical_trajectories
            ),
            'dataset_provenance': {
                'trajectory_cluster': str(cluster.root),
                'trajectory_count_before_deduplication': (
                    cluster.source_trajectory_count
                ),
                'trajectory_count_after_deduplication': cluster.trajectory_count,
                'step_count_before_deduplication': cluster.source_step_count,
                'step_count_after_deduplication': cluster.step_count,
                'duplicate_trajectory_mappings': list(
                    cluster.duplicate_trajectory_mappings
                ),
            },
            'train_loss_history': list(result.train_loss_history),
            'validation_loss_history': list(result.validation_loss_history),
            'best_epoch': result.best_epoch,
            'best_validation_loss': result.best_validation_loss,
            'best_checkpoint': best_path.name,
            'final_checkpoint': final_path.name,
            'split_file': split_path.name,
            'train_statistics': split_payload['train_statistics'],
            'validation_statistics': split_payload['validation_statistics'],
            'finished_at': finished_at,
        }
        if model == 'snn':
            metrics['snn'] = {
                'time_window': actor.time_window,
                'tau': actor.tau,
                'surrogate': actor.surrogate_name,
                'backend': actor.backend,
            }
        _write_strict_json(metrics_path, metrics)
        reporter.finish()
        return metrics
    finally:
        reporter.close()


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description='Train an independent ANN or SNN actor from a V2 trajectory cluster.'
    )
    parser.add_argument('--trajectory-cluster', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--model', choices=('ann', 'snn'), default='ann')
    parser.add_argument('--snn-time-window', type=int, default=4)
    parser.add_argument('--shard-cache-mb', type=float, default=256.0)
    parser.add_argument(
        '--deduplicate-identical-trajectories',
        action='store_true',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metrics = train_v2_behavior_cloning(
        trajectory_cluster=args.trajectory_cluster,
        output_dir=args.output_dir,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        model=args.model,
        snn_time_window=args.snn_time_window,
        shard_cache_mb=args.shard_cache_mb,
        deduplicate_identical_trajectories=(
            args.deduplicate_identical_trajectories
        ),
    )
    print(json.dumps({
        'output_dir': str(args.output_dir),
        'best_epoch': metrics['best_epoch'],
        'best_validation_loss': metrics['best_validation_loss'],
        'requested_device': metrics['requested_device'],
        'resolved_device': metrics['resolved_device'],
    }, allow_nan=False, ensure_ascii=False))


if __name__ == '__main__':
    main()


__all__ = [
    'build_parser',
    'build_v2_bc_actor',
    'main',
    'train_v2_behavior_cloning',
]
