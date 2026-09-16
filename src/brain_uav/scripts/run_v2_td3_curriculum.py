"""One predeclared-seed formal V2 ANN/SNN easy-to-hard curriculum chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from brain_uav.config import ScenarioConfig
from brain_uav.models import V2SNNPolicyActor, require_v2_spikingjelly
from brain_uav.scripts.common import DEVICE_CHOICES, resolve_training_device
from brain_uav.scripts.train_v2_td3 import run_v2_td3_stage
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig,
    prepare_v2_stage_initialization,
)
from brain_uav.trainers.v2_validation import (
    derive_validation_stage_seed,
    generate_v2_validation_pool,
    load_v2_validation_pool,
    save_v2_validation_pool,
)
from brain_uav.v2_curriculum import v2_stage_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run one fixed-seed formal V2 ANN/SNN TD3 curriculum chain.'
    )
    parser.add_argument('--bc-checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--validation-pool-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--validation-seed', type=int, default=20260904)
    parser.add_argument('--max-stage', choices=('easy', 'medium', 'hard'), default='hard')
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--model', choices=('ann', 'snn'), default='ann')
    parser.add_argument('--snn-time-window', type=int, default=4)
    parser.add_argument('--compile-critic-encoder', action='store_true')
    parser.add_argument('--compile-target-encoders', action='store_true')
    parser.add_argument('--compile-shared-relations', action='store_true')
    parser.add_argument('--compile-snn-target-encoder', action='store_true')
    parser.add_argument('--fused-adam', action='store_true')
    parser.add_argument('--compile-actor-loss', action='store_true')
    parser.add_argument('--compile-action-inference', action='store_true')
    parser.add_argument('--aggregate-relation-values-first', action='store_true')
    parser.add_argument('--compile-actors', action='store_true')
    parser.add_argument(
        '--frozen-critic-strategy',
        choices=('eager', 'compiled_no_grad_context'),
        default='eager',
    )
    parser.add_argument('--compile-critic-block', action='store_true')
    parser.add_argument('--compile-target-block', action='store_true')
    return parser


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f'Curriculum summary already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )


def prepare_v2_validation_pools(
    directory: str | Path,
    scenario: ScenarioConfig,
    *,
    validation_seed: int = 20260904,
    scenario_count: int = 100,
    uav_collision_radius: float = 0.0,
) -> dict[str, Path]:
    """Generate missing pools or strictly load existing pools before training."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    # Prepare all three before any formal stage is started.
    for stage in ('easy', 'medium', 'hard'):
        path = root / f'v2_validation_{stage}.json'
        expected_stage_seed = derive_validation_stage_seed(validation_seed, stage)
        if path.exists():
            load_v2_validation_pool(
                path,
                expected_level=stage,
                expected_scenario=scenario,
                expected_count=scenario_count,
                expected_uav_collision_radius=uav_collision_radius,
                expected_master_seed=validation_seed,
                expected_stage_seed=expected_stage_seed,
            )
        else:
            pool = generate_v2_validation_pool(
                scenario,
                stage,
                scenario_count=scenario_count,
                master_seed=validation_seed,
                uav_collision_radius=uav_collision_radius,
            )
            save_v2_validation_pool(path, pool)
        paths[stage] = path
    return paths


def run_v2_curriculum(
    *,
    bc_checkpoint: str | Path,
    output_root: str | Path,
    validation_pool_dir: str | Path,
    seed: int = 7,
    validation_seed: int = 20260904,
    max_stage: str = 'hard',
    device: str = 'auto',
    stage_runner: Callable[..., dict[str, Any]] = run_v2_td3_stage,
    model: str = 'ann',
    snn_time_window: int = 4,
    compile_critic_encoder: bool = False,
    compile_target_encoders: bool = False,
    compile_actors: bool = False,
    frozen_critic_strategy: str = 'eager',
    compile_critic_block: bool = False,
    compile_target_block: bool = False,
    compile_shared_relations: bool = False,
    compile_snn_target_encoder: bool = False,
    fused_adam: bool = False,
    compile_actor_loss: bool = False,
    compile_action_inference: bool = False,
    aggregate_relation_values_first: bool = False,
) -> dict[str, Any]:
    requested_device = device
    resolved_device = resolve_training_device(requested_device)
    if model not in ('ann', 'snn'):
        raise ValueError('model must be "ann" or "snn".')
    if type(snn_time_window) is not int or snn_time_window <= 0:
        raise ValueError('snn_time_window must be a positive integer.')
    if model == 'snn':
        require_v2_spikingjelly()
    stages = v2_stage_sequence(max_stage)
    bc_path = Path(bc_checkpoint)
    if not bc_path.is_file():
        raise FileNotFoundError(f'V2 BC checkpoint does not exist: {bc_path}')
    prepared_initialization = prepare_v2_stage_initialization(
        V2FormalTrainingConfig(stage='easy', seed=seed),
        init_checkpoint=bc_path,
        device='cpu',
        model_type=model,
        snn_time_window=snn_time_window,
    )
    bc_initialization = prepared_initialization.bc_initialization
    if bc_initialization is None:
        raise RuntimeError('Prepared easy initialization is missing the BC actor.')
    if model == 'snn':
        actor = bc_initialization.actor
        if type(actor) is not V2SNNPolicyActor:
            raise TypeError(
                'V2 SNN curriculum BC initialization did not return '
                'a V2SNNPolicyActor.'
            )
        if actor.time_window != snn_time_window:
            raise ValueError(
                'V2 SNN BC checkpoint time_window '
                f'{actor.time_window} does not match requested '
                f'snn_time_window {snn_time_window}.'
            )
        snn_metadata: dict[str, Any] | None = {
            'time_window': actor.time_window,
            'tau': actor.tau,
            'surrogate': actor.surrogate_name,
            'backend': actor.backend,
        }
        effective_snn_time_window = actor.time_window
    else:
        snn_metadata = None
        effective_snn_time_window = snn_time_window
    scenario = prepared_initialization.scenario_config
    uav_collision_radius = prepared_initialization.uav_collision_radius
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Output root is not empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    pools = prepare_v2_validation_pools(
        validation_pool_dir,
        scenario,
        validation_seed=validation_seed,
        scenario_count=100,
        uav_collision_radius=uav_collision_radius,
    )
    validated_pools = {
        stage: load_v2_validation_pool(
            path,
            expected_level=stage,
            expected_scenario=scenario,
            expected_count=100,
            expected_uav_collision_radius=uav_collision_radius,
            expected_master_seed=validation_seed,
            expected_stage_seed=derive_validation_stage_seed(validation_seed, stage),
        )
        for stage, path in pools.items()
    }
    initialization = bc_path
    global_steps = 0
    summaries: list[dict[str, Any]] = []
    failed_stage: str | None = None
    for stage in stages:
        stem = f'v2_td3_{stage}' if model == 'ann' else f'v2_snn_td3_{stage}'
        output = root / f'{stem}.pt'
        metrics = root / f'{stem}_metrics.json'
        summary = stage_runner(
            stage=stage,
            init_checkpoint=initialization,
            output=output,
            metrics_out=metrics,
            validation_pool=pools[stage],
            seed=seed,
            device=resolved_device,
            scenario=scenario,
            uav_collision_radius=uav_collision_radius,
            expected_validation_master_seed=validated_pools[stage].master_seed,
            expected_validation_stage_seed=validated_pools[stage].stage_seed,
            global_steps_start=global_steps,
            model=model,
            snn_time_window=effective_snn_time_window,
            prepared_initialization=(
                prepared_initialization if stage == 'easy' else None
            ),
            compile_critic_encoder=compile_critic_encoder,
            compile_target_encoders=compile_target_encoders,
            compile_actors=compile_actors,
            frozen_critic_strategy=frozen_critic_strategy,
            compile_critic_block=compile_critic_block,
            compile_target_block=compile_target_block,
            compile_shared_relations=compile_shared_relations,
            compile_snn_target_encoder=compile_snn_target_encoder,
            fused_adam=fused_adam,
            compile_actor_loss=compile_actor_loss,
            compile_action_inference=compile_action_inference,
            aggregate_relation_values_first=aggregate_relation_values_first,
        )
        summaries.append(summary)
        global_steps = int(summary.get('global_steps_end', global_steps + int(summary['steps'])))
        if not summary['passed']:
            failed_stage = stage
            break
        initialization = Path(summary['checkpoint'])
    payload = {
        'format': f'v2_formal_{model}_td3_curriculum_summary',
        'format_version': 1,
        'model_type': model,
        'snn': snn_metadata,
        'seed': seed,
        'validation_seed': validated_pools['easy'].master_seed,
        'validation_pools': {
            stage: {
                'path': str(pools[stage]),
                'master_seed': pool.master_seed,
                'stage_seed': pool.stage_seed,
                'scenario_count': pool.scenario_count,
                'content_digest': pool.content_digest,
            }
            for stage, pool in validated_pools.items()
        },
        'requested_device': requested_device,
        'resolved_device': resolved_device,
        'max_stage': max_stage,
        'stage_order': list(stages),
        'passed': failed_stage is None and len(summaries) == len(stages),
        'failed_stage': failed_stage,
        'stages': summaries,
        'global_steps': global_steps,
        'compilation_request': {
            'compile_critic_encoder': compile_critic_encoder,
            'compile_target_encoders': compile_target_encoders,
            'compile_actors': compile_actors,
            'frozen_critic_strategy': frozen_critic_strategy,
            'compile_critic_block': compile_critic_block,
            'compile_target_block': compile_target_block,
            'compile_shared_relations': compile_shared_relations,
            'compile_snn_target_encoder': compile_snn_target_encoder,
            'fused_adam': fused_adam,
            'compile_actor_loss': compile_actor_loss,
            'compile_action_inference': compile_action_inference,
            'aggregate_relation_values_first': aggregate_relation_values_first,
            'cuda_graph': False,
        },
    }
    _write_json(root / 'summary.json', payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_device = resolve_training_device(args.device)
    print(json.dumps({
        'requested_device': args.device,
        'resolved_device': resolved_device,
        'model_type': args.model,
        'snn_time_window': args.snn_time_window if args.model == 'snn' else None,
    }, indent=2))
    summary = run_v2_curriculum(
        bc_checkpoint=args.bc_checkpoint,
        output_root=args.output_root,
        validation_pool_dir=args.validation_pool_dir,
        seed=args.seed,
        validation_seed=args.validation_seed,
        max_stage=args.max_stage,
        device=args.device,
        model=args.model,
        snn_time_window=args.snn_time_window,
        compile_critic_encoder=args.compile_critic_encoder,
        compile_target_encoders=args.compile_target_encoders,
        compile_actors=args.compile_actors,
        frozen_critic_strategy=args.frozen_critic_strategy,
        compile_critic_block=args.compile_critic_block,
        compile_target_block=args.compile_target_block,
        compile_shared_relations=args.compile_shared_relations,
        compile_snn_target_encoder=args.compile_snn_target_encoder,
        fused_adam=args.fused_adam,
        compile_actor_loss=args.compile_actor_loss,
        compile_action_inference=args.compile_action_inference,
        aggregate_relation_values_first=args.aggregate_relation_values_first,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
