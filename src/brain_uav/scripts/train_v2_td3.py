"""Command-line entry point for one formal V2 ANN/SNN TD3 stage."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.models import V2SNNPolicyActor, require_v2_spikingjelly
from brain_uav.observations import V2ObservationBatch, collate_v2_observations
from brain_uav.scripts.common import DEVICE_CHOICES, resolve_training_device
from brain_uav.trainers.v2_formal_training import (
    V2FormalStageTrainer,
    V2FormalTrainingConfig,
    V2PreparedStageInitialization,
    build_v2_formal_checkpoint,
    build_v2_periodic_snapshot,
    build_v2_stage_engine,
    prepare_v2_stage_initialization,
    save_v2_formal_checkpoint,
    save_v2_periodic_snapshot,
    validate_v2_prepared_stage_initialization,
)
from brain_uav.trainers.v2_validation import (
    V2_VALIDATION_POOL_VERSION,
    evaluate_v2_fixed_validation,
    load_v2_validation_pool,
    scenario_config_snapshot,
)
from brain_uav.trainers.v2_reporting import V2ExperimentReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train one formal structured-observation V2 ANN/SNN TD3 stage.'
    )
    parser.add_argument('--stage', choices=('easy', 'medium', 'hard'), required=True)
    parser.add_argument('--init-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metrics-out', type=Path, required=True)
    parser.add_argument('--validation-pool', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--model', choices=('ann', 'snn'), default='ann')
    parser.add_argument('--snn-time-window', type=int, default=4)
    parser.add_argument('--max-stage-steps', type=int, default=None)
    parser.add_argument('--early-stop-min-steps', type=int, default=125_000)
    parser.add_argument('--window-episodes', type=int, default=15)
    parser.add_argument('--consecutive-windows', type=int, default=4)
    parser.add_argument('--max-failures-per-window', type=int, default=1)
    parser.add_argument('--validation-max-failures', type=int, default=6)
    # These default to the 2026-09-22-verified full compile + CUDA Graph
    # combination (see default_v2_cuda_graph_compilation below) unless
    # explicitly overridden with --no-<flag>. compile_critic_encoder,
    # compile_target_encoders, pinned_batch_transfer, and
    # aggregate_relation_values_first are not part of that verified
    # combination and keep their plain False default.
    parser.add_argument('--compile-critic-encoder', action='store_true')
    parser.add_argument('--compile-target-encoders', action='store_true')
    parser.add_argument(
        '--compile-shared-relations', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        '--compile-snn-target-encoder', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument('--fused-adam', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        '--compile-actor-loss', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        '--cache-actor-loss-coefficients', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        '--compile-action-inference', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        '--cuda-graph-action-inference', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument('--pinned-batch-transfer', action='store_true')
    parser.add_argument('--aggregate-relation-values-first', action='store_true')
    parser.add_argument(
        '--reduce-update-stat-syncs', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        '--cuda-graph-updates', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        '--cuda-graph-actor-update', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument('--compile-actors', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        '--frozen-critic-strategy',
        choices=('eager', 'compiled_no_grad_context'),
        default=None,
    )
    parser.add_argument(
        '--compile-critic-block', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        '--compile-target-block', action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        '--periodic-snapshot-interval-steps',
        type=int,
        default=50_000,
        help=(
            'Purely observational: save a mid-stage checkpoint and run a '
            'non-gating fixed validation every N steps. 0 disables both.'
        ),
    )
    return parser


def default_v2_cuda_graph_compilation(
    *, model: str, resolved_device: str,
) -> dict[str, bool | str]:
    """The 2026-09-22-verified full compile + CUDA Graph combination (D1).

    Source: the diagnostic_actor_graph_{ann,snn}_actor_20260922_155254 runs
    under train_result/, both of which passed
    ``profile_v2_td3.py --check-compiled-numerics`` with exactly this
    combination. ``cuda_graph_*`` flags require CUDA and are forced off on
    CPU so ``--device cpu``/``auto`` without a GPU still works; every other
    flag here is device-independent. ``compile_snn_target_encoder`` is only
    valid for an SNN actor, so it tracks ``model``.
    """

    if model not in ('ann', 'snn'):
        raise ValueError('model must be "ann" or "snn".')
    if resolved_device not in ('cpu', 'cuda'):
        raise ValueError('resolved_device must be "cpu" or "cuda".')
    is_cuda = resolved_device == 'cuda'
    return {
        'compile_actors': True,
        'frozen_critic_strategy': 'compiled_no_grad_context',
        'compile_critic_block': True,
        'compile_target_block': True,
        'compile_shared_relations': True,
        'compile_snn_target_encoder': model == 'snn',
        'fused_adam': True,
        'compile_actor_loss': True,
        'cache_actor_loss_coefficients': True,
        'compile_action_inference': True,
        'cuda_graph_action_inference': is_cuda,
        'reduce_update_stat_syncs': True,
        'cuda_graph_updates': is_cuda,
        'cuda_graph_actor_update': is_cuda,
    }


def _resolve_v2_cuda_graph_compilation(
    args: argparse.Namespace, *, resolved_device: str,
) -> dict[str, bool | str]:
    """Apply CLI overrides (non-None) on top of the verified defaults."""

    defaults = default_v2_cuda_graph_compilation(
        model=args.model, resolved_device=resolved_device,
    )
    resolved = dict(defaults)
    for name in defaults:
        value = getattr(args, name)
        if value is not None:
            resolved[name] = value
    return resolved


def _failed_checkpoint_path(path: Path) -> Path:
    suffix = path.suffix or '.pt'
    stem = path.stem if path.suffix else path.name
    return path.with_name(f'{stem}_failed{suffix}')


def _write_strict_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f'Output already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )


def _configure_stage_compilation(
    engine,
    pool,
    prepared: V2PreparedStageInitialization,
    *,
    compile_critic_encoder: bool,
    compile_target_encoders: bool,
    compile_actors: bool,
    frozen_critic_strategy: str,
    compile_critic_block: bool,
    compile_target_block: bool,
    compile_shared_relations: bool = False,
    compile_snn_target_encoder: bool = False,
    compile_actor_loss: bool = False,
    cache_actor_loss_coefficients: bool = False,
    compile_action_inference: bool = False,
    cuda_graph_action_inference: bool = False,
    cuda_graph_actor_update: bool = False,
    cuda_graph_updates: bool = False,
) -> dict[str, Any]:
    if cuda_graph_actor_update and not (
        cuda_graph_updates and compile_actors
        and frozen_critic_strategy == 'compiled_no_grad_context'
    ):
        raise ValueError(
            'cuda_graph_actor_update requires cuda_graph_updates, '
            'compile_actors and compiled_no_grad_context.'
        )
    if cache_actor_loss_coefficients and not compile_actor_loss:
        raise ValueError('cache_actor_loss_coefficients requires compile_actor_loss.')
    requested = any((
        compile_critic_encoder, compile_target_encoders, compile_actors,
        compile_critic_block, compile_target_block,
        compile_shared_relations, compile_snn_target_encoder,
        compile_actor_loss,
        compile_action_inference,
        cuda_graph_action_inference,
        cuda_graph_updates,
    ))
    if not requested:
        if frozen_critic_strategy != 'eager':
            raise ValueError(
                'compiled_no_grad_context requires a compiled critic path.'
            )
        return {
            'requested': False,
            'enabled_objects': [],
            'frozen_critic_strategy': 'eager',
            'select_action_execution': 'eager',
            'cuda_graph': False,
            'cuda_graph_evidence': None,
            'cuda_graph_action_inference': False,
            'cuda_graph_action_inference_evidence': None,
            'optimizer_execution': (
                'fused_adam' if getattr(engine, 'fused_adam', False) else 'adam'
            ),
            'relation_value_execution': (
                'aggregate_then_project'
                if getattr(engine, 'aggregate_relation_values_first', False)
                else 'project_then_aggregate'
            ),
            'actor_loss_granularity': 'eager',
            'actor_loss_coefficients_requested': False,
            'actor_loss_coefficient_execution': 'per_update',
            'action_inference_granularity': 'eager',
            'update_statistics_execution': (
                'batched_device_readback'
                if getattr(engine, 'reduce_update_stat_syncs', False)
                else 'per_scalar'
            ),
            'reduce_update_stat_syncs_requested': bool(
                getattr(engine, 'reduce_update_stat_syncs', False)
            ),
            'batch_transfer_execution': (
                'reusable_pinned_non_blocking'
                if getattr(engine, 'pinned_batch_transfer', False)
                else 'blocking_to_device'
            ),
            'pinned_batch_transfer_requested': bool(
                getattr(engine, 'pinned_batch_transfer', False)
            ),
            'action_inference_warmup_shapes': [],
            'registration_wall_seconds': 0.0,
            'warmup_wall_seconds': 0.0,
            'warmup_batch_shapes': [],
        }
    registration_started = perf_counter()
    metadata = engine.configure_compilation(
        compile_critic_encoder=compile_critic_encoder,
        compile_target_encoders=compile_target_encoders,
        compile_actors=compile_actors,
        frozen_critic_strategy=frozen_critic_strategy,
        compile_critic_block=compile_critic_block,
        compile_target_block=compile_target_block,
        compile_shared_relations=compile_shared_relations,
        compile_snn_target_encoder=compile_snn_target_encoder,
        compile_actor_loss=compile_actor_loss,
        cache_actor_loss_coefficients=cache_actor_loss_coefficients,
        compile_action_inference=compile_action_inference,
        cuda_graph_action_inference=cuda_graph_action_inference,
        cuda_graph_updates=cuda_graph_updates,
        cuda_graph_actor_update=cuda_graph_actor_update,
        backend='inductor', mode='default', fullgraph=True, dynamic=True,
    )
    metadata['requested'] = True
    metadata['reduce_update_stat_syncs_requested'] = bool(
        getattr(engine, 'reduce_update_stat_syncs', False)
    )
    metadata['pinned_batch_transfer_requested'] = bool(
        getattr(engine, 'pinned_batch_transfer', False)
    )
    metadata['actor_loss_coefficients_requested'] = cache_actor_loss_coefficients
    metadata['registration_wall_seconds'] = perf_counter() - registration_started

    records_by_zone_count = {}
    for record in pool.scenarios:
        records_by_zone_count.setdefault(len(record['payload']['zones']), record)
    records = tuple(records_by_zone_count.values())
    warmup_env = V2StaticNoFlyTrajectoryEnv(
        prepared.scenario_config,
        prepared.reward_config,
        seed=pool.stage_seed,
        fixed_scenarios=[record['payload'] for record in records],
        uav_collision_radius=prepared.uav_collision_radius,
    )
    observations = tuple(
        warmup_env.reset(options={'scenario': record['payload']})[0]
        for record in records
    )
    batches = [
        collate_v2_observations([observation] * engine.batch_size).to(engine.device)
        for observation in observations
    ]
    if len(observations) > 1:
        batches.append(collate_v2_observations([
            observations[index % len(observations)]
            for index in range(engine.batch_size)
        ]).to(engine.device))
    warmup_batches = tuple(batches)
    action_inference_batches = tuple(
        collate_v2_observations([observation]).to(engine.device)
        for observation in observations
    )
    if action_inference_batches and all(
        batch.max_zone_count != 0 for batch in action_inference_batches
    ):
        first = action_inference_batches[0]
        action_inference_batches = (V2ObservationBatch(
            ego_features=first.ego_features,
            goal_features=first.goal_features,
            zone_features=first.zone_features[:, :0, :],
            presence_mask=first.presence_mask[:, :0],
        ), *action_inference_batches)
    metadata['warmup_batch_shapes'] = [
        [batch.batch_size, int(batch.zone_features.shape[1])]
        for batch in warmup_batches
    ]
    warmup_started = perf_counter()
    if compile_actors:
        engine.warmup_actor_compile(warmup_batches)
    if compile_shared_relations:
        engine.warmup_shared_relations_compile(warmup_batches)
    if compile_critic_block:
        engine.warmup_full_compile(warmup_batches)
    elif compile_critic_encoder:
        engine.warmup_online_critic_encoder_compile(warmup_batches)
        if compile_target_encoders:
            engine.warmup_target_encoder_compile(warmup_batches)
    if compile_snn_target_encoder and not compile_target_block:
        engine.warmup_snn_target_encoder_compile(warmup_batches)
    if compile_actor_loss:
        engine.warmup_actor_loss_compile(warmup_batches)
    if compile_action_inference:
        engine.warmup_action_inference_compile(action_inference_batches)
        metadata['action_inference_warmup_shapes'] = [
            [batch.batch_size, int(batch.zone_features.shape[1])]
            for batch in action_inference_batches
        ]
    if cuda_graph_action_inference:
        metadata['cuda_graph_action_inference_evidence'] = (
            engine.verify_action_inference_cuda_graph_capture(
                action_inference_batches
            )
        )
    if cuda_graph_updates:
        metadata['cuda_graph_evidence'] = (
            engine.verify_update_cuda_graph_capture(warmup_batches)
        )
    metadata['warmup_wall_seconds'] = perf_counter() - warmup_started
    return metadata


def run_v2_td3_stage(
    *,
    stage: str,
    init_checkpoint: str | Path,
    output: str | Path,
    metrics_out: str | Path,
    validation_pool: str | Path,
    seed: int = 7,
    device: str = 'auto',
    max_stage_steps: int | None = None,
    early_stop_min_steps: int = 125_000,
    window_episodes: int = 15,
    consecutive_windows: int = 4,
    max_failures_per_window: int = 1,
    validation_max_failures: int = 6,
    scenario: ScenarioConfig | None = None,
    rewards: RewardConfig | None = None,
    uav_collision_radius: float | None = None,
    global_steps_start: int = 0,
    validation_scenario_count: int = 100,
    expected_validation_master_seed: int | None = None,
    expected_validation_stage_seed: int | None = None,
    model: str = 'ann',
    snn_time_window: int = 4,
    prepared_initialization: V2PreparedStageInitialization | None = None,
    reporting: bool = True,
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
    cache_actor_loss_coefficients: bool = False,
    compile_action_inference: bool = False,
    cuda_graph_action_inference: bool = False,
    aggregate_relation_values_first: bool = False,
    reduce_update_stat_syncs: bool = False,
    pinned_batch_transfer: bool = False,
    cuda_graph_actor_update: bool = False,
    cuda_graph_updates: bool = False,
    periodic_snapshot_interval_steps: int | None = None,
) -> dict[str, Any]:
    requested_device = device
    resolved_device = resolve_training_device(requested_device)
    if model not in ('ann', 'snn'):
        raise ValueError('model must be "ann" or "snn".')
    if type(snn_time_window) is not int or snn_time_window <= 0:
        raise ValueError('snn_time_window must be a positive integer.')
    if type(reporting) is not bool:
        raise ValueError('reporting must be bool.')
    if model == 'snn':
        require_v2_spikingjelly()
    config = V2FormalTrainingConfig(
        stage=stage,
        seed=seed,
        max_steps=max_stage_steps,
        early_stop_min_steps=early_stop_min_steps,
        window_episode_count=window_episodes,
        consecutive_qualified_windows=consecutive_windows,
        max_failures_per_window=max_failures_per_window,
        validation_max_failures=validation_max_failures,
    )
    if prepared_initialization is None:
        prepared_initialization = prepare_v2_stage_initialization(
            config,
            init_checkpoint=init_checkpoint,
            scenario=scenario,
            rewards=rewards,
            uav_collision_radius=uav_collision_radius,
            device=resolved_device,
            model_type=model,
            snn_time_window=snn_time_window,
        )
    validate_v2_prepared_stage_initialization(
        prepared_initialization,
        config,
        init_checkpoint=init_checkpoint,
        scenario=scenario,
        rewards=rewards,
        uav_collision_radius=uav_collision_radius,
        model_type=model,
        snn_time_window=snn_time_window,
    )
    scenario_config = prepared_initialization.scenario_config
    reward_config = prepared_initialization.reward_config
    effective_uav_collision_radius = (
        prepared_initialization.uav_collision_radius
    )
    if periodic_snapshot_interval_steps is not None:
        if type(periodic_snapshot_interval_steps) is not int or periodic_snapshot_interval_steps <= 0:
            raise ValueError('periodic_snapshot_interval_steps must be a positive integer.')
    output_path = Path(output)
    metrics_path = Path(metrics_out)
    failed_output_path = _failed_checkpoint_path(output_path)
    report_path = metrics_path.with_name(f'{metrics_path.stem}_reports')
    periodic_snapshot_dir = output_path.with_name(f'{output_path.stem}_periodic')
    if (output_path.exists() or failed_output_path.exists() or metrics_path.exists()
            or (reporting and report_path.exists())
            or (periodic_snapshot_interval_steps is not None and periodic_snapshot_dir.exists())):
        raise FileExistsError('Formal V2 stage outputs already exist.')
    pool_path = Path(validation_pool)
    pool = load_v2_validation_pool(
        pool_path,
        expected_level=stage,
        expected_scenario=scenario_config,
        expected_count=validation_scenario_count,
        expected_uav_collision_radius=effective_uav_collision_radius,
        expected_master_seed=expected_validation_master_seed,
        expected_stage_seed=expected_validation_stage_seed,
    )
    if config.validation_max_failures >= pool.scenario_count:
        raise ValueError(
            'validation_max_failures must be below the validation scenario count.'
        )
    components = build_v2_stage_engine(
        scenario_config,
        config,
        init_checkpoint=init_checkpoint,
        rewards=reward_config,
        uav_collision_radius=effective_uav_collision_radius,
        device=resolved_device,
        model_type=model,
        snn_time_window=snn_time_window,
        prepared_initialization=prepared_initialization,
        fused_adam=fused_adam,
        aggregate_relation_values_first=aggregate_relation_values_first,
        reduce_update_stat_syncs=reduce_update_stat_syncs,
        pinned_batch_transfer=pinned_batch_transfer,
    )
    compilation_metadata = _configure_stage_compilation(
        components.engine,
        pool,
        prepared_initialization,
        compile_critic_encoder=compile_critic_encoder,
        compile_target_encoders=compile_target_encoders,
        compile_actors=compile_actors,
        frozen_critic_strategy=frozen_critic_strategy,
        compile_critic_block=compile_critic_block,
        compile_target_block=compile_target_block,
        compile_shared_relations=compile_shared_relations,
        compile_snn_target_encoder=compile_snn_target_encoder,
        compile_actor_loss=compile_actor_loss,
        cache_actor_loss_coefficients=cache_actor_loss_coefficients,
        compile_action_inference=compile_action_inference,
        cuda_graph_action_inference=cuda_graph_action_inference,
        cuda_graph_updates=cuda_graph_updates,
        cuda_graph_actor_update=cuda_graph_actor_update,
    )
    reporter = (
        V2ExperimentReporter(
            report_path,
            stage=stage,
            model_type=model,
            scenario=scenario_config,
            rewards=reward_config,
            uav_collision_radius=effective_uav_collision_radius,
            max_steps=config.max_steps,
            required_qualified_windows=config.consecutive_qualified_windows,
        )
        if reporting
        else None
    )
    if reporter is not None:
        actor = components.engine.actor
        snn_metadata = (
            {
                'time_window': actor.time_window,
                'tau': actor.tau,
                'surrogate': actor.surrogate_name,
                'backend': actor.backend,
            }
            if isinstance(actor, V2SNNPolicyActor)
            else None
        )
        reporter.start_stage({
            'requested_device': requested_device,
            'resolved_device': resolved_device,
            'initialization_source': dict(components.initialization_source),
            'seed': seed,
            'max_steps': config.max_steps,
            'formal_config': config.to_dict(),
            'checkpoint_output': str(output_path),
            'failed_checkpoint_output': str(failed_output_path),
            'metrics_output': str(metrics_path),
            'snn': snn_metadata,
            'compilation': compilation_metadata,
        })

    def validate(actor):
        return evaluate_v2_fixed_validation(
            actor,
            pool,
            reward_config,
            max_failures=config.validation_max_failures,
            device=resolved_device,
            reporter=reporter,
        )

    periodic_snapshot_sink = None
    periodic_validation_sink = None
    if periodic_snapshot_interval_steps is not None:
        # C1: purely observational mid-stage checkpoint + fixed-validation
        # snapshots, so a manually interrupted run keeps usable weights and
        # a periodic validation reading (P5早停判据离线回放结论_20260917.md
        # section 6.1). Deliberately separate from `output_path`/`validate`
        # above: this never calls record_validation and is never accepted
        # as a stage predecessor by load_v2_formal_checkpoint.
        def periodic_snapshot_sink(stage_steps: int) -> None:
            payload = build_v2_periodic_snapshot(
                components.engine,
                config,
                stage_steps=stage_steps,
                scenario=scenario_config,
                rewards=reward_config,
                uav_collision_radius=effective_uav_collision_radius,
                seed_manifest=components.seed_manifest,
                initialization_source=components.initialization_source,
            )
            snapshot_path = periodic_snapshot_dir / f'step_{stage_steps:09d}.pt'
            save_v2_periodic_snapshot(snapshot_path, payload)
            print(
                f"[V2 {model.upper()} {stage}] periodic snapshot "
                f"stage_steps={stage_steps} path={snapshot_path}",
                flush=True,
            )

        def periodic_validation_sink(stage_steps: int) -> None:
            result = evaluate_v2_fixed_validation(
                components.engine.actor,
                pool,
                reward_config,
                max_failures=config.validation_max_failures,
                device=resolved_device,
                reporter=None,
            )
            if reporter is not None:
                record = dict(result.to_dict())
                record['stage_steps'] = stage_steps
                record['global_steps'] = global_steps_start + stage_steps
                reporter.record_periodic_validation(record)

    trainer = V2FormalStageTrainer(
        scenario_config,
        reward_config,
        config,
        components.engine,
        scenario_sources=components.scenario_generators,
        validation_runner=validate,
        selector=components.selector,
        exploration_rng=components.exploration_rng,
        global_steps_start=global_steps_start,
        uav_collision_radius=effective_uav_collision_radius,
        reporter=reporter,
        periodic_snapshot_interval_steps=periodic_snapshot_interval_steps,
        periodic_snapshot_sink=periodic_snapshot_sink,
        periodic_validation_sink=periodic_validation_sink,
    )
    try:
        result = trainer.run()
        validation_metadata = {
            'path': str(pool_path),
            'format_version': V2_VALIDATION_POOL_VERSION,
            'curriculum_level': pool.curriculum_level,
            'master_seed': pool.master_seed,
            'stage_seed': pool.stage_seed,
            'scenario_count': pool.scenario_count,
            'content_digest': pool.content_digest,
        }
        checkpoint_payload = build_v2_formal_checkpoint(
            components.engine,
            result,
            config,
            scenario=scenario_config,
            rewards=reward_config,
            uav_collision_radius=effective_uav_collision_radius,
            seed_manifest=components.seed_manifest,
            validation_pool_metadata=validation_metadata,
            initialization_source=components.initialization_source,
        )
        checkpoint_path = output_path if result.passed_validation else failed_output_path
        if checkpoint_path.exists():
            raise FileExistsError(f'Formal V2 checkpoint already exists: {checkpoint_path}')
        save_v2_formal_checkpoint(checkpoint_path, checkpoint_payload)
        actor = components.engine.actor
        metrics_payload = {
            'format': f'v2_formal_{model}_td3_metrics',
            'format_version': 1,
            'model_type': model,
            'actor_trainable_parameter_count': sum(
                parameter.numel()
                for parameter in actor.parameters()
                if parameter.requires_grad
            ),
            'critic_trainable_parameter_count': sum(
                parameter.numel()
                for critic in (components.engine.critic1, components.engine.critic2)
                for parameter in critic.parameters()
                if parameter.requires_grad
            ),
            'scenario_config': scenario_config_snapshot(scenario_config),
            'reward_config': asdict(reward_config),
            'formal_config': config.to_dict(),
            'seed_manifest': dict(components.seed_manifest),
            'validation_pool': validation_metadata,
            'initialization_source': dict(components.initialization_source),
            'requested_device': requested_device,
            'resolved_device': resolved_device,
            'replay_sampling_implementation': (
                components.engine.replay.sampling_implementation
            ),
            'result': result.to_dict(),
            'checkpoint': str(checkpoint_path),
            'report_directory': str(report_path) if reporter is not None else None,
            'compilation': compilation_metadata,
        }
        if isinstance(actor, V2SNNPolicyActor):
            metrics_payload['snn'] = {
                'time_window': actor.time_window,
                'tau': actor.tau,
                'surrogate': actor.surrogate_name,
                'backend': actor.backend,
            }
        _write_strict_json(metrics_path, metrics_payload)
        if reporter is not None:
            reporter.finish_stage(result.to_dict())
        summary = {
            'stage': stage,
            'model_type': model,
            'passed': result.passed_validation,
            'checkpoint': str(checkpoint_path),
            'metrics': str(metrics_path),
            'steps': result.stage_steps,
            'global_steps_end': result.global_steps_end,
            'outcome_counts': dict(result.outcome_counts),
            'validation_result': result.validation_records[-1] if result.validation_records else None,
            'stop_reason': result.stop_reason,
            'requested_device': requested_device,
            'resolved_device': resolved_device,
            'replay_sampling_implementation': (
                components.engine.replay.sampling_implementation
            ),
            'report_directory': str(report_path) if reporter is not None else None,
            'compilation': compilation_metadata,
        }
        return summary
    finally:
        if reporter is not None:
            reporter.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_device = resolve_training_device(args.device)
    config_preview = V2FormalTrainingConfig(
        stage=args.stage,
        seed=args.seed,
        max_steps=args.max_stage_steps,
        early_stop_min_steps=args.early_stop_min_steps,
        window_episode_count=args.window_episodes,
        consecutive_qualified_windows=args.consecutive_windows,
        max_failures_per_window=args.max_failures_per_window,
        validation_max_failures=args.validation_max_failures,
    )
    print(json.dumps({
        'requested_device': args.device,
        'resolved_device': resolved_device,
        'resolved_v2_formal_config': config_preview.to_dict(),
        'model_type': args.model,
        'snn_time_window': args.snn_time_window if args.model == 'snn' else None,
    }, indent=2))
    compilation = _resolve_v2_cuda_graph_compilation(args, resolved_device=resolved_device)
    print(json.dumps({'resolved_compilation_defaults': compilation}, indent=2))
    summary = run_v2_td3_stage(
        stage=args.stage,
        init_checkpoint=args.init_checkpoint,
        output=args.output,
        metrics_out=args.metrics_out,
        validation_pool=args.validation_pool,
        seed=args.seed,
        device=args.device,
        max_stage_steps=args.max_stage_steps,
        early_stop_min_steps=args.early_stop_min_steps,
        window_episodes=args.window_episodes,
        consecutive_windows=args.consecutive_windows,
        max_failures_per_window=args.max_failures_per_window,
        validation_max_failures=args.validation_max_failures,
        model=args.model,
        snn_time_window=args.snn_time_window,
        compile_critic_encoder=args.compile_critic_encoder,
        compile_target_encoders=args.compile_target_encoders,
        compile_actors=compilation['compile_actors'],
        frozen_critic_strategy=compilation['frozen_critic_strategy'],
        compile_critic_block=compilation['compile_critic_block'],
        compile_target_block=compilation['compile_target_block'],
        compile_shared_relations=compilation['compile_shared_relations'],
        compile_snn_target_encoder=compilation['compile_snn_target_encoder'],
        fused_adam=compilation['fused_adam'],
        compile_actor_loss=compilation['compile_actor_loss'],
        cache_actor_loss_coefficients=compilation['cache_actor_loss_coefficients'],
        compile_action_inference=compilation['compile_action_inference'],
        cuda_graph_action_inference=compilation['cuda_graph_action_inference'],
        aggregate_relation_values_first=args.aggregate_relation_values_first,
        reduce_update_stat_syncs=compilation['reduce_update_stat_syncs'],
        pinned_batch_transfer=args.pinned_batch_transfer,
        cuda_graph_updates=compilation['cuda_graph_updates'],
        cuda_graph_actor_update=compilation['cuda_graph_actor_update'],
        periodic_snapshot_interval_steps=(
            None if args.periodic_snapshot_interval_steps == 0
            else args.periodic_snapshot_interval_steps
        ),
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
