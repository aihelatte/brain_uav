"""Command-line entry point for one formal V2 ANN/SNN TD3 stage."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.v2_curriculum import V2NoiseSchedule
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.models import V2SNNPolicyActor, require_v2_spikingjelly
from brain_uav.models.v2_policy_config import (
    V2SpikeRegularizationConfig, add_policy_training_arguments,
    spike_config_from_arguments, validate_policy_request, validate_policy_dimensions,
    projection_from_architecture,
)
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
    v2_bc_schedule_metadata,
    v2_initial_state_summary,
)
from brain_uav.trainers.v2_validation import (
    V2_VALIDATION_POOL_VERSION,
    evaluate_v2_fixed_validation,
    load_v2_validation_pool,
    scenario_config_snapshot,
)
from brain_uav.trainers.v2_reporting import V2ExperimentReporter


def _resolve_v2_actor_lr(stage: str, model: str, actor_lr: float | None) -> float | None:
    if actor_lr is None and stage in ('easy', 'medium'):
        return 1.55e-4 if model == 'ann' else 9.5e-4
    return actor_lr


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
    parser.add_argument('--target-noise-scale', type=float, default=1.0)
    parser.add_argument('--weight-snapshot-steps', type=int, nargs='+', default=(),
                        help='Diagnostic-only snapshots at episode boundaries; no validation.')
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--model', choices=('ann', 'snn'), default='ann')
    parser.add_argument('--snn-time-window', type=int, default=4)
    add_policy_training_arguments(parser)
    parser.add_argument('--max-stage-steps', type=int, default=None)
    parser.add_argument('--early-stop-min-steps', type=int, default=125_000)
    parser.add_argument('--window-episodes', type=int, default=15)
    parser.add_argument('--consecutive-windows', type=int, default=4)
    parser.add_argument('--max-failures-per-window', type=int, default=1)
    parser.add_argument('--validation-max-failures', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target-q-aggregation', choices=('min', 'mean'), default='min')
    parser.add_argument('--failure-sample-bias', type=float, default=1.0)
    parser.add_argument(
        '--actor-lr', type=float, default=None,
        help='Actor LR override; easy/medium defaults: ANN 1.55e-4, SNN 9.5e-4.',
    )
    parser.add_argument('--critic-lr', type=float, default=None)
    parser.add_argument(
        '--bc-schedule', choices=('default', 'smooth-final', 'smooth-mid', 'smooth-mid-strong'), default='default',
        help='Optional stage-local BC schedule; smooth variants support easy/medium.',
    )
    parser.add_argument(
        '--bc-final-drop-step',
        type=int,
        choices=(300_000, 400_000),
        default=300_000,
        help='Stage-local 15-to-5 BC switch; 400000 is medium-only.',
    )
    # These default to the 2026-09-22-verified full compile + CUDA Graph
    # combination (see default_v2_cuda_graph_compilation below) unless
    # explicitly overridden with --no-<flag>. compile_critic_encoder,
    # compile_target_encoders is not part of that verified
    # combination and keep their plain False default.
    parser.add_argument('--window-update-statistics', action='store_true')
    parser.add_argument('--reuse-update-buffers', action='store_true')
    parser.add_argument('--gpu-td3-compute', action=argparse.BooleanOptionalAction, default=None, help='Auto-select supported CUDA compute; --no-gpu-td3-compute forces original updates.')
    parser.add_argument('--cpu-deduplication', action=argparse.BooleanOptionalAction, default=True, help='CPU reuse enabled by default; --no-cpu-deduplication disables it.')
    parser.add_argument('--gpu-td3-update', action='store_true', help='Experimental full device replay/update; CUDA 2.5.1+cu124 only; default off.')
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


def resolve_v2_gpu_compute(
    requested: bool | None, *, device: str, fused_adam: bool,
    formal_configuration: bool = True, conflicts: bool = False,
) -> dict[str, Any]:
    """Resolve execution only; capture/runtime errors must never fall back."""
    if requested is not None and type(requested) is not bool:
        raise TypeError('gpu_td3_compute must be bool or None (auto).')
    reason = None
    if requested is False:
        reason = 'explicitly disabled'
    elif not str(device).startswith('cuda'):
        reason = 'requires CUDA; original CPU update selected'
    elif not fused_adam or not formal_configuration or conflicts:
        reason = 'requires fused Adam and the supported formal compilation configuration without conflicting experiments'
    else:
        from brain_uav.trainers.v2_gpu_update import require_cuda_abi
        try:
            require_cuda_abi()
        except RuntimeError as exc:
            if requested is True:
                raise
            reason = str(exc)
    if requested is True and reason is not None:
        raise ValueError('gpu_td3_compute: ' + reason)
    return {'requested': requested, 'effective': reason is None,
            'reason': reason or ('supported CUDA auto selection' if requested is None else 'explicitly enabled'),
            'phase': 'configuration', 'actual_execution': None}


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
            'local_optimizations': engine.local_optimization_metadata(),
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
                'project_then_aggregate'
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
                'blocking_to_device'
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
    if (engine.gpu_td3_update or engine.gpu_td3_compute):
        metadata['update_execution_override'] = {
            'requested': True, 'effective_after_initialization': 'manual outer GPU update executor',
            'old_update_compilation': 'registered only; old backward/capture warmup skipped; not invoked by GPU updates',
            'scope_preserved': 'compiled environment action inference; no nested update CUDA Graphs'}
    metadata['local_optimizations'] = engine.local_optimization_metadata()
    metadata['reduce_update_stat_syncs_requested'] = bool(
        getattr(engine, 'reduce_update_stat_syncs', False)
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
    if (engine.gpu_td3_update or engine.gpu_td3_compute) and compile_actors:
        import torch
        # Preserve inference/validation compilation without first creating leaf
        # AccumulateGrad nodes on the default stream. The manual executor owns
        # first backward and its persistent stream across checkpoint reloads.
        with torch.no_grad():
            for batch in warmup_batches:
                engine._mark_cuda_graph_step_begin()
                for actor in ((engine.actor,) if engine.gpu_td3_compute else (engine.actor,engine.bc_reference_actor)):
                    if actor is not None: actor(batch)
    elif compile_actors:
        engine.warmup_actor_compile(warmup_batches)
    if compile_shared_relations and not engine.gpu_td3_compute:
        engine.warmup_shared_relations_compile(warmup_batches)
    if compile_critic_block and not (engine.gpu_td3_update or engine.gpu_td3_compute):
        engine.warmup_full_compile(warmup_batches)
    elif compile_critic_encoder and not (engine.gpu_td3_update or engine.gpu_td3_compute):
        engine.warmup_online_critic_encoder_compile(warmup_batches)
        if compile_target_encoders:
            engine.warmup_target_encoder_compile(warmup_batches)
    if compile_snn_target_encoder and not compile_target_block and not (engine.gpu_td3_update or engine.gpu_td3_compute):
        engine.warmup_snn_target_encoder_compile(warmup_batches)
    if compile_actor_loss and not (engine.gpu_td3_update or engine.gpu_td3_compute):
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
    if cuda_graph_updates and not (engine.gpu_td3_update or engine.gpu_td3_compute):
        metadata['cuda_graph_evidence'] = (
            engine.verify_update_cuda_graph_capture(warmup_batches)
        )
    elif (engine.gpu_td3_update or engine.gpu_td3_compute):
        metadata['cuda_graph_evidence']={'verified':False,'reason':'full executor capture/replay pending; old update probes intentionally skipped'}
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
    gamma: float = 0.99,
    target_q_aggregation: str = 'min',
    failure_sample_bias: float = 1.0,
    actor_lr: float | None = None,
    critic_lr: float | None = None,
    bc_final_drop_step: int = 300_000,
    bc_schedule: str = 'default',
    device: str = 'auto',
    max_stage_steps: int | None = None,
    early_stop_min_steps: int = 125_000,
    window_episodes: int = 15,
    consecutive_windows: int = 4,
    max_failures_per_window: int = 1,
    validation_max_failures: int = 5,
    scenario: ScenarioConfig | None = None,
    rewards: RewardConfig | None = None,
    uav_collision_radius: float | None = None,
    global_steps_start: int = 0,
    validation_scenario_count: int = 100,
    expected_validation_master_seed: int | None = None,
    expected_validation_stage_seed: int | None = None,
    model: str = 'ann',
    snn_time_window: int = 4,
    actor_projection_dim: int | None = None,
    actor_hidden_dim: int | None = None,
    actor_rl_scale_alpha: float = 2.5,
    spike_regularization: V2SpikeRegularizationConfig | None = None,
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

    reduce_update_stat_syncs: bool = False,
    window_update_statistics: bool = False,
    reuse_update_buffers: bool = False,

    gpu_td3_update: bool = False,
    gpu_td3_compute: bool | None = None,
    cpu_deduplication: bool = True,


    cuda_graph_actor_update: bool = False,
    cuda_graph_updates: bool = False,
    periodic_snapshot_interval_steps: int | None = None,
    target_noise_scale: float = 1.0,
    weight_snapshot_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    noise_schedule = V2NoiseSchedule().with_target_noise_scale(target_noise_scale)
    bc_schedule_metadata = v2_bc_schedule_metadata(
        stage,
        final_drop_step=bc_final_drop_step,
        schedule=bc_schedule,
    )
    requested_device = device
    resolved_device = resolve_training_device(requested_device)
    if type(cpu_deduplication) is not bool:
        raise TypeError('cpu_deduplication must be a bool.')
    compute_resolution = resolve_v2_gpu_compute(
        gpu_td3_compute, device=resolved_device, fused_adam=fused_adam,
        formal_configuration=all((compile_actors, compile_critic_block, compile_target_block,
            compile_shared_relations, compile_actor_loss, cache_actor_loss_coefficients,
            compile_action_inference, cuda_graph_action_inference, cuda_graph_updates,
            cuda_graph_actor_update, reduce_update_stat_syncs,
            frozen_critic_strategy == 'compiled_no_grad_context',
            model != 'snn' or compile_snn_target_encoder)),
        conflicts=any((gpu_td3_update, window_update_statistics, reuse_update_buffers)),
    )
    gpu_td3_compute = compute_resolution['effective']
    if gpu_td3_compute and any((gpu_td3_update,    window_update_statistics, reuse_update_buffers)):
        raise ValueError('gpu_td3_compute conflicts with other experimental paths.')
    if type(gpu_td3_update) is not bool:
        raise TypeError('gpu_td3_update must be a bool.')
    if gpu_td3_update and any((   window_update_statistics, reuse_update_buffers)):
        raise ValueError('gpu_td3_update conflicts with other experimental paths.')
    if model not in ('ann', 'snn'):
        raise ValueError('model must be "ann" or "snn".')
    validate_policy_dimensions(actor_projection_dim, actor_hidden_dim)
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
        gamma=gamma,
        target_q_aggregation=target_q_aggregation,
        failure_sample_bias=failure_sample_bias,
        actor_lr=_resolve_v2_actor_lr(stage, model, actor_lr),
        critic_lr=critic_lr,
        actor_rl_scale_alpha=actor_rl_scale_alpha,
        noise_schedule=noise_schedule,
        spike_regularization=V2SpikeRegularizationConfig() if spike_regularization is None else spike_regularization,
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
    if stage == 'easy':
        validate_policy_request(prepared_initialization.bc_initialization.actor, actor_projection_dim, actor_hidden_dim)
        effective_spike = config.spike_regularization
    else:
        previous = prepared_initialization.formal_checkpoint['engine_checkpoint']
        if actor_projection_dim is not None and actor_projection_dim != projection_from_architecture(previous['architecture']):
            raise ValueError('Requested actor projection differs from predecessor.')
        if actor_hidden_dim is not None and actor_hidden_dim != previous['architecture']['actor_hidden_dim']:
            raise ValueError('Requested actor width differs from predecessor.')
        inherited = V2SpikeRegularizationConfig.from_snapshot(previous['algorithm_config'].get('spike_regularization', V2SpikeRegularizationConfig().to_dict()))
        if spike_regularization is not None and spike_regularization != inherited:
            raise ValueError('Requested spike regularization differs from predecessor.')
        effective_spike = inherited
    effective_spike.validate_model(model)
    config = replace(config, spike_regularization=effective_spike)
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
    weight_snapshot_dir = output_path.with_name(f'{output_path.stem}_weights')
    if (output_path.exists() or failed_output_path.exists() or metrics_path.exists()
            or (reporting and report_path.exists())
            or (weight_snapshot_steps and weight_snapshot_dir.exists())
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

        reduce_update_stat_syncs=reduce_update_stat_syncs,
        window_update_statistics=window_update_statistics,
        reuse_update_buffers=reuse_update_buffers,

        gpu_td3_update=gpu_td3_update,
        gpu_td3_compute=gpu_td3_compute,


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
    initial_state_summary = v2_initial_state_summary(components.engine)
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
        stage_report_payload = {
            'initial_state_sha256': initial_state_summary,
            'requested_device': requested_device,
            'resolved_device': resolved_device,
            'initialization_source': dict(components.initialization_source),
            'seed': seed,
            'max_steps': config.max_steps,
            'formal_config': config.to_dict(),
            'bc_schedule': bc_schedule_metadata,
            'checkpoint_output': str(output_path),
            'failed_checkpoint_output': str(failed_output_path),
            'metrics_output': str(metrics_path),
            'snn': snn_metadata,
            'compilation': compilation_metadata,
        }

    def validate(actor):
        return evaluate_v2_fixed_validation(
            actor,
            pool,
            reward_config,
            max_failures=config.validation_max_failures,
            device=resolved_device,
            reporter=reporter,
            cpu_deduplication=cpu_deduplication,
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
                bc_final_drop_step=bc_final_drop_step,
                bc_schedule=bc_schedule,
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
                cpu_deduplication=cpu_deduplication,
            )
            if reporter is not None:
                record = dict(result.to_dict())
                record['stage_steps'] = stage_steps
                record['global_steps'] = global_steps_start + stage_steps
                reporter.record_periodic_validation(record)

    def weight_snapshot_sink(requested_step: int, actual_step: int) -> None:
        # Strict non-terminal format, never a formal-passed course predecessor.
        payload = build_v2_periodic_snapshot(
            components.engine, config, stage_steps=actual_step,
            scenario=scenario_config, rewards=reward_config,
            uav_collision_radius=effective_uav_collision_radius,
            seed_manifest=components.seed_manifest,
            initialization_source=components.initialization_source,
            bc_final_drop_step=bc_final_drop_step, bc_schedule=bc_schedule,
        )
        path = weight_snapshot_dir / f'requested_{requested_step:09d}_actual_{actual_step:09d}.pt'
        save_v2_periodic_snapshot(path, payload)

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
        bc_final_drop_step=bc_final_drop_step,
        bc_schedule=bc_schedule,
        periodic_snapshot_interval_steps=periodic_snapshot_interval_steps,
        periodic_snapshot_sink=periodic_snapshot_sink,
        weight_snapshot_steps=weight_snapshot_steps,
        weight_snapshot_sink=weight_snapshot_sink,
        periodic_validation_sink=periodic_validation_sink,
        cpu_deduplication=cpu_deduplication,
    )
    execution_configuration = trainer.execution_configuration
    execution_configuration['weight_snapshots'] = {
        'requested_steps': list(weight_snapshot_steps),
        'purpose': 'diagnostic_only_not_formal_passed_or_predecessor',
        'directory': str(weight_snapshot_dir), 'fixed_validation': False,
    }
    compute_resolution['configured'] = compute_resolution['effective']
    execution_configuration['gpu_td3_compute'] = compute_resolution
    execution_configuration['cpu_deduplication'].update({
        'candidate_validation_environment': cpu_deduplication,
        'candidate_validation_action_input': cpu_deduplication,
        'periodic_validation_environment': (
            cpu_deduplication if periodic_validation_sink is not None else None),
        'periodic_validation_action_input': (
            cpu_deduplication if periodic_validation_sink is not None else None),
        'compilation_warmup': 'not_applicable',
    })
    if reporter is not None:
        reporter.start_stage({**stage_report_payload,
                              'execution_configuration': execution_configuration})
    try:
        result = trainer.run()
        compilation_metadata['local_optimizations'] = components.engine.local_optimization_metadata()
        compute_resolution.update(phase='completed', actual_execution=compilation_metadata['local_optimizations'].get('gpu_td3_compute'))
        compute_resolution['effective'] = bool((compute_resolution['actual_execution'] or {}).get('effective', False))
        if compute_resolution['configured'] and not compute_resolution['effective']:
            compute_resolution['reason'] = 'configured, but no GPU compute execution was observed'
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
            bc_final_drop_step=bc_final_drop_step,
            bc_schedule=bc_schedule,
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
            'bc_schedule': bc_schedule_metadata,
            'seed_manifest': dict(components.seed_manifest),
            'validation_pool': validation_metadata,
            'initialization_source': dict(components.initialization_source),
            'requested_device': requested_device,
            'resolved_device': resolved_device,
            'replay_sampling_implementation': (
                components.engine.replay.sampling_implementation
            ),
            'result': result.to_dict(),
            'initial_state_sha256': initial_state_summary,
            'checkpoint': str(checkpoint_path),
            'report_directory': str(report_path) if reporter is not None else None,
            'compilation': compilation_metadata,
            'execution_configuration': execution_configuration,
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
            reporter.finish_stage({**result.to_dict(),
                                   'target_q_aggregation': config.target_q_aggregation,
                                   'initial_state_sha256': initial_state_summary,
                                   'execution_configuration': execution_configuration})
        summary = {
            'initial_state_sha256': initial_state_summary,
            'target_q_aggregation': config.target_q_aggregation,
            'stage': stage,
            'model_type': model,
            'bc_schedule': bc_schedule_metadata,
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
            'execution_configuration': execution_configuration,
        }
        return summary
    except BaseException as exc:
        compute_resolution.update(
            phase='failed', actual_execution=components.engine.local_optimization_metadata().get('gpu_td3_compute'))
        compute_resolution['effective'] = bool((compute_resolution['actual_execution'] or {}).get('effective', False))
        compute_resolution['reason'] = type(exc).__name__ + ': ' + str(exc)
        _write_strict_json(metrics_path.with_name(metrics_path.stem + '_execution_error.json'), {
            'error_type': type(exc).__name__, 'error': str(exc),
            'execution_configuration': execution_configuration,
        })
        raise
    finally:
        try:
            if getattr(components.engine, 'window_update_statistics', False):
                components.engine.flush_update_statistics()
        finally:
            if reporter is not None:
                reporter.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.actor_lr = _resolve_v2_actor_lr(args.stage, args.model, args.actor_lr)
    spike_config = spike_config_from_arguments(args)
    resolved_device = resolve_training_device(args.device)
    config_preview = V2FormalTrainingConfig(
        noise_schedule=V2NoiseSchedule().with_target_noise_scale(args.target_noise_scale),
        stage=args.stage,
        seed=args.seed,
        max_steps=args.max_stage_steps,
        early_stop_min_steps=args.early_stop_min_steps,
        window_episode_count=args.window_episodes,
        consecutive_qualified_windows=args.consecutive_windows,
        max_failures_per_window=args.max_failures_per_window,
        validation_max_failures=args.validation_max_failures,
        gamma=args.gamma,
        target_q_aggregation=args.target_q_aggregation,
        failure_sample_bias=args.failure_sample_bias,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        actor_rl_scale_alpha=args.actor_rl_scale_alpha,
        spike_regularization=spike_config,
    )
    print(json.dumps({
        'execution_configuration': {'cpu_deduplication': {'requested': args.cpu_deduplication}},
        'local_optimizations_requested': {'window_update_statistics': args.window_update_statistics,
                                          'reuse_update_buffers': args.reuse_update_buffers,  'gpu_td3_update': args.gpu_td3_update, 'gpu_td3_compute': args.gpu_td3_compute},
        'requested_device': args.device,
        'resolved_device': resolved_device,
        'resolved_v2_formal_config': config_preview.to_dict(),
        'bc_schedule': v2_bc_schedule_metadata(
            args.stage,
            final_drop_step=args.bc_final_drop_step,
            schedule=args.bc_schedule,
        ),
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
        gamma=args.gamma,
        target_q_aggregation=args.target_q_aggregation,
        failure_sample_bias=args.failure_sample_bias,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        bc_final_drop_step=args.bc_final_drop_step,
        bc_schedule=args.bc_schedule,
        model=args.model,
        snn_time_window=args.snn_time_window,
        actor_projection_dim=args.actor_projection_dim,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_rl_scale_alpha=args.actor_rl_scale_alpha,
        spike_regularization=spike_config,
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

        reduce_update_stat_syncs=compilation['reduce_update_stat_syncs'],
        window_update_statistics=args.window_update_statistics,
        reuse_update_buffers=args.reuse_update_buffers,

        gpu_td3_update=args.gpu_td3_update,
        gpu_td3_compute=args.gpu_td3_compute,
        cpu_deduplication=args.cpu_deduplication,


        cuda_graph_updates=compilation['cuda_graph_updates'],
        cuda_graph_actor_update=compilation['cuda_graph_actor_update'],
        periodic_snapshot_interval_steps=(
            # Periodic full-pool validation remains independent of weight snapshots.
            None if args.periodic_snapshot_interval_steps == 0
            else args.periodic_snapshot_interval_steps
        ),
        target_noise_scale=args.target_noise_scale,
        weight_snapshot_steps=tuple(args.weight_snapshot_steps),
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
