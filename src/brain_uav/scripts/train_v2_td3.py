"""Command-line entry point for one formal V2 ANN/SNN TD3 stage."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.models import V2SNNPolicyActor, require_v2_spikingjelly
from brain_uav.scripts.common import DEVICE_CHOICES, resolve_training_device
from brain_uav.trainers.v2_formal_training import (
    V2FormalStageTrainer,
    V2FormalTrainingConfig,
    V2PreparedStageInitialization,
    build_v2_formal_checkpoint,
    build_v2_stage_engine,
    prepare_v2_stage_initialization,
    save_v2_formal_checkpoint,
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
    return parser


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
    output_path = Path(output)
    metrics_path = Path(metrics_out)
    failed_output_path = _failed_checkpoint_path(output_path)
    report_path = metrics_path.with_name(f'{metrics_path.stem}_reports')
    if (output_path.exists() or failed_output_path.exists() or metrics_path.exists()
            or (reporting and report_path.exists())):
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
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
