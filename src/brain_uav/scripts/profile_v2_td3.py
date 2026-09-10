"""Bounded V2 ANN/SNN TD3 timing diagnostic, never a formal curriculum run."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
import torch

from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.models import V2SNNPolicyActor
from brain_uav.scripts.common import DEVICE_CHOICES, resolve_training_device
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig,
    build_v2_stage_engine,
    prepare_v2_stage_initialization,
    v2_bc_lambda,
)
from brain_uav.trainers.v2_validation import (
    V2ValidationPool,
    derive_validation_stage_seed,
    generate_v2_validation_pool,
    load_v2_validation_pool,
    save_v2_validation_pool,
)


DIAGNOSTIC_FORMAT = 'v2_td3_timing_diagnostic'
DIAGNOSTIC_VERSION = 1
DIAGNOSTIC_LEVELS = ('easy', 'medium', 'hard')


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a non-negative integer.')
    return value


def _strict_json_write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite diagnostic output: {path}')
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2)
        + '\n',
        encoding='utf-8',
    )


class _TimingBook:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.wall_seconds: defaultdict[str, float] = defaultdict(float)
        self.calls: defaultdict[str, int] = defaultdict(int)
        self._cuda_pairs: defaultdict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = defaultdict(list)

    def call(
        self,
        name: str,
        operation: Callable[[], Any],
        *,
        cuda_event: bool = False,
    ) -> Any:
        event_pair = None
        if cuda_event and self.device.type == 'cuda':
            event_pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            event_pair[0].record()
        started = perf_counter()
        try:
            return operation()
        finally:
            self.wall_seconds[name] += perf_counter() - started
            self.calls[name] += 1
            if event_pair is not None:
                event_pair[1].record()
                self._cuda_pairs[name].append(event_pair)

    def cuda_stream_interval_seconds(self) -> dict[str, float] | None:
        if self.device.type != 'cuda':
            return None
        torch.cuda.synchronize(self.device)
        return {
            name: sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
            for name, pairs in self._cuda_pairs.items()
        }


def _load_diagnostic_initialization(
    config: V2FormalTrainingConfig,
    *,
    bc_checkpoint: Path,
    device: torch.device,
    model: str,
    snn_time_window: int,
):
    return prepare_v2_stage_initialization(
        config,
        init_checkpoint=bc_checkpoint,
        scenario=None,
        rewards=None,
        uav_collision_radius=None,
        device=device,
        model_type=model,
        snn_time_window=snn_time_window,
    )


def _prepare_diagnostic_pools(
    directory: Path,
    *,
    scenario,
    scenario_count: int,
    master_seed: int,
    uav_collision_radius: float,
) -> dict[str, V2ValidationPool]:
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f'Diagnostic pool path is not a directory: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    pools: dict[str, V2ValidationPool] = {}
    for level in DIAGNOSTIC_LEVELS:
        path = directory / f'{level}.json'
        stage_seed = derive_validation_stage_seed(master_seed, level)
        if path.exists():
            pool = load_v2_validation_pool(
                path,
                expected_level=level,
                expected_scenario=scenario,
                expected_count=scenario_count,
                expected_uav_collision_radius=uav_collision_radius,
                expected_master_seed=master_seed,
                expected_stage_seed=stage_seed,
            )
        else:
            pool = generate_v2_validation_pool(
                scenario,
                level,
                scenario_count=scenario_count,
                master_seed=master_seed,
                uav_collision_radius=uav_collision_radius,
            )
            save_v2_validation_pool(path, pool)
        pools[level] = pool
    return pools


def _near_goal(info: dict[str, Any], *, radius: float) -> bool:
    values = (
        float(info.get('goal_distance', float('inf'))),
        float(info.get('segment_goal_distance', float('inf'))),
    )
    return bool(info.get('goal_reached_by_segment', False)) or min(values) <= radius


def _run_diagnostic_level(
    *,
    level: str,
    pool: V2ValidationPool,
    prepared,
    formal_config: V2FormalTrainingConfig,
    bc_checkpoint: Path,
    model: str,
    snn_time_window: int,
    device: torch.device,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    scenario_count = len(pool.scenarios)
    if scenario_count == 0 or measured_steps < scenario_count:
        raise ValueError('measured_steps must cover every fixed-pool scenario.')
    per_scenario, remainder = divmod(measured_steps, scenario_count)
    budgets = [per_scenario + int(index < remainder) for index in range(scenario_count)]
    coverage = [
        {
            'scenario_id': record['scenario_id'],
            'zone_count': len(record['payload']['zones']),
            'measured_steps': 0,
            'episodes_completed': 0,
        }
        for record in pool.scenarios
    ]
    components = build_v2_stage_engine(
        None,
        formal_config,
        init_checkpoint=bc_checkpoint,
        rewards=None,
        uav_collision_radius=None,
        device=device,
        model_type=model,
        snn_time_window=snn_time_window,
        prepared_initialization=prepared,
    )
    engine = components.engine
    engine.actor.train()
    env = V2StaticNoFlyTrajectoryEnv(
        prepared.scenario_config,
        prepared.reward_config,
        seed=pool.stage_seed,
        fixed_scenarios=[record['payload'] for record in pool.scenarios],
        uav_collision_radius=prepared.uav_collision_radius,
    )
    timing = _TimingBook(device)
    observation = None
    episode_transitions: list[tuple[Any, ...]] = []
    slot_refs: list[tuple[int, int]] = []
    episodes_completed = 0
    measured_started: float | None = None
    actor_updates_before = 0
    critic_updates_before = 0
    original_sample = engine.replay.sample
    measured_phase = False
    step_number = 0
    measured_count = 0
    actual_warmup_steps = 0
    scenario_index = 0
    # Bound failure if the expected update cadence cannot reach the warmup gate.
    warmup_limit = max(
        warmup_steps, engine.batch_size - 1, formal_config.actor_freeze_steps,
    ) + max(4, 2 * formal_config.policy_delay)

    def timed_sample(batch_size: int):
        if not measured_phase:
            return original_sample(batch_size)
        return timing.call(
            'replay_sample_wall_seconds',
            lambda: original_sample(batch_size),
        )

    engine.replay.sample = timed_sample
    try:
        while measured_count < measured_steps:
            if not measured_phase and (
                step_number >= warmup_steps
                and engine.critic_update_count >= 4
                and engine.actor_update_count >= 2
            ):
                # Discard the warmup fragment without inventing a terminal transition.
                observation = None
                episode_transitions = []
                slot_refs = []
                actual_warmup_steps = step_number
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                measured_started = perf_counter()
                actor_updates_before = engine.actor_update_count
                critic_updates_before = engine.critic_update_count
                measured_phase = True
            if not measured_phase and step_number >= warmup_limit:
                raise RuntimeError('Diagnostic warmup did not reach critic/actor update minima.')
            call = timing.call if measured_phase else lambda _n, op, **_k: op()
            if observation is None:
                observation, _ = call(
                    'scenario_reset_wall_seconds',
                    lambda: env.reset(options={
                        'scenario': pool.scenarios[scenario_index]['payload'],
                    }),
                )
                episode_transitions = []
                slot_refs = []
            line_safe = call(
                'pre_action_geometry_wall_seconds',
                lambda: env.line_to_goal_is_safe(
                    env.state[:3],
                    clearance=formal_config.terminal_geo_safe_clearance,
                ),
            )
            action = call(
                'action_inference_wall_seconds',
                lambda: engine.select_action(
                    observation,
                    exploration_noise=formal_config.noise_schedule.exploration_initial,
                    exploration_rng=components.exploration_rng,
                ),
                cuda_event=True,
            )
            next_observation, reward, terminated, truncated, info = call(
                'environment_step_wall_seconds',
                lambda: env.step(action),
            )
            done = bool(terminated or truncated)
            near_goal = _near_goal(info, radius=formal_config.near_goal_radius)
            transition = (
                observation,
                np.asarray(action, dtype=np.float32).copy(),
                float(reward),
                next_observation,
                done,
                near_goal,
                line_safe,
            )
            slot_ref = call(
                'replay_write_wall_seconds',
                lambda: engine.replay.add(
                    observation,
                    action,
                    reward,
                    next_observation,
                    done,
                    success=False,
                    near_goal=near_goal,
                    line_to_goal_safe=line_safe,
                ),
            )
            slot_refs.append(slot_ref)
            episode_transitions.append(transition)
            step_number += 1
            if len(engine.replay) >= engine.batch_size:
                engine.set_target_noise(
                    policy_noise=formal_config.noise_schedule.policy_initial,
                    noise_clip=formal_config.noise_schedule.clip_initial,
                )
                call(
                    'td3_update_wall_seconds',
                    lambda: engine.update_once(
                        total_steps=step_number,
                        bc_lambda=v2_bc_lambda(step_number - 1),
                    ),
                    cuda_event=True,
                )
            observation = next_observation
            if done:
                outcome = str(info.get('outcome', ''))
                if outcome == 'goal':
                    def record_success() -> None:
                        for item in episode_transitions:
                            engine.replay.add_success_transition(
                                item[0], item[1], item[2], item[3], item[4],
                                near_goal=item[5],
                                line_to_goal_safe=item[6],
                            )
                        engine.replay.mark_success_slots(slot_refs, success=True)
                    call('replay_write_wall_seconds', record_success)
                episodes_completed += int(measured_phase)
                if measured_phase:
                    coverage[scenario_index]['episodes_completed'] += 1
                observation = None
            if measured_phase:
                measured_count += 1
                coverage[scenario_index]['measured_steps'] += 1
                if coverage[scenario_index]['measured_steps'] == budgets[scenario_index]:
                    # A budget boundary ends only this fragment, never the environment episode.
                    scenario_index += 1
                    observation = None
                    episode_transitions = []
                    slot_refs = []
        stream_seconds = timing.cuda_stream_interval_seconds()
        measured_total = perf_counter() - measured_started
    finally:
        engine.replay.sample = original_sample

    update_seconds = timing.wall_seconds['td3_update_wall_seconds']
    sample_seconds = timing.wall_seconds['replay_sample_wall_seconds']
    network_update_seconds = max(0.0, update_seconds - sample_seconds)
    if not all(isfinite(value) and value >= 0.0 for value in timing.wall_seconds.values()):
        raise RuntimeError('Diagnostic produced a non-finite timing value.')
    actor_updates = engine.actor_update_count - actor_updates_before
    critic_updates = engine.critic_update_count - critic_updates_before
    if critic_updates == 0 or actor_updates == 0:
        raise RuntimeError('Diagnostic measurement must include both critic and actor updates.')
    return {
        'curriculum_level': level,
        'requested_minimum_warmup_steps': warmup_steps,
        'warmup_steps': actual_warmup_steps,
        'warmup_critic_updates': critic_updates_before,
        'warmup_actor_updates': actor_updates_before,
        'measured_steps': measured_steps,
        'scenario_coverage': coverage,
        'episodes_completed': episodes_completed,
        'actor_updates': actor_updates,
        'critic_updates': critic_updates,
        'actor_target_updates': actor_updates,
        'timing': {
            'total_wall_seconds': measured_total,
            'throughput_environment_steps_per_second': (
                measured_steps / measured_total if measured_total > 0.0 else None
            ),
            'scenario_reset_wall_seconds': timing.wall_seconds['scenario_reset_wall_seconds'],
            'action_inference_wall_seconds': timing.wall_seconds['action_inference_wall_seconds'],
            'pre_action_geometry_wall_seconds': timing.wall_seconds['pre_action_geometry_wall_seconds'],
            'environment_step_wall_seconds': timing.wall_seconds['environment_step_wall_seconds'],
            'replay_write_wall_seconds': timing.wall_seconds['replay_write_wall_seconds'],
            'td3_update_wall_seconds': update_seconds,
            'replay_sample_wall_seconds': sample_seconds,
            'replay_sample_relation': 'within_td3_update',
            'td3_update_excluding_replay_sample_wall_seconds': network_update_seconds,
            'cuda_stream_interval_seconds': stream_seconds,
            'cuda_timing_note': (
                'CUDA stream interval time may include host submission gaps and waits; '
                'not pure GPU compute time and not a GPU compute utilization estimate. '
                'Explicit synchronization occurs once before measurement and once after the level.'
                if device.type == 'cuda'
                else 'CUDA stream interval time is not available on CPU; no CUDA synchronization.'
            ),
            'calls': dict(timing.calls),
            'measurement_note': (
                'Instrumented wall sections include measurement overhead; '
                'replay sample is nested within TD3 update and is not additive. '
                'This is load measurement of short fixed-scenario fragments, '
                'not coverage of all flight positions or formal full-episode curriculum throughput.'
            ),
        },
    }


def run_v2_td3_timing_diagnostic(
    *,
    model: str,
    bc_checkpoint: str | Path,
    output_dir: str | Path,
    scenario_pool_dir: str | Path,
    device: str = 'auto',
    seed: int = 7,
    validation_seed: int = 20260904,
    snn_time_window: int = 4,
    steps_per_level: int = 256,
    warmup_steps: int = 16,
    batch_size: int = 64,
    replay_capacity: int = 2048,
    scenario_count: int = 3,
) -> dict[str, Any]:
    if model not in ('ann', 'snn'):
        raise ValueError('model must be ann or snn.')
    steps = _positive_int(steps_per_level, name='steps_per_level')
    warmup = _nonnegative_int(warmup_steps, name='warmup_steps')
    batch = _positive_int(batch_size, name='batch_size')
    capacity = _positive_int(replay_capacity, name='replay_capacity')
    if capacity < batch:
        raise ValueError('replay_capacity must be at least batch_size.')
    pool_count = _positive_int(scenario_count, name='scenario_count')
    if steps < pool_count:
        raise ValueError('steps_per_level must be at least scenario_count.')
    run_seed = _nonnegative_int(seed, name='seed')
    pool_seed = _nonnegative_int(validation_seed, name='validation_seed')
    if type(snn_time_window) is not int or snn_time_window <= 0:
        raise ValueError('snn_time_window must be a positive integer.')
    requested_device = device
    resolved_device = resolve_training_device(device)
    target_device = torch.device(resolved_device)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Use a fresh diagnostic output directory: {output}')
    config = V2FormalTrainingConfig(
        stage='easy',
        seed=run_seed,
        max_steps=max(1, warmup + steps),
        replay_capacity=capacity,
        batch_size=batch,
        warmup_steps=0,
        actor_freeze_steps=0,
    )
    checkpoint = Path(bc_checkpoint)
    prepared = _load_diagnostic_initialization(
        config,
        bc_checkpoint=checkpoint,
        device=target_device,
        model=model,
        snn_time_window=snn_time_window,
    )
    actor = prepared.bc_initialization.actor
    if model == 'snn':
        if not isinstance(actor, V2SNNPolicyActor):
            raise ValueError('SNN diagnostic requires a V2 SNN BC actor.')
        snn_metadata: dict[str, Any] | None = {
            'time_window': actor.time_window,
            'tau': actor.tau,
            'surrogate': actor.surrogate_name,
            'backend': actor.backend,
        }
    else:
        snn_metadata = None
    pool_started = perf_counter()
    pools = _prepare_diagnostic_pools(
        Path(scenario_pool_dir),
        scenario=prepared.scenario_config,
        scenario_count=pool_count,
        master_seed=pool_seed,
        uav_collision_radius=prepared.uav_collision_radius,
    )
    pool_prepare_seconds = perf_counter() - pool_started
    output.mkdir(parents=True, exist_ok=False)
    level_results: dict[str, Any] = {}
    for level in DIAGNOSTIC_LEVELS:
        result = _run_diagnostic_level(
            level=level,
            pool=pools[level],
            prepared=prepared,
            formal_config=config,
            bc_checkpoint=checkpoint,
            model=model,
            snn_time_window=(
                actor.time_window if isinstance(actor, V2SNNPolicyActor) else snn_time_window
            ),
            device=target_device,
            warmup_steps=warmup,
            measured_steps=steps,
        )
        level_results[level] = result
        print(json.dumps({
            'level': level,
            'steps': result['measured_steps'],
            'wall_seconds': result['timing']['total_wall_seconds'],
            'actor_updates': result['actor_updates'],
            'critic_updates': result['critic_updates'],
        }, allow_nan=False, ensure_ascii=False), flush=True)
    summary = {
        'format': DIAGNOSTIC_FORMAT,
        'format_version': DIAGNOSTIC_VERSION,
        'formal_stage_passed': False,
        'purpose': 'short_timing_diagnostic_not_for_promotion_or_ranking',
        'model': model,
        'snn': snn_metadata,
        'requested_device': requested_device,
        'resolved_device': resolved_device,
        'seed': run_seed,
        'validation_seed': pool_seed,
        'bc_checkpoint': str(checkpoint.resolve()),
        'scenario_pool_directory': str(Path(scenario_pool_dir).resolve()),
        'scenario_pool_prepare_wall_seconds': pool_prepare_seconds,
        'scenario_config': asdict(prepared.scenario_config),
        'uav_collision_radius': prepared.uav_collision_radius,
        'diagnostic_config': {
            'minimum_warmup_steps_per_level': warmup,
            'minimum_warmup_critic_updates': 4,
            'minimum_warmup_actor_updates': 2,
            'measured_steps_per_level': steps,
            'batch_size': batch,
            'replay_capacity': capacity,
            'actor_freeze_steps': 0,
            'scenario_count_per_level': pool_count,
            'bc_lambda_schedule': 'formal_v2_stage_local_schedule',
            'exploration_noise': config.noise_schedule.exploration_initial,
            'policy_noise': config.noise_schedule.policy_initial,
            'noise_clip': config.noise_schedule.clip_initial,
            'geometry_subtiming': 'pre_action_line_safety_only',
            'observation_subtiming': 'included_in_reset_and_environment_step',
        },
        'actor_trainable_parameter_count': sum(
            parameter.numel() for parameter in actor.parameters() if parameter.requires_grad
        ),
        'pools': {
            level: {
                'master_seed': pools[level].master_seed,
                'stage_seed': pools[level].stage_seed,
                'scenario_count': pools[level].scenario_count,
                'content_digest': pools[level].content_digest,
            }
            for level in DIAGNOSTIC_LEVELS
        },
        'levels': level_results,
    }
    summary = json.loads(json.dumps(
        summary,
        allow_nan=False,
        ensure_ascii=False,
    ))
    _strict_json_write(output / 'diagnostic_summary.json', summary)
    print(json.dumps(summary, allow_nan=False, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('ann', 'snn'), required=True)
    parser.add_argument('--bc-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scenario-pool-dir', type=Path, required=True)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--validation-seed', type=int, default=20260904)
    parser.add_argument('--snn-time-window', type=int, default=4)
    parser.add_argument('--steps-per-level', type=int, default=256)
    parser.add_argument('--warmup-steps', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--replay-capacity', type=int, default=2048)
    parser.add_argument('--scenario-count', type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_v2_td3_timing_diagnostic(
        model=args.model,
        bc_checkpoint=args.bc_checkpoint,
        output_dir=args.output_dir,
        scenario_pool_dir=args.scenario_pool_dir,
        device=args.device,
        seed=args.seed,
        validation_seed=args.validation_seed,
        snn_time_window=args.snn_time_window,
        steps_per_level=args.steps_per_level,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        scenario_count=args.scenario_count,
    )


if __name__ == '__main__':
    main()


__all__ = [
    'DIAGNOSTIC_FORMAT',
    'DIAGNOSTIC_VERSION',
    'build_parser',
    'main',
    'run_v2_td3_timing_diagnostic',
]
