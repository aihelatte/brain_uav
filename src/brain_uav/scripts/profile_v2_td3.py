"""Bounded V2 ANN/SNN TD3 timing diagnostic, never a formal curriculum run."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

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
from brain_uav.trainers.v2_td3 import V2_TD3_UPDATE_TIMING_SECTIONS
from brain_uav.trainers.v2_validation import (
    V2ValidationPool,
    derive_validation_stage_seed,
    generate_v2_validation_pool,
    load_v2_validation_pool,
    save_v2_validation_pool,
)


DIAGNOSTIC_FORMAT = 'v2_td3_timing_diagnostic'
DIAGNOSTIC_VERSION = 3
DIAGNOSTIC_LEVELS = ('easy', 'medium', 'hard')
UPDATE_TIMING_SECTION_NAMES = V2_TD3_UPDATE_TIMING_SECTIONS


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
        on_complete: Callable[[Any, float], None] | None = None,
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
            result = operation()
        finally:
            elapsed = perf_counter() - started
            self.wall_seconds[name] += elapsed
            self.calls[name] += 1
            if event_pair is not None:
                event_pair[1].record()
                self._cuda_pairs[name].append(event_pair)
        if on_complete is not None:
            on_complete(result, elapsed)
        return result

    def cuda_stream_interval_seconds(self) -> dict[str, float] | None:
        if self.device.type != 'cuda':
            return None
        torch.cuda.synchronize(self.device)
        return {
            name: sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
            for name, pairs in self._cuda_pairs.items()
        }


class _UpdateTimingSummary:
    """Aggregate mutually exclusive update sections by actual update result."""

    def __init__(self) -> None:
        self._buckets = {
            name: {
                'update_count': 0,
                'total_wall_seconds': 0.0,
                'sections': {section: 0.0 for section in UPDATE_TIMING_SECTION_NAMES},
            }
            for name in ('critic_only', 'actor_updated', 'overall_weighted')
        }

    def record(
        self,
        *,
        actor_updated: bool,
        total_wall_seconds: float,
        sections: Mapping[str, float],
    ) -> None:
        if set(sections) != set(UPDATE_TIMING_SECTION_NAMES):
            raise RuntimeError('TD3 update timing sections are missing or unknown.')
        total = float(total_wall_seconds)
        values = {name: float(sections[name]) for name in UPDATE_TIMING_SECTION_NAMES}
        if not isfinite(total) or total < 0.0:
            raise RuntimeError('TD3 update wall time must be finite and non-negative.')
        if any(not isfinite(value) or value < 0.0 for value in values.values()):
            raise RuntimeError('TD3 update section times must be finite and non-negative.')
        classification = 'actor_updated' if actor_updated else 'critic_only'
        for name in (classification, 'overall_weighted'):
            bucket = self._buckets[name]
            bucket['update_count'] += 1
            bucket['total_wall_seconds'] += total
            for section, value in values.items():
                bucket['sections'][section] += value

    @staticmethod
    def _bucket_payload(bucket: dict[str, Any]) -> dict[str, Any]:
        count = int(bucket['update_count'])
        total = float(bucket['total_wall_seconds'])
        section_totals = dict(bucket['sections'])
        section_totals['other_uncovered'] = total - sum(section_totals.values())
        return {
            'update_count': count,
            'total_wall_seconds': total,
            'average_wall_seconds': total / count if count else None,
            'sections': {
                name: {
                    'total_wall_seconds': value,
                    'average_wall_seconds': value / count if count else None,
                    'percent_of_update_wall_seconds': (
                        100.0 * value / total if total > 0.0 else None
                    ),
                }
                for name, value in section_totals.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            name: self._bucket_payload(bucket)
            for name, bucket in self._buckets.items()
        }


class _DetailedUpdateProfiler:
    """Bounded profiler collection for measured TD3 updates only."""

    def __init__(
        self,
        device: torch.device,
        *,
        requested_updates: int,
        output_dir: Path | None,
    ) -> None:
        self.device = device
        self.requested_updates = _nonnegative_int(
            requested_updates,
            name='detailed_profiler_updates',
        )
        if self.requested_updates and output_dir is None:
            raise ValueError('profiler_output_dir is required when profiling is enabled.')
        self.output_dir = output_dir
        self.captured_updates = 0
        self._profiler = None
        self._activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            self._activities.append(torch.profiler.ProfilerActivity.CUDA)

    @property
    def enabled(self) -> bool:
        return self.requested_updates > 0

    def _ensure_started(self) -> None:
        if self._profiler is not None:
            return
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._profiler = torch.profiler.profile(
            activities=self._activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self._profiler.__enter__()
        self._profiler.toggle_collection_dynamic(False, self._activities)

    def run(self, operation: Callable[[bool], Any]) -> Any:
        if not self.enabled or self.captured_updates >= self.requested_updates:
            return operation(False)
        self._ensure_started()
        self._profiler.toggle_collection_dynamic(True, self._activities)
        try:
            result = operation(True)
        finally:
            self._profiler.toggle_collection_dynamic(False, self._activities)
        self.captured_updates += 1
        self._profiler.step()
        return result

    def finish(self) -> dict[str, Any]:
        output_paths: dict[str, str] = {}
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            assert self.output_dir is not None
            averages = self._profiler.key_averages()
            cpu_path = self.output_dir / 'operators_cpu.txt'
            cuda_path = self.output_dir / 'operators_cuda.txt'
            trace_path = self.output_dir / 'trace.json'
            cpu_path.write_text(
                averages.table(sort_by='self_cpu_time_total', row_limit=50),
                encoding='utf-8',
            )
            if self.device.type == 'cuda':
                cuda_text = averages.table(
                    sort_by='self_cuda_time_total',
                    row_limit=50,
                )
            else:
                cuda_text = 'CUDA activity was not collected for this CPU diagnostic.\n'
            cuda_path.write_text(cuda_text, encoding='utf-8')
            self._profiler.export_chrome_trace(str(trace_path))
            output_paths = {
                'cpu_operator_table': str(cpu_path.resolve()),
                'cuda_operator_table': str(cuda_path.resolve()),
                'trace': str(trace_path.resolve()),
            }
        return {
            'enabled': self.enabled,
            'requested_updates': self.requested_updates,
            'captured_updates': self.captured_updates,
            'output_paths': output_paths,
            'measurement_note': (
                'The profiler adds overhead. Its sampled updates are separately '
                'identified and excluded from the ordinary update breakdown. CPU '
                'region time is not pure GPU compute time; CUDA synchronization waits '
                'may appear in later regions such as gradient checks. Parent and child '
                'regions must not be added together, and no per-region CUDA '
                'synchronization is introduced.'
            ),
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
    detailed_profiler_updates: int = 0,
    profiler_output_dir: Path | None = None,
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
    update_timing = _UpdateTimingSummary()
    detailed_profiler = _DetailedUpdateProfiler(
        device,
        requested_updates=detailed_profiler_updates,
        output_dir=profiler_output_dir,
    )
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
    profiled_update_active = False
    # Bound failure if the expected update cadence cannot reach the warmup gate.
    warmup_limit = max(
        warmup_steps, engine.batch_size - 1, formal_config.actor_freeze_steps,
    ) + max(4, 2 * formal_config.policy_delay)

    def timed_sample(batch_size: int):
        if not measured_phase:
            return original_sample(batch_size)
        return timing.call(
            (
                'profiled_replay_sample_wall_seconds'
                if profiled_update_active
                else 'replay_sample_wall_seconds'
            ),
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
                update_sections: dict[str, float] = {}

                def capture_update_sections(sections: Mapping[str, float]) -> None:
                    update_sections.update(sections)

                def record_update(result: Any, elapsed: float) -> None:
                    update_timing.record(
                        actor_updated=bool(result.actor_updated),
                        total_wall_seconds=elapsed,
                        sections=update_sections,
                    )

                def perform_update(profile_sections: bool):
                    nonlocal profiled_update_active
                    profiled_update_active = profile_sections
                    try:
                        return call(
                            (
                                'td3_profiled_update_wall_seconds'
                                if profile_sections
                                else 'td3_update_wall_seconds'
                            ),
                            lambda: engine.update_once(
                                total_steps=step_number,
                                bc_lambda=v2_bc_lambda(step_number - 1),
                                timing_recorder=(
                                    capture_update_sections
                                    if measured_phase and not profile_sections
                                    else None
                                ),
                                profile_sections=profile_sections,
                            ),
                            cuda_event=True,
                            on_complete=(None if profile_sections else record_update),
                        )
                    finally:
                        profiled_update_active = False

                if measured_phase:
                    detailed_profiler.run(perform_update)
                else:
                    perform_update(False)
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
        profiler_metadata = detailed_profiler.finish()

    update_seconds = timing.wall_seconds['td3_update_wall_seconds']
    sample_seconds = timing.wall_seconds['replay_sample_wall_seconds']
    network_update_seconds = max(0.0, update_seconds - sample_seconds)
    if not all(isfinite(value) and value >= 0.0 for value in timing.wall_seconds.values()):
        raise RuntimeError('Diagnostic produced a non-finite timing value.')
    actor_updates = engine.actor_update_count - actor_updates_before
    critic_updates = engine.critic_update_count - critic_updates_before
    if critic_updates == 0 or actor_updates == 0:
        raise RuntimeError('Diagnostic measurement must include both critic and actor updates.')
    profiler_overhead_included = bool(profiler_metadata['enabled'])
    total_wall_seconds_note = (
        'This is wall time for the diagnostic run including detailed profiler '
        'collection, startup, and activity-switching overhead. It must not be used '
        'as normal training throughput or for optimization speedup comparisons.'
        if profiler_overhead_included
        else 'Detailed profiler is disabled; throughput is calculated from this '
        'ordinary short-diagnostic wall time.'
    )
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
        'detailed_profiler': profiler_metadata,
        'timing': {
            'total_wall_seconds': measured_total,
            'total_wall_seconds_includes_detailed_profiler_overhead': (
                profiler_overhead_included
            ),
            'total_wall_seconds_note': total_wall_seconds_note,
            'throughput_environment_steps_per_second': (
                measured_steps / measured_total
                if not profiler_overhead_included and measured_total > 0.0
                else None
            ),
            'scenario_reset_wall_seconds': timing.wall_seconds['scenario_reset_wall_seconds'],
            'action_inference_wall_seconds': timing.wall_seconds['action_inference_wall_seconds'],
            'pre_action_geometry_wall_seconds': timing.wall_seconds['pre_action_geometry_wall_seconds'],
            'environment_step_wall_seconds': timing.wall_seconds['environment_step_wall_seconds'],
            'replay_write_wall_seconds': timing.wall_seconds['replay_write_wall_seconds'],
            'td3_update_wall_seconds': update_seconds,
            'td3_profiled_update_wall_seconds': timing.wall_seconds[
                'td3_profiled_update_wall_seconds'
            ],
            'td3_update_breakdown': update_timing.to_dict(),
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
                'Update sub-sections use CPU wall time without per-section CUDA '
                'synchronization; asynchronous work or waits may therefore appear '
                'in a later section. The uncovered value is the outer update_once '
                'wall time minus measured sections and is not clamped. '
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
    detailed_profiler_updates: int = 0,
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
    profiler_updates = _nonnegative_int(
        detailed_profiler_updates,
        name='detailed_profiler_updates',
    )
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
            detailed_profiler_updates=profiler_updates,
            profiler_output_dir=(
                output / 'profiler' / level if profiler_updates else None
            ),
        )
        level_results[level] = result
        print(json.dumps({
            'level': level,
            'steps': result['measured_steps'],
            'wall_seconds': result['timing']['total_wall_seconds'],
            'actor_updates': result['actor_updates'],
            'critic_updates': result['critic_updates'],
            'detailed_profiler_updates': result['detailed_profiler'][
                'captured_updates'
            ],
            'update_breakdown_average_ms': {
                name: {
                    'count': bucket['update_count'],
                    'update_total': (
                        None if bucket['average_wall_seconds'] is None
                        else 1000.0 * bucket['average_wall_seconds']
                    ),
                    'sections': {
                        section: (
                            None if values['average_wall_seconds'] is None
                            else 1000.0 * values['average_wall_seconds']
                        )
                        for section, values in bucket['sections'].items()
                    },
                }
                for name, bucket in result['timing']['td3_update_breakdown'].items()
                if name != 'overall_weighted'
            },
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
            'detailed_profiler_updates_per_level': profiler_updates,
            'detailed_profiler_scope': (
                'first measured TD3 updates after warmup; excluded from ordinary '
                'update timing breakdown'
            ),
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
    parser.add_argument(
        '--detailed-profiler-updates',
        type=int,
        nargs='?',
        const=16,
        default=0,
        metavar='N',
        help='Profile at most N post-warmup TD3 updates per level (bare flag: 16).',
    )
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
        detailed_profiler_updates=args.detailed_profiler_updates,
    )


if __name__ == '__main__':
    main()


__all__ = [
    'DIAGNOSTIC_FORMAT',
    'DIAGNOSTIC_VERSION',
    'UPDATE_TIMING_SECTION_NAMES',
    'build_parser',
    'main',
    'run_v2_td3_timing_diagnostic',
]
