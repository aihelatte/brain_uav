"""Bounded V2 ANN/SNN TD3 timing diagnostic, never a formal curriculum run."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import gc
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import brain_uav.trainers.v2_td3 as v2_td3_module
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.models import V2SNNPolicyActor
from brain_uav.observations import V2ObservationBatch, collate_v2_observations
from brain_uav.scripts.common import DEVICE_CHOICES, resolve_training_device
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig,
    build_v2_stage_engine,
    prepare_v2_stage_initialization,
    v2_bc_lambda,
)
from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch
from brain_uav.trainers.v2_td3 import V2_TD3_UPDATE_TIMING_SECTIONS
from brain_uav.trainers.v2_validation import (
    V2ValidationPool,
    derive_validation_stage_seed,
    generate_v2_validation_pool,
    load_v2_validation_pool,
    save_v2_validation_pool,
)


DIAGNOSTIC_FORMAT = 'v2_td3_timing_diagnostic'
DIAGNOSTIC_VERSION = 5
DIAGNOSTIC_LEVELS = ('easy', 'medium', 'hard')
UPDATE_TIMING_SECTION_NAMES = V2_TD3_UPDATE_TIMING_SECTIONS
COMPILED_NUMERIC_RTOL = 1e-4
COMPILED_NUMERIC_ATOL = 1e-5


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


def _dynamo_unique_graph_count() -> int:
    return int(torch._dynamo.utils.counters['stats'].get('unique_graphs', 0))


def _compile_warmup_batches(
    *,
    pool: V2ValidationPool,
    prepared,
    batch_size: int,
    device: torch.device,
) -> tuple[Any, ...]:
    warmup_env = V2StaticNoFlyTrajectoryEnv(
        prepared.scenario_config,
        prepared.reward_config,
        seed=pool.stage_seed,
        fixed_scenarios=[record['payload'] for record in pool.scenarios],
        uav_collision_radius=prepared.uav_collision_radius,
    )
    observations = tuple(
        warmup_env.reset(options={'scenario': record['payload']})[0]
        for record in pool.scenarios
    )
    batches = [
        collate_v2_observations([observation] * batch_size).to(device)
        for observation in observations
    ]
    if len(observations) > 1:
        mixed = [
            observations[index % len(observations)]
            for index in range(batch_size)
        ]
        batches.append(collate_v2_observations(mixed).to(device))
    return tuple(batches)


def _warmup_online_critic_normal_only(
    engine,
    batches: Sequence[V2ObservationBatch],
) -> None:
    """Compile ordinary critic forwards without exercising the frozen path."""

    warmup_batches = tuple(batches)
    if not warmup_batches:
        raise ValueError('At least one compile warmup batch is required.')
    critic_parameters = tuple(engine.critic1.parameters()) + tuple(
        engine.critic2.parameters()
    )
    requires_grad = tuple(parameter.requires_grad for parameter in critic_parameters)
    gradient_state = tuple(
        (
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in critic_parameters
    )
    torch_rng_state = torch.random.get_rng_state()
    numpy_rng_state = np.random.get_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if engine.device.type == 'cuda' else None
    )
    counts_before = (
        engine.update_count,
        engine.critic_update_count,
        engine.critic_target_update_count,
        engine.actor_update_count,
        engine.last_total_steps,
    )
    try:
        for batch in warmup_batches:
            device_batch = batch.to(engine.device)
            shared_relations = engine._build_shared_relations(device_batch)
            action = torch.zeros(
                (device_batch.batch_size, engine.action_dim),
                dtype=torch.float32,
                device=engine.device,
            )
            engine.critic_optimizer.zero_grad(set_to_none=True)
            critic_sum = engine.critic1(
                device_batch,
                action,
                shared_relations=shared_relations,
            ).sum() + engine.critic2(
                device_batch,
                action,
                shared_relations=shared_relations,
            ).sum()
            critic_sum.backward()
            engine.critic_optimizer.zero_grad(set_to_none=True)
    finally:
        for parameter, required, (original_grad, saved_grad) in zip(
            critic_parameters,
            requires_grad,
            gradient_state,
        ):
            parameter.requires_grad_(required)
            if original_grad is None:
                parameter.grad = None
            else:
                original_grad.copy_(saved_grad)
                parameter.grad = original_grad
        torch.random.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
    counts_after = (
        engine.update_count,
        engine.critic_update_count,
        engine.critic_target_update_count,
        engine.actor_update_count,
        engine.last_total_steps,
    )
    if counts_after != counts_before:
        raise RuntimeError('Compile warmup must not change TD3 update counters.')


def _configure_group_compilation(
    engine,
    group: str,
    batches: Sequence[V2ObservationBatch],
) -> list[str]:
    group_name = str(group).upper()
    if group_name not in ('A', 'B', 'C'):
        raise ValueError('compiled numerics group must be A, B, or C.')
    enabled = list(engine.enable_online_critic_encoder_compile(
        backend='inductor', mode='default', fullgraph=True, dynamic=True,
    ) or ('critic1.zone_set_encoder', 'critic2.zone_set_encoder'))
    if group_name == 'A':
        _warmup_online_critic_normal_only(engine, batches)
    else:
        engine.warmup_online_critic_encoder_compile(batches)
    if group_name == 'C':
        targets = engine.enable_target_encoder_compile(
            backend='inductor', mode='default', fullgraph=True, dynamic=True,
        ) or (
            'actor_target.zone_set_encoder',
            'critic1_target.zone_set_encoder',
            'critic2_target.zone_set_encoder',
        )
        enabled.extend(targets)
        engine.warmup_target_encoder_compile(batches)
    return enabled


def _compare_compiled_numeric_tensors(
    reference: Mapping[str, torch.Tensor],
    compiled: Mapping[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if reference.keys() != compiled.keys():
        raise AssertionError('Compiled numeric snapshot keys do not match reference.')
    if not reference:
        raise ValueError('Compiled numeric snapshots must not be empty.')
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for name, expected in reference.items():
        actual = compiled[name]
        if expected.shape != actual.shape:
            raise AssertionError(
                f'Compiled numeric mismatch for {name}: shapes differ.'
            )
        if expected.dtype != actual.dtype:
            raise AssertionError(
                f'Compiled numeric mismatch for {name}: dtypes differ.'
            )
        difference = (actual - expected).abs()
        if difference.numel():
            maximum_absolute_error = max(
                maximum_absolute_error,
                float(difference.max().item()),
            )
            denominator = expected.abs().clamp_min(atol)
            maximum_relative_error = max(
                maximum_relative_error,
                float((difference / denominator).max().item()),
            )
        try:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise AssertionError(
                f'Compiled numeric mismatch for {name}: {exc}'
            ) from exc
    return {
        'tensor_count': len(reference),
        'maximum_absolute_error': maximum_absolute_error,
        'maximum_relative_error': maximum_relative_error,
        'rtol': rtol,
        'atol': atol,
    }


def _zero_zone_sample_count(observation: V2ObservationBatch) -> int:
    return int((~observation.presence_mask.any(dim=1)).sum().item())


def _gradient_diagnostic_summary(
    gradient: torch.Tensor | None,
) -> dict[str, Any]:
    if gradient is None:
        return {
            'state': 'none',
            'maximum_absolute_value': None,
            'norm': None,
            'finite': None,
        }
    detached = gradient.detach()
    finite = bool(torch.isfinite(detached).all().item())
    maximum_absolute_value = (
        float(detached.abs().max().item()) if finite and detached.numel() else None
    )
    norm = float(torch.linalg.vector_norm(detached).item()) if finite else None
    state = (
        'zero'
        if maximum_absolute_value == 0.0
        else 'nonzero'
    )
    return {
        'state': state,
        'maximum_absolute_value': maximum_absolute_value,
        'norm': norm,
        'finite': finite,
    }


def _compare_optional_numeric_tensor(
    reference: torch.Tensor | None,
    compiled: torch.Tensor | None,
    *,
    name: str,
    rtol: float,
    atol: float,
) -> float | None:
    if (reference is None) != (compiled is None):
        raise AssertionError(
            f'Compiled numeric mismatch for {name}: one value is None.'
        )
    if reference is None:
        return None
    try:
        torch.testing.assert_close(compiled, reference, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(
            f'Compiled numeric mismatch for {name}: {exc}'
        ) from exc
    difference = (compiled - reference).abs()
    return float(difference.max().item()) if difference.numel() else 0.0


def _numeric_diagnostic_observation_batches(
    batches: Sequence[V2ObservationBatch],
    *,
    device: torch.device,
) -> tuple[V2ObservationBatch, V2ObservationBatch]:
    if not batches:
        raise ValueError('Numeric checking requires compile warmup batches.')
    candidates = tuple(batch.to(device) for batch in batches)
    source = max(candidates, key=lambda batch: batch.max_zone_count)
    if source.batch_size < 2:
        raise ValueError('Numeric checking requires a batch size of at least two.')
    nonempty_rows = source.presence_mask.any(dim=1).nonzero(as_tuple=False)
    if nonempty_rows.numel() == 0:
        raise AssertionError(
            'Numeric diagnostic prerequisite not met: no nonempty scene is available.'
        )
    row = int(nonempty_rows[0, 0].item())
    repeats = (source.batch_size,) + (1,) * (source.ego_features.ndim - 1)
    ego = source.ego_features[row:row + 1].repeat(repeats)
    repeats = (source.batch_size,) + (1,) * (source.goal_features.ndim - 1)
    goal = source.goal_features[row:row + 1].repeat(repeats)
    repeats = (source.batch_size,) + (1,) * (source.zone_features.ndim - 1)
    zones = source.zone_features[row:row + 1].repeat(repeats)
    repeats = (source.batch_size,) + (1,) * (source.presence_mask.ndim - 1)
    mask = source.presence_mask[row:row + 1].repeat(repeats)
    nonempty = V2ObservationBatch(ego, goal, zones, mask)
    mixed_zones = zones.clone()
    mixed_mask = mask.clone()
    mixed_zones[0].zero_()
    mixed_mask[0].zero_()
    mixed = V2ObservationBatch(ego.clone(), goal.clone(), mixed_zones, mixed_mask)
    return mixed, nonempty


def _fixed_numeric_replay_batch(
    observation: V2ObservationBatch,
    *,
    action_dim: int,
    device: torch.device,
) -> V2ReplayBatch:
    observation = observation.to(device)
    batch_size = observation.batch_size
    zeros = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    return V2ReplayBatch(
        obs=observation,
        action=torch.zeros(
            (batch_size, action_dim), dtype=torch.float32, device=device
        ),
        reward=torch.linspace(
            0.25, 0.75, batch_size, dtype=torch.float32, device=device
        ).unsqueeze(-1),
        next_obs=observation,
        done=zeros.clone(),
        success=zeros.clone(),
        near_goal=torch.ones_like(zeros),
        line_to_goal_safe=torch.ones_like(zeros),
    )


_LOCALIZATION_STAGE_ORDER = (
    'target_forward',
    'online_forward_and_loss',
    'backward_pre_clip_gradients',
    'optimizer_step',
)


def _capture_critic_only_update_stages(
    engine,
    batch: V2ReplayBatch,
    *,
    total_steps: int,
    execution: str = 'unspecified',
) -> dict[str, Any]:
    """Capture one real critic-only update with temporary, removable hooks."""

    captures: dict[str, Any] = {
        stage: {} for stage in _LOCALIZATION_STAGE_ORDER
    }
    handles: list[Any] = []
    original_sample = engine.replay.sample
    original_functional = v2_td3_module.F
    parameter_before: dict[str, torch.Tensor] = {}

    def save_forward(stage: str, name: str):
        def hook(_module, _inputs, output):
            captures[stage][name] = output.detach().cpu().clone()
        return hook

    for module, stage, name in (
        (engine.actor_target, 'target_forward', 'actor_target'),
        (engine.critic1_target, 'target_forward', 'critic1_target'),
        (engine.critic2_target, 'target_forward', 'critic2_target'),
        (engine.critic1, 'online_forward_and_loss', 'critic1'),
        (engine.critic2, 'online_forward_and_loss', 'critic2'),
    ):
        handles.append(module.register_forward_hook(save_forward(stage, name)))

    qualified_parameters: list[tuple[str, torch.nn.Parameter]] = []
    for critic_name in ('critic1', 'critic2'):
        critic = getattr(engine, critic_name)
        for parameter_name, parameter in critic.named_parameters():
            qualified_name = f'{critic_name}.{parameter_name}'
            qualified_parameters.append((qualified_name, parameter))
            parameter_before[qualified_name] = parameter.detach().cpu().clone()
            captures['backward_pre_clip_gradients'][qualified_name] = None

            def gradient_hook(gradient, name=qualified_name):
                captures['backward_pre_clip_gradients'][name] = (
                    gradient.detach().cpu().clone()
                )
                return gradient

            handles.append(parameter.register_hook(gradient_hook))

    mse_calls = 0
    original_mse_loss = original_functional.mse_loss

    class _FunctionalProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_functional, name)

        def mse_loss(self, input, target, *args, **kwargs):
            nonlocal mse_calls
            if mse_calls == 0:
                captures['target_forward']['td_target'] = (
                    target.detach().cpu().clone()
                )
            mse_calls += 1
            return original_mse_loss(input, target, *args, **kwargs)

    try:
        engine.replay.sample = lambda batch_size: batch
        v2_td3_module.F = _FunctionalProxy()
        metrics = engine.update_once(total_steps=total_steps, bc_lambda=0.0)
        if metrics.actor_updated or metrics.critic_targets_updated:
            raise AssertionError(
                'Grouped numeric localization requires a critic-only update.'
            )
        captures['online_forward_and_loss']['critic_loss'] = torch.as_tensor(
            metrics.critic_loss,
            dtype=torch.float32,
        )
        for qualified_name, parameter in qualified_parameters:
            prefix = f'{qualified_name}'
            optimizer_state = engine.critic_optimizer.state.get(parameter)
            captures['optimizer_step'][f'parameter_before.{prefix}'] = (
                parameter_before[qualified_name]
            )
            captures['optimizer_step'][f'parameter_after.{prefix}'] = (
                parameter.detach().cpu().clone()
            )
            captures['optimizer_step'][f'adam_exists.{prefix}'] = torch.tensor(
                optimizer_state is not None
            )
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                value = None if optimizer_state is None else optimizer_state.get(field)
                captures['optimizer_step'][f'adam_{field}.{prefix}'] = (
                    None
                    if value is None
                    else torch.as_tensor(value).detach().cpu().clone()
                )
        captures['empty_scene_tokens'] = _empty_token_states(engine)
        captures['batch'] = {
            'batch_size': batch.batch_size,
            'zero_zone_sample_count': _zero_zone_sample_count(batch.obs),
        }
        return captures
    except BaseException as exc:
        print(json.dumps({
            'compiled_numeric_group_execution_error': {
                'execution': execution,
                'exception_type': type(exc).__name__,
                'message': str(exc),
                'captured_stages': [
                    stage for stage in _LOCALIZATION_STAGE_ORDER
                    if any(value is not None for value in captures[stage].values())
                ],
            },
        }, allow_nan=False, ensure_ascii=False), flush=True)
        raise
    finally:
        engine.replay.sample = original_sample
        v2_td3_module.F = original_functional
        for handle in reversed(handles):
            handle.remove()


def _numeric_value_difference(
    reference: torch.Tensor | None,
    compiled: torch.Tensor | None,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, float | None]:
    if (reference is None) != (compiled is None):
        return True, None
    if reference is None:
        return False, None
    if reference.shape != compiled.shape or reference.dtype != compiled.dtype:
        return True, None
    if reference.dtype == torch.bool:
        difference = compiled != reference
        maximum = float(difference.any().item()) if difference.numel() else 0.0
    else:
        difference = (compiled - reference).abs()
        maximum = float(difference.max().item()) if difference.numel() else 0.0
    return not torch.allclose(compiled, reference, rtol=rtol, atol=atol), maximum


def _compare_localization_stages(
    eager: Mapping[str, Mapping[str, torch.Tensor | None]],
    compiled: Mapping[str, Mapping[str, torch.Tensor | None]],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    earliest = None
    for stage in _LOCALIZATION_STAGE_ORDER:
        eager_values = eager.get(stage, {})
        compiled_values = compiled.get(stage, {})
        names = sorted(set(eager_values) | set(compiled_values))
        differences: list[tuple[str, float | None]] = []
        for name in names:
            if name not in eager_values or name not in compiled_values:
                differences.append((name, None))
                continue
            differs, maximum = _numeric_value_difference(
                eager_values[name],
                compiled_values[name],
                rtol=rtol,
                atol=atol,
            )
            if differs:
                differences.append((name, maximum))
        finite_maxima = [value for _, value in differences if value is not None]
        other = [
            item for item in differences if '.empty_scene_token' not in item[0]
        ]
        other_maxima = [value for _, value in other if value is not None]
        stages[stage] = {
            'difference_count': len(differences),
            'maximum_absolute_difference': (
                max(finite_maxima) if finite_maxima else None
            ),
            'first_difference_name': differences[0][0] if differences else None,
            'other_parameter_difference_count': len(other),
            'other_parameter_maximum_absolute_difference': (
                max(other_maxima) if other_maxima else None
            ),
            'other_parameter_first_difference_name': (
                other[0][0] if other else None
            ),
        }
        if differences and earliest is None:
            earliest = stage
    return {
        'passed': earliest is None,
        'earliest_difference_stage': earliest,
        'stages': stages,
        'rtol': rtol,
        'atol': atol,
    }


def _empty_token_localization_record(
    captures: Mapping[str, Any],
    critic_name: str,
) -> dict[str, Any]:
    states = captures.get('empty_scene_tokens', {})
    state = states.get(critic_name)
    if state is None:
        return {}
    prefix = f'{critic_name}.zone_set_encoder.empty_scene_token'
    optimizer = captures['optimizer_step']
    before = optimizer[f'parameter_before.{prefix}']
    after = optimizer[f'parameter_after.{prefix}']
    return {
        'gradient': _gradient_diagnostic_summary(state['gradient']),
        'parameter_delta': _gradient_diagnostic_summary(after - before),
        'adam': _adam_diagnostic_summary(state),
    }


def _report_group_localization(
    group: str,
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = _compare_localization_stages(
        eager,
        compiled,
        rtol=COMPILED_NUMERIC_RTOL,
        atol=COMPILED_NUMERIC_ATOL,
    )
    comparison['group'] = str(group).upper()
    comparison['update_index'] = 1
    comparison['update'] = 'critic_only_with_one_zero_zone_sample'
    comparison['batch'] = eager.get('batch', {})
    comparison['empty_scene_tokens'] = {}
    for critic_name in ('critic1', 'critic2'):
        eager_state = eager.get('empty_scene_tokens', {}).get(critic_name)
        compiled_state = compiled.get('empty_scene_tokens', {}).get(critic_name)
        record = {
            'eager': _empty_token_localization_record(eager, critic_name),
            'compiled': _empty_token_localization_record(compiled, critic_name),
            'differences': {},
        }
        if eager_state is not None and compiled_state is not None:
            differences = _empty_token_pair_differences(
                eager_state,
                compiled_state,
            )
            prefix = f'{critic_name}.zone_set_encoder.empty_scene_token'
            eager_optimizer = eager['optimizer_step']
            compiled_optimizer = compiled['optimizer_step']
            eager_delta = (
                eager_optimizer[f'parameter_after.{prefix}']
                - eager_optimizer[f'parameter_before.{prefix}']
            )
            compiled_delta = (
                compiled_optimizer[f'parameter_after.{prefix}']
                - compiled_optimizer[f'parameter_before.{prefix}']
            )
            differences['parameter_delta_maximum_absolute_difference'] = (
                _optional_maximum_absolute_difference(eager_delta, compiled_delta)
            )
            record['differences'] = differences
        comparison['empty_scene_tokens'][critic_name] = record
    print(json.dumps(
        {'compiled_numeric_group_localization': comparison},
        allow_nan=False,
        ensure_ascii=False,
    ), flush=True)
    if not comparison['passed']:
        raise AssertionError(
            'Compiled numeric group localization first differed at '
            f"{comparison['earliest_difference_stage']}."
        )
    return comparison


def _empty_token_state(engine, critic_name: str) -> dict[str, Any]:
    critic = getattr(engine, critic_name)
    parameter = critic.zone_set_encoder.empty_scene_token
    optimizer_state = engine.critic_optimizer.state.get(parameter)
    return {
        'parameter': parameter.detach().cpu().clone(),
        'gradient': (
            None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        ),
        'adam_exists': optimizer_state is not None,
        'adam_step': (
            None
            if optimizer_state is None or 'step' not in optimizer_state
            else torch.as_tensor(optimizer_state['step']).detach().cpu().clone()
        ),
        'exp_avg': (
            None
            if optimizer_state is None or 'exp_avg' not in optimizer_state
            else optimizer_state['exp_avg'].detach().cpu().clone()
        ),
        'exp_avg_sq': (
            None
            if optimizer_state is None or 'exp_avg_sq' not in optimizer_state
            else optimizer_state['exp_avg_sq'].detach().cpu().clone()
        ),
    }


def _empty_token_states(engine) -> dict[str, dict[str, Any]]:
    return {
        critic_name: _empty_token_state(engine, critic_name)
        for critic_name in ('critic1', 'critic2')
    }


def _optional_maximum_absolute_difference(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    difference = (left - right).abs()
    return float(difference.max().item()) if difference.numel() else 0.0


def _adam_diagnostic_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    step = state['adam_step']
    return {
        'exists': bool(state['adam_exists']),
        'step': None if step is None else float(step.item()),
        'exp_avg': _gradient_diagnostic_summary(state['exp_avg']),
        'exp_avg_sq': _gradient_diagnostic_summary(state['exp_avg_sq']),
    }


def _empty_token_execution_record(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        'before': {
            'gradient': _gradient_diagnostic_summary(before['gradient']),
            'adam': _adam_diagnostic_summary(before),
        },
        'after': {
            'gradient': _gradient_diagnostic_summary(after['gradient']),
            'adam': _adam_diagnostic_summary(after),
        },
        'parameter_delta': _gradient_diagnostic_summary(
            after['parameter'] - before['parameter']
        ),
    }


def _empty_token_pair_differences(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> dict[str, float | None]:
    return {
        'parameter_maximum_absolute_difference': (
            _optional_maximum_absolute_difference(
                eager['parameter'], compiled['parameter']
            )
        ),
        'gradient_maximum_absolute_difference': (
            _optional_maximum_absolute_difference(
                eager['gradient'], compiled['gradient']
            )
        ),
        'adam_step_maximum_absolute_difference': (
            _optional_maximum_absolute_difference(
                eager['adam_step'], compiled['adam_step']
            )
        ),
        'exp_avg_maximum_absolute_difference': (
            _optional_maximum_absolute_difference(
                eager['exp_avg'], compiled['exp_avg']
            )
        ),
        'exp_avg_sq_maximum_absolute_difference': (
            _optional_maximum_absolute_difference(
                eager['exp_avg_sq'], compiled['exp_avg_sq']
            )
        ),
    }


def _historical_momentum_prerequisite_met(
    eager_after: Mapping[str, Mapping[str, Any]],
    compiled_after: Mapping[str, Mapping[str, Any]],
) -> bool:
    for states in (eager_after, compiled_after):
        for critic_name in ('critic1', 'critic2'):
            gradient = _gradient_diagnostic_summary(
                states[critic_name]['gradient']
            )
            momentum = _gradient_diagnostic_summary(
                states[critic_name]['exp_avg']
            )
            if not (
                gradient['state'] == 'nonzero'
                and gradient['finite'] is True
                and momentum['state'] == 'nonzero'
                and momentum['finite'] is True
            ):
                return False
    return True


def _empty_token_update_record(
    *,
    update_index: int,
    update: str,
    batch_construction: str,
    batch: V2ReplayBatch,
    eager_before: Mapping[str, Mapping[str, Any]],
    eager_after: Mapping[str, Mapping[str, Any]],
    compiled_before: Mapping[str, Mapping[str, Any]],
    compiled_after: Mapping[str, Mapping[str, Any]],
    require_historical_momentum: bool,
) -> dict[str, Any]:
    prerequisite_met = (
        _historical_momentum_prerequisite_met(eager_after, compiled_after)
        if require_historical_momentum
        else None
    )
    critics: dict[str, Any] = {}
    for critic_name in ('critic1', 'critic2'):
        eager_before_state = eager_before[critic_name]
        eager_after_state = eager_after[critic_name]
        compiled_before_state = compiled_before[critic_name]
        compiled_after_state = compiled_after[critic_name]
        critics[critic_name] = {
            'eager': _empty_token_execution_record(
                eager_before_state, eager_after_state
            ),
            'compiled': _empty_token_execution_record(
                compiled_before_state, compiled_after_state
            ),
            'before_differences': _empty_token_pair_differences(
                eager_before_state, compiled_before_state
            ),
            'after_differences': _empty_token_pair_differences(
                eager_after_state, compiled_after_state
            ),
        }
    return {
        'update_index': update_index,
        'update': update,
        'batch_construction': batch_construction,
        'batch_size': batch.batch_size,
        'zero_zone_sample_count': _zero_zone_sample_count(batch.obs),
        'historical_momentum_prerequisite_met': prerequisite_met,
        'critics': critics,
    }


def _compare_empty_token_states(
    eager: Mapping[str, Mapping[str, Any]],
    compiled: Mapping[str, Mapping[str, Any]],
    *,
    rtol: float,
    atol: float,
) -> None:
    for critic_name in ('critic1', 'critic2'):
        eager_state = eager[critic_name]
        compiled_state = compiled[critic_name]
        if eager_state['adam_exists'] != compiled_state['adam_exists']:
            raise AssertionError(
                f'Compiled numeric mismatch for {critic_name}.empty_scene_token.'
                'adam.exists.'
            )
        for field in (
            'parameter', 'gradient', 'adam_step', 'exp_avg', 'exp_avg_sq',
        ):
            _compare_optional_numeric_tensor(
                eager_state[field],
                compiled_state[field],
                name=f'{critic_name}.empty_scene_token.{field}',
                rtol=rtol,
                atol=atol,
            )


def _report_and_validate_empty_token_update(
    *,
    update_index: int,
    update: str,
    batch_construction: str,
    batch: V2ReplayBatch,
    eager_before: Mapping[str, Mapping[str, Any]],
    eager_after: Mapping[str, Mapping[str, Any]],
    compiled_before: Mapping[str, Mapping[str, Any]],
    compiled_after: Mapping[str, Mapping[str, Any]],
    require_historical_momentum: bool,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    diagnostic_record = _empty_token_update_record(
        update_index=update_index,
        update=update,
        batch_construction=batch_construction,
        batch=batch,
        eager_before=eager_before,
        eager_after=eager_after,
        compiled_before=compiled_before,
        compiled_after=compiled_after,
        require_historical_momentum=require_historical_momentum,
    )
    print(json.dumps(
        {'compiled_numeric_empty_token': diagnostic_record},
        allow_nan=False,
        ensure_ascii=False,
    ), flush=True)
    if require_historical_momentum and not diagnostic_record[
        'historical_momentum_prerequisite_met'
    ]:
        raise AssertionError(
            'Numeric diagnostic prerequisite not met: the first update did not '
            'produce nonzero empty_scene_token gradients and Adam momentum for '
            'both critics and execution paths.'
        )
    _compare_empty_token_states(
        eager_after,
        compiled_after,
        rtol=rtol,
        atol=atol,
    )
    return diagnostic_record


def _numeric_update_snapshot(engine, batch: V2ReplayBatch, metrics) -> dict[str, torch.Tensor]:
    snapshot = {
        f'metrics.{name}': torch.as_tensor(value, dtype=torch.float32).cpu()
        for name, value in asdict(metrics).items()
    }
    for model_name in (
        'actor', 'critic1', 'critic2',
        'actor_target', 'critic1_target', 'critic2_target',
    ):
        model = getattr(engine, model_name)
        for parameter_name, parameter in model.named_parameters():
            snapshot[f'parameters.{model_name}.{parameter_name}'] = (
                parameter.detach().cpu().clone()
            )
    for critic_name in ('critic1', 'critic2'):
        critic = getattr(engine, critic_name)
        for parameter_name, parameter in critic.named_parameters():
            if parameter.grad is None:
                raise AssertionError(
                    f'Numeric check expected gradient for {critic_name}.{parameter_name}.'
                )
            snapshot[f'gradients.{critic_name}.{parameter_name}'] = (
                parameter.grad.detach().cpu().clone()
            )
    with torch.no_grad():
        next_observation = batch.next_obs.to(engine.device)
        shared_relations = engine._build_shared_relations(next_observation)
        target_action = engine.actor_target(
            next_observation,
            shared_relations=shared_relations,
        )
        target_q1 = engine.critic1_target(
            next_observation,
            target_action,
            shared_relations=shared_relations,
        )
        target_q2 = engine.critic2_target(
            next_observation,
            target_action,
            shared_relations=shared_relations,
        )
    snapshot['forward.actor_target'] = target_action.detach().cpu().clone()
    snapshot['forward.critic1_target'] = target_q1.detach().cpu().clone()
    snapshot['forward.critic2_target'] = target_q2.detach().cpu().clone()
    return snapshot


def _run_compiled_numerics_check(
    *,
    pool: V2ValidationPool,
    prepared,
    formal_config: V2FormalTrainingConfig,
    bc_checkpoint: Path,
    device: torch.device,
    snn_time_window: int,
) -> dict[str, Any]:
    torch_rng_state = torch.random.get_rng_state()
    numpy_rng_state = np.random.get_state()
    cuda_rng_state = torch.cuda.get_rng_state_all()
    started = perf_counter()
    reference_components = None
    compiled_components = None
    reference = None
    compiled = None
    fixed_batch = None
    diagnostic_batches = None
    try:
        reference_components = build_v2_stage_engine(
            None,
            formal_config,
            init_checkpoint=bc_checkpoint,
            rewards=None,
            uav_collision_radius=None,
            device=device,
            model_type='ann',
            snn_time_window=snn_time_window,
            prepared_initialization=prepared,
        )
        compiled_components = build_v2_stage_engine(
            None,
            formal_config,
            init_checkpoint=bc_checkpoint,
            rewards=None,
            uav_collision_radius=None,
            device=device,
            model_type='ann',
            snn_time_window=snn_time_window,
            prepared_initialization=prepared,
        )
        reference = reference_components.engine
        compiled = compiled_components.engine
        compiled.load_checkpoint_state_dict(reference.checkpoint_state_dict())
        compiled.enable_online_critic_encoder_compile(
            backend='inductor', mode='default', fullgraph=True, dynamic=True,
        )
        compiled.enable_target_encoder_compile(
            backend='inductor', mode='default', fullgraph=True, dynamic=True,
        )
        warmup_batches = _compile_warmup_batches(
            pool=pool,
            prepared=prepared,
            batch_size=reference.batch_size,
            device=device,
        )
        compiled.warmup_online_critic_encoder_compile(warmup_batches)
        compiled.warmup_target_encoder_compile(warmup_batches)
        mixed_observation, nonempty_observation = (
            _numeric_diagnostic_observation_batches(
                warmup_batches,
                device=device,
            )
        )
        diagnostic_batches = (
            _fixed_numeric_replay_batch(
                mixed_observation,
                action_dim=reference.action_dim,
                device=device,
            ),
            _fixed_numeric_replay_batch(
                nonempty_observation,
                action_dim=reference.action_dim,
                device=device,
            ),
        )
        update_results: dict[str, Any] = {}
        maximum_absolute_error = 0.0
        maximum_relative_error = 0.0
        for (
            update_index,
            label,
            diagnostic_label,
            batch_construction,
            total_steps,
            fixed_batch,
            require_momentum,
        ) in (
            (
                1,
                'critic_only',
                'critic_only_with_empty_scene',
                'synthetic_from_fixed_pool_with_one_zero_zone_sample',
                1,
                diagnostic_batches[0],
                True,
            ),
            (
                2,
                'actor_and_target_updated',
                'actor_and_target_updated_nonempty',
                'fixed_pool_nonempty_sample_repeated',
                formal_config.policy_delay,
                diagnostic_batches[1],
                False,
            ),
        ):
            reference.replay.sample = lambda batch_size, batch=fixed_batch: batch
            compiled.replay.sample = lambda batch_size, batch=fixed_batch: batch
            eager_before = _empty_token_states(reference)
            compiled_before = _empty_token_states(compiled)
            update_rng_state = torch.random.get_rng_state()
            update_cuda_rng_state = torch.cuda.get_rng_state_all()
            reference_metrics = reference.update_once(
                total_steps=total_steps,
                bc_lambda=v2_bc_lambda(total_steps - 1),
            )
            torch.random.set_rng_state(update_rng_state)
            torch.cuda.set_rng_state_all(update_cuda_rng_state)
            compiled_metrics = compiled.update_once(
                total_steps=total_steps,
                bc_lambda=v2_bc_lambda(total_steps - 1),
            )
            eager_after = _empty_token_states(reference)
            compiled_after = _empty_token_states(compiled)
            if label == 'critic_only' and (
                reference_metrics.actor_updated or compiled_metrics.actor_updated
            ):
                raise AssertionError('Numeric critic-only update unexpectedly updated actor.')
            if label == 'actor_and_target_updated' and not (
                reference_metrics.actor_updated and compiled_metrics.actor_updated
                and reference_metrics.critic_targets_updated
                and compiled_metrics.critic_targets_updated
            ):
                raise AssertionError('Numeric actor/target update did not execute fully.')
            diagnostic_record = _report_and_validate_empty_token_update(
                update_index=update_index,
                update=diagnostic_label,
                batch_construction=batch_construction,
                batch=fixed_batch,
                eager_before=eager_before,
                eager_after=eager_after,
                compiled_before=compiled_before,
                compiled_after=compiled_after,
                require_historical_momentum=require_momentum,
                rtol=COMPILED_NUMERIC_RTOL,
                atol=COMPILED_NUMERIC_ATOL,
            )
            comparison = _compare_compiled_numeric_tensors(
                _numeric_update_snapshot(reference, fixed_batch, reference_metrics),
                _numeric_update_snapshot(compiled, fixed_batch, compiled_metrics),
                rtol=COMPILED_NUMERIC_RTOL,
                atol=COMPILED_NUMERIC_ATOL,
            )
            update_results[label] = comparison
            maximum_absolute_error = max(
                maximum_absolute_error, comparison['maximum_absolute_error']
            )
            maximum_relative_error = max(
                maximum_relative_error, comparison['maximum_relative_error']
            )
        return {
            'requested': True,
            'passed': True,
            'device': str(device),
            'rtol': COMPILED_NUMERIC_RTOL,
            'atol': COMPILED_NUMERIC_ATOL,
            'maximum_absolute_error': maximum_absolute_error,
            'maximum_relative_error': maximum_relative_error,
            'updates': update_results,
            'wall_seconds': perf_counter() - started,
            'measurement_note': (
                'Uses independent eager and compiled engines before timed diagnosis; '
                'this check time is excluded from compile warmup and stable timing.'
            ),
        }
    finally:
        fixed_batch = None
        diagnostic_batches = None
        reference = None
        compiled = None
        reference_components = None
        compiled_components = None
        gc.collect()
        torch.random.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_state)


def _run_grouped_compiled_numerics_diagnostic(
    *,
    group: str,
    pool: V2ValidationPool,
    prepared,
    formal_config: V2FormalTrainingConfig,
    bc_checkpoint: Path,
    device: torch.device,
    snn_time_window: int,
) -> dict[str, Any]:
    """Run one isolated eager/compiled critic-only localization update."""

    group_name = str(group).upper()
    if group_name not in ('A', 'B', 'C'):
        raise ValueError('compiled numerics group must be A, B, or C.')
    torch_rng_state = torch.random.get_rng_state()
    numpy_rng_state = np.random.get_state()
    cuda_rng_state = torch.cuda.get_rng_state_all()
    eager_components = None
    compiled_components = None
    eager = None
    compiled = None
    fixed_batch = None
    try:
        eager_components = build_v2_stage_engine(
            None,
            formal_config,
            init_checkpoint=bc_checkpoint,
            rewards=None,
            uav_collision_radius=None,
            device=device,
            model_type='ann',
            snn_time_window=snn_time_window,
            prepared_initialization=prepared,
        )
        compiled_components = build_v2_stage_engine(
            None,
            formal_config,
            init_checkpoint=bc_checkpoint,
            rewards=None,
            uav_collision_radius=None,
            device=device,
            model_type='ann',
            snn_time_window=snn_time_window,
            prepared_initialization=prepared,
        )
        eager = eager_components.engine
        compiled = compiled_components.engine
        compiled.load_checkpoint_state_dict(eager.checkpoint_state_dict())
        warmup_batches = _compile_warmup_batches(
            pool=pool,
            prepared=prepared,
            batch_size=eager.batch_size,
            device=device,
        )
        mixed_observation, _ = _numeric_diagnostic_observation_batches(
            warmup_batches,
            device=device,
        )
        compilation_batches = tuple(warmup_batches) + (mixed_observation,)
        enabled_objects = _configure_group_compilation(
            compiled,
            group_name,
            compilation_batches,
        )
        fixed_batch = _fixed_numeric_replay_batch(
            mixed_observation,
            action_dim=eager.action_dim,
            device=device,
        )
        update_torch_rng_state = torch.random.get_rng_state()
        update_numpy_rng_state = np.random.get_state()
        update_cuda_rng_state = torch.cuda.get_rng_state_all()
        eager_capture = _capture_critic_only_update_stages(
            eager,
            fixed_batch,
            total_steps=1,
            execution='eager',
        )
        torch.random.set_rng_state(update_torch_rng_state)
        np.random.set_state(update_numpy_rng_state)
        torch.cuda.set_rng_state_all(update_cuda_rng_state)
        compiled_capture = _capture_critic_only_update_stages(
            compiled,
            fixed_batch,
            total_steps=1,
            execution='compiled',
        )
        comparison = _report_group_localization(
            group_name,
            eager_capture,
            compiled_capture,
        )
        return {
            'requested': True,
            'group': group_name,
            'passed': comparison['passed'],
            'device': str(device),
            'compiled_objects': enabled_objects,
            'batch_size': fixed_batch.batch_size,
            'zero_zone_sample_count': _zero_zone_sample_count(fixed_batch.obs),
            'earliest_difference_stage': comparison[
                'earliest_difference_stage'
            ],
            'rtol': COMPILED_NUMERIC_RTOL,
            'atol': COMPILED_NUMERIC_ATOL,
            'measurement_note': (
                'Localization-only run in an independent process; compilation, '
                'warmup, and this update are not stable timing samples.'
            ),
        }
    finally:
        fixed_batch = None
        eager = None
        compiled = None
        eager_components = None
        compiled_components = None
        gc.collect()
        torch.random.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_state)


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
    compile_critic_encoder: bool = False,
    compile_target_encoders: bool = False,
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
    compile_metadata: dict[str, Any] = {
        'requested': bool(compile_critic_encoder),
        'target_encoders_requested': bool(compile_target_encoders),
        'enabled_objects': [],
        'backend': 'inductor',
        'mode': 'default',
        'fullgraph': True,
        'dynamic': True,
        'registration_wall_seconds': 0.0,
        'warmup_wall_seconds': 0.0,
        'online_registration_wall_seconds': 0.0,
        'online_warmup_wall_seconds': 0.0,
        'target_registration_wall_seconds': 0.0,
        'target_warmup_wall_seconds': 0.0,
        'warmup_batch_shapes': [],
        'target_warmup_batch_shapes': [],
        'measurement_graph_count_before': None,
        'measurement_graph_count_after': None,
        'measurement_new_graph_count': None,
        'stable_timing': None,
        'valid_for_speed_comparison': None,
        'measurement_note': 'Critic encoder compilation is disabled.',
    }
    if compile_critic_encoder:
        registration_started = perf_counter()
        enabled_objects = engine.enable_online_critic_encoder_compile(
            backend='inductor',
            mode='default',
            fullgraph=True,
            dynamic=True,
        )
        online_registration_seconds = (
            perf_counter() - registration_started
        )
        compile_metadata['online_registration_wall_seconds'] = (
            online_registration_seconds
        )
        compile_metadata['registration_wall_seconds'] += online_registration_seconds
        compile_metadata['enabled_objects'].extend(enabled_objects)
        if compile_target_encoders:
            target_registration_started = perf_counter()
            target_objects = engine.enable_target_encoder_compile(
                backend='inductor',
                mode='default',
                fullgraph=True,
                dynamic=True,
            )
            target_registration_seconds = (
                perf_counter() - target_registration_started
            )
            compile_metadata['target_registration_wall_seconds'] = (
                target_registration_seconds
            )
            compile_metadata['registration_wall_seconds'] += (
                target_registration_seconds
            )
            compile_metadata['enabled_objects'].extend(target_objects)
        warmup_batches = _compile_warmup_batches(
            pool=pool,
            prepared=prepared,
            batch_size=engine.batch_size,
            device=device,
        )
        compile_metadata['warmup_batch_shapes'] = [
            [batch.batch_size, int(batch.zone_features.shape[1])]
            for batch in warmup_batches
        ]
        warmup_started = perf_counter()
        engine.warmup_online_critic_encoder_compile(warmup_batches)
        online_warmup_seconds = perf_counter() - warmup_started
        compile_metadata['online_warmup_wall_seconds'] = online_warmup_seconds
        compile_metadata['warmup_wall_seconds'] += online_warmup_seconds
        if compile_target_encoders:
            compile_metadata['target_warmup_batch_shapes'] = [
                [batch.batch_size, int(batch.zone_features.shape[1])]
                for batch in warmup_batches
            ]
            target_warmup_started = perf_counter()
            engine.warmup_target_encoder_compile(warmup_batches)
            target_warmup_seconds = perf_counter() - target_warmup_started
            compile_metadata['target_warmup_wall_seconds'] = target_warmup_seconds
            compile_metadata['warmup_wall_seconds'] += target_warmup_seconds
        compile_metadata['measurement_note'] = (
            'Registration and compile-triggering pure-compute warmup are excluded '
            'from measured update wall time. Measurement is valid for speed '
            'comparison only if no new Dynamo graphs appear after the warmup boundary.'
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
                if compile_critic_encoder:
                    compile_metadata['measurement_graph_count_before'] = (
                        _dynamo_unique_graph_count()
                    )
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
    if compile_critic_encoder:
        graph_count_after = _dynamo_unique_graph_count()
        graph_count_before = compile_metadata['measurement_graph_count_before']
        new_graph_count = graph_count_after - graph_count_before
        stable_timing = new_graph_count == 0
        compile_metadata['measurement_graph_count_after'] = graph_count_after
        compile_metadata['measurement_new_graph_count'] = new_graph_count
        compile_metadata['stable_timing'] = stable_timing
        compile_metadata['valid_for_speed_comparison'] = stable_timing
        if not stable_timing:
            compile_metadata['measurement_note'] = (
                'New Dynamo graphs were compiled during measurement. This level is '
                'not stable timing and must not be used for speedup comparison.'
            )
    profiler_overhead_included = bool(profiler_metadata['enabled'])
    compile_timing_valid = compile_metadata['valid_for_speed_comparison'] is not False
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
        'critic_encoder_compile': compile_metadata,
        'timing': {
            'total_wall_seconds': measured_total,
            'total_wall_seconds_includes_detailed_profiler_overhead': (
                profiler_overhead_included
            ),
            'total_wall_seconds_note': total_wall_seconds_note,
            'throughput_environment_steps_per_second': (
                measured_steps / measured_total
                if (
                    not profiler_overhead_included
                    and compile_timing_valid
                    and measured_total > 0.0
                )
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
    compile_critic_encoder: bool = False,
    compile_target_encoders: bool = False,
    check_compiled_numerics: bool = False,
    compiled_numerics_group: str | None = None,
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
    if type(compile_critic_encoder) is not bool:
        raise TypeError('compile_critic_encoder must be a bool.')
    if type(compile_target_encoders) is not bool:
        raise TypeError('compile_target_encoders must be a bool.')
    if type(check_compiled_numerics) is not bool:
        raise TypeError('check_compiled_numerics must be a bool.')
    if compiled_numerics_group is not None:
        if type(compiled_numerics_group) is not str:
            raise TypeError('compiled_numerics_group must be a string or None.')
        compiled_numerics_group = compiled_numerics_group.upper()
        if compiled_numerics_group not in ('A', 'B', 'C'):
            raise ValueError('compiled_numerics_group must be A, B, or C.')
        if (
            compile_critic_encoder
            or compile_target_encoders
            or check_compiled_numerics
            or profiler_updates
        ):
            raise ValueError(
                'compiled_numerics_group is an isolated mode and cannot be '
                'combined with compile/profiler timing options.'
            )
    if compile_target_encoders and not compile_critic_encoder:
        raise ValueError(
            'compile_target_encoders requires compile_critic_encoder.'
        )
    if check_compiled_numerics and not compile_target_encoders:
        raise ValueError(
            'check_compiled_numerics requires compile_target_encoders.'
        )
    if compile_critic_encoder and profiler_updates:
        raise ValueError(
            'compile_critic_encoder cannot be combined with detailed profiler.'
        )
    if compile_critic_encoder and model != 'ann':
        raise ValueError('compile_critic_encoder is limited to the ANN diagnostic.')
    if compiled_numerics_group is not None and model != 'ann':
        raise ValueError('compiled_numerics_group is limited to the ANN diagnostic.')
    if steps < pool_count:
        raise ValueError('steps_per_level must be at least scenario_count.')
    run_seed = _nonnegative_int(seed, name='seed')
    pool_seed = _nonnegative_int(validation_seed, name='validation_seed')
    if type(snn_time_window) is not int or snn_time_window <= 0:
        raise ValueError('snn_time_window must be a positive integer.')
    requested_device = device
    resolved_device = resolve_training_device(device)
    target_device = torch.device(resolved_device)
    if check_compiled_numerics and target_device.type != 'cuda':
        raise ValueError('check_compiled_numerics requires a CUDA diagnostic.')
    if compiled_numerics_group is not None and target_device.type != 'cuda':
        raise ValueError('compiled_numerics_group requires a CUDA diagnostic.')
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
    if compiled_numerics_group is not None:
        localization = _run_grouped_compiled_numerics_diagnostic(
            group=compiled_numerics_group,
            pool=pools['easy'],
            prepared=prepared,
            formal_config=config,
            bc_checkpoint=checkpoint,
            device=target_device,
            snn_time_window=snn_time_window,
        )
        output.mkdir(parents=True, exist_ok=False)
        summary = {
            'format': DIAGNOSTIC_FORMAT,
            'format_version': DIAGNOSTIC_VERSION,
            'formal_stage_passed': False,
            'purpose': 'compiled_numeric_group_localization_only',
            'model': model,
            'requested_device': requested_device,
            'resolved_device': resolved_device,
            'seed': run_seed,
            'validation_seed': pool_seed,
            'bc_checkpoint': str(checkpoint.resolve()),
            'scenario_pool_directory': str(Path(scenario_pool_dir).resolve()),
            'scenario_pool_prepare_wall_seconds': pool_prepare_seconds,
            'compiled_numeric_group_localization': localization,
            'timing_levels_executed': 0,
        }
        summary = json.loads(json.dumps(
            summary,
            allow_nan=False,
            ensure_ascii=False,
        ))
        _strict_json_write(output / 'diagnostic_summary.json', summary)
        print(json.dumps(summary, allow_nan=False, ensure_ascii=False), flush=True)
        return summary
    compiled_numerics = {
        'requested': False,
        'passed': None,
        'device': None,
        'rtol': COMPILED_NUMERIC_RTOL,
        'atol': COMPILED_NUMERIC_ATOL,
        'maximum_absolute_error': None,
        'maximum_relative_error': None,
        'updates': {},
        'wall_seconds': 0.0,
        'measurement_note': 'Compiled numeric checking is disabled.',
    }
    if check_compiled_numerics:
        compiled_numerics = _run_compiled_numerics_check(
            pool=pools['easy'],
            prepared=prepared,
            formal_config=config,
            bc_checkpoint=checkpoint,
            device=target_device,
            snn_time_window=snn_time_window,
        )
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
            compile_critic_encoder=compile_critic_encoder,
            compile_target_encoders=compile_target_encoders,
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
        'compiled_numerics': compiled_numerics,
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
            'compile_critic_encoder_requested': compile_critic_encoder,
            'compile_target_encoders_requested': compile_target_encoders,
            'check_compiled_numerics_requested': check_compiled_numerics,
            'compile_critic_encoder_backend': (
                'inductor' if compile_critic_encoder else None
            ),
            'compile_critic_encoder_mode': (
                'default' if compile_critic_encoder else None
            ),
            'compile_critic_encoder_fullgraph': (
                True if compile_critic_encoder else None
            ),
            'compile_critic_encoder_dynamic': (
                True if compile_critic_encoder else None
            ),
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
    parser.add_argument(
        '--compile-critic-encoder',
        action='store_true',
        help='Compile only critic1/critic2 ZoneSetEncoder tensor computation.',
    )
    parser.add_argument(
        '--compile-target-encoders',
        action='store_true',
        help='Also compile actor/critic target ZoneSetEncoder tensor computation.',
    )
    parser.add_argument(
        '--check-compiled-numerics',
        action='store_true',
        help='Run an isolated CUDA eager-versus-compiled TD3 numeric check.',
    )
    parser.add_argument(
        '--compiled-numerics-group',
        choices=('A', 'B', 'C'),
        default=None,
        help=(
            'Run only grouped CUDA localization A, B, or C, then exit without '
            'the normal timing loop.'
        ),
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
        compile_critic_encoder=args.compile_critic_encoder,
        compile_target_encoders=args.compile_target_encoders,
        check_compiled_numerics=args.check_compiled_numerics,
        compiled_numerics_group=args.compiled_numerics_group,
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
