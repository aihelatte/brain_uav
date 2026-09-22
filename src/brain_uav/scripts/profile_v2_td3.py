"""Bounded V2 ANN/SNN TD3 timing diagnostic, never a formal curriculum run."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import gc
import json
from math import isfinite
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import brain_uav.trainers.v2_td3 as v2_td3_module
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.models import V2SNNPolicyActor
from brain_uav.observations import (
    GOAL_FEATURE_INDEX,
    V2ObservationBatch,
    collate_v2_observations,
)
from brain_uav.scripts.common import DEVICE_CHOICES, resolve_training_device
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig,
    build_v2_stage_engine,
    prepare_v2_stage_initialization,
    v2_bc_lambda,
)
from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch
from brain_uav.trainers.v2_td3 import (
    V2_TD3_UPDATE_DETAIL_SECTIONS,
    V2_TD3_UPDATE_TIMING_SECTIONS,
)
from brain_uav.trainers.v2_validation import (
    V2ValidationPool,
    derive_validation_stage_seed,
    generate_v2_validation_pool,
    load_v2_validation_pool,
    save_v2_validation_pool,
)


DIAGNOSTIC_FORMAT = 'v2_td3_timing_diagnostic'
DIAGNOSTIC_VERSION = 7
DIAGNOSTIC_LEVELS = ('easy', 'medium', 'hard')
UPDATE_TIMING_SECTION_NAMES = V2_TD3_UPDATE_TIMING_SECTIONS
COMPILED_NUMERIC_RTOL = 1e-4
COMPILED_NUMERIC_ATOL = 1e-5
FROZEN_CRITIC_ACTOR_ENCODER_EXECUTION = 'eager'


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


class _NestedUpdateTimingSummary:
    """Aggregate diagnostic child intervals without treating parents as additive."""

    _CRITIC_CHILDREN = ('critic_zero_grad', 'critic_loss_backward')
    _ACTOR_CHILDREN = tuple(
        name for name in V2_TD3_UPDATE_DETAIL_SECTIONS
        if name not in ('critic_zero_grad', 'critic_loss_backward')
    )

    def __init__(self) -> None:
        self.update_count = 0
        self.actor_update_count = 0
        self.parent_wall_seconds = {
            'critic_backward': 0.0,
            'actor_update': 0.0,
        }
        self.wall_seconds = {
            name: 0.0 for name in V2_TD3_UPDATE_DETAIL_SECTIONS
        }
        self.calls = {name: 0 for name in V2_TD3_UPDATE_DETAIL_SECTIONS}

    def record(
        self,
        *,
        actor_updated: bool,
        outer_sections: Mapping[str, float],
        detail: Mapping[str, Mapping[str, float | int]],
    ) -> None:
        for parent in self.parent_wall_seconds:
            value = float(outer_sections[parent])
            if not isfinite(value) or value < 0.0:
                raise RuntimeError('Nested TD3 parent timing must be non-negative.')
            self.parent_wall_seconds[parent] += value
        detail_wall = detail['wall_seconds']
        detail_calls = detail['calls']
        unknown = (set(detail_wall) | set(detail_calls)) - set(self.wall_seconds)
        if unknown:
            raise RuntimeError(f'Unknown nested TD3 timing sections: {sorted(unknown)}.')
        for name in self.wall_seconds:
            value = float(detail_wall.get(name, 0.0))
            calls = int(detail_calls.get(name, 0))
            if not isfinite(value) or value < 0.0 or calls < 0:
                raise RuntimeError('Nested TD3 timing values must be non-negative.')
            self.wall_seconds[name] += value
            self.calls[name] += calls
        self.update_count += 1
        self.actor_update_count += int(actor_updated)

    def _parent_payload(
        self,
        parent: str,
        children: Sequence[str],
        *,
        environment_steps: int,
        actor_parent: bool,
    ) -> dict[str, Any]:
        parent_total = self.parent_wall_seconds[parent]
        event_count = self.actor_update_count if actor_parent else self.update_count
        child_total = sum(self.wall_seconds[name] for name in children)
        return {
            'total_wall_seconds': parent_total,
            'event_count': event_count,
            'average_wall_seconds_per_event': (
                parent_total / event_count if event_count else None
            ),
            'average_wall_seconds_per_environment_step': (
                parent_total / environment_steps if environment_steps else None
            ),
            'other_uncovered_wall_seconds': parent_total - child_total,
            'children_are_mutually_exclusive': True,
            'children': {
                name: {
                    'execution_call_count': self.calls[name],
                    'total_wall_seconds': self.wall_seconds[name],
                    'average_wall_seconds_per_update': (
                        self.wall_seconds[name] / self.update_count
                        if self.update_count else None
                    ),
                    'average_wall_seconds_per_actor_update': (
                        self.wall_seconds[name] / self.actor_update_count
                        if self.actor_update_count else None
                    ),
                    'average_wall_seconds_per_environment_step': (
                        self.wall_seconds[name] / environment_steps
                        if environment_steps else None
                    ),
                    'percent_of_parent_wall_seconds': (
                        100.0 * self.wall_seconds[name] / parent_total
                        if parent_total > 0.0 else None
                    ),
                }
                for name in children
            },
        }

    def to_dict(self, *, environment_steps: int) -> dict[str, Any]:
        steps = _positive_int(environment_steps, name='environment_steps')
        return {
            'update_count': self.update_count,
            'actor_update_count': self.actor_update_count,
            'environment_step_count': steps,
            'critic_backward': self._parent_payload(
                'critic_backward',
                self._CRITIC_CHILDREN,
                environment_steps=steps,
                actor_parent=False,
            ),
            'actor_update': self._parent_payload(
                'actor_update',
                self._ACTOR_CHILDREN,
                environment_steps=steps,
                actor_parent=True,
            ),
            'measurement_note': (
                'Child intervals are mutually exclusive within each parent. Parent '
                'and child times must not be added together. Wall intervals use the '
                'CPU clock without per-section CUDA synchronization, so asynchronous '
                'CUDA waits may be charged to a later interval.'
            ),
        }


class _EnvironmentGeometryDiagnostic:
    """Scoped environment phase and geometry-call recorder for this script."""

    _STEP_SECTIONS = (
        'dynamics_position_progress',
        'termination',
        'current_point_clearance',
        'reward',
        'observation_construction',
        'info_construction',
    )

    def __init__(
        self,
        *,
        duplicate_audit_steps: int = 8,
        duplicate_example_limit: int = 5,
    ) -> None:
        self.duplicate_audit_steps = _nonnegative_int(
            duplicate_audit_steps,
            name='duplicate_audit_steps',
        )
        self.duplicate_example_limit = _nonnegative_int(
            duplicate_example_limit,
            name='duplicate_example_limit',
        )
        self.step_count = 0
        self.reset_count = 0
        self._step_started: float | None = None
        self._reset_started: float | None = None
        self._step_total = 0.0
        self._reset_total = 0.0
        self._section_wall = defaultdict(float)
        self._section_calls = defaultdict(int)
        self._source_stack: list[str] = []
        self._operation_stack: list[str] = []
        self._geometry: dict[str, dict[str, float | int]] = {}
        self._restorations: list[tuple[Any, str, bool, Any]] = []
        self._attached: set[int] = set()
        self._audit_seen: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._audit_cycle: str | None = None
        self._audit_cycle_index: int | None = None
        self._reset_audit_active = False
        self._audited_step_count = 0
        self._audited_reset_count = 0
        self._duplicate_groups = 0
        self._duplicate_calls = 0
        self._duplicate_examples: list[dict[str, Any]] = []
        self._closed = False

    @contextmanager
    def source(self, name: str):
        self._source_stack.append(str(name))
        try:
            yield
        finally:
            self._source_stack.pop()

    @contextmanager
    def section(self, name: str):
        started = perf_counter()
        try:
            with self.source(name):
                yield
        finally:
            self._section_wall[name] += perf_counter() - started
            self._section_calls[name] += 1

    def begin_step(self) -> None:
        if self._step_started is not None:
            raise RuntimeError('Environment diagnostic step is already active.')
        self._step_started = perf_counter()

    def end_step(self) -> None:
        if self._step_started is None:
            raise RuntimeError('Environment diagnostic step is not active.')
        self._step_total += perf_counter() - self._step_started
        self._step_started = None
        self.step_count += 1

    def begin_reset(self) -> None:
        if self._reset_started is not None:
            raise RuntimeError('Environment diagnostic reset is already active.')
        self._reset_audit_active = self._begin_audit_cycle('reset')
        self._reset_started = perf_counter()

    def end_reset(self) -> None:
        if self._reset_started is None:
            raise RuntimeError('Environment diagnostic reset is not active.')
        self._reset_total += perf_counter() - self._reset_started
        self._reset_started = None
        self.reset_count += 1
        if self._reset_audit_active:
            self._end_audit_cycle('reset')
            self._reset_audit_active = False

    def _begin_audit_cycle(self, kind: str) -> bool:
        if self._audit_cycle is not None:
            raise RuntimeError(
                f'Geometry audit cycle {self._audit_cycle!r} is already active.'
            )
        count = (
            self._audited_step_count
            if kind == 'step'
            else self._audited_reset_count
        )
        if count >= self.duplicate_audit_steps:
            return False
        self._audit_cycle = kind
        self._audit_cycle_index = count + 1
        self._audit_seen.clear()
        return True

    def _end_audit_cycle(self, kind: str) -> None:
        if self._audit_cycle != kind:
            raise RuntimeError(
                f'Geometry audit cycle {kind!r} is not active.'
            )
        if kind == 'step':
            self._audited_step_count += 1
        else:
            self._audited_reset_count += 1
        self._audit_seen.clear()
        self._audit_cycle = None
        self._audit_cycle_index = None

    @contextmanager
    def audit_step(self):
        active = self._begin_audit_cycle('step')
        try:
            yield
        finally:
            if active:
                self._end_audit_cycle('step')

    @staticmethod
    def _strict_argument(value: Any) -> tuple[Any, ...]:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return ('repr', type(value).__name__, repr(value))
        if array.ndim == 0:
            scalar = array.item()
            if isinstance(scalar, (bool, int, float, str)):
                return ('scalar', type(scalar).__name__, scalar)
        contiguous = np.ascontiguousarray(array)
        return (
            'array',
            contiguous.dtype.str,
            tuple(contiguous.shape),
            contiguous.tobytes(),
        )

    def _audit_query(
        self,
        owner: Any,
        operation: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        if self._audit_cycle is None:
            return
        zone = owner if hasattr(owner, 'shape') else None
        shape = owner.shape if zone is not None else owner
        key = (
            id(zone) if zone is not None else None,
            id(shape),
            operation,
            tuple(self._strict_argument(value) for value in args),
            tuple(
                (name, self._strict_argument(value))
                for name, value in sorted(kwargs.items())
            ),
        )
        source = self._source_stack[-1] if self._source_stack else 'unattributed'
        existing = self._audit_seen.get(key)
        if existing is None:
            self._audit_seen[key] = {
                'count': 1,
                'sources': [source],
                'shape_type': type(shape).__name__,
                'operation': operation,
            }
            return
        existing['count'] += 1
        if source not in existing['sources']:
            existing['sources'].append(source)
        self._duplicate_calls += 1
        if existing['count'] == 2:
            self._duplicate_groups += 1
            if len(self._duplicate_examples) < self.duplicate_example_limit:
                self._duplicate_examples.append({
                    '_audit_key': key,
                    'audit_cycle': self._audit_cycle,
                    'audit_cycle_index': self._audit_cycle_index,
                    'shape_type': existing['shape_type'],
                    'operation': operation,
                    'sources': list(existing['sources']),
                    'strictly_identical_call_count': existing['count'],
                })
        elif self._duplicate_examples:
            for example in self._duplicate_examples:
                if (
                    example['_audit_key'] == key
                    and example['audit_cycle'] == self._audit_cycle
                    and example['audit_cycle_index'] == self._audit_cycle_index
                ):
                    example['strictly_identical_call_count'] = existing['count']
                    example['sources'] = list(existing['sources'])
                    break

    @contextmanager
    def _query(
        self,
        owner: Any,
        operation: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ):
        source = self._source_stack[-1] if self._source_stack else 'unattributed'
        shape = owner.shape if hasattr(owner, 'shape') else owner
        effective_operation = operation
        if (
            operation == 'segment_clearance'
            and self._operation_stack
            and self._operation_stack[-1] == 'segment_safety'
        ):
            effective_operation = 'exact_segment_clearance_within_safety'
        key = f'{source}|{type(shape).__name__}|{effective_operation}'
        record = self._geometry.setdefault(key, {'calls': 0, 'wall_seconds': 0.0})
        self._audit_query(owner, effective_operation, args, kwargs)
        started = perf_counter()
        self._operation_stack.append(effective_operation)
        try:
            yield
        finally:
            self._operation_stack.pop()
            record['calls'] = int(record['calls']) + 1
            record['wall_seconds'] = float(record['wall_seconds']) + (
                perf_counter() - started
            )

    def _patch_method(self, owner: Any, name: str, operation: str) -> None:
        if not hasattr(owner, name):
            return
        original = getattr(owner, name)
        had_instance_value = name in getattr(owner, '__dict__', {})
        instance_value = owner.__dict__.get(name) if had_instance_value else None

        def wrapped(*args, __original=original, __operation=operation, **kwargs):
            with self._query(owner, __operation, args, kwargs):
                return __original(*args, **kwargs)

        setattr(owner, name, wrapped)
        self._restorations.append((owner, name, had_instance_value, instance_value))

    def attach_zones(self, zones: Sequence[Any]) -> None:
        if self._closed:
            raise RuntimeError('Environment geometry diagnostic is closed.')
        for zone in zones:
            if id(zone) not in self._attached:
                self._attached.add(id(zone))
                self._patch_method(zone, 'point_clearance', 'point_clearance')
                self._patch_method(
                    zone,
                    'point_clearance_and_surface_normal',
                    'point_clearance_and_surface_normal',
                )
                self._patch_method(zone, 'violates_segment', 'segment_safety')
                self._patch_method(zone, 'segment_clearance', 'segment_clearance')
            shape = zone.shape
            if id(shape) not in self._attached:
                self._attached.add(id(shape))
                self._patch_method(shape, 'surface_normal', 'surface_normal')
                self._patch_method(
                    shape,
                    'segment_intersection',
                    'segment_intersection',
                )
                self._patch_method(
                    shape,
                    'segment_clearance',
                    'shape_segment_clearance',
                )

    def close(self) -> None:
        if self._closed:
            return
        for owner, name, had_value, value in reversed(self._restorations):
            if had_value:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self._restorations.clear()
        self._audit_seen.clear()
        self._audit_cycle = None
        self._audit_cycle_index = None
        self._reset_audit_active = False
        self._closed = True

    def to_dict(self) -> dict[str, Any]:
        covered = sum(self._section_wall[name] for name in self._STEP_SECTIONS)
        return {
            'enabled': True,
            'step_count': self.step_count,
            'reset_count': self.reset_count,
            'step_total_wall_seconds': self._step_total,
            'reset_total_wall_seconds': self._reset_total,
            'step_sections': {
                name: {
                    'calls': self._section_calls[name],
                    'total_wall_seconds': self._section_wall[name],
                    'average_wall_seconds_per_step': (
                        self._section_wall[name] / self.step_count
                        if self.step_count else None
                    ),
                }
                for name in self._STEP_SECTIONS
            } | {
                'other_uncovered': {
                    'calls': self.step_count,
                    'total_wall_seconds': self._step_total - covered,
                    'average_wall_seconds_per_step': (
                        (self._step_total - covered) / self.step_count
                        if self.step_count else None
                    ),
                },
            },
            'geometry_queries': {
                key: dict(value) for key, value in sorted(self._geometry.items())
            },
            'duplicate_query_audit': {
                'sampled_step_limit': self.duplicate_audit_steps,
                'audited_step_count': self._audited_step_count,
                'audited_reset_count': self._audited_reset_count,
                'strict_duplicate_group_count': self._duplicate_groups,
                'duplicate_call_count_after_first': self._duplicate_calls,
                'representative_examples': [
                    {
                        name: value for name, value in example.items()
                        if name != '_audit_key'
                    }
                    for example in self._duplicate_examples
                ],
                'comparison_note': (
                    'Identity, complete point/segment arrays, and exact keyword '
                    'arguments are compared without rounding. Calls are reported '
                    'only; no query cache is used.'
                ),
            },
            'parent_child_note': (
                'Geometry parent and child calls are not additive. segment_safety '
                'may return from its AABB test without an '
                'exact_segment_clearance_within_safety child call.'
            ),
        }


class _CompiledPathProfiler:
    """Bounded post-measurement profiler that keeps compiled paths enabled."""

    def __init__(
        self,
        device: torch.device,
        *,
        requested_updates: int,
        output_dir: Path,
        expected_compiled_entries: Sequence[str],
    ) -> None:
        self.device = device
        self.requested_updates = _positive_int(
            requested_updates,
            name='compiled_path_profiler_updates',
        )
        self.output_dir = Path(output_dir)
        self.expected_compiled_entries = tuple(expected_compiled_entries)
        self.captured_updates = 0
        self.compiled_entry_calls: defaultdict[str, int] = defaultdict(int)
        self._profiler = None
        self._closed = False
        self._activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            self._activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._graph_count_before: int | None = None
        self._zone_counts: list[int] = []
        self._bc_effective_updates = 0
        self._terminal_geometry_effective_updates = 0

    def _ensure_started(self) -> None:
        if self._profiler is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._graph_count_before = _dynamo_unique_graph_count()
        self._profiler = torch.profiler.profile(
            activities=self._activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            acc_events=True,
        )
        self._profiler.__enter__()
        self._profiler.toggle_collection_dynamic(False, self._activities)

    def record_compiled_entry(self, name: str) -> None:
        self.compiled_entry_calls[str(name)] += 1

    def record_update_inputs(self, batch: V2ReplayBatch, metrics: Any) -> None:
        counts = batch.obs.presence_mask.sum(dim=1).detach().cpu().tolist()
        self._zone_counts.extend(int(value) for value in counts)
        self._bc_effective_updates += int(
            float(getattr(metrics, 'bc_lambda', 0.0)) > 0.0
            and float(getattr(metrics, 'bc_loss', 0.0)) != 0.0
        )
        self._terminal_geometry_effective_updates += int(
            float(getattr(metrics, 'terminal_geo_lambda', 0.0)) > 0.0
            and float(getattr(metrics, 'terminal_geo_loss', 0.0)) != 0.0
        )

    def run(self, operation: Callable[[], Any]) -> Any:
        if self.captured_updates >= self.requested_updates:
            return None
        self._ensure_started()
        self._profiler.toggle_collection_dynamic(True, self._activities)
        try:
            with torch.profiler.record_function(
                'v2_td3.compiled_path_profiled_update'
            ):
                result = operation()
        finally:
            self._profiler.toggle_collection_dynamic(False, self._activities)
        self.captured_updates += 1
        self._profiler.step()
        return result

    @staticmethod
    def _event_value(event: Any, name: str) -> float | None:
        value = getattr(event, name, None)
        return None if value is None else float(value)

    @classmethod
    def _operation_summary(
        cls,
        averages: Sequence[Any],
        *,
        self_metric: str,
        total_metric: str,
        limit: int = 25,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        available = [
            event for event in averages
            if cls._event_value(event, self_metric) is not None
        ]
        if not available:
            return [], {
                'available': False,
                'operation_count': None,
                'calls': None,
                'self_time_us': None,
                'total_time_us': None,
                'total_time_note': (
                    'Inclusive total is unavailable and is never aggregated '
                    'across parent/child events.'
                ),
            }
        available.sort(
            key=lambda event: cls._event_value(event, self_metric) or 0.0,
            reverse=True,
        )

        def payload(event: Any) -> dict[str, Any]:
            return {
                'name': str(event.key),
                'calls': int(event.count),
                'self_time_us': cls._event_value(event, self_metric),
                'total_time_us': cls._event_value(event, total_metric),
            }

        top = [payload(event) for event in available[:limit]]
        other_events = available[limit:]
        other = {
            'available': True,
            'operation_count': len(other_events),
            'calls': sum(int(event.count) for event in other_events),
            'self_time_us': sum(
                cls._event_value(event, self_metric) or 0.0
                for event in other_events
            ),
            'total_time_us': None,
            'total_time_note': (
                'Inclusive total is not aggregated for other because parent '
                'and child event totals overlap.'
            ),
        }
        return top, other

    @staticmethod
    def _device_type_name(event: Any) -> str | None:
        device_type = getattr(event, 'device_type', None)
        if device_type is None:
            return None
        name = getattr(device_type, 'name', None)
        if name is not None:
            return str(name).upper()
        text = str(device_type).rsplit('.', 1)[-1]
        return text.upper() if text else None

    @classmethod
    def _trace_cuda_device_task_summary(
        cls,
        trace_events: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        category_to_kind = {
            'kernel': 'kernel',
            'gpu_memcpy': 'memcpy',
            'gpu_memset': 'memset',
        }
        grouped: dict[str, dict[str, list[float | None]]] = {
            kind: defaultdict(list) for kind in category_to_kind.values()
        }
        excluded_interval_markers = 0
        for event in trace_events:
            category = event.get('cat')
            if category == 'gpu_user_annotation':
                excluded_interval_markers += 1
                continue
            kind = category_to_kind.get(category)
            if kind is None or event.get('ph') != 'X':
                continue
            duration = event.get('dur')
            try:
                duration_value = float(duration) if duration is not None else None
            except (TypeError, ValueError, OverflowError):
                duration_value = None
            if duration_value is not None and not isfinite(duration_value):
                duration_value = None
            grouped[kind][str(event.get('name', '<unnamed>'))].append(
                duration_value
            )
        available = any(grouped[kind] for kind in grouped)
        result: dict[str, Any] = {
            'available': available,
            'data_source': 'Chrome trace event categories',
            'excluded_interval_markers': excluded_interval_markers,
            'aggregation_note': (
                'Counts and durations use only trace categories kernel, '
                'gpu_memcpy, and gpu_memset. gpu_user_annotation intervals are '
                'excluded. Durations are not GPU busy time or utilization '
                'because streams may overlap.'
            ),
        }
        for kind in ('kernel', 'memcpy', 'memset'):
            operations = grouped[kind]
            if not operations:
                result[kind] = {
                    'available': False,
                    'calls': None,
                    'self_time_us': None,
                    'operations': None,
                }
                continue
            rows = []
            for name, durations in operations.items():
                total = (
                    sum(float(value) for value in durations)
                    if all(value is not None for value in durations)
                    else None
                )
                rows.append({
                    'name': name,
                    'calls': len(durations),
                    'self_time_us': total,
                    'total_time_us': None,
                })
            rows.sort(
                key=lambda row: (
                    row['self_time_us'] is not None,
                    row['self_time_us'] or 0.0,
                ),
                reverse=True,
            )
            all_durations = [
                duration
                for durations in operations.values()
                for duration in durations
            ]
            result[kind] = {
                'available': True,
                'calls': len(all_durations),
                'self_time_us': (
                    sum(float(value) for value in all_durations)
                    if all(value is not None for value in all_durations)
                    else None
                ),
                'operations': rows[:50],
            }
        return result

    def finish(self) -> dict[str, Any]:
        if self._profiler is None:
            raise RuntimeError('Compiled path profiler captured no updates.')
        self._profiler.__exit__(None, None, None)
        self._closed = True
        averages = list(self._profiler.key_averages())
        graph_count_after = _dynamo_unique_graph_count()
        cpu_events = [
            event for event in averages
            if self._device_type_name(event) == 'CPU'
        ]
        cuda_events = [
            event for event in averages
            if self._device_type_name(event) == 'CUDA'
        ]
        cpu_top, cpu_other = self._operation_summary(
            cpu_events,
            self_metric='self_cpu_time_total',
            total_metric='cpu_time_total',
        )
        cpu_associated_cuda_events = [
            event for event in cpu_events
            if (self._event_value(event, 'self_device_time_total') or 0.0) > 0.0
            or (self._event_value(event, 'device_time_total') or 0.0) > 0.0
        ]
        cpu_associated_cuda_top, cpu_associated_cuda_other = (
            self._operation_summary(
                cpu_associated_cuda_events,
                self_metric='self_device_time_total',
                total_metric='device_time_total',
            )
        )
        cuda_top, cuda_other = self._operation_summary(
            cuda_events,
            self_metric='self_device_time_total',
            total_metric='device_time_total',
        )
        cpu_path = self.output_dir / 'compiled_operators_cpu.txt'
        cuda_path = self.output_dir / 'compiled_operators_cuda.txt'
        trace_path = self.output_dir / 'compiled_medium_trace.json'
        report_path = self.output_dir / 'compiled_profile_report.txt'
        device_task_path = self.output_dir / 'compiled_cuda_device_tasks.txt'
        self._profiler.export_chrome_trace(str(trace_path))
        try:
            trace_payload = json.loads(trace_path.read_text(encoding='utf-8'))
            trace_events = trace_payload.get('traceEvents', ())
            if not isinstance(trace_events, list):
                trace_events = ()
        except (OSError, UnicodeError, json.JSONDecodeError):
            trace_events = ()
        cuda_device_tasks = self._trace_cuda_device_task_summary(trace_events)
        cuda_timing_available = bool(
            self.device.type == 'cuda' and cuda_device_tasks['available']
        )

        def format_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
            return [
                f'{row["name"]}\tcalls={row["calls"]}\t'
                f'self_time_us={row["self_time_us"]}\t'
                f'inclusive_time_us={row["total_time_us"]}'
                for row in rows
            ]

        cpu_path.write_text(
            '\n'.join((
                'CPU operations by self CPU time',
                *format_rows(cpu_top),
                f'other_self_time_us={cpu_other["self_time_us"]}; '
                'other_inclusive_time_us=null',
                '',
                'CPU operations by associated device time (attribution only)',
                *format_rows(cpu_associated_cuda_top),
                f'other_self_time_us='
                f'{cpu_associated_cuda_other["self_time_us"]}; '
                'other_inclusive_time_us=null',
            )) + '\n',
            encoding='utf-8',
        )
        cuda_text = (
            '\n'.join(
                f'{kind}: calls={cuda_device_tasks[kind]["calls"]}, '
                f'self_time_us={cuda_device_tasks[kind]["self_time_us"]}'
                for kind in ('kernel', 'memcpy', 'memset')
            ) + '\n'
            if cuda_timing_available
            else (
                'CUDA device events were requested but are unavailable.\n'
                if self.device.type == 'cuda'
                else 'CUDA activity was not requested for this CPU diagnostic.\n'
            )
        )
        cuda_path.write_text(cuda_text, encoding='utf-8')
        device_task_path.write_text(
            (
                '\n'.join(
                    f'{kind}\t{item["name"]}\tcalls={item["calls"]}\t'
                    f'self_time_us={item["self_time_us"]}\t'
                    f'inclusive_time_us={item["total_time_us"]}'
                    for kind in ('kernel', 'memcpy', 'memset')
                    for item in (cuda_device_tasks[kind]['operations'] or ())
                ) + '\n'
                if cuda_timing_available
                else 'CUDA device-event detail is unavailable from this profiler.\n'
            ),
            encoding='utf-8',
        )
        report_path.write_text(
            '\n'.join((
                f'captured_updates: {self.captured_updates}',
                f'device: {self.device}',
                f'compiled_entry_calls: {dict(self.compiled_entry_calls)}',
                f'graph_count_before: {self._graph_count_before}',
                f'graph_count_after: {graph_count_after}',
                'cuda_capture_status: '
                + (
                    'available'
                    if cuda_timing_available
                    else (
                        'unavailable: no positive CUDA timing events were produced'
                        if self.device.type == 'cuda'
                        else 'not requested for CPU diagnostic'
                    )
                ),
                'CPU self time and total time are distinct; parent/child rows '
                'must not be added.',
                'CUDA event spans are not reported as GPU busy time. Inspect the '
                'trace for launch gaps and stream idle periods.',
                'Inductor fused kernel names are retained and are not assigned to '
                'individual attention or FFN layers.',
            )) + '\n',
            encoding='utf-8',
        )
        paths = {
            'cpu_operator_table': str(cpu_path.resolve()),
            'cuda_operator_table': str(cuda_path.resolve()),
            'trace': str(trace_path.resolve()),
            'text_report': str(report_path.resolve()),
            'cuda_device_task_table': str(device_task_path.resolve()),
        }
        missing_entries = [
            name for name in self.expected_compiled_entries
            if self.compiled_entry_calls[name] == 0
        ]
        if missing_entries:
            raise RuntimeError(
                'Requested compiled entries were not executed: '
                f'{missing_entries}.'
            )
        graph_delta = graph_count_after - int(self._graph_count_before)
        valid_for_stable_analysis = graph_delta == 0 and (
            self.device.type != 'cuda' or cuda_timing_available
        )
        return {
            'enabled': True,
            'requested_updates': self.requested_updates,
            'captured_updates': self.captured_updates,
            'activities': [activity.name for activity in self._activities],
            'cuda_activity_requested': self.device.type == 'cuda',
            'cuda_activity_collected': cuda_timing_available,
            'cuda_capture_status': (
                'available'
                if cuda_timing_available
                else (
                    'unavailable: no positive CUDA timing events were produced'
                    if self.device.type == 'cuda'
                    else 'not requested for CPU diagnostic'
                )
            ),
            'compiled_entry_calls': dict(self.compiled_entry_calls),
            'expected_compiled_entries': list(self.expected_compiled_entries),
            'graph_count_before': self._graph_count_before,
            'graph_count_after': graph_count_after,
            'new_graph_count': graph_delta,
            'valid_for_stable_analysis': valid_for_stable_analysis,
            'sample_zone_count_min': min(self._zone_counts) if self._zone_counts else None,
            'sample_zone_count_max': max(self._zone_counts) if self._zone_counts else None,
            'bc_effective_updates': self._bc_effective_updates,
            'terminal_geometry_effective_updates': (
                self._terminal_geometry_effective_updates
            ),
            'top_cpu_operations': cpu_top,
            'other_cpu_operations': cpu_other,
            'cpu_associated_cuda_operations': (
                cpu_associated_cuda_top if self.device.type == 'cuda' else None
            ),
            'other_cpu_associated_cuda_operations': (
                cpu_associated_cuda_other if self.device.type == 'cuda' else None
            ),
            'top_cuda_operations': cuda_top if cuda_timing_available else None,
            'other_cuda_operations': cuda_other if cuda_timing_available else None,
            'cuda_device_tasks': cuda_device_tasks,
            'top_cuda_kernels': cuda_device_tasks['kernel']['operations'],
            'cuda_kernel_details_available': bool(
                cuda_timing_available
                and cuda_device_tasks['kernel']['operations']
            ),
            'cpu_operator_timing_available': cpu_other['available'],
            'cuda_operator_timing_available': (
                cuda_timing_available
            ),
            'quantitative_cpu_submission_wait_split': None,
            'quantitative_gpu_idle_time': None,
            'timeline_interpretation_note': (
                'PyTorch 2.5.1 does not expose a reliable aggregate split between '
                'CPU submission work, synchronization waits, and GPU idle time here. '
                'Use the retained operator names/counts and Chrome trace; unavailable '
                'aggregates are null rather than measured zero.'
            ),
            'output_paths': paths,
            'output_file_sizes_bytes': {
                name: Path(path).stat().st_size for name, path in paths.items()
            },
            'measurement_note': (
                'These are profiler-instrumented updates after ordinary measurement '
                'and are excluded from throughput. CPU self/total and CUDA self/total '
                'times are reported separately. CPU time is not pure GPU compute; '
                'timeline gaps require trace inspection. Missing profiler fields are '
                'reported as null, never as measured zero.'
            ),
        }

    def close(self) -> None:
        if self._profiler is not None and not self._closed:
            self._profiler.__exit__(None, None, None)
            self._closed = True


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


def _action_inference_warmup_batches(
    *, pool: V2ValidationPool, prepared, device: torch.device,
) -> tuple[V2ObservationBatch, ...]:
    batches = list(_compile_warmup_batches(
        pool=pool, prepared=prepared, batch_size=1, device=device,
    ))
    if not batches:
        raise ValueError('At least one action inference warmup batch is required.')
    if all(batch.max_zone_count != 0 for batch in batches):
        first = batches[0]
        batches.insert(0, V2ObservationBatch(
            ego_features=first.ego_features,
            goal_features=first.goal_features,
            zone_features=first.zone_features[:, :0, :],
            presence_mask=first.presence_mask[:, :0],
        ))
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
        reference_only = sorted(reference.keys() - compiled.keys())
        compiled_only = sorted(compiled.keys() - reference.keys())
        raise AssertionError(
            'Compiled numeric snapshot keys do not match reference: '
            f'reference_only={reference_only[:1]}, '
            f'compiled_only={compiled_only[:1]}.'
        )
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
        if expected.dtype == torch.bool:
            difference = (actual != expected).to(torch.float32)
        else:
            difference = (actual - expected).abs()
        if difference.numel():
            maximum_absolute_error = max(
                maximum_absolute_error,
                float(difference.max().item()),
            )
            denominator = (
                torch.ones_like(difference)
                if expected.dtype == torch.bool
                else expected.abs().clamp_min(atol)
            )
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
    line_to_goal_safe: bool = True,
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
        line_to_goal_safe=(
            torch.ones_like(zeros) if line_to_goal_safe else torch.zeros_like(zeros)
        ),
    )


def _terminal_numeric_observation_batch(
    observation: V2ObservationBatch,
    engine,
) -> V2ObservationBatch:
    radius = float(engine.terminal_geo_radius)
    horizontal_span = float(engine.actor.scales.horizontal_span)
    if not engine.terminal_geo_regularization_enabled or radius <= 0.0:
        raise AssertionError(
            'Numeric diagnostic prerequisite not met: terminal geometry '
            'regularization must be enabled with a positive radius.'
        )
    forward = min(radius * 0.25, horizontal_span * 0.05)
    right = forward * 0.5
    up = forward * 0.25
    distance = (forward * forward + right * right + up * up) ** 0.5
    goal = observation.goal_features.clone()
    goal[:, GOAL_FEATURE_INDEX['goal_forward_norm']] = forward / horizontal_span
    goal[:, GOAL_FEATURE_INDEX['goal_right_norm']] = right / horizontal_span
    goal[:, GOAL_FEATURE_INDEX['goal_up_norm']] = (
        up / float(engine.actor.scales.vertical_span)
    )
    goal[:, GOAL_FEATURE_INDEX['goal_distance_norm']] = distance / horizontal_span
    return V2ObservationBatch(
        observation.ego_features.clone(),
        goal,
        observation.zone_features.clone(),
        observation.presence_mask.clone(),
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
    comparison['frozen_critic_actor_encoder_execution'] = (
        FROZEN_CRITIC_ACTOR_ENCODER_EXECUTION
    )
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
            gradient_name = f'gradients.{critic_name}.{parameter_name}'
            snapshot[f'{gradient_name}.present'] = torch.tensor(
                parameter.grad is not None,
                dtype=torch.bool,
            )
            if parameter.grad is not None:
                snapshot[gradient_name] = parameter.grad.detach().cpu().clone()
            optimizer_state = engine.critic_optimizer.state.get(parameter)
            optimizer_name = f'optimizer.critic.{critic_name}.{parameter_name}'
            snapshot[f'{optimizer_name}.present'] = torch.tensor(
                optimizer_state is not None,
                dtype=torch.bool,
            )
            if optimizer_state is None:
                continue
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                if field not in optimizer_state:
                    raise AssertionError(
                        f'Numeric check expected Adam {field} for '
                        f'{critic_name}.{parameter_name}.'
                    )
                snapshot[f'{optimizer_name}.{field}'] = (
                    torch.as_tensor(optimizer_state[field]).detach().cpu().clone()
                )
    if metrics.actor_updated:
        for parameter_name, parameter in engine.actor.named_parameters():
            gradient_name = f'gradients.actor.{parameter_name}'
            snapshot[f'{gradient_name}.present'] = torch.tensor(
                parameter.grad is not None,
                dtype=torch.bool,
            )
            if parameter.grad is not None:
                snapshot[gradient_name] = parameter.grad.detach().cpu().clone()
            optimizer_state = engine.actor_optimizer.state.get(parameter)
            optimizer_name = f'optimizer.actor.{parameter_name}'
            snapshot[f'{optimizer_name}.present'] = torch.tensor(
                optimizer_state is not None,
                dtype=torch.bool,
            )
            if optimizer_state is None:
                continue
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                snapshot[f'{optimizer_name}.{field}'] = (
                    torch.as_tensor(optimizer_state[field]).detach().cpu().clone()
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


def _run_actor_update_with_rl_gradient_capture(
    engine,
    *,
    total_steps: int,
    bc_lambda: float = 0.0,
    capture_regularizers: bool = False,
    regularizer_probe_batch: V2ReplayBatch | None = None,
) -> tuple[Any, dict[str, Any]]:
    q_output_gradient: torch.Tensor | None = None
    bc_action_gradient: torch.Tensor | None = None
    terminal_geo_action_gradient: torch.Tensor | None = None

    def regularizer_action_gradient(name: str) -> torch.Tensor | None:
        if regularizer_probe_batch is None:
            raise AssertionError(
                'Regularizer gradient capture requires its fixed replay batch.'
            )
        batch = regularizer_probe_batch.to(engine.device)
        engine._mark_cuda_graph_step_begin()
        shared_relations = engine._build_shared_relations(batch.obs)
        actor_actions: torch.Tensor | None = None

        def capture_actor_actions(_module, _inputs, output):
            nonlocal actor_actions
            actor_actions = output

        actor_handle = engine.actor.register_forward_hook(capture_actor_actions)
        guidance = engine._actor_critic_guidance(
            batch.obs,
            shared_relations=shared_relations,
            profile_sections=False,
        )
        guidance_entered = False
        try:
            critic_context = guidance.__enter__()
            guidance_entered = True
            terms = engine._compute_actor_loss_terms(
                batch.obs,
                batch.line_to_goal_safe,
                bc_lambda=bc_lambda,
                shared_relations=shared_relations,
                critic_context=critic_context,
            )
            if actor_actions is None:
                raise AssertionError(
                    'Numeric diagnostic prerequisite not met: the actual actor '
                    'forward output was not captured.'
                )
            loss = getattr(terms, name)
            if not loss.requires_grad or not actor_actions.requires_grad:
                return None
            gradient = torch.autograd.grad(
                loss,
                actor_actions,
                allow_unused=True,
            )[0]
            return (
                None if gradient is None
                else gradient.detach().cpu().clone()
            )
        finally:
            if guidance_entered:
                guidance.__exit__(None, None, None)
            actor_handle.remove()

    def capture_frozen_q_gradient(_module, _inputs, output):
        if not any(parameter.requires_grad for parameter in engine.critic1.head.parameters()):
            def save_gradient(gradient):
                nonlocal q_output_gradient
                q_output_gradient = gradient.detach().cpu().clone()
                return gradient
            output.register_hook(save_gradient)

    if capture_regularizers:
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all()
            if engine.device.type == 'cuda' else None
        )
        try:
            bc_action_gradient = regularizer_action_gradient('bc_loss')
            terminal_geo_action_gradient = regularizer_action_gradient(
                'terminal_geo_loss'
            )
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    # Hook the Q head rather than the enclosing critic: the isolated-context
    # guidance path deliberately calls forward_from_context() and therefore
    # does not invoke the critic module's outer forward hooks.
    handle = engine.critic1.head.register_forward_hook(
        capture_frozen_q_gradient
    )
    try:
        metrics = engine.update_once(
            total_steps=total_steps,
            bc_lambda=bc_lambda,
        )
    finally:
        handle.remove()
    actor_gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in engine.actor.named_parameters()
        if parameter.grad is not None
    }
    return metrics, {
        'q_output_gradient': q_output_gradient,
        'actor_gradients': actor_gradients,
        'bc_action_gradient': bc_action_gradient,
        'terminal_geo_action_gradient': terminal_geo_action_gradient,
    }


def _report_and_validate_actor_regularizers(
    eager_metrics,
    compiled_metrics,
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
    *,
    expected_bc_lambda: float,
) -> dict[str, Any]:
    def summarize(metrics, capture: Mapping[str, Any]) -> dict[str, Any]:
        return {
            'bc_lambda': float(metrics.bc_lambda),
            'bc_loss': float(metrics.bc_loss),
            'bc_action_gradient': _gradient_diagnostic_summary(
                capture['bc_action_gradient']
            ),
            'terminal_geo_lambda': float(metrics.terminal_geo_lambda),
            'terminal_geo_loss': float(metrics.terminal_geo_loss),
            'terminal_geo_action_gradient': _gradient_diagnostic_summary(
                capture['terminal_geo_action_gradient']
            ),
        }

    record = {
        'bc_lambda': float(expected_bc_lambda),
        'eager': summarize(eager_metrics, eager),
        'compiled': summarize(compiled_metrics, compiled),
    }
    print(json.dumps(
        {'compiled_numeric_actor_regularizers': record},
        allow_nan=False,
        ensure_ascii=False,
    ), flush=True)
    for execution in ('eager', 'compiled'):
        values = record[execution]
        if not (
            values['bc_lambda'] == expected_bc_lambda
            and expected_bc_lambda > 0.0
            and values['bc_loss'] > 0.0
            and values['bc_action_gradient']['state'] == 'nonzero'
            and values['bc_action_gradient']['finite'] is True
            and values['terminal_geo_lambda'] > 0.0
            and values['terminal_geo_loss'] > 0.0
            and values['terminal_geo_action_gradient']['state'] == 'nonzero'
            and values['terminal_geo_action_gradient']['finite'] is True
        ):
            raise AssertionError(
                'Numeric diagnostic prerequisite not met: nonzero BC and '
                'terminal geometry losses must both contribute finite action '
                f'gradients for {execution}.'
            )
    _compare_optional_numeric_tensor(
        eager['bc_action_gradient'],
        compiled['bc_action_gradient'],
        name='actor_regularizers.bc_action_gradient',
        rtol=COMPILED_NUMERIC_RTOL,
        atol=COMPILED_NUMERIC_ATOL,
    )
    _compare_optional_numeric_tensor(
        eager['terminal_geo_action_gradient'],
        compiled['terminal_geo_action_gradient'],
        name='actor_regularizers.terminal_geo_action_gradient',
        rtol=COMPILED_NUMERIC_RTOL,
        atol=COMPILED_NUMERIC_ATOL,
    )
    return record


def _report_and_validate_actor_rl_gradients(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
    *,
    require_snn_module_coverage: bool = False,
) -> dict[str, Any]:
    required_snn_modules = ('zone_set_encoder', 'snn_head.fc1')

    def module_coverage(
        gradients: Mapping[str, torch.Tensor],
        module_name: str,
    ) -> dict[str, Any]:
        selected = tuple(
            gradient
            for name, gradient in gradients.items()
            if name == module_name or name.startswith(f'{module_name}.')
        )
        finite = tuple(
            bool(torch.isfinite(gradient).all()) for gradient in selected
        )
        return {
            'gradient_count': len(selected),
            'finite_nonzero_gradient_count': sum(
                int(is_finite and bool(torch.count_nonzero(gradient)))
                for gradient, is_finite in zip(selected, finite)
            ),
            'all_present_gradients_finite': bool(
                selected and all(finite)
            ),
        }

    def summarize(capture: Mapping[str, Any]) -> dict[str, Any]:
        gradients = capture['actor_gradients']
        nonzero = sum(
            int(bool(torch.count_nonzero(gradient)))
            for gradient in gradients.values()
        )
        summary = {
            'q_output_gradient': _gradient_diagnostic_summary(
                capture['q_output_gradient']
            ),
            'actor_parameter_gradient_count': len(gradients),
            'nonzero_actor_parameter_gradients': nonzero,
            'all_actor_parameter_gradients_finite': bool(
                gradients
                and all(
                    bool(torch.isfinite(gradient).all())
                    for gradient in gradients.values()
                )
            ),
        }
        if require_snn_module_coverage:
            summary['snn_module_gradient_coverage'] = {
                module_name: module_coverage(gradients, module_name)
                for module_name in required_snn_modules
            }
        return summary

    record = {'eager': summarize(eager), 'compiled': summarize(compiled)}
    print(json.dumps(
        {'compiled_numeric_actor_rl_gradient': record},
        allow_nan=False,
        ensure_ascii=False,
    ), flush=True)
    for execution, summary in record.items():
        if not (
            summary['q_output_gradient']['state'] == 'nonzero'
            and summary['q_output_gradient']['finite'] is True
            and summary['nonzero_actor_parameter_gradients'] > 0
            and summary['all_actor_parameter_gradients_finite']
        ):
            raise AssertionError(
                f'Numeric actor update did not preserve a finite RL gradient '
                f'path for {execution}.'
            )
        if require_snn_module_coverage:
            for module_name, coverage in summary[
                'snn_module_gradient_coverage'
            ].items():
                if not (
                    coverage['gradient_count'] > 0
                    and coverage['finite_nonzero_gradient_count'] > 0
                    and coverage['all_present_gradients_finite']
                ):
                    raise AssertionError(
                        'Numeric diagnostic prerequisite not met: SNN pure RL '
                        f'gradient did not reach {module_name} for {execution}.'
                    )
    _compare_optional_numeric_tensor(
        eager['q_output_gradient'],
        compiled['q_output_gradient'],
        name='actor_rl.q_output_gradient',
        rtol=COMPILED_NUMERIC_RTOL,
        atol=COMPILED_NUMERIC_ATOL,
    )
    _compare_compiled_numeric_tensors(
        eager['actor_gradients'],
        compiled['actor_gradients'],
        rtol=COMPILED_NUMERIC_RTOL,
        atol=COMPILED_NUMERIC_ATOL,
    )
    return record


def _set_numeric_engine_modes(engine) -> None:
    """Establish the modes used by a real TD3 learning update."""

    engine.actor.train()
    engine.critic1.train()
    engine.critic2.train()
    for target in (
        engine.actor_target,
        engine.critic1_target,
        engine.critic2_target,
    ):
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
    if engine.bc_reference_actor is not None:
        engine.bc_reference_actor.eval()
        for parameter in engine.bc_reference_actor.parameters():
            parameter.requires_grad_(False)


def _numeric_engine_mode_summary(engine, *, model: str) -> dict[str, Any]:
    def frozen_module_summary(module) -> dict[str, Any]:
        return {
            'training': bool(module.training),
            'all_parameters_frozen': all(
                not parameter.requires_grad for parameter in module.parameters()
            ),
        }

    summary: dict[str, Any] = {
        'online_actor_training': bool(engine.actor.training),
        'online_critics_training': {
            'critic1': bool(engine.critic1.training),
            'critic2': bool(engine.critic2.training),
        },
        'target_networks': {
            name: frozen_module_summary(module)
            for name, module in (
                ('actor_target', engine.actor_target),
                ('critic1_target', engine.critic1_target),
                ('critic2_target', engine.critic2_target),
            )
        },
        'bc_reference_actor': (
            None
            if engine.bc_reference_actor is None
            else frozen_module_summary(engine.bc_reference_actor)
        ),
    }
    if model == 'snn':
        if not isinstance(engine.actor, V2SNNPolicyActor):
            summary['snn_lif_training'] = {}
        else:
            summary['snn_lif_training'] = {
                f'snn_head.{name}': bool(module.training)
                for name, module in engine.actor.snn_head.named_modules()
                if name and 'lif' in name.lower()
            }
    return summary


def _report_and_validate_numeric_engine_modes(
    eager,
    compiled,
    *,
    model: str,
    phase: str,
) -> dict[str, Any]:
    record = {
        'phase': phase,
        'model': model,
        'engines': {
            'eager': _numeric_engine_mode_summary(eager, model=model),
            'compiled': _numeric_engine_mode_summary(compiled, model=model),
        },
    }
    print(json.dumps(
        {'compiled_numeric_module_modes': record},
        allow_nan=False,
        ensure_ascii=False,
    ), flush=True)

    problems: list[str] = []
    for execution, summary in record['engines'].items():
        if not summary['online_actor_training']:
            problems.append(f'{execution}.online_actor')
        for critic_name, training in summary['online_critics_training'].items():
            if not training:
                problems.append(f'{execution}.{critic_name}')
        for target_name, target in summary['target_networks'].items():
            if target['training'] or not target['all_parameters_frozen']:
                problems.append(f'{execution}.{target_name}')
        bc_reference = summary['bc_reference_actor']
        if bc_reference is None:
            problems.append(f'{execution}.bc_reference_actor_missing')
        elif (
            bc_reference['training']
            or not bc_reference['all_parameters_frozen']
        ):
            problems.append(f'{execution}.bc_reference_actor')
        if model == 'snn':
            lif_modes = summary.get('snn_lif_training', {})
            if not lif_modes:
                problems.append(f'{execution}.snn_lif_modules_missing')
            problems.extend(
                f'{execution}.{name}'
                for name, training in lif_modes.items()
                if not training
            )
    if problems:
        raise AssertionError(
            'Numeric diagnostic module modes are invalid: '
            + ', '.join(problems)
            + '.'
        )
    return record


def _run_compiled_numerics_check(
    *,
    pool: V2ValidationPool,
    prepared,
    formal_config: V2FormalTrainingConfig,
    bc_checkpoint: Path,
    device: torch.device,
    snn_time_window: int,
    model: str = 'ann',
    compile_critic_encoder: bool = True,
    compile_target_encoders: bool = True,
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
    cuda_graph_updates: bool = False,
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
            model_type=model,
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
            model_type=model,
            snn_time_window=snn_time_window,
            prepared_initialization=prepared,
            fused_adam=fused_adam,
            aggregate_relation_values_first=aggregate_relation_values_first,
            reduce_update_stat_syncs=reduce_update_stat_syncs,
            pinned_batch_transfer=pinned_batch_transfer,
        )
        reference = reference_components.engine
        compiled = compiled_components.engine
        compiled.load_checkpoint_state_dict(reference.checkpoint_state_dict())
        if reference.bc_reference_actor is None or compiled.bc_reference_actor is None:
            raise AssertionError(
                'Numeric diagnostic prerequisite not met: a frozen BC reference '
                'actor is required.'
            )
        _set_numeric_engine_modes(reference)
        _set_numeric_engine_modes(compiled)
        module_modes = {
            'before_compile_warmup': (
                _report_and_validate_numeric_engine_modes(
                    reference,
                    compiled,
                    model=model,
                    phase='before_compile_warmup',
                )
            ),
        }
        configured = compiled.configure_compilation(
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
            backend='inductor', mode='default', fullgraph=True, dynamic=True,
        )
        warmup_batches = _compile_warmup_batches(
            pool=pool,
            prepared=prepared,
            batch_size=reference.batch_size,
            device=device,
        )
        if compile_actors:
            compiled.warmup_actor_compile(warmup_batches)
        if compile_shared_relations:
            compiled.warmup_shared_relations_compile(warmup_batches)
        if compile_critic_block:
            compiled.warmup_full_compile(warmup_batches)
        elif compile_critic_encoder:
            compiled.warmup_online_critic_encoder_compile(warmup_batches)
            if compile_target_encoders:
                compiled.warmup_target_encoder_compile(warmup_batches)
        if compile_snn_target_encoder and not compile_target_block:
            compiled.warmup_snn_target_encoder_compile(warmup_batches)
        if compile_actor_loss:
            compiled.warmup_actor_loss_compile(warmup_batches)
        if cuda_graph_updates:
            configured['cuda_graph_evidence'] = (
                compiled.verify_update_cuda_graph_capture(warmup_batches)
            )
        action_inference_batches = _action_inference_warmup_batches(
            pool=pool, prepared=prepared, device=device,
        )
        if compile_action_inference:
            compiled.warmup_action_inference_compile(action_inference_batches)
        if cuda_graph_action_inference:
            configured['cuda_graph_action_inference_evidence'] = (
                compiled.verify_action_inference_cuda_graph_capture(
                    action_inference_batches
                )
            )
        action_inference_comparison = []
        with torch.inference_mode():
            for batch in action_inference_batches:
                expected = reference._action_inference_tensor(batch)
                actual = compiled._action_inference_tensor(batch)
                torch.testing.assert_close(
                    actual, expected,
                    rtol=COMPILED_NUMERIC_RTOL,
                    atol=COMPILED_NUMERIC_ATOL,
                )
                action_inference_comparison.append({
                    'batch_size': batch.batch_size,
                    'zone_count': batch.max_zone_count,
                })
        module_modes['before_consecutive_updates'] = (
            _report_and_validate_numeric_engine_modes(
                reference,
                compiled,
                model=model,
                phase='before_consecutive_updates',
            )
        )
        mixed_observation, nonempty_observation = (
            _numeric_diagnostic_observation_batches(
                warmup_batches,
                device=device,
            )
        )
        fixed_batch = _fixed_numeric_replay_batch(
            mixed_observation,
            action_dim=reference.action_dim,
            device=device,
            line_to_goal_safe=False,
        )
        terminal_observation = _terminal_numeric_observation_batch(
            mixed_observation,
            reference,
        )
        terminal_batch = _fixed_numeric_replay_batch(
            terminal_observation,
            action_dim=reference.action_dim,
            device=device,
            line_to_goal_safe=True,
        )
        nonempty_batch = _fixed_numeric_replay_batch(
            nonempty_observation,
            action_dim=reference.action_dim,
            device=device,
            line_to_goal_safe=False,
        )
        diagnostic_batches = (
            fixed_batch,
            fixed_batch,
            fixed_batch,
            terminal_batch,
            nonempty_batch,
        )
        if formal_config.policy_delay <= 1:
            raise ValueError(
                'Compiled numeric consecutive-update checking requires '
                'policy_delay > 1.'
            )
        update_results: dict[str, Any] = {}
        actor_rl_gradient: dict[str, Any] | None = None
        regularizer_gradient_check: dict[str, Any] | None = None
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
            capture_actor_rl,
            bc_lambda,
            capture_regularizers,
        ) in (
            (
                1,
                'critic_only',
                'critic_only_with_empty_scene',
                'synthetic_from_fixed_pool_with_one_zero_zone_sample',
                1,
                diagnostic_batches[0],
                True,
                False,
                0.0,
                False,
            ),
            (
                2,
                'actor_and_target_updated',
                'actor_and_target_updated_with_empty_scene',
                'synthetic_from_fixed_pool_with_one_zero_zone_sample',
                formal_config.policy_delay,
                diagnostic_batches[1],
                False,
                True,
                0.0,
                False,
            ),
            (
                3,
                'critic_only_after_actor',
                'critic_only_after_actor_with_empty_scene',
                'synthetic_from_fixed_pool_with_one_zero_zone_sample',
                formal_config.policy_delay + 1,
                diagnostic_batches[2],
                False,
                False,
                0.0,
                False,
            ),
            (
                4,
                'actor_with_bc_and_terminal_geometry',
                'actor_with_nonzero_bc_and_terminal_geometry',
                'synthetic_near_goal_safe_batch_with_one_zero_zone_sample',
                formal_config.policy_delay * 2,
                diagnostic_batches[3],
                True,
                False,
                1.5,
                True,
            ),
            (
                5,
                'critic_only_nonempty_after_momentum',
                'critic_only_all_nonempty_after_empty_scene_momentum',
                'synthetic_all_nonempty_batch_after_mixed_batch_momentum',
                formal_config.policy_delay * 2 + 1,
                diagnostic_batches[4],
                False,
                False,
                0.0,
                False,
            ),
        ):
            reference.replay.sample = lambda batch_size, batch=fixed_batch: batch
            compiled.replay.sample = lambda batch_size, batch=fixed_batch: batch
            eager_before = _empty_token_states(reference)
            compiled_before = _empty_token_states(compiled)
            if label == 'critic_only_nonempty_after_momentum':
                switch_record = {
                    'zero_zone_sample_count': _zero_zone_sample_count(
                        fixed_batch.obs
                    ),
                    'historical_momentum_prerequisite_met': (
                        _historical_momentum_prerequisite_met(
                            eager_before,
                            compiled_before,
                        )
                    ),
                }
                print(json.dumps(
                    {'compiled_numeric_nonempty_switch': switch_record},
                    allow_nan=False,
                    ensure_ascii=False,
                ), flush=True)
                if switch_record['zero_zone_sample_count'] != 0:
                    raise AssertionError(
                        'Numeric diagnostic prerequisite not met: the momentum '
                        'switch batch contains an empty scene.'
                    )
                if not switch_record['historical_momentum_prerequisite_met']:
                    raise AssertionError(
                        'Numeric diagnostic prerequisite not met: empty_scene_token '
                        'does not have nonzero historical Adam momentum before the '
                        'all-nonempty batch.'
                    )
            update_rng_state = torch.random.get_rng_state()
            update_cuda_rng_state = torch.cuda.get_rng_state_all()
            if capture_actor_rl or capture_regularizers:
                reference_metrics, reference_actor_capture = (
                    _run_actor_update_with_rl_gradient_capture(
                        reference,
                        total_steps=total_steps,
                        bc_lambda=bc_lambda,
                        capture_regularizers=capture_regularizers,
                        regularizer_probe_batch=(
                            fixed_batch if capture_regularizers else None
                        ),
                    )
                )
            else:
                reference_metrics = reference.update_once(
                    total_steps=total_steps,
                    bc_lambda=bc_lambda,
                )
            torch.random.set_rng_state(update_rng_state)
            torch.cuda.set_rng_state_all(update_cuda_rng_state)
            if capture_actor_rl or capture_regularizers:
                compiled_metrics, compiled_actor_capture = (
                    _run_actor_update_with_rl_gradient_capture(
                        compiled,
                        total_steps=total_steps,
                        bc_lambda=bc_lambda,
                        capture_regularizers=capture_regularizers,
                        regularizer_probe_batch=(
                            fixed_batch if capture_regularizers else None
                        ),
                    )
                )
            else:
                compiled_metrics = compiled.update_once(
                    total_steps=total_steps,
                    bc_lambda=bc_lambda,
                )
            if capture_actor_rl:
                if any(
                    metrics.bc_lambda != 0.0
                    or metrics.terminal_geo_loss != 0.0
                    for metrics in (reference_metrics, compiled_metrics)
                ):
                    raise AssertionError(
                        'Numeric diagnostic prerequisite not met: the pure RL '
                        'actor update included BC or terminal geometry loss.'
                    )
                actor_rl_gradient = _report_and_validate_actor_rl_gradients(
                    reference_actor_capture,
                    compiled_actor_capture,
                    require_snn_module_coverage=(model == 'snn'),
                )
            if capture_regularizers:
                regularizer_gradient_check = (
                    _report_and_validate_actor_regularizers(
                        reference_metrics,
                        compiled_metrics,
                        reference_actor_capture,
                        compiled_actor_capture,
                        expected_bc_lambda=bc_lambda,
                    )
                )
            eager_after = _empty_token_states(reference)
            compiled_after = _empty_token_states(compiled)
            if label in ('critic_only', 'critic_only_after_actor') and (
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
            try:
                comparison = _compare_compiled_numeric_tensors(
                    _numeric_update_snapshot(reference, fixed_batch, reference_metrics),
                    _numeric_update_snapshot(compiled, fixed_batch, compiled_metrics),
                    rtol=COMPILED_NUMERIC_RTOL,
                    atol=COMPILED_NUMERIC_ATOL,
                )
            except AssertionError as exc:
                print(json.dumps({
                    'compiled_numeric_failure': {
                        'update_index': update_index,
                        'update': diagnostic_label,
                        'error': str(exc),
                    },
                }, allow_nan=False, ensure_ascii=False), flush=True)
                raise
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
            'actor_rl_gradient': actor_rl_gradient,
            'regularizer_gradient_check': regularizer_gradient_check,
            'module_modes': module_modes,
            'frozen_critic_actor_encoder_execution': (
                frozen_critic_strategy
            ),
            'compilation': configured,
            'action_inference_comparison': action_inference_comparison,
            'wall_seconds': perf_counter() - started,
            'measurement_note': (
                'Uses independent original and optimized engines before timed diagnosis; '
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
            'frozen_critic_actor_encoder_execution': (
                FROZEN_CRITIC_ACTOR_ENCODER_EXECUTION
            ),
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
    cuda_graph_updates: bool = False,
    compiled_path_profiler_updates: int = 0,
    compiled_profiler_output_dir: Path | None = None,
    environment_performance_diagnostic: bool = False,
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
        fused_adam=fused_adam,
        aggregate_relation_values_first=aggregate_relation_values_first,
        reduce_update_stat_syncs=reduce_update_stat_syncs,
        pinned_batch_transfer=pinned_batch_transfer,
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
    performance_diagnostic_enabled = bool(
        environment_performance_diagnostic or compiled_path_profiler_updates > 0
    )
    environment_diagnostic = (
        _EnvironmentGeometryDiagnostic(duplicate_audit_steps=8)
        if performance_diagnostic_enabled
        else None
    )
    nested_update_timing = _NestedUpdateTimingSummary()
    compile_requested = any((
        compile_critic_encoder,
        compile_target_encoders,
        compile_actors,
        compile_critic_block,
        compile_target_block,
        compile_shared_relations,
        compile_snn_target_encoder,
        compile_actor_loss,
        compile_action_inference,
        cuda_graph_action_inference,
        cuda_graph_updates,
    ))
    extended_compile_requested = any((
        compile_actors,
        compile_critic_block,
        compile_target_block,
        compile_shared_relations,
        compile_snn_target_encoder,
        compile_actor_loss,
        compile_action_inference,
        cuda_graph_action_inference,
        frozen_critic_strategy != 'eager',
    ))
    compile_metadata: dict[str, Any] = {
        'requested': compile_requested,
        'target_encoders_requested': bool(compile_target_encoders),
        'actors_requested': bool(compile_actors),
        'critic_block_requested': bool(compile_critic_block),
        'target_block_requested': bool(compile_target_block),
        'shared_relations_requested': bool(compile_shared_relations),
        'snn_target_encoder_requested': bool(compile_snn_target_encoder),
        'actor_loss_requested': bool(compile_actor_loss),
        'actor_loss_coefficients_requested': bool(cache_actor_loss_coefficients),
        'action_inference_requested': bool(compile_action_inference),
        'cuda_graph_action_inference_requested': bool(
            cuda_graph_action_inference
        ),
        'pinned_batch_transfer_requested': bool(pinned_batch_transfer),
        'reduce_update_stat_syncs_requested': bool(reduce_update_stat_syncs),
        'cuda_graph_updates_requested': bool(cuda_graph_updates),
        'optimizer_execution': 'fused_adam' if fused_adam else 'adam',
        'relation_value_execution': (
            'aggregate_then_project' if aggregate_relation_values_first
            else 'project_then_aggregate'
        ),
        'actor_loss_granularity': 'eager',
        'actor_loss_coefficient_execution': 'per_update',
        'action_inference_granularity': 'eager',
        'update_statistics_execution': (
            'batched_device_readback'
            if reduce_update_stat_syncs else 'per_scalar'
        ),
        'batch_transfer_execution': (
            'reusable_pinned_non_blocking'
            if pinned_batch_transfer else 'blocking_to_device'
        ),
        'enabled_objects': [],
        'backend': 'inductor',
        'mode': 'default',
        'fullgraph': True,
        'dynamic': True,
        'frozen_critic_strategy': frozen_critic_strategy,
        'frozen_critic_actor_encoder_execution': (
            frozen_critic_strategy if compile_requested else None
        ),
        'select_action_execution': 'eager',
        'cuda_graph': False,
        'cuda_graph_evidence': None,
        'cuda_graph_action_inference': False,
        'cuda_graph_action_inference_evidence': None,
        'registration_wall_seconds': 0.0,
        'warmup_wall_seconds': 0.0,
        'online_registration_wall_seconds': 0.0,
        'online_warmup_wall_seconds': 0.0,
        'target_registration_wall_seconds': 0.0,
        'target_warmup_wall_seconds': 0.0,
        'warmup_batch_shapes': [],
        'action_inference_warmup_shapes': [],
        'target_warmup_batch_shapes': [],
        'measurement_graph_count_before': None,
        'measurement_graph_count_after': None,
        'measurement_new_graph_count': None,
        'stable_timing': None,
        'valid_for_speed_comparison': None,
        'measurement_note': 'Critic encoder compilation is disabled.',
    }
    if compile_requested and extended_compile_requested:
        registration_started = perf_counter()
        configured = engine.configure_compilation(
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
            backend='inductor', mode='default', fullgraph=True, dynamic=True,
        )
        compile_metadata.update(configured)
        compile_metadata['registration_wall_seconds'] = (
            perf_counter() - registration_started
        )
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
            action_batches = _action_inference_warmup_batches(
                pool=pool, prepared=prepared, device=device,
            )
            engine.warmup_action_inference_compile(action_batches)
            compile_metadata['action_inference_warmup_shapes'] = [
                [batch.batch_size, batch.max_zone_count]
                for batch in action_batches
            ]
            if cuda_graph_action_inference:
                compile_metadata['cuda_graph_action_inference_evidence'] = (
                    engine.verify_action_inference_cuda_graph_capture(
                        action_batches
                    )
                )
        if cuda_graph_updates:
            compile_metadata['cuda_graph_evidence'] = (
                engine.verify_update_cuda_graph_capture(warmup_batches)
            )
        compile_metadata['warmup_wall_seconds'] = perf_counter() - warmup_started
        compile_metadata['measurement_note'] = (
            'Registration and compile-triggering pure-compute warmup are excluded '
            'from measured update wall time. Measurement is valid for speed '
            'comparison only if no new Dynamo graphs appear after the warmup boundary.'
        )
    elif compile_critic_encoder:
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
    compiled_path_profiler = None
    compiled_path_profiler_metadata: dict[str, Any] = {
        'enabled': False,
        'requested_updates': 0,
        'captured_updates': 0,
        'output_paths': {},
        'measurement_note': 'Compiled-path performance profiling is disabled.',
    }
    if compiled_path_profiler_updates > 0:
        if compiled_profiler_output_dir is None:
            raise ValueError(
                'compiled_profiler_output_dir is required for compiled profiling.'
            )
        expected_compiled_entries = ['critic_block', 'target_block']
        if compile_actors:
            expected_compiled_entries.extend(('actor', 'bc_reference_actor'))
        if frozen_critic_strategy == 'compiled_no_grad_context':
            expected_compiled_entries.append('frozen_critic_context')
        if compile_shared_relations:
            expected_compiled_entries.append('shared_relations')
        if compile_snn_target_encoder:
            expected_compiled_entries.append('snn_target_encoder')
        if compile_actor_loss:
            expected_compiled_entries.append('actor_loss')
        if model == 'snn':
            expected_compiled_entries[expected_compiled_entries.index(
                'target_block'
            )] = 'target_critic_td_block'
        compiled_path_profiler = _CompiledPathProfiler(
            device,
            requested_updates=compiled_path_profiler_updates,
            output_dir=compiled_profiler_output_dir,
            expected_compiled_entries=expected_compiled_entries,
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
                if compile_requested:
                    compile_metadata['measurement_graph_count_before'] = (
                        _dynamo_unique_graph_count()
                    )
                if environment_diagnostic is not None:
                    env.set_performance_diagnostic(environment_diagnostic)
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
            audit_step = (
                environment_diagnostic.audit_step()
                if environment_diagnostic is not None and measured_phase
                else nullcontext()
            )
            with audit_step:
                if environment_diagnostic is None or not measured_phase:
                    line_safe = call(
                        'pre_action_geometry_wall_seconds',
                        lambda: env.line_to_goal_is_safe(
                            env.state[:3],
                            clearance=formal_config.terminal_geo_safe_clearance,
                        ),
                    )
                else:
                    with environment_diagnostic.source('pre_action_geometry'):
                        line_safe = call(
                            'pre_action_geometry_wall_seconds',
                            lambda: env.line_to_goal_is_safe(
                                env.state[:3],
                                clearance=(
                                    formal_config.terminal_geo_safe_clearance
                                ),
                            ),
                        )
                action = call(
                    'action_inference_wall_seconds',
                    lambda: engine.select_action(
                        observation,
                        exploration_noise=(
                            formal_config.noise_schedule.exploration_initial
                        ),
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
                update_details: dict[str, Any] = {}

                def capture_update_sections(sections: Mapping[str, float]) -> None:
                    update_sections.update(sections)

                def capture_update_details(details: Mapping[str, Any]) -> None:
                    update_details.update(details)

                def record_update(result: Any, elapsed: float) -> None:
                    update_timing.record(
                        actor_updated=bool(result.actor_updated),
                        total_wall_seconds=elapsed,
                        sections=update_sections,
                    )
                    if performance_diagnostic_enabled:
                        nested_update_timing.record(
                            actor_updated=bool(result.actor_updated),
                            outer_sections=update_sections,
                            detail=update_details,
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
                                **(
                                    {
                                        'diagnostic_timing_recorder': (
                                            capture_update_details
                                        ),
                                    }
                                    if (
                                        measured_phase
                                        and performance_diagnostic_enabled
                                        and not profile_sections
                                    )
                                    else {}
                                ),
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
        ordinary_actor_updates = engine.actor_update_count - actor_updates_before
        ordinary_critic_updates = engine.critic_update_count - critic_updates_before
        ordinary_graph_count_after = (
            _dynamo_unique_graph_count() if compile_requested else None
        )
        if environment_diagnostic is not None:
            env.set_performance_diagnostic(None)
            environment_diagnostic.close()
        if compiled_path_profiler is not None:
            profiled_actor_results: list[bool] = []
            for offset in range(compiled_path_profiler.requested_updates):
                sampled_batch: list[V2ReplayBatch] = []

                def capture_profile_sample(batch_size: int) -> V2ReplayBatch:
                    batch = original_sample(batch_size)
                    sampled_batch.append(batch)
                    return batch

                engine.replay.sample = capture_profile_sample
                profile_total_steps = step_number + offset + 1
                engine.set_target_noise(
                    policy_noise=formal_config.noise_schedule.policy_initial,
                    noise_clip=formal_config.noise_schedule.clip_initial,
                )
                metrics = compiled_path_profiler.run(
                    lambda total_steps=profile_total_steps: engine.update_once(
                        total_steps=total_steps,
                        bc_lambda=v2_bc_lambda(total_steps - 1),
                        profile_sections=False,
                        diagnostic_profile_sections=True,
                        compiled_execution_recorder=(
                            compiled_path_profiler.record_compiled_entry
                        ),
                    )
                )
                if len(sampled_batch) != 1:
                    raise RuntimeError(
                        'Compiled profiler update must sample Replay exactly once.'
                    )
                compiled_path_profiler.record_update_inputs(
                    sampled_batch[0], metrics
                )
                profiled_actor_results.append(bool(metrics.actor_updated))
            if not (any(profiled_actor_results) and not all(profiled_actor_results)):
                raise RuntimeError(
                    'Compiled profiler must capture critic-only and actor updates.'
                )
            compiled_path_profiler_metadata = compiled_path_profiler.finish()
    finally:
        engine.replay.sample = original_sample
        env.set_performance_diagnostic(None)
        if environment_diagnostic is not None:
            environment_diagnostic.close()
        if compiled_path_profiler is not None:
            compiled_path_profiler.close()
        profiler_metadata = detailed_profiler.finish()

    update_seconds = timing.wall_seconds['td3_update_wall_seconds']
    sample_seconds = timing.wall_seconds['replay_sample_wall_seconds']
    network_update_seconds = max(0.0, update_seconds - sample_seconds)
    if not all(isfinite(value) and value >= 0.0 for value in timing.wall_seconds.values()):
        raise RuntimeError('Diagnostic produced a non-finite timing value.')
    actor_updates = ordinary_actor_updates
    critic_updates = ordinary_critic_updates
    if critic_updates == 0 or actor_updates == 0:
        raise RuntimeError('Diagnostic measurement must include both critic and actor updates.')
    if compile_requested:
        graph_count_after = ordinary_graph_count_after
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
        else (
            'The optional environment and nested-update diagnostic is enabled. '
            'This ordinary short-run wall time includes its instrumentation '
            'overhead; the separate compiled profiler updates are excluded.'
            if performance_diagnostic_enabled
            else 'Detailed profiler is disabled; throughput is calculated from this '
            'ordinary short-diagnostic wall time.'
        )
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
        'compiled_path_profiler': compiled_path_profiler_metadata,
        'environment_geometry_diagnostic': (
            environment_diagnostic.to_dict()
            if environment_diagnostic is not None
            else {
                'enabled': False,
                'measurement_note': 'Environment geometry diagnostics are disabled.',
            }
        ),
        'critic_encoder_compile': compile_metadata,
        'timing': {
            'total_wall_seconds': measured_total,
            'total_wall_seconds_includes_detailed_profiler_overhead': (
                profiler_overhead_included
            ),
            'total_wall_seconds_includes_performance_diagnostic_overhead': (
                performance_diagnostic_enabled
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
            'td3_nested_update_breakdown': (
                nested_update_timing.to_dict(environment_steps=measured_steps)
                if performance_diagnostic_enabled
                else None
            ),
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
    cuda_graph_updates: bool = False,
    check_compiled_numerics: bool = False,
    compiled_numerics_only: bool = False,
    compiled_numerics_group: str | None = None,
    compiled_performance_diagnostic_updates: int = 0,
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
    compiled_profiler_updates = _nonnegative_int(
        compiled_performance_diagnostic_updates,
        name='compiled_performance_diagnostic_updates',
    )
    if compiled_profiler_updates not in (0, *range(2, 9)):
        raise ValueError(
            'compiled_performance_diagnostic_updates must be 0 or between 2 '
            'and 8 so both critic-only and actor updates can be observed.'
        )
    if type(compile_critic_encoder) is not bool:
        raise TypeError('compile_critic_encoder must be a bool.')
    if type(compile_target_encoders) is not bool:
        raise TypeError('compile_target_encoders must be a bool.')
    for name, value in (
        ('compile_actors', compile_actors),
        ('compile_critic_block', compile_critic_block),
        ('compile_target_block', compile_target_block),
        ('compile_shared_relations', compile_shared_relations),
        ('compile_snn_target_encoder', compile_snn_target_encoder),
        ('fused_adam', fused_adam),
        ('compile_actor_loss', compile_actor_loss),
        ('cache_actor_loss_coefficients', cache_actor_loss_coefficients),
        ('compile_action_inference', compile_action_inference),
        ('aggregate_relation_values_first', aggregate_relation_values_first),
        ('reduce_update_stat_syncs', reduce_update_stat_syncs),
        ('cuda_graph_updates', cuda_graph_updates),
    ):
        if type(value) is not bool:
            raise TypeError(f'{name} must be a bool.')
    if frozen_critic_strategy not in ('eager', 'compiled_no_grad_context'):
        raise ValueError(
            'frozen_critic_strategy must be eager or compiled_no_grad_context.'
        )
    if cache_actor_loss_coefficients and not compile_actor_loss:
        raise ValueError('cache_actor_loss_coefficients requires compile_actor_loss.')
    if type(check_compiled_numerics) is not bool:
        raise TypeError('check_compiled_numerics must be a bool.')
    if type(compiled_numerics_only) is not bool:
        raise TypeError('compiled_numerics_only must be a bool.')
    if compiled_numerics_only and not check_compiled_numerics:
        raise ValueError(
            'compiled_numerics_only requires check_compiled_numerics.'
        )
    if compiled_numerics_group is not None:
        if type(compiled_numerics_group) is not str:
            raise TypeError('compiled_numerics_group must be a string or None.')
        compiled_numerics_group = compiled_numerics_group.upper()
        if compiled_numerics_group not in ('A', 'B', 'C'):
            raise ValueError('compiled_numerics_group must be A, B, or C.')
        if (
            compile_critic_encoder
            or compile_target_encoders
            or compile_actors
            or compile_critic_block
            or compile_target_block
            or compile_shared_relations
            or compile_snn_target_encoder
            or fused_adam
            or compile_actor_loss
            or compile_action_inference
            or aggregate_relation_values_first
            or reduce_update_stat_syncs
            or cuda_graph_updates
            or frozen_critic_strategy != 'eager'
            or check_compiled_numerics
            or compiled_numerics_only
            or profiler_updates
            or compiled_profiler_updates
        ):
            raise ValueError(
                'compiled_numerics_group is an isolated mode and cannot be '
                'combined with compile/profiler timing options.'
            )
    if compile_target_encoders and compile_snn_target_encoder:
        raise ValueError(
            'compile_target_encoders and compile_snn_target_encoder are mutually exclusive.'
        )
    if compile_target_encoders and not compile_critic_encoder:
        raise ValueError(
            'compile_target_encoders requires compile_critic_encoder.'
        )
    if compile_critic_encoder and compile_critic_block:
        raise ValueError(
            'compile_critic_encoder and compile_critic_block are mutually exclusive.'
        )
    if compile_target_encoders and compile_target_block:
        raise ValueError(
            'compile_target_encoders and compile_target_block are mutually exclusive.'
        )
    if compile_snn_target_encoder and model != 'snn':
        raise ValueError('compile_snn_target_encoder requires the SNN diagnostic.')
    if compile_target_block and not compile_critic_block:
        raise ValueError('compile_target_block requires compile_critic_block.')
    if frozen_critic_strategy == 'compiled_no_grad_context' and not (
        compile_critic_encoder or compile_critic_block
    ):
        raise ValueError(
            'compiled_no_grad_context requires a compiled critic path.'
        )
    if check_compiled_numerics and not (
        compile_target_encoders or compile_target_block or compile_snn_target_encoder
        or fused_adam or compile_actor_loss or compile_action_inference
        or aggregate_relation_values_first or reduce_update_stat_syncs
        or cuda_graph_updates
    ):
        raise ValueError(
            'check_compiled_numerics requires a target compile scope or one '
            'of the new optimization flags.'
        )
    compile_requested = any((
        compile_critic_encoder,
        compile_target_encoders,
        compile_actors,
        compile_critic_block,
        compile_target_block,
        compile_shared_relations,
        compile_snn_target_encoder,
        compile_actor_loss,
        compile_action_inference,
        cuda_graph_action_inference,
    ))
    if compile_requested and profiler_updates:
        raise ValueError(
            'compilation cannot be combined with detailed profiler.'
        )
    if compiled_profiler_updates:
        if profiler_updates or check_compiled_numerics or compiled_numerics_only:
            raise ValueError(
                'compiled performance diagnostic cannot be combined with '
                'detailed profiler or compiled numerics modes.'
            )
        if not (
            compile_actors
            and compile_critic_block
            and compile_target_block
            and frozen_critic_strategy == 'compiled_no_grad_context'
        ):
            raise ValueError(
                'compiled performance diagnostic requires the full compiled '
                'actor, critic block, target block, and compiled_no_grad_context '
                'configuration.'
            )
    if (compile_critic_encoder or compile_target_encoders) and model != 'ann':
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
    if cuda_graph_updates and target_device.type != 'cuda':
        raise ValueError('cuda_graph_updates requires a CUDA diagnostic.')
    if cuda_graph_updates and not compile_critic_block:
        raise ValueError('cuda_graph_updates requires compile_critic_block.')
    if cuda_graph_action_inference and not compile_action_inference:
        raise ValueError(
            'cuda_graph_action_inference requires compile_action_inference.'
        )
    if cuda_graph_action_inference and target_device.type != 'cuda':
        raise ValueError(
            'cuda_graph_action_inference requires a CUDA diagnostic.'
        )
    if pinned_batch_transfer and target_device.type != 'cuda':
        raise ValueError('pinned_batch_transfer requires a CUDA diagnostic.')
    if compiled_numerics_group is not None and target_device.type != 'cuda':
        raise ValueError('compiled_numerics_group requires a CUDA diagnostic.')
    if compiled_profiler_updates and target_device.type != 'cuda':
        raise ValueError(
            'compiled performance diagnostic requires a CUDA diagnostic.'
        )
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
            model=model,
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
            cache_actor_loss_coefficients=cache_actor_loss_coefficients,
            compile_action_inference=compile_action_inference,
            cuda_graph_action_inference=cuda_graph_action_inference,
            aggregate_relation_values_first=aggregate_relation_values_first,
            reduce_update_stat_syncs=reduce_update_stat_syncs,
            pinned_batch_transfer=pinned_batch_transfer,
            cuda_graph_updates=cuda_graph_updates,
        )
    if compiled_numerics_only:
        output.mkdir(parents=True, exist_ok=False)
        summary = {
            'format': DIAGNOSTIC_FORMAT,
            'format_version': DIAGNOSTIC_VERSION,
            'formal_stage_passed': False,
            'purpose': 'compiled_numeric_correctness_check_only',
            'model': model,
            'requested_device': requested_device,
            'resolved_device': resolved_device,
            'seed': run_seed,
            'validation_seed': pool_seed,
            'bc_checkpoint': str(checkpoint.resolve()),
            'scenario_pool_directory': str(Path(scenario_pool_dir).resolve()),
            'scenario_pool_prepare_wall_seconds': pool_prepare_seconds,
            'compiled_numerics': compiled_numerics,
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
            compile_actors=compile_actors,
            frozen_critic_strategy=frozen_critic_strategy,
            compile_critic_block=compile_critic_block,
            compile_target_block=compile_target_block,
            compile_shared_relations=compile_shared_relations,
            compile_snn_target_encoder=compile_snn_target_encoder,
            fused_adam=fused_adam,
            compile_actor_loss=compile_actor_loss,
            cache_actor_loss_coefficients=cache_actor_loss_coefficients,
            compile_action_inference=compile_action_inference,
            cuda_graph_action_inference=cuda_graph_action_inference,
            aggregate_relation_values_first=aggregate_relation_values_first,
            reduce_update_stat_syncs=reduce_update_stat_syncs,
            pinned_batch_transfer=pinned_batch_transfer,
            cuda_graph_updates=cuda_graph_updates,
            compiled_path_profiler_updates=(
                compiled_profiler_updates if level == 'medium' else 0
            ),
            compiled_profiler_output_dir=(
                output / 'compiled_path_profiler' / 'medium'
                if level == 'medium' and compiled_profiler_updates
                else None
            ),
            environment_performance_diagnostic=bool(compiled_profiler_updates),
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
    performance_report: dict[str, Any] = {
        'enabled': False,
        'path': None,
        'size_bytes': None,
    }
    if compiled_profiler_updates:
        report_path = output / 'performance_diagnostic_summary.txt'
        report_lines = [
            'UAV V2 TD3 bounded compiled-path performance diagnostic',
            'Wall subtimings include diagnostic overhead and do not synchronize '
            'CUDA per child interval.',
        ]
        for level in DIAGNOSTIC_LEVELS:
            environment = level_results[level]['environment_geometry_diagnostic']
            report_lines.append(
                f'{level}: environment_steps={environment["step_count"]}, '
                f'environment_step_wall_seconds='
                f'{environment["step_total_wall_seconds"]:.9f}'
            )
            for name, values in environment['step_sections'].items():
                report_lines.append(
                    f'  {name}: calls={values["calls"]}, '
                    f'average_ms_per_step='
                    f'{1000.0 * values["average_wall_seconds_per_step"]:.6f}'
                )
        medium_profile = level_results['medium']['compiled_path_profiler']
        report_lines.extend((
            f'medium_profiled_updates={medium_profile["captured_updates"]}',
            f'medium_new_graph_count={medium_profile["new_graph_count"]}',
            f'medium_valid_for_stable_analysis='
            f'{medium_profile["valid_for_stable_analysis"]}',
            f'medium_cuda_capture_status='
            f'{medium_profile["cuda_capture_status"]}',
            f'medium_sample_zone_count_range='
            f'{medium_profile["sample_zone_count_min"]}..'
            f'{medium_profile["sample_zone_count_max"]}',
            f'medium_bc_effective_updates='
            f'{medium_profile["bc_effective_updates"]}',
            f'medium_terminal_geometry_effective_updates='
            f'{medium_profile["terminal_geometry_effective_updates"]}',
            f'medium_compiled_entry_calls='
            f'{medium_profile["compiled_entry_calls"]}',
            'See compiled operator tables and trace for CPU submission/waits, '
            'kernel counts, and timeline gaps. Fused kernels retain profiler names.',
        ))
        report_path.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
        performance_report = {
            'enabled': True,
            'path': str(report_path.resolve()),
            'size_bytes': report_path.stat().st_size,
        }
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
        'performance_diagnostic_report': performance_report,
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
            'compiled_performance_profiler_updates_medium': (
                compiled_profiler_updates
            ),
            'environment_performance_diagnostic': bool(
                compiled_profiler_updates
            ),
            'compile_critic_encoder_requested': compile_critic_encoder,
            'compile_target_encoders_requested': compile_target_encoders,
            'compile_actors_requested': compile_actors,
            'frozen_critic_strategy': frozen_critic_strategy,
            'compile_critic_block_requested': compile_critic_block,
            'compile_target_block_requested': compile_target_block,
            'compile_shared_relations_requested': compile_shared_relations,
            'compile_snn_target_encoder_requested': compile_snn_target_encoder,
            'fused_adam_requested': fused_adam,
            'compile_actor_loss_requested': compile_actor_loss,
            'cache_actor_loss_coefficients_requested': cache_actor_loss_coefficients,
            'compile_action_inference_requested': compile_action_inference,
            'cuda_graph_action_inference_requested': (
                cuda_graph_action_inference
            ),
            'pinned_batch_transfer_requested': pinned_batch_transfer,
            'aggregate_relation_values_first_requested': (
                aggregate_relation_values_first
            ),
            'reduce_update_stat_syncs_requested': reduce_update_stat_syncs,
            'cuda_graph_updates_requested': cuda_graph_updates,
            'cuda_graph': cuda_graph_updates,
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
        '--compile-actors',
        action='store_true',
        help='Compile ANN actor full forwards or SNN actor encoders.',
    )
    parser.add_argument(
        '--frozen-critic-strategy',
        choices=('eager', 'compiled_no_grad_context'),
        default='eager',
        help='Execution strategy for critic1 while it guides the actor.',
    )
    parser.add_argument(
        '--compile-critic-block',
        action='store_true',
        help='Compile twin critic full forwards and their existing loss.',
    )
    parser.add_argument(
        '--compile-target-block',
        action='store_true',
        help='Compile the applicable target forward and TD-target tensor block.',
    )
    parser.add_argument(
        '--compile-shared-relations',
        action='store_true',
        help='Compile the canonical shared relation tensor build.',
    )
    parser.add_argument(
        '--compile-snn-target-encoder',
        action='store_true',
        help='Compile only the SNN target actor ZoneSetEncoder.',
    )
    parser.add_argument('--fused-adam', action='store_true')
    parser.add_argument('--compile-actor-loss', action='store_true')
    parser.add_argument('--cache-actor-loss-coefficients', action='store_true')
    parser.add_argument('--compile-action-inference', action='store_true')
    parser.add_argument('--cuda-graph-action-inference', action='store_true')
    parser.add_argument('--pinned-batch-transfer', action='store_true')
    parser.add_argument('--aggregate-relation-values-first', action='store_true')
    parser.add_argument('--reduce-update-stat-syncs', action='store_true')
    parser.add_argument('--cuda-graph-updates', action='store_true')
    parser.add_argument(
        '--check-compiled-numerics',
        action='store_true',
        help='Run an isolated CUDA eager-versus-compiled TD3 numeric check.',
    )
    parser.add_argument(
        '--compiled-numerics-only',
        action='store_true',
        help='Exit after the isolated compiled numeric correctness check.',
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
    parser.add_argument(
        '--compiled-performance-diagnostic-updates',
        type=int,
        nargs='?',
        const=8,
        default=0,
        metavar='N',
        help=(
            'Enable environment/update detail on all levels and profile at most '
            'N additional real compiled TD3 updates after medium measurement '
            '(bare flag: 8).'
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
        compile_actors=args.compile_actors,
        frozen_critic_strategy=args.frozen_critic_strategy,
        compile_critic_block=args.compile_critic_block,
        compile_target_block=args.compile_target_block,
        compile_shared_relations=args.compile_shared_relations,
        compile_snn_target_encoder=args.compile_snn_target_encoder,
        fused_adam=args.fused_adam,
        compile_actor_loss=args.compile_actor_loss,
        cache_actor_loss_coefficients=args.cache_actor_loss_coefficients,
        compile_action_inference=args.compile_action_inference,
        cuda_graph_action_inference=args.cuda_graph_action_inference,
        aggregate_relation_values_first=args.aggregate_relation_values_first,
        reduce_update_stat_syncs=args.reduce_update_stat_syncs,
        pinned_batch_transfer=args.pinned_batch_transfer,
        cuda_graph_updates=args.cuda_graph_updates,
        check_compiled_numerics=args.check_compiled_numerics,
        compiled_numerics_only=args.compiled_numerics_only,
        compiled_numerics_group=args.compiled_numerics_group,
        compiled_performance_diagnostic_updates=(
            args.compiled_performance_diagnostic_updates
        ),
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
