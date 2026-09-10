"""Behavior cloning support for ragged V2 expert trajectory clusters.

This module is independent from the legacy flat-observation BC path.  It
validates one trajectory shard at a time, retains only trajectory-level index
metadata, and optionally reuses strictly loaded arrays through a bounded LRU.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from math import ceil, isfinite
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from brain_uav.config import ScenarioConfig
from brain_uav.models import V2ANNPolicyActor, V2SNNPolicyActor
from brain_uav.models.zone_set_encoder import ZoneSetEncoderConfig
from brain_uav.observations import (
    EGO_FEATURE_DIM,
    GOAL_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    V2Observation,
    V2ObservationBatch,
    V2ObservationScales,
)
from brain_uav.scripts.generate_v2_trajectory_clusters import (
    V2_TRAJECTORY_CLUSTER_FORMAT,
    V2_TRAJECTORY_CLUSTER_VERSION,
    load_easy_scenario_pool,
)
from brain_uav.scripts.v2_trajectory_io import load_v2_trajectory_shard


V2_BC_CHECKPOINT_FORMAT = 'v2_bc_actor_checkpoint'
V2_BC_CHECKPOINT_VERSION = 1
V2_SNN_BC_CHECKPOINT_FORMAT = 'v2_snn_bc_actor_checkpoint'
V2_SNN_BC_CHECKPOINT_VERSION = 1
V2_BC_OBSERVATION_CONTRACT_ID = 'v2_dynamic_zone_set_observation_v1'
V2_BC_PLANNER_NAMES = ('v2_heuristic', 'v2_apf')

_MANIFEST_FIELDS = {
    'format',
    'format_version',
    'status',
    'scenario_pool_file',
    'master_seed',
    'requested_scenarios',
    'shard_size',
    'scenario_config',
    'uav_collision_radius',
    'shards',
    'statistics',
}
_SHARD_ENTRY_FIELDS = {
    'file',
    'scenario_sequence_start',
    'scenario_sequence_end',
    'trajectory_count',
    'step_count',
    'zone_token_count',
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f'Non-finite JSON number {value!r} is not allowed.')


def _strict_json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True),
        parse_constant=_reject_json_constant,
    )


def _load_strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'{name} file does not exist: {path}')
    try:
        value = json.loads(
            path.read_text(encoding='utf-8'),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{name} is not valid strict JSON.') from exc
    if type(value) is not dict:
        raise ValueError(f'{name} must be a strict JSON object.')
    return _strict_json_copy(value)


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a non-negative integer.')
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f'{name} must be finite.')
    result = float(value)
    if not isfinite(result):
        raise ValueError(f'{name} must be finite.')
    return result


def _nonnegative_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f'{name} must be non-negative.')
    return result


def _nonempty_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f'{name} must be a non-empty string.')
    return value


def _bound_file(root: Path, relative: Any, *, name: str) -> Path:
    text = _nonempty_text(relative, name=name)
    candidate_part = Path(text)
    if candidate_part.is_absolute():
        raise ValueError(f'{name} must be relative to the trajectory cluster.')
    resolved_root = root.resolve()
    candidate = (resolved_root / candidate_part).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f'{name} escapes the trajectory cluster directory.')
    return candidate


def _scenario_config_from_snapshot(value: Any) -> ScenarioConfig:
    if type(value) is not dict:
        raise ValueError('scenario_config must be a JSON object.')
    try:
        scenario = ScenarioConfig(**deepcopy(value))
    except (TypeError, ValueError) as exc:
        raise ValueError('scenario_config cannot construct ScenarioConfig.') from exc
    if _strict_json_copy(asdict(scenario)) != value:
        raise ValueError('scenario_config is not a complete normalized snapshot.')
    return scenario


class V2ShardArrayCache:
    """Process-local LRU of strictly loaded shard arrays, bounded by nbytes."""

    def __init__(self, *, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('max_bytes must be a non-negative integer.')
        self.max_bytes = max_bytes
        self._cached_bytes = 0
        self._entries: OrderedDict[
            Path, tuple[dict[str, np.ndarray], int]
        ] = OrderedDict()

    @property
    def cached_bytes(self) -> int:
        return self._cached_bytes

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def load(self, path: str | Path) -> dict[str, np.ndarray]:
        key = Path(path).resolve()
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached[0]
        arrays = load_v2_trajectory_shard(key)
        for array in arrays.values():
            array.setflags(write=False)
        size = sum(array.nbytes for array in arrays.values())
        if self.max_bytes == 0 or size > self.max_bytes:
            return arrays
        while self._entries and self._cached_bytes + size > self.max_bytes:
            _, (_, removed_size) = self._entries.popitem(last=False)
            self._cached_bytes -= removed_size
        self._entries[key] = (arrays, size)
        self._cached_bytes += size
        return arrays


@dataclass(frozen=True, slots=True)
class V2BCShardRecord:
    """Validated summary of one shard kept without its step arrays."""

    path: Path
    manifest_index: int
    trajectory_count: int
    step_count: int
    zone_token_count: int


@dataclass(frozen=True, slots=True)
class V2BCTrajectoryRecord:
    """Trajectory-level index data retained after a shard is released."""

    shard_index: int
    trajectory_index: int
    trajectory_id: str
    scenario_id: str
    planner_name: str
    step_start: int
    step_end: int
    zone_count: int

    @property
    def step_count(self) -> int:
        return self.step_end - self.step_start


@dataclass(frozen=True, slots=True, eq=False)
class V2BCTrajectoryCluster:
    """Strict cluster provenance plus bounded trajectory/shard indexes."""

    root: Path
    manifest: dict[str, Any]
    scenario_pool: dict[str, Any]
    scenario_config: ScenarioConfig
    uav_collision_radius: float
    shards: tuple[V2BCShardRecord, ...]
    source_trajectories: tuple[V2BCTrajectoryRecord, ...]
    trajectories: tuple[V2BCTrajectoryRecord, ...]
    scenario_ids: tuple[str, ...]
    deduplicate_identical_trajectories: bool
    duplicate_trajectory_mappings: tuple[dict[str, str], ...]
    shard_cache: V2ShardArrayCache

    @property
    def trajectory_count(self) -> int:
        return len(self.trajectories)

    @property
    def source_trajectory_count(self) -> int:
        return len(self.source_trajectories)

    @property
    def step_count(self) -> int:
        return sum(record.step_count for record in self.trajectories)

    @property
    def source_step_count(self) -> int:
        return sum(record.step_count for record in self.source_trajectories)

    @property
    def zone_token_count(self) -> int:
        return sum(shard.zone_token_count for shard in self.shards)


def _trajectory_arrays_equal(
    first: V2BCTrajectoryRecord,
    second: V2BCTrajectoryRecord,
    cache: V2ShardArrayCache,
    shards: Sequence[V2BCShardRecord],
) -> bool:
    if first.scenario_id != second.scenario_id or first.step_count != second.step_count:
        return False
    first_arrays = cache.load(shards[first.shard_index].path)
    second_arrays = (
        first_arrays
        if first.shard_index == second.shard_index
        else cache.load(shards[second.shard_index].path)
    )
    first_slice = slice(first.step_start, first.step_end)
    second_slice = slice(second.step_start, second.step_end)
    for name in ('ego_features', 'goal_features', 'actions', 'states_before_action'):
        if not np.array_equal(first_arrays[name][first_slice], second_arrays[name][second_slice]):
            return False
    for name in ('terminal_states', 'outcomes', 'step_counts', 'scenario_seeds'):
        if not np.array_equal(
            first_arrays[name][first.trajectory_index],
            second_arrays[name][second.trajectory_index],
        ):
            return False
    first_offsets = first_arrays['zone_offsets'][first.step_start : first.step_end + 1]
    second_offsets = second_arrays['zone_offsets'][second.step_start : second.step_end + 1]
    first_relative = first_offsets - first_offsets[0]
    second_relative = second_offsets - second_offsets[0]
    if not np.array_equal(first_relative, second_relative):
        return False
    first_zones = first_arrays['zone_features'][int(first_offsets[0]) : int(first_offsets[-1])]
    second_zones = second_arrays['zone_features'][int(second_offsets[0]) : int(second_offsets[-1])]
    return np.array_equal(first_zones, second_zones)


def _deduplicate_expert_trajectories(
    records: Sequence[V2BCTrajectoryRecord],
    cache: V2ShardArrayCache,
    shards: Sequence[V2BCShardRecord],
) -> tuple[tuple[V2BCTrajectoryRecord, ...], tuple[dict[str, str], ...]]:
    by_scenario: dict[str, list[V2BCTrajectoryRecord]] = {}
    for record in records:
        by_scenario.setdefault(record.scenario_id, []).append(record)
    removed: set[str] = set()
    mappings: list[dict[str, str]] = []
    for scenario_id in sorted(by_scenario):
        scenario_records = by_scenario[scenario_id]
        heuristics = [
            record for record in scenario_records
            if record.planner_name == 'v2_heuristic'
        ]
        apf_records = [
            record for record in scenario_records
            if record.planner_name == 'v2_apf'
        ]
        for apf_record in apf_records:
            identical = next(
                (
                    heuristic
                    for heuristic in heuristics
                    if _trajectory_arrays_equal(heuristic, apf_record, cache, shards)
                ),
                None,
            )
            if identical is None:
                continue
            removed.add(apf_record.trajectory_id)
            mappings.append({
                'scenario_id': scenario_id,
                'kept_trajectory_id': identical.trajectory_id,
                'kept_planner_name': identical.planner_name,
                'removed_trajectory_id': apf_record.trajectory_id,
                'removed_planner_name': apf_record.planner_name,
            })
    effective = tuple(
        record for record in records if record.trajectory_id not in removed
    )
    return effective, tuple(mappings)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError('V2 trajectory cluster manifest has missing or unknown fields.')
    if manifest['format'] != V2_TRAJECTORY_CLUSTER_FORMAT:
        raise ValueError('Unsupported V2 trajectory cluster manifest format.')
    if manifest['format_version'] != V2_TRAJECTORY_CLUSTER_VERSION:
        raise ValueError('Unsupported V2 trajectory cluster manifest version.')
    if manifest['status'] != 'complete':
        raise ValueError('V2 trajectory cluster status must be complete.')
    _nonnegative_int(manifest['master_seed'], name='manifest master_seed')
    _positive_int(manifest['requested_scenarios'], name='manifest requested_scenarios')
    _positive_int(manifest['shard_size'], name='manifest shard_size')
    _nonnegative_float(
        manifest['uav_collision_radius'],
        name='manifest uav_collision_radius',
    )
    if type(manifest['shards']) is not list or not manifest['shards']:
        raise ValueError('V2 trajectory cluster manifest must reference at least one shard.')
    statistics = manifest['statistics']
    if type(statistics) is not dict:
        raise ValueError('manifest statistics must be a JSON object.')
    successful = statistics.get('successful_trajectories')
    if type(successful) is not int or successful <= 0:
        raise ValueError('manifest successful_trajectories must be greater than zero.')


def load_v2_bc_trajectory_cluster(
    path: str | Path,
    *,
    shard_cache_mb: float = 0.0,
    deduplicate_identical_trajectories: bool = False,
) -> V2BCTrajectoryCluster:
    """Strictly validate a complete cluster while holding one shard at a time."""

    cache_megabytes = _nonnegative_float(shard_cache_mb, name='shard_cache_mb')
    if type(deduplicate_identical_trajectories) is not bool:
        raise TypeError('deduplicate_identical_trajectories must be a bool.')
    cache_bytes = int(cache_megabytes * 1024 * 1024)
    cache = V2ShardArrayCache(max_bytes=cache_bytes)

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f'Trajectory cluster directory does not exist: {root}')
    root = root.resolve()
    manifest = _load_strict_json(root / 'manifest.json', name='manifest')
    _validate_manifest(manifest)
    pool_path = _bound_file(root, manifest['scenario_pool_file'], name='scenario_pool_file')
    scenario_pool = load_easy_scenario_pool(pool_path)
    if manifest['master_seed'] != scenario_pool['master_seed']:
        raise ValueError('manifest and scenario pool master_seed mismatch.')
    if manifest['requested_scenarios'] != scenario_pool['scenario_count']:
        raise ValueError('manifest and scenario pool requested_scenarios mismatch.')
    if manifest['scenario_config'] != scenario_pool['scenario_config']:
        raise ValueError('manifest and scenario pool scenario_config mismatch.')
    if manifest['uav_collision_radius'] != scenario_pool['uav_collision_radius']:
        raise ValueError('manifest and scenario pool uav_collision_radius mismatch.')
    scenario_config = _scenario_config_from_snapshot(manifest['scenario_config'])
    collision_radius = _nonnegative_float(
        manifest['uav_collision_radius'],
        name='uav_collision_radius',
    )
    scenario_items = {
        item['scenario_id']: item
        for item in scenario_pool['scenarios']
    }
    valid_scenario_ids = set(scenario_items)
    action_low = np.asarray(
        [-scenario_config.delta_gamma_max, -scenario_config.delta_psi_max],
        dtype=np.float32,
    )
    action_high = -action_low
    if not np.all(np.isfinite(action_high)) or np.any(action_high <= 0.0):
        raise ValueError('ScenarioConfig action limits must be finite and positive.')

    shard_records: list[V2BCShardRecord] = []
    trajectory_records: list[V2BCTrajectoryRecord] = []
    seen_files: set[Path] = set()
    seen_trajectory_ids: set[str] = set()
    total_trajectories = 0
    for shard_index, entry in enumerate(manifest['shards']):
        if type(entry) is not dict or set(entry) != _SHARD_ENTRY_FIELDS:
            raise ValueError(f'manifest shard {shard_index} has invalid fields.')
        shard_path = _bound_file(root, entry['file'], name=f'shards[{shard_index}].file')
        if shard_path in seen_files:
            raise ValueError('manifest references the same shard file more than once.')
        if not shard_path.is_file():
            raise FileNotFoundError(f'missing shard file: {shard_path}')
        seen_files.add(shard_path)
        expected_trajectories = _positive_int(
            entry['trajectory_count'], name=f'shards[{shard_index}].trajectory_count'
        )
        expected_steps = _positive_int(
            entry['step_count'], name=f'shards[{shard_index}].step_count'
        )
        expected_zones = _nonnegative_int(
            entry['zone_token_count'], name=f'shards[{shard_index}].zone_token_count'
        )
        start_sequence = _nonnegative_int(
            entry['scenario_sequence_start'],
            name=f'shards[{shard_index}].scenario_sequence_start',
        )
        end_sequence = _nonnegative_int(
            entry['scenario_sequence_end'],
            name=f'shards[{shard_index}].scenario_sequence_end',
        )
        if start_sequence > end_sequence or end_sequence >= manifest['requested_scenarios']:
            raise ValueError(f'manifest shard {shard_index} has invalid scenario bounds.')

        arrays = cache.load(shard_path)
        actual_summary = (
            int(arrays['trajectory_ids'].shape[0]),
            int(arrays['ego_features'].shape[0]),
            int(arrays['zone_features'].shape[0]),
        )
        expected_summary = (expected_trajectories, expected_steps, expected_zones)
        if actual_summary != expected_summary:
            raise ValueError(
                f'shard {entry["file"]!r} statistics mismatch: '
                f'actual={actual_summary}, manifest={expected_summary}.'
            )
        if np.any(arrays['actions'] < action_low) or np.any(arrays['actions'] > action_high):
            raise ValueError(f'shard {entry["file"]!r} contains an out-of-range expert action.')
        offsets = arrays['trajectory_offsets']
        for trajectory_index in range(expected_trajectories):
            trajectory_id = str(arrays['trajectory_ids'][trajectory_index])
            scenario_id = str(arrays['scenario_ids'][trajectory_index])
            payload_ref = str(arrays['scenario_payload_refs'][trajectory_index])
            planner_name = str(arrays['planner_names'][trajectory_index])
            if trajectory_id in seen_trajectory_ids:
                raise ValueError(f'Duplicate trajectory_id {trajectory_id!r} across shards.')
            seen_trajectory_ids.add(trajectory_id)
            if scenario_id not in valid_scenario_ids:
                raise ValueError(f'trajectory scenario_id {scenario_id!r} is absent from scenario pool.')
            if payload_ref != scenario_id:
                raise ValueError('trajectory scenario_payload_ref must equal scenario_id.')
            if planner_name not in V2_BC_PLANNER_NAMES:
                raise ValueError(f'Unsupported V2 expert planner_name {planner_name!r}.')
            if str(arrays['outcomes'][trajectory_index]) != 'goal':
                raise ValueError('V2 BC accepts only outcome="goal" trajectories.')
            scenario_item = scenario_items[scenario_id]
            if int(arrays['scenario_seeds'][trajectory_index]) != scenario_item['scenario_seed']:
                raise ValueError('trajectory scenario_seed does not match scenario pool.')
            step_start = int(offsets[trajectory_index])
            step_end = int(offsets[trajectory_index + 1])
            zone_counts = np.diff(
                arrays['zone_offsets'][step_start : step_end + 1]
            )
            expected_zone_count = len(scenario_item['payload']['zones'])
            if np.any(zone_counts != expected_zone_count):
                raise ValueError('trajectory observation zone count does not match scenario payload.')
            trajectory_records.append(V2BCTrajectoryRecord(
                shard_index=shard_index,
                trajectory_index=trajectory_index,
                trajectory_id=trajectory_id,
                scenario_id=scenario_id,
                planner_name=planner_name,
                step_start=step_start,
                step_end=step_end,
                zone_count=expected_zone_count,
            ))
        shard_records.append(V2BCShardRecord(
            path=shard_path,
            manifest_index=shard_index,
            trajectory_count=expected_trajectories,
            step_count=expected_steps,
            zone_token_count=expected_zones,
        ))
        total_trajectories += expected_trajectories
        del arrays

    if total_trajectories != manifest['statistics']['successful_trajectories']:
        raise ValueError('manifest successful_trajectories does not match shard totals.')
    if not trajectory_records:
        raise ValueError('V2 BC trajectory cluster contains no successful data.')
    source_records = tuple(trajectory_records)
    if deduplicate_identical_trajectories:
        effective_records, duplicate_mappings = _deduplicate_expert_trajectories(
            source_records,
            cache,
            shard_records,
        )
    else:
        effective_records = source_records
        duplicate_mappings = ()
    effective_scenarios = tuple(sorted({record.scenario_id for record in effective_records}))
    return V2BCTrajectoryCluster(
        root=root,
        manifest=manifest,
        scenario_pool=scenario_pool,
        scenario_config=scenario_config,
        uav_collision_radius=collision_radius,
        shards=tuple(shard_records),
        source_trajectories=source_records,
        trajectories=effective_records,
        scenario_ids=effective_scenarios,
        deduplicate_identical_trajectories=deduplicate_identical_trajectories,
        duplicate_trajectory_mappings=duplicate_mappings,
        shard_cache=cache,
    )


def restore_v2_observation(
    shard_arrays: Mapping[str, np.ndarray],
    step_index: int,
) -> V2Observation:
    """Restore one ragged step without flattening or padding its zone rows."""

    if not isinstance(shard_arrays, Mapping):
        raise TypeError('shard_arrays must be a mapping returned by the V2 shard loader.')
    if type(step_index) is not int:
        raise TypeError('step_index must be an integer.')
    required = ('ego_features', 'goal_features', 'zone_features', 'zone_offsets')
    if any(name not in shard_arrays for name in required):
        raise ValueError('shard_arrays is missing observation arrays.')
    total_steps = int(shard_arrays['ego_features'].shape[0])
    if step_index < 0 or step_index >= total_steps:
        raise IndexError(f'step_index {step_index} is outside [0, {total_steps}).')
    zone_start = int(shard_arrays['zone_offsets'][step_index])
    zone_end = int(shard_arrays['zone_offsets'][step_index + 1])
    zone_rows = shard_arrays['zone_features'][zone_start:zone_end]
    return V2Observation(
        ego_features=shard_arrays['ego_features'][step_index],
        goal_features=shard_arrays['goal_features'][step_index],
        zone_features=zone_rows,
        presence_mask=np.ones(zone_end - zone_start, dtype=np.bool_),
    )


def _selected_step_indices(
    arrays: Mapping[str, np.ndarray],
    scenario_ids: set[str],
    trajectory_indices: set[int] | None = None,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    offsets = arrays['trajectory_offsets']
    for trajectory_index, raw_scenario_id in enumerate(arrays['scenario_ids']):
        if (
            trajectory_indices is not None
            and trajectory_index not in trajectory_indices
        ):
            continue
        if str(raw_scenario_id) not in scenario_ids:
            continue
        start = int(offsets[trajectory_index])
        end = int(offsets[trajectory_index + 1])
        selected.append(np.arange(start, end, dtype=np.int64))
    if not selected:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(selected)


def assemble_v2_bc_batch(
    arrays: Mapping[str, np.ndarray],
    step_indices: np.ndarray,
    *,
    device: torch.device | str = 'cpu',
) -> tuple[V2ObservationBatch, torch.Tensor]:
    """Assemble one validated shard slice on CPU, then transfer whole tensors."""

    if not isinstance(arrays, Mapping):
        raise TypeError('arrays must be a validated shard mapping.')
    required = (
        'ego_features',
        'goal_features',
        'zone_features',
        'zone_offsets',
        'actions',
    )
    if any(name not in arrays for name in required):
        raise ValueError('arrays is missing V2 BC batch data.')
    indices = np.asarray(step_indices)
    if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in 'iu':
        raise ValueError('step_indices must be a non-empty integer vector.')
    indices = indices.astype(np.int64, copy=False)
    total_steps = int(arrays['ego_features'].shape[0])
    if np.any(indices < 0) or np.any(indices >= total_steps):
        raise IndexError('step_indices contains an out-of-range step.')
    starts = arrays['zone_offsets'][indices].astype(np.int64, copy=False)
    ends = arrays['zone_offsets'][indices + 1].astype(np.int64, copy=False)
    counts = ends - starts
    max_zone_count = int(counts.max(initial=0))
    batch_size = int(indices.size)
    zone_rows = np.zeros(
        (batch_size, max_zone_count, ZONE_FEATURE_DIM), dtype=np.float32
    )
    for row, (zone_start, zone_end) in enumerate(zip(starts, ends)):
        zone_count = int(zone_end - zone_start)
        if zone_count:
            zone_rows[row, :zone_count] = arrays['zone_features'][zone_start:zone_end]
    presence_mask = (
        np.arange(max_zone_count, dtype=np.int64)[None, :] < counts[:, None]
    )
    cpu_batch = V2ObservationBatch(
        ego_features=torch.from_numpy(
            np.ascontiguousarray(arrays['ego_features'][indices], dtype=np.float32)
        ),
        goal_features=torch.from_numpy(
            np.ascontiguousarray(arrays['goal_features'][indices], dtype=np.float32)
        ),
        zone_features=torch.from_numpy(zone_rows),
        presence_mask=torch.from_numpy(presence_mask),
    )
    actions = torch.from_numpy(
        np.ascontiguousarray(arrays['actions'][indices], dtype=np.float32)
    )
    target_device = torch.device(device)
    return (
        cpu_batch.to(target_device),
        actions.to(device=target_device, dtype=torch.float32),
    )


def iter_v2_bc_batches(
    cluster: V2BCTrajectoryCluster,
    scenario_ids: Sequence[str],
    *,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator | None,
    device: torch.device | str = 'cpu',
) -> Iterator[tuple[V2ObservationBatch, torch.Tensor]]:
    """Stream batches from validated shard arrays.

    This intentionally uses no DataLoader workers.  Padding is performed only
    to the largest real zone count in each yielded batch.  A cluster-level
    bounded cache may reuse the same strict shard load across epochs and splits.
    """

    if not isinstance(cluster, V2BCTrajectoryCluster):
        raise TypeError('cluster must be a V2BCTrajectoryCluster.')
    size = _positive_int(batch_size, name='batch_size')
    if type(shuffle) is not bool:
        raise TypeError('shuffle must be a bool.')
    if isinstance(scenario_ids, (str, bytes)) or not isinstance(scenario_ids, Sequence):
        raise TypeError('scenario_ids must be a non-empty sequence of strings.')
    selected_ids = tuple(scenario_ids)
    if not selected_ids or any(type(value) is not str for value in selected_ids):
        raise ValueError('scenario_ids must be a non-empty sequence of strings.')
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError('scenario_ids must not contain duplicates.')
    unknown = set(selected_ids) - set(cluster.scenario_ids)
    if unknown:
        raise ValueError(f'scenario_ids contains unknown values: {sorted(unknown)}.')
    if shuffle and not isinstance(rng, np.random.Generator):
        raise TypeError('A numpy.random.Generator is required when shuffle=True.')
    target_device = torch.device(device)
    shard_order = (
        rng.permutation(len(cluster.shards)).tolist()
        if shuffle
        else list(range(len(cluster.shards)))
    )
    selected_set = set(selected_ids)
    for shard_index in shard_order:
        arrays = cluster.shard_cache.load(cluster.shards[shard_index].path)
        selected_trajectory_indices = {
            record.trajectory_index
            for record in cluster.trajectories
            if record.shard_index == shard_index
            and record.scenario_id in selected_set
        }
        step_indices = _selected_step_indices(
            arrays,
            selected_set,
            selected_trajectory_indices,
        )
        if shuffle and step_indices.size:
            step_indices = step_indices[rng.permutation(step_indices.size)]
        for start in range(0, int(step_indices.size), size):
            batch_indices = step_indices[start : start + size]
            if not batch_indices.size:
                continue
            observation_batch, actions = assemble_v2_bc_batch(
                arrays,
                batch_indices,
                device=target_device,
            )
            yield observation_batch, actions
        del arrays


def _split_statistics(
    cluster: V2BCTrajectoryCluster,
    scenario_ids: tuple[str, ...],
) -> dict[str, Any]:
    selected = set(scenario_ids)
    records = [record for record in cluster.trajectories if record.scenario_id in selected]
    source_records = [
        record for record in cluster.source_trajectories
        if record.scenario_id in selected
    ]
    mappings = [
        _strict_json_copy(mapping)
        for mapping in cluster.duplicate_trajectory_mappings
        if mapping['scenario_id'] in selected
    ]
    planners = {
        name: {'trajectory_count': 0, 'step_count': 0}
        for name in V2_BC_PLANNER_NAMES
    }
    zone_count_steps = {'0': 0, '1': 0, '2': 0}
    for record in records:
        planners[record.planner_name]['trajectory_count'] += 1
        planners[record.planner_name]['step_count'] += record.step_count
        key = str(record.zone_count)
        if key not in zone_count_steps:
            zone_count_steps[key] = 0
        zone_count_steps[key] += record.step_count
    return {
        'scenario_count': len(scenario_ids),
        'trajectory_count': len(records),
        'step_count': sum(record.step_count for record in records),
        'planners': planners,
        'zone_count_steps': zone_count_steps,
        'deduplication': {
            'enabled': cluster.deduplicate_identical_trajectories,
            'trajectory_count_before': len(source_records),
            'trajectory_count_after': len(records),
            'step_count_before': sum(record.step_count for record in source_records),
            'step_count_after': sum(record.step_count for record in records),
            'removed_trajectory_count': len(source_records) - len(records),
            'duplicate_trajectory_mappings': mappings,
        },
    }


@dataclass(frozen=True, slots=True)
class V2BCScenarioSplit:
    """Leak-free deterministic split grouped by scenario identifier."""

    seed: int
    validation_fraction: float
    train_scenario_ids: tuple[str, ...]
    validation_scenario_ids: tuple[str, ...]
    train_statistics: dict[str, Any]
    validation_statistics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'seed': self.seed,
            'validation_fraction': self.validation_fraction,
            'training_fraction': 1.0 - self.validation_fraction,
            'train_scenario_ids': list(self.train_scenario_ids),
            'validation_scenario_ids': list(self.validation_scenario_ids),
            'train_statistics': _strict_json_copy(self.train_statistics),
            'validation_statistics': _strict_json_copy(self.validation_statistics),
        }


def split_v2_bc_scenarios(
    cluster: V2BCTrajectoryCluster,
    *,
    validation_fraction: float = 0.2,
    seed: int,
) -> V2BCScenarioSplit:
    """Split sorted successful scenario IDs using a seeded permutation."""

    if not isinstance(cluster, V2BCTrajectoryCluster):
        raise TypeError('cluster must be a V2BCTrajectoryCluster.')
    fraction = _finite_float(validation_fraction, name='validation_fraction')
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError('validation_fraction must be strictly between 0 and 1.')
    split_seed = _nonnegative_int(seed, name='seed')
    ordered_ids = tuple(sorted(cluster.scenario_ids))
    validation_count = int(ceil(len(ordered_ids) * fraction))
    if validation_count < 1 or validation_count >= len(ordered_ids):
        raise ValueError(
            'The successful data must provide at least one training and one '
            'validation scenario for the requested validation_fraction.'
        )
    rng = np.random.default_rng(split_seed)
    permutation = rng.permutation(len(ordered_ids))
    validation_ids = tuple(sorted(ordered_ids[int(i)] for i in permutation[:validation_count]))
    train_ids = tuple(sorted(ordered_ids[int(i)] for i in permutation[validation_count:]))
    return V2BCScenarioSplit(
        seed=split_seed,
        validation_fraction=fraction,
        train_scenario_ids=train_ids,
        validation_scenario_ids=validation_ids,
        train_statistics=_split_statistics(cluster, train_ids),
        validation_statistics=_split_statistics(cluster, validation_ids),
    )


class ValidationBestTracker:
    """Keep the first actor state achieving each strictly lower validation loss."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_validation_loss = float('inf')
        self._best_state_dict: dict[str, torch.Tensor] | None = None

    @property
    def best_state_dict(self) -> dict[str, torch.Tensor]:
        if self._best_state_dict is None:
            raise RuntimeError('No finite validation loss has produced a best state.')
        return self._best_state_dict

    def consider(
        self,
        *,
        epoch: int,
        validation_loss: float,
        actor: nn.Module,
    ) -> bool:
        epoch_number = _positive_int(epoch, name='epoch')
        loss = _finite_float(validation_loss, name='validation_loss')
        if not isinstance(actor, nn.Module):
            raise TypeError('actor must be a torch.nn.Module.')
        if loss >= self.best_validation_loss:
            return False
        self.best_epoch = epoch_number
        self.best_validation_loss = loss
        self._best_state_dict = _cpu_state_dict(actor.state_dict())
        return True


@dataclass(frozen=True, slots=True)
class V2BCTrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    validation_fraction: float
    device: str | torch.device = 'cpu'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'epochs', _positive_int(self.epochs, name='epochs'))
        object.__setattr__(self, 'batch_size', _positive_int(self.batch_size, name='batch_size'))
        learning_rate = _finite_float(self.learning_rate, name='learning_rate')
        if learning_rate <= 0.0:
            raise ValueError('learning_rate must be greater than zero.')
        object.__setattr__(self, 'learning_rate', learning_rate)
        object.__setattr__(self, 'seed', _nonnegative_int(self.seed, name='seed'))
        fraction = _finite_float(self.validation_fraction, name='validation_fraction')
        if fraction <= 0.0 or fraction >= 1.0:
            raise ValueError('validation_fraction must be strictly between 0 and 1.')
        object.__setattr__(self, 'validation_fraction', fraction)
        object.__setattr__(self, 'device', torch.device(self.device))


@dataclass(frozen=True, slots=True)
class V2BCTrainingResult:
    train_loss_history: tuple[float, ...]
    validation_loss_history: tuple[float, ...]
    best_epoch: int
    best_validation_loss: float
    best_state_dict: dict[str, torch.Tensor]
    final_state_dict: dict[str, torch.Tensor]


def _cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state_dict.items()
    }


def _run_v2_bc_epoch(
    actor: V2ANNPolicyActor | V2SNNPolicyActor,
    cluster: V2BCTrajectoryCluster,
    scenario_ids: tuple[str, ...],
    *,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator | None,
) -> float:
    training = optimizer is not None
    if training:
        actor.train()
    else:
        actor.eval()
    weighted_loss = 0.0
    sample_count = 0
    iterator = iter_v2_bc_batches(
        cluster,
        scenario_ids,
        batch_size=batch_size,
        shuffle=training,
        rng=rng,
        device=device,
    )
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for observation_batch, expert_actions in iterator:
            predicted_actions = actor(observation_batch)
            loss = F.mse_loss(predicted_actions, expert_actions)
            loss_value = float(loss.detach().cpu().item())
            if not isfinite(loss_value):
                stage = 'training' if training else 'validation'
                raise RuntimeError(f'Non-finite V2 BC {stage} loss.')
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            current_batch = observation_batch.batch_size
            weighted_loss += loss_value * current_batch
            sample_count += current_batch
    if sample_count == 0:
        stage = 'training' if training else 'validation'
        raise RuntimeError(f'V2 BC {stage} split contains no step samples.')
    return weighted_loss / sample_count


def train_v2_bc_actor(
    actor: V2ANNPolicyActor | V2SNNPolicyActor,
    cluster: V2BCTrajectoryCluster,
    split: V2BCScenarioSplit,
    config: V2BCTrainingConfig,
    *,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> V2BCTrainingResult:
    """Train a strict V2 ANN or SNN actor for fixed epochs using MSE."""

    if not isinstance(actor, (V2ANNPolicyActor, V2SNNPolicyActor)):
        raise TypeError('actor must be a V2ANNPolicyActor or V2SNNPolicyActor.')
    if not isinstance(cluster, V2BCTrajectoryCluster):
        raise TypeError('cluster must be a V2BCTrajectoryCluster.')
    if not isinstance(split, V2BCScenarioSplit):
        raise TypeError('split must be a V2BCScenarioSplit.')
    if not isinstance(config, V2BCTrainingConfig):
        raise TypeError('config must be a V2BCTrainingConfig.')
    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError('epoch_callback must be callable when provided.')
    device = torch.device(config.device)
    actor.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    training_rng = np.random.default_rng(config.seed)
    train_history: list[float] = []
    validation_history: list[float] = []
    best = ValidationBestTracker()
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_v2_bc_epoch(
            actor,
            cluster,
            split.train_scenario_ids,
            batch_size=config.batch_size,
            device=device,
            optimizer=optimizer,
            rng=training_rng,
        )
        validation_loss = _run_v2_bc_epoch(
            actor,
            cluster,
            split.validation_scenario_ids,
            batch_size=config.batch_size,
            device=device,
            optimizer=None,
            rng=None,
        )
        train_history.append(train_loss)
        validation_history.append(validation_loss)
        refreshed_best = best.consider(
            epoch=epoch,
            validation_loss=validation_loss,
            actor=actor,
        )
        if epoch_callback is not None:
            epoch_callback({
                'epoch': epoch,
                'epochs': config.epochs,
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'best_epoch': best.best_epoch,
                'best_validation_loss': best.best_validation_loss,
                'refreshed_best': refreshed_best,
            })
    actor.train()
    if best.best_epoch is None:
        raise RuntimeError('V2 BC training did not produce a finite best checkpoint.')
    return V2BCTrainingResult(
        train_loss_history=tuple(train_history),
        validation_loss_history=tuple(validation_history),
        best_epoch=best.best_epoch,
        best_validation_loss=best.best_validation_loss,
        best_state_dict=best.best_state_dict,
        final_state_dict=_cpu_state_dict(actor.state_dict()),
    )


def v2_bc_observation_contract() -> dict[str, Any]:
    return {
        'id': V2_BC_OBSERVATION_CONTRACT_ID,
        'ego_dim': EGO_FEATURE_DIM,
        'goal_dim': GOAL_FEATURE_DIM,
        'zone_dim': ZONE_FEATURE_DIM,
        'presence_mask': True,
    }


def _cluster_scales(cluster: V2BCTrajectoryCluster) -> V2ObservationScales:
    scenario = cluster.scenario_config
    return V2ObservationScales(
        world_xy=float(scenario.world_xy),
        world_z_min=float(scenario.world_z_min),
        world_z_max=float(scenario.world_z_max),
        gamma_max=float(scenario.gamma_max),
    )


def _cluster_action_bounds(cluster: V2BCTrajectoryCluster) -> tuple[np.ndarray, np.ndarray]:
    scenario = cluster.scenario_config
    high = np.asarray(
        [scenario.delta_gamma_max, scenario.delta_psi_max],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(high)) or np.any(high <= 0.0):
        raise ValueError('ScenarioConfig action limits must be finite and positive.')
    return -high, high


def _actor_architecture(actor: V2ANNPolicyActor | V2SNNPolicyActor) -> dict[str, Any]:
    return {
        'scales': asdict(actor.scales),
        'encoder_config': asdict(actor.encoder_config),
        'uav_radius': actor.uav_radius,
        'actor_hidden_dim': actor.hidden_dim,
    }


def _validate_actor_for_cluster(
    actor: V2ANNPolicyActor | V2SNNPolicyActor,
    cluster: V2BCTrajectoryCluster,
) -> None:
    if actor.action_dim != 2:
        raise ValueError('V2 BC actor action_dim must be 2.')
    expected_scales = _cluster_scales(cluster)
    _, expected_high = _cluster_action_bounds(cluster)
    if actor.scales != expected_scales:
        raise ValueError('V2 BC actor scales do not match trajectory provenance.')
    if actor.uav_radius != cluster.uav_collision_radius:
        raise ValueError('V2 BC actor uav_radius does not match trajectory provenance.')
    if not torch.equal(
        actor.action_limit.detach().cpu(),
        torch.from_numpy(expected_high),
    ):
        raise ValueError('V2 BC actor action limits do not match trajectory provenance.')


def _snn_actor_architecture(actor: V2SNNPolicyActor) -> dict[str, Any]:
    architecture = _actor_architecture(actor)
    architecture.update({
        'time_window': actor.time_window,
        'tau': actor.tau,
        'surrogate': actor.surrogate_name,
        'backend': actor.backend,
    })
    return architecture


def _build_v2_bc_checkpoint_common(
    *,
    checkpoint_kind: str,
    actor: V2ANNPolicyActor | V2SNNPolicyActor,
    actor_state_dict: Mapping[str, torch.Tensor],
    cluster: V2BCTrajectoryCluster,
    split: V2BCScenarioSplit,
    config: V2BCTrainingConfig,
    result: V2BCTrainingResult,
    finished_at: str,
) -> dict[str, Any]:
    if checkpoint_kind not in ('best', 'final'):
        raise ValueError('checkpoint_kind must be "best" or "final".')
    _validate_actor_for_cluster(actor, cluster)
    timestamp = _nonempty_text(finished_at, name='finished_at')
    state = _cpu_state_dict(actor_state_dict)
    _validate_state_dict(actor, state, name='actor_state_dict')
    _validate_architecture_buffers(actor, state)
    action_low, action_high = _cluster_action_bounds(cluster)
    return {
        'format': V2_BC_CHECKPOINT_FORMAT,
        'format_version': V2_BC_CHECKPOINT_VERSION,
        'checkpoint_kind': checkpoint_kind,
        'observation_contract': v2_bc_observation_contract(),
        'architecture': _actor_architecture(actor),
        'action_dim': actor.action_dim,
        'action_low': action_low.tolist(),
        'action_high': action_high.tolist(),
        'actor_state_dict': state,
        'training_config': {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'seed': config.seed,
            'validation_fraction': config.validation_fraction,
        },
        'train_loss_history': list(result.train_loss_history),
        'validation_loss_history': list(result.validation_loss_history),
        'best_epoch': result.best_epoch,
        'best_validation_loss': result.best_validation_loss,
        'dataset_provenance': {
            'trajectory_cluster': str(cluster.root),
            'manifest_format': cluster.manifest['format'],
            'manifest_version': cluster.manifest['format_version'],
            'manifest': _strict_json_copy(cluster.manifest),
            'scenario_config': _strict_json_copy(cluster.manifest['scenario_config']),
            'uav_collision_radius': cluster.uav_collision_radius,
            'master_seed': cluster.manifest['master_seed'],
        },
        'split': split.to_dict(),
        'finished_at': timestamp,
    }


def build_v2_bc_checkpoint_payload(
    *,
    checkpoint_kind: str,
    actor: V2ANNPolicyActor,
    actor_state_dict: Mapping[str, torch.Tensor],
    cluster: V2BCTrajectoryCluster,
    split: V2BCScenarioSplit,
    config: V2BCTrainingConfig,
    result: V2BCTrainingResult,
    finished_at: str,
) -> dict[str, Any]:
    """Build the explicit, unchanged V2 ANN BC checkpoint schema."""

    if not isinstance(actor, V2ANNPolicyActor):
        raise TypeError('actor must be a V2ANNPolicyActor.')
    return _build_v2_bc_checkpoint_common(
        checkpoint_kind=checkpoint_kind,
        actor=actor,
        actor_state_dict=actor_state_dict,
        cluster=cluster,
        split=split,
        config=config,
        result=result,
        finished_at=finished_at,
    )


def build_v2_snn_bc_checkpoint_payload(
    *,
    checkpoint_kind: str,
    actor: V2SNNPolicyActor,
    actor_state_dict: Mapping[str, torch.Tensor],
    cluster: V2BCTrajectoryCluster,
    split: V2BCScenarioSplit,
    config: V2BCTrainingConfig,
    result: V2BCTrainingResult,
    finished_at: str,
) -> dict[str, Any]:
    """Build the strict SNN-specific V2 BC checkpoint schema."""

    if not isinstance(actor, V2SNNPolicyActor):
        raise TypeError('actor must be a V2SNNPolicyActor.')
    common = _build_v2_bc_checkpoint_common(
        checkpoint_kind=checkpoint_kind,
        actor=actor,
        actor_state_dict=actor_state_dict,
        cluster=cluster,
        split=split,
        config=config,
        result=result,
        finished_at=finished_at,
    )
    common['format'] = V2_SNN_BC_CHECKPOINT_FORMAT
    common['format_version'] = V2_SNN_BC_CHECKPOINT_VERSION
    common['model_type'] = 'snn'
    common['architecture'] = _snn_actor_architecture(actor)
    return common


_CHECKPOINT_FIELDS = {
    'format',
    'format_version',
    'checkpoint_kind',
    'observation_contract',
    'architecture',
    'action_dim',
    'action_low',
    'action_high',
    'actor_state_dict',
    'training_config',
    'train_loss_history',
    'validation_loss_history',
    'best_epoch',
    'best_validation_loss',
    'dataset_provenance',
    'split',
    'finished_at',
}
_SNN_CHECKPOINT_FIELDS = _CHECKPOINT_FIELDS | {'model_type'}


def _validate_state_dict(
    actor: nn.Module,
    state_dict: Any,
    *,
    name: str,
) -> None:
    if not isinstance(state_dict, Mapping):
        raise ValueError(f'{name} must be a state_dict mapping.')
    expected = actor.state_dict()
    if expected.keys() != state_dict.keys():
        raise ValueError(f'{name} keys do not match the V2 ANN actor architecture.')
    for key, expected_value in expected.items():
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'{name}[{key!r}] must be a tensor.')
        if value.shape != expected_value.shape or value.dtype != expected_value.dtype:
            raise ValueError(f'{name}[{key!r}] is incompatible with the V2 ANN actor.')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name}[{key!r}] must contain only finite values.')


def _validate_architecture_buffers(
    actor: V2ANNPolicyActor | V2SNNPolicyActor,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    for name, expected in actor.named_buffers():
        candidate = state_dict[name]
        if not torch.equal(candidate.detach().cpu(), expected.detach().cpu()):
            raise ValueError(
                f'actor_state_dict buffer {name!r} mismatches checkpoint architecture.'
            )


def _float_list(value: Any, *, length: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite vector.') from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be a finite vector of length {length}.')
    return array


def _history(value: Any, *, name: str, epochs: int) -> tuple[float, ...]:
    if type(value) is not list or len(value) != epochs:
        raise ValueError(f'{name} must contain one value per epoch.')
    return tuple(_finite_float(item, name=f'{name} value') for item in value)


def _validate_checkpoint_payload(
    payload: Any,
) -> tuple[
    V2ObservationScales,
    ZoneSetEncoderConfig,
    float,
    int,
    np.ndarray,
    np.ndarray,
]:
    if type(payload) is not dict:
        raise ValueError('V2 BC checkpoint payload must be a dictionary.')
    if payload.get('format') != V2_BC_CHECKPOINT_FORMAT:
        raise ValueError(
            f'checkpoint format must be {V2_BC_CHECKPOINT_FORMAT!r}; '
            'legacy flat BC checkpoints are not supported.'
        )
    if payload.get('format_version') != V2_BC_CHECKPOINT_VERSION:
        raise ValueError('Unsupported V2 BC checkpoint format_version.')
    if set(payload) != _CHECKPOINT_FIELDS:
        raise ValueError('V2 BC checkpoint has missing or unknown fields.')
    if payload['checkpoint_kind'] not in ('best', 'final'):
        raise ValueError('V2 BC checkpoint_kind is invalid.')
    if payload['observation_contract'] != v2_bc_observation_contract():
        raise ValueError('V2 BC observation contract is incompatible.')

    architecture = payload['architecture']
    if type(architecture) is not dict or set(architecture) != {
        'scales', 'encoder_config', 'uav_radius', 'actor_hidden_dim'
    }:
        raise ValueError('V2 BC checkpoint architecture is invalid.')
    try:
        scales = V2ObservationScales(**architecture['scales'])
        encoder_config = ZoneSetEncoderConfig(**architecture['encoder_config'])
    except (TypeError, ValueError) as exc:
        raise ValueError('V2 BC checkpoint architecture is incompatible.') from exc
    uav_radius = _nonnegative_float(architecture['uav_radius'], name='architecture uav_radius')
    hidden_dim = _positive_int(architecture['actor_hidden_dim'], name='actor_hidden_dim')
    action_dim = _positive_int(payload['action_dim'], name='action_dim')
    if action_dim != 2:
        raise ValueError('V2 BC checkpoint action_dim must be 2.')
    action_low = _float_list(payload['action_low'], length=action_dim, name='action_low')
    action_high = _float_list(payload['action_high'], length=action_dim, name='action_high')
    if not np.array_equal(action_low, -action_high) or np.any(action_high <= 0.0):
        raise ValueError('V2 BC checkpoint action range is incompatible with the actor contract.')

    training_config = payload['training_config']
    if type(training_config) is not dict or set(training_config) != {
        'epochs', 'batch_size', 'learning_rate', 'seed', 'validation_fraction'
    }:
        raise ValueError('V2 BC checkpoint training_config is invalid.')
    epochs = _positive_int(training_config['epochs'], name='training epochs')
    _positive_int(training_config['batch_size'], name='training batch_size')
    learning_rate = _finite_float(training_config['learning_rate'], name='training learning_rate')
    if learning_rate <= 0.0:
        raise ValueError('training learning_rate must be positive.')
    _nonnegative_int(training_config['seed'], name='training seed')
    fraction = _finite_float(
        training_config['validation_fraction'], name='training validation_fraction'
    )
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError('training validation_fraction must be in (0, 1).')
    train_history = _history(payload['train_loss_history'], name='train_loss_history', epochs=epochs)
    validation_history = _history(
        payload['validation_loss_history'], name='validation_loss_history', epochs=epochs
    )
    if any(value < 0.0 for value in train_history + validation_history):
        raise ValueError('V2 BC MSE loss histories must be non-negative.')
    best_epoch = _positive_int(payload['best_epoch'], name='best_epoch')
    best_loss = _finite_float(payload['best_validation_loss'], name='best_validation_loss')
    first_minimum_epoch = min(
        range(1, epochs + 1),
        key=lambda epoch: validation_history[epoch - 1],
    )
    if (
        best_epoch != first_minimum_epoch
        or best_loss != validation_history[best_epoch - 1]
    ):
        raise ValueError('best checkpoint metadata does not match validation loss history.')
    if any(not isfinite(value) for value in train_history + validation_history):
        raise ValueError('V2 BC loss histories must be finite.')

    provenance = payload['dataset_provenance']
    if type(provenance) is not dict or set(provenance) != {
        'trajectory_cluster',
        'manifest_format',
        'manifest_version',
        'manifest',
        'scenario_config',
        'uav_collision_radius',
        'master_seed',
    }:
        raise ValueError('V2 BC dataset provenance is invalid.')
    _nonempty_text(provenance['trajectory_cluster'], name='trajectory_cluster')
    if (
        provenance['manifest_format'] != V2_TRAJECTORY_CLUSTER_FORMAT
        or provenance['manifest_version'] != V2_TRAJECTORY_CLUSTER_VERSION
    ):
        raise ValueError('V2 BC dataset manifest provenance is incompatible.')
    manifest = provenance['manifest']
    if type(manifest) is not dict:
        raise ValueError('V2 BC dataset manifest snapshot is invalid.')
    _strict_json_copy(manifest)
    _validate_manifest(manifest)
    if (
        manifest.get('format') != provenance['manifest_format']
        or manifest.get('format_version') != provenance['manifest_version']
        or manifest.get('scenario_config') != provenance['scenario_config']
        or manifest.get('uav_collision_radius') != provenance['uav_collision_radius']
        or manifest.get('master_seed') != provenance['master_seed']
    ):
        raise ValueError('V2 BC dataset provenance fields mismatch its manifest snapshot.')
    scenario = _scenario_config_from_snapshot(provenance['scenario_config'])
    expected_scales = V2ObservationScales(
        float(scenario.world_xy),
        float(scenario.world_z_min),
        float(scenario.world_z_max),
        float(scenario.gamma_max),
    )
    expected_high = np.asarray(
        [scenario.delta_gamma_max, scenario.delta_psi_max], dtype=np.float32
    )
    provenance_radius = _nonnegative_float(
        provenance['uav_collision_radius'], name='provenance uav_collision_radius'
    )
    if scales != expected_scales:
        raise ValueError('V2 BC checkpoint scales mismatch dataset provenance.')
    if uav_radius != provenance_radius:
        raise ValueError('V2 BC checkpoint uav_radius mismatch dataset provenance.')
    if not np.array_equal(action_high, expected_high):
        raise ValueError('V2 BC checkpoint action bounds mismatch dataset provenance.')

    split = payload['split']
    if type(split) is not dict or set(split) != {
        'seed',
        'validation_fraction',
        'training_fraction',
        'train_scenario_ids',
        'validation_scenario_ids',
        'train_statistics',
        'validation_statistics',
    }:
        raise ValueError('V2 BC checkpoint split metadata is invalid.')
    train_ids = split['train_scenario_ids']
    validation_ids = split['validation_scenario_ids']
    if (
        type(train_ids) is not list
        or type(validation_ids) is not list
        or not train_ids
        or not validation_ids
        or any(type(value) is not str or not value for value in train_ids + validation_ids)
        or len(set(train_ids)) != len(train_ids)
        or len(set(validation_ids)) != len(validation_ids)
        or set(train_ids) & set(validation_ids)
    ):
        raise ValueError('V2 BC checkpoint split scenario IDs are invalid.')
    if split['seed'] != training_config['seed']:
        raise ValueError('V2 BC checkpoint split seed mismatch.')
    if split['validation_fraction'] != fraction:
        raise ValueError('V2 BC checkpoint split validation_fraction mismatch.')
    if split['training_fraction'] != 1.0 - fraction:
        raise ValueError('V2 BC checkpoint split training_fraction mismatch.')
    _strict_json_copy(split['train_statistics'])
    _strict_json_copy(split['validation_statistics'])
    _nonempty_text(payload['finished_at'], name='finished_at')
    return scales, encoder_config, uav_radius, hidden_dim, action_low, action_high


def _validate_expected_actor(
    expected_actor: V2ANNPolicyActor,
    *,
    scales: V2ObservationScales,
    encoder_config: ZoneSetEncoderConfig,
    uav_radius: float,
    hidden_dim: int,
    action_high: np.ndarray,
) -> None:
    if not isinstance(expected_actor, V2ANNPolicyActor):
        raise TypeError('expected_actor must be a V2ANNPolicyActor.')
    if (
        expected_actor.scales != scales
        or expected_actor.encoder_config != encoder_config
        or expected_actor.uav_radius != uav_radius
        or expected_actor.hidden_dim != hidden_dim
        or expected_actor.action_dim != int(action_high.shape[0])
        or not torch.equal(
            expected_actor.action_limit.detach().cpu(),
            torch.from_numpy(action_high),
        )
    ):
        raise ValueError('expected_actor architecture is incompatible with V2 BC checkpoint.')


def _validate_snn_checkpoint_payload(
    payload: Any,
) -> tuple[
    V2ObservationScales,
    ZoneSetEncoderConfig,
    float,
    int,
    np.ndarray,
    np.ndarray,
    int,
    float,
]:
    if type(payload) is not dict:
        raise ValueError('V2 SNN BC checkpoint payload must be a dictionary.')
    if payload.get('format') != V2_SNN_BC_CHECKPOINT_FORMAT:
        raise ValueError(
            f'checkpoint format must be {V2_SNN_BC_CHECKPOINT_FORMAT!r}; '
            'ANN and legacy flat BC checkpoints are not supported.'
        )
    if payload.get('format_version') != V2_SNN_BC_CHECKPOINT_VERSION:
        raise ValueError('Unsupported V2 SNN BC checkpoint format_version.')
    if set(payload) != _SNN_CHECKPOINT_FIELDS or payload.get('model_type') != 'snn':
        raise ValueError('V2 SNN BC checkpoint has missing, unknown, or invalid fields.')
    architecture = payload.get('architecture')
    expected_fields = {
        'scales',
        'encoder_config',
        'uav_radius',
        'actor_hidden_dim',
        'time_window',
        'tau',
        'surrogate',
        'backend',
    }
    if type(architecture) is not dict or set(architecture) != expected_fields:
        raise ValueError('V2 SNN BC checkpoint architecture is invalid.')
    time_window = _positive_int(architecture['time_window'], name='time_window')
    tau = _finite_float(architecture['tau'], name='tau')
    if tau <= 0.0:
        raise ValueError('V2 SNN BC checkpoint tau must be positive.')
    if architecture['surrogate'] != 'atan' or architecture['backend'] != 'torch':
        raise ValueError('V2 SNN BC checkpoint implementation is incompatible.')

    common = deepcopy(payload)
    common.pop('model_type')
    common['format'] = V2_BC_CHECKPOINT_FORMAT
    common['format_version'] = V2_BC_CHECKPOINT_VERSION
    common['architecture'] = {
        name: deepcopy(architecture[name])
        for name in ('scales', 'encoder_config', 'uav_radius', 'actor_hidden_dim')
    }
    scales, encoder, radius, hidden, low, high = _validate_checkpoint_payload(common)
    return scales, encoder, radius, hidden, low, high, time_window, tau


def _validate_expected_snn_actor(
    expected_actor: V2SNNPolicyActor,
    *,
    scales: V2ObservationScales,
    encoder_config: ZoneSetEncoderConfig,
    uav_radius: float,
    hidden_dim: int,
    action_high: np.ndarray,
    time_window: int,
    tau: float,
) -> None:
    if not isinstance(expected_actor, V2SNNPolicyActor):
        raise TypeError('expected_actor must be a V2SNNPolicyActor.')
    if (
        expected_actor.scales != scales
        or expected_actor.encoder_config != encoder_config
        or expected_actor.uav_radius != uav_radius
        or expected_actor.hidden_dim != hidden_dim
        or expected_actor.action_dim != int(action_high.shape[0])
        or expected_actor.time_window != time_window
        or expected_actor.tau != tau
        or expected_actor.surrogate_name != 'atan'
        or expected_actor.backend != 'torch'
        or not torch.equal(
            expected_actor.action_limit.detach().cpu(),
            torch.from_numpy(action_high),
        )
    ):
        raise ValueError(
            'expected_actor architecture is incompatible with V2 SNN BC checkpoint.'
        )


def load_v2_bc_actor_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = 'cpu',
    expected_actor: V2ANNPolicyActor | None = None,
) -> V2ANNPolicyActor:
    """Strictly load a V2 BC actor; legacy flat checkpoints are rejected."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'V2 BC checkpoint does not exist: {checkpoint_path}')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    scales, encoder_config, uav_radius, hidden_dim, _, action_high = (
        _validate_checkpoint_payload(payload)
    )
    if expected_actor is None:
        actor = V2ANNPolicyActor(
            scales=scales,
            action_dim=int(action_high.shape[0]),
            hidden_dim=hidden_dim,
            action_limit=torch.from_numpy(action_high.copy()),
            uav_radius=uav_radius,
            encoder_config=encoder_config,
        )
    else:
        _validate_expected_actor(
            expected_actor,
            scales=scales,
            encoder_config=encoder_config,
            uav_radius=uav_radius,
            hidden_dim=hidden_dim,
            action_high=action_high,
        )
        actor = expected_actor
    _validate_state_dict(actor, payload['actor_state_dict'], name='actor_state_dict')
    _validate_architecture_buffers(actor, payload['actor_state_dict'])
    actor.load_state_dict(payload['actor_state_dict'], strict=True)
    actor.to(torch.device(device))
    actor.eval()
    return actor


def load_v2_snn_bc_actor_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = 'cpu',
    expected_actor: V2SNNPolicyActor | None = None,
) -> V2SNNPolicyActor:
    """Strictly load a best V2 SNN BC actor; ANN checkpoints are rejected."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'V2 SNN BC checkpoint does not exist: {checkpoint_path}')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    (
        scales,
        encoder_config,
        uav_radius,
        hidden_dim,
        _,
        action_high,
        time_window,
        tau,
    ) = _validate_snn_checkpoint_payload(payload)
    if payload['checkpoint_kind'] != 'best':
        raise ValueError('V2 SNN TD3 initialization requires a best BC checkpoint.')
    if expected_actor is None:
        actor = V2SNNPolicyActor(
            scales=scales,
            action_dim=int(action_high.shape[0]),
            hidden_dim=hidden_dim,
            action_limit=torch.from_numpy(action_high.copy()),
            time_window=time_window,
            tau=tau,
            uav_radius=uav_radius,
            encoder_config=encoder_config,
        )
    else:
        _validate_expected_snn_actor(
            expected_actor,
            scales=scales,
            encoder_config=encoder_config,
            uav_radius=uav_radius,
            hidden_dim=hidden_dim,
            action_high=action_high,
            time_window=time_window,
            tau=tau,
        )
        actor = expected_actor
    _validate_state_dict(actor, payload['actor_state_dict'], name='actor_state_dict')
    _validate_architecture_buffers(actor, payload['actor_state_dict'])
    actor.load_state_dict(payload['actor_state_dict'], strict=True)
    actor.to(torch.device(device))
    actor.eval()
    return actor


__all__ = [
    'V2_BC_CHECKPOINT_FORMAT',
    'V2_BC_CHECKPOINT_VERSION',
    'V2_SNN_BC_CHECKPOINT_FORMAT',
    'V2_SNN_BC_CHECKPOINT_VERSION',
    'V2BCScenarioSplit',
    'V2BCShardRecord',
    'V2BCTrainingConfig',
    'V2BCTrainingResult',
    'V2BCTrajectoryCluster',
    'V2BCTrajectoryRecord',
    'V2ShardArrayCache',
    'ValidationBestTracker',
    'assemble_v2_bc_batch',
    'build_v2_bc_checkpoint_payload',
    'build_v2_snn_bc_checkpoint_payload',
    'iter_v2_bc_batches',
    'load_v2_bc_actor_checkpoint',
    'load_v2_snn_bc_actor_checkpoint',
    'load_v2_bc_trajectory_cluster',
    'restore_v2_observation',
    'split_v2_bc_scenarios',
    'train_v2_bc_actor',
    'v2_bc_observation_contract',
]
