"""Validated ragged storage for successful V2 expert trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from brain_uav.observations import (
    EGO_FEATURE_DIM,
    GOAL_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    V2Observation,
)


V2_TRAJECTORY_SHARD_FORMAT = 'v2_easy_expert_trajectory_shard'
V2_TRAJECTORY_SHARD_VERSION = 2
_V2_TRAJECTORY_SHARD_FIELDS = {
    'format',
    'format_version',
    'ego_features',
    'goal_features',
    'zone_features',
    'zone_offsets',
    'actions',
    'states_before_action',
    'trajectory_offsets',
    'trajectory_ids',
    'scenario_ids',
    'scenario_payload_refs',
    'planner_names',
    'outcomes',
    'step_counts',
    'terminal_states',
    'scenario_seeds',
}


def _nonempty_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f'{name} must be a non-empty string.')
    return value


def _finite_float32_array(
    value: Any,
    *,
    shape: tuple[int | None, ...],
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite float32 array.') from exc
    shape_matches = array.ndim == len(shape) and all(
        expected is None or actual == expected
        for actual, expected in zip(array.shape, shape)
    )
    if not shape_matches or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must have shape {shape} and finite values.')
    result = array.copy()
    result.setflags(write=False)
    return result


def _offsets(value: Any, *, length: int, final: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain integers.')
    array = np.asarray(array, dtype=np.int64)
    if array.shape != (length,) or array[0] != 0:
        raise ValueError(f'{name} must have shape ({length},) and start at zero.')
    if np.any(array[1:] < array[:-1]) or int(array[-1]) != final:
        raise ValueError(f'{name} must be monotonic and end at {final}.')
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SuccessfulV2Trajectory:
    """One validated goal trajectory with ragged per-step zone rows."""

    trajectory_id: str
    scenario_id: str
    planner_name: str
    scenario_seed: int
    outcome: str
    terminal_state: np.ndarray
    ego_features: np.ndarray
    goal_features: np.ndarray
    zone_features: np.ndarray
    zone_offsets: np.ndarray
    actions: np.ndarray
    states_before_action: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'trajectory_id',
            _nonempty_text(self.trajectory_id, name='trajectory_id'),
        )
        object.__setattr__(
            self,
            'scenario_id',
            _nonempty_text(self.scenario_id, name='scenario_id'),
        )
        object.__setattr__(
            self,
            'planner_name',
            _nonempty_text(self.planner_name, name='planner_name'),
        )
        if type(self.scenario_seed) is not int or self.scenario_seed < 0:
            raise ValueError('scenario_seed must be a non-negative integer.')
        if self.outcome != 'goal':
            raise ValueError('Only outcome="goal" trajectories may be stored.')
        terminal_state = _finite_float32_array(
            self.terminal_state,
            shape=(5,),
            name='terminal_state',
        )
        ego = _finite_float32_array(
            self.ego_features,
            shape=(None, EGO_FEATURE_DIM),
            name='ego_features',
        )
        step_count = ego.shape[0]
        if step_count < 1:
            raise ValueError('A successful trajectory must contain at least one step.')
        goal = _finite_float32_array(
            self.goal_features,
            shape=(step_count, GOAL_FEATURE_DIM),
            name='goal_features',
        )
        zones = _finite_float32_array(
            self.zone_features,
            shape=(None, ZONE_FEATURE_DIM),
            name='zone_features',
        )
        zone_offsets = _offsets(
            self.zone_offsets,
            length=step_count + 1,
            final=zones.shape[0],
            name='zone_offsets',
        )
        actions = _finite_float32_array(
            self.actions,
            shape=(step_count, 2),
            name='actions',
        )
        states = _finite_float32_array(
            self.states_before_action,
            shape=(step_count, 5),
            name='states_before_action',
        )
        object.__setattr__(self, 'terminal_state', terminal_state)
        object.__setattr__(self, 'ego_features', ego)
        object.__setattr__(self, 'goal_features', goal)
        object.__setattr__(self, 'zone_features', zones)
        object.__setattr__(self, 'zone_offsets', zone_offsets)
        object.__setattr__(self, 'actions', actions)
        object.__setattr__(self, 'states_before_action', states)

    @property
    def step_count(self) -> int:
        return int(self.ego_features.shape[0])

    @property
    def scenario_payload_ref(self) -> str:
        return self.scenario_id


def build_successful_v2_trajectory(
    *,
    trajectory_id: str,
    scenario_id: str,
    planner_name: str,
    scenario_seed: int,
    observations: Sequence[V2Observation],
    states_before_action: Any,
    actions: Any,
    terminal_state: Any,
    outcome: str,
) -> SuccessfulV2Trajectory:
    """Copy structured observations and pack their zone rows without padding."""

    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise TypeError('observations must be a sequence of V2Observation values.')
    if not observations:
        raise ValueError('A successful trajectory must contain at least one observation.')
    ego_rows: list[np.ndarray] = []
    goal_rows: list[np.ndarray] = []
    zone_rows: list[np.ndarray] = []
    zone_offsets = [0]
    for index, observation in enumerate(observations):
        if not isinstance(observation, V2Observation):
            raise TypeError(f'observations[{index}] must be a V2Observation.')
        ego_rows.append(np.asarray(observation.ego_features, dtype=np.float32).copy())
        goal_rows.append(np.asarray(observation.goal_features, dtype=np.float32).copy())
        zones = np.asarray(observation.zone_features, dtype=np.float32).copy()
        if zones.shape[0]:
            zone_rows.append(zones)
        zone_offsets.append(zone_offsets[-1] + zones.shape[0])
    packed_zones = (
        np.concatenate(zone_rows, axis=0)
        if zone_rows
        else np.empty((0, ZONE_FEATURE_DIM), dtype=np.float32)
    )
    return SuccessfulV2Trajectory(
        trajectory_id=trajectory_id,
        scenario_id=scenario_id,
        planner_name=planner_name,
        scenario_seed=scenario_seed,
        outcome=outcome,
        terminal_state=terminal_state,
        ego_features=np.stack(ego_rows).astype(np.float32, copy=False),
        goal_features=np.stack(goal_rows).astype(np.float32, copy=False),
        zone_features=packed_zones,
        zone_offsets=np.asarray(zone_offsets, dtype=np.int64),
        actions=actions,
        states_before_action=states_before_action,
    )


class V2TrajectoryShardBuffer:
    """Hold only the current shard's successful trajectories in memory."""

    def __init__(self) -> None:
        self._trajectories: list[SuccessfulV2Trajectory] = []

    @property
    def trajectory_count(self) -> int:
        return len(self._trajectories)

    @property
    def step_count(self) -> int:
        return sum(item.step_count for item in self._trajectories)

    def add(self, trajectory: SuccessfulV2Trajectory) -> None:
        if not isinstance(trajectory, SuccessfulV2Trajectory):
            raise TypeError('trajectory must be a SuccessfulV2Trajectory.')
        self._trajectories.append(trajectory)

    def flush(self, path: str | Path) -> dict[str, int]:
        if not self._trajectories:
            raise ValueError('Cannot write an empty V2 trajectory shard.')
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite existing shard: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = self._pack()
        _validate_shard_arrays(arrays)
        np.savez_compressed(destination, **arrays)
        summary = {
            'trajectory_count': self.trajectory_count,
            'step_count': self.step_count,
            'zone_token_count': int(arrays['zone_features'].shape[0]),
        }
        self._trajectories.clear()
        return summary

    def _pack(self) -> dict[str, np.ndarray]:
        trajectories = self._trajectories
        ego = np.concatenate([item.ego_features for item in trajectories], axis=0)
        goal = np.concatenate([item.goal_features for item in trajectories], axis=0)
        actions = np.concatenate([item.actions for item in trajectories], axis=0)
        states = np.concatenate(
            [item.states_before_action for item in trajectories],
            axis=0,
        )
        zone_parts = [item.zone_features for item in trajectories if item.zone_features.shape[0]]
        zones = (
            np.concatenate(zone_parts, axis=0)
            if zone_parts
            else np.empty((0, ZONE_FEATURE_DIM), dtype=np.float32)
        )
        zone_offsets = [0]
        trajectory_offsets = [0]
        zone_total = 0
        for trajectory in trajectories:
            zone_offsets.extend(
                (trajectory.zone_offsets[1:] + zone_total).astype(np.int64).tolist()
            )
            zone_total += trajectory.zone_features.shape[0]
            trajectory_offsets.append(trajectory_offsets[-1] + trajectory.step_count)
        return {
            'format': np.asarray(V2_TRAJECTORY_SHARD_FORMAT),
            'format_version': np.asarray(V2_TRAJECTORY_SHARD_VERSION, dtype=np.int64),
            'ego_features': ego.astype(np.float32, copy=False),
            'goal_features': goal.astype(np.float32, copy=False),
            'zone_features': zones.astype(np.float32, copy=False),
            'zone_offsets': np.asarray(zone_offsets, dtype=np.int64),
            'actions': actions.astype(np.float32, copy=False),
            'states_before_action': states.astype(np.float32, copy=False),
            'trajectory_offsets': np.asarray(trajectory_offsets, dtype=np.int64),
            'trajectory_ids': np.asarray([item.trajectory_id for item in trajectories]),
            'scenario_ids': np.asarray([item.scenario_id for item in trajectories]),
            'scenario_payload_refs': np.asarray(
                [item.scenario_payload_ref for item in trajectories]
            ),
            'planner_names': np.asarray([item.planner_name for item in trajectories]),
            'outcomes': np.asarray([item.outcome for item in trajectories]),
            'step_counts': np.asarray(
                [item.step_count for item in trajectories],
                dtype=np.int64,
            ),
            'terminal_states': np.stack(
                [item.terminal_state for item in trajectories]
            ).astype(np.float32, copy=False),
            'scenario_seeds': np.asarray(
                [item.scenario_seed for item in trajectories],
                dtype=np.int64,
            ),
        }


def _validate_shard_arrays(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != _V2_TRAJECTORY_SHARD_FIELDS:
        raise ValueError('V2 trajectory shard has missing or unknown arrays.')
    ego = arrays['ego_features']
    goal = arrays['goal_features']
    zones = arrays['zone_features']
    actions = arrays['actions']
    states = arrays['states_before_action']
    total_steps = ego.shape[0]
    if ego.shape != (total_steps, EGO_FEATURE_DIM):
        raise ValueError('ego_features has an invalid shard shape.')
    if goal.shape != (total_steps, GOAL_FEATURE_DIM):
        raise ValueError('goal_features has an invalid shard shape.')
    if zones.ndim != 2 or zones.shape[1] != ZONE_FEATURE_DIM:
        raise ValueError('zone_features has an invalid shard shape.')
    if actions.shape != (total_steps, 2) or states.shape != (total_steps, 5):
        raise ValueError('Actions and states must align with shard step count.')
    for name in (
        'ego_features',
        'goal_features',
        'zone_features',
        'actions',
        'states_before_action',
        'terminal_states',
    ):
        if arrays[name].dtype != np.float32 or not np.all(np.isfinite(arrays[name])):
            raise ValueError(f'{name} must contain finite float32 values.')
    zone_offsets = arrays['zone_offsets']
    trajectory_offsets = arrays['trajectory_offsets']
    if (
        zone_offsets.shape != (total_steps + 1,)
        or zone_offsets[0] != 0
        or np.any(zone_offsets[1:] < zone_offsets[:-1])
        or int(zone_offsets[-1]) != zones.shape[0]
    ):
        raise ValueError('zone_offsets is inconsistent with zone_features.')
    trajectory_count = arrays['trajectory_ids'].shape[0]
    if (
        trajectory_offsets.shape != (trajectory_count + 1,)
        or trajectory_offsets[0] != 0
        or np.any(trajectory_offsets[1:] <= trajectory_offsets[:-1])
        or int(trajectory_offsets[-1]) != total_steps
    ):
        raise ValueError('trajectory_offsets is inconsistent with shard steps.')
    per_trajectory_names = (
        'trajectory_ids',
        'scenario_ids',
        'scenario_payload_refs',
        'planner_names',
        'outcomes',
        'step_counts',
        'scenario_seeds',
    )
    if any(arrays[name].shape != (trajectory_count,) for name in per_trajectory_names):
        raise ValueError('Per-trajectory shard arrays must have matching lengths.')
    if arrays['terminal_states'].shape != (trajectory_count, 5):
        raise ValueError('terminal_states must have shape [trajectory_count, 5].')
    if not np.array_equal(arrays['step_counts'], np.diff(trajectory_offsets)):
        raise ValueError('step_counts do not match trajectory_offsets.')
    for name in ('trajectory_ids', 'scenario_ids', 'scenario_payload_refs', 'planner_names'):
        if arrays[name].dtype.kind not in 'US' or np.any(arrays[name] == ''):
            raise ValueError(f'{name} must contain non-empty strings.')
    if arrays['outcomes'].dtype.kind not in 'US':
        raise ValueError('outcomes must be a string array.')
    if arrays['step_counts'].dtype != np.int64 or np.any(arrays['step_counts'] < 1):
        raise ValueError('step_counts must contain positive int64 values.')
    if arrays['scenario_seeds'].dtype != np.int64 or np.any(arrays['scenario_seeds'] < 0):
        raise ValueError('scenario_seeds must contain non-negative int64 values.')
    if not np.all(arrays['outcomes'] == 'goal'):
        raise ValueError('A V2 expert shard may contain only goal trajectories.')
    for name, value in arrays.items():
        if value.dtype == object:
            raise ValueError(f'{name} must not use an object dtype.')


def load_v2_trajectory_shard(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a shard without enabling pickle."""

    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if str(arrays.get('format')) != V2_TRAJECTORY_SHARD_FORMAT:
        raise ValueError('Unsupported V2 trajectory shard format.')
    version = arrays.get('format_version')
    if version is None or version.shape != () or int(version) != V2_TRAJECTORY_SHARD_VERSION:
        raise ValueError('Unsupported V2 trajectory shard version.')
    _validate_shard_arrays(arrays)
    return arrays


__all__ = [
    'SuccessfulV2Trajectory',
    'V2TrajectoryShardBuffer',
    'V2_TRAJECTORY_SHARD_FORMAT',
    'V2_TRAJECTORY_SHARD_VERSION',
    'build_successful_v2_trajectory',
    'load_v2_trajectory_shard',
]
