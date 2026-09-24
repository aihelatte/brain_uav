"""Structured replay storage for dynamic V2 observations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any

import numpy as np
import torch

from brain_uav.observations import (
    EGO_FEATURE_DIM,
    GOAL_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    V2Observation,
    V2ObservationBatch,
)


V2_REPLAY_SAMPLING_IMPLEMENTATION = 'fenwick_ppswor_v1'


class _FenwickWeightTree:
    """Fenwick tree for exact sequential weighted sampling."""

    __slots__ = ('capacity', '_tree')

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree = np.zeros(capacity + 1, dtype=np.float64)

    def add(self, index: int, delta: float) -> None:
        tree_index = index + 1
        while tree_index <= self.capacity:
            self._tree[tree_index] += delta
            tree_index += tree_index & -tree_index

    @property
    def total(self) -> float:
        result = 0.0
        tree_index = self.capacity
        while tree_index > 0:
            result += float(self._tree[tree_index])
            tree_index -= tree_index & -tree_index
        return result

    def find_by_cumulative_weight(self, target: float) -> int:
        """Return the first zero-based index whose cumulative weight exceeds target."""

        index = 0
        bit = 1 << (self.capacity.bit_length() - 1)
        remaining = target
        while bit:
            candidate = index + bit
            if candidate <= self.capacity and self._tree[candidate] <= remaining:
                index = candidate
                remaining -= float(self._tree[candidate])
            bit >>= 1
        if index >= self.capacity:
            raise RuntimeError('Replay cumulative weight lookup exceeded capacity.')
        return index

    def remove_temporarily(
        self,
        index: int,
        weight: float,
        original_nodes: dict[int, float],
    ) -> None:
        tree_index = index + 1
        while tree_index <= self.capacity:
            original_nodes.setdefault(tree_index, float(self._tree[tree_index]))
            self._tree[tree_index] -= weight
            tree_index += tree_index & -tree_index

    def restore_nodes(self, original_nodes: dict[int, float]) -> None:
        for tree_index, value in original_nodes.items():
            self._tree[tree_index] = value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer.')
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a positive integer.') from exc
    if result <= 0 or result != value:
        raise ValueError(f'{name} must be a positive integer.')
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a non-negative integer.')
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a non-negative integer.') from exc
    if result < 0 or result != value:
        raise ValueError(f'{name} must be a non-negative integer.')
    return result


def _finite_at_least_one(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number greater than or equal to 1.') from exc
    if not isfinite(result) or result < 1.0:
        raise ValueError(f'{name} must be a finite number greater than or equal to 1.')
    return result


def _finite_fraction(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number in [0, 1].') from exc
    if not isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f'{name} must be a finite number in [0, 1].')
    return result


def _optional_seed(value: Any) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name='seed')


@dataclass(frozen=True, slots=True, eq=False)
class V2ReplayBatch:
    """One sampled V2 replay batch with independently padded states."""

    obs: V2ObservationBatch
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: V2ObservationBatch
    done: torch.Tensor
    success: torch.Tensor
    near_goal: torch.Tensor
    line_to_goal_safe: torch.Tensor
    sample_failure_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.obs, V2ObservationBatch):
            raise TypeError('obs must be a V2ObservationBatch.')
        if not isinstance(self.next_obs, V2ObservationBatch):
            raise TypeError('next_obs must be a V2ObservationBatch.')
        batch_size = self.obs.batch_size
        if self.next_obs.batch_size != batch_size:
            raise ValueError('obs and next_obs batch sizes must match.')

        tensor_fields = (
            ('action', self.action),
            ('reward', self.reward),
            ('done', self.done),
            ('success', self.success),
            ('near_goal', self.near_goal),
            ('line_to_goal_safe', self.line_to_goal_safe),
        )
        for name, value in tensor_fields:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'{name} must be a torch.Tensor.')
            if value.dtype != torch.float32:
                raise TypeError(f'{name} must have dtype torch.float32.')
        if self.action.ndim != 2 or self.action.shape[0] != batch_size:
            raise ValueError('action must have shape (B, action_dim).')
        if self.action.shape[1] <= 0:
            raise ValueError('action_dim must be greater than zero.')
        for name, value in tensor_fields[1:]:
            if value.shape != (batch_size, 1):
                raise ValueError(f'{name} must have shape ({batch_size}, 1).')

        device = self.obs.ego_features.device
        if self.next_obs.ego_features.device != device:
            raise ValueError('obs and next_obs must be on the same device.')
        for name, value in tensor_fields:
            if value.device != device:
                raise ValueError(
                    f'All replay tensors must share one device; {name} is on '
                    f'{value.device}, expected {device}.'
                )
        failure_fraction = _finite_fraction(
            self.sample_failure_fraction,
            name='sample_failure_fraction',
        )
        object.__setattr__(self, 'sample_failure_fraction', failure_fraction)

    @property
    def batch_size(self) -> int:
        return self.obs.batch_size

    def to(self, device: torch.device | str) -> V2ReplayBatch:
        """Return a new replay batch on ``device`` without reading tensor data."""

        target = torch.device(device)
        return V2ReplayBatch(
            obs=self.obs.to(target),
            action=self.action.to(target),
            reward=self.reward.to(target),
            next_obs=self.next_obs.to(target),
            done=self.done.to(target),
            success=self.success.to(target),
            near_goal=self.near_goal.to(target),
            line_to_goal_safe=self.line_to_goal_safe.to(target),
            sample_failure_fraction=self.sample_failure_fraction,
        )


class V2ReplayBuffer:
    """Preallocated ring buffer for dynamic structured V2 observations.

    ``zone_storage_capacity`` limits only preallocated replay memory. It is
    not passed to the policy and is not a model capacity.
    """

    def __init__(
        self,
        capacity: int,
        action_dim: int,
        zone_storage_capacity: int,
        success_sample_bias: float = 1.0,
        near_goal_sample_bias: float = 1.0,
        success_replay_fraction: float = 0.25,
        success_batch_fraction: float = 0.25,
        *,
        failure_sample_bias: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.capacity = _positive_int(capacity, name='capacity')
        self.action_dim = _positive_int(action_dim, name='action_dim')
        self.zone_storage_capacity = _nonnegative_int(
            zone_storage_capacity,
            name='zone_storage_capacity',
        )
        self.success_sample_bias = _finite_at_least_one(
            success_sample_bias,
            name='success_sample_bias',
        )
        self.near_goal_sample_bias = _finite_at_least_one(
            near_goal_sample_bias,
            name='near_goal_sample_bias',
        )
        self.failure_sample_bias = _finite_at_least_one(
            failure_sample_bias,
            name='failure_sample_bias',
        )
        self.success_replay_fraction = _finite_fraction(
            success_replay_fraction,
            name='success_replay_fraction',
        )
        self.success_batch_fraction = _finite_fraction(
            success_batch_fraction,
            name='success_batch_fraction',
        )
        self.seed = _optional_seed(seed)
        self.rng = np.random.default_rng(self.seed)

        self.ego_features = np.zeros(
            (self.capacity, EGO_FEATURE_DIM), dtype=np.float32
        )
        self.goal_features = np.zeros(
            (self.capacity, GOAL_FEATURE_DIM), dtype=np.float32
        )
        self.zone_features = np.zeros(
            (self.capacity, self.zone_storage_capacity, ZONE_FEATURE_DIM),
            dtype=np.float32,
        )
        self.zone_count = np.zeros(self.capacity, dtype=np.int32)
        self.next_ego_features = np.zeros_like(self.ego_features)
        self.next_goal_features = np.zeros_like(self.goal_features)
        self.next_zone_features = np.zeros_like(self.zone_features)
        self.next_zone_count = np.zeros_like(self.zone_count)
        self.action = np.zeros(
            (self.capacity, self.action_dim), dtype=np.float32
        )
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.success = np.zeros(self.capacity, dtype=np.bool_)
        self.failure = np.zeros(self.capacity, dtype=np.bool_)
        self.near_goal = np.zeros(self.capacity, dtype=np.bool_)
        self.line_to_goal_safe = np.zeros(self.capacity, dtype=np.bool_)
        self.sample_weight = np.zeros(self.capacity, dtype=np.float64)
        self._sampling_weight_tree = _FenwickWeightTree(self.capacity)
        self.write_id = np.full(self.capacity, -1, dtype=np.int64)
        self.next_write_id = 0
        self.size = 0
        self.position = 0
        self.success_count = 0
        self.near_goal_count = 0
        self.total_sample_weight = 0.0

        self.success_capacity = int(
            self.capacity * self.success_replay_fraction
        )
        self.success_ego_features = np.zeros(
            (self.success_capacity, EGO_FEATURE_DIM), dtype=np.float32
        )
        self.success_goal_features = np.zeros(
            (self.success_capacity, GOAL_FEATURE_DIM), dtype=np.float32
        )
        self.success_zone_features = np.zeros(
            (
                self.success_capacity,
                self.zone_storage_capacity,
                ZONE_FEATURE_DIM,
            ),
            dtype=np.float32,
        )
        self.success_zone_count = np.zeros(
            self.success_capacity, dtype=np.int32
        )
        self.success_next_ego_features = np.zeros_like(
            self.success_ego_features
        )
        self.success_next_goal_features = np.zeros_like(
            self.success_goal_features
        )
        self.success_next_zone_features = np.zeros_like(
            self.success_zone_features
        )
        self.success_next_zone_count = np.zeros_like(
            self.success_zone_count
        )
        self.success_action = np.zeros(
            (self.success_capacity, self.action_dim), dtype=np.float32
        )
        self.success_reward = np.zeros(
            (self.success_capacity, 1), dtype=np.float32
        )
        self.success_done = np.zeros(
            (self.success_capacity, 1), dtype=np.float32
        )
        self.success_near_goal = np.zeros(
            self.success_capacity, dtype=np.bool_
        )
        self.success_line_to_goal_safe = np.zeros(
            self.success_capacity, dtype=np.bool_
        )
        self.success_size = 0
        self.success_position = 0

    def __len__(self) -> int:
        return self.size

    def _validate_transition(
        self,
        obs: V2Observation,
        action: np.ndarray,
        reward: float,
        next_obs: V2Observation,
    ) -> tuple[np.ndarray, float]:
        if not isinstance(obs, V2Observation):
            raise TypeError('obs must be a V2Observation.')
        if not isinstance(next_obs, V2Observation):
            raise TypeError('next_obs must be a V2Observation.')
        for name, observation in (('obs', obs), ('next_obs', next_obs)):
            zone_count = int(observation.zone_features.shape[0])
            if zone_count > self.zone_storage_capacity:
                raise ValueError(
                    f'{name} has {zone_count} zones, exceeding '
                    f'zone_storage_capacity={self.zone_storage_capacity}.'
                )
        try:
            action_array = np.array(action, dtype=np.float32, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('action must be a finite float32 vector.') from exc
        if action_array.shape != (self.action_dim,):
            raise ValueError(
                f'action must have shape ({self.action_dim},); '
                f'got {action_array.shape}.'
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError('action must contain only finite values.')
        try:
            reward_value = float(reward)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('reward must be finite.') from exc
        if not isfinite(reward_value):
            raise ValueError('reward must be finite.')
        return action_array, reward_value

    @staticmethod
    def _write_observation(
        observation: V2Observation,
        index: int,
        ego_storage: np.ndarray,
        goal_storage: np.ndarray,
        zone_storage: np.ndarray,
        count_storage: np.ndarray,
    ) -> None:
        zone_count = int(observation.zone_features.shape[0])
        ego_storage[index] = observation.ego_features
        goal_storage[index] = observation.goal_features
        zone_storage[index].fill(0.0)
        if zone_count > 0:
            zone_storage[index, :zone_count] = observation.zone_features
        count_storage[index] = zone_count

    def add(
        self,
        obs: V2Observation,
        action: np.ndarray,
        reward: float,
        next_obs: V2Observation,
        done: bool,
        success: bool = False,
        near_goal: bool = False,
        line_to_goal_safe: bool = False,
    ) -> tuple[int, int]:
        action_array, reward_value = self._validate_transition(
            obs, action, reward, next_obs
        )
        index = self.position
        write_id = int(self.next_write_id)
        self.next_write_id += 1
        if self.size == self.capacity:
            if self.success[index]:
                self.success_count -= 1
            if self.near_goal[index]:
                self.near_goal_count -= 1
        else:
            self.size += 1

        self._write_observation(
            obs,
            index,
            self.ego_features,
            self.goal_features,
            self.zone_features,
            self.zone_count,
        )
        self._write_observation(
            next_obs,
            index,
            self.next_ego_features,
            self.next_goal_features,
            self.next_zone_features,
            self.next_zone_count,
        )
        self.action[index] = action_array
        self.reward[index, 0] = reward_value
        self.done[index, 0] = float(bool(done))
        self.success[index] = bool(success)
        self.failure[index] = False
        self.near_goal[index] = bool(near_goal)
        self.line_to_goal_safe[index] = bool(line_to_goal_safe)
        self.write_id[index] = write_id
        if self.success[index]:
            self.success_count += 1
        if self.near_goal[index]:
            self.near_goal_count += 1
        self._set_primary_slot_weight(index, self._slot_weight(index))
        self.position = (self.position + 1) % self.capacity
        return index, write_id

    def mark_success_slots(
        self,
        slot_refs: list[tuple[int, int]],
        success: bool = True,
    ) -> int:
        """Update success flags for primary slots that have not been replaced."""

        updated = 0
        for index, write_id in slot_refs:
            index = int(index)
            if index < 0 or index >= self.capacity:
                continue
            if int(self.write_id[index]) != int(write_id):
                continue
            current = bool(self.success[index])
            desired = bool(success)
            if current == desired:
                if desired and self.failure[index]:
                    self.failure[index] = False
                    self._set_primary_slot_weight(index, self._slot_weight(index))
                    updated += 1
                continue
            self.success[index] = desired
            if desired:
                self.failure[index] = False
            self.success_count += 1 if desired else -1
            self._set_primary_slot_weight(index, self._slot_weight(index))
            updated += 1
        return updated

    def mark_failure_slots(
        self,
        slot_refs: list[tuple[int, int]],
        failure: bool = True,
    ) -> int:
        """Update confirmed-failure flags for live primary slots only."""

        updated = 0
        desired = bool(failure)
        for index, write_id in slot_refs:
            index = int(index)
            if index < 0 or index >= self.capacity:
                continue
            if int(self.write_id[index]) != int(write_id):
                continue
            current_failure = bool(self.failure[index])
            current_success = bool(self.success[index])
            new_success = False if desired else current_success
            if current_failure == desired and current_success == new_success:
                continue
            self.failure[index] = desired
            if current_success != new_success:
                self.success[index] = new_success
                self.success_count += 1 if new_success else -1
            self._set_primary_slot_weight(index, self._slot_weight(index))
            updated += 1
        return updated

    def add_success_transition(
        self,
        obs: V2Observation,
        action: np.ndarray,
        reward: float,
        next_obs: V2Observation,
        done: bool,
        near_goal: bool = False,
        line_to_goal_safe: bool = False,
    ) -> None:
        action_array, reward_value = self._validate_transition(
            obs, action, reward, next_obs
        )
        if self.success_capacity <= 0:
            return
        index = self.success_position
        if self.success_size < self.success_capacity:
            self.success_size += 1
        self._write_observation(
            obs,
            index,
            self.success_ego_features,
            self.success_goal_features,
            self.success_zone_features,
            self.success_zone_count,
        )
        self._write_observation(
            next_obs,
            index,
            self.success_next_ego_features,
            self.success_next_goal_features,
            self.success_next_zone_features,
            self.success_next_zone_count,
        )
        self.success_action[index] = action_array
        self.success_reward[index, 0] = reward_value
        self.success_done[index, 0] = float(bool(done))
        self.success_near_goal[index] = bool(near_goal)
        self.success_line_to_goal_safe[index] = bool(line_to_goal_safe)
        self.success_position = (index + 1) % self.success_capacity

    def success_fraction(self) -> float:
        if self.size == 0:
            return 0.0
        return float(self.success_count / self.size)

    def near_goal_fraction(self) -> float:
        if self.size == 0:
            return 0.0
        return float(self.near_goal_count / self.size)

    def _slot_weight(self, index: int) -> float:
        weight = 1.0
        if self.success_sample_bias > 1.0 and self.success[index]:
            weight *= self.success_sample_bias
        if self.near_goal_sample_bias > 1.0 and self.near_goal[index]:
            weight *= self.near_goal_sample_bias
        if self.failure_sample_bias > 1.0 and self.failure[index]:
            weight *= self.failure_sample_bias
        return float(weight)

    def _set_primary_slot_weight(self, index: int, weight: float) -> None:
        previous = float(self.sample_weight[index])
        delta = weight - previous
        self.sample_weight[index] = weight
        self._sampling_weight_tree.add(index, delta)
        self.total_sample_weight += delta

    @property
    def sampling_implementation(self) -> str:
        return V2_REPLAY_SAMPLING_IMPLEMENTATION

    @property
    def sampling_weight_total(self) -> float:
        """Current primary replay weight represented by the sampling tree."""

        return self._sampling_weight_tree.total

    def _sample_primary_indices(self, batch_size: int) -> np.ndarray:
        """Sequential probability-proportional-to-weight sampling without replacement."""

        if batch_size <= 0 or batch_size > self.size:
            raise ValueError('primary batch_size must be in [1, replay size].')
        represented_total = self._sampling_weight_tree.total
        if (
            not isfinite(represented_total)
            or represented_total <= 0.0
            or not isclose(
                represented_total,
                self.total_sample_weight,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError('Replay cumulative weight state is inconsistent.')

        indices = np.empty(batch_size, dtype=np.int64)
        original_nodes: dict[int, float] = {}
        try:
            for output_index in range(batch_size):
                remaining_total = self._sampling_weight_tree.total
                if not isfinite(remaining_total) or remaining_total <= 0.0:
                    raise RuntimeError(
                        'Replay cumulative weight state became non-positive.'
                    )
                draw = float(self.rng.random())
                if not isfinite(draw) or draw < 0.0 or draw >= 1.0:
                    raise RuntimeError('Replay RNG returned a value outside [0, 1).')
                selected = self._sampling_weight_tree.find_by_cumulative_weight(
                    draw * remaining_total
                )
                if selected >= self.size:
                    raise RuntimeError('Replay selected an invalid primary slot.')
                weight = float(self.sample_weight[selected])
                if not isfinite(weight) or weight <= 0.0:
                    raise RuntimeError('Replay selected a non-positive slot weight.')
                indices[output_index] = selected
                self._sampling_weight_tree.remove_temporarily(
                    selected,
                    weight,
                    original_nodes,
                )
        finally:
            self._sampling_weight_tree.restore_nodes(original_nodes)
        if not isclose(
            self._sampling_weight_tree.total,
            represented_total,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError('Replay cumulative weight restoration failed.')
        return indices

    def _sample_primary(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = self._sample_primary_indices(batch_size)
        return {
            'ego': self.ego_features[indices].copy(),
            'goal': self.goal_features[indices].copy(),
            'zones': self.zone_features[indices].copy(),
            'count': self.zone_count[indices].copy(),
            'action': self.action[indices].copy(),
            'reward': self.reward[indices].copy(),
            'next_ego': self.next_ego_features[indices].copy(),
            'next_goal': self.next_goal_features[indices].copy(),
            'next_zones': self.next_zone_features[indices].copy(),
            'next_count': self.next_zone_count[indices].copy(),
            'done': self.done[indices].copy(),
            'success': self.success[indices].astype(np.float32).reshape(-1, 1),
            'failure': self.failure[indices].copy(),
            'near_goal': self.near_goal[indices].astype(np.float32).reshape(-1, 1),
            'line_to_goal_safe': self.line_to_goal_safe[indices]
            .astype(np.float32)
            .reshape(-1, 1),
        }

    def _sample_success(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = self.rng.choice(
            self.success_size,
            batch_size,
            replace=False,
        )
        return {
            'ego': self.success_ego_features[indices].copy(),
            'goal': self.success_goal_features[indices].copy(),
            'zones': self.success_zone_features[indices].copy(),
            'count': self.success_zone_count[indices].copy(),
            'action': self.success_action[indices].copy(),
            'reward': self.success_reward[indices].copy(),
            'next_ego': self.success_next_ego_features[indices].copy(),
            'next_goal': self.success_next_goal_features[indices].copy(),
            'next_zones': self.success_next_zone_features[indices].copy(),
            'next_count': self.success_next_zone_count[indices].copy(),
            'done': self.success_done[indices].copy(),
            'success': np.ones((batch_size, 1), dtype=np.float32),
            'failure': np.zeros(batch_size, dtype=np.bool_),
            'near_goal': self.success_near_goal[indices]
            .astype(np.float32)
            .reshape(-1, 1),
            'line_to_goal_safe': self.success_line_to_goal_safe[indices]
            .astype(np.float32)
            .reshape(-1, 1),
        }

    @staticmethod
    def _merge_samples(
        primary: dict[str, np.ndarray] | None,
        success: dict[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        if primary is None:
            if success is None:
                raise RuntimeError('Replay sampling produced no data.')
            return success
        if success is None:
            return primary
        return {
            name: np.concatenate((primary[name], success[name]), axis=0)
            for name in primary
        }

    @staticmethod
    def _observation_batch(
        ego: np.ndarray,
        goal: np.ndarray,
        zones: np.ndarray,
        counts: np.ndarray,
    ) -> V2ObservationBatch:
        batch_size = int(ego.shape[0])
        max_count = int(np.max(counts)) if batch_size > 0 else 0
        cropped_zones = zones[:, :max_count].copy()
        mask = (
            np.arange(max_count, dtype=np.int32)[None, :]
            < counts[:, None]
        )
        return V2ObservationBatch(
            ego_features=torch.from_numpy(ego.copy()),
            goal_features=torch.from_numpy(goal.copy()),
            zone_features=torch.from_numpy(cropped_zones),
            presence_mask=torch.from_numpy(mask),
        )

    def sample(self, batch_size: int) -> V2ReplayBatch:
        batch_size = _positive_int(batch_size, name='batch_size')
        if batch_size > self.size:
            raise ValueError(
                f'batch_size={batch_size} exceeds replay size={self.size}'
            )
        desired_success = int(batch_size * self.success_batch_fraction)
        success_count = min(desired_success, self.success_size)
        primary_count = batch_size - success_count
        primary = (
            self._sample_primary(primary_count)
            if primary_count > 0
            else None
        )
        success = (
            self._sample_success(success_count)
            if success_count > 0
            else None
        )
        sample = self._merge_samples(primary, success)
        obs = self._observation_batch(
            sample['ego'],
            sample['goal'],
            sample['zones'],
            sample['count'],
        )
        next_obs = self._observation_batch(
            sample['next_ego'],
            sample['next_goal'],
            sample['next_zones'],
            sample['next_count'],
        )
        return V2ReplayBatch(
            obs=obs,
            action=torch.from_numpy(sample['action']),
            reward=torch.from_numpy(sample['reward']),
            next_obs=next_obs,
            done=torch.from_numpy(sample['done']),
            success=torch.from_numpy(sample['success']),
            near_goal=torch.from_numpy(sample['near_goal']),
            line_to_goal_safe=torch.from_numpy(sample['line_to_goal_safe']),
            sample_failure_fraction=float(np.mean(sample['failure'])),
        )
