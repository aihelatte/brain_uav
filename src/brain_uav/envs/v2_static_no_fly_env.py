"""Independent V2 environment over structured observations and unified geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from typing import Any

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.geometry import NoFlyZone, no_fly_zone_from_dict
from brain_uav.observations import (
    V2Observation,
    V2ObservationScales,
    build_v2_observation,
)

from .static_no_fly_env_runtime import StaticNoFlyTrajectoryEnv
from .v2_scenario_generator import (
    V2_ENV_SCENARIO_FORMAT,
    V2_ENV_SCENARIO_VERSION,
    V2ScenarioGenerator,
)


def _nonnegative_finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and non-negative.') from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and non-negative.')
    return result


def _finite_vector(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite vector with shape {shape}.') from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be a finite vector with shape {shape}.')
    return result.copy()


def _copy_json_value(value: Any, *, path: str, active: set[int]) -> Any:
    value_type = type(value)
    if value is None or value_type in (bool, str, int):
        return value
    if value_type is float:
        if not isfinite(value):
            raise ValueError(f'{path} must contain only finite JSON numbers.')
        return value
    if value_type not in (list, dict):
        raise ValueError(
            f'{path} contains unsupported JSON value type {value_type.__name__!r}.'
        )
    identity = id(value)
    if identity in active:
        raise ValueError(f'{path} contains a circular reference.')
    active.add(identity)
    try:
        if value_type is list:
            return [
                _copy_json_value(item, path=f'{path}[{index}]', active=active)
                for index, item in enumerate(value)
            ]
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f'{path} dictionary keys must be strings.')
            result[key] = _copy_json_value(
                item,
                path=f'{path}[{key!r}]',
                active=active,
            )
        return result
    finally:
        active.remove(identity)


def _copy_scenario_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if type(value) is not dict:
        raise ValueError('V2 scenario metadata must be a strict JSON object.')
    return _copy_json_value(value, path='V2 scenario metadata', active=set())


class V2StaticNoFlyTrajectoryEnv(StaticNoFlyTrajectoryEnv):
    """Fixed-scenario V2 environment with dynamic NoFlyZone geometry.

    The class deliberately reuses the legacy environment's geometry-independent
    dynamics and reward helpers. Scenario loading, observations, collision,
    clearance, line-of-sight safety, and zone-warning terms are V2-specific.

    A Gym Box cannot represent this environment's variable-length structured
    observation. Consequently observation_space is explicitly None and
    observation_contract identifies V2Observation.
    """

    observation_contract = V2Observation

    def __init__(
        self,
        scenario: ScenarioConfig,
        rewards: RewardConfig,
        seed: int | None = None,
        fixed_scenarios: Sequence[Mapping[str, Any]] | None = None,
        *,
        uav_collision_radius: float = 0.0,
        scenario_generator: V2ScenarioGenerator | None = None,
    ) -> None:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError('scenario must be a ScenarioConfig.')
        if not isinstance(rewards, RewardConfig):
            raise TypeError('rewards must be a RewardConfig.')
        if fixed_scenarios is not None and (
            isinstance(fixed_scenarios, (str, bytes))
            or not isinstance(fixed_scenarios, Sequence)
        ):
            raise TypeError('fixed_scenarios must be a sequence of V2 scenario mappings.')
        if scenario_generator is not None and not isinstance(
            scenario_generator, V2ScenarioGenerator
        ):
            raise TypeError('scenario_generator must be a V2ScenarioGenerator.')
        if scenario_generator is not None and scenario_generator.scenario != scenario:
            raise ValueError('scenario_generator and environment ScenarioConfig must match.')

        super().__init__(
            scenario=scenario,
            rewards=rewards,
            seed=seed,
            fixed_scenarios=None,
            curriculum_mix=None,
        )
        self.fixed_scenarios = (
            tuple(deepcopy(item) for item in fixed_scenarios)
            if fixed_scenarios is not None
            else ()
        )
        self._fixed_idx = 0
        self.scenario_generator = scenario_generator
        self.uav_collision_radius = _nonnegative_finite(
            uav_collision_radius,
            name='uav_collision_radius',
        )
        self.observation_scales = V2ObservationScales(
            world_xy=float(self.scenario.world_xy),
            world_z_min=float(self.scenario.world_z_min),
            world_z_max=float(self.scenario.world_z_max),
            gamma_max=float(self.scenario.gamma_max),
        )
        self.observation_space = None
        self.zones: list[NoFlyZone] = []
        self.scenario_metadata: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[V2Observation, dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        if options is not None and not isinstance(options, dict):
            raise TypeError('options must be a dict when provided.')
        if options and 'scenario' in options:
            payload = options['scenario']
        elif self.fixed_scenarios:
            payload = self.fixed_scenarios[self._fixed_idx % len(self.fixed_scenarios)]
            self._fixed_idx += 1
        elif self.scenario_generator is not None:
            if seed is not None:
                self.scenario_generator.seed(seed)
            payload = self.scenario_generator.generate()
        else:
            raise RuntimeError(
                'No V2 scenario was provided; random V2 scenario generation '
                'requires an explicitly configured V2ScenarioGenerator.'
            )
        self._load_scenario(payload)
        self.initial_state = self.state.copy()
        self.steps = 0
        self.last_delta_z = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.recent_progress = []
        self.trajectory = [self.state[:3].copy()]
        self.best_goal_distance_so_far = self._goal_distance(self.state[:3])
        self.last_segment_goal_distance = self.best_goal_distance_so_far
        self.last_goal_reached_by_segment = False
        point_clearances = self._point_clearances(self.state[:3])
        return self._get_obs(
            zone_point_clearances=point_clearances,
        ), self._info(
            progress=0.0,
            zone_point_clearances=point_clearances,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[V2Observation, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).clip(
            self.action_space.low,
            self.action_space.high,
        )
        prev_state = self.state.copy()
        prev_action = self.prev_action.copy()
        prev_distance = self._goal_distance(prev_state[:3])
        prev_best_goal_distance = self.best_goal_distance_so_far
        self._apply_action(action)
        self.last_delta_z = float(self.state[2] - prev_state[2])
        self.steps += 1
        self.trajectory.append(self.state[:3].copy())
        new_distance = self._goal_distance(self.state[:3])
        self.last_segment_goal_distance = self._segment_goal_distance(
            prev_state[:3],
            self.state[:3],
        )
        self.last_goal_reached_by_segment = (
            self.last_segment_goal_distance <= self._active_goal_radius()
        )
        step_progress = prev_distance - new_distance
        self._record_progress(step_progress)
        terminated, truncated, outcome = self._termination(prev_state[:3])
        point_clearances = self._point_clearances(self.state[:3])
        reward = self._compute_reward(
            prev_state,
            prev_action,
            prev_distance,
            new_distance,
            action,
            outcome,
            prev_best_goal_distance,
            zone_point_clearances=point_clearances,
        )
        if new_distance < self.best_goal_distance_so_far:
            self.best_goal_distance_so_far = new_distance
        self.prev_action = action.copy()
        return (
            self._get_obs(zone_point_clearances=point_clearances),
            float(reward),
            terminated,
            truncated,
            self._info(
                progress=step_progress,
                outcome=outcome,
                zone_point_clearances=point_clearances,
            ),
        )

    def render(self):
        raise NotImplementedError(
            'Shape-aware V2 rendering is outside this validation stage.'
        )

    def export_scenario(self) -> dict[str, Any]:
        payload = {
            'format': V2_ENV_SCENARIO_FORMAT,
            'format_version': V2_ENV_SCENARIO_VERSION,
            'state': self.initial_state.copy().tolist(),
            'goal': self.goal.copy().tolist(),
            'zones': [zone.to_dict() for zone in self.zones],
            'curriculum_level': self.last_curriculum_level,
        }
        if self.scenario_metadata:
            payload['metadata'] = _copy_scenario_metadata(self.scenario_metadata)
        return payload

    def set_goal(
        self,
        new_goal: Any,
        *,
        reset_leg_timer: bool = True,
    ) -> tuple[V2Observation, dict[str, Any]]:
        self.goal = _finite_vector(new_goal, shape=(3,), name='new_goal')
        if reset_leg_timer:
            self.steps = 0
        self.recent_progress = []
        self.best_goal_distance_so_far = self._goal_distance(self.state[:3])
        self.last_segment_goal_distance = self.best_goal_distance_so_far
        self.last_goal_reached_by_segment = False
        point_clearances = self._point_clearances(self.state[:3])
        return self._get_obs(
            zone_point_clearances=point_clearances,
        ), self._info(
            progress=0.0,
            zone_point_clearances=point_clearances,
        )

    def _load_scenario(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError('V2 scenario payload must be a mapping.')
        required = {'format', 'format_version', 'state', 'goal', 'zones'}
        allowed = required | {'curriculum_level', 'metadata'}
        keys = set(payload.keys())
        missing = required - keys
        unknown = keys - allowed
        if missing:
            raise ValueError(f'V2 scenario is missing fields: {sorted(missing)}.')
        if unknown:
            raise ValueError(f'V2 scenario has unknown fields: {sorted(map(repr, unknown))}.')
        if payload['format'] != V2_ENV_SCENARIO_FORMAT:
            raise ValueError(f'V2 scenario format must be {V2_ENV_SCENARIO_FORMAT!r}.')
        if payload['format_version'] != V2_ENV_SCENARIO_VERSION:
            raise ValueError(
                f'V2 scenario format_version must be {V2_ENV_SCENARIO_VERSION}.'
            )
        raw_zones = payload['zones']
        if isinstance(raw_zones, (str, bytes)) or not isinstance(raw_zones, Sequence):
            raise ValueError('V2 scenario zones must be a sequence of NoFlyZone payloads.')
        zones: list[NoFlyZone] = []
        seen_ids: set[str] = set()
        for index, raw_zone in enumerate(raw_zones):
            try:
                zone = no_fly_zone_from_dict(raw_zone)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'V2 scenario zones[{index}] must be a strict NoFlyZone payload.'
                ) from exc
            if zone.zone_id in seen_ids:
                raise ValueError(f'V2 scenario contains duplicate zone_id {zone.zone_id!r}.')
            seen_ids.add(zone.zone_id)
            zones.append(zone)

        self.state = _finite_vector(payload['state'], shape=(5,), name='scenario state')
        self.goal = _finite_vector(payload['goal'], shape=(3,), name='scenario goal')
        self.zones = zones
        self.last_curriculum_level = str(payload.get('curriculum_level', 'custom'))
        self.scenario_metadata = _copy_scenario_metadata(payload.get('metadata'))

    def _get_obs(
        self,
        *,
        zone_point_clearances: Sequence[float] | None = None,
    ) -> V2Observation:
        return build_v2_observation(
            state=self.state,
            goal=self.goal,
            zones=self.zones,
            scales=self.observation_scales,
            uav_radius=self.uav_collision_radius,
            zone_point_clearances=zone_point_clearances,
        )

    def _validated_zone_point_clearances(
        self,
        values: Sequence[float],
    ) -> tuple[float, ...]:
        try:
            clearances = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                'zone_point_clearances must contain one finite value per zone.'
            ) from exc
        if clearances.shape != (len(self.zones),) or not np.all(
            np.isfinite(clearances)
        ):
            raise ValueError(
                'zone_point_clearances must contain one finite value per zone.'
            )
        return tuple(float(value) for value in clearances)

    def _point_clearances(self, position: Any) -> tuple[float, ...]:
        point = _finite_vector(position, shape=(3,), name='position')
        return self._validated_zone_point_clearances(
            tuple(
                zone.point_clearance(
                    point,
                    uav_radius=self.uav_collision_radius,
                )
                for zone in self.zones
            )
        )

    def _zone_query_identity(self) -> tuple[tuple[int, int, float], ...]:
        return tuple(
            (id(zone), id(zone.shape), float(zone.safety_margin))
            for zone in self.zones
        )

    def _compute_reward(
        self,
        prev_state: np.ndarray,
        prev_action: np.ndarray,
        prev_distance: float,
        new_distance: float,
        action: np.ndarray,
        outcome: str,
        prev_best_goal_distance: float,
        *,
        zone_point_clearances: Sequence[float] | None = None,
    ) -> float:
        if zone_point_clearances is None:
            return super()._compute_reward(
                prev_state,
                prev_action,
                prev_distance,
                new_distance,
                action,
                outcome,
                prev_best_goal_distance,
            )
        clearances = self._validated_zone_point_clearances(zone_point_clearances)
        context = (
            self.state[:3].copy(),
            self._zone_query_identity(),
            float(self.uav_collision_radius),
            clearances,
        )
        sentinel = object()
        previous = getattr(self, '_active_zone_point_clearances', sentinel)
        self._active_zone_point_clearances = context
        try:
            return super()._compute_reward(
                prev_state,
                prev_action,
                prev_distance,
                new_distance,
                action,
                outcome,
                prev_best_goal_distance,
            )
        finally:
            if previous is sentinel:
                del self._active_zone_point_clearances
            else:
                self._active_zone_point_clearances = previous

    def _active_clearances_for(
        self,
        position: Any,
    ) -> tuple[float, ...] | None:
        context = getattr(self, '_active_zone_point_clearances', None)
        if context is None:
            return None
        point, zone_identity, radius, clearances = context
        try:
            candidate = np.asarray(position, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            candidate.shape != (3,)
            or not np.array_equal(candidate, point)
            or zone_identity != self._zone_query_identity()
            or radius != self.uav_collision_radius
        ):
            return None
        return clearances

    def _termination(
        self,
        previous_position: np.ndarray | None = None,
    ) -> tuple[bool, bool, str]:
        cfg = self.scenario
        position = self.state[:3]
        if (
            self._goal_distance(position) <= self._active_goal_radius()
            or self.last_goal_reached_by_segment
        ):
            return True, False, 'goal'
        if position[2] <= cfg.world_z_min:
            return True, False, 'ground'
        if (
            abs(position[0]) > cfg.world_xy
            or abs(position[1]) > cfg.world_xy
            or position[2] > cfg.world_z_max
        ):
            return True, False, 'boundary'
        segment_start = position if previous_position is None else previous_position
        if any(
            zone.violates_segment(
                segment_start,
                position,
                uav_radius=self.uav_collision_radius,
            )
            for zone in self.zones
        ):
            return True, False, 'collision'
        if self.steps >= cfg.max_steps:
            return False, True, 'timeout'
        return False, False, 'running'

    def min_zone_clearance(self, position: Any | None = None) -> float:
        point = (
            self.state[:3]
            if position is None
            else _finite_vector(position, shape=(3,), name='position')
        )
        return self._minimum_zone_clearance(self._point_clearances(point))

    @staticmethod
    def _minimum_zone_clearance(clearances: Sequence[float]) -> float:
        return min(clearances) if clearances else float('inf')

    def _nearest_zone_surface_clearance(self, pos: np.ndarray) -> float:
        return self.min_zone_clearance(pos)

    def line_to_goal_is_safe(
        self,
        position: Any | None = None,
        *,
        clearance: float = 0.0,
    ) -> bool:
        point = (
            self.state[:3]
            if position is None
            else _finite_vector(position, shape=(3,), name='position')
        )
        extra_clearance = _nonnegative_finite(clearance, name='clearance')
        effective_uav_radius = self.uav_collision_radius + extra_clearance
        return not any(
            zone.violates_segment(
                point,
                self.goal,
                uav_radius=effective_uav_radius,
            )
            for zone in self.zones
        )

    def _line_to_goal_is_safe(
        self,
        pos: np.ndarray,
        samples: int = 32,
        clearance: float = 0.0,
    ) -> bool:
        del samples
        return self.line_to_goal_is_safe(pos, clearance=clearance)

    def _zone_warning_penalty(self, pos: np.ndarray) -> float:
        clearances = self._active_clearances_for(pos)
        if clearances is None:
            clearances = self._point_clearances(pos)
        return self._zone_warning_penalty_from_clearances(clearances)

    def _zone_warning_penalty_from_clearances(
        self,
        clearances: Sequence[float],
    ) -> float:
        clearance_values = self._validated_zone_point_clearances(clearances)
        warning_distance = max(self.scenario.warning_distance, 1e-6)
        penalties: list[float] = []
        for clearance in clearance_values:
            intrusion = warning_distance - clearance
            if intrusion <= 0.0:
                continue
            ratio = float(np.clip(intrusion / warning_distance, 0.0, 1.0))
            penalties.append(self.rewards.zone_penalty_weight * ratio**2)
        if not penalties:
            return 0.0
        primary = max(penalties)
        secondary = sum(penalties) - primary
        combined = primary + self.rewards.zone_secondary_penalty_ratio * min(
            secondary,
            primary,
        )
        return min(combined, self.rewards.zone_penalty_cap)

    def _info(
        self,
        *,
        progress: float,
        outcome: str = 'running',
        zone_point_clearances: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        point_clearances = (
            self._point_clearances(self.state[:3])
            if zone_point_clearances is None
            else self._validated_zone_point_clearances(zone_point_clearances)
        )
        requested_zone_count = self.scenario_metadata.get('requested_zone_count')
        if type(requested_zone_count) is not int or requested_zone_count < 0:
            requested_zone_count = len(self.zones)
        direct_path_blocker_count = self.scenario_metadata.get(
            'direct_path_blocker_count'
        )
        if type(direct_path_blocker_count) is not int or direct_path_blocker_count < 0:
            direct_path_blocker_count = sum(
                zone.violates_segment(
                    self.initial_state[:3],
                    self.goal,
                    uav_radius=self.scenario.corridor_blocking_margin,
                )
                for zone in self.zones
            )
        overlap_allowed = self.scenario_metadata.get('overlap_allowed')
        if type(overlap_allowed) is not bool:
            overlap_allowed = None
        info = super()._info(progress=progress, outcome=outcome)
        info.update(
            {
                'line_to_goal_safe': self.line_to_goal_is_safe(),
                'min_zone_clearance': self._minimum_zone_clearance(point_clearances),
                'zone_warning_penalty': self._zone_warning_penalty_from_clearances(
                    point_clearances
                ),
                'zone_count': len(self.zones),
                'uav_collision_radius': self.uav_collision_radius,
                'requested_zone_count': requested_zone_count,
                'direct_path_blocker_count': direct_path_blocker_count,
                'overlap_allowed': overlap_allowed,
            }
        )
        scenario_seed = self.scenario_metadata.get('scenario_seed')
        if type(scenario_seed) is int and scenario_seed >= 0:
            info['scenario_seed'] = scenario_seed
        return info
