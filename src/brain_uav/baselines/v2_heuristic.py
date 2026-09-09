"""Heuristic expert for the structured V2 environment."""

from __future__ import annotations

from math import isfinite

import numpy as np

from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.geometry import NoFlyZone
from brain_uav.observations import V2Observation

from .common import heading_to_action


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and non-negative.')
    return result


def _outward_direction(zone: NoFlyZone, position: np.ndarray) -> np.ndarray:
    """Return a deterministic unit vector pointing away from a V2 shape."""

    closest = zone.shape.closest_point(position)
    direction = np.asarray(position, dtype=np.float64) - closest
    length = float(np.linalg.norm(direction))
    if zone.shape.contains(position) or length <= 1e-12:
        direction = zone.shape.surface_normal(position)
        length = float(np.linalg.norm(direction))
    if not np.all(np.isfinite(direction)) or not isfinite(length) or length <= 0.0:
        raise ValueError(f'Zone {zone.zone_id!r} returned an invalid outward direction.')
    return direction / length


def _bounded_action(
    env: V2StaticNoFlyTrajectoryEnv,
    direction: np.ndarray,
) -> np.ndarray:
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError('Expert desired direction must be a finite 3D vector.')
    limits = np.asarray(env.action_space.high, dtype=np.float32)
    action = heading_to_action(
        float(env.state[3]),
        float(env.state[4]),
        direction,
        limits,
    )
    action = np.clip(action, env.action_space.low, env.action_space.high).astype(
        np.float32,
        copy=False,
    )
    if action.shape != (2,) or not np.all(np.isfinite(action)):
        raise ValueError('V2 heuristic produced an invalid action.')
    return action.copy()


class V2HeuristicPlanner:
    """Goal attraction plus unified-geometry repulsion for easy V2 scenes."""

    planner_name = 'v2_heuristic'

    def __init__(
        self,
        env: V2StaticNoFlyTrajectoryEnv,
        *,
        repulsive_gain: float = 3.0,
        influence_margin: float = 4.0,
    ) -> None:
        if not isinstance(env, V2StaticNoFlyTrajectoryEnv):
            raise TypeError('env must be a V2StaticNoFlyTrajectoryEnv.')
        self.env = env
        self.repulsive_gain = _finite_nonnegative(
            repulsive_gain,
            name='repulsive_gain',
        )
        self.influence_margin = _finite_nonnegative(
            influence_margin,
            name='influence_margin',
        )

    def act(self, observation: V2Observation) -> np.ndarray:
        if not isinstance(observation, V2Observation):
            raise TypeError('observation must be a V2Observation.')
        position = np.asarray(self.env.state[:3], dtype=np.float64)
        direction = np.asarray(self.env.goal, dtype=np.float64) - position
        repulsion = np.zeros(3, dtype=np.float64)
        influence = float(self.env.scenario.warning_distance) + self.influence_margin
        for zone in self.env.zones:
            clearance = zone.point_clearance(
                position,
                uav_radius=self.env.uav_collision_radius,
            )
            if clearance < influence:
                strength = (influence - clearance) * self.repulsive_gain
                repulsion += strength * _outward_direction(zone, position)
        return _bounded_action(self.env, direction + repulsion)


__all__ = ['V2HeuristicPlanner']
