"""Artificial-potential-field expert for the structured V2 environment."""

from __future__ import annotations

import numpy as np

from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.observations import V2Observation

from .v2_heuristic import (
    _bounded_action,
    _finite_nonnegative,
    _outward_direction,
)


class V2ArtificialPotentialFieldPlanner:
    """Attractive target force plus per-zone unified-surface repulsion."""

    planner_name = 'v2_apf'

    def __init__(
        self,
        env: V2StaticNoFlyTrajectoryEnv,
        *,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 5000.0,
        influence_margin: float = 6.0,
    ) -> None:
        if not isinstance(env, V2StaticNoFlyTrajectoryEnv):
            raise TypeError('env must be a V2StaticNoFlyTrajectoryEnv.')
        self.env = env
        self.attractive_gain = _finite_nonnegative(
            attractive_gain,
            name='attractive_gain',
        )
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
        force = self.attractive_gain * (
            np.asarray(self.env.goal, dtype=np.float64) - position
        )
        threshold = float(self.env.scenario.warning_distance) + self.influence_margin
        for zone in self.env.zones:
            clearance = zone.point_clearance(
                position,
                uav_radius=self.env.uav_collision_radius,
            )
            if clearance < threshold:
                effective_distance = max(float(clearance), 1e-6)
                strength = self.repulsive_gain * (
                    (1.0 / effective_distance) - (1.0 / threshold)
                ) / (effective_distance**2)
                force += strength * _outward_direction(zone, position)
        return _bounded_action(self.env, force)


__all__ = ['V2ArtificialPotentialFieldPlanner']
