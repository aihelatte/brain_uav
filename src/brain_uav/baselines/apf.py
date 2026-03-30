"""Artificial potential field baseline."""

from __future__ import annotations

import numpy as np

from ..envs import StaticNoFlyTrajectoryEnv
from .common import heading_to_action


class ArtificialPotentialFieldPlanner:
    """Use attraction to goal and repulsion from no-fly zones."""

    def __init__(self, env: StaticNoFlyTrajectoryEnv, attractive_gain: float = 1.0, repulsive_gain: float = 5000.0):
        self.env = env
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain

    def act(self, obs: np.ndarray) -> np.ndarray:
        del obs
        pos = self.env.state[:3]
        force = self.attractive_gain * (self.env.goal - pos)
        for zone in self.env.zones:
            delta = np.array([pos[0] - zone.center_xy[0], pos[1] - zone.center_xy[1], pos[2]], dtype=np.float32)
            distance = max(float(np.linalg.norm(delta)), 1e-6)
            threshold = zone.radius + self.env.scenario.warning_distance + 60.0
            if distance < threshold:
                # APF 的核心：进入作用范围后，障碍物会产生斥力。
                strength = self.repulsive_gain * ((1.0 / distance) - (1.0 / threshold)) / (distance**2)
                force += strength * (delta / distance)
        limits = np.array(
            [self.env.scenario.delta_gamma_max, self.env.scenario.delta_psi_max], dtype=np.float32
        )
        return heading_to_action(self.env.state[3], self.env.state[4], force, limits)

    def rollout(self, max_steps: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        obs, _ = self.env.reset()
        steps = max_steps or self.env.scenario.max_steps
        samples: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(steps):
            action = self.act(obs)
            samples.append((obs.copy(), action.copy()))
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        return samples
