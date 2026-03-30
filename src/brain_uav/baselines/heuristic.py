from __future__ import annotations

import numpy as np

from ..envs import StaticNoFlyTrajectoryEnv
from .common import heading_to_action


class HeuristicPlanner:
    def __init__(self, env: StaticNoFlyTrajectoryEnv) -> None:
        self.env = env

    def act(self, obs: np.ndarray) -> np.ndarray:
        del obs
        pos = self.env.state[:3]
        direction = self.env.goal - pos
        repulsion = np.zeros(3, dtype=np.float32)
        for zone in self.env.zones:
            vector = np.array([pos[0] - zone.center_xy[0], pos[1] - zone.center_xy[1], pos[2]], dtype=np.float32)
            distance = max(float(np.linalg.norm(vector)), 1e-6)
            influence = zone.radius + self.env.scenario.warning_distance + 40.0
            if distance < influence:
                repulsion += vector / distance * (influence - distance) * 3.0
        limits = np.array(
            [self.env.scenario.delta_gamma_max, self.env.scenario.delta_psi_max], dtype=np.float32
        )
        return heading_to_action(self.env.state[3], self.env.state[4], direction + repulsion, limits)

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

