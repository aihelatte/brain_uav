"""Main environment implementation used by training and evaluation.

这是项目最核心的文件之一。
它负责：
- 生成飞行场景
- 推进飞行器状态
- 判断是否撞禁飞区/出界/到达目标
- 计算奖励
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import RewardConfig, ScenarioConfig
from ..utils.gym_compat import gym, spaces


@dataclass(slots=True)
class Zone:
    """One static hemisphere no-fly zone.

    center_xy 只记录地面投影圆心，高度方向默认从地面向上隆起成半球。
    """

    center_xy: np.ndarray
    radius: float


class StaticNoFlyTrajectoryEnv(gym.Env):
    """Gymnasium-style environment for static no-fly-zone trajectory planning."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario: ScenarioConfig | None = None,
        rewards: RewardConfig | None = None,
        seed: int | None = None,
        fixed_scenarios: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario or ScenarioConfig()
        self.rewards = rewards or RewardConfig()
        self.fixed_scenarios = fixed_scenarios or []
        self._fixed_idx = 0
        self.rng = np.random.default_rng(seed)

        obs_dim = 5 + 3 + 3 * self.scenario.nearest_zone_count
        self.action_space = spaces.Box(
            low=np.array(
                [-self.scenario.delta_gamma_max, -self.scenario.delta_psi_max], dtype=np.float32
            ),
            high=np.array(
                [self.scenario.delta_gamma_max, self.scenario.delta_psi_max], dtype=np.float32
            ),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.state = np.zeros(5, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.zones: list[Zone] = []
        self.steps = 0
        self.trajectory: list[np.ndarray] = []

    def seed(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.seed(seed)
        if options and "scenario" in options:
            self._load_scenario(options["scenario"])
        elif self.fixed_scenarios:
            scenario = self.fixed_scenarios[self._fixed_idx % len(self.fixed_scenarios)]
            self._fixed_idx += 1
            self._load_scenario(scenario)
        else:
            self._sample_scenario()
        self.steps = 0
        self.trajectory = [self.state[:3].copy()]
        return self._get_obs(), self._info(progress=0.0)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).clip(self.action_space.low, self.action_space.high)
        prev_distance = self._goal_distance(self.state[:3])
        self._apply_action(action)
        self.steps += 1
        self.trajectory.append(self.state[:3].copy())
        new_distance = self._goal_distance(self.state[:3])
        terminated, truncated, outcome = self._termination()
        reward = self._compute_reward(prev_distance, new_distance, action, outcome)
        return self._get_obs(), float(reward), terminated, truncated, self._info(
            progress=prev_distance - new_distance,
            outcome=outcome,
        )

    def render(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        traj = np.array(self.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:blue", label="trajectory")
        ax.scatter(*self.goal, color="tab:green", s=80, label="goal")
        for zone in self.zones:
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi / 2 : 10j]
            x = zone.radius * np.cos(u) * np.sin(v) + zone.center_xy[0]
            y = zone.radius * np.sin(u) * np.sin(v) + zone.center_xy[1]
            z = zone.radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color="tab:red", alpha=0.25)
        ax.legend(loc="upper left")
        return fig

    def export_scenario(self) -> dict[str, Any]:
        return {
            "state": self.state.copy(),
            "goal": self.goal.copy(),
            "zones": [
                {"center_xy": zone.center_xy.copy(), "radius": float(zone.radius)} for zone in self.zones
            ],
        }

    def _sample_scenario(self) -> None:
        cfg = self.scenario
        self.state = np.array(
            [
                self.rng.uniform(-0.8 * cfg.world_xy, -0.5 * cfg.world_xy),
                self.rng.uniform(-0.2 * cfg.world_xy, 0.2 * cfg.world_xy),
                self.rng.uniform(80.0, 160.0),
                0.0,
                self.rng.uniform(-0.2, 0.2),
            ],
            dtype=np.float32,
        )
        self.goal = np.array(
            [
                self.rng.uniform(0.45 * cfg.world_xy, 0.8 * cfg.world_xy),
                self.rng.uniform(-0.3 * cfg.world_xy, 0.3 * cfg.world_xy),
                self.rng.uniform(80.0, 220.0),
            ],
            dtype=np.float32,
        )
        self.zones = []
        zone_count = int(self.rng.integers(cfg.min_no_fly_zones, cfg.max_no_fly_zones + 1))
        for _ in range(zone_count):
            self.zones.append(
                Zone(
                    center_xy=np.array(
                        [
                            self.rng.uniform(-0.2 * cfg.world_xy, 0.5 * cfg.world_xy),
                            self.rng.uniform(-0.5 * cfg.world_xy, 0.5 * cfg.world_xy),
                        ],
                        dtype=np.float32,
                    ),
                    radius=float(self.rng.uniform(*cfg.no_fly_radius_range)),
                )
            )

    def _load_scenario(self, payload: dict[str, Any]) -> None:
        self.state = np.asarray(payload["state"], dtype=np.float32).copy()
        self.goal = np.asarray(payload["goal"], dtype=np.float32).copy()
        self.zones = [
            Zone(center_xy=np.asarray(zone["center_xy"], dtype=np.float32), radius=float(zone["radius"]))
            for zone in payload["zones"]
        ]

    def _apply_action(self, action: np.ndarray) -> None:
        x, y, z, gamma, psi = self.state
        cfg = self.scenario
        gamma = float(np.clip(gamma + action[0], -cfg.gamma_max, cfg.gamma_max))
        psi = self._wrap_angle(float(psi + action[1]))
        x += cfg.speed * math.cos(gamma) * math.cos(psi) * cfg.dt
        y += cfg.speed * math.cos(gamma) * math.sin(psi) * cfg.dt
        z += cfg.speed * math.sin(gamma) * cfg.dt
        self.state = np.array([x, y, z, gamma, psi], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        own_state = self.state
        rel_goal = self.goal - self.state[:3]
        zone_features: list[float] = []
        sorted_zones = sorted(self.zones, key=lambda zone: np.linalg.norm(zone.center_xy - self.state[:2]))
        for zone in sorted_zones[: self.scenario.nearest_zone_count]:
            dx, dy = zone.center_xy - self.state[:2]
            zone_features.extend([float(dx), float(dy), float(zone.radius)])
        while len(zone_features) < self.scenario.nearest_zone_count * 3:
            zone_features.extend([0.0, 0.0, 0.0])
        return np.concatenate(
            [own_state, rel_goal.astype(np.float32), np.array(zone_features, dtype=np.float32)]
        ).astype(np.float32)

    def _termination(self) -> tuple[bool, bool, str]:
        cfg = self.scenario
        pos = self.state[:3]
        if self._goal_distance(pos) <= cfg.goal_radius:
            return True, False, "goal"
        if pos[2] <= cfg.world_z_min:
            return True, False, "ground"
        if abs(pos[0]) > cfg.world_xy or abs(pos[1]) > cfg.world_xy or pos[2] > cfg.world_z_max:
            return True, False, "boundary"
        if any(self._inside_zone(pos, zone) for zone in self.zones):
            return True, False, "collision"
        if self.steps >= cfg.max_steps:
            return False, True, "timeout"
        return False, False, "running"

    def _compute_reward(self, prev_distance: float, new_distance: float, action: np.ndarray, outcome: str) -> float:
        rew = self.rewards.progress_weight * (prev_distance - new_distance)
        rew -= self.rewards.step_penalty
        rew -= self.rewards.smoothness_weight * float(np.square(action).sum())
        rew -= self._zone_warning_penalty(self.state[:3])
        rew -= self._boundary_warning_penalty(self.state[:3])
        if outcome == "goal":
            rew += self.rewards.goal_reward
        elif outcome in {"collision", "ground"}:
            rew -= self.rewards.collision_penalty
        elif outcome == "boundary":
            rew -= self.rewards.boundary_penalty
        elif outcome == "timeout":
            rew -= self.rewards.timeout_penalty
        return rew

    def _zone_warning_penalty(self, pos: np.ndarray) -> float:
        """Quadratic soft penalty around no-fly zones with a hard cap."""

        warning_distance = max(self.scenario.warning_distance, 1e-6)
        total_penalty = 0.0
        for zone in self.zones:
            center_distance = float(
                np.linalg.norm(np.array([pos[0] - zone.center_xy[0], pos[1] - zone.center_xy[1], pos[2]]))
            )
            intrusion = zone.radius + warning_distance - center_distance
            if intrusion <= 0.0:
                continue
            ratio = float(np.clip(intrusion / warning_distance, 0.0, 1.0))
            total_penalty += self.rewards.zone_penalty_weight * (ratio**2)
        return min(total_penalty, self.rewards.zone_penalty_cap)

    def _boundary_warning_penalty(self, pos: np.ndarray) -> float:
        """Quadratic soft penalty when approaching x/y/ceiling boundaries."""

        warning_distance = max(self.scenario.boundary_warning_distance, 1e-6)
        distances = [
            self.scenario.world_xy - abs(float(pos[0])),
            self.scenario.world_xy - abs(float(pos[1])),
            self.scenario.world_z_max - float(pos[2]),
        ]
        min_distance = min(distances)
        if min_distance >= warning_distance:
            return 0.0
        ratio = float(np.clip((warning_distance - min_distance) / warning_distance, 0.0, 1.0))
        penalty = self.rewards.boundary_soft_penalty_weight * (ratio**2)
        return min(penalty, self.rewards.boundary_soft_penalty_cap)

    def _goal_distance(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - self.goal))

    @staticmethod
    def _inside_zone(pos: np.ndarray, zone: Zone) -> bool:
        distance = (pos[0] - zone.center_xy[0]) ** 2 + (pos[1] - zone.center_xy[1]) ** 2 + pos[2] ** 2
        return bool(distance <= zone.radius**2)

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return ((value + math.pi) % (2 * math.pi)) - math.pi

    def _info(self, *, progress: float, outcome: str = "running") -> dict[str, Any]:
        return {
            "goal_distance": self._goal_distance(self.state[:3]),
            "progress": progress,
            "outcome": outcome,
            "steps": self.steps,
        }
