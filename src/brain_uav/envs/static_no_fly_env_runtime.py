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
from ..curriculum import CURRICULUM_LEVELS, normalize_curriculum_mix
from ..utils.gym_compat import gym, spaces


@dataclass(slots=True)
class Zone:
    """One static hemisphere no-fly zone."""

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
        curriculum_mix: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario or ScenarioConfig()
        self.rewards = rewards or RewardConfig()
        self.fixed_scenarios = fixed_scenarios or []
        self.curriculum_mix = normalize_curriculum_mix(curriculum_mix, fallback_level='hard') if curriculum_mix else None
        self._fixed_idx = 0
        self.rng = np.random.default_rng(seed)

        obs_dim = 5 + 3 + 4 + 3 * self.scenario.nearest_zone_count
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
        self.initial_state = np.zeros(5, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.zones: list[Zone] = []
        self.steps = 0
        self.last_delta_z = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.recent_progress: list[float] = []
        self.trajectory: list[np.ndarray] = []
        self.last_curriculum_level = 'random'

    def seed(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.seed(seed)
        if options and 'scenario' in options:
            self._load_scenario(options['scenario'])
        elif self.fixed_scenarios:
            scenario = self.fixed_scenarios[self._fixed_idx % len(self.fixed_scenarios)]
            self._fixed_idx += 1
            self._load_scenario(scenario)
        else:
            self._sample_scenario()
        self.initial_state = self.state.copy()
        self.steps = 0
        self.last_delta_z = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.recent_progress = []
        self.trajectory = [self.state[:3].copy()]
        return self._get_obs(), self._info(progress=0.0)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).clip(self.action_space.low, self.action_space.high)
        prev_state = self.state.copy()
        prev_action = self.prev_action.copy()
        prev_distance = self._goal_distance(prev_state[:3])
        self._apply_action(action)
        self.last_delta_z = float(self.state[2] - prev_state[2])
        self.steps += 1
        self.trajectory.append(self.state[:3].copy())
        new_distance = self._goal_distance(self.state[:3])
        step_progress = prev_distance - new_distance
        self._record_progress(step_progress)
        terminated, truncated, outcome = self._termination()
        reward = self._compute_reward(prev_state, prev_action, prev_distance, new_distance, action, outcome)
        self.prev_action = action.copy()
        return self._get_obs(), float(reward), terminated, truncated, self._info(
            progress=step_progress,
            outcome=outcome,
        )

    def render(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        traj = np.array(self.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='tab:blue', label='trajectory')
        ax.scatter(*self.goal, color='tab:green', s=80, label='goal')
        for zone in self.zones:
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi / 2 : 10j]
            x = zone.radius * np.cos(u) * np.sin(v) + zone.center_xy[0]
            y = zone.radius * np.sin(u) * np.sin(v) + zone.center_xy[1]
            z = zone.radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color='tab:red', alpha=0.25)
        ax.legend(loc='upper left')
        return fig

    def export_scenario(self) -> dict[str, Any]:
        return {
            'state': self.initial_state.copy().tolist(),
            'goal': self.goal.copy().tolist(),
            'zones': [
                {'center_xy': zone.center_xy.copy().tolist(), 'radius': float(zone.radius)}
                for zone in self.zones
            ],
            'curriculum_level': self.last_curriculum_level,
        }

    def _sample_scenario(self) -> None:
        if self.curriculum_mix:
            for _ in range(self.scenario.scenario_max_sampling_attempts):
                level = self._sample_curriculum_level()
                scenario = self._sample_curriculum_scenario(level)
                if scenario is not None:
                    self._load_scenario(scenario)
                    self.last_curriculum_level = level
                    return
            raise RuntimeError('Failed to sample a curriculum scenario under current constraints.')

        scenario = self._sample_curriculum_scenario('hard')
        if scenario is None:
            raise RuntimeError('Failed to sample a random hard scenario under current constraints.')
        self._load_scenario(scenario)
        self.last_curriculum_level = 'hard'

    def _sample_curriculum_level(self) -> str:
        levels = list(self.curriculum_mix.keys())
        weights = np.array([self.curriculum_mix[level] for level in levels], dtype=np.float64)
        weights = weights / weights.sum()
        return str(self.rng.choice(levels, p=weights))

    def _sample_curriculum_scenario(self, level: str) -> dict[str, Any] | None:
        if level == 'easy':
            return self._sample_easy_scenario()
        if level == 'medium':
            return self._sample_medium_scenario()
        if level == 'hard':
            return self._sample_hard_scenario()
        raise ValueError(f'Unsupported curriculum level: {level}')

    def _sample_easy_scenario(self) -> dict[str, Any] | None:
        cfg = self.scenario
        for _ in range(40):
            state = np.array(
                [
                    self.rng.uniform(-0.8 * cfg.world_xy, -0.58 * cfg.world_xy),
                    self.rng.uniform(-0.10 * cfg.world_xy, 0.10 * cfg.world_xy),
                    self.rng.uniform(110.0, 155.0),
                    0.0,
                    self.rng.uniform(-0.10, 0.10),
                ],
                dtype=np.float32,
            )
            goal = np.array(
                [
                    self.rng.uniform(0.52 * cfg.world_xy, 0.80 * cfg.world_xy),
                    self.rng.uniform(-0.12 * cfg.world_xy, 0.12 * cfg.world_xy),
                    self.rng.uniform(105.0, 165.0),
                ],
                dtype=np.float32,
            )
            if abs(float(goal[2] - state[2])) > 55.0:
                continue
            radius = float(self.rng.uniform(70.0, 105.0))
            line_y = float((state[1] + goal[1]) * 0.5)
            zone = Zone(
                center_xy=np.array(
                    [
                        self.rng.uniform(-0.05 * cfg.world_xy, 0.35 * cfg.world_xy),
                        line_y + self.rng.choice([-1.0, 1.0]) * self.rng.uniform(170.0, 280.0),
                    ],
                    dtype=np.float32,
                ),
                radius=radius,
            )
            if not self._zone_candidate_is_valid(state, goal, [], zone.center_xy, zone.radius):
                continue
            blockers = self._count_corridor_blockers(state, goal, [zone], margin=20.0)
            if blockers != 0:
                continue
            return {
                'state': state.tolist(),
                'goal': goal.tolist(),
                'zones': [{'center_xy': zone.center_xy.tolist(), 'radius': zone.radius}],
                'curriculum_level': 'easy',
            }
        return None

    def _sample_medium_scenario(self) -> dict[str, Any] | None:
        cfg = self.scenario
        for _ in range(60):
            state = np.array(
                [
                    self.rng.uniform(-0.82 * cfg.world_xy, -0.60 * cfg.world_xy),
                    self.rng.uniform(-0.16 * cfg.world_xy, 0.16 * cfg.world_xy),
                    self.rng.uniform(105.0, 165.0),
                    0.0,
                    self.rng.uniform(-0.15, 0.15),
                ],
                dtype=np.float32,
            )
            goal = np.array(
                [
                    self.rng.uniform(0.50 * cfg.world_xy, 0.82 * cfg.world_xy),
                    self.rng.uniform(-0.20 * cfg.world_xy, 0.20 * cfg.world_xy),
                    self.rng.uniform(95.0, 185.0),
                ],
                dtype=np.float32,
            )
            if abs(float(goal[2] - state[2])) > 75.0:
                continue
            mode = str(self.rng.choice(['single_block', 'double_detour']))
            if mode == 'single_block':
                zones = self._sample_medium_single_block(state, goal)
            else:
                zones = self._sample_medium_double_detour(state, goal)
            if not zones:
                continue
            blockers = self._count_corridor_blockers(state, goal, zones, margin=cfg.corridor_blocking_margin)
            if blockers < 1 or blockers > 2:
                continue
            return {
                'state': state.tolist(),
                'goal': goal.tolist(),
                'zones': [
                    {'center_xy': zone.center_xy.tolist(), 'radius': zone.radius}
                    for zone in zones
                ],
                'curriculum_level': 'medium',
            }
        return None

    def _sample_medium_single_block(self, state: np.ndarray, goal: np.ndarray) -> list[Zone] | None:
        cfg = self.scenario
        zones: list[Zone] = []
        center_xy = np.array(
            [
                self.rng.uniform(0.00 * cfg.world_xy, 0.30 * cfg.world_xy),
                self.rng.uniform(-60.0, 60.0) + 0.5 * (state[1] + goal[1]),
            ],
            dtype=np.float32,
        )
        radius = float(self.rng.uniform(105.0, 145.0))
        if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius):
            return None
        zones.append(Zone(center_xy=center_xy, radius=radius))
        return zones

    def _sample_medium_double_detour(self, state: np.ndarray, goal: np.ndarray) -> list[Zone] | None:
        cfg = self.scenario
        zones: list[Zone] = []
        base_x = self.rng.uniform(-0.05 * cfg.world_xy, 0.20 * cfg.world_xy)
        offsets = [self.rng.uniform(120.0, 190.0), -self.rng.uniform(120.0, 190.0)]
        self.rng.shuffle(offsets)
        for idx, offset in enumerate(offsets):
            center_xy = np.array(
                [
                    base_x + idx * self.rng.uniform(120.0, 190.0),
                    0.5 * (state[1] + goal[1]) + offset,
                ],
                dtype=np.float32,
            )
            radius = float(self.rng.uniform(85.0, 120.0))
            if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius):
                return None
            zones.append(Zone(center_xy=center_xy, radius=radius))
        return zones

    def _sample_hard_scenario(self) -> dict[str, Any] | None:
        cfg = self.scenario
        for _attempt in range(cfg.scenario_max_sampling_attempts):
            state = np.array(
                [
                    self.rng.uniform(-0.8 * cfg.world_xy, -0.5 * cfg.world_xy),
                    self.rng.uniform(-0.2 * cfg.world_xy, 0.2 * cfg.world_xy),
                    self.rng.uniform(80.0, 160.0),
                    0.0,
                    self.rng.uniform(-0.2, 0.2),
                ],
                dtype=np.float32,
            )
            goal = np.array(
                [
                    self.rng.uniform(0.45 * cfg.world_xy, 0.8 * cfg.world_xy),
                    self.rng.uniform(-0.3 * cfg.world_xy, 0.3 * cfg.world_xy),
                    self.rng.uniform(80.0, 220.0),
                ],
                dtype=np.float32,
            )
            if abs(float(goal[2] - state[2])) > cfg.max_start_goal_height_gap:
                continue
            zones = self._sample_zones_for_pair(state, goal)
            if zones is None:
                continue
            if not self._corridor_is_reasonable(state, goal, zones):
                continue
            return {
                'state': state.tolist(),
                'goal': goal.tolist(),
                'zones': [
                    {'center_xy': zone.center_xy.tolist(), 'radius': zone.radius}
                    for zone in zones
                ],
                'curriculum_level': 'hard',
            }
        return None

    def _sample_zones_for_pair(self, state: np.ndarray, goal: np.ndarray) -> list[Zone] | None:
        cfg = self.scenario
        zones: list[Zone] = []
        zone_count = int(self.rng.integers(max(2, cfg.min_no_fly_zones), cfg.max_no_fly_zones + 1))
        for _ in range(zone_count):
            accepted = False
            for _attempt in range(50):
                center_xy = np.array(
                    [
                        self.rng.uniform(-0.2 * cfg.world_xy, 0.5 * cfg.world_xy),
                        self.rng.uniform(-0.5 * cfg.world_xy, 0.5 * cfg.world_xy),
                    ],
                    dtype=np.float32,
                )
                radius = float(self.rng.uniform(*cfg.no_fly_radius_range))
                if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius):
                    continue
                zones.append(Zone(center_xy=center_xy, radius=radius))
                accepted = True
                break
            if not accepted:
                return None
        return zones

    def _zone_candidate_is_valid(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        existing_zones: list[Zone],
        center_xy: np.ndarray,
        radius: float,
    ) -> bool:
        cfg = self.scenario
        dist_to_goal = float(
            np.linalg.norm(
                np.array([goal[0] - center_xy[0], goal[1] - center_xy[1], goal[2]], dtype=np.float32)
            )
        )
        safe_margin = radius + cfg.warning_distance + cfg.goal_radius + 10.0
        if dist_to_goal <= safe_margin:
            return False

        dist_to_start = float(
            np.linalg.norm(
                np.array([state[0] - center_xy[0], state[1] - center_xy[1], state[2]], dtype=np.float32)
            )
        )
        if dist_to_start <= radius + cfg.warning_distance + cfg.start_zone_clearance:
            return False

        for zone in existing_zones:
            center_distance = float(np.linalg.norm(center_xy - zone.center_xy))
            min_allowed = cfg.zone_overlap_ratio_limit * (radius + zone.radius)
            if center_distance <= min_allowed:
                return False
        return True

    def _corridor_is_reasonable(self, state: np.ndarray, goal: np.ndarray, zones: list[Zone]) -> bool:
        blockers = self._count_corridor_blockers(state, goal, zones, margin=self.scenario.corridor_blocking_margin)
        return blockers <= self.scenario.max_corridor_blockers

    def _count_corridor_blockers(self, state: np.ndarray, goal: np.ndarray, zones: list[Zone], margin: float) -> int:
        start_xy = state[:2]
        goal_xy = goal[:2]
        segment = goal_xy - start_xy
        segment_norm_sq = float(np.dot(segment, segment))
        if segment_norm_sq <= 1e-6:
            return 0
        blockers = 0
        for zone in zones:
            t = float(np.dot(zone.center_xy - start_xy, segment) / segment_norm_sq)
            if t <= 0.08 or t >= 0.92:
                continue
            projection = start_xy + t * segment
            distance_to_segment = float(np.linalg.norm(zone.center_xy - projection))
            if distance_to_segment <= zone.radius + margin:
                blockers += 1
        return blockers

    def _load_scenario(self, payload: dict[str, Any]) -> None:
        self.state = np.asarray(payload['state'], dtype=np.float32).copy()
        self.goal = np.asarray(payload['goal'], dtype=np.float32).copy()
        self.zones = [
            Zone(center_xy=np.asarray(zone['center_xy'], dtype=np.float32), radius=float(zone['radius']))
            for zone in payload['zones']
        ]
        self.last_curriculum_level = str(payload.get('curriculum_level', 'custom'))

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
        extra_features = np.array(
            [
                float(self.state[2] - self.scenario.ground_warning_height),
                float(self.scenario.world_z_max - self.state[2]),
                float(self.last_delta_z),
                1.0 if self.last_delta_z < 0.0 else 0.0,
            ],
            dtype=np.float32,
        )
        zone_features: list[float] = []
        sorted_zones = sorted(self.zones, key=lambda zone: np.linalg.norm(zone.center_xy - self.state[:2]))
        for zone in sorted_zones[: self.scenario.nearest_zone_count]:
            dx, dy = zone.center_xy - self.state[:2]
            zone_features.extend([float(dx), float(dy), float(zone.radius)])
        while len(zone_features) < self.scenario.nearest_zone_count * 3:
            zone_features.extend([0.0, 0.0, 0.0])
        return np.concatenate(
            [own_state, rel_goal.astype(np.float32), extra_features, np.array(zone_features, dtype=np.float32)]
        ).astype(np.float32)

    def _termination(self) -> tuple[bool, bool, str]:
        cfg = self.scenario
        pos = self.state[:3]
        if self._goal_distance(pos) <= cfg.goal_radius:
            return True, False, 'goal'
        if pos[2] <= cfg.world_z_min:
            return True, False, 'ground'
        if abs(pos[0]) > cfg.world_xy or abs(pos[1]) > cfg.world_xy or pos[2] > cfg.world_z_max:
            return True, False, 'boundary'
        if any(self._inside_zone(pos, zone) for zone in self.zones):
            return True, False, 'collision'
        if self.steps >= cfg.max_steps:
            return False, True, 'timeout'
        return False, False, 'running'

    def _compute_reward(
        self,
        prev_state: np.ndarray,
        prev_action: np.ndarray,
        prev_distance: float,
        new_distance: float,
        action: np.ndarray,
        outcome: str,
    ) -> float:
        rew = self.rewards.progress_weight * (prev_distance - new_distance)
        rew += self._terminal_convergence_reward(prev_distance, new_distance)
        rew -= self.rewards.step_penalty
        rew -= self.rewards.smoothness_weight * float(np.square(action).sum())
        rew -= self._action_change_penalty(prev_action, action)
        rew -= self._zone_warning_penalty(self.state[:3])
        rew -= self._boundary_warning_penalty(self.state[:3])
        rew -= self._ground_warning_penalty(self.state[:3])
        rew -= self._descent_trend_penalty(prev_state, self.state)
        rew -= self._climb_trend_penalty(self.state)
        rew -= self._inefficiency_penalty()
        if outcome == 'goal':
            rew += self.rewards.goal_reward
        elif outcome in {'collision', 'ground'}:
            rew -= self.rewards.collision_penalty
        elif outcome == 'boundary':
            rew -= self.rewards.boundary_penalty
        elif outcome == 'timeout':
            rew -= self.rewards.timeout_penalty
        return rew

    def _record_progress(self, step_progress: float) -> None:
        self.recent_progress.append(float(step_progress))
        if len(self.recent_progress) > self.rewards.progress_window_size:
            self.recent_progress.pop(0)

    def _inefficiency_penalty(self) -> float:
        if len(self.recent_progress) < self.rewards.progress_window_size:
            return 0.0
        total_progress = float(sum(self.recent_progress))
        if total_progress >= self.rewards.min_progress_per_window:
            return 0.0
        deficit_ratio = float(
            np.clip(
                (self.rewards.min_progress_per_window - total_progress)
                / max(self.rewards.min_progress_per_window, 1e-6),
                0.0,
                1.0,
            )
        )
        penalty = self.rewards.inefficiency_penalty_weight * deficit_ratio
        return min(penalty, self.rewards.inefficiency_penalty_cap)

    def _terminal_convergence_reward(self, prev_distance: float, new_distance: float) -> float:
        if new_distance >= self.rewards.terminal_convergence_distance or new_distance >= prev_distance:
            return 0.0
        reward = self.rewards.terminal_convergence_progress_weight * (prev_distance - new_distance)
        return min(reward, self.rewards.terminal_convergence_reward_cap)

    def _action_change_penalty(self, prev_action: np.ndarray, action: np.ndarray) -> float:
        delta = action - prev_action
        return (
            self.rewards.action_delta_gamma_weight * float(delta[0] ** 2)
            + self.rewards.action_delta_psi_weight * float(delta[1] ** 2)
        )

    def _zone_warning_penalty(self, pos: np.ndarray) -> float:
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

    def _ground_warning_penalty(self, pos: np.ndarray) -> float:
        warning_height = min(self.scenario.ground_warning_height, 80.0)
        effective_span = max(warning_height - self.scenario.world_z_min, 1e-6)
        if float(pos[2]) >= warning_height:
            return 0.0
        ratio = float(np.clip((warning_height - float(pos[2])) / effective_span, 0.0, 1.0))
        penalty = self.rewards.ground_soft_penalty_weight * (ratio**2)
        return min(penalty, self.rewards.ground_soft_penalty_cap)

    def _descent_trend_penalty(self, prev_state: np.ndarray, new_state: np.ndarray) -> float:
        delta_z = float(new_state[2] - prev_state[2])
        gamma = float(new_state[3])
        if delta_z >= 0.0 or gamma >= -self.scenario.descent_gamma_threshold:
            return 0.0

        max_vertical_step = max(self.scenario.speed * self.scenario.dt * math.sin(self.scenario.gamma_max), 1e-6)
        gamma_ratio = float(
            np.clip(
                (abs(gamma) - self.scenario.descent_gamma_threshold)
                / max(self.scenario.gamma_max - self.scenario.descent_gamma_threshold, 1e-6),
                0.0,
                1.0,
            )
        )
        descent_ratio = float(np.clip(abs(delta_z) / max_vertical_step, 0.0, 1.0))
        if float(new_state[2]) >= self.scenario.descent_penalty_height:
            height_factor = 0.35
        else:
            height_factor = 0.35 + 0.65 * float(
                np.clip(
                    (self.scenario.descent_penalty_height - float(new_state[2]))
                    / max(self.scenario.descent_penalty_height - self.scenario.world_z_min, 1e-6),
                    0.0,
                    1.0,
                )
            )
        penalty = self.rewards.descent_trend_penalty_weight * gamma_ratio * descent_ratio * height_factor
        return min(penalty, self.rewards.descent_trend_penalty_cap)

    def _climb_trend_penalty(self, state: np.ndarray) -> float:
        high_altitude_start = self.rewards.climb_trend_high_altitude_ratio * self.scenario.world_z_max
        altitude = float(state[2])
        gamma = float(state[3])
        if altitude <= high_altitude_start or gamma <= self.rewards.climb_trend_gamma_threshold:
            return 0.0

        altitude_ratio = float(
            np.clip(
                (altitude - high_altitude_start)
                / max(self.scenario.world_z_max - high_altitude_start, 1e-6),
                0.0,
                1.0,
            )
        )
        gamma_ratio = float(
            np.clip(
                (gamma - self.rewards.climb_trend_gamma_threshold)
                / max(self.scenario.gamma_max - self.rewards.climb_trend_gamma_threshold, 1e-6),
                0.0,
                1.0,
            )
        )
        penalty = self.rewards.climb_trend_penalty_weight * altitude_ratio * gamma_ratio
        return min(penalty, self.rewards.climb_trend_penalty_cap)

    def _goal_distance(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - self.goal))

    @staticmethod
    def _inside_zone(pos: np.ndarray, zone: Zone) -> bool:
        distance = (pos[0] - zone.center_xy[0]) ** 2 + (pos[1] - zone.center_xy[1]) ** 2 + pos[2] ** 2
        return bool(distance <= zone.radius**2)

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return ((value + math.pi) % (2 * math.pi)) - math.pi

    def _info(self, *, progress: float, outcome: str = 'running') -> dict[str, Any]:
        return {
            'goal_distance': self._goal_distance(self.state[:3]),
            'progress': progress,
            'outcome': outcome,
            'steps': self.steps,
            'curriculum_level': self.last_curriculum_level,
        }
