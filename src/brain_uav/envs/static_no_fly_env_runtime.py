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
from ..curriculum import normalize_curriculum_mix
from ..utils.gym_compat import gym, spaces


@dataclass(slots=True)
class Zone:
    """One static hemisphere no-fly zone."""

    center_xy: np.ndarray
    radius: float


class StaticNoFlyTrajectoryEnv(gym.Env):
    """Gymnasium-style environment for static no-fly-zone trajectory planning."""

    metadata = {"render_modes": ["human"]}
    _distance_reward_scale_compensation = 10.0

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

        obs_dim = 5 + 3 + 4 + 4 * self.scenario.nearest_zone_count
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
        self.best_goal_distance_so_far = 0.0
        self.last_segment_goal_distance = float('inf')
        self.last_goal_reached_by_segment = False

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
        self.best_goal_distance_so_far = self._goal_distance(self.state[:3])
        self.last_segment_goal_distance = self.best_goal_distance_so_far
        self.last_goal_reached_by_segment = False
        return self._get_obs(), self._info(progress=0.0)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).clip(self.action_space.low, self.action_space.high)
        prev_state = self.state.copy()
        prev_action = self.prev_action.copy()
        prev_distance = self._goal_distance(prev_state[:3])
        prev_best_goal_distance = self.best_goal_distance_so_far
        self._apply_action(action)
        self.last_delta_z = float(self.state[2] - prev_state[2])
        self.steps += 1
        self.trajectory.append(self.state[:3].copy())
        new_distance = self._goal_distance(self.state[:3])
        self.last_segment_goal_distance = self._segment_goal_distance(prev_state[:3], self.state[:3])
        self.last_goal_reached_by_segment = self.last_segment_goal_distance <= self._active_goal_radius()
        step_progress = prev_distance - new_distance
        self._record_progress(step_progress)
        terminated, truncated, outcome = self._termination()
        reward = self._compute_reward(
            prev_state,
            prev_action,
            prev_distance,
            new_distance,
            action,
            outcome,
            prev_best_goal_distance,
        )
        if new_distance < self.best_goal_distance_so_far:
            self.best_goal_distance_so_far = new_distance
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

    def set_goal(self, new_goal, *, reset_leg_timer: bool = True):
        self.goal = np.asarray(new_goal, dtype=np.float32).copy()
        if reset_leg_timer:
            self.steps = 0
        self.recent_progress = []
        self.best_goal_distance_so_far = self._goal_distance(self.state[:3])
        self.last_segment_goal_distance = self.best_goal_distance_so_far
        self.last_goal_reached_by_segment = False
        return self._get_obs(), self._info(progress=0.0)

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
        if level == 'easy_two_zone':
            return self._sample_easy_two_zone_scenario()
        if level == 'medium':
            return self._sample_medium_scenario()
        if level == 'hard':
            return self._sample_hard_scenario()
        raise ValueError(f'Unsupported curriculum level: {level}')

    def _distance_range_for_level(self, level: str) -> tuple[float, float]:
        return self.scenario.distance_range_for_level(level)

    def _z_sampling_spec(
        self,
        level: str,
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        cfg = self.scenario
        z_specs = {
            'easy': ((0.16, 0.26), (0.16, 0.30), 0.10),
            'easy_two_zone': ((0.17, 0.28), (0.17, 0.33), 0.12),
            'medium': ((0.18, 0.30), (0.18, 0.35), 0.14),
            'hard': ((0.18, 0.30), (0.18, 0.36), 0.15),
            'benchmark': ((0.18, 0.30), (0.18, 0.36), 0.15),
        }
        try:
            state_ratio_range, goal_ratio_range, max_gap_ratio = z_specs[level]
        except KeyError as exc:
            raise ValueError(f'Unsupported z sampling level: {level}') from exc
        state_z_range = tuple(ratio * cfg.world_z_max for ratio in state_ratio_range)
        goal_z_range = tuple(ratio * cfg.world_z_max for ratio in goal_ratio_range)
        max_height_gap = max_gap_ratio * cfg.world_z_max
        return state_z_range, goal_z_range, max_height_gap

    def _sample_start_goal_pair(
        self,
        level: str,
        *,
        mean_y_ratio: float,
        lateral_offset_ratio: float,
        psi_range: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        cfg = self.scenario
        dist_min, dist_max = self._distance_range_for_level(level)
        state_z_range, goal_z_range, max_height_gap = self._z_sampling_spec(level)
        for _ in range(40):
            distance = float(self.rng.uniform(dist_min, dist_max))
            mean_y = float(self.rng.uniform(-mean_y_ratio * cfg.world_xy, mean_y_ratio * cfg.world_xy))
            state_z = float(self.rng.uniform(*state_z_range))
            goal_z = float(self.rng.uniform(*goal_z_range))
            delta_z = goal_z - state_z
            if abs(delta_z) > max_height_gap:
                continue
            lateral_offset = float(self.rng.uniform(-lateral_offset_ratio * distance, lateral_offset_ratio * distance))
            remaining_sq = distance**2 - lateral_offset**2 - delta_z**2
            if remaining_sq <= 1e-6:
                continue
            delta_x = math.sqrt(remaining_sq)
            state = np.array(
                [
                    -0.5 * delta_x,
                    mean_y - 0.5 * lateral_offset,
                    state_z,
                    0.0,
                    self.rng.uniform(*psi_range),
                ],
                dtype=np.float32,
            )
            goal = np.array(
                [
                    0.5 * delta_x,
                    mean_y + 0.5 * lateral_offset,
                    goal_z,
                ],
                dtype=np.float32,
            )
            if max(abs(float(state[0])), abs(float(goal[0]))) > cfg.world_xy:
                continue
            if max(abs(float(state[1])), abs(float(goal[1]))) > cfg.world_xy:
                continue
            return state, goal
        return None

    def _sample_easy_scenario(self) -> dict[str, Any] | None:
        cfg = self.scenario
        for _ in range(40):
            pair = self._sample_start_goal_pair(
                'easy',
                mean_y_ratio=0.10,
                lateral_offset_ratio=0.06,
                psi_range=(-0.10, 0.10),
            )
            if pair is None:
                continue
            state, goal = pair
            radius = self._sample_zone_radius('easy')
            line_y = float((state[1] + goal[1]) * 0.5)
            side_offset = float(self.rng.uniform(1.35 * radius, 1.60 * radius))
            zone = Zone(
                center_xy=np.array(
                    [
                        self.rng.uniform(-0.05 * cfg.world_xy, 0.35 * cfg.world_xy),
                        line_y + self.rng.choice([-1.0, 1.0]) * side_offset,
                    ],
                    dtype=np.float32,
                ),
                radius=radius,
            )
            if not self._zone_candidate_is_valid(state, goal, [], zone.center_xy, zone.radius):
                continue
            blockers = self._count_corridor_blockers(
                state,
                goal,
                [zone],
                margin=cfg.corridor_blocking_margin,
            )
            if blockers != 0:
                continue
            return {
                'state': state.tolist(),
                'goal': goal.tolist(),
                'zones': [{'center_xy': zone.center_xy.tolist(), 'radius': zone.radius}],
                'curriculum_level': 'easy',
            }
        return None

    def _sample_easy_two_zone_scenario(self) -> dict[str, Any] | None:
        cfg = self.scenario
        for _ in range(70):
            pair = self._sample_start_goal_pair(
                'easy_two_zone',
                mean_y_ratio=0.12,
                lateral_offset_ratio=0.08,
                psi_range=(-0.12, 0.12),
            )
            if pair is None:
                continue
            state, goal = pair
            force_blocker = bool(self.rng.random() < cfg.easy_two_zone_blocker_probability)
            zones = self._sample_easy_two_zone_pair(state, goal, force_blocker)
            if not zones:
                continue
            blockers = self._count_corridor_blockers(state, goal, zones, margin=cfg.corridor_blocking_margin)
            if force_blocker and not (1 <= blockers <= 2):
                continue
            if not force_blocker and blockers > 1:
                continue
            return {
                'state': state.tolist(),
                'goal': goal.tolist(),
                'zones': [
                    {'center_xy': zone.center_xy.tolist(), 'radius': zone.radius}
                    for zone in zones
                ],
                'curriculum_level': 'easy_two_zone',
            }
        return None

    def _sample_easy_two_zone_pair(self, state: np.ndarray, goal: np.ndarray, force_blocker: bool) -> list[Zone] | None:
        cfg = self.scenario
        zones: list[Zone] = []
        mean_y = 0.5 * (state[1] + goal[1])
        radius_min, radius_max = cfg.radius_range_for_level('easy_two_zone')
        if force_blocker:
            radius_1 = float(self.rng.uniform(radius_min, radius_max))
            center_1 = np.array(
                [
                    self.rng.uniform(-0.02 * cfg.world_xy, 0.22 * cfg.world_xy),
                    mean_y + self.rng.uniform(-0.15 * radius_1, 0.15 * radius_1),
                ],
                dtype=np.float32,
            )
            if not self._zone_candidate_is_valid(state, goal, zones, center_1, radius_1):
                return None
            zones.append(Zone(center_xy=center_1, radius=radius_1))

            side = float(self.rng.choice([-1.0, 1.0]))
            radius_2 = float(self.rng.uniform(radius_min, radius_max))
            center_2 = np.array(
                [
                    center_1[0] + self.rng.uniform(0.85 * radius_1, 1.25 * radius_1),
                    mean_y + side * self.rng.uniform(0.75 * radius_2, 1.10 * radius_2),
                ],
                dtype=np.float32,
            )
            center_2 = self._separate_zone_center(center_2, zones, radius_2, cfg.easy_two_zone_min_gap)
            if not self._zone_candidate_is_valid(state, goal, zones, center_2, radius_2):
                return None
            zones.append(Zone(center_xy=center_2, radius=radius_2))
        else:
            base_x = self.rng.uniform(-0.05 * cfg.world_xy, 0.20 * cfg.world_xy)
            signs = [1.0, -1.0]
            self.rng.shuffle(signs)
            x_cursor = 0.0
            for idx, side in enumerate(signs):
                radius = float(self.rng.uniform(radius_min, radius_max))
                if idx > 0:
                    x_cursor += float(self.rng.uniform(0.85 * radius, 1.20 * radius))
                offset = side * float(self.rng.uniform(0.80 * radius, 1.20 * radius))
                center_xy = np.array(
                    [
                        base_x + x_cursor,
                        mean_y + offset,
                    ],
                    dtype=np.float32,
                )
                center_xy = self._separate_zone_center(center_xy, zones, radius, cfg.easy_two_zone_min_gap)
                if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius):
                    return None
                zones.append(Zone(center_xy=center_xy, radius=radius))

        if not self._double_zone_layout_is_reasonable(state, goal, zones, cfg.easy_two_zone_min_gap):
            return None
        return zones

    def _sample_medium_scenario(self) -> dict[str, Any] | None:
        cfg = self.scenario
        for _ in range(60):
            pair = self._sample_start_goal_pair(
                'medium',
                mean_y_ratio=0.16,
                lateral_offset_ratio=0.10,
                psi_range=(-0.15, 0.15),
            )
            if pair is None:
                continue
            state, goal = pair
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
        radius = self._sample_zone_radius('medium')
        center_xy = np.array(
            [
                self.rng.uniform(0.00 * cfg.world_xy, 0.30 * cfg.world_xy),
                self.rng.uniform(-0.12 * radius, 0.12 * radius) + 0.5 * (state[1] + goal[1]),
            ],
            dtype=np.float32,
        )
        if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius):
            return None
        zones.append(Zone(center_xy=center_xy, radius=radius))
        return zones

    def _sample_medium_double_detour(self, state: np.ndarray, goal: np.ndarray) -> list[Zone] | None:
        cfg = self.scenario
        zones: list[Zone] = []
        base_x = self.rng.uniform(-0.05 * cfg.world_xy, 0.20 * cfg.world_xy)
        signs = [1.0, -1.0]
        self.rng.shuffle(signs)
        x_cursor = 0.0
        for idx, side in enumerate(signs):
            radius = self._sample_zone_radius('medium')
            if idx > 0:
                x_cursor += float(self.rng.uniform(0.80 * radius, 1.15 * radius))
            offset = side * float(self.rng.uniform(0.70 * radius, 1.05 * radius))
            center_xy = np.array(
                [
                    base_x + x_cursor,
                    0.5 * (state[1] + goal[1]) + offset,
                ],
                dtype=np.float32,
            )
            center_xy = self._separate_zone_center(center_xy, zones, radius, cfg.dual_zone_min_margin)
            if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius):
                return None
            zones.append(Zone(center_xy=center_xy, radius=radius))
        if not self._double_zone_layout_is_reasonable(state, goal, zones, cfg.dual_zone_min_margin):
            return None
        return zones

    def _sample_hard_scenario(self) -> dict[str, Any] | None:
        cfg = self.scenario
        for _attempt in range(cfg.scenario_max_sampling_attempts):
            pair = self._sample_start_goal_pair(
                'hard',
                mean_y_ratio=0.22,
                lateral_offset_ratio=0.12,
                psi_range=(-0.2, 0.2),
            )
            if pair is None:
                continue
            state, goal = pair
            zones = self._sample_zones_for_pair(state, goal, level='hard')
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

    def _flight_direction(self, gamma: float, psi: float) -> np.ndarray:
        direction = np.array(
            [
                math.cos(gamma) * math.cos(psi),
                math.cos(gamma) * math.sin(psi),
                math.sin(gamma),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return direction / norm

    def _goal_direction(self, pos: np.ndarray) -> np.ndarray | None:
        rel_goal = self.goal - pos
        norm = float(np.linalg.norm(rel_goal))
        if norm <= 1e-6:
            return None
        return rel_goal / norm

    def _inside_zone_with_clearance(self, pos: np.ndarray, zone: Zone, clearance: float = 0.0) -> bool:
        effective_radius = zone.radius + max(clearance, 0.0)
        distance = (pos[0] - zone.center_xy[0]) ** 2 + (pos[1] - zone.center_xy[1]) ** 2 + pos[2] ** 2
        return bool(distance <= effective_radius**2)

    def _line_to_goal_is_safe(self, pos: np.ndarray, samples: int = 32, clearance: float = 0.0) -> bool:
        if not self.zones:
            return True
        sample_count = max(int(samples), 1)
        for idx in range(1, sample_count + 1):
            t = idx / sample_count
            point = pos + t * (self.goal - pos)
            if any(self._inside_zone_with_clearance(point, zone, clearance) for zone in self.zones):
                return False
        return True

    def _terminal_reward_active(self, pos: np.ndarray, radius: float, outcome: str) -> bool:
        if outcome in {'collision', 'ground', 'boundary'}:
            return False
        if self._goal_distance(pos) > radius:
            return False
        return self._line_to_goal_is_safe(pos, clearance=0.0)

    def _terminal_los_reward(self, pos: np.ndarray, gamma: float, psi: float, outcome: str) -> float:
        if not self._terminal_reward_active(pos, self.rewards.terminal_guidance_radius, outcome):
            return 0.0
        goal_dir = self._goal_direction(pos)
        if goal_dir is None:
            return 0.0
        alignment = float(np.dot(self._flight_direction(gamma, psi), goal_dir))
        if alignment >= 0.0:
            reward = self.rewards.terminal_los_weight * alignment
        else:
            reward = -self.rewards.terminal_los_penalty_weight * abs(alignment)
        cap = self.rewards.terminal_los_reward_cap
        return float(np.clip(reward, -cap, cap))

    def _terminal_radial_tangential_reward(self, pos: np.ndarray, gamma: float, psi: float, outcome: str) -> float:
        if not self._terminal_reward_active(pos, self.rewards.terminal_tangential_radius, outcome):
            return 0.0
        goal_dir = self._goal_direction(pos)
        if goal_dir is None:
            return 0.0
        radial_component = float(np.dot(self._flight_direction(gamma, psi), goal_dir))
        tangential_component = math.sqrt(max(1.0 - radial_component**2, 0.0))
        reward = self.rewards.terminal_radial_weight * max(radial_component, 0.0)
        penalty = min(
            self.rewards.terminal_tangential_penalty_weight * tangential_component,
            self.rewards.terminal_tangential_penalty_cap,
        )
        return reward - penalty

    def _sample_zone_radius(self, level: str) -> float:
        radius_min, radius_max = self.scenario.radius_range_for_level(level)
        return float(self.rng.uniform(radius_min, radius_max))

    def _separate_zone_center(
        self,
        center_xy: np.ndarray,
        existing_zones: list[Zone],
        radius: float,
        min_margin: float,
    ) -> np.ndarray:
        adjusted = np.asarray(center_xy, dtype=np.float32).copy()
        for zone in existing_zones:
            delta = adjusted - zone.center_xy
            distance = float(np.linalg.norm(delta))
            min_distance = zone.radius + radius + min_margin + 1.0
            if distance > min_distance:
                continue
            if distance <= 1e-6:
                delta = np.array([1.0, 0.0], dtype=np.float32)
                distance = 1.0
            adjusted = zone.center_xy + delta / distance * min_distance
        limit = float(self.scenario.world_xy)
        return np.clip(adjusted, -limit, limit).astype(np.float32)

    def _sample_zones_for_pair(self, state: np.ndarray, goal: np.ndarray, level: str = 'hard') -> list[Zone] | None:
        cfg = self.scenario
        zones: list[Zone] = []
        radius_min, radius_max = cfg.radius_range_for_level(level)
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
                radius = float(self.rng.uniform(radius_min, radius_max))
                if zones:
                    center_xy = self._separate_zone_center(center_xy, zones, radius, cfg.dual_zone_min_margin)
                if not self._zone_gap_is_valid(zones, center_xy, radius, cfg.dual_zone_min_margin):
                    continue
                if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius):
                    continue
                zones.append(Zone(center_xy=center_xy, radius=radius))
                accepted = True
                break
            if not accepted:
                return None
        return zones

    @staticmethod
    def _zone_gap_is_valid(
        existing_zones: list[Zone],
        center_xy: np.ndarray,
        radius: float,
        min_margin: float,
    ) -> bool:
        for zone in existing_zones:
            center_distance = float(np.linalg.norm(center_xy - zone.center_xy))
            if center_distance <= zone.radius + radius + min_margin:
                return False
        return True

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
        safe_margin = radius + cfg.warning_distance + cfg.goal_radius + 1.0
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

    def _double_zone_layout_is_reasonable(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        zones: list[Zone],
        min_margin: float,
    ) -> bool:
        if len(zones) < 2:
            return True
        for idx, zone_a in enumerate(zones):
            for zone_b in zones[idx + 1 :]:
                center_distance = float(np.linalg.norm(zone_a.center_xy - zone_b.center_xy))
                if center_distance <= zone_a.radius + zone_b.radius + min_margin:
                    return False
        blockers = self._count_corridor_blockers(state, goal, zones, margin=self.scenario.corridor_blocking_margin)
        if blockers > 2:
            return False
        return True

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
            r_xy = float(np.linalg.norm([dx, dy]))
            if r_xy < zone.radius:
                z_cap = math.sqrt(max(zone.radius ** 2 - r_xy ** 2, 0.0))
            else:
                z_cap = 0.0
            z_margin_to_dome = float(self.state[2] - z_cap)
            zone_features.extend([float(dx), float(dy), float(zone.radius), z_margin_to_dome])
        while len(zone_features) < self.scenario.nearest_zone_count * 4:
            zone_features.extend([0.0, 0.0, 0.0, 0.0])
        return np.concatenate(
            [own_state, rel_goal.astype(np.float32), extra_features, np.array(zone_features, dtype=np.float32)]
        ).astype(np.float32)

    def _termination(self) -> tuple[bool, bool, str]:
        cfg = self.scenario
        pos = self.state[:3]
        if self._goal_distance(pos) <= self._active_goal_radius() or self.last_goal_reached_by_segment:
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
        prev_best_goal_distance: float,
    ) -> float:
        rew = self.rewards.progress_weight * (
            (prev_distance - new_distance) * self._distance_reward_scale_compensation
        )
        rew += self._breakthrough_reward(new_distance, prev_best_goal_distance, outcome)
        rew += self._terminal_los_reward(self.state[:3], float(self.state[3]), float(self.state[4]), outcome)
        rew += self._terminal_radial_tangential_reward(
            self.state[:3],
            float(self.state[3]),
            float(self.state[4]),
            outcome,
        )
        rew -= self.rewards.step_penalty
        rew -= self.rewards.smoothness_weight * float(np.square(action).sum())
        rew -= self._action_change_penalty(prev_action, action)
        rew -= self._zone_warning_penalty(self.state[:3])
        rew -= self._boundary_warning_penalty(self.state[:3])
        rew -= self._ground_warning_penalty(self.state[:3])
        rew -= self._descent_trend_penalty(prev_state, self.state)
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

    def _breakthrough_reward(
        self,
        new_distance: float,
        prev_best_goal_distance: float,
        outcome: str,
    ) -> float:
        if outcome in {'collision', 'ground', 'boundary'}:
            return 0.0
        if new_distance >= prev_best_goal_distance:
            return 0.0
        if len(self.recent_progress) < self.rewards.progress_window_size:
            return 0.0
        window_progress = float(sum(self.recent_progress))
        if window_progress < self.rewards.breakthrough_progress_threshold:
            return 0.0
        if self._nearest_zone_surface_clearance(self.state[:3]) > self.rewards.breakthrough_reward_distance:
            return 0.0
        reward = self.rewards.breakthrough_reward_weight * (
            window_progress * self._distance_reward_scale_compensation
        )
        return min(reward, self.rewards.breakthrough_reward_cap)

    def _nearest_zone_surface_clearance(self, pos: np.ndarray) -> float:
        if not self.zones:
            return float('inf')
        clearances = []
        for zone in self.zones:
            center_distance = float(
                np.linalg.norm(np.array([pos[0] - zone.center_xy[0], pos[1] - zone.center_xy[1], pos[2]]))
            )
            clearances.append(center_distance - zone.radius)
        return min(clearances)

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
        warning_height = min(self.scenario.ground_warning_height, 8.0)
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

    def _goal_distance(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - self.goal))

    def _active_goal_radius(self) -> float:
        if self.scenario.goal_radius_curriculum_enabled:
            level = self.last_curriculum_level
            if level in self.scenario.goal_radius_curriculum:
                return float(self.scenario.goal_radius_curriculum[level])
        return float(self.scenario.goal_radius)

    def _segment_goal_distance(self, start: np.ndarray, end: np.ndarray) -> float:
        segment = end - start
        segment_norm_sq = float(np.dot(segment, segment))
        if segment_norm_sq <= 1e-12:
            return self._goal_distance(end)
        projection = float(np.dot(self.goal - start, segment) / segment_norm_sq)
        projection = float(np.clip(projection, 0.0, 1.0))
        closest = start + projection * segment
        return float(np.linalg.norm(closest - self.goal))

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
            'segment_goal_distance': float(self.last_segment_goal_distance),
            'goal_reached_by_segment': bool(self.last_goal_reached_by_segment),
            'progress': progress,
            'outcome': outcome,
            'steps': self.steps,
            'curriculum_level': self.last_curriculum_level,
            'active_goal_radius': self._active_goal_radius(),
        }
