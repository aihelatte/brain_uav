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
    """Gymnasium-style environment for static no-fly-zone trajectory planning.

    Distances are interpreted as km, time as s, and speed as km/s. The goal
    success check intentionally uses the fixed ``scenario.goal_radius`` only.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario: ScenarioConfig | None = None,
        rewards: RewardConfig | None = None,
        seed: int | None = None,
        fixed_scenarios: list[dict[str, Any]] | None = None,
        curriculum_mix: dict[str, float] | None = None,
        goal_radius_curriculum_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.scenario = scenario or ScenarioConfig()
        self._normalize_time_limit()
        self.rewards = rewards or RewardConfig()
        self.fixed_scenarios = fixed_scenarios or []
        self.curriculum_mix = normalize_curriculum_mix(curriculum_mix, fallback_level='hard') if curriculum_mix else None
        self.goal_radius_curriculum_enabled = bool(goal_radius_curriculum_enabled)
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
        self.active_goal_radius = float(self.scenario.goal_radius)
        self.last_segment_goal_distance = 0.0
        self.last_goal_reached_by_segment = False

    def _normalize_time_limit(self) -> None:
        """Fill missing time-limit fields without overriding explicit max_steps."""

        if self.scenario.dt <= 0.0:
            raise ValueError('scenario.dt must be positive.')
        max_time_s = getattr(self.scenario, 'max_time_s', None)
        if self.scenario.max_steps is None:
            if max_time_s is None:
                raise ValueError('scenario.max_steps or scenario.max_time_s must be set.')
            self.scenario.max_steps = int(float(max_time_s) / float(self.scenario.dt))

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
        self.last_segment_goal_distance = self._segment_goal_distance(prev_state[:3], self.state[:3])
        self.steps += 1
        self.trajectory.append(self.state[:3].copy())
        new_distance = self._goal_distance(self.state[:3])
        self.last_goal_reached_by_segment = (
            new_distance > self.active_goal_radius
            and self.last_segment_goal_distance <= self.active_goal_radius
        )
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
            'distance_unit': 'km',
            'time_unit': 's',
            'goal_radius': float(self.scenario.goal_radius),
            'active_goal_radius': float(self.active_goal_radius),
            **self._start_goal_metadata(self.initial_state, self.goal),
        }

    def _sample_scenario(self) -> None:
        if self.curriculum_mix:
            for _ in range(self.scenario.scenario_max_sampling_attempts):
                level = self._sample_curriculum_level()
                scenario = self._sample_curriculum_scenario(level)
                if scenario is not None:
                    self._load_scenario(scenario, use_curriculum_radius=self.goal_radius_curriculum_enabled)
                    self.last_curriculum_level = level
                    return
            raise RuntimeError('Failed to sample a curriculum scenario under current constraints.')

        scenario = self._sample_curriculum_scenario('hard')
        if scenario is None:
            raise RuntimeError('Failed to sample a random hard scenario under current constraints.')
        self._load_scenario(scenario, use_curriculum_radius=self.goal_radius_curriculum_enabled)
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

    def _start_goal_distance_range(self, level: str) -> tuple[float, float]:
        return getattr(self.scenario, f'{level}_start_goal_distance_range')

    def _no_fly_radius_range(self, level: str) -> tuple[float, float]:
        return getattr(self.scenario, f'{level}_no_fly_radius_range')

    def _min_zone_surface_gap(self, level: str) -> float:
        return float(getattr(self.scenario, f'{level}_min_zone_surface_gap'))

    def _sample_zone_radius(self, level: str) -> float:
        return float(self.rng.uniform(*self._no_fly_radius_range(level)))

    def _sample_start_goal_pair(
        self,
        level: str,
        start_x_fraction: tuple[float, float],
        goal_x_fraction: tuple[float, float],
        start_y_fraction: tuple[float, float],
        goal_y_fraction: tuple[float, float],
        psi_range: tuple[float, float],
        attempts: int = 80,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        cfg = self.scenario
        for _ in range(attempts):
            state = np.array(
                [
                    self.rng.uniform(start_x_fraction[0] * cfg.world_xy, start_x_fraction[1] * cfg.world_xy),
                    self.rng.uniform(start_y_fraction[0] * cfg.world_xy, start_y_fraction[1] * cfg.world_xy),
                    self.rng.uniform(*cfg.start_z_range),
                    0.0,
                    self.rng.uniform(*psi_range),
                ],
                dtype=np.float32,
            )
            goal = np.array(
                [
                    self.rng.uniform(goal_x_fraction[0] * cfg.world_xy, goal_x_fraction[1] * cfg.world_xy),
                    self.rng.uniform(goal_y_fraction[0] * cfg.world_xy, goal_y_fraction[1] * cfg.world_xy),
                    self.rng.uniform(*cfg.goal_z_range),
                ],
                dtype=np.float32,
            )
            if self._start_goal_profile_is_valid(state, goal, level):
                return state, goal
        return None

    def _start_goal_metadata(self, state: np.ndarray, goal: np.ndarray) -> dict[str, float]:
        delta = goal.astype(np.float32) - state[:3].astype(np.float32)
        return {
            'start_goal_distance_km': float(np.linalg.norm(delta)),
            'start_goal_horizontal_distance_km': float(np.linalg.norm(delta[:2])),
        }

    def _start_goal_distance_is_valid(self, state: np.ndarray, goal: np.ndarray, level: str) -> bool:
        distance = self._start_goal_metadata(state, goal)['start_goal_distance_km']
        low, high = self._start_goal_distance_range(level)
        return low <= distance <= high

    def _height_profile_is_valid(self, state: np.ndarray, goal: np.ndarray) -> bool:
        cfg = self.scenario
        if not (cfg.world_z_min < float(state[2]) < cfg.world_z_max):
            return False
        if not (cfg.world_z_min < float(goal[2]) < cfg.world_z_max):
            return False
        return abs(float(goal[2] - state[2])) <= cfg.max_start_goal_height_gap

    def _start_goal_profile_is_valid(self, state: np.ndarray, goal: np.ndarray, level: str) -> bool:
        return (
            self._point_is_in_bounds(state[:3])
            and self._point_is_in_bounds(goal)
            and self._height_profile_is_valid(state, goal)
            and self._start_goal_distance_is_valid(state, goal, level)
        )

    def _build_scenario_payload(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        zones: list[Zone],
        level: str,
    ) -> dict[str, Any]:
        return {
            'state': state.tolist(),
            'goal': goal.tolist(),
            'zones': [
                {'center_xy': zone.center_xy.tolist(), 'radius': zone.radius}
                for zone in zones
            ],
            'curriculum_level': level,
            'distance_unit': 'km',
            'time_unit': 's',
            'goal_radius': float(self.scenario.goal_radius),
            'active_goal_radius': float(self._goal_radius_for_level(level)),
            **self._start_goal_metadata(state, goal),
        }

    def _point_is_in_bounds(self, pos: np.ndarray) -> bool:
        cfg = self.scenario
        return bool(
            abs(float(pos[0])) <= cfg.world_xy
            and abs(float(pos[1])) <= cfg.world_xy
            and cfg.world_z_min < float(pos[2]) < cfg.world_z_max
        )

    def _zone_layout_is_valid(self, zones: list[Zone], level: str) -> bool:
        min_surface_gap = self._min_zone_surface_gap(level)
        for idx, zone in enumerate(zones):
            if not self._zone_boundary_is_valid(zone.center_xy, zone.radius):
                return False
            for other in zones[idx + 1 :]:
                center_distance = float(np.linalg.norm(zone.center_xy - other.center_xy))
                min_allowed = zone.radius + other.radius + min_surface_gap
                if center_distance <= min_allowed:
                    return False
        return True

    def _scenario_geometry_is_valid(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        zones: list[Zone],
        level: str,
    ) -> bool:
        if not self._start_goal_profile_is_valid(state, goal, level):
            return False
        if not self._zone_layout_is_valid(zones, level):
            return False
        if any(self._inside_zone(state[:3], zone) or self._inside_zone(goal, zone) for zone in zones):
            return False
        if not self._corridor_is_reasonable(state, goal, zones):
            return False
        return True

    def _sample_easy_scenario(self) -> dict[str, Any] | None:
        cfg = self.scenario
        for _ in range(40):
            pair = self._sample_start_goal_pair(
                'easy',
                start_x_fraction=(-0.58, -0.42),
                goal_x_fraction=(0.25, 0.42),
                start_y_fraction=(-0.10, 0.10),
                goal_y_fraction=(-0.12, 0.12),
                psi_range=(-0.10, 0.10),
            )
            if pair is None:
                continue
            state, goal = pair
            radius = self._sample_zone_radius('easy')
            line_y = float((state[1] + goal[1]) * 0.5)
            zone = Zone(
                center_xy=np.array(
                    [
                        self.rng.uniform(-0.05 * cfg.world_xy, 0.35 * cfg.world_xy),
                        line_y + self.rng.choice([-1.0, 1.0]) * self.rng.uniform(300.0, 450.0),
                    ],
                    dtype=np.float32,
                ),
                radius=radius,
            )
            if not self._zone_candidate_is_valid(state, goal, [], zone.center_xy, zone.radius, 'easy'):
                continue
            blockers = self._count_corridor_blockers(state, goal, [zone], margin=20.0)
            if blockers != 0:
                continue
            zones = [zone]
            if not self._scenario_geometry_is_valid(state, goal, zones, 'easy'):
                continue
            return self._build_scenario_payload(state, goal, zones, 'easy')
        return None

    def _sample_easy_two_zone_scenario(self) -> dict[str, Any] | None:
        cfg = self.scenario
        for _ in range(70):
            pair = self._sample_start_goal_pair(
                'easy_two_zone',
                start_x_fraction=(-0.62, -0.45),
                goal_x_fraction=(0.35, 0.55),
                start_y_fraction=(-0.12, 0.12),
                goal_y_fraction=(-0.16, 0.16),
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
            if not self._scenario_geometry_is_valid(state, goal, zones, 'easy_two_zone'):
                continue
            return self._build_scenario_payload(state, goal, zones, 'easy_two_zone')
        return None

    def _sample_easy_two_zone_pair(self, state: np.ndarray, goal: np.ndarray, force_blocker: bool) -> list[Zone] | None:
        cfg = self.scenario
        zones: list[Zone] = []
        mean_y = 0.5 * (state[1] + goal[1])
        if force_blocker:
            center_1 = np.array(
                [
                    self.rng.uniform(-0.02 * cfg.world_xy, 0.22 * cfg.world_xy),
                    mean_y + self.rng.uniform(-45.0, 45.0),
                ],
                dtype=np.float32,
            )
            radius_1 = self._sample_zone_radius('easy_two_zone')
            if not self._zone_candidate_is_valid(state, goal, zones, center_1, radius_1, 'easy_two_zone'):
                return None
            zones.append(Zone(center_xy=center_1, radius=radius_1))

            side = float(self.rng.choice([-1.0, 1.0]))
            center_2 = np.array(
                [
                    center_1[0] + self.rng.uniform(230.0, 360.0),
                    mean_y + side * self.rng.uniform(230.0, 380.0),
                ],
                dtype=np.float32,
            )
            radius_2 = self._sample_zone_radius('easy_two_zone')
            if not self._zone_candidate_is_valid(state, goal, zones, center_2, radius_2, 'easy_two_zone'):
                return None
            zones.append(Zone(center_xy=center_2, radius=radius_2))
        else:
            base_x = self.rng.uniform(-0.05 * cfg.world_xy, 0.20 * cfg.world_xy)
            offsets = [self.rng.uniform(220.0, 360.0), -self.rng.uniform(220.0, 360.0)]
            self.rng.shuffle(offsets)
            for idx, offset in enumerate(offsets):
                center_xy = np.array(
                    [
                        base_x + idx * self.rng.uniform(240.0, 380.0),
                        mean_y + offset,
                    ],
                    dtype=np.float32,
                )
                radius = self._sample_zone_radius('easy_two_zone')
                if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius, 'easy_two_zone'):
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
                start_x_fraction=(-0.68, -0.50),
                goal_x_fraction=(0.45, 0.65),
                start_y_fraction=(-0.16, 0.16),
                goal_y_fraction=(-0.20, 0.20),
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
            if not self._scenario_geometry_is_valid(state, goal, zones, 'medium'):
                continue
            return self._build_scenario_payload(state, goal, zones, 'medium')
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
        radius = self._sample_zone_radius('medium')
        if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius, 'medium'):
            return None
        zones.append(Zone(center_xy=center_xy, radius=radius))
        return zones

    def _sample_medium_double_detour(self, state: np.ndarray, goal: np.ndarray) -> list[Zone] | None:
        cfg = self.scenario
        zones: list[Zone] = []
        base_x = self.rng.uniform(-0.05 * cfg.world_xy, 0.20 * cfg.world_xy)
        offsets = [self.rng.uniform(260.0, 420.0), -self.rng.uniform(260.0, 420.0)]
        self.rng.shuffle(offsets)
        for idx, offset in enumerate(offsets):
            center_xy = np.array(
                [
                    base_x + idx * self.rng.uniform(260.0, 420.0),
                    0.5 * (state[1] + goal[1]) + offset,
                ],
                dtype=np.float32,
            )
            radius = self._sample_zone_radius('medium')
            if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius, 'medium'):
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
                start_x_fraction=(-0.72, -0.55),
                goal_x_fraction=(0.55, 0.78),
                start_y_fraction=(-0.20, 0.20),
                goal_y_fraction=(-0.30, 0.30),
                psi_range=(-0.20, 0.20),
            )
            if pair is None:
                continue
            state, goal = pair
            zones = self._sample_zones_for_pair(state, goal, 'hard')
            if zones is None:
                continue
            if not self._scenario_geometry_is_valid(state, goal, zones, 'hard'):
                continue
            return self._build_scenario_payload(state, goal, zones, 'hard')
        return None

    def _sample_zones_for_pair(self, state: np.ndarray, goal: np.ndarray, level: str) -> list[Zone] | None:
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
                radius = self._sample_zone_radius(level)
                if not self._zone_candidate_is_valid(state, goal, zones, center_xy, radius, level):
                    continue
                zones.append(Zone(center_xy=center_xy, radius=radius))
                accepted = True
                break
            if not accepted:
                return None
        return zones

    def _zone_boundary_is_valid(self, center_xy: np.ndarray, radius: float) -> bool:
        cfg = self.scenario
        lateral_clearance = radius + cfg.warning_distance
        return bool(
            abs(float(center_xy[0])) + lateral_clearance <= cfg.world_xy
            and abs(float(center_xy[1])) + lateral_clearance <= cfg.world_xy
            and radius < cfg.world_z_max
        )

    def _zone_candidate_is_valid(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        existing_zones: list[Zone],
        center_xy: np.ndarray,
        radius: float,
        level: str,
    ) -> bool:
        cfg = self.scenario
        if not self._zone_boundary_is_valid(center_xy, radius):
            return False

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
        if level == 'easy' and not self._easy_corridor_clearance_is_valid(state, goal, center_xy, radius):
            return False

        for zone in existing_zones:
            center_distance = float(np.linalg.norm(center_xy - zone.center_xy))
            min_allowed = radius + zone.radius + self._min_zone_surface_gap(level)
            if center_distance <= min_allowed:
                return False
        return True

    def _distance_to_start_goal_segment_xy(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        center_xy: np.ndarray,
    ) -> float:
        start_xy = state[:2]
        goal_xy = goal[:2]
        segment = goal_xy - start_xy
        segment_norm_sq = float(np.dot(segment, segment))
        if segment_norm_sq <= 1e-6:
            return float(np.linalg.norm(center_xy - start_xy))
        t = float(np.dot(center_xy - start_xy, segment) / segment_norm_sq)
        t = float(np.clip(t, 0.0, 1.0))
        projection = start_xy + t * segment
        return float(np.linalg.norm(center_xy - projection))

    def _easy_corridor_clearance_is_valid(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        center_xy: np.ndarray,
        radius: float,
    ) -> bool:
        distance_to_segment = self._distance_to_start_goal_segment_xy(state, goal, center_xy)
        min_distance = radius + self.scenario.warning_distance + self.scenario.easy_min_corridor_warning_gap
        return distance_to_segment >= min_distance

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

    def _goal_radius_for_level(self, level: str) -> float:
        return float(getattr(self.scenario, f'{level}_goal_radius', self.scenario.goal_radius))

    def _load_scenario(self, payload: dict[str, Any], *, use_curriculum_radius: bool = False) -> None:
        self.state = np.asarray(payload['state'], dtype=np.float32).copy()
        self.goal = np.asarray(payload['goal'], dtype=np.float32).copy()
        self.zones = [
            Zone(center_xy=np.asarray(zone['center_xy'], dtype=np.float32), radius=float(zone['radius']))
            for zone in payload['zones']
        ]
        self.last_curriculum_level = str(payload.get('curriculum_level', 'custom'))
        self.active_goal_radius = (
            self._goal_radius_for_level(self.last_curriculum_level)
            if use_curriculum_radius
            else float(self.scenario.goal_radius)
        )

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
        if self._goal_distance(pos) <= self.active_goal_radius or self.last_goal_reached_by_segment:
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
        rew = self.rewards.progress_weight * (prev_distance - new_distance)
        rew += self._breakthrough_reward(new_distance, prev_best_goal_distance, outcome)
        rew -= self.rewards.step_penalty
        rew -= self.rewards.smoothness_weight * float(np.square(action).sum())
        rew -= self._action_change_penalty(prev_action, action)
        rew -= self._zone_warning_penalty(self.state[:3])
        rew -= self._boundary_warning_penalty(self.state[:3])
        rew -= self._ground_warning_penalty(self.state[:3])
        rew -= self._descent_trend_penalty(prev_state, self.state)
        rew -= self._inefficiency_penalty()
        rew += self._terminal_guidance_reward(prev_state, prev_distance, new_distance, outcome)
        rew += self._terminal_los_reward(new_distance, outcome)
        if outcome == 'goal':
            rew += self.rewards.goal_reward
        elif outcome in {'collision', 'ground'}:
            rew -= self.rewards.collision_penalty
        elif outcome == 'boundary':
            rew -= self.rewards.boundary_penalty
        elif outcome == 'timeout':
            rew -= self.rewards.timeout_penalty
        return rew

    def _terminal_guidance_reward(
        self,
        prev_state: np.ndarray,
        prev_distance: float,
        new_distance: float,
        outcome: str,
    ) -> float:
        if outcome in {'collision', 'ground', 'boundary'}:
            return 0.0
        if new_distance > self.rewards.terminal_guidance_radius:
            return 0.0

        reward = self.rewards.terminal_progress_weight * (prev_distance - new_distance)
        prev_z_error = abs(float(prev_state[2] - self.goal[2]))
        new_z_error = abs(float(self.state[2] - self.goal[2]))
        reward += self.rewards.terminal_z_weight * (prev_z_error - new_z_error)
        if new_distance <= self.rewards.terminal_stall_radius and new_distance >= prev_distance:
            reward -= self.rewards.terminal_stall_penalty
        return float(reward)

    def _terminal_los_reward(self, goal_distance: float, outcome: str) -> float:
        if outcome in {'collision', 'ground', 'boundary'}:
            return 0.0
        if goal_distance > self.rewards.terminal_los_radius:
            return 0.0
        if goal_distance <= 1e-6:
            return 0.0

        gamma = float(self.state[3])
        psi = float(self.state[4])
        v_hat = np.array(
            [
                math.cos(gamma) * math.cos(psi),
                math.cos(gamma) * math.sin(psi),
                math.sin(gamma),
            ],
            dtype=np.float32,
        )
        u_goal = (self.goal - self.state[:3]) / goal_distance
        alignment = float(np.dot(v_hat, u_goal))
        if alignment >= 0.0:
            return self.rewards.terminal_los_weight * alignment
        return -self.rewards.terminal_los_penalty_weight * abs(alignment)

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
        reward = self.rewards.breakthrough_reward_weight * window_progress
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

    def _goal_distance(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - self.goal))

    def _segment_goal_distance(self, prev_pos: np.ndarray, curr_pos: np.ndarray) -> float:
        prev = np.asarray(prev_pos, dtype=np.float32)
        curr = np.asarray(curr_pos, dtype=np.float32)
        segment = curr - prev
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq <= 1e-9:
            return self._goal_distance(curr)
        t = float(np.dot(self.goal - prev, segment) / segment_len_sq)
        t = float(np.clip(t, 0.0, 1.0))
        closest = prev + t * segment
        return self._goal_distance(closest)

    def _segment_reaches_goal(self, prev_pos: np.ndarray, curr_pos: np.ndarray) -> bool:
        return self._segment_goal_distance(prev_pos, curr_pos) <= self.active_goal_radius

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
            'goal_radius': float(self.scenario.goal_radius),
            'active_goal_radius': float(self.active_goal_radius),
            'progress': progress,
            'outcome': outcome,
            'steps': self.steps,
            'curriculum_level': self.last_curriculum_level,
        }
