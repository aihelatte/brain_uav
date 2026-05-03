"""Utilities for target-switch evaluation and curriculum training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


TARGET_SWITCH_LEVELS = ('target_switch_easy', 'target_switch_medium', 'target_switch_hard')


@dataclass(slots=True)
class TargetSwitchConfig:
    level: str
    base_curriculum_level: str = 'hard'
    switch_step_ratio_range: tuple[float, float] = (0.25, 0.40)
    switch_angle_deg_range: tuple[float, float] = (0.0, 15.0)
    new_goal_distance_ratio_range: tuple[float, float] = (0.55, 0.75)
    new_goal_z_ratio_range: tuple[float, float] = (0.16, 0.28)
    max_height_gap_ratio: float = 0.10
    lateral_offset_ratio: float = 0.06
    target_switch_progress_scale: float = 1.25
    switch_alignment_window: int = 180
    switch_alignment_reward_weight: float = 6.0
    ceiling_warning_margin: float = 80.0
    ceiling_warning_penalty_weight: float = 120.0
    ceiling_warning_penalty_cap: float = 220.0
    upward_near_ceiling_penalty_weight: float = 160.0
    upward_near_ceiling_penalty_cap: float = 260.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def target_switch_config_for_level(level: str) -> TargetSwitchConfig:
    if level == 'target_switch_easy':
        return TargetSwitchConfig(level='target_switch_easy')
    if level == 'target_switch_medium':
        return TargetSwitchConfig(
            level='target_switch_medium',
            switch_step_ratio_range=(0.25, 0.45),
            switch_angle_deg_range=(0.0, 35.0),
            new_goal_distance_ratio_range=(0.65, 0.90),
            new_goal_z_ratio_range=(0.14, 0.30),
            max_height_gap_ratio=0.14,
            lateral_offset_ratio=0.10,
            target_switch_progress_scale=1.35,
            switch_alignment_window=220,
            switch_alignment_reward_weight=7.5,
            ceiling_warning_margin=90.0,
            ceiling_warning_penalty_weight=150.0,
            ceiling_warning_penalty_cap=260.0,
            upward_near_ceiling_penalty_weight=200.0,
            upward_near_ceiling_penalty_cap=320.0,
        )
    if level == 'target_switch_hard':
        return TargetSwitchConfig(
            level='target_switch_hard',
            switch_step_ratio_range=(0.20, 0.50),
            switch_angle_deg_range=(0.0, 55.0),
            new_goal_distance_ratio_range=(0.75, 1.10),
            new_goal_z_ratio_range=(0.12, 0.32),
            max_height_gap_ratio=0.18,
            lateral_offset_ratio=0.14,
            target_switch_progress_scale=1.45,
            switch_alignment_window=260,
            switch_alignment_reward_weight=9.0,
            ceiling_warning_margin=100.0,
            ceiling_warning_penalty_weight=180.0,
            ceiling_warning_penalty_cap=320.0,
            upward_near_ceiling_penalty_weight=240.0,
            upward_near_ceiling_penalty_cap=380.0,
        )
    raise ValueError(f'Unsupported target switch level: {level}')


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _max_ray_distance_to_world(xy: np.ndarray, direction_xy: np.ndarray, world_xy: float) -> float:
    values: list[float] = []
    for axis in range(2):
        direction = float(direction_xy[axis])
        if abs(direction) <= 1e-9:
            continue
        limit = world_xy if direction > 0.0 else -world_xy
        distance = (limit - float(xy[axis])) / direction
        if distance > 0.0:
            values.append(float(distance))
    return min(values) if values else float('inf')


def _angle_offset(rng: np.random.Generator, cfg: TargetSwitchConfig, switch_mode: str) -> float:
    if switch_mode == 'forward':
        sign = float(rng.choice([-1.0, 1.0]))
        return math.radians(sign * rng.uniform(*cfg.switch_angle_deg_range))
    if switch_mode == 'lateral':
        sign = float(rng.choice([-1.0, 1.0]))
        return math.radians(sign * rng.uniform(60.0, 100.0))
    if switch_mode == 'reverse':
        sign = float(rng.choice([-1.0, 1.0]))
        return math.radians(sign * rng.uniform(120.0, 180.0))
    raise ValueError(f'Unsupported switch_mode: {switch_mode}')


def _fallback_offsets(rng: np.random.Generator, cfg: TargetSwitchConfig) -> list[float]:
    max_angle = max(float(cfg.switch_angle_deg_range[1]), 45.0)
    base = [0.0, -max_angle, max_angle, -75.0, 75.0]
    jitter = float(rng.uniform(-8.0, 8.0))
    return [math.radians(angle + jitter) for angle in base]


def _sample_goal_z(env, rng: np.random.Generator, cfg: TargetSwitchConfig) -> float:
    scenario = env.scenario
    current_z = float(env.state[2])
    max_gap = cfg.max_height_gap_ratio * float(scenario.world_z_max)
    z_min = float(scenario.world_z_min) + 1e-3
    z_max = float(scenario.world_z_max) - 1e-3
    preferred_low = cfg.new_goal_z_ratio_range[0] * float(scenario.world_z_max)
    preferred_high = cfg.new_goal_z_ratio_range[1] * float(scenario.world_z_max)
    low = max(z_min, preferred_low, current_z - max_gap)
    high = min(z_max, preferred_high, current_z + max_gap)
    if low <= high:
        return float(rng.uniform(low, high))
    low = max(z_min, current_z - max_gap)
    high = min(z_max, current_z + max_gap)
    if current_z > 0.65 * float(scenario.world_z_max):
        high = min(high, current_z)
    if low <= high:
        return float(rng.uniform(low, high))
    return float(np.clip(current_z, z_min, z_max))


def _valid_goal(env, goal: np.ndarray) -> bool:
    scenario = env.scenario
    if abs(float(goal[0])) > float(scenario.world_xy) or abs(float(goal[1])) > float(scenario.world_xy):
        return False
    if not (float(scenario.world_z_min) < float(goal[2]) < float(scenario.world_z_max)):
        return False
    for zone in env.zones:
        if env._inside_zone_with_clearance(goal, zone, scenario.start_zone_clearance):
            return False
    return True


def sample_valid_new_goal(
    env,
    rng: np.random.Generator,
    cfg: TargetSwitchConfig | None = None,
    *,
    switch_mode: str = 'forward',
    max_attempts: int = 240,
) -> np.ndarray:
    cfg = cfg or target_switch_config_for_level('target_switch_easy')
    scenario = env.scenario
    current = np.asarray(env.state[:3], dtype=np.float32)
    psi = float(env.state[4])
    distance_min = cfg.new_goal_distance_ratio_range[0] * float(scenario.target_distance)
    distance_max = cfg.new_goal_distance_ratio_range[1] * float(scenario.target_distance)
    min_fallback = max(0.08 * float(scenario.target_distance), 2.0 * float(scenario.goal_radius))

    for attempt in range(max_attempts):
        if attempt < max_attempts * 0.65:
            offset = _angle_offset(rng, cfg, switch_mode)
        else:
            offset = _fallback_offsets(rng, cfg)[attempt % 5]
            center_vec = -current[:2]
            if float(np.linalg.norm(center_vec)) > 1e-6 and attempt > max_attempts * 0.82:
                offset = _wrap_angle(math.atan2(float(center_vec[1]), float(center_vec[0])) - psi)
        direction_xy = np.array([math.cos(psi + offset), math.sin(psi + offset)], dtype=np.float32)
        ray_limit = _max_ray_distance_to_world(current[:2], direction_xy, float(scenario.world_xy)) * 0.92
        high = min(distance_max, ray_limit)
        low = min(distance_min, high)
        if high < min_fallback:
            continue
        if high < distance_min:
            low = min_fallback
        if low > high:
            low = max(min_fallback, 0.5 * high)
        distance = float(rng.uniform(low, high))
        z = _sample_goal_z(env, rng, cfg)
        goal = np.array(
            [
                current[0] + distance * direction_xy[0],
                current[1] + distance * direction_xy[1],
                z,
            ],
            dtype=np.float32,
        )
        if _valid_goal(env, goal):
            return goal

    raise RuntimeError(f'Failed to sample valid target-switch goal for {cfg.level} mode={switch_mode}')


def summarize_switch_episode(
    *,
    trajectory: list[np.ndarray],
    old_goal: np.ndarray | None,
    new_goal: np.ndarray | None,
    switch_position: np.ndarray | None,
    switch_index: int | None,
    post_switch_steps: int,
    pre_switch_done: bool,
) -> dict[str, Any]:
    if new_goal is None or switch_position is None:
        return {
            'switch_to_new_distance': None,
            'final_to_new_distance': None,
            'min_to_new_distance': None,
            'distance_reduction': None,
            'post_switch_steps': int(post_switch_steps),
            'pre_switch_done': bool(pre_switch_done),
            'old_goal': old_goal.tolist() if old_goal is not None else None,
            'new_goal': None,
            'switch_position': None,
        }
    traj = np.asarray(trajectory, dtype=np.float32)
    start_idx = int(switch_index or 0)
    post_traj = traj[start_idx:] if len(traj) > start_idx else traj[-1:]
    distances = np.linalg.norm(post_traj - np.asarray(new_goal, dtype=np.float32), axis=1)
    switch_to_new = float(np.linalg.norm(np.asarray(switch_position, dtype=np.float32) - new_goal))
    final_to_new = float(np.linalg.norm(traj[-1] - new_goal))
    return {
        'switch_to_new_distance': switch_to_new,
        'final_to_new_distance': final_to_new,
        'min_to_new_distance': float(np.min(distances)),
        'distance_reduction': switch_to_new - final_to_new,
        'post_switch_steps': int(post_switch_steps),
        'pre_switch_done': bool(pre_switch_done),
        'old_goal': old_goal.tolist() if old_goal is not None else None,
        'new_goal': np.asarray(new_goal, dtype=np.float32).tolist(),
        'switch_position': np.asarray(switch_position, dtype=np.float32).tolist(),
    }
