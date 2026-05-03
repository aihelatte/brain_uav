"""Target-switch wrapper environment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..config import RewardConfig, ScenarioConfig
from ..target_switch import TargetSwitchConfig, sample_valid_new_goal, summarize_switch_episode
from .static_no_fly_env_runtime import StaticNoFlyTrajectoryEnv


class TargetSwitchTrajectoryEnv(StaticNoFlyTrajectoryEnv):
    """Static no-fly env with one mid-flight active-goal switch."""

    def __init__(
        self,
        scenario: ScenarioConfig | None = None,
        rewards: RewardConfig | None = None,
        target_switch: TargetSwitchConfig | None = None,
        seed: int | None = None,
        fixed_scenarios: list[dict[str, Any]] | None = None,
        switch_mode: str = 'forward',
    ) -> None:
        self.target_switch = target_switch or TargetSwitchConfig(level='target_switch_easy')
        self.switch_mode = switch_mode
        super().__init__(
            scenario=scenario,
            rewards=rewards,
            seed=seed,
            fixed_scenarios=fixed_scenarios,
            curriculum_mix={self.target_switch.base_curriculum_level: 1.0},
        )
        self.old_goal: np.ndarray | None = None
        self.new_goal: np.ndarray | None = None
        self.switch_position: np.ndarray | None = None
        self.switch_step = 1
        self.switch_index: int | None = None
        self.switched = False
        self.pre_switch_done = False
        self.last_episode_summary: dict[str, Any] | None = None
        self._switch_alignment_rewards: list[float] = []
        self._ceiling_penalties: list[float] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.old_goal = self.goal.copy()
        self.new_goal = None
        self.switch_position = None
        self.switch_index = None
        self.switched = False
        self.pre_switch_done = False
        self.last_episode_summary = None
        self._switch_alignment_rewards = []
        self._ceiling_penalties = []
        lo, hi = self.target_switch.switch_step_ratio_range
        ratio = float(self.rng.uniform(lo, hi))
        self.switch_step = max(1, min(self.scenario.max_steps - 1, int(round(ratio * self.scenario.max_steps))))
        info['switch_step'] = self.switch_step
        info['switch_transition'] = False
        info['post_switch_steps'] = 0
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        switch_transition = False

        if not self.switched and done:
            self.pre_switch_done = True

        if not self.switched and not done and self.steps >= self.switch_step:
            self._switch_to_new_goal()
            obs = self._get_obs()
            info = self._info(progress=0.0, outcome='running')
            switch_transition = True
        elif self.switched and not switch_transition:
            extra_reward, reward_info = self._target_switch_reward(info)
            reward += extra_reward
            info.update(reward_info)

        info['switch_transition'] = switch_transition
        info['switch_step'] = self.switch_step
        info['switched'] = self.switched
        info['post_switch_steps'] = self.steps if self.switched else 0
        info['pre_switch_done'] = self.pre_switch_done
        if self.switch_position is not None:
            info['switch_position'] = self.switch_position.tolist()
        if self.old_goal is not None:
            info['old_goal'] = self.old_goal.tolist()
        if self.new_goal is not None:
            info['new_goal'] = self.new_goal.tolist()

        if terminated or truncated:
            self._finalize_episode_summary(info)
        return obs, float(reward), terminated, truncated, info

    def _switch_to_new_goal(self) -> None:
        self.old_goal = self.old_goal if self.old_goal is not None else self.goal.copy()
        self.switch_position = self.state[:3].copy()
        self.switch_index = len(self.trajectory) - 1
        self.new_goal = sample_valid_new_goal(self, self.rng, self.target_switch, switch_mode=self.switch_mode)
        self.set_goal(self.new_goal, reset_leg_timer=True)
        self.switched = True
        self._switch_alignment_rewards = []
        self._ceiling_penalties = []

    def _target_switch_reward(self, info: dict[str, Any]) -> tuple[float, dict[str, float]]:
        progress = float(info.get('progress', 0.0))
        extra = self.rewards.progress_weight * (self.target_switch.target_switch_progress_scale - 1.0) * (
            progress * self._distance_reward_scale_compensation
        )
        alignment_reward = self._switch_alignment_reward()
        ceiling_penalty = self._ceiling_safety_penalty()
        self._switch_alignment_rewards.append(alignment_reward)
        self._ceiling_penalties.append(ceiling_penalty)
        extra += alignment_reward + ceiling_penalty
        return extra, {
            'switch_alignment_reward': alignment_reward,
            'ceiling_penalty': ceiling_penalty,
            'switch_alignment_reward_mean': float(np.mean(self._switch_alignment_rewards)),
            'ceiling_penalty_mean': float(np.mean(self._ceiling_penalties)),
        }

    def _switch_alignment_reward(self) -> float:
        if not self.switched or self.new_goal is None:
            return 0.0
        if self.steps > self.target_switch.switch_alignment_window:
            return 0.0
        rel = self.new_goal - self.state[:3]
        norm = float(np.linalg.norm(rel))
        if norm <= 1e-6:
            return 0.0
        goal_dir = rel / norm
        flight_dir = self._flight_direction(float(self.state[3]), float(self.state[4]))
        alignment = max(float(np.dot(flight_dir, goal_dir)), 0.0)
        return self.target_switch.switch_alignment_reward_weight * alignment

    def _ceiling_safety_penalty(self) -> float:
        clearance = float(self.scenario.world_z_max - self.state[2])
        if clearance >= self.target_switch.ceiling_warning_margin:
            return 0.0
        ratio = float(
            np.clip(
                (self.target_switch.ceiling_warning_margin - clearance)
                / max(self.target_switch.ceiling_warning_margin, 1e-6),
                0.0,
                1.0,
            )
        )
        penalty = min(
            self.target_switch.ceiling_warning_penalty_weight * ratio**2,
            self.target_switch.ceiling_warning_penalty_cap,
        )
        if float(self.state[3]) > 0.0:
            gamma_ratio = float(np.clip(float(self.state[3]) / max(self.scenario.gamma_max, 1e-6), 0.0, 1.0))
            penalty += min(
                self.target_switch.upward_near_ceiling_penalty_weight * ratio * gamma_ratio,
                self.target_switch.upward_near_ceiling_penalty_cap,
            )
        return -float(penalty)

    def _boundary_reason(self) -> str | None:
        pos = self.state[:3]
        if float(pos[2]) > self.scenario.world_z_max:
            return 'z_high'
        if abs(float(pos[0])) > self.scenario.world_xy:
            return 'x'
        if abs(float(pos[1])) > self.scenario.world_xy:
            return 'y'
        if float(pos[2]) <= self.scenario.world_z_min:
            return 'ground'
        return None

    def _finalize_episode_summary(self, info: dict[str, Any]) -> None:
        summary = summarize_switch_episode(
            trajectory=self.trajectory,
            old_goal=self.old_goal,
            new_goal=self.new_goal,
            switch_position=self.switch_position,
            switch_index=self.switch_index,
            post_switch_steps=self.steps if self.switched else 0,
            pre_switch_done=self.pre_switch_done,
        )
        summary.update(
            {
                'switch_step': self.switch_step,
                'switched': self.switched,
                'boundary_reason': self._boundary_reason() if info.get('outcome') == 'boundary' else None,
                'switch_alignment_reward_mean': float(np.mean(self._switch_alignment_rewards))
                if self._switch_alignment_rewards
                else 0.0,
                'ceiling_penalty_mean': float(np.mean(self._ceiling_penalties)) if self._ceiling_penalties else 0.0,
            }
        )
        self.last_episode_summary = summary
