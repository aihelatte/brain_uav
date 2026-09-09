"""Minimal CPU-oriented interaction loop for the standalone V2 TD3 stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np

from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.observations import V2Observation

from .v2_td3 import V2TD3UpdateEngine, V2TD3UpdateMetrics


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a non-negative integer.')
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a non-negative integer.') from exc
    if result < 0 or result != value:
        raise ValueError(f'{name} must be a non-negative integer.')
    return result


def _nonnegative_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and non-negative.') from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and non-negative.')
    return result


@dataclass(slots=True)
class V2TrainingLoopMetrics:
    """Small structured metric set for V2 closure smoke tests."""

    total_steps: int = 0
    update_count: int = 0
    episodes: int = 0
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    outcomes: dict[str, int] = field(default_factory=dict)
    replay_success_fraction: float = 0.0
    success_replay_size: int = 0
    last_update_metrics: V2TD3UpdateMetrics | None = None


@dataclass(frozen=True, slots=True)
class _EpisodeTransition:
    observation: V2Observation
    action: np.ndarray
    reward: float
    next_observation: V2Observation
    done: bool
    near_goal: bool
    line_to_goal_safe: bool


class V2TD3TrainingLoop:
    """Minimal V2 environment/replay/update loop, not a production trainer.

    This class intentionally omits curriculum, early stopping, checkpoint files,
    noise schedules, BC schedules, benchmark logic, and full RNG/replay state.
    """

    def __init__(
        self,
        env: V2StaticNoFlyTrajectoryEnv,
        engine: V2TD3UpdateEngine,
        *,
        warmup_steps: int,
        exploration_noise: float = 0.0,
        near_goal_radius: float = 250.0,
        terminal_geo_safe_clearance: float = 40.0,
        seed: int | None = None,
    ) -> None:
        if not isinstance(env, V2StaticNoFlyTrajectoryEnv):
            raise TypeError('env must be a V2StaticNoFlyTrajectoryEnv.')
        if not isinstance(engine, V2TD3UpdateEngine):
            raise TypeError('engine must be a V2TD3UpdateEngine.')
        if engine.action_dim != int(env.action_space.shape[0]):
            raise ValueError('Environment and V2 TD3 engine action dimensions must match.')
        if not np.array_equal(
            engine.action_low.detach().cpu().numpy(),
            env.action_space.low,
        ) or not np.array_equal(
            engine.action_high.detach().cpu().numpy(),
            env.action_space.high,
        ):
            raise ValueError('Environment and V2 TD3 engine action ranges must match.')
        expected_scales = env.observation_scales
        if engine.actor.scales != expected_scales:
            raise ValueError('Environment and V2 Actor observation scales must match.')
        if engine.actor.uav_radius != env.uav_collision_radius:
            raise ValueError('Environment and V2 Actor UAV collision radii must match.')

        self.env = env
        self.engine = engine
        self.warmup_steps = _nonnegative_int(warmup_steps, name='warmup_steps')
        self.exploration_noise = _nonnegative_float(
            exploration_noise,
            name='exploration_noise',
        )
        self.near_goal_radius = _nonnegative_float(
            near_goal_radius,
            name='near_goal_radius',
        )
        self.terminal_geo_safe_clearance = _nonnegative_float(
            terminal_geo_safe_clearance,
            name='terminal_geo_safe_clearance',
        )
        self.rng = np.random.default_rng(seed)
        self.metrics = V2TrainingLoopMetrics()
        self._observation: V2Observation | None = None
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_transitions: list[_EpisodeTransition] = []
        self._episode_slot_refs: list[tuple[int, int]] = []

    def _warmup_action(self) -> np.ndarray:
        return self.rng.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
        ).astype(np.float32)

    def _is_near_goal(self, info: dict[str, Any]) -> bool:
        goal_distance = float(info.get('goal_distance', float('inf')))
        segment_goal_distance = float(
            info.get('segment_goal_distance', float('inf'))
        )
        goal_reached_by_segment = bool(
            info.get('goal_reached_by_segment', False)
        )
        return (
            goal_distance <= self.near_goal_radius
            or segment_goal_distance <= self.near_goal_radius
            or goal_reached_by_segment
        )

    def _finish_episode(self, outcome: str) -> None:
        self.metrics.episodes += 1
        self.metrics.episode_returns.append(float(self._episode_return))
        self.metrics.episode_lengths.append(int(self._episode_length))
        self.metrics.outcomes[outcome] = self.metrics.outcomes.get(outcome, 0) + 1
        if outcome == 'goal':
            for transition in self._episode_transitions:
                self.engine.replay.add_success_transition(
                    transition.observation,
                    transition.action,
                    transition.reward,
                    transition.next_observation,
                    transition.done,
                    near_goal=transition.near_goal,
                    line_to_goal_safe=transition.line_to_goal_safe,
                )
            self.engine.replay.mark_success_slots(
                self._episode_slot_refs,
                success=True,
            )
        self._observation, _ = self.env.reset()
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_transitions = []
        self._episode_slot_refs = []

    def run(
        self,
        total_steps: int,
        *,
        bc_lambda: float = 0.0,
    ) -> V2TrainingLoopMetrics:
        """Execute a bounded number of interaction steps on the current CPU loop."""

        step_count = _nonnegative_int(total_steps, name='total_steps')
        bc_lambda_value = _nonnegative_float(bc_lambda, name='bc_lambda')
        if self._observation is None:
            self._observation, _ = self.env.reset()

        for _ in range(step_count):
            observation = self._observation
            line_to_goal_safe = self.env.line_to_goal_is_safe(
                self.env.state[:3],
                clearance=self.terminal_geo_safe_clearance,
            )
            self.metrics.total_steps += 1
            if self.metrics.total_steps <= self.warmup_steps:
                action = self._warmup_action()
            else:
                action = self.engine.select_action(
                    observation,
                    exploration_noise=self.exploration_noise,
                )
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            done = bool(terminated or truncated)
            near_goal = self._is_near_goal(info)
            slot_ref = self.engine.replay.add(
                observation,
                action,
                reward,
                next_observation,
                done,
                success=False,
                near_goal=near_goal,
                line_to_goal_safe=line_to_goal_safe,
            )
            self._episode_slot_refs.append(slot_ref)
            self._episode_transitions.append(
                _EpisodeTransition(
                    observation=observation,
                    action=np.asarray(action, dtype=np.float32).copy(),
                    reward=float(reward),
                    next_observation=next_observation,
                    done=done,
                    near_goal=near_goal,
                    line_to_goal_safe=line_to_goal_safe,
                )
            )
            self._episode_return += float(reward)
            self._episode_length += 1
            self._observation = next_observation

            if len(self.engine.replay) >= self.engine.batch_size:
                update_metrics = self.engine.update_once(
                    total_steps=self.metrics.total_steps,
                    bc_lambda=bc_lambda_value,
                )
                self.metrics.update_count += 1
                self.metrics.last_update_metrics = update_metrics

            if done:
                self._finish_episode(str(info.get('outcome', 'unknown')))

        self.metrics.replay_success_fraction = (
            self.engine.replay.success_fraction()
        )
        self.metrics.success_replay_size = self.engine.replay.success_size
        return self.metrics
