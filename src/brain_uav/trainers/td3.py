"""TD3 trainer.

����ǿ��ѧϰ����ѭ����
����԰������ɣ�
- Actor ���������
- ���� Critic ������
- �طŻ��渺�𷴸�ѧϰ��ȥ����
"""

from __future__ import annotations

import statistics
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .replay_buffer import ReplayBuffer


@dataclass(slots=True)
class TD3Metrics:
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    rl_actor_loss: float = 0.0
    bc_loss: float = 0.0
    bc_lambda: float = 0.0
    bc_regularization_enabled: bool = False
    near_goal_sample_bias: float = 1.0
    near_goal_sample_radius: float = 0.0
    replay_near_goal_fraction: float = 0.0
    steps: int = 0
    episodes: int = 0
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    outcomes: dict[str, int] = field(default_factory=dict)
    episode_window_stats: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'actor_loss': self.actor_loss,
            'critic_loss': self.critic_loss,
            'rl_actor_loss': self.rl_actor_loss,
            'bc_loss': self.bc_loss,
            'bc_lambda': self.bc_lambda,
            'bc_regularization_enabled': self.bc_regularization_enabled,
            'near_goal_sample_bias': self.near_goal_sample_bias,
            'near_goal_sample_radius': self.near_goal_sample_radius,
            'replay_near_goal_fraction': self.replay_near_goal_fraction,
            'steps': self.steps,
            'episodes': self.episodes,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'outcomes': self.outcomes,
            'episode_window_stats': self.episode_window_stats,
            'avg_return': statistics.mean(self.episode_returns) if self.episode_returns else 0.0,
            'avg_length': statistics.mean(self.episode_lengths) if self.episode_lengths else 0.0,
        }


class TD3Trainer:
    def __init__(
        self,
        env,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_delay: int,
        replay_size: int,
        batch_size: int,
        warmup_steps: int,
        exploration_noise: float,
        success_sample_bias: float,
        near_goal_sample_bias: float = 1.0,
        near_goal_sample_radius: float = 100.0,
        min_exploration_noise: float | None = None,
        exploration_noise_decay_start_fraction: float = 0.5,
        exploration_noise_decay_end_fraction: float = 1.0,
        actor_freeze_steps: int = 0,
        critic_grad_clip_norm: float | None = None,
        warmup_strategy: str = 'random',
        device: str = 'cpu',
        bc_reference_actor: nn.Module | None = None,
        curriculum_level: str | None = None,
        envs: list[Any] | None = None,
    ) -> None:
        self.env = env
        self.envs = envs if envs is not None else [env]
        self.actor = actor.to(device)
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.actor_target = deepcopy(self.actor)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr
        )
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.exploration_noise = exploration_noise
        self.actor_freeze_steps = actor_freeze_steps
        self.critic_grad_clip_norm = critic_grad_clip_norm
        self.success_sample_bias = success_sample_bias
        self.near_goal_sample_bias = float(near_goal_sample_bias)
        self.near_goal_sample_radius = float(near_goal_sample_radius)
        self.min_exploration_noise = min_exploration_noise
        self.exploration_noise_decay_start_fraction = float(exploration_noise_decay_start_fraction)
        self.exploration_noise_decay_end_fraction = float(exploration_noise_decay_end_fraction)
        self.training_horizon = 1
        self.exploration_noise_base = exploration_noise
        self.policy_noise_base = policy_noise
        self.noise_clip_base = noise_clip
        self.exploration_noise_current = exploration_noise
        self.policy_noise_current = policy_noise
        self.noise_clip_current = noise_clip
        self.warmup_strategy = warmup_strategy
        self.device = device
        self.replay = ReplayBuffer(
            replay_size,
            success_sample_bias=success_sample_bias,
            near_goal_sample_bias=near_goal_sample_bias,
        )
        self.total_steps = 0
        self.metrics = TD3Metrics()
        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
        self._current_window: list[dict] = []
        self.stop_reason: str | None = None
        self.curriculum_level: str | None = curriculum_level
        self.bc_reference_actor: nn.Module | None = None
        self.metrics.near_goal_sample_bias = self.near_goal_sample_bias
        self.metrics.near_goal_sample_radius = self.near_goal_sample_radius
        if bc_reference_actor is not None:
            self.set_bc_reference_actor(bc_reference_actor)

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 500,
        verbose: bool = True,
        summary_every_episodes: int = 50,
        episode_callback: Callable[[dict[str, Any]], None] | None = None,
        window_callback: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> TD3Metrics:
        if len(self.envs) > 1:
            return self._train_parallel(
                total_timesteps=total_timesteps,
                log_interval=log_interval,
                verbose=verbose,
                summary_every_episodes=summary_every_episodes,
                episode_callback=episode_callback,
                window_callback=window_callback,
            )

        self.training_horizon = max(int(total_timesteps), 1)
        obs, _ = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        episode_goal_diagnostics = self._new_episode_goal_diagnostics()
        if verbose:
            print(
                f"[TD3] start total_timesteps={total_timesteps} warmup_steps={self.warmup_steps} "
                f"actor_freeze_steps={self.actor_freeze_steps} batch_size={self.batch_size} "
                f"replay_size={self.replay.buffer.maxlen} warmup_strategy={self.warmup_strategy} "
                f"summary_every_episodes={summary_every_episodes} "
                f"near_goal_sample_bias={self.near_goal_sample_bias} "
                f"near_goal_sample_radius={self.near_goal_sample_radius}"
            )
        for step_idx in range(total_timesteps):
            self.total_steps += 1
            if self.bc_reference_actor is not None:
                self.metrics.bc_lambda = float(self._bc_lambda())
            if self.total_steps <= self.warmup_steps:
                action = self._warmup_action(obs)
            else:
                action = self.select_action(obs, with_noise=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            near_goal = float(info.get('goal_distance', float('inf'))) <= self.near_goal_sample_radius
            # Include timeouts as terminal transitions in replay.
            self.replay.add(
                obs,
                action,
                reward,
                next_obs,
                done,
                success=bool(info.get('outcome') == 'goal'),
                near_goal=bool(near_goal),
            )
            self.metrics.replay_near_goal_fraction = self.replay.near_goal_fraction()
            episode_return += reward
            episode_length += 1
            self._update_episode_goal_diagnostics(episode_goal_diagnostics, episode_length)
            obs = next_obs
            if len(self.replay) >= self.batch_size:
                self._update()
            if done:
                self.metrics.episodes += 1
                self.metrics.episode_returns.append(float(episode_return))
                self.metrics.episode_lengths.append(int(episode_length))
                outcome = info.get('outcome', 'unknown')
                self.metrics.outcomes[outcome] = self.metrics.outcomes.get(outcome, 0) + 1
                episode_record = {
                    'episode': self.metrics.episodes,
                    'total_steps': self.total_steps,
                    'return': float(episode_return),
                    'length': int(episode_length),
                    'outcome': outcome,
                    'actor_loss': float(self.metrics.actor_loss),
                    'critic_loss': float(self.metrics.critic_loss),
                    **self._finalize_episode_goal_diagnostics(episode_goal_diagnostics),
                    'scenario': self.env.export_scenario(),
                    'trajectory': [point.tolist() for point in self.env.trajectory],
                    'final_state': self.env.state.copy().tolist(),
                    'info': {
                        'goal_distance': float(info.get('goal_distance', 0.0)),
                        'segment_goal_distance': float(info.get('segment_goal_distance', 0.0)),
                        'goal_reached_by_segment': bool(info.get('goal_reached_by_segment', False)),
                        'goal_radius': float(info.get('goal_radius', self.env.scenario.goal_radius)),
                        'active_goal_radius': float(info.get('active_goal_radius', self.env.scenario.goal_radius)),
                        'progress': float(info.get('progress', 0.0)),
                        'steps': int(info.get('steps', episode_length)),
                        'curriculum_level': info.get('curriculum_level'),
                    },
                }
                self._current_window.append(
                    {
                        'episode': episode_record['episode'],
                        'total_steps': episode_record['total_steps'],
                        'return': episode_record['return'],
                        'length': episode_record['length'],
                        'outcome': episode_record['outcome'],
                        'actor_loss': episode_record['actor_loss'],
                        'critic_loss': episode_record['critic_loss'],
                        'min_goal_distance': episode_record['min_goal_distance'],
                        'min_xy_goal_distance': episode_record['min_xy_goal_distance'],
                        'min_z_goal_error': episode_record['min_z_goal_error'],
                        'final_goal_distance': episode_record['final_goal_distance'],
                        'final_xy_goal_distance': episode_record['final_xy_goal_distance'],
                        'final_z_goal_error': episode_record['final_z_goal_error'],
                        'min_segment_goal_distance': episode_record['min_segment_goal_distance'],
                        'near_goal_step_count': episode_record['near_goal_step_count'],
                        'goal_reached_by_segment_count': episode_record['goal_reached_by_segment_count'],
                        'active_goal_radius': episode_record['active_goal_radius'],
                    }
                )
                if episode_callback is not None:
                    episode_callback(episode_record)
                if summary_every_episodes > 0 and len(self._current_window) >= summary_every_episodes:
                    window_row = self._flush_window_stats()
                    if window_callback is not None and window_row is not None:
                        stop_reason = window_callback(window_row)
                        if stop_reason:
                            self.stop_reason = stop_reason
                            if verbose:
                                print(f"[TD3] early stop triggered: {stop_reason}")
                            obs, _ = self.env.reset()
                            episode_return = 0.0
                            episode_length = 0
                            episode_goal_diagnostics = self._new_episode_goal_diagnostics()
                            break
                if verbose:
                    print(
                        f"[TD3] episode={self.metrics.episodes} step={self.total_steps}/{total_timesteps} "
                        f"return={episode_return:.2f} length={episode_length} outcome={outcome}"
                    )
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_length = 0
                episode_goal_diagnostics = self._new_episode_goal_diagnostics()
            if verbose and ((step_idx + 1) % log_interval == 0 or (step_idx + 1) == total_timesteps):
                avg_return = statistics.mean(self.metrics.episode_returns[-5:]) if self.metrics.episode_returns else 0.0
                actor_phase = 'frozen' if self.total_steps <= self.actor_freeze_steps else 'active'
                print(
                    f"[TD3] progress={step_idx + 1}/{total_timesteps} episodes={self.metrics.episodes} "
                    f"buffer={len(self.replay)} success_frac={self.replay.success_fraction():.3f} "
                    f"near_goal_frac={self.replay.near_goal_fraction():.3f} "
                    f"actor_phase={actor_phase} actor_loss={self.metrics.actor_loss:.4f} "
                    f"critic_loss={self.metrics.critic_loss:.4f} recent_avg_return={avg_return:.2f}"
                )
        if self._current_window:
            window_row = self._flush_window_stats()
            if window_callback is not None and window_row is not None and self.stop_reason is None:
                stop_reason = window_callback(window_row)
                if stop_reason:
                    self.stop_reason = stop_reason
                    if verbose:
                        print(f"[TD3] early stop triggered: {stop_reason}")
        self.metrics.steps = self.total_steps
        self.metrics.replay_near_goal_fraction = self.replay.near_goal_fraction()
        self.actor.to('cpu')
        self.critic1.to('cpu')
        self.critic2.to('cpu')
        return self.metrics

    def _train_parallel(
        self,
        total_timesteps: int,
        log_interval: int = 500,
        verbose: bool = True,
        summary_every_episodes: int = 50,
        episode_callback: Callable[[dict[str, Any]], None] | None = None,
        window_callback: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> TD3Metrics:
        self.training_horizon = max(int(total_timesteps), 1)
        obs_list: list[np.ndarray] = []
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        episode_diagnostics: list[dict[str, Any]] = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
            episode_returns.append(0.0)
            episode_lengths.append(0)
            episode_diagnostics.append(self._new_episode_goal_diagnostics(env))

        if verbose:
            print(
                f"[TD3] start total_timesteps={total_timesteps} warmup_steps={self.warmup_steps} "
                f"actor_freeze_steps={self.actor_freeze_steps} batch_size={self.batch_size} "
                f"replay_size={self.replay.buffer.maxlen} warmup_strategy={self.warmup_strategy} "
                f"summary_every_episodes={summary_every_episodes} num_envs={len(self.envs)} "
                f"near_goal_sample_bias={self.near_goal_sample_bias} "
                f"near_goal_sample_radius={self.near_goal_sample_radius}"
            )

        stop_training = False
        while self.total_steps < total_timesteps and not stop_training:
            remaining = total_timesteps - self.total_steps
            active_indices = list(range(min(len(self.envs), remaining)))
            if not active_indices:
                break

            if self.total_steps < self.warmup_steps and self.warmup_strategy == 'random':
                actions = [self.envs[idx].action_space.sample() for idx in active_indices]
            else:
                obs_batch = np.stack([obs_list[idx] for idx in active_indices]).astype(np.float32)
                actions_batch = self.select_actions(obs_batch, with_noise=True)
                actions = [actions_batch[item_idx] for item_idx in range(len(active_indices))]

            for env_idx, action in zip(active_indices, actions):
                if self.total_steps >= total_timesteps:
                    break
                env = self.envs[env_idx]
                obs = obs_list[env_idx]
                self.total_steps += 1
                if self.bc_reference_actor is not None:
                    self.metrics.bc_lambda = float(self._bc_lambda())
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                near_goal = float(info.get('goal_distance', float('inf'))) <= self.near_goal_sample_radius
                self.replay.add(
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                    success=bool(info.get('outcome') == 'goal'),
                    near_goal=bool(near_goal),
                )
                self.metrics.replay_near_goal_fraction = self.replay.near_goal_fraction()
                episode_returns[env_idx] += reward
                episode_lengths[env_idx] += 1
                self._update_episode_goal_diagnostics(
                    episode_diagnostics[env_idx],
                    episode_lengths[env_idx],
                    env,
                )
                obs_list[env_idx] = next_obs
                if len(self.replay) >= self.batch_size:
                    self._update()
                if done:
                    episode_record = self._record_episode(
                        env=env,
                        info=info,
                        episode_return=episode_returns[env_idx],
                        episode_length=episode_lengths[env_idx],
                        diagnostics=episode_diagnostics[env_idx],
                        env_id=env_idx,
                    )
                    if episode_callback is not None:
                        episode_callback(episode_record)
                    if summary_every_episodes > 0 and len(self._current_window) >= summary_every_episodes:
                        window_row = self._flush_window_stats()
                        if window_callback is not None and window_row is not None:
                            stop_reason = window_callback(window_row)
                            if stop_reason:
                                self.stop_reason = stop_reason
                                stop_training = True
                                if verbose:
                                    print(f"[TD3] early stop triggered: {stop_reason}")
                    if verbose:
                        print(
                            f"[TD3] episode={self.metrics.episodes} env={env_idx} "
                            f"step={self.total_steps}/{total_timesteps} "
                            f"return={episode_returns[env_idx]:.2f} length={episode_lengths[env_idx]} "
                            f"outcome={episode_record['outcome']}"
                        )
                    if stop_training:
                        break
                    obs_reset, _ = env.reset()
                    obs_list[env_idx] = obs_reset
                    episode_returns[env_idx] = 0.0
                    episode_lengths[env_idx] = 0
                    episode_diagnostics[env_idx] = self._new_episode_goal_diagnostics(env)
                if verbose and (self.total_steps % log_interval == 0 or self.total_steps == total_timesteps):
                    avg_return = (
                        statistics.mean(self.metrics.episode_returns[-5:]) if self.metrics.episode_returns else 0.0
                    )
                    actor_phase = 'frozen' if self.total_steps <= self.actor_freeze_steps else 'active'
                    print(
                        f"[TD3] progress={self.total_steps}/{total_timesteps} episodes={self.metrics.episodes} "
                        f"buffer={len(self.replay)} success_frac={self.replay.success_fraction():.3f} "
                        f"near_goal_frac={self.replay.near_goal_fraction():.3f} "
                        f"actor_phase={actor_phase} actor_loss={self.metrics.actor_loss:.4f} "
                        f"critic_loss={self.metrics.critic_loss:.4f} recent_avg_return={avg_return:.2f}"
                    )

        if self._current_window:
            window_row = self._flush_window_stats()
            if window_callback is not None and window_row is not None and self.stop_reason is None:
                stop_reason = window_callback(window_row)
                if stop_reason:
                    self.stop_reason = stop_reason
                    if verbose:
                        print(f"[TD3] early stop triggered: {stop_reason}")
        self.metrics.steps = self.total_steps
        self.metrics.replay_near_goal_fraction = self.replay.near_goal_fraction()
        self.actor.to('cpu')
        self.critic1.to('cpu')
        self.critic2.to('cpu')
        return self.metrics

    def set_bc_reference_actor(self, actor: nn.Module) -> None:
        self.bc_reference_actor = actor.to(self.device)
        self.bc_reference_actor.eval()
        for param in self.bc_reference_actor.parameters():
            param.requires_grad = False
        self.metrics.bc_regularization_enabled = True

    def _record_episode(
        self,
        env,
        info: dict[str, Any],
        episode_return: float,
        episode_length: int,
        diagnostics: dict[str, Any],
        env_id: int = 0,
    ) -> dict[str, Any]:
        self.metrics.episodes += 1
        self.metrics.episode_returns.append(float(episode_return))
        self.metrics.episode_lengths.append(int(episode_length))
        outcome = info.get('outcome', 'unknown')
        self.metrics.outcomes[outcome] = self.metrics.outcomes.get(outcome, 0) + 1
        episode_record = {
            'episode': self.metrics.episodes,
            'env_id': int(env_id),
            'total_steps': self.total_steps,
            'return': float(episode_return),
            'length': int(episode_length),
            'outcome': outcome,
            'actor_loss': float(self.metrics.actor_loss),
            'critic_loss': float(self.metrics.critic_loss),
            **self._finalize_episode_goal_diagnostics(diagnostics, env),
            'scenario': env.export_scenario(),
            'trajectory': [point.tolist() for point in env.trajectory],
            'final_state': env.state.copy().tolist(),
            'info': {
                'goal_distance': float(info.get('goal_distance', 0.0)),
                'segment_goal_distance': float(info.get('segment_goal_distance', 0.0)),
                'goal_reached_by_segment': bool(info.get('goal_reached_by_segment', False)),
                'goal_radius': float(info.get('goal_radius', env.scenario.goal_radius)),
                'active_goal_radius': float(info.get('active_goal_radius', env.scenario.goal_radius)),
                'progress': float(info.get('progress', 0.0)),
                'steps': int(info.get('steps', episode_length)),
                'curriculum_level': info.get('curriculum_level'),
            },
        }
        self._current_window.append(
            {
                'episode': episode_record['episode'],
                'total_steps': episode_record['total_steps'],
                'return': episode_record['return'],
                'length': episode_record['length'],
                'outcome': episode_record['outcome'],
                'actor_loss': episode_record['actor_loss'],
                'critic_loss': episode_record['critic_loss'],
                'min_goal_distance': episode_record['min_goal_distance'],
                'min_xy_goal_distance': episode_record['min_xy_goal_distance'],
                'min_z_goal_error': episode_record['min_z_goal_error'],
                'final_goal_distance': episode_record['final_goal_distance'],
                'final_xy_goal_distance': episode_record['final_xy_goal_distance'],
                'final_z_goal_error': episode_record['final_z_goal_error'],
                'min_segment_goal_distance': episode_record['min_segment_goal_distance'],
                'near_goal_step_count': episode_record['near_goal_step_count'],
                'goal_reached_by_segment_count': episode_record['goal_reached_by_segment_count'],
                'active_goal_radius': episode_record['active_goal_radius'],
            }
        )
        return episode_record

    def _goal_error_components(self, env=None) -> dict[str, float | list[float]]:
        env = env or self.env
        pos = np.asarray(env.state[:3], dtype=float)
        goal = np.asarray(env.goal, dtype=float)
        delta = pos - goal
        return {
            'goal_distance': float(np.linalg.norm(delta)),
            'xy_goal_distance': float(np.linalg.norm(delta[:2])),
            'z_goal_error': float(abs(delta[2])),
            'position': pos.astype(float).tolist(),
            'segment_goal_distance': float(getattr(env, 'last_segment_goal_distance', np.linalg.norm(delta))),
            'goal_reached_by_segment': bool(getattr(env, 'last_goal_reached_by_segment', False)),
            'active_goal_radius': float(getattr(env, 'active_goal_radius', env.scenario.goal_radius)),
        }

    def _new_episode_goal_diagnostics(self, env=None) -> dict[str, Any]:
        components = self._goal_error_components(env)
        return {
            'min_goal_distance': components['goal_distance'],
            'min_xy_goal_distance': components['xy_goal_distance'],
            'min_z_goal_error': components['z_goal_error'],
            'min_segment_goal_distance': components['segment_goal_distance'],
            'step_at_min_goal_distance': 0,
            'near_goal_step_count': 0,
            'goal_reached_by_segment_count': 0,
            'active_goal_radius': components['active_goal_radius'],
        }

    def _update_episode_goal_diagnostics(self, diagnostics: dict[str, Any], step: int, env=None) -> None:
        components = self._goal_error_components(env)
        if float(components['goal_distance']) <= 50.0:
            diagnostics['near_goal_step_count'] += 1
        if bool(components['goal_reached_by_segment']):
            diagnostics['goal_reached_by_segment_count'] += 1
        if float(components['goal_distance']) < float(diagnostics['min_goal_distance']):
            diagnostics['min_goal_distance'] = components['goal_distance']
            diagnostics['min_xy_goal_distance'] = components['xy_goal_distance']
            diagnostics['min_z_goal_error'] = components['z_goal_error']
            diagnostics['step_at_min_goal_distance'] = int(step)
        if float(components['segment_goal_distance']) < float(diagnostics['min_segment_goal_distance']):
            diagnostics['min_segment_goal_distance'] = components['segment_goal_distance']
        diagnostics['active_goal_radius'] = components['active_goal_radius']

    def _finalize_episode_goal_diagnostics(self, diagnostics: dict[str, Any], env=None) -> dict[str, float | int]:
        components = self._goal_error_components(env)
        return {
            'min_goal_distance': float(diagnostics['min_goal_distance']),
            'min_xy_goal_distance': float(diagnostics['min_xy_goal_distance']),
            'min_z_goal_error': float(diagnostics['min_z_goal_error']),
            'final_goal_distance': float(components['goal_distance']),
            'final_xy_goal_distance': float(components['xy_goal_distance']),
            'final_z_goal_error': float(components['z_goal_error']),
            'min_segment_goal_distance': float(diagnostics['min_segment_goal_distance']),
            'step_at_min_goal_distance': int(diagnostics['step_at_min_goal_distance']),
            'near_goal_step_count': int(diagnostics['near_goal_step_count']),
            'goal_reached_by_segment_count': int(diagnostics['goal_reached_by_segment_count']),
            'active_goal_radius': float(diagnostics['active_goal_radius']),
        }

    def _bc_lambda(self) -> float:
        if self.curriculum_level == 'hard':
            if self.total_steps <= 50_000:
                return 600.0
            if self.total_steps <= 100_000:
                return 150.0
            return 20.0
        if self.total_steps <= 300_000:
            return 1000.0
        if self.total_steps <= 500_000:
            return 300.0
        return 50.0

    def _current_noise(self) -> tuple[float, float, float]:
        if self.curriculum_level != 'hard':
            self.exploration_noise_current = self._scheduled_exploration_noise()
            self.policy_noise_current = self.policy_noise_base
            self.noise_clip_current = self.noise_clip_base
            return self.exploration_noise_current, self.policy_noise_current, self.noise_clip_current
        if self.total_steps <= 100_000:
            factor = 0.0
        elif self.total_steps <= 200_000:
            factor = (self.total_steps - 100_000) / 100_000
        else:
            factor = 1.0
        self.exploration_noise_current = self.exploration_noise_base + (0.01 - self.exploration_noise_base) * factor
        self.policy_noise_current = self.policy_noise_base + (0.008 - self.policy_noise_base) * factor
        self.noise_clip_current = self.noise_clip_base + (0.015 - self.noise_clip_base) * factor
        return self.exploration_noise_current, self.policy_noise_current, self.noise_clip_current

    def _scheduled_exploration_noise(self) -> float:
        if self.min_exploration_noise is None:
            return self.exploration_noise_base
        min_noise = float(self.min_exploration_noise)
        start_step = self.exploration_noise_decay_start_fraction * float(self.training_horizon)
        end_step = self.exploration_noise_decay_end_fraction * float(self.training_horizon)
        if self.total_steps <= start_step:
            return self.exploration_noise_base
        if end_step <= start_step:
            factor = 1.0
        else:
            factor = (float(self.total_steps) - start_step) / (end_step - start_step)
            factor = float(np.clip(factor, 0.0, 1.0))
        return self.exploration_noise_base + (min_noise - self.exploration_noise_base) * factor

    def select_action(self, obs: np.ndarray, with_noise: bool = False) -> np.ndarray:
        obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        if with_noise:
            exploration_noise, _, _ = self._current_noise()
            action = action + np.random.normal(0.0, exploration_noise, size=action.shape)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high).astype(np.float32)

    def select_actions(self, obs_batch: np.ndarray, with_noise: bool = False) -> np.ndarray:
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.actor(obs_tensor).cpu().numpy()
        if with_noise:
            exploration_noise, _, _ = self._current_noise()
            actions = actions + np.random.normal(0.0, exploration_noise, size=actions.shape)
        return np.clip(actions, self.env.action_space.low, self.env.action_space.high).astype(np.float32)

    def _warmup_action(self, obs: np.ndarray) -> np.ndarray:
        if self.warmup_strategy == 'policy':
            return self.select_action(obs, with_noise=True)
        return self.env.action_space.sample()

    def _flush_window_stats(self) -> dict[str, Any] | None:
        window = self._current_window
        if not window:
            return None
        outcome_counts: dict[str, int] = {}
        for item in window:
            outcome_counts[item['outcome']] = outcome_counts.get(item['outcome'], 0) + 1
        row = {
            'episode_start': window[0]['episode'],
            'episode_end': window[-1]['episode'],
            'episode_count': len(window),
            'total_steps': window[-1]['total_steps'],
            'avg_return': round(statistics.mean(item['return'] for item in window), 6),
            'avg_length': round(statistics.mean(item['length'] for item in window), 6),
            'avg_actor_loss': round(statistics.mean(item['actor_loss'] for item in window), 6),
            'avg_critic_loss': round(statistics.mean(item['critic_loss'] for item in window), 6),
            'avg_min_goal_distance': round(statistics.mean(item['min_goal_distance'] for item in window), 6),
            'avg_min_xy_goal_distance': round(statistics.mean(item['min_xy_goal_distance'] for item in window), 6),
            'avg_min_z_goal_error': round(statistics.mean(item['min_z_goal_error'] for item in window), 6),
            'avg_final_goal_distance': round(statistics.mean(item['final_goal_distance'] for item in window), 6),
            'avg_final_xy_goal_distance': round(statistics.mean(item['final_xy_goal_distance'] for item in window), 6),
            'avg_final_z_goal_error': round(statistics.mean(item['final_z_goal_error'] for item in window), 6),
            'avg_min_segment_goal_distance': round(statistics.mean(item['min_segment_goal_distance'] for item in window), 6),
            'avg_near_goal_step_count': round(statistics.mean(item['near_goal_step_count'] for item in window), 6),
            'goal_reached_by_segment_count': sum(item['goal_reached_by_segment_count'] for item in window),
            'avg_active_goal_radius': round(statistics.mean(item['active_goal_radius'] for item in window), 6),
            'replay_near_goal_fraction': round(self.replay.near_goal_fraction(), 6),
            'goal_count': outcome_counts.get('goal', 0),
            'timeout_count': outcome_counts.get('timeout', 0),
            'boundary_count': outcome_counts.get('boundary', 0),
            'ground_count': outcome_counts.get('ground', 0),
            'collision_count': outcome_counts.get('collision', 0),
            'other_count': sum(
                v for k, v in outcome_counts.items() if k not in {'goal', 'timeout', 'boundary', 'ground', 'collision'}
            ),
        }
        self.metrics.episode_window_stats.append(row)
        self._current_window = []
        return row

    def _update(self) -> None:
        batch = self.replay.sample(self.batch_size)
        obs = batch['obs'].to(self.device)
        actions = batch['action'].to(self.device)
        rewards = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)

        with torch.no_grad():
            _, policy_noise, noise_clip = self._current_noise()
            noise = (torch.randn_like(actions) * policy_noise).clamp(-noise_clip, noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(self.action_low, self.action_high)
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = rewards + (1.0 - done) * self.gamma * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.critic_grad_clip_norm is not None and self.critic_grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                list(self.critic1.parameters()) + list(self.critic2.parameters()),
                max_norm=self.critic_grad_clip_norm,
            )
        self.critic_optimizer.step()
        self.metrics.critic_loss = float(critic_loss.item())

        if self.total_steps % self.policy_delay == 0:
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            if self.total_steps > self.actor_freeze_steps:
                actor_actions = self.actor(obs)
                rl_actor_loss = -self.critic1(obs, actor_actions).mean()
                bc_lambda = 0.0
                bc_loss = torch.zeros((), device=self.device)
                if self.bc_reference_actor is not None:
                    bc_lambda = self._bc_lambda()
                    with torch.no_grad():
                        bc_actions = self.bc_reference_actor(obs)
                    bc_loss = F.mse_loss(actor_actions, bc_actions)
                actor_loss = rl_actor_loss + bc_lambda * bc_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self._soft_update(self.actor, self.actor_target)
                self.metrics.actor_loss = float(actor_loss.item())
                self.metrics.rl_actor_loss = float(rl_actor_loss.item())
                self.metrics.bc_loss = float(bc_loss.item())
                self.metrics.bc_lambda = float(bc_lambda)

    def _soft_update(self, model: nn.Module, target: nn.Module) -> None:
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
