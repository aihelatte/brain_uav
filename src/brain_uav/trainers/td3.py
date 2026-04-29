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
    scaled_rl_actor_loss: float = 0.0
    actor_rl_scale: float = 1.0
    bc_loss: float = 0.0
    bc_lambda: float = 0.0
    terminal_geo_loss: float = 0.0
    terminal_geo_lambda: float = 0.0
    replay_success_fraction: float = 0.0
    replay_near_goal_fraction: float = 0.0
    success_replay_size: int = 0
    success_replay_fraction: float = 0.0
    sample_success_fraction: float = 0.0
    sample_near_goal_fraction: float = 0.0
    bc_regularization_enabled: bool = False
    reference_source: str | None = None
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
            'scaled_rl_actor_loss': self.scaled_rl_actor_loss,
            'actor_rl_scale': self.actor_rl_scale,
            'bc_loss': self.bc_loss,
            'bc_lambda': self.bc_lambda,
            'terminal_geo_loss': self.terminal_geo_loss,
            'terminal_geo_lambda': self.terminal_geo_lambda,
            'replay_success_fraction': self.replay_success_fraction,
            'replay_near_goal_fraction': self.replay_near_goal_fraction,
            'success_replay_size': self.success_replay_size,
            'success_replay_fraction': self.success_replay_fraction,
            'sample_success_fraction': self.sample_success_fraction,
            'sample_near_goal_fraction': self.sample_near_goal_fraction,
            'bc_regularization_enabled': self.bc_regularization_enabled,
            'reference_source': self.reference_source,
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
        actor_freeze_steps: int = 0,
        actor_grad_clip_norm: float | None = None,
        actor_rl_scale_alpha: float = 2.5,
        terminal_geo_regularization_enabled: bool = True,
        terminal_geo_radius: float = 250.0,
        terminal_geo_lambda: float = 3000.0,
        terminal_geo_safe_clearance: float = 40.0,
        near_goal_radius: float = 250.0,
        success_replay_fraction: float = 0.25,
        success_batch_fraction: float = 0.25,
        noise_decay_fraction: float = 0.5,
        exploration_noise_final: float = 0.005,
        policy_noise_final: float = 0.006,
        noise_clip_final: float = 0.012,
        critic_grad_clip_norm: float | None = None,
        warmup_strategy: str = 'random',
        device: str = 'cpu',
        bc_reference_actor: nn.Module | None = None,
        curriculum_level: str | None = None,
    ) -> None:
        self.env = env
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
        self.actor_grad_clip_norm = actor_grad_clip_norm
        self.actor_rl_scale_alpha = actor_rl_scale_alpha
        self.terminal_geo_regularization_enabled = terminal_geo_regularization_enabled
        self.terminal_geo_radius = terminal_geo_radius
        self.terminal_geo_lambda = terminal_geo_lambda
        self.terminal_geo_safe_clearance = terminal_geo_safe_clearance
        self.near_goal_radius = near_goal_radius
        self.success_replay_fraction = success_replay_fraction
        self.success_batch_fraction = success_batch_fraction
        self.noise_decay_fraction = noise_decay_fraction
        self.exploration_noise_final = exploration_noise_final
        self.policy_noise_final = policy_noise_final
        self.noise_clip_final = noise_clip_final
        self.critic_grad_clip_norm = critic_grad_clip_norm
        self.success_sample_bias = success_sample_bias
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
            success_replay_fraction=success_replay_fraction,
            success_batch_fraction=success_batch_fraction,
        )
        self.total_steps = 0
        self.metrics = TD3Metrics()
        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
        self._current_window: list[dict] = []
        self.stop_reason: str | None = None
        self.early_stopped = False
        self.curriculum_level: str | None = curriculum_level
        self.current_stage_timesteps = 1
        self.bc_reference_actor: nn.Module | None = None
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
        log_prefix: str = '[TD3]',
    ) -> TD3Metrics:
        self.current_stage_timesteps = max(int(total_timesteps), 1)
        obs, _ = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        episode_transitions: list[dict[str, Any]] = []
        if verbose:
            print(
                f"{log_prefix} start total_timesteps={total_timesteps} warmup_steps={self.warmup_steps} "
                f"actor_freeze_steps={self.actor_freeze_steps} batch_size={self.batch_size} "
                f"replay_size={self.replay.capacity} warmup_strategy={self.warmup_strategy} "
                f"summary_every_episodes={summary_every_episodes}"
            )
        for step_idx in range(total_timesteps):
            self.total_steps += 1
            line_to_goal_safe = bool(
                getattr(self.env, '_line_to_goal_is_safe', lambda pos, clearance=0.0: True)(
                    self.env.state[:3],
                    clearance=self.terminal_geo_safe_clearance,
                )
            )
            if self.total_steps <= self.warmup_steps:
                action = self._warmup_action(obs)
            else:
                action = self.select_action(obs, with_noise=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            near_goal = self._is_near_goal(info)
            # Include timeouts as terminal transitions in replay.
            self.replay.add(
                obs,
                action,
                reward,
                next_obs,
                done,
                success=bool(info.get('outcome') == 'goal'),
                near_goal=near_goal,
                line_to_goal_safe=line_to_goal_safe,
            )
            episode_transitions.append(
                {
                    'obs': obs.copy(),
                    'action': action.copy(),
                    'reward': float(reward),
                    'next_obs': next_obs.copy(),
                    'done': bool(done),
                    'near_goal': near_goal,
                    'line_to_goal_safe': line_to_goal_safe,
                }
            )
            episode_return += reward
            episode_length += 1
            obs = next_obs
            if len(self.replay) >= self.batch_size:
                self._update()
            if done:
                self.metrics.episodes += 1
                self.metrics.episode_returns.append(float(episode_return))
                self.metrics.episode_lengths.append(int(episode_length))
                outcome = info.get('outcome', 'unknown')
                self.metrics.outcomes[outcome] = self.metrics.outcomes.get(outcome, 0) + 1
                if outcome == 'goal':
                    for transition in episode_transitions:
                        self.replay.add_success_transition(**transition)
                episode_record = {
                    'episode': self.metrics.episodes,
                    'total_steps': self.total_steps,
                    'return': float(episode_return),
                    'length': int(episode_length),
                    'outcome': outcome,
                    'actor_loss': float(self.metrics.actor_loss),
                    'critic_loss': float(self.metrics.critic_loss),
                    'scenario': self.env.export_scenario(),
                    'trajectory': [point.tolist() for point in self.env.trajectory],
                    'final_state': self.env.state.copy().tolist(),
                    'info': {
                        'goal_distance': float(info.get('goal_distance', 0.0)),
                        'segment_goal_distance': float(info.get('segment_goal_distance', float('inf'))),
                        'goal_reached_by_segment': bool(info.get('goal_reached_by_segment', False)),
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
                            self.early_stopped = True
                            if verbose:
                                print(f"{log_prefix} early stop triggered: {stop_reason}")
                            obs, _ = self.env.reset()
                            episode_return = 0.0
                            episode_length = 0
                            break
                if verbose:
                    print(
                        f"{log_prefix} episode={self.metrics.episodes} step={self.total_steps}/{total_timesteps} "
                        f"return={episode_return:.2f} length={episode_length} outcome={outcome}"
                    )
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_length = 0
                episode_transitions = []
            if verbose and ((step_idx + 1) % log_interval == 0 or (step_idx + 1) == total_timesteps):
                avg_return = statistics.mean(self.metrics.episode_returns[-5:]) if self.metrics.episode_returns else 0.0
                actor_phase = 'frozen' if self.total_steps <= self.actor_freeze_steps else 'active'
                print(
                    f"{log_prefix} progress={step_idx + 1}/{total_timesteps} episodes={self.metrics.episodes} "
                    f"buffer={len(self.replay)} success_frac={self.replay.success_fraction():.3f} "
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
                        print(f"{log_prefix} early stop triggered: {stop_reason}")
        self.metrics.steps = self.total_steps
        self.actor.to('cpu')
        self.critic1.to('cpu')
        self.critic2.to('cpu')
        return self.metrics

    def set_bc_reference_actor(self, actor: nn.Module, source: str = 'bc') -> None:
        self.bc_reference_actor = actor.to(self.device)
        self.bc_reference_actor.eval()
        for param in self.bc_reference_actor.parameters():
            param.requires_grad = False
        self.metrics.bc_regularization_enabled = True
        self.metrics.reference_source = source

    def _bc_lambda(self) -> float:
        if self.total_steps < 75_000:
            return 500.0
        if self.total_steps < 150_000:
            return 150.0
        if self.total_steps < 250_000:
            return 30.0
        return 5.0

    def _is_near_goal(self, info: dict[str, Any]) -> bool:
        goal_distance = float(info.get('goal_distance', float('inf')))
        segment_goal_distance = float(info.get('segment_goal_distance', float('inf')))
        goal_reached_by_segment = bool(info.get('goal_reached_by_segment', False))
        return (
            goal_distance <= self.near_goal_radius
            or segment_goal_distance <= self.near_goal_radius
            or goal_reached_by_segment
        )

    def _current_noise(self) -> tuple[float, float, float]:
        stage_progress = min(float(self.total_steps) / max(float(self.current_stage_timesteps), 1.0), 1.0)
        decay_progress = min(stage_progress / max(self.noise_decay_fraction, 1e-6), 1.0)
        if decay_progress >= 1.0:
            self.exploration_noise_current = self.exploration_noise_final
            self.policy_noise_current = self.policy_noise_final
            self.noise_clip_current = self.noise_clip_final
            return self.exploration_noise_current, self.policy_noise_current, self.noise_clip_current
        self.exploration_noise_current = self.exploration_noise_base + (
            self.exploration_noise_final - self.exploration_noise_base
        ) * decay_progress
        self.policy_noise_current = self.policy_noise_base + (
            self.policy_noise_final - self.policy_noise_base
        ) * decay_progress
        self.noise_clip_current = self.noise_clip_base + (
            self.noise_clip_final - self.noise_clip_base
        ) * decay_progress
        return self.exploration_noise_current, self.policy_noise_current, self.noise_clip_current

    def select_action(self, obs: np.ndarray, with_noise: bool = False) -> np.ndarray:
        obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        if with_noise:
            exploration_noise, _, _ = self._current_noise()
            action = action + np.random.normal(0.0, exploration_noise, size=action.shape)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high).astype(np.float32)

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

    def _compute_actor_loss_terms(
        self,
        obs: torch.Tensor,
        line_to_goal_safe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, torch.Tensor, float]:
        actor_actions = self.actor(obs)
        q_values = self.critic1(obs, actor_actions)
        rl_actor_loss = -q_values.mean()
        q_scale = q_values.detach().abs().mean().clamp(min=1.0)
        actor_rl_scale = float(self.actor_rl_scale_alpha / q_scale.item())
        scaled_rl_actor_loss = rl_actor_loss * actor_rl_scale
        bc_lambda = 0.0
        bc_loss = torch.zeros((), device=self.device)
        if self.bc_reference_actor is not None:
            bc_lambda = self._bc_lambda()
            with torch.no_grad():
                bc_actions = self.bc_reference_actor(obs)
            bc_loss = F.mse_loss(actor_actions, bc_actions)
        terminal_geo_loss = torch.zeros((), device=self.device)
        terminal_geo_lambda = 0.0
        if self.terminal_geo_regularization_enabled:
            terminal_geo_loss = self._terminal_geo_loss(obs, actor_actions, line_to_goal_safe)
            if terminal_geo_loss.detach().abs().item() > 0.0:
                terminal_geo_lambda = self.terminal_geo_lambda
        actor_loss = scaled_rl_actor_loss + bc_lambda * bc_loss + terminal_geo_lambda * terminal_geo_loss
        return (
            actor_loss,
            rl_actor_loss,
            scaled_rl_actor_loss,
            bc_loss,
            bc_lambda,
            actor_rl_scale,
            terminal_geo_loss,
            terminal_geo_lambda,
        )

    def _terminal_geo_loss(
        self,
        obs: torch.Tensor,
        actor_actions: torch.Tensor,
        line_to_goal_safe: torch.Tensor,
    ) -> torch.Tensor:
        if not self.terminal_geo_regularization_enabled:
            return torch.zeros((), device=self.device)

        rel_goal = obs[:, 5:8]
        goal_distance = torch.linalg.norm(rel_goal, dim=1)
        safe_mask = line_to_goal_safe.squeeze(-1) > 0.5
        eligible_mask = safe_mask & (goal_distance <= self.terminal_geo_radius)
        if not torch.any(eligible_mask):
            return torch.zeros((), device=self.device)

        gamma = obs[:, 3]
        psi = obs[:, 4]
        dx = rel_goal[:, 0]
        dy = rel_goal[:, 1]
        dz = rel_goal[:, 2]
        horizontal = torch.sqrt(torch.clamp(dx.square() + dy.square(), min=1e-9))
        target_gamma = torch.atan2(dz, horizontal)
        target_psi = torch.atan2(dy, dx)

        delta_gamma = torch.clamp(
            target_gamma - gamma,
            min=float(self.action_low[0].item()),
            max=float(self.action_high[0].item()),
        )
        delta_psi = torch.clamp(
            self._wrap_angle_tensor(target_psi - psi),
            min=float(self.action_low[1].item()),
            max=float(self.action_high[1].item()),
        )
        target_action = torch.stack([delta_gamma, delta_psi], dim=-1)
        return F.mse_loss(actor_actions[eligible_mask], target_action[eligible_mask])

    @staticmethod
    def _wrap_angle_tensor(value: torch.Tensor) -> torch.Tensor:
        two_pi = float(2.0 * np.pi)
        return torch.remainder(value + float(np.pi), two_pi) - float(np.pi)

    def _update(self) -> None:
        batch = self.replay.sample(self.batch_size)
        obs = batch['obs'].to(self.device)
        actions = batch['action'].to(self.device)
        rewards = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)
        line_to_goal_safe = batch['line_to_goal_safe'].to(self.device)
        self.metrics.replay_success_fraction = self.replay.success_fraction()
        self.metrics.replay_near_goal_fraction = self.replay.near_goal_fraction()
        self.metrics.success_replay_size = self.replay.success_size
        self.metrics.success_replay_fraction = self.success_replay_fraction
        self.metrics.sample_success_fraction = float(batch['success'].float().mean().item())
        self.metrics.sample_near_goal_fraction = float(batch['near_goal'].float().mean().item())

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
                (
                    actor_loss,
                    rl_actor_loss,
                    scaled_rl_actor_loss,
                    bc_loss,
                    bc_lambda,
                    actor_rl_scale,
                    terminal_geo_loss,
                    terminal_geo_lambda,
                ) = self._compute_actor_loss_terms(obs, line_to_goal_safe)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.actor_grad_clip_norm is not None and self.actor_grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        max_norm=self.actor_grad_clip_norm,
                    )
                self.actor_optimizer.step()
                self._soft_update(self.actor, self.actor_target)
                self.metrics.actor_loss = float(actor_loss.item())
                self.metrics.rl_actor_loss = float(rl_actor_loss.item())
                self.metrics.scaled_rl_actor_loss = float(scaled_rl_actor_loss.item())
                self.metrics.actor_rl_scale = float(actor_rl_scale)
                self.metrics.bc_loss = float(bc_loss.item())
                self.metrics.bc_lambda = float(bc_lambda)
                self.metrics.terminal_geo_loss = float(terminal_geo_loss.item())
                self.metrics.terminal_geo_lambda = float(terminal_geo_lambda)

    def _soft_update(self, model: nn.Module, target: nn.Module) -> None:
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
