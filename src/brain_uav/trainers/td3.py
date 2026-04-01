"""TD3 trainer.

这是强化学习的主循环。
你可以把它理解成：
- Actor 负责出动作
- 两个 Critic 负责打分
- 回放缓存负责反复学习过去经验
"""

from __future__ import annotations

import statistics
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .replay_buffer import ReplayBuffer


@dataclass(slots=True)
class TD3Metrics:
    actor_loss: float = 0.0
    critic_loss: float = 0.0
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
        warmup_strategy: str = 'random',
        device: str = 'cpu',
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
        self.warmup_strategy = warmup_strategy
        self.device = device
        self.replay = ReplayBuffer(replay_size)
        self.total_steps = 0
        self.metrics = TD3Metrics()
        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
        self._current_window: list[dict] = []

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 500,
        verbose: bool = True,
        summary_every_episodes: int = 50,
    ) -> TD3Metrics:
        obs, _ = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        if verbose:
            print(
                f"[TD3] start total_timesteps={total_timesteps} warmup_steps={self.warmup_steps} "
                f"batch_size={self.batch_size} replay_size={self.replay.buffer.maxlen} "
                f"warmup_strategy={self.warmup_strategy} summary_every_episodes={summary_every_episodes}"
            )
        for step_idx in range(total_timesteps):
            self.total_steps += 1
            if self.total_steps <= self.warmup_steps:
                action = self._warmup_action(obs)
            else:
                action = self.select_action(obs, with_noise=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            # 注意：只把 terminated 当成 TD target 的“真实结束”掩码。
            self.replay.add(obs, action, reward, next_obs, terminated)
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
                self._current_window.append(
                    {
                        'episode': self.metrics.episodes,
                        'return': float(episode_return),
                        'length': int(episode_length),
                        'outcome': outcome,
                        'actor_loss': float(self.metrics.actor_loss),
                        'critic_loss': float(self.metrics.critic_loss),
                    }
                )
                if summary_every_episodes > 0 and len(self._current_window) >= summary_every_episodes:
                    self._flush_window_stats()
                if verbose:
                    print(
                        f"[TD3] episode={self.metrics.episodes} step={self.total_steps}/{total_timesteps} "
                        f"return={episode_return:.2f} length={episode_length} outcome={outcome}"
                    )
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_length = 0
            if verbose and ((step_idx + 1) % log_interval == 0 or (step_idx + 1) == total_timesteps):
                avg_return = statistics.mean(self.metrics.episode_returns[-5:]) if self.metrics.episode_returns else 0.0
                print(
                    f"[TD3] progress={step_idx + 1}/{total_timesteps} episodes={self.metrics.episodes} "
                    f"buffer={len(self.replay)} actor_loss={self.metrics.actor_loss:.4f} "
                    f"critic_loss={self.metrics.critic_loss:.4f} recent_avg_return={avg_return:.2f}"
                )
        if self._current_window:
            self._flush_window_stats()
        self.metrics.steps = self.total_steps
        self.actor.to('cpu')
        self.critic1.to('cpu')
        self.critic2.to('cpu')
        return self.metrics

    def select_action(self, obs: np.ndarray, with_noise: bool = False) -> np.ndarray:
        obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        if with_noise:
            action = action + np.random.normal(0.0, self.exploration_noise, size=action.shape)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high).astype(np.float32)

    def _warmup_action(self, obs: np.ndarray) -> np.ndarray:
        if self.warmup_strategy == 'policy':
            return self.select_action(obs, with_noise=True)
        return self.env.action_space.sample()

    def _flush_window_stats(self) -> None:
        """Aggregate recent episodes into one AI-friendly summary block."""

        window = self._current_window
        outcome_counts: dict[str, int] = {}
        for item in window:
            outcome_counts[item['outcome']] = outcome_counts.get(item['outcome'], 0) + 1
        row = {
            'episode_start': window[0]['episode'],
            'episode_end': window[-1]['episode'],
            'episode_count': len(window),
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

    def _update(self) -> None:
        batch = self.replay.sample(self.batch_size)
        obs = batch['obs'].to(self.device)
        actions = batch['action'].to(self.device)
        rewards = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(self.action_low, self.action_high)
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = rewards + (1.0 - done) * self.gamma * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.metrics.critic_loss = float(critic_loss.item())

        if self.total_steps % self.policy_delay == 0:
            actor_actions = self.actor(obs)
            actor_loss = -self.critic1(obs, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            self.metrics.actor_loss = float(actor_loss.item())

    def _soft_update(self, model: nn.Module, target: nn.Module) -> None:
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
