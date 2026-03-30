from __future__ import annotations

import json
import math
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

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

    def to_dict(self) -> dict:
        return {
            'actor_loss': self.actor_loss,
            'critic_loss': self.critic_loss,
            'steps': self.steps,
            'episodes': self.episodes,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'outcomes': self.outcomes,
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
        self.device = device
        self.replay = ReplayBuffer(replay_size)
        self.total_steps = 0
        self.metrics = TD3Metrics()
        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)

    def train(self, total_timesteps: int) -> TD3Metrics:
        obs, _ = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        for _ in range(total_timesteps):
            self.total_steps += 1
            if self.total_steps <= self.warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs, with_noise=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.replay.add(obs, action, reward, next_obs, done)
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
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_length = 0
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
