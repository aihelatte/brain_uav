"""Replay buffer used by TD3."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Store transitions in ring-buffer arrays for off-policy learning."""

    def __init__(
        self,
        capacity: int,
        success_sample_bias: float = 1.0,
        near_goal_sample_bias: float = 1.0,
    ) -> None:
        self.capacity = int(capacity)
        self.success_sample_bias = max(float(success_sample_bias), 1.0)
        self.near_goal_sample_bias = max(float(near_goal_sample_bias), 1.0)
        self.obs: np.ndarray | None = None
        self.action: np.ndarray | None = None
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs: np.ndarray | None = None
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.success = np.zeros(self.capacity, dtype=np.bool_)
        self.near_goal = np.zeros(self.capacity, dtype=np.bool_)
        self.line_to_goal_safe = np.zeros(self.capacity, dtype=np.bool_)
        self.sample_weight = np.ones(self.capacity, dtype=np.float64)
        self.size = 0
        self.position = 0
        self.success_count = 0
        self.near_goal_count = 0
        self.total_sample_weight = 0.0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        success: bool = False,
        near_goal: bool = False,
        line_to_goal_safe: bool = False,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        self._ensure_storage(obs, action, next_obs)

        idx = self.position
        previous_weight = 0.0
        if self.size == self.capacity:
            if self.success[idx]:
                self.success_count -= 1
            if self.near_goal[idx]:
                self.near_goal_count -= 1
            previous_weight = float(self.sample_weight[idx])
        else:
            self.size += 1

        self.total_sample_weight -= previous_weight
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx, 0] = float(reward)
        self.next_obs[idx] = next_obs
        self.done[idx, 0] = float(done)
        self.success[idx] = bool(success)
        self.near_goal[idx] = bool(near_goal)
        self.line_to_goal_safe[idx] = bool(line_to_goal_safe)

        if self.success[idx]:
            self.success_count += 1
        if self.near_goal[idx]:
            self.near_goal_count += 1

        self.sample_weight[idx] = self._slot_weight(idx)
        self.total_sample_weight += float(self.sample_weight[idx])
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size > self.size:
            raise ValueError(f'batch_size={batch_size} exceeds replay size={self.size}')

        probs = self._sampling_probabilities()
        if probs is None:
            idx = np.random.choice(self.size, batch_size, replace=False)
        else:
            idx = np.random.choice(self.size, batch_size, replace=False, p=probs)
        return {
            'obs': torch.from_numpy(self.obs[idx].copy()),
            'action': torch.from_numpy(self.action[idx].copy()),
            'reward': torch.from_numpy(self.reward[idx].copy()),
            'next_obs': torch.from_numpy(self.next_obs[idx].copy()),
            'done': torch.from_numpy(self.done[idx].copy()),
            'success': torch.from_numpy(self.success[idx].astype(np.float32).reshape(-1, 1)),
            'near_goal': torch.from_numpy(self.near_goal[idx].astype(np.float32).reshape(-1, 1)),
            'line_to_goal_safe': torch.from_numpy(self.line_to_goal_safe[idx].astype(np.float32).reshape(-1, 1)),
        }

    def success_fraction(self) -> float:
        if self.size == 0:
            return 0.0
        return float(self.success_count / self.size)

    def near_goal_fraction(self) -> float:
        if self.size == 0:
            return 0.0
        return float(self.near_goal_count / self.size)

    def _ensure_storage(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> None:
        if self.obs is None:
            self.obs = np.zeros((self.capacity, *obs.shape), dtype=np.float32)
            self.action = np.zeros((self.capacity, *action.shape), dtype=np.float32)
            self.next_obs = np.zeros((self.capacity, *next_obs.shape), dtype=np.float32)

    def _slot_weight(self, idx: int) -> float:
        weight = 1.0
        if self.success_sample_bias > 1.0 and self.success[idx]:
            weight *= self.success_sample_bias
        if self.near_goal_sample_bias > 1.0 and self.near_goal[idx]:
            weight *= self.near_goal_sample_bias
        return float(weight)

    def _sampling_probabilities(self) -> np.ndarray | None:
        if self.size == 0:
            return None
        if self.success_sample_bias <= 1.0 and self.near_goal_sample_bias <= 1.0:
            return None
        if self.total_sample_weight <= 0.0:
            return None
        return self.sample_weight[: self.size] / self.total_sample_weight
