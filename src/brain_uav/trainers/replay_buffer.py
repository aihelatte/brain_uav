"""Replay buffer used by TD3."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class Transition:
    """One RL transition."""

    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: float
    success: float


class ReplayBuffer:
    """Store transitions and sample random mini-batches for off-policy learning."""

    def __init__(self, capacity: int, success_sample_bias: float = 1.0) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.success_sample_bias = max(float(success_sample_bias), 1.0)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        success: bool = False,
    ) -> None:
        self.buffer.append(Transition(obs, action, reward, next_obs, float(done), float(success)))

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.success_sample_bias > 1.0:
            weights = np.array(
                [self.success_sample_bias if item.success > 0.5 else 1.0 for item in self.buffer],
                dtype=np.float64,
            )
            probs = weights / weights.sum()
            idx = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        else:
            idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return {
            'obs': torch.tensor(np.stack([item.obs for item in batch]), dtype=torch.float32),
            'action': torch.tensor(np.stack([item.action for item in batch]), dtype=torch.float32),
            'reward': torch.tensor([[item.reward] for item in batch], dtype=torch.float32),
            'next_obs': torch.tensor(np.stack([item.next_obs for item in batch]), dtype=torch.float32),
            'done': torch.tensor([[item.done] for item in batch], dtype=torch.float32),
            'success': torch.tensor([[item.success] for item in batch], dtype=torch.float32),
        }

    def success_fraction(self) -> float:
        if not self.buffer:
            return 0.0
        return float(sum(item.success for item in self.buffer) / len(self.buffer))
