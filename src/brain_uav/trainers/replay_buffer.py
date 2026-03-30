from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.buffer.append(Transition(obs, action, reward, next_obs, float(done)))

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return {
            "obs": torch.tensor(np.stack([item.obs for item in batch]), dtype=torch.float32),
            "action": torch.tensor(np.stack([item.action for item in batch]), dtype=torch.float32),
            "reward": torch.tensor([[item.reward] for item in batch], dtype=torch.float32),
            "next_obs": torch.tensor(np.stack([item.next_obs for item in batch]), dtype=torch.float32),
            "done": torch.tensor([[item.done] for item in batch], dtype=torch.float32),
        }

