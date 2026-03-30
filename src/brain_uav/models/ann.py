"""Standard ANN models used as baseline actor and critic."""

from __future__ import annotations

import torch
from torch import nn


class ANNPolicyActor(nn.Module):
    """Continuous control actor implemented with a plain MLP."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_limit: torch.Tensor) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # action_limit 用来把 [-1, 1] 的输出缩放到环境动作范围。
        self.register_buffer("action_limit", action_limit)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.action_limit


class ANNCritic(nn.Module):
    """Q-value network for TD3.

    输入是 (state, action)，输出是这对状态动作的价值估计 Q。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))
