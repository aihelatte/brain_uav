"""Standard ANN models used as baseline actor and critic."""

from __future__ import annotations

import torch
from torch import nn


class ANNPolicyActor(nn.Module):
    """Continuous control actor implemented with a plain MLP.

    这里使用 LeakyReLU 而不是 ReLU，主要是为了避免 BC 预训练阶段
    因 ReLU 死区导致网络长期输出接近 0 的“交白卷”现象。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_limit: torch.Tensor) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # action_limit 用来把 [-1, 1] 的输出缩放到环境动作范围。
        self.register_buffer("action_limit", action_limit)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Use a stable initialization for BC and TD3 training."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # 最后一层缩小一点，避免一开始动作幅值过大。
        final_linear = self.net[4]
        nn.init.uniform_(final_linear.weight, -1e-3, 1e-3)
        nn.init.uniform_(final_linear.bias, -1e-3, 1e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.action_limit


class ANNCritic(nn.Module):
    """Q-value network for TD3.

    输入是 (state, action)，输出是这对状态动作的价值估计 Q。
    同样使用 LeakyReLU，避免 critic 也出现死神经元问题。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))
