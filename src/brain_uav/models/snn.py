"""Spiking neural network actor.

优先使用 SpikingJelly；如果环境里没有这个库，就回退到一个简化版 LIF 实现。
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False
    functional = None
    neuron = None
    surrogate = None


class FallbackLIFLayer(nn.Module):
    """A tiny differentiable LIF approximation used only as fallback."""

    def __init__(self, decay: float = 0.5, threshold: float = 1.0) -> None:
        super().__init__()
        self.decay = decay
        self.threshold = threshold

    def forward(self, current: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        membrane = torch.zeros_like(current)
        spike_sum = torch.zeros_like(current)
        last_membrane = torch.zeros_like(current)
        for _ in range(steps):
            membrane = self.decay * membrane + current
            spikes = torch.sigmoid(5.0 * (membrane - self.threshold))
            hard = (membrane >= self.threshold).to(membrane.dtype)
            # 这里用 straight-through 的方式，让前向是硬脉冲，反向仍可求梯度。
            spikes = spikes + (hard - spikes).detach()
            membrane = membrane * (1.0 - hard)
            spike_sum += hard
            last_membrane = membrane
        return spike_sum / steps, last_membrane, spike_sum


class SNNPolicyActor(nn.Module):
    """Actor network for the proposed SNN-based method."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        time_window: int,
        action_limit: torch.Tensor,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.time_window = time_window
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        if HAS_SPIKINGJELLY:
            self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), step_mode='m')
            self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), step_mode='m')
        else:
            self.lif1 = FallbackLIFLayer()
            self.lif2 = FallbackLIFLayer()
        self.register_buffer('action_limit', action_limit)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if HAS_SPIKINGJELLY:
            action, _, _ = self._forward_spikingjelly(obs)
            return action
        action, _, _ = self._forward_fallback(obs)
        return action

    def forward_with_diagnostics(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass plus some extra statistics for profiling.

        这些统计包括：
        - 每层平均脉冲发放率
        - 稠密 MACs 估计
        - 根据脉冲活动率折算后的有效 MACs
        """

        if HAS_SPIKINGJELLY:
            action, spike_sum1, spike_sum2 = self._forward_spikingjelly(obs)
        else:
            action, spike_sum1, spike_sum2 = self._forward_fallback(obs)
        macs_per_timestep = float(
            self.state_dim * self.hidden_dim + self.hidden_dim * self.hidden_dim + self.hidden_dim * self.action_dim
        )
        dense_macs = macs_per_timestep * float(self.time_window)
        effective_macs = float(
            (self.state_dim * self.hidden_dim)
            + (self.hidden_dim * self.hidden_dim) * float((spike_sum1 / self.time_window).mean().detach().cpu())
            + (self.hidden_dim * self.action_dim) * float((spike_sum2 / self.time_window).mean().detach().cpu())
        )
        effective_macs *= float(self.time_window)
        diagnostics = {
            'backend': 'spikingjelly' if HAS_SPIKINGJELLY else 'fallback',
            'time_window': float(self.time_window),
            'spike_rate_l1': float((spike_sum1 / self.time_window).mean().detach().cpu()),
            'spike_rate_l2': float((spike_sum2 / self.time_window).mean().detach().cpu()),
            'dense_macs_estimate': float(dense_macs),
            'effective_macs_estimate': float(effective_macs),
            'dense_macs_per_timestep': float(macs_per_timestep),
        }
        return action, diagnostics

    def _forward_spikingjelly(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        functional.reset_net(self)
        encoded = self.fc1(obs)
        encoded_seq = encoded.unsqueeze(0).expand(self.time_window, -1, -1)
        spikes1_seq = self.lif1(encoded_seq)
        hidden_seq = self.fc2(spikes1_seq)
        spikes2_seq = self.lif2(hidden_seq)
        membrane = self.lif2.v
        features = 0.5 * (spikes2_seq.mean(0) + membrane)
        out = self.fc3(features)
        action = torch.tanh(out) * self.action_limit
        spike_sum1 = spikes1_seq.sum(0)
        spike_sum2 = spikes2_seq.sum(0)
        functional.reset_net(self)
        return action, spike_sum1, spike_sum2

    def _forward_fallback(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.fc1(obs)
        spikes1, _, spike_sum1 = self.lif1(encoded, self.time_window)
        hidden = self.fc2(spikes1)
        spikes2, membrane, spike_sum2 = self.lif2(hidden, self.time_window)
        out = self.fc3(0.5 * (spikes2 + membrane))
        action = torch.tanh(out) * self.action_limit
        return action, spike_sum1, spike_sum2
