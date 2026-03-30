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
            self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
            self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        else:
            self.lif1 = FallbackLIFLayer()
            self.lif2 = FallbackLIFLayer()
        self.register_buffer('action_limit', action_limit)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action, _ = self.forward_with_diagnostics(obs)
        return action

    def forward_with_diagnostics(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass plus some extra statistics for profiling.

        这些统计包括：
        - 每层平均脉冲发放率
        - 稠密 MACs 估计
        - 根据脉冲活动率折算后的有效 MACs
        """

        if HAS_SPIKINGJELLY:
            functional.reset_net(self)
            encoded = self.fc1(obs)
            spike_trace_1 = []
            for _ in range(self.time_window):
                spike_trace_1.append(self.lif1(encoded))
            spikes1 = torch.stack(spike_trace_1, dim=0).mean(0)

            hidden = self.fc2(spikes1)
            spike_trace_2 = []
            mem_trace_2 = []
            for _ in range(self.time_window):
                spk = self.lif2(hidden)
                spike_trace_2.append(spk)
                mem_trace_2.append(self.lif2.v.clone())
            spikes2 = torch.stack(spike_trace_2, dim=0).mean(0)
            membrane = torch.stack(mem_trace_2, dim=0).mean(0)
            spike_sum1 = torch.stack(spike_trace_1, dim=0).sum(0)
            spike_sum2 = torch.stack(spike_trace_2, dim=0).sum(0)
            out = self.fc3(0.5 * (spikes2 + membrane))
            functional.reset_net(self)
        else:
            encoded = self.fc1(obs)
            spikes1, _, spike_sum1 = self.lif1(encoded, self.time_window)
            hidden = self.fc2(spikes1)
            spikes2, membrane, spike_sum2 = self.lif2(hidden, self.time_window)
            out = self.fc3(0.5 * (spikes2 + membrane))

        action = torch.tanh(out) * self.action_limit
        diagnostics = {
            'backend': 'spikingjelly' if HAS_SPIKINGJELLY else 'fallback',
            'spike_rate_l1': float((spike_sum1 / self.time_window).mean().detach().cpu()),
            'spike_rate_l2': float((spike_sum2 / self.time_window).mean().detach().cpu()),
            'dense_macs_estimate': float(
                self.state_dim * self.hidden_dim + self.hidden_dim * self.hidden_dim + self.hidden_dim * self.action_dim
            ),
        }
        diagnostics['effective_macs_estimate'] = float(
            self.state_dim * self.hidden_dim
            + self.hidden_dim * self.hidden_dim * diagnostics['spike_rate_l1']
            + self.hidden_dim * self.action_dim * diagnostics['spike_rate_l2']
        )
        return action, diagnostics
