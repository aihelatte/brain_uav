from __future__ import annotations

import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        scale = torch.clamp(1.0 - (membrane - threshold).abs(), min=0.0)
        return grad_output * scale, None


class LIFLayer(nn.Module):
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
            spikes = SurrogateSpike.apply(membrane, self.threshold)
            membrane = membrane * (1.0 - spikes)
            spike_sum += spikes
            last_membrane = membrane
        return spike_sum / steps, last_membrane, spike_sum


class SNNPolicyActor(nn.Module):
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
        self.lif1 = LIFLayer()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = LIFLayer()
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.time_window = time_window
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.register_buffer("action_limit", action_limit)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action, _ = self.forward_with_diagnostics(obs)
        return action

    def forward_with_diagnostics(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        encoded = self.fc1(obs)
        spikes1, _, spike_sum1 = self.lif1(encoded, self.time_window)
        hidden = self.fc2(spikes1)
        spikes2, membrane, spike_sum2 = self.lif2(hidden, self.time_window)
        out = self.fc3(0.5 * (spikes2 + membrane))
        action = torch.tanh(out) * self.action_limit
        diagnostics = {
            "spike_rate_l1": float((spike_sum1 / self.time_window).mean().detach().cpu()),
            "spike_rate_l2": float((spike_sum2 / self.time_window).mean().detach().cpu()),
            "dense_macs_estimate": float(
                self.state_dim * self.hidden_dim + self.hidden_dim * self.hidden_dim + self.hidden_dim * self.action_dim
            ),
        }
        diagnostics["effective_macs_estimate"] = float(
            self.state_dim * self.hidden_dim
            + self.hidden_dim * self.hidden_dim * diagnostics["spike_rate_l1"]
            + self.hidden_dim * self.action_dim * diagnostics["spike_rate_l2"]
        )
        return action, diagnostics
