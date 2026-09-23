"""Spiking policy actor for structured dynamic V2 observations."""

from __future__ import annotations

from contextlib import contextmanager
from math import isfinite
from typing import Any, Iterator

import torch
from torch import nn

from brain_uav.observations import V2ObservationBatch, V2ObservationScales

from .zone_set_encoder import (
    ZoneSetEncoder,
    ZoneSetEncoderConfig,
    ZoneSetSharedRelations,
)

try:
    from spikingjelly.activation_based import functional, neuron, surrogate

    _SPIKINGJELLY_AVAILABLE = True
    _SPIKINGJELLY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised with a controlled mock
    functional = None
    neuron = None
    surrogate = None
    _SPIKINGJELLY_AVAILABLE = False
    _SPIKINGJELLY_IMPORT_ERROR = exc


V2_SNN_BACKEND = 'torch'
V2_SNN_SURROGATE = 'atan'


def require_v2_spikingjelly() -> None:
    """Fail explicitly when the required SpikingJelly implementation is absent."""

    if not _SPIKINGJELLY_AVAILABLE:
        raise RuntimeError(
            'V2 SNN requires SpikingJelly activation_based with the torch backend; '
            'no fallback implementation is permitted.'
        ) from _SPIKINGJELLY_IMPORT_ERROR


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer.')
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a positive integer.') from exc
    if result <= 0 or result != value:
        raise ValueError(f'{name} must be a positive integer.')
    return result


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite and greater than zero.')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and greater than zero.') from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f'{name} must be finite and greater than zero.')
    return result


def _reset_linear_layers(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            nn.init.zeros_(child.bias)


class V2SNNPolicyHead(nn.Module):
    """Two-layer multi-step LIF decision head over one encoded scene context."""

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        action_dim: int,
        *,
        time_window: int,
        tau: float,
    ) -> None:
        super().__init__()
        require_v2_spikingjelly()
        self.context_dim = _positive_int(context_dim, name='context_dim')
        self.hidden_dim = _positive_int(hidden_dim, name='hidden_dim')
        self.action_dim = _positive_int(action_dim, name='action_dim')
        self.time_window = _positive_int(time_window, name='time_window')
        self.tau = _positive_float(tau, name='tau')
        self.fc1 = nn.Linear(self.context_dim, self.hidden_dim)
        self.lif1 = neuron.LIFNode(
            tau=self.tau,
            surrogate_function=surrogate.ATan(),
            step_mode='m',
            backend=V2_SNN_BACKEND,
        )
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lif2 = neuron.LIFNode(
            tau=self.tau,
            surrogate_function=surrogate.ATan(),
            step_mode='m',
            backend=V2_SNN_BACKEND,
        )
        self.action_layer = nn.Linear(self.hidden_dim, self.action_dim)
        _reset_linear_layers(self)
        nn.init.uniform_(self.action_layer.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.action_layer.bias, -1e-3, 1e-3)

    def _forward_impl(
        self, context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(context, torch.Tensor):
            raise TypeError('context must be a torch.Tensor.')
        if context.dtype != torch.float32:
            raise TypeError('context must have dtype torch.float32.')
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(
                f'context must have shape [B, {self.context_dim}]; '
                f'got {tuple(context.shape)}.'
            )
        projected = self.fc1(context)
        current_sequence = projected.unsqueeze(0).expand(
            self.time_window, -1, -1
        )
        first_spikes = self.lif1(current_sequence)
        second_spikes = self.lif2(self.fc2(first_spikes))
        final_membrane = self.lif2.v
        if not isinstance(final_membrane, torch.Tensor) or final_membrane.shape != (
            context.shape[0],
            self.hidden_dim,
        ):
            raise RuntimeError(
                'SpikingJelly LIF membrane state has an incompatible shape.'
            )
        readout = 0.5 * (second_spikes.mean(dim=0) + final_membrane)
        action = self.action_layer(readout)
        return action, first_spikes, second_spikes

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        action, _, _ = self._forward_impl(context)
        return action

    def forward_with_diagnostics(
        self, context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass plus per-layer firing-rate statistics (H9 in docs).

        Reuses the exact same computation as :meth:`forward` (via
        ``_forward_impl``) so the returned action and the training numerics
        are unaffected; this only additionally exposes the spike tensors
        that ``forward`` already computes and discards.
        """

        action, first_spikes, second_spikes = self._forward_impl(context)
        diagnostics = {
            'spike_rate_l1': float(first_spikes.mean().detach().cpu()),
            'spike_rate_l2': float(second_spikes.mean().detach().cpu()),
        }
        return action, diagnostics


class V2SNNPolicyActor(nn.Module):
    """V2 dynamic-set encoder followed by a strict SpikingJelly SNN head."""

    model_type = 'snn'
    surrogate_name = V2_SNN_SURROGATE
    backend = V2_SNN_BACKEND

    def __init__(
        self,
        scales: V2ObservationScales,
        action_dim: int,
        hidden_dim: int,
        action_limit: torch.Tensor,
        *,
        time_window: int = 4,
        tau: float = 2.0,
        uav_radius: float = 0.0,
        encoder_config: ZoneSetEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        require_v2_spikingjelly()
        if not isinstance(scales, V2ObservationScales):
            raise TypeError('scales must be a V2ObservationScales instance.')
        self.action_dim = _positive_int(action_dim, name='action_dim')
        self.hidden_dim = _positive_int(hidden_dim, name='hidden_dim')
        self.time_window = _positive_int(time_window, name='time_window')
        self.tau = _positive_float(tau, name='tau')
        if encoder_config is None:
            encoder_config = ZoneSetEncoderConfig()
        if not isinstance(encoder_config, ZoneSetEncoderConfig):
            raise TypeError('encoder_config must be a ZoneSetEncoderConfig.')
        if not isinstance(action_limit, torch.Tensor):
            raise TypeError('action_limit must be a torch.Tensor.')
        limit = action_limit.detach().to(dtype=torch.float32).clone()
        if limit.shape != (self.action_dim,):
            raise ValueError(
                f'action_limit must have shape ({self.action_dim},); '
                f'got {tuple(limit.shape)}.'
            )
        if not bool(torch.isfinite(limit).all()) or not bool((limit > 0.0).all()):
            raise ValueError('action_limit must contain finite positive values.')

        self.scales = scales
        self.encoder_config = encoder_config
        self.uav_radius = float(uav_radius)
        self.zone_set_encoder = ZoneSetEncoder(
            scales,
            uav_radius=uav_radius,
            config=encoder_config,
        )
        self.snn_head = V2SNNPolicyHead(
            self.zone_set_encoder.output_dim,
            self.hidden_dim,
            self.action_dim,
            time_window=self.time_window,
            tau=self.tau,
        )
        self.register_buffer('action_limit', limit)

    @contextmanager
    def reset_state_context(self) -> Iterator[None]:
        functional.reset_net(self.snn_head)
        try:
            yield
        finally:
            functional.reset_net(self.snn_head)

    def action_from_context(self, context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.snn_head(context)) * self.action_limit

    def action_from_context_with_diagnostics(
        self, context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raw_action, diagnostics = self.snn_head.forward_with_diagnostics(context)
        return torch.tanh(raw_action) * self.action_limit, diagnostics

    def forward(
        self,
        observation: V2ObservationBatch,
        *,
        shared_relations: ZoneSetSharedRelations | None = None,
        profile_sections: bool = False,
    ) -> torch.Tensor:
        if not isinstance(observation, V2ObservationBatch):
            raise TypeError('observation must be a V2ObservationBatch.')
        with self.reset_state_context():
            context = self.zone_set_encoder(
                observation.ego_features,
                observation.goal_features,
                observation.zone_features,
                observation.presence_mask,
                shared_relations=shared_relations,
                profile_sections=profile_sections,
            )
            return self.action_from_context(context)

    def forward_with_diagnostics(
        self,
        observation: V2ObservationBatch,
        *,
        shared_relations: ZoneSetSharedRelations | None = None,
        profile_sections: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass plus LIF firing-rate diagnostics (H9 in docs).

        Diagnostic-only: does not change ``forward``'s computation path or
        numerics. Intended for low-frequency, offline instrumentation, not
        the per-step training hot path.
        """

        if not isinstance(observation, V2ObservationBatch):
            raise TypeError('observation must be a V2ObservationBatch.')
        with self.reset_state_context():
            context = self.zone_set_encoder(
                observation.ego_features,
                observation.goal_features,
                observation.zone_features,
                observation.presence_mask,
                shared_relations=shared_relations,
                profile_sections=profile_sections,
            )
            return self.action_from_context_with_diagnostics(context)


__all__ = [
    'V2_SNN_BACKEND',
    'V2_SNN_SURROGATE',
    'V2SNNPolicyActor',
    'V2SNNPolicyHead',
    'require_v2_spikingjelly',
]
