"""ANN actor and critic for structured dynamic V2 observations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch
from torch import nn

from brain_uav.observations import V2ObservationBatch, V2ObservationScales

from .zone_set_encoder import (
    ZoneSetEncoder,
    ZoneSetEncoderConfig,
    ZoneSetSharedRelations,
)


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


def _reset_linear_layers(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            nn.init.zeros_(child.bias)


class V2ANNPolicyActor(nn.Module):
    """ANN policy over a shared-parameter dynamic zone-set encoder."""

    def __init__(
        self,
        scales: V2ObservationScales,
        action_dim: int,
        hidden_dim: int,
        action_limit: torch.Tensor,
        *,
        uav_radius: float = 0.0,
        encoder_config: ZoneSetEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(scales, V2ObservationScales):
            raise TypeError('scales must be a V2ObservationScales instance.')
        self.action_dim = _positive_int(action_dim, name='action_dim')
        self.hidden_dim = _positive_int(hidden_dim, name='hidden_dim')
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
        self.head = nn.Sequential(
            nn.Linear(self.zone_set_encoder.output_dim, self.hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh(),
        )
        self.register_buffer('action_limit', limit)
        self._compiled_full_forward: Callable[..., torch.Tensor] | None = None
        self._compiled_full_forward_config: dict[str, object] | None = None
        self._force_eager_full_forward = False
        _reset_linear_layers(self.head)
        final_linear = self.head[4]
        nn.init.uniform_(final_linear.weight, -1e-3, 1e-3)
        nn.init.uniform_(final_linear.bias, -1e-3, 1e-3)

    @property
    def compiled_full_forward_enabled(self) -> bool:
        return self._compiled_full_forward is not None

    @property
    def compiled_full_forward_config(self) -> dict[str, object] | None:
        if self._compiled_full_forward_config is None:
            return None
        return dict(self._compiled_full_forward_config)

    def _compute_full_forward_tensors(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        clean_zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        valid_token_mask: torch.Tensor,
        token_pair_relations: torch.Tensor,
        relation_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = self.zone_set_encoder._compute_policy_context_tensors(
            ego_features,
            goal_features,
            clean_zone_features,
            presence_mask,
            valid_token_mask,
            token_pair_relations,
            relation_pair_mask,
        )
        return self.head(context) * self.action_limit

    def enable_compiled_full_forward(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> None:
        if self.compiled_full_forward_enabled:
            raise RuntimeError('ANN actor full forward is already compiled.')
        if self.zone_set_encoder.compiled_tensor_forward_enabled:
            raise RuntimeError(
                'ANN actor full forward cannot nest a compiled encoder.'
            )
        self._compiled_full_forward = torch.compile(
            self._compute_full_forward_tensors,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        self._compiled_full_forward_config = {
            'backend': backend,
            'mode': mode,
            'fullgraph': bool(fullgraph),
            'dynamic': bool(dynamic),
        }

    @contextmanager
    def eager_full_forward(self) -> Iterator[None]:
        previous = self._force_eager_full_forward
        self._force_eager_full_forward = True
        try:
            with self.zone_set_encoder.eager_tensor_forward():
                yield
        finally:
            self._force_eager_full_forward = previous

    def forward(
        self,
        observation: V2ObservationBatch,
        *,
        shared_relations: ZoneSetSharedRelations | None = None,
        profile_sections: bool = False,
    ) -> torch.Tensor:
        if not isinstance(observation, V2ObservationBatch):
            raise TypeError('observation must be a V2ObservationBatch.')
        if (
            self._compiled_full_forward is not None
            and not self._force_eager_full_forward
            and not profile_sections
            and (
                torch.is_grad_enabled()
                or not any(parameter.requires_grad for parameter in self.parameters())
            )
        ):
            tensor_arguments = self.zone_set_encoder.prepare_tensor_forward_arguments(
                observation.ego_features,
                observation.goal_features,
                observation.zone_features,
                observation.presence_mask,
                shared_relations=shared_relations,
            )
            return self._compiled_full_forward(*tensor_arguments)
        context = self.zone_set_encoder(
            observation.ego_features,
            observation.goal_features,
            observation.zone_features,
            observation.presence_mask,
            shared_relations=shared_relations,
            profile_sections=profile_sections,
        )
        return self.head(context) * self.action_limit


class V2ANNCritic(nn.Module):
    """ANN Q-network with an independent dynamic zone-set encoder."""

    def __init__(
        self,
        scales: V2ObservationScales,
        action_dim: int,
        hidden_dim: int,
        *,
        uav_radius: float = 0.0,
        encoder_config: ZoneSetEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(scales, V2ObservationScales):
            raise TypeError('scales must be a V2ObservationScales instance.')
        self.action_dim = _positive_int(action_dim, name='action_dim')
        self.hidden_dim = _positive_int(hidden_dim, name='hidden_dim')
        if encoder_config is None:
            encoder_config = ZoneSetEncoderConfig()
        if not isinstance(encoder_config, ZoneSetEncoderConfig):
            raise TypeError('encoder_config must be a ZoneSetEncoderConfig.')

        self.scales = scales
        self.encoder_config = encoder_config
        self.uav_radius = float(uav_radius)
        self.zone_set_encoder = ZoneSetEncoder(
            scales,
            uav_radius=uav_radius,
            config=encoder_config,
        )
        self.head = nn.Sequential(
            nn.Linear(
                self.zone_set_encoder.output_dim + self.action_dim,
                self.hidden_dim,
            ),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_dim, 1),
        )
        _reset_linear_layers(self.head)

    def encode_context(
        self,
        observation: V2ObservationBatch,
        *,
        shared_relations: ZoneSetSharedRelations | None = None,
        profile_sections: bool = False,
    ) -> torch.Tensor:
        if not isinstance(observation, V2ObservationBatch):
            raise TypeError('observation must be a V2ObservationBatch.')
        return self.zone_set_encoder(
            observation.ego_features,
            observation.goal_features,
            observation.zone_features,
            observation.presence_mask,
            shared_relations=shared_relations,
            profile_sections=profile_sections,
        )

    def _compute_q_tensors(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(torch.cat((context, action), dim=-1))

    def _compute_full_forward_tensors(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        clean_zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        valid_token_mask: torch.Tensor,
        token_pair_relations: torch.Tensor,
        relation_pair_mask: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        context = self.zone_set_encoder._compute_policy_context_tensors(
            ego_features,
            goal_features,
            clean_zone_features,
            presence_mask,
            valid_token_mask,
            token_pair_relations,
            relation_pair_mask,
        )
        return self._compute_q_tensors(context, action)

    def forward_from_context(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(context, torch.Tensor):
            raise TypeError('context must be a torch.Tensor.')
        if context.dtype != torch.float32:
            raise TypeError('context must have dtype torch.float32.')
        expected_context_shape = (action.shape[0], self.zone_set_encoder.output_dim)
        if context.shape != expected_context_shape:
            raise ValueError(
                f'context must have shape {expected_context_shape}; '
                f'got {tuple(context.shape)}.'
            )
        return self._compute_q_tensors(context, action)

    def forward(
        self,
        observation: V2ObservationBatch,
        action: torch.Tensor,
        *,
        shared_relations: ZoneSetSharedRelations | None = None,
        profile_sections: bool = False,
    ) -> torch.Tensor:
        if not isinstance(observation, V2ObservationBatch):
            raise TypeError('observation must be a V2ObservationBatch.')
        if not isinstance(action, torch.Tensor):
            raise TypeError('action must be a torch.Tensor.')
        if action.dtype != torch.float32:
            raise TypeError('action must have dtype torch.float32.')
        expected_shape = (observation.batch_size, self.action_dim)
        if action.shape != expected_shape:
            raise ValueError(
                f'action must have shape {expected_shape}; got {tuple(action.shape)}.'
            )
        if action.device != observation.ego_features.device:
            raise ValueError('action and observation must be on the same device.')
        context = self.encode_context(
            observation,
            shared_relations=shared_relations,
            profile_sections=profile_sections,
        )
        return self._compute_q_tensors(context, action)
