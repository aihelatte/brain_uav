"""Relation-aware encoder for dynamic sets of V2 no-fly zones."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Callable, Iterator

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function

from brain_uav.observations import (
    EGO_FEATURE_DIM,
    GOAL_FEATURE_DIM,
    PAIR_RELATION_FEATURE_DIM,
    PairRelationBuilder,
    V2ObservationScales,
    ZONE_FEATURE_DIM,
)


@dataclass(frozen=True, slots=True)
class ZoneSetEncoderConfig:
    """Validated architecture settings with fixed V2 input dimensions."""

    ego_dim: int = EGO_FEATURE_DIM
    goal_dim: int = GOAL_FEATURE_DIM
    zone_dim: int = ZONE_FEATURE_DIM
    relation_dim: int = PAIR_RELATION_FEATURE_DIM
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            'ego_dim',
            'goal_dim',
            'zone_dim',
            'relation_dim',
            'hidden_dim',
            'num_heads',
            'num_layers',
            'ffn_dim',
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        fixed_dimensions = (
            ('ego_dim', self.ego_dim, EGO_FEATURE_DIM),
            ('goal_dim', self.goal_dim, GOAL_FEATURE_DIM),
            ('zone_dim', self.zone_dim, ZONE_FEATURE_DIM),
            ('relation_dim', self.relation_dim, PAIR_RELATION_FEATURE_DIM),
        )
        for name, value, expected in fixed_dimensions:
            if value != expected:
                raise ValueError(
                    f'{name} is fixed by the V2 contract at {expected}; got {value}.'
                )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads.')
        try:
            dropout = float(self.dropout)
        except (TypeError, ValueError) as exc:
            raise ValueError('dropout must be exactly 0.0 in the first V2 encoder.') from exc
        if not isfinite(dropout) or dropout != 0.0:
            raise ValueError('dropout must be exactly 0.0 in the first V2 encoder.')
        object.__setattr__(self, 'dropout', dropout)


class ZoneEncoder(nn.Module):
    """Shared per-zone MLP with no slot-specific parameters."""

    def __init__(self, config: ZoneSetEncoderConfig) -> None:
        super().__init__()
        if not isinstance(config, ZoneSetEncoderConfig):
            raise TypeError('config must be a ZoneSetEncoderConfig.')
        self.net = nn.Sequential(
            nn.Linear(config.zone_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, zone_features: torch.Tensor) -> torch.Tensor:
        return self.net(zone_features)


class TaskEncoder(nn.Module):
    """Encode UAV ego state and goal features into a task query."""

    def __init__(self, config: ZoneSetEncoderConfig) -> None:
        super().__init__()
        if not isinstance(config, ZoneSetEncoderConfig):
            raise TypeError('config must be a ZoneSetEncoderConfig.')
        self.net = nn.Sequential(
            nn.Linear(config.ego_dim + config.goal_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat((ego_features, goal_features), dim=-1))


class RelationAwareSelfAttention(nn.Module):
    """Masked multi-head attention with learned pair bias and pair value."""

    def __init__(self, config: ZoneSetEncoderConfig) -> None:
        super().__init__()
        if not isinstance(config, ZoneSetEncoderConfig):
            raise TypeError('config must be a ZoneSetEncoderConfig.')
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.score_scale = 1.0 / sqrt(float(self.head_dim))
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.relation_bias = nn.Linear(config.relation_dim, config.num_heads)
        self.relation_value = nn.Linear(config.relation_dim, config.hidden_dim)
        self.aggregate_relation_values_first = False

    def enable_relation_value_aggregation(self) -> None:
        self.aggregate_relation_values_first = True

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = value.shape
        return value.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_token_mask: torch.Tensor,
        pair_relations: torch.Tensor,
        relation_pair_mask: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, hidden_dim = tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f'tokens last dimension must be {self.hidden_dim}; got {hidden_dim}.'
            )
        if valid_token_mask.shape != (batch_size, token_count):
            raise ValueError('valid_token_mask shape must match the token axes.')
        expected_pair_shape = (
            batch_size,
            token_count,
            token_count,
            self.relation_bias.in_features,
        )
        if pair_relations.shape != expected_pair_shape:
            raise ValueError(
                f'pair_relations must have shape {expected_pair_shape}; '
                f'got {tuple(pair_relations.shape)}.'
            )
        if relation_pair_mask.shape != (
            batch_size,
            token_count,
            token_count,
        ):
            raise ValueError('relation_pair_mask shape must match both token axes.')
        if valid_token_mask.dtype != torch.bool or relation_pair_mask.dtype != torch.bool:
            raise TypeError('Attention masks must have dtype torch.bool.')

        clean_relations = torch.where(
            relation_pair_mask.unsqueeze(-1),
            pair_relations,
            torch.zeros_like(pair_relations),
        )
        qkv_weight = torch.cat(
            (self.query.weight, self.key.weight, self.value.weight),
            dim=0,
        )
        qkv_bias = torch.cat(
            (self.query.bias, self.key.bias, self.value.bias),
            dim=0,
        )
        queries, keys, values = F.linear(tokens, qkv_weight, qkv_bias).split(
            self.hidden_dim,
            dim=-1,
        )
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        if self.aggregate_relation_values_first:
            relation_bias = self.relation_bias(clean_relations)
        else:
            relation_weight = torch.cat(
                (self.relation_bias.weight, self.relation_value.weight),
                dim=0,
            )
            relation_projection_bias = torch.cat(
                (self.relation_bias.bias, self.relation_value.bias),
                dim=0,
            )
            relation_bias, relation_values = F.linear(
                clean_relations,
                relation_weight,
                relation_projection_bias,
            ).split(
                (
                    self.relation_bias.out_features,
                    self.relation_value.out_features,
                ),
                dim=-1,
            )
        relation_bias = relation_bias.permute(0, 3, 1, 2)
        relation_bias = relation_bias * relation_pair_mask.unsqueeze(1).to(
            relation_bias.dtype
        )
        if not self.aggregate_relation_values_first:
            relation_values = relation_values.reshape(
                batch_size,
                token_count,
                token_count,
                self.num_heads,
                self.head_dim,
            ).permute(0, 3, 1, 2, 4)
            relation_values = relation_values * relation_pair_mask[
                :, None, :, :, None
            ].to(relation_values.dtype)

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores * self.score_scale + relation_bias
        key_mask = valid_token_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attention_weights = torch.softmax(scores, dim=-1)
        valid_attention = (
            valid_token_mask[:, None, :, None]
            & valid_token_mask[:, None, None, :]
        )
        attention_weights = torch.where(
            valid_attention,
            attention_weights,
            torch.zeros_like(attention_weights),
        )
        normalizer = attention_weights.sum(dim=-1, keepdim=True)
        attention_weights = torch.where(
            valid_token_mask[:, None, :, None],
            attention_weights / normalizer.clamp_min(
                torch.finfo(attention_weights.dtype).tiny
            ),
            torch.zeros_like(attention_weights),
        )

        standard_messages = torch.matmul(attention_weights, values)
        if self.aggregate_relation_values_first:
            weighted_relations = attention_weights * relation_pair_mask[
                :, None, :, :
            ].to(attention_weights.dtype)
            relation_sum = torch.einsum(
                'bhij,bijr->bhir', weighted_relations, clean_relations,
            )
            relation_messages = torch.einsum(
                'bhir,hdr->bhid',
                relation_sum,
                self.relation_value.weight.reshape(
                    self.num_heads, self.head_dim, self.relation_value.in_features,
                ),
            )
            relation_messages = relation_messages + (
                weighted_relations.sum(dim=-1, keepdim=True)
                * self.relation_value.bias.reshape(self.num_heads, self.head_dim)[
                    None, :, None, :
                ]
            )
        else:
            relation_messages = torch.einsum(
                'bhij,bhijd->bhid',
                attention_weights,
                relation_values,
            )
        contextual = standard_messages + relation_messages
        contextual = contextual.transpose(1, 2).contiguous().reshape(
            batch_size,
            token_count,
            self.hidden_dim,
        )
        contextual = self.output(contextual)
        contextual = contextual * valid_token_mask.unsqueeze(-1).to(contextual.dtype)
        if return_attention_weights:
            return contextual, attention_weights
        return contextual


class RelationAttentionBlock(nn.Module):
    """Pre-LayerNorm relation-aware attention followed by an FFN."""

    def __init__(self, config: ZoneSetEncoderConfig) -> None:
        super().__init__()
        if not isinstance(config, ZoneSetEncoderConfig):
            raise TypeError('config must be a ZoneSetEncoderConfig.')
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = RelationAwareSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.hidden_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        valid_token_mask: torch.Tensor,
        pair_relations: torch.Tensor,
        relation_pair_mask: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention_result = self.attention(
            self.attention_norm(tokens),
            valid_token_mask,
            pair_relations,
            relation_pair_mask,
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            attention_output, weights = attention_result
        else:
            attention_output = attention_result
            weights = None
        mask = valid_token_mask.unsqueeze(-1).to(tokens.dtype)
        tokens = (tokens + attention_output) * mask
        tokens = (tokens + self.ffn(self.ffn_norm(tokens))) * mask
        if return_attention_weights:
            return tokens, weights
        return tokens


class TaskConditionedPooling(nn.Module):
    """Attention-pool contextual tokens using the encoded UAV task."""

    def __init__(self, config: ZoneSetEncoderConfig) -> None:
        super().__init__()
        if not isinstance(config, ZoneSetEncoderConfig):
            raise TypeError('config must be a ZoneSetEncoderConfig.')
        self.hidden_dim = config.hidden_dim
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.score_scale = 1.0 / sqrt(float(config.hidden_dim))

    def forward(
        self,
        task_embedding: torch.Tensor,
        contextual_tokens: torch.Tensor,
        valid_token_mask: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query = self.query(task_embedding).unsqueeze(1)
        key_value_weight = torch.cat((self.key.weight, self.value.weight), dim=0)
        key_value_bias = torch.cat((self.key.bias, self.value.bias), dim=0)
        keys, values = F.linear(
            contextual_tokens,
            key_value_weight,
            key_value_bias,
        ).split(
            (self.key.out_features, self.value.out_features),
            dim=-1,
        )
        scores = torch.sum(query * keys, dim=-1) * self.score_scale
        scores = scores.masked_fill(
            ~valid_token_mask,
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(
            valid_token_mask,
            weights,
            torch.zeros_like(weights),
        )
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        summary = torch.sum(weights.unsqueeze(-1) * values, dim=1)
        if return_attention_weights:
            return summary, weights
        return summary


@dataclass(frozen=True, slots=True)
class ZoneSetEncoderDiagnostics:
    """Optional tensors used to validate attention and masking behavior."""

    policy_context: torch.Tensor
    pair_relations: torch.Tensor
    contextual_zone_tokens: torch.Tensor
    self_attention_weights: tuple[torch.Tensor, ...]
    pooling_weights: torch.Tensor
    valid_token_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class ZoneSetSharedRelations:
    """Parameter-free relation tensors bound to one exact observation batch."""

    scales: V2ObservationScales
    uav_radius: float
    ego_features: torch.Tensor
    goal_features: torch.Tensor
    zone_features: torch.Tensor
    presence_mask: torch.Tensor
    clean_zone_features: torch.Tensor
    pair_relations: torch.Tensor
    valid_token_mask: torch.Tensor
    token_pair_relations: torch.Tensor
    relation_pair_mask: torch.Tensor


class ZoneSetEncoder(nn.Module):
    """Convert a dynamic V2 no-fly-zone set into a fixed policy context."""

    def __init__(
        self,
        scales: V2ObservationScales,
        *,
        uav_radius: float = 0.0,
        config: ZoneSetEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(scales, V2ObservationScales):
            raise TypeError('scales must be a V2ObservationScales instance.')
        if config is None:
            config = ZoneSetEncoderConfig()
        if not isinstance(config, ZoneSetEncoderConfig):
            raise TypeError('config must be a ZoneSetEncoderConfig.')
        self.scales = scales
        self.uav_radius = float(uav_radius)
        self.config = config
        self.pair_relation_builder = PairRelationBuilder(
            scales,
            uav_radius=uav_radius,
        )
        self.zone_encoder = ZoneEncoder(config)
        self.task_encoder = TaskEncoder(config)
        self.layers = nn.ModuleList(
            RelationAttentionBlock(config)
            for _ in range(config.num_layers)
        )
        self.pooling = TaskConditionedPooling(config)
        self.empty_scene_token = nn.Parameter(
            torch.empty(1, 1, config.hidden_dim)
        )
        nn.init.normal_(self.empty_scene_token, mean=0.0, std=0.02)
        self._compiled_tensor_forward: Callable[..., torch.Tensor] | None = None
        self._compiled_tensor_forward_config: dict[str, object] | None = None
        self._force_eager_tensor_forward = False
        self._compiled_shared_relations: Callable[..., tuple[torch.Tensor, ...]] | None = None
        self._compiled_shared_relations_config: dict[str, object] | None = None

    @property
    def output_dim(self) -> int:
        return 2 * self.config.hidden_dim

    @property
    def compiled_tensor_forward_enabled(self) -> bool:
        return self._compiled_tensor_forward is not None

    @property
    def compiled_tensor_forward_config(self) -> dict[str, object] | None:
        if self._compiled_tensor_forward_config is None:
            return None
        return dict(self._compiled_tensor_forward_config)

    @property
    def compiled_shared_relations_enabled(self) -> bool:
        return self._compiled_shared_relations is not None

    @property
    def compiled_shared_relations_config(self) -> dict[str, object] | None:
        if self._compiled_shared_relations_config is None:
            return None
        return dict(self._compiled_shared_relations_config)

    def enable_compiled_shared_relations(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> None:
        if self.compiled_shared_relations_enabled:
            raise RuntimeError('ZoneSetEncoder shared relations are already compiled.')
        self._compiled_shared_relations = torch.compile(
            self._compute_shared_relation_tensors,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        self._compiled_shared_relations_config = {
            'backend': backend,
            'mode': mode,
            'fullgraph': bool(fullgraph),
            'dynamic': bool(dynamic),
        }

    def enable_compiled_tensor_forward(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> None:
        if self.compiled_tensor_forward_enabled:
            raise RuntimeError('ZoneSetEncoder tensor forward is already compiled.')
        compiled = torch.compile(
            self._compute_policy_context_tensors,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        self._compiled_tensor_forward = compiled
        self._compiled_tensor_forward_config = {
            'backend': backend,
            'mode': mode,
            'fullgraph': bool(fullgraph),
            'dynamic': bool(dynamic),
        }

    @contextmanager
    def eager_tensor_forward(self) -> Iterator[None]:
        """Temporarily select the original tensor path without disabling compile."""

        previous = self._force_eager_tensor_forward
        self._force_eager_tensor_forward = True
        try:
            yield
        finally:
            self._force_eager_tensor_forward = previous

    def _validate_inputs(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> tuple[int, int]:
        for name, value in (
            ('ego_features', ego_features),
            ('goal_features', goal_features),
            ('zone_features', zone_features),
            ('presence_mask', presence_mask),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'{name} must be a torch.Tensor.')
        for name, value in (
            ('ego_features', ego_features),
            ('goal_features', goal_features),
            ('zone_features', zone_features),
        ):
            if value.dtype != torch.float32:
                raise TypeError(f'{name} must have dtype torch.float32.')
        if presence_mask.dtype != torch.bool:
            raise TypeError('presence_mask must have dtype torch.bool.')
        if ego_features.ndim != 2 or ego_features.shape[1] != self.config.ego_dim:
            raise ValueError(
                f'ego_features must have shape (B, {self.config.ego_dim}); '
                f'got {tuple(ego_features.shape)}.'
            )
        batch_size = int(ego_features.shape[0])
        if batch_size <= 0:
            raise ValueError('Batch size must be greater than zero.')
        if goal_features.shape != (batch_size, self.config.goal_dim):
            raise ValueError(
                f'goal_features must have shape ({batch_size}, '
                f'{self.config.goal_dim}); got {tuple(goal_features.shape)}.'
            )
        if (
            zone_features.ndim != 3
            or zone_features.shape[0] != batch_size
            or zone_features.shape[2] != self.config.zone_dim
        ):
            raise ValueError(
                f'zone_features must have shape (B, N, {self.config.zone_dim}); '
                f'got {tuple(zone_features.shape)}.'
            )
        zone_count = int(zone_features.shape[1])
        if presence_mask.shape != (batch_size, zone_count):
            raise ValueError(
                f'presence_mask must have shape ({batch_size}, {zone_count}); '
                f'got {tuple(presence_mask.shape)}.'
            )
        device = ego_features.device
        for name, value in (
            ('goal_features', goal_features),
            ('zone_features', zone_features),
            ('presence_mask', presence_mask),
        ):
            if value.device != device:
                raise ValueError(
                    f'All encoder inputs must share one device; {name} is on '
                    f'{value.device}, expected {device}.'
                )
        if self.empty_scene_token.device != device:
            raise ValueError('ZoneSetEncoder parameters and inputs must share one device.')
        return batch_size, zone_count

    @staticmethod
    def _validate_finite_inputs(
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        zone_features: torch.Tensor,
    ) -> None:
        for name, value in (
            ('ego_features', ego_features),
            ('goal_features', goal_features),
            ('zone_features', zone_features),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must contain only finite values.')

    def _build_shared_relations_validated(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        *,
        batch_size: int,
        zone_count: int,
    ) -> ZoneSetSharedRelations:
        tensor_build = (
            self._compiled_shared_relations
            if (
                self._compiled_shared_relations is not None
                and not self._force_eager_tensor_forward
            )
            else self._compute_shared_relation_tensors
        )
        (
            clean_zone_features,
            pair_relations,
            valid_token_mask,
            token_pair_relations,
            relation_pair_mask,
        ) = tensor_build(ego_features, zone_features, presence_mask)
        return ZoneSetSharedRelations(
            scales=self.scales,
            uav_radius=self.uav_radius,
            ego_features=ego_features,
            goal_features=goal_features,
            zone_features=zone_features,
            presence_mask=presence_mask,
            clean_zone_features=clean_zone_features,
            pair_relations=pair_relations,
            valid_token_mask=valid_token_mask,
            token_pair_relations=token_pair_relations,
            relation_pair_mask=relation_pair_mask,
        )

    def _compute_shared_relation_tensors(
        self,
        ego_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_zone_features = torch.where(
            presence_mask.unsqueeze(-1),
            zone_features,
            torch.zeros_like(zone_features),
        )
        pair_relations = self.pair_relation_builder.compute_relations(
            ego_features,
            clean_zone_features,
            presence_mask,
        )
        empty_valid = ~presence_mask.any(dim=1)
        valid_token_mask = torch.cat(
            (empty_valid.unsqueeze(1), presence_mask),
            dim=1,
        )
        token_pair_relations = F.pad(
            pair_relations,
            (0, 0, 1, 0, 1, 0),
            value=0.0,
        )
        zone_pair_mask = (
            presence_mask.unsqueeze(1)
            & presence_mask.unsqueeze(2)
            & ~torch.eye(
                zone_features.shape[1],
                dtype=torch.bool,
                device=presence_mask.device,
            ).unsqueeze(0)
        )
        relation_pair_mask = F.pad(
            zone_pair_mask,
            (1, 0, 1, 0),
            value=False,
        )
        return (
            clean_zone_features,
            pair_relations,
            valid_token_mask,
            token_pair_relations,
            relation_pair_mask,
        )

    def build_shared_relations(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        *,
        profile_sections: bool = False,
    ) -> ZoneSetSharedRelations:
        batch_size, zone_count = self._validate_inputs(
            ego_features,
            goal_features,
            zone_features,
            presence_mask,
        )
        if profile_sections:
            with record_function('v2_encoder.shared_relation_build'):
                return self._build_shared_relations_validated(
                    ego_features,
                    goal_features,
                    zone_features,
                    presence_mask,
                    batch_size=batch_size,
                    zone_count=zone_count,
                )
        return self._build_shared_relations_validated(
            ego_features,
            goal_features,
            zone_features,
            presence_mask,
            batch_size=batch_size,
            zone_count=zone_count,
        )

    def _validate_shared_relations(
        self,
        shared: ZoneSetSharedRelations,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> None:
        if not isinstance(shared, ZoneSetSharedRelations):
            raise TypeError('shared_relations must be a ZoneSetSharedRelations.')
        if shared.scales != self.scales or shared.uav_radius != self.uav_radius:
            raise ValueError('shared_relations fixed geometry configuration is incompatible.')
        expected_inputs = (
            ego_features,
            goal_features,
            zone_features,
            presence_mask,
        )
        actual_inputs = (
            shared.ego_features,
            shared.goal_features,
            shared.zone_features,
            shared.presence_mask,
        )
        if any(actual is not expected for actual, expected in zip(
            actual_inputs, expected_inputs
        )):
            raise ValueError('shared_relations belongs to a different observation batch.')
        batch_size = int(ego_features.shape[0])
        zone_count = int(zone_features.shape[1])
        expected = (
            (shared.clean_zone_features, zone_features.shape, torch.float32),
            (
                shared.pair_relations,
                (batch_size, zone_count, zone_count, self.config.relation_dim),
                torch.float32,
            ),
            (shared.valid_token_mask, (batch_size, zone_count + 1), torch.bool),
            (
                shared.token_pair_relations,
                (batch_size, zone_count + 1, zone_count + 1, self.config.relation_dim),
                torch.float32,
            ),
            (
                shared.relation_pair_mask,
                (batch_size, zone_count + 1, zone_count + 1),
                torch.bool,
            ),
        )
        for value, shape, dtype in expected:
            if value.shape != shape or value.dtype != dtype:
                raise ValueError('shared_relations tensor contract is incompatible.')
            if value.device != ego_features.device:
                raise ValueError('shared_relations tensors must share the input device.')

    def _encode_zone_task_tensors(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        clean_zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        valid_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zone_tokens = self.zone_encoder(clean_zone_features)
        zone_tokens = zone_tokens * presence_mask.unsqueeze(-1).to(
            zone_tokens.dtype
        )
        task_embedding = self.task_encoder(ego_features, goal_features)
        empty_tokens = self.empty_scene_token.expand(
            ego_features.shape[0],
            -1,
            -1,
        )
        tokens = torch.cat((empty_tokens, zone_tokens), dim=1)
        tokens = tokens * valid_token_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, task_embedding

    def _apply_relation_attention_tensors(
        self,
        tokens: torch.Tensor,
        valid_token_mask: torch.Tensor,
        token_pair_relations: torch.Tensor,
        relation_pair_mask: torch.Tensor,
        *,
        diagnostics: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        attention_weights: list[torch.Tensor] = []
        for layer in self.layers:
            layer_result = layer(
                tokens,
                valid_token_mask,
                token_pair_relations,
                relation_pair_mask,
                return_attention_weights=diagnostics,
            )
            if diagnostics:
                tokens, weights = layer_result
                attention_weights.append(weights)
            else:
                tokens = layer_result
        return tokens, tuple(attention_weights)

    def _pool_policy_context_tensors(
        self,
        task_embedding: torch.Tensor,
        tokens: torch.Tensor,
        valid_token_mask: torch.Tensor,
        *,
        diagnostics: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pooling_result = self.pooling(
            task_embedding,
            tokens,
            valid_token_mask,
            return_attention_weights=diagnostics,
        )
        if diagnostics:
            zone_summary, pooling_weights = pooling_result
        else:
            zone_summary = pooling_result
            pooling_weights = None
        return torch.cat((task_embedding, zone_summary), dim=-1), pooling_weights

    def _compute_policy_context_tensors(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        clean_zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        valid_token_mask: torch.Tensor,
        token_pair_relations: torch.Tensor,
        relation_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens, task_embedding = self._encode_zone_task_tensors(
            ego_features,
            goal_features,
            clean_zone_features,
            presence_mask,
            valid_token_mask,
        )
        tokens, _ = self._apply_relation_attention_tensors(
            tokens,
            valid_token_mask,
            token_pair_relations,
            relation_pair_mask,
            diagnostics=False,
        )
        policy_context, _ = self._pool_policy_context_tensors(
            task_embedding,
            tokens,
            valid_token_mask,
            diagnostics=False,
        )
        return policy_context

    def prepare_tensor_forward_arguments(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        *,
        shared_relations: ZoneSetSharedRelations | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Validate inputs and expose the parameter-free compiled tensor inputs."""

        batch_size, zone_count = self._validate_inputs(
            ego_features,
            goal_features,
            zone_features,
            presence_mask,
        )
        if shared_relations is None:
            shared_relations = self._build_shared_relations_validated(
                ego_features,
                goal_features,
                zone_features,
                presence_mask,
                batch_size=batch_size,
                zone_count=zone_count,
            )
        else:
            self._validate_shared_relations(
                shared_relations,
                ego_features,
                goal_features,
                zone_features,
                presence_mask,
            )
        return (
            ego_features,
            goal_features,
            shared_relations.clean_zone_features,
            presence_mask,
            shared_relations.valid_token_mask,
            shared_relations.token_pair_relations,
            shared_relations.relation_pair_mask,
        )

    def _forward_impl(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        *,
        diagnostics: bool,
        shared_relations: ZoneSetSharedRelations | None = None,
        profile_sections: bool = False,
    ) -> torch.Tensor | ZoneSetEncoderDiagnostics:
        batch_size, zone_count = self._validate_inputs(
            ego_features,
            goal_features,
            zone_features,
            presence_mask,
        )
        if shared_relations is None:
            if profile_sections:
                with record_function('v2_encoder.shared_relation_build'):
                    shared_relations = self._build_shared_relations_validated(
                        ego_features,
                        goal_features,
                        zone_features,
                        presence_mask,
                        batch_size=batch_size,
                        zone_count=zone_count,
                    )
            else:
                shared_relations = self._build_shared_relations_validated(
                    ego_features,
                    goal_features,
                    zone_features,
                    presence_mask,
                    batch_size=batch_size,
                    zone_count=zone_count,
                )
        else:
            self._validate_shared_relations(
                shared_relations,
                ego_features,
                goal_features,
                zone_features,
                presence_mask,
            )
        pair_relations = shared_relations.pair_relations
        valid_token_mask = shared_relations.valid_token_mask
        token_pair_relations = shared_relations.token_pair_relations
        relation_pair_mask = shared_relations.relation_pair_mask

        tensor_arguments = (
            ego_features,
            goal_features,
            shared_relations.clean_zone_features,
            presence_mask,
            valid_token_mask,
            token_pair_relations,
            relation_pair_mask,
        )
        if not diagnostics and not profile_sections:
            tensor_forward = (
                self._compiled_tensor_forward
                if (
                    self._compiled_tensor_forward is not None
                    and not self._force_eager_tensor_forward
                )
                else self._compute_policy_context_tensors
            )
            return tensor_forward(*tensor_arguments)

        if profile_sections:
            with record_function('v2_encoder.zone_task_encoding'):
                tokens, task_embedding = self._encode_zone_task_tensors(
                    ego_features,
                    goal_features,
                    shared_relations.clean_zone_features,
                    presence_mask,
                    valid_token_mask,
                )
        else:
            tokens, task_embedding = self._encode_zone_task_tensors(
                ego_features,
                goal_features,
                shared_relations.clean_zone_features,
                presence_mask,
                valid_token_mask,
            )

        if profile_sections:
            with record_function('v2_encoder.relation_attention'):
                tokens, attention_weights = self._apply_relation_attention_tensors(
                    tokens,
                    valid_token_mask,
                    token_pair_relations,
                    relation_pair_mask,
                    diagnostics=diagnostics,
                )
        else:
            tokens, attention_weights = self._apply_relation_attention_tensors(
                tokens,
                valid_token_mask,
                token_pair_relations,
                relation_pair_mask,
                diagnostics=diagnostics,
            )

        if profile_sections:
            with record_function('v2_encoder.task_conditioned_pooling'):
                policy_context, pooling_weights = self._pool_policy_context_tensors(
                    task_embedding,
                    tokens,
                    valid_token_mask,
                    diagnostics=diagnostics,
                )
        else:
            policy_context, pooling_weights = self._pool_policy_context_tensors(
                task_embedding,
                tokens,
                valid_token_mask,
                diagnostics=diagnostics,
            )
        if not diagnostics:
            return policy_context
        return ZoneSetEncoderDiagnostics(
            policy_context=policy_context,
            pair_relations=pair_relations,
            contextual_zone_tokens=tokens[:, 1:],
            self_attention_weights=attention_weights,
            pooling_weights=pooling_weights,
            valid_token_mask=valid_token_mask,
        )

    def forward(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        *,
        shared_relations: ZoneSetSharedRelations | None = None,
        profile_sections: bool = False,
    ) -> torch.Tensor:
        return self._forward_impl(
            ego_features,
            goal_features,
            zone_features,
            presence_mask,
            diagnostics=False,
            shared_relations=shared_relations,
            profile_sections=profile_sections,
        )

    def forward_checked(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run ``forward`` with explicit finite-value checks.

        The checks synchronize when tensors are on CUDA, so this entry point
        is intended for input boundaries, debugging, and tests rather than
        latency-sensitive inference loops.
        """

        self._validate_inputs(
            ego_features,
            goal_features,
            zone_features,
            presence_mask,
        )
        self._validate_finite_inputs(
            ego_features,
            goal_features,
            zone_features,
        )
        policy_context = self.forward(
            ego_features,
            goal_features,
            zone_features,
            presence_mask,
        )
        if not bool(torch.isfinite(policy_context).all()):
            raise ValueError('ZoneSetEncoder output must contain only finite values.')
        return policy_context

    def forward_with_diagnostics(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> ZoneSetEncoderDiagnostics:
        return self._forward_impl(
            ego_features,
            goal_features,
            zone_features,
            presence_mask,
            diagnostics=True,
        )
