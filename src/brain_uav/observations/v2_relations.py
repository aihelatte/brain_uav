"""Vectorized pairwise relations for V2 no-fly-zone feature sets."""

from __future__ import annotations

from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import nn

from .v2_contract import (
    EGO_FEATURE_DIM,
    EGO_FEATURE_INDEX,
    ZONE_FEATURE_DIM,
    ZONE_FEATURE_INDEX,
    V2ObservationScales,
)


PAIR_RELATION_FEATURE_NAMES = (
    'delta_forward_norm',
    'delta_right_norm',
    'delta_up_norm',
    'center_distance_norm',
    'gap_forward_norm',
    'gap_right_norm',
    'gap_up_norm',
    'expanded_aabb_clearance_norm',
    'expanded_aabb_overlap',
    'straddles_uav_right_axis',
    'straddles_uav_up_axis',
    'both_block_raw_goal_path',
)
PAIR_RELATION_FEATURE_DIM = 12
PAIR_RELATION_FEATURE_INDEX: Mapping[str, int] = MappingProxyType(
    {
        name: index
        for index, name in enumerate(PAIR_RELATION_FEATURE_NAMES)
    }
)


def _nonnegative_scalar(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'{name} must be finite and greater than or equal to zero.'
        ) from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(
            f'{name} must be finite and greater than or equal to zero.'
        )
    return result


class PairRelationBuilder(nn.Module):
    """Construct conservative expanded-AABB relations without trainable state."""

    def __init__(
        self,
        scales: V2ObservationScales,
        *,
        uav_radius: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(scales, V2ObservationScales):
            raise TypeError('scales must be a V2ObservationScales instance.')
        radius = _nonnegative_scalar(uav_radius, name='uav_radius')
        self.register_buffer(
            'horizontal_span',
            torch.tensor(scales.horizontal_span, dtype=torch.float32),
        )
        self.register_buffer(
            'vertical_span',
            torch.tensor(scales.vertical_span, dtype=torch.float32),
        )
        self.register_buffer(
            'world_diagonal',
            torch.tensor(scales.world_diagonal, dtype=torch.float32),
        )
        self.register_buffer(
            'uav_radius',
            torch.tensor(radius, dtype=torch.float32),
        )

    def _validate_inputs(
        self,
        ego_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> tuple[int, int]:
        for name, value in (
            ('ego_features', ego_features),
            ('zone_features', zone_features),
            ('presence_mask', presence_mask),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'{name} must be a torch.Tensor.')
        if ego_features.dtype != torch.float32:
            raise TypeError('ego_features must have dtype torch.float32.')
        if zone_features.dtype != torch.float32:
            raise TypeError('zone_features must have dtype torch.float32.')
        if presence_mask.dtype != torch.bool:
            raise TypeError('presence_mask must have dtype torch.bool.')
        if ego_features.ndim != 2 or ego_features.shape[1] != EGO_FEATURE_DIM:
            raise ValueError(
                f'ego_features must have shape (B, {EGO_FEATURE_DIM}); '
                f'got {tuple(ego_features.shape)}.'
            )
        batch_size = int(ego_features.shape[0])
        if batch_size <= 0:
            raise ValueError('Batch size must be greater than zero.')
        if (
            zone_features.ndim != 3
            or zone_features.shape[0] != batch_size
            or zone_features.shape[2] != ZONE_FEATURE_DIM
        ):
            raise ValueError(
                f'zone_features must have shape (B, N, {ZONE_FEATURE_DIM}); '
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
            ('zone_features', zone_features),
            ('presence_mask', presence_mask),
        ):
            if value.device != device:
                raise ValueError(
                    f'All inputs must share one device; {name} is on '
                    f'{value.device}, expected {device}.'
                )
        if self.horizontal_span.device != device:
            raise ValueError(
                'PairRelationBuilder buffers and inputs must be on the same device.'
            )
        return batch_size, zone_count

    @staticmethod
    def _validate_finite_inputs(
        ego_features: torch.Tensor,
        zone_features: torch.Tensor,
    ) -> None:
        for name, value in (
            ('ego_features', ego_features),
            ('zone_features', zone_features),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must contain only finite values.')

    def forward(
        self,
        ego_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return pair relations with shape [B, N, N, 12]."""

        batch_size, zone_count = self._validate_inputs(
            ego_features,
            zone_features,
            presence_mask,
        )
        clean_zones = torch.where(
            presence_mask.unsqueeze(-1),
            zone_features,
            torch.zeros_like(zone_features),
        )

        forward = (
            clean_zones[..., ZONE_FEATURE_INDEX['zone_forward_norm']]
            * self.horizontal_span
        )
        right = (
            clean_zones[..., ZONE_FEATURE_INDEX['zone_right_norm']]
            * self.horizontal_span
        )
        up = (
            clean_zones[..., ZONE_FEATURE_INDEX['zone_up_norm']]
            * self.vertical_span
        )
        centers = torch.stack((forward, right, up), dim=-1)

        extent_x = (
            clean_zones[..., ZONE_FEATURE_INDEX['extent_x_norm']]
            * self.horizontal_span
        )
        extent_y = (
            clean_zones[..., ZONE_FEATURE_INDEX['extent_y_norm']]
            * self.horizontal_span
        )
        extent_z = (
            clean_zones[..., ZONE_FEATURE_INDEX['extent_z_norm']]
            * self.vertical_span
        )
        margin = (
            clean_zones[..., ZONE_FEATURE_INDEX['safety_margin_norm']]
            * self.world_diagonal
            + self.uav_radius
        )
        absolute_sin = torch.abs(
            ego_features[:, EGO_FEATURE_INDEX['sin_psi']]
        ).unsqueeze(1)
        absolute_cos = torch.abs(
            ego_features[:, EGO_FEATURE_INDEX['cos_psi']]
        ).unsqueeze(1)
        half_forward = (
            0.5 * (absolute_cos * extent_x + absolute_sin * extent_y)
            + margin
        )
        half_right = (
            0.5 * (absolute_sin * extent_x + absolute_cos * extent_y)
            + margin
        )
        half_up = 0.5 * extent_z + margin
        half_extents = torch.stack(
            (half_forward, half_right, half_up),
            dim=-1,
        )

        delta = centers.unsqueeze(1) - centers.unsqueeze(2)
        gaps = (
            torch.abs(delta)
            - half_extents.unsqueeze(1)
            - half_extents.unsqueeze(2)
        )
        center_distance = torch.linalg.vector_norm(delta, dim=-1)
        expanded_clearance = torch.linalg.vector_norm(
            torch.relu(gaps),
            dim=-1,
        )
        overlap = torch.all(gaps <= 0.0, dim=-1)
        straddles_right = (
            right.unsqueeze(1) * right.unsqueeze(2) < 0.0
        )
        straddles_up = up.unsqueeze(1) * up.unsqueeze(2) < 0.0
        blocks = (
            clean_zones[
                ..., ZONE_FEATURE_INDEX['raw_goal_path_intersects']
            ]
            > 0.5
        )
        both_block = blocks.unsqueeze(1) & blocks.unsqueeze(2)

        relations = torch.stack(
            (
                delta[..., 0] / self.horizontal_span,
                delta[..., 1] / self.horizontal_span,
                delta[..., 2] / self.vertical_span,
                center_distance / self.world_diagonal,
                gaps[..., 0] / self.horizontal_span,
                gaps[..., 1] / self.horizontal_span,
                gaps[..., 2] / self.vertical_span,
                expanded_clearance / self.world_diagonal,
                overlap.to(dtype=torch.float32),
                straddles_right.to(dtype=torch.float32),
                straddles_up.to(dtype=torch.float32),
                both_block.to(dtype=torch.float32),
            ),
            dim=-1,
        )
        pair_mask = (
            presence_mask.unsqueeze(1)
            & presence_mask.unsqueeze(2)
            & ~torch.eye(
                zone_count,
                dtype=torch.bool,
                device=presence_mask.device,
            ).unsqueeze(0)
        )
        relations = relations * pair_mask.unsqueeze(-1).to(relations.dtype)
        if relations.shape != (
            batch_size,
            zone_count,
            zone_count,
            PAIR_RELATION_FEATURE_DIM,
        ):
            raise RuntimeError('Internal pair-relation shape invariant failed.')
        return relations

    def forward_checked(
        self,
        ego_features: torch.Tensor,
        zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run ``forward`` with explicit finite-value checks.

        The checks synchronize when tensors are on CUDA, so this entry point
        is intended for input boundaries, debugging, and tests rather than
        latency-sensitive inference loops.
        """

        self._validate_inputs(ego_features, zone_features, presence_mask)
        self._validate_finite_inputs(ego_features, zone_features)
        relations = self.forward(ego_features, zone_features, presence_mask)
        if not bool(torch.isfinite(relations).all()):
            raise ValueError(
                'PairRelationBuilder output must contain only finite values.'
            )
        return relations
