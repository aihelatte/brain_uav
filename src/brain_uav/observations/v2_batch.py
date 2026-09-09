"""Dynamic PyTorch batching for structured V2 observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .v2_contract import (
    EGO_FEATURE_DIM,
    GOAL_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    V2Observation,
)


def _require_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor.')
    return value


@dataclass(frozen=True, slots=True, eq=False)
class V2ObservationBatch:
    """Validated batch with padding only along the dynamic zone axis."""

    ego_features: torch.Tensor
    goal_features: torch.Tensor
    zone_features: torch.Tensor
    presence_mask: torch.Tensor

    def __post_init__(self) -> None:
        ego = _require_tensor(self.ego_features, name='ego_features')
        goal = _require_tensor(self.goal_features, name='goal_features')
        zones = _require_tensor(self.zone_features, name='zone_features')
        mask = _require_tensor(self.presence_mask, name='presence_mask')

        if ego.ndim != 2 or ego.shape[1] != EGO_FEATURE_DIM:
            raise ValueError(
                f'ego_features must have shape (B, {EGO_FEATURE_DIM}); '
                f'got {tuple(ego.shape)}.'
            )
        batch_size = int(ego.shape[0])
        if batch_size <= 0:
            raise ValueError('V2ObservationBatch batch size must be greater than zero.')
        if goal.shape != (batch_size, GOAL_FEATURE_DIM):
            raise ValueError(
                f'goal_features must have shape ({batch_size}, {GOAL_FEATURE_DIM}); '
                f'got {tuple(goal.shape)}.'
            )
        if zones.ndim != 3 or zones.shape[0] != batch_size or zones.shape[2] != ZONE_FEATURE_DIM:
            raise ValueError(
                f'zone_features must have shape (B, N, {ZONE_FEATURE_DIM}); '
                f'got {tuple(zones.shape)}.'
            )
        max_zone_count = int(zones.shape[1])
        if mask.shape != (batch_size, max_zone_count):
            raise ValueError(
                f'presence_mask must have shape ({batch_size}, {max_zone_count}); '
                f'got {tuple(mask.shape)}.'
            )

        for name, tensor in (
            ('ego_features', ego),
            ('goal_features', goal),
            ('zone_features', zones),
        ):
            if tensor.dtype != torch.float32:
                raise TypeError(f'{name} must have dtype torch.float32.')
        if mask.dtype != torch.bool:
            raise TypeError('presence_mask must have dtype torch.bool.')

        device = ego.device
        for name, tensor in (
            ('goal_features', goal),
            ('zone_features', zones),
            ('presence_mask', mask),
        ):
            if tensor.device != device:
                raise ValueError(
                    f'All batch tensors must share one device; {name} is on '
                    f'{tensor.device}, expected {device}.'
                )

    def validate_finite(self) -> None:
        """Explicitly reject non-finite feature data.

        Calling this method for CUDA tensors synchronizes the host with the
        device. Use it at input boundaries, during debugging, or in tests;
        do not call it on every latency-sensitive inference step.
        """

        for name, tensor in (
            ('ego_features', self.ego_features),
            ('goal_features', self.goal_features),
            ('zone_features', self.zone_features),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'{name} must contain only finite values.')

    @property
    def batch_size(self) -> int:
        return int(self.ego_features.shape[0])

    @property
    def max_zone_count(self) -> int:
        return int(self.zone_features.shape[1])

    def to(self, device: torch.device | str) -> V2ObservationBatch:
        """Return a new validated batch on the requested device."""

        target = torch.device(device)
        return V2ObservationBatch(
            ego_features=self.ego_features.to(target),
            goal_features=self.goal_features.to(target),
            zone_features=self.zone_features.to(target),
            presence_mask=self.presence_mask.to(target),
        )


def collate_v2_observations(
    observations: Sequence[V2Observation],
    *,
    device: torch.device | str | None = None,
) -> V2ObservationBatch:
    """Pad observations to this batch's largest real zone count."""

    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise TypeError('observations must be a non-empty sequence of V2Observation objects.')
    if len(observations) == 0:
        raise ValueError('observations must not be empty.')
    for index, observation in enumerate(observations):
        if not isinstance(observation, V2Observation):
            raise TypeError(f'observations[{index}] must be a V2Observation.')

    batch_size = len(observations)
    max_zone_count = max(
        int(observation.zone_features.shape[0])
        for observation in observations
    )
    ego_features = torch.from_numpy(
        np.stack([observation.ego_features for observation in observations]).copy()
    )
    goal_features = torch.from_numpy(
        np.stack([observation.goal_features for observation in observations]).copy()
    )
    zone_features = torch.zeros(
        (batch_size, max_zone_count, ZONE_FEATURE_DIM),
        dtype=torch.float32,
    )
    presence_mask = torch.zeros(
        (batch_size, max_zone_count),
        dtype=torch.bool,
    )
    for batch_index, observation in enumerate(observations):
        zone_count = int(observation.zone_features.shape[0])
        if zone_count == 0:
            continue
        zone_features[batch_index, :zone_count] = torch.from_numpy(
            observation.zone_features.copy()
        )
        presence_mask[batch_index, :zone_count] = torch.from_numpy(
            observation.presence_mask.copy()
        )

    cpu_batch = V2ObservationBatch(
        ego_features=ego_features,
        goal_features=goal_features,
        zone_features=zone_features,
        presence_mask=presence_mask,
    )
    if device is None:
        return cpu_batch
    return cpu_batch.to(device)
