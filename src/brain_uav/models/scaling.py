"""Fixed observation scaling for mixed-unit physical features."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import ScenarioConfig


class FixedObsScaler(nn.Module):
    """Scale mixed-unit observations by fixed physical units."""

    def __init__(self, scenario: ScenarioConfig, state_dim: int, clip_value: float = 5.0) -> None:
        super().__init__()
        nearest_zone_count = scenario.nearest_zone_count
        expected_state_dim = 5 + 3 + 4 + 4 * nearest_zone_count
        if state_dim != expected_state_dim:
            raise ValueError(
                f'FixedObsScaler expected state_dim={expected_state_dim} for nearest_zone_count='
                f'{nearest_zone_count}, got {state_dim}.'
            )

        world_xy = self._positive(float(scenario.world_xy), 'world_xy')
        world_z_max = self._positive(float(scenario.world_z_max), 'world_z_max')
        gamma_max = self._positive(float(scenario.gamma_max), 'gamma_max')
        step_distance = self._positive(float(scenario.speed * scenario.dt), 'speed * dt')
        radius_scale = self._positive(float(scenario.no_fly_radius_range[1]), 'no_fly_radius_range[1]')
        angle_scale = math.pi

        scale_values = [
            world_xy,
            world_xy,
            world_z_max,
            gamma_max,
            angle_scale,
            world_xy,
            world_xy,
            world_z_max,
            world_z_max,
            world_z_max,
            step_distance,
            1.0,
        ]
        for _ in range(nearest_zone_count):
            scale_values.extend([world_xy, world_xy, radius_scale, world_z_max])

        scale = torch.tensor(scale_values, dtype=torch.float32)
        if torch.any(scale <= 0.0):
            raise ValueError('FixedObsScaler scale values must all be positive.')
        self.register_buffer('scale', scale)
        self.clip_value = float(clip_value)

    @staticmethod
    def _positive(value: float, name: str) -> float:
        if value <= 0.0:
            raise ValueError(f'{name} must be positive for FixedObsScaler, got {value}.')
        return value

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        scaled = obs / self.scale.to(device=obs.device)
        return torch.clamp(scaled, -self.clip_value, self.clip_value)
