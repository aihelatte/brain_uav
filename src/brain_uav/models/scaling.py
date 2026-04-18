"""Fixed physical scaling for environment observations."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import ScenarioConfig


class FixedObsScaler(nn.Module):
    """Scale mixed-unit observations by fixed physical units.

    The observation layout follows ``StaticNoFlyTrajectoryEnv._get_obs``:
    own state, relative goal, extra altitude/descent features, then four
    features per nearest no-fly zone.
    """

    def __init__(
        self,
        state_dim: int,
        scenario_cfg: ScenarioConfig | None = None,
        clamp_value: float | None = 5.0,
    ) -> None:
        super().__init__()
        cfg = scenario_cfg or ScenarioConfig()
        expected_dim = 5 + 3 + 4 + 4 * cfg.nearest_zone_count
        if state_dim != expected_dim:
            raise ValueError(
                f'FixedObsScaler expected state_dim={expected_dim} for nearest_zone_count='
                f'{cfg.nearest_zone_count}, got {state_dim}.'
            )

        world_xy = self._positive(cfg.world_xy, 'world_xy')
        world_z_max = self._positive(cfg.world_z_max, 'world_z_max')
        gamma_scale = self._positive(cfg.gamma_max, 'gamma_max')
        step_distance = self._positive(cfg.speed * cfg.dt, 'speed * dt')
        radius_scale = self._positive(float(cfg.hard_no_fly_radius_range[1]), 'hard_no_fly_radius_range[1]')

        scale_values = [
            world_xy,
            world_xy,
            world_z_max,
            gamma_scale,
            math.pi,
            world_xy,
            world_xy,
            world_z_max,
            world_z_max,
            world_z_max,
            step_distance,
            1.0,
        ]
        for _ in range(cfg.nearest_zone_count):
            scale_values.extend([world_xy, world_xy, radius_scale, world_z_max])

        scale = torch.tensor(scale_values, dtype=torch.float32)
        if torch.any(scale <= 0):
            raise ValueError('FixedObsScaler scale values must all be positive.')
        self.register_buffer('scale', scale)
        self.clamp_value = clamp_value

    @staticmethod
    def _positive(value: float, name: str) -> float:
        value = float(value)
        if value <= 0.0:
            raise ValueError(f'{name} must be positive for FixedObsScaler, got {value}.')
        return value

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        scaled = obs / self.scale.to(device=obs.device, dtype=obs.dtype)
        if self.clamp_value is not None:
            scaled = torch.clamp(scaled, -self.clamp_value, self.clamp_value)
        return scaled
