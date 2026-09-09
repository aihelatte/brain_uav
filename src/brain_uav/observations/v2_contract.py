"""Data contracts for a single structured V2 observation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


EGO_FEATURE_NAMES = (
    'uav_x_norm',
    'uav_y_norm',
    'uav_z_fraction',
    'gamma_fraction',
    'sin_psi',
    'cos_psi',
)

GOAL_FEATURE_NAMES = (
    'goal_forward_norm',
    'goal_right_norm',
    'goal_up_norm',
    'goal_distance_norm',
)

ZONE_FEATURE_NAMES = (
    'shape_sphere',
    'shape_ellipsoid',
    'shape_box',
    'shape_triangular_pyramid',
    'shape_quadrangular_pyramid',
    'zone_forward_norm',
    'zone_right_norm',
    'zone_up_norm',
    'extent_x_norm',
    'extent_y_norm',
    'extent_z_norm',
    'safety_margin_norm',
    'point_clearance_norm',
    'surface_normal_forward',
    'surface_normal_right',
    'surface_normal_up',
    'approach_cosine',
    'raw_goal_path_intersects',
    'raw_first_intersection_fraction',
)

EGO_FEATURE_DIM = 6
GOAL_FEATURE_DIM = 4
ZONE_FEATURE_DIM = 19


def _feature_index(names: tuple[str, ...]) -> Mapping[str, int]:
    return MappingProxyType({name: index for index, name in enumerate(names)})


EGO_FEATURE_INDEX = _feature_index(EGO_FEATURE_NAMES)
GOAL_FEATURE_INDEX = _feature_index(GOAL_FEATURE_NAMES)
ZONE_FEATURE_INDEX = _feature_index(ZONE_FEATURE_NAMES)


def _finite_scalar(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number.') from exc
    if not isfinite(result):
        raise ValueError(f'{name} must be a finite number.')
    return result


@dataclass(frozen=True, slots=True)
class V2ObservationScales:
    """World and attitude scales used by the V2 feature contract."""

    world_xy: float
    world_z_min: float
    world_z_max: float
    gamma_max: float

    def __post_init__(self) -> None:
        world_xy = _finite_scalar(self.world_xy, name='world_xy')
        world_z_min = _finite_scalar(self.world_z_min, name='world_z_min')
        world_z_max = _finite_scalar(self.world_z_max, name='world_z_max')
        gamma_max = _finite_scalar(self.gamma_max, name='gamma_max')
        if world_xy <= 0.0:
            raise ValueError('world_xy must be greater than zero.')
        if world_z_max <= world_z_min:
            raise ValueError('world_z_max must be greater than world_z_min.')
        if gamma_max <= 0.0:
            raise ValueError('gamma_max must be greater than zero.')
        object.__setattr__(self, 'world_xy', world_xy)
        object.__setattr__(self, 'world_z_min', world_z_min)
        object.__setattr__(self, 'world_z_max', world_z_max)
        object.__setattr__(self, 'gamma_max', gamma_max)

    @property
    def horizontal_span(self) -> float:
        return 2.0 * self.world_xy

    @property
    def vertical_span(self) -> float:
        return self.world_z_max - self.world_z_min

    @property
    def world_diagonal(self) -> float:
        horizontal_span = self.horizontal_span
        return sqrt(
            horizontal_span**2
            + horizontal_span**2
            + self.vertical_span**2
        )


def _float32_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float32, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be convertible to a finite float32 array.') from exc
    if array.shape != shape:
        raise ValueError(f'{name} must have shape {shape}; got {array.shape}.')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite float32 values.')
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, eq=False)
class V2Observation:
    """Immutable per-scenario observation with a dynamic zone axis."""

    ego_features: np.ndarray
    goal_features: np.ndarray
    zone_features: np.ndarray
    presence_mask: np.ndarray

    def __post_init__(self) -> None:
        try:
            raw_zone_features = np.asarray(self.zone_features)
        except (TypeError, ValueError) as exc:
            raise ValueError('zone_features must be an array with shape (N, 19).') from exc
        if raw_zone_features.ndim != 2:
            raise ValueError('zone_features must be an array with shape (N, 19).')
        zone_count = int(raw_zone_features.shape[0])

        ego_features = _float32_array(
            self.ego_features,
            shape=(EGO_FEATURE_DIM,),
            name='ego_features',
        )
        goal_features = _float32_array(
            self.goal_features,
            shape=(GOAL_FEATURE_DIM,),
            name='goal_features',
        )
        zone_features = _float32_array(
            self.zone_features,
            shape=(zone_count, ZONE_FEATURE_DIM),
            name='zone_features',
        )
        try:
            presence_mask = np.array(self.presence_mask, dtype=np.bool_, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError('presence_mask must be convertible to a boolean array.') from exc
        if presence_mask.shape != (zone_count,):
            raise ValueError(
                f'presence_mask must have shape ({zone_count},); got {presence_mask.shape}.'
            )
        if not np.all(presence_mask):
            raise ValueError(
                'A per-scenario V2Observation contains only real zones, '
                'so presence_mask must contain only True.'
            )
        presence_mask.setflags(write=False)

        object.__setattr__(self, 'ego_features', ego_features)
        object.__setattr__(self, 'goal_features', goal_features)
        object.__setattr__(self, 'zone_features', zone_features)
        object.__setattr__(self, 'presence_mask', presence_mask)
