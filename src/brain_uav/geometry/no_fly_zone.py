"""Safety-boundary wrapper around raw geometry shapes."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

import numpy as np

from .base import GEOMETRY_TOLERANCE, GeometryShape, as_point3, nonnegative_scalar


def _copy_json_value(value: Any, *, path: str, active_containers: set[int]) -> Any:
    value_type = type(value)
    if value is None or value_type in (bool, str, int):
        return value
    if value_type is float:
        if not isfinite(value):
            raise ValueError(f'{path} must contain only finite JSON numbers.')
        return value
    if value_type not in (list, dict):
        raise ValueError(
            f'{path} contains unsupported JSON value type {value_type.__name__!r}.'
        )

    identity = id(value)
    if identity in active_containers:
        raise ValueError(f'{path} contains a circular metadata reference.')
    active_containers.add(identity)
    try:
        if value_type is list:
            return [
                _copy_json_value(item, path=f'{path}[{index}]', active_containers=active_containers)
                for index, item in enumerate(value)
            ]

        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(
                    f'{path} dictionary keys must be strings; got {type(key).__name__!r}.'
                )
            result[key] = _copy_json_value(
                item,
                path=f'{path}[{key!r}]',
                active_containers=active_containers,
            )
        return result
    finally:
        active_containers.remove(identity)


def _copy_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if type(metadata) is not dict:
        raise ValueError('metadata must be a JSON object with string keys.')
    return _copy_json_value(metadata, path='metadata', active_containers=set())


class NoFlyZone:
    """A named geometry plus safety margin and non-geometric metadata."""

    def __init__(
        self,
        zone_id: str,
        shape: GeometryShape,
        safety_margin: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(zone_id, str) or not zone_id.strip():
            raise ValueError('zone_id must be a non-empty string.')
        if not isinstance(shape, GeometryShape):
            raise ValueError('shape must implement GeometryShape.')
        self._zone_id = zone_id
        self._shape = shape
        self._safety_margin = nonnegative_scalar(safety_margin, name='safety_margin')
        self._metadata = _copy_metadata(metadata)

    @property
    def zone_id(self) -> str:
        return self._zone_id

    @property
    def shape(self) -> GeometryShape:
        return self._shape

    @property
    def safety_margin(self) -> float:
        return self._safety_margin

    @property
    def metadata(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    def point_clearance(self, point: Any, uav_radius: float = 0.0) -> float:
        radius = nonnegative_scalar(uav_radius, name='uav_radius')
        return float(self.shape.signed_distance(point) - self.safety_margin - radius)

    def violates_point(self, point: Any, uav_radius: float = 0.0) -> bool:
        return bool(self.point_clearance(point, uav_radius=uav_radius) <= GEOMETRY_TOLERANCE)

    def segment_clearance(self, start: Any, end: Any, uav_radius: float = 0.0) -> float:
        radius = nonnegative_scalar(uav_radius, name='uav_radius')
        return float(self.shape.segment_clearance(start, end) - self.safety_margin - radius)

    def violates_segment(self, start: Any, end: Any, uav_radius: float = 0.0) -> bool:
        first = as_point3(start, name='start')
        second = as_point3(end, name='end')
        radius = nonnegative_scalar(uav_radius, name='uav_radius')
        bounds = self.shape.bounding_box()
        # A disjoint segment AABB proves safety; overlapping bounds prove nothing.
        # Include the existing collision tolerance plus an outward rounding guard.
        scale = max(1.0, float(np.max(np.abs(bounds.min_corner))),
                    float(np.max(np.abs(bounds.max_corner))), self.safety_margin, radius)
        expansion = (self.safety_margin + radius + 2.0 * GEOMETRY_TOLERANCE
                     + 8.0 * np.finfo(np.float64).eps * scale)
        if (np.any(np.maximum(first, second) < bounds.min_corner - expansion)
                or np.any(np.minimum(first, second) > bounds.max_corner + expansion)):
            return False
        return bool(self.segment_clearance(first, second, uav_radius=radius) <= GEOMETRY_TOLERANCE)

    def to_dict(self) -> dict[str, Any]:
        from .serialization import no_fly_zone_to_dict

        return no_fly_zone_to_dict(self)
