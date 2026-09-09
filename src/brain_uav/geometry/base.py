"""Shared contracts and numerical validation for no-fly geometry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


GEOMETRY_TOLERANCE = 1e-9
MAX_NUMERICAL_ITERATIONS = 256


class GeometryConvergenceError(RuntimeError):
    """Raised when a bounded geometry iteration cannot meet its tolerance."""


def as_point3(value: Any, *, name: str = 'point') -> np.ndarray:
    """Return a finite float64 vector with exactly three coordinates."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite three-dimensional point.') from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be a finite three-dimensional point.')
    return array.copy()


def positive_scalar(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and greater than zero.') from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f'{name} must be finite and greater than zero.')
    return result


def nonnegative_scalar(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and greater than or equal to zero.') from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and greater than or equal to zero.')
    return result


def _readonly_point(value: Any, *, name: str) -> np.ndarray:
    point = as_point3(value, name=name)
    point.setflags(write=False)
    return point


@dataclass(frozen=True, slots=True)
class AABB:
    """Axis-aligned bounding box with immutable corner arrays."""

    min_corner: np.ndarray
    max_corner: np.ndarray

    def __post_init__(self) -> None:
        minimum = _readonly_point(self.min_corner, name='min_corner')
        maximum = _readonly_point(self.max_corner, name='max_corner')
        if np.any(minimum > maximum):
            raise ValueError('min_corner must not exceed max_corner on any axis.')
        object.__setattr__(self, 'min_corner', minimum)
        object.__setattr__(self, 'max_corner', maximum)

    def contains(self, point: Any) -> bool:
        candidate = as_point3(point)
        return bool(
            np.all(candidate >= self.min_corner - GEOMETRY_TOLERANCE)
            and np.all(candidate <= self.max_corner + GEOMETRY_TOLERANCE)
        )


@dataclass(frozen=True, slots=True)
class SegmentHit:
    """The first contact between a closed segment and a closed solid."""

    t: float
    point: np.ndarray
    normal: np.ndarray
    starts_inside: bool

    def __post_init__(self) -> None:
        parameter = float(self.t)
        if not np.isfinite(parameter) or parameter < -GEOMETRY_TOLERANCE or parameter > 1.0 + GEOMETRY_TOLERANCE:
            raise ValueError('SegmentHit.t must be finite and in [0, 1].')
        point = _readonly_point(self.point, name='point')
        normal = _readonly_point(self.normal, name='normal')
        norm = float(np.linalg.norm(normal))
        if norm <= GEOMETRY_TOLERANCE:
            raise ValueError('SegmentHit.normal must have non-zero length.')
        normal = normal / norm
        normal.setflags(write=False)
        object.__setattr__(self, 't', float(np.clip(parameter, 0.0, 1.0)))
        object.__setattr__(self, 'point', point)
        object.__setattr__(self, 'normal', normal)
        object.__setattr__(self, 'starts_inside', bool(self.starts_inside))


class GeometryShape(ABC):
    """Common interface implemented by every V2 no-fly solid."""

    def contains(self, point: Any) -> bool:
        return bool(self.signed_distance(point) <= GEOMETRY_TOLERANCE)

    @abstractmethod
    def signed_distance(self, point: Any) -> float:
        """Return Euclidean distance to the boundary, negative inside."""

    @abstractmethod
    def closest_point(self, point: Any) -> np.ndarray:
        """Return the nearest point on the solid boundary."""

    @abstractmethod
    def surface_normal(self, point: Any) -> np.ndarray:
        """Return a deterministic outward unit normal at the closest point."""

    @abstractmethod
    def segment_intersection(self, start: Any, end: Any) -> SegmentHit | None:
        """Return first closed-segment contact, or None when disjoint."""

    @abstractmethod
    def segment_clearance(self, start: Any, end: Any) -> float:
        """Return non-negative Euclidean distance from segment to solid."""

    @abstractmethod
    def bounding_box(self) -> AABB:
        """Return the exact axis-aligned bounding box."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the shape using the strict V2 schema."""

        from .serialization import shape_to_dict

        return shape_to_dict(self)


def convex_segment_clearance(shape: GeometryShape, start: Any, end: Any) -> float:
    """Minimize distance to a convex solid continuously along a segment."""

    start_point = as_point3(start, name='start')
    end_point = as_point3(end, name='end')
    if shape.segment_intersection(start_point, end_point) is not None:
        return 0.0
    direction = end_point - start_point
    segment_length = float(np.linalg.norm(direction))
    if segment_length <= GEOMETRY_TOLERANCE:
        return max(float(shape.signed_distance(start_point)), 0.0)

    def distance(parameter: float) -> float:
        value = float(shape.signed_distance(start_point + parameter * direction))
        if not np.isfinite(value):
            raise GeometryConvergenceError('Segment-clearance objective returned a non-finite value.')
        return max(value, 0.0)

    left = 0.0
    right = 1.0
    inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
    inner_left = right - inverse_phi * (right - left)
    inner_right = left + inverse_phi * (right - left)
    value_left = distance(inner_left)
    value_right = distance(inner_right)
    parameter_tolerance = max(1e-14, GEOMETRY_TOLERANCE / max(segment_length, 1.0))

    for _ in range(MAX_NUMERICAL_ITERATIONS):
        if right - left <= parameter_tolerance:
            candidates = (
                distance(0.0),
                distance(1.0),
                distance(left),
                distance(right),
                value_left,
                value_right,
                distance(0.5 * (left + right)),
            )
            return max(float(min(candidates)), 0.0)
        if value_left <= value_right:
            right = inner_right
            inner_right = inner_left
            value_right = value_left
            inner_left = right - inverse_phi * (right - left)
            value_left = distance(inner_left)
        else:
            left = inner_left
            inner_left = inner_right
            value_left = value_right
            inner_right = left + inverse_phi * (right - left)
            value_right = distance(inner_right)
    raise GeometryConvergenceError('Segment-clearance minimization did not converge.')
