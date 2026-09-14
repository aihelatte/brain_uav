"""Analytic axis-aligned geometry primitives."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import (
    AABB,
    GEOMETRY_TOLERANCE,
    MAX_NUMERICAL_ITERATIONS,
    GeometryConvergenceError,
    GeometryShape,
    SegmentHit,
    as_point3,
    convex_segment_clearance,
    positive_scalar,
)


class Sphere(GeometryShape):
    """A complete, possibly suspended sphere with no rotation."""

    def __init__(self, center: Any, radius: float) -> None:
        self._center = as_point3(center, name='center')
        self._radius = positive_scalar(radius, name='radius')
        if self._center[2] - self._radius < -GEOMETRY_TOLERANCE:
            raise ValueError('Sphere must not extend below z=0.')

    @property
    def center(self) -> np.ndarray:
        return self._center.copy()

    @property
    def radius(self) -> float:
        return self._radius

    def signed_distance(self, point: Any) -> float:
        candidate = as_point3(point)
        return float(np.linalg.norm(candidate - self._center) - self.radius)

    def closest_point(self, point: Any) -> np.ndarray:
        candidate = as_point3(point)
        delta = candidate - self._center
        norm = float(np.linalg.norm(delta))
        if norm <= GEOMETRY_TOLERANCE:
            delta = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            delta = delta / norm
        return self._center + self.radius * delta

    def surface_normal(self, point: Any) -> np.ndarray:
        closest = self.closest_point(point)
        return (closest - self._center) / self.radius

    def segment_intersection(self, start: Any, end: Any) -> SegmentHit | None:
        start_point = as_point3(start, name='start')
        end_point = as_point3(end, name='end')
        if self.contains(start_point):
            return SegmentHit(0.0, start_point, self.surface_normal(start_point), True)

        direction = end_point - start_point
        segment_length = float(np.linalg.norm(direction))
        if segment_length <= GEOMETRY_TOLERANCE:
            return None
        a = segment_length**2
        offset = start_point - self._center
        center_parameter = -float(np.dot(offset, direction)) / a
        nearest_offset = offset + center_parameter * direction
        nearest_distance = float(np.linalg.norm(nearest_offset))
        if nearest_distance - self.radius > GEOMETRY_TOLERANCE:
            return None

        chord_squared = max(self.radius**2 - nearest_distance**2, 0.0)
        half_span = float(np.sqrt(chord_squared / a))
        roots = (center_parameter - half_span, center_parameter + half_span)
        parameter_tolerance = GEOMETRY_TOLERANCE / segment_length
        for parameter in roots:
            if -parameter_tolerance <= parameter <= 1.0 + parameter_tolerance:
                parameter = float(np.clip(parameter, 0.0, 1.0))
                point = start_point + parameter * direction
                return SegmentHit(parameter, point, self.surface_normal(point), False)
        return None

    def segment_clearance(self, start: Any, end: Any) -> float:
        start_point = as_point3(start, name='start')
        end_point = as_point3(end, name='end')
        if self.segment_intersection(start_point, end_point) is not None:
            return 0.0
        direction = end_point - start_point
        length_squared = float(np.dot(direction, direction))
        if length_squared <= GEOMETRY_TOLERANCE**2:
            center_distance = float(np.linalg.norm(start_point - self._center))
        else:
            parameter = float(np.clip(np.dot(self._center - start_point, direction) / length_squared, 0.0, 1.0))
            center_distance = float(np.linalg.norm(start_point + parameter * direction - self._center))
        return max(center_distance - self.radius, 0.0)

    def bounding_box(self) -> AABB:
        radius_vector = np.full(3, self.radius, dtype=np.float64)
        return AABB(self._center - radius_vector, self._center + radius_vector)


class Ellipsoid(GeometryShape):
    """Axis-aligned ellipsoid using exact Euclidean boundary projection."""

    def __init__(
        self,
        center: Any,
        radius_x: float,
        radius_y: float,
        radius_z: float,
    ) -> None:
        self._center = as_point3(center, name='center')
        self._radii = np.array(
            [
                positive_scalar(radius_x, name='radius_x'),
                positive_scalar(radius_y, name='radius_y'),
                positive_scalar(radius_z, name='radius_z'),
            ],
            dtype=np.float64,
        )
        if self._center[2] - self._radii[2] < -GEOMETRY_TOLERANCE:
            raise ValueError('Ellipsoid must not extend below z=0.')

    @property
    def center(self) -> np.ndarray:
        return self._center.copy()

    @property
    def radii(self) -> np.ndarray:
        return self._radii.copy()

    @property
    def radius_x(self) -> float:
        return float(self._radii[0])

    @property
    def radius_y(self) -> float:
        return float(self._radii[1])

    @property
    def radius_z(self) -> float:
        return float(self._radii[2])

    def _level(self, point: np.ndarray) -> float:
        return float(np.sum(((point - self._center) / self._radii) ** 2))

    def _regular_projection(self, magnitudes: np.ndarray, *, outside: bool) -> np.ndarray | None:
        radii_squared = self._radii**2
        active = magnitudes > 0.0
        if not np.any(active):
            return None

        if outside:
            denominator_offsets = radii_squared
            lower = 0.0
            upper = float(np.max(radii_squared))
        else:
            smallest_radius_squared = float(np.min(radii_squared))
            denominator_offsets = radii_squared - smallest_radius_squared
            lower = 0.0
            upper = smallest_radius_squared

        def equation(parameter: float) -> float:
            denominator = denominator_offsets[active] + parameter
            if np.any(denominator == 0.0):
                return float('inf')
            terms = self._radii[active] * magnitudes[active] / denominator
            return float(np.dot(terms, terms) - 1.0)

        if outside:
            for _ in range(MAX_NUMERICAL_ITERATIONS):
                if equation(upper) <= 0.0:
                    break
                upper *= 2.0
            else:
                raise GeometryConvergenceError('Failed to bracket exterior ellipsoid projection.')
        else:
            if equation(lower) <= 0.0:
                return None

        best_parameter = upper
        best_residual = abs(equation(upper))
        residual_tolerance = GEOMETRY_TOLERANCE * 1e-3
        for _ in range(MAX_NUMERICAL_ITERATIONS):
            parameter = lower + 0.5 * (upper - lower)
            if parameter <= lower or parameter >= upper:
                break
            value = equation(parameter)
            residual = abs(value)
            if residual < best_residual:
                best_parameter = parameter
                best_residual = residual
            if best_residual <= residual_tolerance:
                break
            if value > 0.0:
                lower = parameter
            else:
                upper = parameter

        projected = np.zeros(3, dtype=np.float64)
        projected[active] = (
            radii_squared[active]
            * magnitudes[active]
            / (denominator_offsets[active] + best_parameter)
        )
        surface_error = abs(float(np.sum((projected / self._radii) ** 2)) - 1.0)
        if surface_error > GEOMETRY_TOLERANCE:
            raise GeometryConvergenceError('Ellipsoid projection did not converge to the boundary.')
        return projected

    def _singular_internal_candidates(self, magnitudes: np.ndarray) -> list[np.ndarray]:
        radii_squared = self._radii**2
        positive = magnitudes > 0.0
        candidates: list[np.ndarray] = []
        for axis in range(3):
            if positive[axis]:
                continue
            multiplier = -radii_squared[axis]
            normalized = np.zeros(3, dtype=np.float64)
            valid = True
            for index in np.flatnonzero(positive):
                denominator = radii_squared[index] + multiplier
                if denominator <= 0.0:
                    valid = False
                    break
                normalized[index] = self._radii[index] * magnitudes[index] / denominator
            if not valid:
                continue
            used = float(np.dot(normalized, normalized))
            if used > 1.0 + GEOMETRY_TOLERANCE:
                continue
            equal_axes = [
                index
                for index in range(3)
                if not positive[index]
                and radii_squared[index] == radii_squared[axis]
            ]
            if not equal_axes:
                continue
            normalized[equal_axes[0]] = np.sqrt(max(1.0 - used, 0.0))
            candidates.append(self._radii * normalized)
        return candidates

    def closest_point(self, point: Any) -> np.ndarray:
        candidate = as_point3(point)
        relative = candidate - self._center
        magnitudes = np.abs(relative)
        signs = np.where(relative < 0.0, -1.0, 1.0)
        level = float(np.sum((magnitudes / self._radii) ** 2))
        if abs(level - 1.0) <= 1e-14:
            return candidate.copy()

        local_candidates: list[np.ndarray] = []
        regular = self._regular_projection(magnitudes, outside=level > 1.0)
        if regular is not None:
            local_candidates.append(regular)
        if level < 1.0:
            local_candidates.extend(self._singular_internal_candidates(magnitudes))
        if not local_candidates:
            raise GeometryConvergenceError('No valid ellipsoid boundary projection candidate was found.')

        best = local_candidates[0]
        best_distance = float(np.linalg.norm(best - magnitudes))
        for projected in local_candidates[1:]:
            distance = float(np.linalg.norm(projected - magnitudes))
            if distance < best_distance - GEOMETRY_TOLERANCE:
                best = projected
                best_distance = distance
        return self._center + signs * best

    def signed_distance(self, point: Any) -> float:
        candidate = as_point3(point)
        closest = self.closest_point(candidate)
        distance = float(np.linalg.norm(candidate - closest))
        return -distance if self._level(candidate) < 1.0 else distance

    def signed_distance_and_surface_normal(
        self,
        point: Any,
    ) -> tuple[float, np.ndarray]:
        candidate = as_point3(point)
        closest = self.closest_point(candidate)
        distance = float(np.linalg.norm(candidate - closest))
        signed_distance = -distance if self._level(candidate) < 1.0 else distance
        return signed_distance, self._surface_normal_from_closest(closest)

    def _surface_normal_from_closest(self, closest: np.ndarray) -> np.ndarray:
        gradient = (closest - self._center) / (self._radii**2)
        norm = float(np.linalg.norm(gradient))
        if norm <= GEOMETRY_TOLERANCE:
            raise GeometryConvergenceError('Ellipsoid surface normal is numerically undefined.')
        return gradient / norm

    def surface_normal(self, point: Any) -> np.ndarray:
        closest = self.closest_point(point)
        return self._surface_normal_from_closest(closest)

    def segment_intersection(self, start: Any, end: Any) -> SegmentHit | None:
        start_point = as_point3(start, name='start')
        end_point = as_point3(end, name='end')
        if self.contains(start_point):
            return SegmentHit(0.0, start_point, self.surface_normal(start_point), True)
        world_direction = end_point - start_point
        segment_length = float(np.linalg.norm(world_direction))
        if segment_length <= GEOMETRY_TOLERANCE:
            return None
        transformed_start = (start_point - self._center) / self._radii
        transformed_direction = world_direction / self._radii
        a = float(np.dot(transformed_direction, transformed_direction))
        if a == 0.0:
            return None
        center_parameter = -float(np.dot(transformed_start, transformed_direction)) / a
        nearest_transformed = transformed_start + center_parameter * transformed_direction
        nearest_radius = float(np.linalg.norm(nearest_transformed))
        if nearest_radius > 1.0:
            normalized_gap = nearest_radius - 1.0
            if float(np.min(self._radii)) * normalized_gap > GEOMETRY_TOLERANCE:
                return None
            nearest_world = start_point + center_parameter * world_direction
            if self.signed_distance(nearest_world) > GEOMETRY_TOLERANCE:
                return None
            half_span = 0.0
        else:
            chord_squared = max(1.0 - nearest_radius**2, 0.0)
            half_span = float(np.sqrt(chord_squared / a))

        parameter_tolerance = GEOMETRY_TOLERANCE / segment_length
        for parameter in (center_parameter - half_span, center_parameter + half_span):
            if -parameter_tolerance <= parameter <= 1.0 + parameter_tolerance:
                parameter = float(np.clip(parameter, 0.0, 1.0))
                hit_point = start_point + parameter * world_direction
                return SegmentHit(parameter, hit_point, self.surface_normal(hit_point), False)
        return None

    def segment_clearance(self, start: Any, end: Any) -> float:
        return convex_segment_clearance(self, start, end)

    def bounding_box(self) -> AABB:
        return AABB(self._center - self._radii, self._center + self._radii)


class Box(GeometryShape):
    """Axis-aligned rectangular box; equal sizes represent a cube."""

    def __init__(
        self,
        center: Any,
        size_x: float,
        size_y: float,
        size_z: float,
    ) -> None:
        self._center = as_point3(center, name='center')
        self._sizes = np.array(
            [
                positive_scalar(size_x, name='size_x'),
                positive_scalar(size_y, name='size_y'),
                positive_scalar(size_z, name='size_z'),
            ],
            dtype=np.float64,
        )
        self._half_sizes = 0.5 * self._sizes
        if self._center[2] - self._half_sizes[2] < -GEOMETRY_TOLERANCE:
            raise ValueError('Box must not extend below z=0.')

    @property
    def center(self) -> np.ndarray:
        return self._center.copy()

    @property
    def sizes(self) -> np.ndarray:
        return self._sizes.copy()

    @property
    def half_sizes(self) -> np.ndarray:
        return self._half_sizes.copy()

    @property
    def size_x(self) -> float:
        return float(self._sizes[0])

    @property
    def size_y(self) -> float:
        return float(self._sizes[1])

    @property
    def size_z(self) -> float:
        return float(self._sizes[2])

    def signed_distance(self, point: Any) -> float:
        candidate = as_point3(point)
        offset = np.abs(candidate - self._center) - self._half_sizes
        outside = float(np.linalg.norm(np.maximum(offset, 0.0)))
        inside = min(float(np.max(offset)), 0.0)
        return outside + inside

    def _nearest_face(self, candidate: np.ndarray) -> tuple[int, float]:
        relative = candidate - self._center
        clearances = self._half_sizes - np.abs(relative)
        axis = int(np.argmin(clearances))
        sign = -1.0 if relative[axis] < 0.0 else 1.0
        return axis, sign

    def closest_point(self, point: Any) -> np.ndarray:
        candidate = as_point3(point)
        minimum = self._center - self._half_sizes
        maximum = self._center + self._half_sizes
        clamped = np.clip(candidate, minimum, maximum)
        if np.any(candidate < minimum) or np.any(candidate > maximum):
            return clamped
        axis, sign = self._nearest_face(candidate)
        closest = candidate.copy()
        closest[axis] = self._center[axis] + sign * self._half_sizes[axis]
        return closest

    def surface_normal(self, point: Any) -> np.ndarray:
        candidate = as_point3(point)
        if self.signed_distance(candidate) < -GEOMETRY_TOLERANCE:
            axis, sign = self._nearest_face(candidate)
            normal = np.zeros(3, dtype=np.float64)
            normal[axis] = sign
            return normal
        closest = self.closest_point(candidate)
        outward = candidate - closest
        norm = float(np.linalg.norm(outward))
        if norm > GEOMETRY_TOLERANCE:
            return outward / norm
        axis, sign = self._nearest_face(closest)
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = sign
        return normal

    def segment_intersection(self, start: Any, end: Any) -> SegmentHit | None:
        start_point = as_point3(start, name='start')
        end_point = as_point3(end, name='end')
        if self.contains(start_point):
            return SegmentHit(0.0, start_point, self.surface_normal(start_point), True)

        direction = end_point - start_point
        minimum = self._center - self._half_sizes
        maximum = self._center + self._half_sizes
        enter = 0.0
        leave = 1.0
        enter_axis = -1
        enter_sign = 1.0
        for axis in range(3):
            component = float(direction[axis])
            if abs(component) <= GEOMETRY_TOLERANCE:
                if start_point[axis] < minimum[axis] - GEOMETRY_TOLERANCE or start_point[axis] > maximum[axis] + GEOMETRY_TOLERANCE:
                    return None
                continue
            first = float((minimum[axis] - start_point[axis]) / component)
            second = float((maximum[axis] - start_point[axis]) / component)
            near = min(first, second)
            far = max(first, second)
            near_sign = -1.0 if component > 0.0 else 1.0
            if near > enter + GEOMETRY_TOLERANCE:
                enter = near
                enter_axis = axis
                enter_sign = near_sign
            leave = min(leave, far)
            if enter > leave + GEOMETRY_TOLERANCE:
                return None
        if leave < -GEOMETRY_TOLERANCE or enter > 1.0 + GEOMETRY_TOLERANCE:
            return None
        parameter = float(np.clip(enter, 0.0, 1.0))
        hit_point = start_point + parameter * direction
        if enter_axis < 0:
            normal = self.surface_normal(hit_point)
        else:
            normal = np.zeros(3, dtype=np.float64)
            normal[enter_axis] = enter_sign
        return SegmentHit(parameter, hit_point, normal, False)

    def segment_clearance(self, start: Any, end: Any) -> float:
        return convex_segment_clearance(self, start, end)

    def bounding_box(self) -> AABB:
        return AABB(self._center - self._half_sizes, self._center + self._half_sizes)
