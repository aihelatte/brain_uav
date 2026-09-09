"""Axis-aligned convex polyhedra used by V2 no-fly zones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .base import (
    AABB,
    GEOMETRY_TOLERANCE,
    GeometryConvergenceError,
    GeometryShape,
    SegmentHit,
    as_point3,
    convex_segment_clearance,
    positive_scalar,
)


@dataclass(frozen=True, slots=True)
class _Face:
    indices: tuple[int, ...]
    normal: np.ndarray
    offset: float


def _closest_point_on_triangle(point: np.ndarray, first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    """Exact closest point on a non-degenerate closed triangle."""

    edge_ab = second - first
    edge_ac = third - first
    from_a = point - first
    d1 = float(np.dot(edge_ab, from_a))
    d2 = float(np.dot(edge_ac, from_a))
    if d1 <= 0.0 and d2 <= 0.0:
        return first.copy()

    from_b = point - second
    d3 = float(np.dot(edge_ab, from_b))
    d4 = float(np.dot(edge_ac, from_b))
    if d3 >= 0.0 and d4 <= d3:
        return second.copy()

    edge_region_c = d1 * d4 - d3 * d2
    if edge_region_c <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        fraction = d1 / (d1 - d3)
        return first + fraction * edge_ab

    from_c = point - third
    d5 = float(np.dot(edge_ab, from_c))
    d6 = float(np.dot(edge_ac, from_c))
    if d6 >= 0.0 and d5 <= d6:
        return third.copy()

    edge_region_b = d5 * d2 - d1 * d6
    if edge_region_b <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        fraction = d2 / (d2 - d6)
        return first + fraction * edge_ac

    edge_region_a = d3 * d6 - d5 * d4
    if edge_region_a <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return second + fraction * (third - second)

    denominator = edge_region_a + edge_region_b + edge_region_c
    if abs(denominator) <= GEOMETRY_TOLERANCE:
        raise GeometryConvergenceError('Degenerate triangle encountered in convex polyhedron.')
    inverse = 1.0 / denominator
    coordinate_b = edge_region_b * inverse
    coordinate_c = edge_region_c * inverse
    return first + edge_ab * coordinate_b + edge_ac * coordinate_c


class _ConvexPolyhedron(GeometryShape):
    def __init__(self, vertices: np.ndarray, face_indices: Sequence[Sequence[int]]) -> None:
        array = np.asarray(vertices, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
            raise ValueError('vertices must be a finite array with shape (N, 3).')
        self._vertices = array.copy()
        centroid = np.mean(self._vertices, axis=0)
        faces: list[_Face] = []
        for raw_indices in face_indices:
            indices = tuple(int(index) for index in raw_indices)
            if len(indices) < 3 or any(index < 0 or index >= len(self._vertices) for index in indices):
                raise ValueError('Each face must reference at least three valid vertices.')
            first, second, third = self._vertices[list(indices[:3])]
            normal = np.cross(second - first, third - first)
            norm = float(np.linalg.norm(normal))
            if norm <= GEOMETRY_TOLERANCE:
                raise ValueError('Pyramid faces must be non-degenerate.')
            normal = normal / norm
            if float(np.dot(normal, centroid - first)) > 0.0:
                normal = -normal
            faces.append(_Face(indices, normal, float(np.dot(normal, first))))
        self._faces = tuple(faces)

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices.copy()

    def _plane_values(self, point: np.ndarray) -> np.ndarray:
        return np.array([float(np.dot(face.normal, point) - face.offset) for face in self._faces])

    def _closest_boundary(self, point: np.ndarray) -> tuple[np.ndarray, int, bool]:
        plane_values = self._plane_values(point)
        inside = bool(np.all(plane_values <= 0.0))
        if inside:
            slacks = -plane_values
            face_index = int(np.argmin(slacks))
            face = self._faces[face_index]
            return point + float(slacks[face_index]) * face.normal, face_index, True

        best_point: np.ndarray | None = None
        best_face = -1
        best_distance_squared = float('inf')
        for face_index, face in enumerate(self._faces):
            anchor = face.indices[0]
            for triangle_index in range(1, len(face.indices) - 1):
                closest = _closest_point_on_triangle(
                    point,
                    self._vertices[anchor],
                    self._vertices[face.indices[triangle_index]],
                    self._vertices[face.indices[triangle_index + 1]],
                )
                distance_squared = float(np.dot(point - closest, point - closest))
                if distance_squared < best_distance_squared - GEOMETRY_TOLERANCE**2:
                    best_point = closest
                    best_face = face_index
                    best_distance_squared = distance_squared
        if best_point is None:
            raise GeometryConvergenceError('No closest boundary point was found for convex polyhedron.')
        return best_point, best_face, False

    def signed_distance(self, point: Any) -> float:
        candidate = as_point3(point)
        closest, _, inside = self._closest_boundary(candidate)
        distance = float(np.linalg.norm(candidate - closest))
        return -distance if inside else distance

    def closest_point(self, point: Any) -> np.ndarray:
        candidate = as_point3(point)
        closest, _, _ = self._closest_boundary(candidate)
        return closest

    def surface_normal(self, point: Any) -> np.ndarray:
        candidate = as_point3(point)
        closest, face_index, inside = self._closest_boundary(candidate)
        outward = candidate - closest
        norm = float(np.linalg.norm(outward))
        if not inside and norm > GEOMETRY_TOLERANCE:
            return outward / norm
        return self._faces[face_index].normal.copy()

    def segment_intersection(self, start: Any, end: Any) -> SegmentHit | None:
        start_point = as_point3(start, name='start')
        end_point = as_point3(end, name='end')
        if self.contains(start_point):
            return SegmentHit(0.0, start_point, self.surface_normal(start_point), True)

        direction = end_point - start_point
        enter = 0.0
        leave = 1.0
        enter_face = -1
        for face_index, face in enumerate(self._faces):
            numerator = float(face.offset - np.dot(face.normal, start_point))
            denominator = float(np.dot(face.normal, direction))
            if abs(denominator) <= GEOMETRY_TOLERANCE:
                if numerator < -GEOMETRY_TOLERANCE:
                    return None
                continue
            parameter = numerator / denominator
            if denominator < 0.0:
                if parameter > enter + GEOMETRY_TOLERANCE:
                    enter = parameter
                    enter_face = face_index
            else:
                leave = min(leave, parameter)
            if enter > leave + GEOMETRY_TOLERANCE:
                return None
        if leave < -GEOMETRY_TOLERANCE or enter > 1.0 + GEOMETRY_TOLERANCE:
            return None
        parameter = float(np.clip(enter, 0.0, 1.0))
        hit_point = start_point + parameter * direction
        normal = self._faces[enter_face].normal if enter_face >= 0 else self.surface_normal(hit_point)
        return SegmentHit(parameter, hit_point, normal, False)

    def segment_clearance(self, start: Any, end: Any) -> float:
        return convex_segment_clearance(self, start, end)

    def bounding_box(self) -> AABB:
        return AABB(np.min(self._vertices, axis=0), np.max(self._vertices, axis=0))


def _ground_base_center(value: Any) -> np.ndarray:
    center = as_point3(value, name='base_center')
    if abs(float(center[2])) > GEOMETRY_TOLERANCE:
        raise ValueError('Pyramid base_center must have z=0.')
    center[2] = 0.0
    return center


class TriangularPyramid(_ConvexPolyhedron):
    """Ground-based pyramid with a fixed isosceles triangular base."""

    def __init__(self, base_center: Any, base_size_x: float, base_size_y: float, height: float) -> None:
        self._base_center = _ground_base_center(base_center)
        self._base_size_x = positive_scalar(base_size_x, name='base_size_x')
        self._base_size_y = positive_scalar(base_size_y, name='base_size_y')
        self._height = positive_scalar(height, name='height')
        center_x, center_y, _ = self._base_center
        vertices = np.array(
            [
                [center_x - self._base_size_x / 2.0, center_y - self._base_size_y / 3.0, 0.0],
                [center_x + self._base_size_x / 2.0, center_y - self._base_size_y / 3.0, 0.0],
                [center_x, center_y + 2.0 * self._base_size_y / 3.0, 0.0],
                [center_x, center_y, self._height],
            ],
            dtype=np.float64,
        )
        super().__init__(vertices, ((0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)))

    @property
    def base_center(self) -> np.ndarray:
        return self._base_center.copy()

    @property
    def base_size_x(self) -> float:
        return self._base_size_x

    @property
    def base_size_y(self) -> float:
        return self._base_size_y

    @property
    def height(self) -> float:
        return self._height


class QuadrangularPyramid(_ConvexPolyhedron):
    """Ground-based pyramid with an axis-aligned rectangular base."""

    def __init__(self, base_center: Any, base_size_x: float, base_size_y: float, height: float) -> None:
        self._base_center = _ground_base_center(base_center)
        self._base_size_x = positive_scalar(base_size_x, name='base_size_x')
        self._base_size_y = positive_scalar(base_size_y, name='base_size_y')
        self._height = positive_scalar(height, name='height')
        center_x, center_y, _ = self._base_center
        half_x = self._base_size_x / 2.0
        half_y = self._base_size_y / 2.0
        vertices = np.array(
            [
                [center_x - half_x, center_y - half_y, 0.0],
                [center_x + half_x, center_y - half_y, 0.0],
                [center_x + half_x, center_y + half_y, 0.0],
                [center_x - half_x, center_y + half_y, 0.0],
                [center_x, center_y, self._height],
            ],
            dtype=np.float64,
        )
        super().__init__(vertices, ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))

    @property
    def base_center(self) -> np.ndarray:
        return self._base_center.copy()

    @property
    def base_size_x(self) -> float:
        return self._base_size_x

    @property
    def base_size_y(self) -> float:
        return self._base_size_y

    @property
    def height(self) -> float:
        return self._height
