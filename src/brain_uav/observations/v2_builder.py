"""Build one structured V2 observation from state, goal, and no-fly zones."""

from __future__ import annotations

from collections.abc import Sequence
from math import cos, isfinite, sin
from typing import Any

import numpy as np

from brain_uav.geometry import (
    Box,
    Ellipsoid,
    NoFlyZone,
    QuadrangularPyramid,
    Sphere,
    TriangularPyramid,
)

from .v2_contract import (
    EGO_FEATURE_DIM,
    EGO_FEATURE_INDEX,
    GOAL_FEATURE_DIM,
    GOAL_FEATURE_INDEX,
    ZONE_FEATURE_DIM,
    ZONE_FEATURE_INDEX,
    V2Observation,
    V2ObservationScales,
)


_SHAPE_FEATURE_NAME = {
    Sphere: 'shape_sphere',
    Ellipsoid: 'shape_ellipsoid',
    Box: 'shape_box',
    TriangularPyramid: 'shape_triangular_pyramid',
    QuadrangularPyramid: 'shape_quadrangular_pyramid',
}


def _finite_vector(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite array with shape {shape}.') from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be a finite array with shape {shape}.')
    return array.copy()


def _nonnegative_scalar(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and greater than or equal to zero.') from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and greater than or equal to zero.')
    return result


def _validated_zones(zones: Any) -> tuple[NoFlyZone, ...]:
    if isinstance(zones, (str, bytes)) or not isinstance(zones, Sequence):
        raise TypeError('zones must be a sequence of NoFlyZone objects.')
    zone_items = tuple(zones)
    seen_ids: set[str] = set()
    for index, zone in enumerate(zone_items):
        if not isinstance(zone, NoFlyZone):
            raise ValueError(f'zones[{index}] must be a NoFlyZone.')
        if not isinstance(zone.zone_id, str) or not zone.zone_id.strip():
            raise ValueError(f'zones[{index}].zone_id must be a non-empty string.')
        if zone.zone_id in seen_ids:
            raise ValueError(f'Duplicate zone_id is not allowed: {zone.zone_id!r}.')
        seen_ids.add(zone.zone_id)
    return zone_items


def _validated_zone_point_clearances(
    values: Any,
    *,
    zone_count: int,
) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError('zone_point_clearances must be a finite one-dimensional sequence.')
    try:
        clearances = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            'zone_point_clearances must be a finite one-dimensional sequence.'
        ) from exc
    if clearances.shape != (zone_count,) or not np.all(np.isfinite(clearances)):
        raise ValueError(
            'zone_point_clearances must have one finite value for every zone.'
        )
    return clearances.copy()


def _validated_zone_surface_normals(
    values: Any,
    *,
    zone_count: int,
) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError('zone_surface_normals must be a finite two-dimensional sequence.')
    try:
        normals = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            'zone_surface_normals must be a finite two-dimensional sequence.'
        ) from exc
    if zone_count == 0 and normals.size == 0:
        normals = np.empty((0, 3), dtype=np.float64)
    if normals.shape != (zone_count, 3) or not np.all(np.isfinite(normals)):
        raise ValueError(
            'zone_surface_normals must contain one finite 3-vector for every zone.'
        )
    return normals.copy()


def build_v2_observation(
    state: Any,
    goal: Any,
    zones: Sequence[NoFlyZone],
    scales: V2ObservationScales,
    *,
    uav_radius: float = 0.0,
    zone_point_clearances: Sequence[float] | None = None,
    zone_surface_normals: Sequence[Sequence[float]] | None = None,
) -> V2Observation:
    """Build one dynamic-length V2 observation without padding or sorting."""

    if not isinstance(scales, V2ObservationScales):
        raise TypeError('scales must be a V2ObservationScales instance.')
    state_array = _finite_vector(state, shape=(5,), name='state')
    goal_array = _finite_vector(goal, shape=(3,), name='goal')
    radius = _nonnegative_scalar(uav_radius, name='uav_radius')
    zone_items = _validated_zones(zones)
    point_clearances = _validated_zone_point_clearances(
        zone_point_clearances,
        zone_count=len(zone_items),
    )
    surface_normals = _validated_zone_surface_normals(
        zone_surface_normals,
        zone_count=len(zone_items),
    )

    uav_position = state_array[:3]
    gamma = float(state_array[3])
    psi = float(state_array[4])
    cos_psi = cos(psi)
    sin_psi = sin(psi)
    forward_axis = np.array([cos_psi, sin_psi, 0.0], dtype=np.float64)
    right_axis = np.array([-sin_psi, cos_psi, 0.0], dtype=np.float64)
    flight_direction = np.array(
        [cos(gamma) * cos_psi, cos(gamma) * sin_psi, sin(gamma)],
        dtype=np.float64,
    )

    ego_features = np.empty(EGO_FEATURE_DIM, dtype=np.float64)
    ego_features[EGO_FEATURE_INDEX['uav_x_norm']] = state_array[0] / scales.world_xy
    ego_features[EGO_FEATURE_INDEX['uav_y_norm']] = state_array[1] / scales.world_xy
    ego_features[EGO_FEATURE_INDEX['uav_z_fraction']] = (
        state_array[2] - scales.world_z_min
    ) / scales.vertical_span
    ego_features[EGO_FEATURE_INDEX['gamma_fraction']] = gamma / scales.gamma_max
    ego_features[EGO_FEATURE_INDEX['sin_psi']] = sin_psi
    ego_features[EGO_FEATURE_INDEX['cos_psi']] = cos_psi

    goal_delta = goal_array - uav_position
    goal_features = np.empty(GOAL_FEATURE_DIM, dtype=np.float64)
    goal_features[GOAL_FEATURE_INDEX['goal_forward_norm']] = (
        np.dot(goal_delta, forward_axis) / scales.horizontal_span
    )
    goal_features[GOAL_FEATURE_INDEX['goal_right_norm']] = (
        np.dot(goal_delta, right_axis) / scales.horizontal_span
    )
    goal_features[GOAL_FEATURE_INDEX['goal_up_norm']] = (
        goal_delta[2] / scales.vertical_span
    )
    goal_features[GOAL_FEATURE_INDEX['goal_distance_norm']] = (
        np.linalg.norm(goal_delta) / scales.world_diagonal
    )

    zone_features = np.zeros(
        (len(zone_items), ZONE_FEATURE_DIM),
        dtype=np.float64,
    )
    for row_index, zone in enumerate(zone_items):
        shape = zone.shape
        shape_feature_name = _SHAPE_FEATURE_NAME.get(type(shape))
        if shape_feature_name is None:
            raise ValueError(
                f'Unsupported GeometryShape implementation: {type(shape).__name__}.'
            )
        row = zone_features[row_index]
        row[ZONE_FEATURE_INDEX[shape_feature_name]] = 1.0

        bounds = shape.bounding_box()
        reference_point = 0.5 * (bounds.min_corner + bounds.max_corner)
        extents = bounds.max_corner - bounds.min_corner
        reference_delta = reference_point - uav_position
        row[ZONE_FEATURE_INDEX['zone_forward_norm']] = (
            np.dot(reference_delta, forward_axis) / scales.horizontal_span
        )
        row[ZONE_FEATURE_INDEX['zone_right_norm']] = (
            np.dot(reference_delta, right_axis) / scales.horizontal_span
        )
        row[ZONE_FEATURE_INDEX['zone_up_norm']] = (
            reference_delta[2] / scales.vertical_span
        )
        row[ZONE_FEATURE_INDEX['extent_x_norm']] = extents[0] / scales.horizontal_span
        row[ZONE_FEATURE_INDEX['extent_y_norm']] = extents[1] / scales.horizontal_span
        row[ZONE_FEATURE_INDEX['extent_z_norm']] = extents[2] / scales.vertical_span
        row[ZONE_FEATURE_INDEX['safety_margin_norm']] = (
            zone.safety_margin / scales.world_diagonal
        )
        point_clearance = (
            zone.point_clearance(uav_position, uav_radius=radius)
            if point_clearances is None
            else point_clearances[row_index]
        )
        row[ZONE_FEATURE_INDEX['point_clearance_norm']] = (
            point_clearance / scales.world_diagonal
        )

        outward_normal = _finite_vector(
            (
                shape.surface_normal(uav_position)
                if surface_normals is None
                else surface_normals[row_index]
            ),
            shape=(3,),
            name=f'zones[{row_index}] surface normal',
        )
        normal_length = float(np.linalg.norm(outward_normal))
        if not np.isclose(normal_length, 1.0, rtol=0.0, atol=1e-7):
            raise ValueError(
                f'zones[{row_index}] surface normal must have unit length; '
                f'got {normal_length}.'
            )
        row[ZONE_FEATURE_INDEX['surface_normal_forward']] = np.dot(
            outward_normal, forward_axis
        )
        row[ZONE_FEATURE_INDEX['surface_normal_right']] = np.dot(
            outward_normal, right_axis
        )
        row[ZONE_FEATURE_INDEX['surface_normal_up']] = outward_normal[2]
        row[ZONE_FEATURE_INDEX['approach_cosine']] = np.clip(
            -np.dot(flight_direction, outward_normal),
            -1.0,
            1.0,
        )

        hit = shape.segment_intersection(uav_position, goal_array)
        row[ZONE_FEATURE_INDEX['raw_goal_path_intersects']] = (
            1.0 if hit is not None else 0.0
        )
        row[ZONE_FEATURE_INDEX['raw_first_intersection_fraction']] = (
            hit.t if hit is not None else 1.0
        )

    presence_mask = np.ones(len(zone_items), dtype=np.bool_)
    return V2Observation(
        ego_features=ego_features,
        goal_features=goal_features,
        zone_features=zone_features,
        presence_mask=presence_mask,
    )
