"""Strict versioned serialization for V2 geometry objects."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .base import GeometryShape
from .no_fly_zone import NoFlyZone
from .polyhedra import QuadrangularPyramid, TriangularPyramid
from .primitives import Box, Ellipsoid, Sphere


GEOMETRY_SCHEMA_VERSION = 2


def _mapping_with_exact_fields(data: Any, required: set[str], *, object_name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f'{object_name} payload must be a mapping.')
    keys = set(data.keys())
    missing = required - keys
    unknown = keys - required
    if missing:
        raise ValueError(f'{object_name} payload is missing fields: {sorted(missing)}.')
    if unknown:
        rendered_unknown = sorted(repr(key) for key in unknown)
        raise ValueError(f'{object_name} payload has unknown fields: {rendered_unknown}.')
    return data


def _validate_schema_version(data: Mapping[str, Any], *, object_name: str) -> None:
    version = data.get('schema_version')
    if type(version) is not int or version != GEOMETRY_SCHEMA_VERSION:
        raise ValueError(f'{object_name} schema_version must be {GEOMETRY_SCHEMA_VERSION}.')


_SHAPE_FIELDS = {
    'sphere': {'center', 'radius'},
    'ellipsoid': {'center', 'radius_x', 'radius_y', 'radius_z'},
    'box': {'center', 'size_x', 'size_y', 'size_z'},
    'triangular_pyramid': {'base_center', 'base_size_x', 'base_size_y', 'height'},
    'quadrangular_pyramid': {'base_center', 'base_size_x', 'base_size_y', 'height'},
}


def shape_to_dict(shape: GeometryShape) -> dict[str, Any]:
    if isinstance(shape, Sphere):
        return {
            'schema_version': GEOMETRY_SCHEMA_VERSION,
            'shape_type': 'sphere',
            'center': shape.center.tolist(),
            'radius': shape.radius,
        }
    if isinstance(shape, Ellipsoid):
        return {
            'schema_version': GEOMETRY_SCHEMA_VERSION,
            'shape_type': 'ellipsoid',
            'center': shape.center.tolist(),
            'radius_x': shape.radius_x,
            'radius_y': shape.radius_y,
            'radius_z': shape.radius_z,
        }
    if isinstance(shape, Box):
        return {
            'schema_version': GEOMETRY_SCHEMA_VERSION,
            'shape_type': 'box',
            'center': shape.center.tolist(),
            'size_x': shape.size_x,
            'size_y': shape.size_y,
            'size_z': shape.size_z,
        }
    if isinstance(shape, TriangularPyramid):
        return {
            'schema_version': GEOMETRY_SCHEMA_VERSION,
            'shape_type': 'triangular_pyramid',
            'base_center': shape.base_center.tolist(),
            'base_size_x': shape.base_size_x,
            'base_size_y': shape.base_size_y,
            'height': shape.height,
        }
    if isinstance(shape, QuadrangularPyramid):
        return {
            'schema_version': GEOMETRY_SCHEMA_VERSION,
            'shape_type': 'quadrangular_pyramid',
            'base_center': shape.base_center.tolist(),
            'base_size_x': shape.base_size_x,
            'base_size_y': shape.base_size_y,
            'height': shape.height,
        }
    raise ValueError(f'Unsupported GeometryShape implementation: {type(shape).__name__}.')


def shape_from_dict(data: Any) -> GeometryShape:
    if not isinstance(data, Mapping):
        raise ValueError('Geometry shape payload must be a mapping.')
    if 'schema_version' not in data:
        raise ValueError('Geometry shape payload is missing fields: [\'schema_version\'].')
    if 'shape_type' not in data:
        raise ValueError('Geometry shape payload is missing fields: [\'shape_type\'].')
    _validate_schema_version(data, object_name='Geometry shape')
    shape_type = data['shape_type']
    if not isinstance(shape_type, str) or shape_type not in _SHAPE_FIELDS:
        raise ValueError(f'Unknown shape_type: {shape_type!r}.')
    required = {'schema_version', 'shape_type'} | _SHAPE_FIELDS[shape_type]
    payload = _mapping_with_exact_fields(data, required, object_name=f'{shape_type} shape')

    if shape_type == 'sphere':
        return Sphere(payload['center'], payload['radius'])
    if shape_type == 'ellipsoid':
        return Ellipsoid(
            payload['center'], payload['radius_x'], payload['radius_y'], payload['radius_z']
        )
    if shape_type == 'box':
        return Box(payload['center'], payload['size_x'], payload['size_y'], payload['size_z'])
    if shape_type == 'triangular_pyramid':
        return TriangularPyramid(
            payload['base_center'], payload['base_size_x'], payload['base_size_y'], payload['height']
        )
    return QuadrangularPyramid(
        payload['base_center'], payload['base_size_x'], payload['base_size_y'], payload['height']
    )


def no_fly_zone_to_dict(zone: NoFlyZone) -> dict[str, Any]:
    if not isinstance(zone, NoFlyZone):
        raise ValueError('zone must be a NoFlyZone.')
    return {
        'schema_version': GEOMETRY_SCHEMA_VERSION,
        'zone_id': zone.zone_id,
        'shape': shape_to_dict(zone.shape),
        'safety_margin': zone.safety_margin,
        'metadata': deepcopy(zone.metadata),
    }


def no_fly_zone_from_dict(data: Any) -> NoFlyZone:
    required = {'schema_version', 'zone_id', 'shape', 'safety_margin', 'metadata'}
    payload = _mapping_with_exact_fields(data, required, object_name='NoFlyZone')
    _validate_schema_version(payload, object_name='NoFlyZone')
    metadata = payload['metadata']
    if not isinstance(metadata, Mapping):
        raise ValueError('NoFlyZone metadata must be a mapping.')
    return NoFlyZone(
        zone_id=payload['zone_id'],
        shape=shape_from_dict(payload['shape']),
        safety_margin=payload['safety_margin'],
        metadata=metadata,
    )
