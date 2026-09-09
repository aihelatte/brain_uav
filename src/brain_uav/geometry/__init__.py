"""Unified V2 no-fly-zone geometry API."""

from .base import AABB, GEOMETRY_TOLERANCE, GeometryConvergenceError, GeometryShape, SegmentHit
from .no_fly_zone import NoFlyZone
from .polyhedra import QuadrangularPyramid, TriangularPyramid
from .primitives import Box, Ellipsoid, Sphere
from .serialization import GEOMETRY_SCHEMA_VERSION, no_fly_zone_from_dict, shape_from_dict

__all__ = [
    'AABB',
    'Box',
    'GEOMETRY_TOLERANCE',
    'GEOMETRY_SCHEMA_VERSION',
    'GeometryConvergenceError',
    'GeometryShape',
    'Ellipsoid',
    'NoFlyZone',
    'QuadrangularPyramid',
    'SegmentHit',
    'Sphere',
    'TriangularPyramid',
    'no_fly_zone_from_dict',
    'shape_from_dict',
]
