"""Tests for strict V2 geometry and no-fly-zone serialization."""

from __future__ import annotations

import json
import unittest

import numpy as np

from brain_uav.geometry import (
    GEOMETRY_SCHEMA_VERSION,
    GEOMETRY_TOLERANCE,
    Box,
    Ellipsoid,
    NoFlyZone,
    QuadrangularPyramid,
    Sphere,
    TriangularPyramid,
    no_fly_zone_from_dict,
    shape_from_dict,
)


class TestGeometrySerialization(unittest.TestCase):
    def shape_cases(self):
        return (
            (
                Sphere([1.0, 2.0, 3.0], 1.0),
                {
                    'schema_version': 2,
                    'shape_type': 'sphere',
                    'center': [1.0, 2.0, 3.0],
                    'radius': 1.0,
                },
            ),
            (
                Ellipsoid([1.0, 2.0, 3.0], 2.0, 1.5, 1.0),
                {
                    'schema_version': 2,
                    'shape_type': 'ellipsoid',
                    'center': [1.0, 2.0, 3.0],
                    'radius_x': 2.0,
                    'radius_y': 1.5,
                    'radius_z': 1.0,
                },
            ),
            (
                Box([1.0, 2.0, 3.0], 4.0, 3.0, 2.0),
                {
                    'schema_version': 2,
                    'shape_type': 'box',
                    'center': [1.0, 2.0, 3.0],
                    'size_x': 4.0,
                    'size_y': 3.0,
                    'size_z': 2.0,
                },
            ),
            (
                TriangularPyramid([1.0, 2.0, 0.0], 4.0, 3.0, 5.0),
                {
                    'schema_version': 2,
                    'shape_type': 'triangular_pyramid',
                    'base_center': [1.0, 2.0, 0.0],
                    'base_size_x': 4.0,
                    'base_size_y': 3.0,
                    'height': 5.0,
                },
            ),
            (
                QuadrangularPyramid([1.0, 2.0, 0.0], 4.0, 3.0, 5.0),
                {
                    'schema_version': 2,
                    'shape_type': 'quadrangular_pyramid',
                    'base_center': [1.0, 2.0, 0.0],
                    'base_size_x': 4.0,
                    'base_size_y': 3.0,
                    'height': 5.0,
                },
            ),
        )

    def test_all_shape_types_have_stable_strict_round_trip(self):
        point = [6.0, 2.5, 4.0]
        segment_start = [-10.0, 2.0, 3.0]
        segment_end = [10.0, 2.0, 3.0]
        for shape, expected in self.shape_cases():
            with self.subTest(shape=expected['shape_type']):
                payload = shape.to_dict()
                self.assertEqual(payload, expected)
                self.assertNotIn('orientation', payload)
                json_payload = json.loads(json.dumps(payload))
                restored = shape_from_dict(json_payload)
                self.assertIs(type(restored), type(shape))
                self.assertEqual(restored.to_dict(), expected)
                self.assertAlmostEqual(restored.signed_distance(point), shape.signed_distance(point), places=9)
                np.testing.assert_allclose(restored.closest_point(point), shape.closest_point(point), atol=1e-9)
                self.assertAlmostEqual(
                    restored.segment_clearance(segment_start, segment_end),
                    shape.segment_clearance(segment_start, segment_end),
                    places=9,
                )
                original_hit = shape.segment_intersection(segment_start, segment_end)
                restored_hit = restored.segment_intersection(segment_start, segment_end)
                self.assertIsNotNone(original_hit)
                self.assertIsNotNone(restored_hit)
                self.assertAlmostEqual(restored_hit.t, original_hit.t, places=10)
                np.testing.assert_allclose(restored_hit.point, original_hit.point, atol=1e-9)
                np.testing.assert_allclose(restored_hit.normal, original_hit.normal, atol=1e-9)
                self.assertEqual(restored_hit.starts_inside, original_hit.starts_inside)
                np.testing.assert_allclose(restored.bounding_box().min_corner, shape.bounding_box().min_corner)
                np.testing.assert_allclose(restored.bounding_box().max_corner, shape.bounding_box().max_corner)

    def test_unknown_missing_or_forbidden_shape_fields_fail(self):
        valid = Sphere([0.0, 0.0, 1.0], 1.0).to_dict()
        invalid_payloads = (
            None,
            {'shape_type': 'sphere', 'center': [0.0, 0.0, 1.0], 'radius': 1.0},
            {**valid, 'schema_version': 1},
            {**valid, 'schema_version': '2'},
            {**valid, 'shape_type': 'cube'},
            {key: value for key, value in valid.items() if key != 'radius'},
            {**valid, 'extra': 1},
            {**valid, 'extra': 1, 7: 'non-string-key'},
            {**valid, 'orientation': [0.0, 0.0, 0.0, 1.0]},
            {**valid, 'center': [0.0, 0.0, 0.5], 'radius': 1.0},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                shape_from_dict(payload)

    def test_orientation_is_rejected_for_every_shape_type(self):
        for shape, _ in self.shape_cases():
            payload = shape.to_dict()
            payload['orientation'] = None
            with self.subTest(shape=payload['shape_type']), self.assertRaises(ValueError):
                shape_from_dict(payload)

    def test_no_fly_zone_round_trip_preserves_metadata_and_geometry(self):
        zone = NoFlyZone(
            zone_id='zone-a',
            shape=Ellipsoid([1.0, 2.0, 3.0], 2.0, 1.5, 1.0),
            safety_margin=0.25,
            metadata={'label': 'alpha', 'nested': {'enabled': True}},
        )
        expected = {
            'schema_version': GEOMETRY_SCHEMA_VERSION,
            'zone_id': 'zone-a',
            'shape': zone.shape.to_dict(),
            'safety_margin': 0.25,
            'metadata': {'label': 'alpha', 'nested': {'enabled': True}},
        }
        self.assertEqual(zone.to_dict(), expected)
        restored = no_fly_zone_from_dict(json.loads(json.dumps(expected)))
        self.assertEqual(restored.to_dict(), expected)
        self.assertEqual(restored.metadata, expected['metadata'])
        self.assertAlmostEqual(restored.point_clearance([4.0, 2.0, 3.0]), zone.point_clearance([4.0, 2.0, 3.0]))

    def test_no_fly_zone_schema_is_strict(self):
        valid = NoFlyZone('zone', Sphere([0.0, 0.0, 1.0], 1.0), 0.1, {'key': 'value'}).to_dict()
        invalid_payloads = (
            {key: value for key, value in valid.items() if key != 'zone_id'},
            {**valid, 'schema_version': 3},
            {**valid, 'unknown': True},
            {**valid, 'orientation': None},
            {**valid, 'metadata': []},
            {**valid, 'shape': {**valid['shape'], 'orientation': None}},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                no_fly_zone_from_dict(payload)

    def test_common_geometry_contract_is_consistent_for_all_shapes(self):
        queries = (
            (Sphere([1.0, 2.0, 3.0], 1.0), [1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 2.0, 5.0]),
            (
                Ellipsoid([1.0, 2.0, 3.0], 2.0, 1.5, 1.0),
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 3.0],
                [1.0, 2.0, 5.0],
            ),
            (Box([1.0, 2.0, 3.0], 4.0, 3.0, 2.0), [1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [1.0, 2.0, 5.0]),
            (
                TriangularPyramid([1.0, 2.0, 0.0], 4.0, 3.0, 5.0),
                [1.0, 2.0, 1.0],
                [1.0, 2.0, 5.0],
                [1.0, 2.0, 6.0],
            ),
            (
                QuadrangularPyramid([1.0, 2.0, 0.0], 4.0, 3.0, 5.0),
                [1.0, 2.0, 1.0],
                [1.0, 2.0, 5.0],
                [1.0, 2.0, 6.0],
            ),
        )
        for shape, inside, boundary, outside in queries:
            with self.subTest(shape=type(shape).__name__):
                for point in (inside, boundary, outside):
                    signed_distance = shape.signed_distance(point)
                    self.assertEqual(shape.contains(point), signed_distance <= GEOMETRY_TOLERANCE)
                    closest = shape.closest_point(point)
                    self.assertLessEqual(abs(shape.signed_distance(closest)), 1e-7)
                    self.assertAlmostEqual(
                        float(np.linalg.norm(np.asarray(point) - closest)),
                        abs(signed_distance),
                        places=8,
                    )
                    normal = shape.surface_normal(point)
                    self.assertTrue(np.all(np.isfinite(normal)))
                    self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=9)
                    self.assertTrue(shape.bounding_box().contains(closest))

                crossing_start = [1.0, 2.0, -1.0]
                crossing_end = [1.0, 2.0, 7.0]
                self.assertIsNotNone(shape.segment_intersection(crossing_start, crossing_end))
                self.assertEqual(shape.segment_clearance(crossing_start, crossing_end), 0.0)
                miss_start = [100.0, 100.0, 0.0]
                miss_end = [100.0, 100.0, 10.0]
                self.assertIsNone(shape.segment_intersection(miss_start, miss_end))
                self.assertGreater(shape.segment_clearance(miss_start, miss_end), 0.0)


if __name__ == '__main__':
    unittest.main()
