"""Tests for no-fly-zone safety-boundary semantics."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from brain_uav.geometry import (GEOMETRY_TOLERANCE, NoFlyZone, Sphere, Ellipsoid, Box,
    TriangularPyramid, QuadrangularPyramid, no_fly_zone_from_dict)


class TestNoFlyZone(unittest.TestCase):
    def test_distant_segment_skips_exact_ellipsoid_solver(self):
        zone = NoFlyZone('far', Ellipsoid([0, 0, 150], 180, 140, 100), 5)
        with patch.object(Ellipsoid, 'segment_clearance', wraps=zone.shape.segment_clearance) as exact:
            self.assertFalse(zone.violates_segment([-600, 350, 150], [600, 350, 150], 2))
            self.assertEqual(exact.call_count, 0)

    def test_segment_broad_phase_matches_exact_geometry(self):
        shapes = [Sphere([0, 0, 3], 2), Ellipsoid([0, 0, 3], 2, 1.5, 1),
                  Box([0, 0, 3], 4, 4, 4),
                  TriangularPyramid([0, 0, 0], 4, 4, 4),
                  QuadrangularPyramid([0, 0, 0], 4, 4, 4)]
        rng = np.random.default_rng(47)
        segments = [(rng.uniform(-6, 6, 3), rng.uniform(-6, 6, 3)) for _ in range(12)]
        segments += [([-5, 0, 2], [5, 0, 2]), ([0, 0, 2], [0, 0, 2]),
                     ([-5, 8, 2], [5, 8, 2])]
        for shape in shapes:
            for margin, radius in ((0, 0), (0.5, 0.25), (0.5, 40)):
                zone = NoFlyZone('reference', shape, margin)
                for start, end in segments:
                    with self.subTest(shape=type(shape).__name__, margin=margin, radius=radius):
                        expected = zone.segment_clearance(start, end, radius) <= GEOMETRY_TOLERANCE
                        self.assertEqual(zone.violates_segment(start, end, radius), expected)

    def test_segment_tolerance_and_overlapping_bounds_keep_exact_semantics(self):
        zone = NoFlyZone('boundary', Sphere([0, 0, 3], 1), 0.5)
        for offset, expected in ((0, True), (0.5e-9, True), (2e-9, False)):
            y = 1.75 + offset
            self.assertEqual(zone.violates_segment([-3, y, 3], [3, y, 3], 0.25), expected)
        self.assertFalse(zone.violates_segment([1.6, 1.6, 3], [1.6, 1.6, 3], 0.25))
        for start, end, radius in (([0, float('nan'), 0], [8, 8, 8], 0),
                                   ([8, 8, 8], [9, 9], 0), ([8, 8, 8], [9, 9, 9], -1)):
            with self.assertRaises(ValueError):
                zone.violates_segment(start, end, radius)

    def setUp(self) -> None:
        self.shape = Sphere(center=[0.0, 0.0, 2.0], radius=1.0)
        self.zone = NoFlyZone(
            zone_id='zone-001',
            shape=self.shape,
            safety_margin=0.5,
            metadata={'label': 'test', 'priority': 3},
        )

    def test_point_clearance_subtracts_margin_and_uav_radius(self):
        point = [2.0, 0.0, 2.0]
        self.assertAlmostEqual(self.shape.signed_distance(point), 1.0)
        self.assertAlmostEqual(self.zone.point_clearance(point), 0.5)
        self.assertAlmostEqual(self.zone.point_clearance(point, uav_radius=0.25), 0.25)

        boundary = [1.5, 0.0, 2.0]
        self.assertAlmostEqual(self.zone.point_clearance(boundary), 0.0, places=10)
        self.assertTrue(self.zone.violates_point(boundary))
        self.assertTrue(self.zone.violates_point([1.5 + 0.5 * GEOMETRY_TOLERANCE, 0.0, 2.0]))
        self.assertFalse(self.zone.violates_point([1.5 + 2.0 * GEOMETRY_TOLERANCE, 0.0, 2.0]))

    def test_segment_clearance_keeps_raw_and_safety_semantics_distinct(self):
        start = [2.0, 0.0, 2.0]
        end = [3.0, 0.0, 2.0]
        self.assertAlmostEqual(self.shape.segment_clearance(start, end), 1.0)
        self.assertAlmostEqual(self.zone.segment_clearance(start, end), 0.5)
        self.assertAlmostEqual(self.zone.segment_clearance(start, end, uav_radius=0.25), 0.25)
        self.assertFalse(self.zone.violates_segment(start, end))

        crossing_start = [-2.0, 0.0, 2.0]
        crossing_end = [2.0, 0.0, 2.0]
        self.assertEqual(self.shape.segment_clearance(crossing_start, crossing_end), 0.0)
        self.assertAlmostEqual(self.zone.segment_clearance(crossing_start, crossing_end), -0.5)
        self.assertTrue(self.zone.violates_segment(crossing_start, crossing_end))

    def test_metadata_is_copied_and_does_not_change_geometry(self):
        metadata = {'nested': {'value': 1}}
        first = NoFlyZone('first', self.shape, 0.5, metadata)
        second = NoFlyZone('second', self.shape, 0.5, {'different': True})
        baseline = first.point_clearance([2.0, 0.0, 2.0])
        metadata['nested']['value'] = 99
        returned = first.metadata
        returned['nested']['value'] = -1

        self.assertEqual(first.metadata, {'nested': {'value': 1}})
        self.assertEqual(first.point_clearance([2.0, 0.0, 2.0]), baseline)
        self.assertEqual(second.point_clearance([2.0, 0.0, 2.0]), baseline)

    def test_metadata_supports_stable_json_round_trip_and_deep_copy(self):
        metadata = {
            'none': None,
            'bool': True,
            'str': 'label',
            'int': 7,
            'float': 1.25,
            'list': [None, False, 'value', 3, 2.5, {'nested': ['original']}],
            'dict': {'enabled': True},
        }
        zone = NoFlyZone('json-zone', self.shape, 0.25, metadata)
        expected_metadata = json.loads(json.dumps(metadata))

        metadata['list'][5]['nested'][0] = 'mutated-input'
        returned = zone.metadata
        returned['dict']['enabled'] = False

        payload = json.loads(json.dumps(zone.to_dict()))
        restored = no_fly_zone_from_dict(payload)
        self.assertEqual(zone.metadata, expected_metadata)
        self.assertEqual(restored.metadata, expected_metadata)
        self.assertEqual(restored.to_dict(), payload)

    def test_metadata_rejects_non_json_values_recursively(self):
        cyclic = []
        cyclic.append(cyclic)
        invalid_metadata = (
            {'bad': {1, 2}},
            {'bad': (1, 2)},
            {'bad': np.array([1.0, 2.0])},
            {'bad': np.float64(1.0)},
            {'bad': np.int64(1)},
            {1: 'non-string-key'},
            {'bad': np.nan},
            {'bad': np.inf},
            {'bad': -np.inf},
            {'bad': object()},
            {'nested': [{'bad': np.array([1.0])}]},
            {'cycle': cyclic},
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata_type=type(next(iter(metadata.values()))).__name__):
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    NoFlyZone('invalid-metadata', self.shape, 0.0, metadata)

    def test_configuration_is_read_only(self):
        baseline_payload = self.zone.to_dict()
        baseline_clearance = self.zone.point_clearance([2.0, 0.0, 2.0])
        replacements = (
            ('zone_id', ''),
            ('shape', Sphere([10.0, 0.0, 1.0], 1.0)),
            ('safety_margin', -1.0),
        )
        for attribute, value in replacements:
            with self.subTest(attribute=attribute), self.assertRaises(AttributeError):
                setattr(self.zone, attribute, value)

        self.assertEqual(self.zone.to_dict(), baseline_payload)
        self.assertEqual(self.zone.point_clearance([2.0, 0.0, 2.0]), baseline_clearance)

    def test_invalid_zone_and_uav_values_are_rejected(self):
        for zone_id in ('', '   ', None, 42):
            with self.subTest(zone_id=zone_id), self.assertRaises(ValueError):
                NoFlyZone(zone_id, self.shape, 0.0)
        for margin in (-1.0, np.nan, np.inf):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                NoFlyZone('zone', self.shape, margin)
        with self.assertRaises(ValueError):
            NoFlyZone('zone', object(), 0.0)
        with self.assertRaises(ValueError):
            NoFlyZone('zone', self.shape, 0.0, metadata=['not', 'a', 'mapping'])

        for radius in (-1.0, np.nan, np.inf):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.zone.point_clearance([2.0, 0.0, 2.0], uav_radius=radius)
            with self.subTest(segment_radius=radius), self.assertRaises(ValueError):
                self.zone.segment_clearance([2.0, 0.0, 2.0], [3.0, 0.0, 2.0], uav_radius=radius)


if __name__ == '__main__':
    unittest.main()
