"""Tests for fixed-orientation ground-based pyramid geometry."""

from __future__ import annotations

import unittest

import numpy as np

from brain_uav.geometry import QuadrangularPyramid, SegmentHit, TriangularPyramid


class TestTriangularPyramid(unittest.TestCase):
    def setUp(self) -> None:
        self.pyramid = TriangularPyramid(
            base_center=[0.0, 0.0, 0.0],
            base_size_x=6.0,
            base_size_y=3.0,
            height=4.0,
        )

    def test_vertices_follow_fixed_centroid_formula(self):
        expected = np.array(
            [
                [-3.0, -1.0, 0.0],
                [3.0, -1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
        np.testing.assert_allclose(self.pyramid.vertices, expected)
        np.testing.assert_allclose(np.mean(self.pyramid.vertices[:3], axis=0), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(self.pyramid.vertices[3], [0.0, 0.0, 4.0])
        np.testing.assert_allclose(self.pyramid.bounding_box().min_corner, [-3.0, -1.0, 0.0])
        np.testing.assert_allclose(self.pyramid.bounding_box().max_corner, [3.0, 2.0, 4.0])

    def test_contains_base_side_edge_apex_and_interior(self):
        side_point = np.mean(self.pyramid.vertices[[0, 1, 3]], axis=0)
        edge_point = np.mean(self.pyramid.vertices[[0, 3]], axis=0)
        for point in ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], side_point, edge_point, [0.0, 0.0, 4.0]):
            with self.subTest(point=np.asarray(point).tolist()):
                self.assertTrue(self.pyramid.contains(point))
                self.assertLessEqual(self.pyramid.signed_distance(point), 1e-9)
        self.assertFalse(self.pyramid.contains([0.0, 0.0, 5.0]))
        self.assertFalse(self.pyramid.contains([-4.0, 0.0, 0.0]))

    def test_closest_point_distance_and_normal_are_consistent(self):
        cases = ([0.0, 0.0, 1.0], [0.0, 0.0, 5.0], [4.0, -2.0, 1.0], [0.0, 0.0, -1.0])
        for point in cases:
            with self.subTest(point=point):
                closest = self.pyramid.closest_point(point)
                self.assertTrue(self.pyramid.contains(closest))
                self.assertLessEqual(abs(self.pyramid.signed_distance(closest)), 1e-8)
                self.assertAlmostEqual(
                    float(np.linalg.norm(np.asarray(point) - closest)),
                    abs(self.pyramid.signed_distance(point)),
                    places=9,
                )
                normal = self.pyramid.surface_normal(point)
                self.assertTrue(np.all(np.isfinite(normal)))
                self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=10)
                np.testing.assert_allclose(normal, self.pyramid.surface_normal(point), atol=0.0)
        np.testing.assert_allclose(self.pyramid.closest_point([0.0, 0.0, 5.0]), [0.0, 0.0, 4.0])
        np.testing.assert_allclose(self.pyramid.closest_point([0.0, 0.0, -1.0]), [0.0, 0.0, 0.0])

    def test_segments_cross_base_side_touch_apex_and_miss(self):
        base_hit = self.pyramid.segment_intersection([0.0, 0.0, -2.0], [0.0, 0.0, 5.0])
        self.assertIsInstance(base_hit, SegmentHit)
        self.assertAlmostEqual(base_hit.t, 2.0 / 7.0, places=10)
        np.testing.assert_allclose(base_hit.point, [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(base_hit.normal, [0.0, 0.0, -1.0], atol=1e-10)

        side_hit = self.pyramid.segment_intersection([-10.0, 0.0, 1.0], [10.0, 0.0, 1.0])
        self.assertIsNotNone(side_hit)
        self.assertGreater(side_hit.t, 0.0)
        self.assertLess(side_hit.t, 1.0)

        apex_touch = self.pyramid.segment_intersection([-1.0, 0.0, 4.0], [1.0, 0.0, 4.0])
        self.assertIsNotNone(apex_touch)
        self.assertAlmostEqual(apex_touch.t, 0.5, places=9)
        np.testing.assert_allclose(apex_touch.point, [0.0, 0.0, 4.0], atol=1e-8)

        self.assertIsNone(self.pyramid.segment_intersection([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0]))
        self.assertAlmostEqual(
            self.pyramid.segment_clearance([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0]), 1.0, places=8
        )

        inside = self.pyramid.segment_intersection([0.0, 0.0, 1.0], [1000.0, 0.0, 1.0])
        self.assertIsNotNone(inside)
        self.assertEqual(inside.t, 0.0)
        self.assertTrue(inside.starts_inside)

    def test_invalid_base_and_degenerate_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            TriangularPyramid([0.0, 0.0, 0.1], 6.0, 3.0, 4.0)
        for values in ((0.0, 3.0, 4.0), (6.0, -1.0, 4.0), (6.0, 3.0, np.inf)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TriangularPyramid([0.0, 0.0, 0.0], *values)
        with self.assertRaises(ValueError):
            self.pyramid.contains([0.0, np.nan, 1.0])

    def test_dimensions_are_read_only_and_cannot_desynchronize_vertices(self):
        baseline_vertices = self.pyramid.vertices
        baseline_payload = self.pyramid.to_dict()
        baseline_distance = self.pyramid.signed_distance([0.0, 0.0, 5.0])
        for attribute, value in (('base_size_x', 12.0), ('base_size_y', 9.0), ('height', 10.0)):
            with self.subTest(attribute=attribute), self.assertRaises(AttributeError):
                setattr(self.pyramid, attribute, value)

        np.testing.assert_array_equal(self.pyramid.vertices, baseline_vertices)
        self.assertEqual(self.pyramid.to_dict(), baseline_payload)
        self.assertEqual(self.pyramid.signed_distance([0.0, 0.0, 5.0]), baseline_distance)


class TestQuadrangularPyramid(unittest.TestCase):
    def setUp(self) -> None:
        self.pyramid = QuadrangularPyramid(
            base_center=[1.0, 2.0, 0.0],
            base_size_x=4.0,
            base_size_y=6.0,
            height=5.0,
        )

    def test_rectangular_and_square_base_vertices_are_fixed(self):
        expected = np.array(
            [
                [-1.0, -1.0, 0.0],
                [3.0, -1.0, 0.0],
                [3.0, 5.0, 0.0],
                [-1.0, 5.0, 0.0],
                [1.0, 2.0, 5.0],
            ]
        )
        np.testing.assert_allclose(self.pyramid.vertices, expected)
        np.testing.assert_allclose(np.mean(self.pyramid.vertices[:4], axis=0), [1.0, 2.0, 0.0])
        np.testing.assert_allclose(self.pyramid.vertices[4], [1.0, 2.0, 5.0])
        np.testing.assert_allclose(self.pyramid.bounding_box().min_corner, [-1.0, -1.0, 0.0])
        np.testing.assert_allclose(self.pyramid.bounding_box().max_corner, [3.0, 5.0, 5.0])

        square = QuadrangularPyramid([0.0, 0.0, 0.0], 4.0, 4.0, 3.0)
        np.testing.assert_allclose(square.vertices[0], [-2.0, -2.0, 0.0])
        np.testing.assert_allclose(square.vertices[2], [2.0, 2.0, 0.0])

    def test_contains_base_side_edge_apex_and_interior(self):
        side_point = np.mean(self.pyramid.vertices[[0, 1, 4]], axis=0)
        edge_point = np.mean(self.pyramid.vertices[[0, 4]], axis=0)
        for point in ([1.0, 2.0, 1.0], [1.0, 2.0, 0.0], side_point, edge_point, [1.0, 2.0, 5.0]):
            with self.subTest(point=np.asarray(point).tolist()):
                self.assertTrue(self.pyramid.contains(point))
                self.assertLessEqual(self.pyramid.signed_distance(point), 1e-9)
        self.assertFalse(self.pyramid.contains([1.0, 2.0, 6.0]))
        self.assertFalse(self.pyramid.contains([4.0, 2.0, 0.0]))

    def test_closest_point_distance_and_normal_are_consistent(self):
        cases = ([1.0, 2.0, 1.0], [1.0, 2.0, 6.0], [5.0, 6.0, 1.0], [1.0, 2.0, -1.0])
        for point in cases:
            with self.subTest(point=point):
                closest = self.pyramid.closest_point(point)
                self.assertTrue(self.pyramid.contains(closest))
                self.assertLessEqual(abs(self.pyramid.signed_distance(closest)), 1e-8)
                self.assertAlmostEqual(
                    float(np.linalg.norm(np.asarray(point) - closest)),
                    abs(self.pyramid.signed_distance(point)),
                    places=9,
                )
                normal = self.pyramid.surface_normal(point)
                self.assertTrue(np.all(np.isfinite(normal)))
                self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=10)
                np.testing.assert_allclose(normal, self.pyramid.surface_normal(point), atol=0.0)
        np.testing.assert_allclose(self.pyramid.closest_point([1.0, 2.0, 6.0]), [1.0, 2.0, 5.0])
        np.testing.assert_allclose(self.pyramid.closest_point([1.0, 2.0, -1.0]), [1.0, 2.0, 0.0])

    def test_segments_cross_base_side_touch_apex_and_miss(self):
        base_hit = self.pyramid.segment_intersection([1.0, 2.0, -2.0], [1.0, 2.0, 7.0])
        self.assertIsNotNone(base_hit)
        self.assertAlmostEqual(base_hit.t, 2.0 / 9.0, places=10)
        np.testing.assert_allclose(base_hit.normal, [0.0, 0.0, -1.0], atol=1e-10)

        side_hit = self.pyramid.segment_intersection([-10.0, 2.0, 1.0], [10.0, 2.0, 1.0])
        self.assertIsNotNone(side_hit)

        apex_touch = self.pyramid.segment_intersection([0.0, 2.0, 5.0], [2.0, 2.0, 5.0])
        self.assertIsNotNone(apex_touch)
        self.assertAlmostEqual(apex_touch.t, 0.5, places=9)
        np.testing.assert_allclose(apex_touch.point, [1.0, 2.0, 5.0], atol=1e-8)

        self.assertIsNone(self.pyramid.segment_intersection([0.0, 2.0, 6.0], [2.0, 2.0, 6.0]))
        self.assertAlmostEqual(
            self.pyramid.segment_clearance([0.0, 2.0, 6.0], [2.0, 2.0, 6.0]), 1.0, places=8
        )

        inside = self.pyramid.segment_intersection([1.0, 2.0, 1.0], [1000.0, 2.0, 1.0])
        self.assertIsNotNone(inside)
        self.assertEqual(inside.t, 0.0)
        self.assertTrue(inside.starts_inside)

    def test_invalid_base_and_degenerate_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            QuadrangularPyramid([0.0, 0.0, -0.1], 4.0, 6.0, 5.0)
        for values in ((0.0, 6.0, 5.0), (4.0, 0.0, 5.0), (4.0, 6.0, -1.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                QuadrangularPyramid([0.0, 0.0, 0.0], *values)
        with self.assertRaises(ValueError):
            self.pyramid.segment_intersection([0.0, 0.0], [1.0, 2.0, 3.0])

    def test_dimensions_are_read_only_and_cannot_desynchronize_vertices(self):
        baseline_vertices = self.pyramid.vertices
        baseline_payload = self.pyramid.to_dict()
        baseline_distance = self.pyramid.signed_distance([1.0, 2.0, 6.0])
        for attribute, value in (('base_size_x', 12.0), ('base_size_y', 9.0), ('height', 10.0)):
            with self.subTest(attribute=attribute), self.assertRaises(AttributeError):
                setattr(self.pyramid, attribute, value)

        np.testing.assert_array_equal(self.pyramid.vertices, baseline_vertices)
        self.assertEqual(self.pyramid.to_dict(), baseline_payload)
        self.assertEqual(self.pyramid.signed_distance([1.0, 2.0, 6.0]), baseline_distance)


if __name__ == '__main__':
    unittest.main()
