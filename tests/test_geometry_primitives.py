"""Tests for axis-aligned analytic geometry primitives."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from brain_uav.geometry import GEOMETRY_TOLERANCE, AABB, Box, Ellipsoid, SegmentHit, Sphere


class TestSphere(unittest.TestCase):
    def setUp(self) -> None:
        self.sphere = Sphere(center=[0.0, 0.0, 2.0], radius=2.0)

    def test_query_contract_for_center_inside_boundary_and_outside(self):
        cases = (
            ([0.0, 0.0, 2.0], True, -2.0, [2.0, 0.0, 2.0]),
            ([1.0, 0.0, 2.0], True, -1.0, [2.0, 0.0, 2.0]),
            ([2.0, 0.0, 2.0], True, 0.0, [2.0, 0.0, 2.0]),
            ([3.0, 0.0, 2.0], False, 1.0, [2.0, 0.0, 2.0]),
        )
        for point, contains, signed_distance, closest in cases:
            with self.subTest(point=point):
                self.assertEqual(self.sphere.contains(point), contains)
                self.assertAlmostEqual(self.sphere.signed_distance(point), signed_distance, places=10)
                np.testing.assert_allclose(self.sphere.closest_point(point), closest, atol=1e-10)
                normal = self.sphere.surface_normal(point)
                self.assertTrue(np.all(np.isfinite(normal)))
                self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=10)

    def test_segments_cover_crossing_tangent_miss_and_inside_start(self):
        crossing = self.sphere.segment_intersection([-3.0, 0.0, 2.0], [3.0, 0.0, 2.0])
        self.assertIsInstance(crossing, SegmentHit)
        self.assertAlmostEqual(crossing.t, 1.0 / 6.0, places=10)
        np.testing.assert_allclose(crossing.point, [-2.0, 0.0, 2.0], atol=1e-10)
        np.testing.assert_allclose(crossing.normal, [-1.0, 0.0, 0.0], atol=1e-10)
        self.assertFalse(crossing.starts_inside)
        self.assertEqual(self.sphere.segment_clearance([-3.0, 0.0, 2.0], [3.0, 0.0, 2.0]), 0.0)

        tangent = self.sphere.segment_intersection([-3.0, 2.0, 2.0], [3.0, 2.0, 2.0])
        self.assertIsNotNone(tangent)
        self.assertAlmostEqual(tangent.t, 0.5, places=10)
        np.testing.assert_allclose(tangent.point, [0.0, 2.0, 2.0], atol=1e-10)

        self.assertIsNone(self.sphere.segment_intersection([-3.0, 2.5, 2.0], [3.0, 2.5, 2.0]))
        self.assertAlmostEqual(
            self.sphere.segment_clearance([-3.0, 2.5, 2.0], [3.0, 2.5, 2.0]), 0.5, places=10
        )

        inside = self.sphere.segment_intersection([0.0, 0.0, 2.0], [100.0, 0.0, 2.0])
        self.assertIsNotNone(inside)
        self.assertEqual(inside.t, 0.0)
        np.testing.assert_allclose(inside.point, [0.0, 0.0, 2.0], atol=0.0)
        self.assertTrue(inside.starts_inside)

    def test_high_speed_segment_cannot_skip_sphere(self):
        hit = self.sphere.segment_intersection([-1000.0, 0.0, 2.0], [1000.0, 0.0, 2.0])
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.t, 0.499, places=10)
        np.testing.assert_allclose(hit.point, [-2.0, 0.0, 2.0], atol=1e-9)

    def test_long_near_tangent_segment_remains_disjoint(self):
        sphere = Sphere(center=[0.0, 0.0, 1.0], radius=1.0)
        start = [-1000.0, 1.000001, 1.0]
        end = [1000.0, 1.000001, 1.0]

        self.assertIsNone(sphere.segment_intersection(start, end))
        self.assertAlmostEqual(sphere.segment_clearance(start, end), 1e-6, places=10)

    def test_bounding_box_and_ground_constraints(self):
        bounds = self.sphere.bounding_box()
        self.assertIsInstance(bounds, AABB)
        np.testing.assert_allclose(bounds.min_corner, [-2.0, -2.0, 0.0])
        np.testing.assert_allclose(bounds.max_corner, [2.0, 2.0, 4.0])
        self.assertTrue(bounds.contains([0.0, 0.0, 2.0]))
        self.assertFalse(bounds.contains([2.1, 0.0, 2.0]))

        Sphere(center=[0.0, 0.0, 3.0], radius=1.0)
        with self.assertRaises(ValueError):
            Sphere(center=[0.0, 0.0, 0.5], radius=1.0)

    def test_invalid_inputs_are_rejected(self):
        for center in ([0.0, 0.0], [0.0, np.nan, 2.0], [0.0, 0.0, np.inf]):
            with self.subTest(center=center), self.assertRaises(ValueError):
                Sphere(center=center, radius=1.0)
        for radius in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                Sphere(center=[0.0, 0.0, 2.0], radius=radius)
        for point in ([0.0, 0.0], [0.0, np.nan, 2.0], [0.0, 0.0, np.inf]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                self.sphere.signed_distance(point)
        with self.assertRaises(ValueError):
            self.sphere.segment_intersection([0.0, 0.0], [1.0, 0.0, 0.0])

    def test_radius_is_read_only(self):
        baseline_distance = self.sphere.signed_distance([3.0, 0.0, 2.0])
        baseline_bounds = self.sphere.bounding_box()
        baseline_payload = self.sphere.to_dict()

        with self.assertRaises(AttributeError):
            self.sphere.radius = -5.0

        self.assertEqual(self.sphere.radius, 2.0)
        self.assertEqual(self.sphere.signed_distance([3.0, 0.0, 2.0]), baseline_distance)
        np.testing.assert_array_equal(self.sphere.bounding_box().min_corner, baseline_bounds.min_corner)
        np.testing.assert_array_equal(self.sphere.bounding_box().max_corner, baseline_bounds.max_corner)
        self.assertEqual(self.sphere.to_dict(), baseline_payload)

    def test_contains_matches_signed_distance_tolerance(self):
        point = [2.0 + 0.5 * GEOMETRY_TOLERANCE, 0.0, 2.0]
        self.assertEqual(
            self.sphere.contains(point),
            self.sphere.signed_distance(point) <= GEOMETRY_TOLERANCE,
        )


class TestEllipsoid(unittest.TestCase):
    def setUp(self) -> None:
        self.ellipsoid = Ellipsoid(
            center=[0.0, 0.0, 3.0],
            radius_x=3.0,
            radius_y=2.0,
            radius_z=1.0,
        )

    def assert_projection_consistent(self, ellipsoid, point, *, inside):
        point = np.asarray(point, dtype=np.float64)
        closest = ellipsoid.closest_point(point)
        relative = closest - ellipsoid.center
        surface_error = abs(float(np.sum((relative / ellipsoid.radii) ** 2)) - 1.0)
        self.assertLessEqual(surface_error, 1e-9)

        distance = float(np.linalg.norm(point - closest))
        signed_distance = ellipsoid.signed_distance(point)
        self.assertAlmostEqual(abs(signed_distance), distance, places=9)
        self.assertLess(signed_distance, 0.0) if inside else self.assertGreater(signed_distance, 0.0)

        normal = ellipsoid.surface_normal(point)
        self.assertTrue(np.all(np.isfinite(normal)))
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=10)
        normal_direction = closest - point if inside else point - closest
        direction_norm = float(np.linalg.norm(normal_direction))
        self.assertGreater(direction_norm, 0.0)
        self.assertGreater(float(np.dot(normal_direction, normal)), 0.0)
        self.assertLessEqual(
            float(np.linalg.norm(np.cross(normal_direction, normal))) / direction_norm,
            1e-9,
        )

        expected_clearance = 0.0 if inside else distance
        self.assertAlmostEqual(
            ellipsoid.segment_clearance(point, point),
            expected_clearance,
            places=9,
        )
        return closest

    def test_axis_distances_center_and_ground_constraints(self):
        self.assertAlmostEqual(self.ellipsoid.signed_distance([4.0, 0.0, 3.0]), 1.0, places=10)
        self.assertAlmostEqual(self.ellipsoid.signed_distance([0.0, 0.0, 4.5]), 0.5, places=10)
        self.assertAlmostEqual(self.ellipsoid.signed_distance([0.0, 0.0, 3.0]), -1.0, places=10)
        np.testing.assert_allclose(self.ellipsoid.closest_point([0.0, 0.0, 3.0]), [0.0, 0.0, 4.0])
        np.testing.assert_allclose(self.ellipsoid.bounding_box().min_corner, [-3.0, -2.0, 2.0])
        np.testing.assert_allclose(self.ellipsoid.bounding_box().max_corner, [3.0, 2.0, 4.0])

        Ellipsoid(center=[0.0, 0.0, 1.0], radius_x=2.0, radius_y=1.5, radius_z=1.0)
        Ellipsoid(center=[0.0, 0.0, 5.0], radius_x=2.0, radius_y=1.5, radius_z=1.0)
        with self.assertRaises(ValueError):
            Ellipsoid(center=[0.0, 0.0, 0.5], radius_x=2.0, radius_y=1.5, radius_z=1.0)

    def test_non_axis_closest_point_is_true_euclidean_projection(self):
        point = np.array([4.0, 3.0, 5.0])
        closest = self.ellipsoid.closest_point(point)
        relative = closest - np.array([0.0, 0.0, 3.0])
        radii = np.array([3.0, 2.0, 1.0])
        self.assertAlmostEqual(float(np.sum((relative / radii) ** 2)), 1.0, places=10)
        self.assertAlmostEqual(
            float(np.linalg.norm(point - closest)),
            abs(self.ellipsoid.signed_distance(point)),
            places=10,
        )
        gradient = relative / radii**2
        self.assertLess(float(np.linalg.norm(np.cross(point - closest, gradient))), 1e-9)

        point_relative = point - np.array([0.0, 0.0, 3.0])
        radial = point_relative / np.sqrt(np.sum((point_relative / radii) ** 2))
        radial_point = np.array([0.0, 0.0, 3.0]) + radial
        self.assertLess(
            float(np.linalg.norm(point - closest)),
            float(np.linalg.norm(point - radial_point)) - 1e-5,
        )

    def test_internal_non_axis_projection_and_normal_are_consistent(self):
        point = np.array([0.2, 0.2, 3.2])
        self.assertTrue(self.ellipsoid.contains(point))
        closest = self.ellipsoid.closest_point(point)
        distance = float(np.linalg.norm(point - closest))
        self.assertAlmostEqual(distance, abs(self.ellipsoid.signed_distance(point)), places=10)
        self.assertAlmostEqual(
            float(np.sum(((closest - self.ellipsoid.center) / self.ellipsoid.radii) ** 2)),
            1.0,
            places=10,
        )
        normal = self.ellipsoid.surface_normal(point)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=10)
        self.assertGreater(float(np.dot(closest - point, normal)), 0.0)

    def test_distance_and_normal_share_one_exact_projection(self):
        original = self.ellipsoid.closest_point
        cases = (
            np.array([4.0, 3.0, 5.0]),
            np.array([0.2, 0.2, 3.2]),
            np.array([3.0, 0.0, 3.0]),
        )
        for point in cases:
            with self.subTest(point=point.tolist()):
                expected_distance = self.ellipsoid.signed_distance(point)
                expected_normal = self.ellipsoid.surface_normal(point)
                with mock.patch.object(
                    self.ellipsoid,
                    'closest_point',
                    wraps=original,
                ) as closest_point:
                    distance, normal = (
                        self.ellipsoid.signed_distance_and_surface_normal(point)
                    )
                self.assertEqual(closest_point.call_count, 1)
                self.assertEqual(distance, expected_distance)
                np.testing.assert_array_equal(normal, expected_normal)

    def test_internal_projection_remains_finite_at_inactive_axis_pole(self):
        ellipsoid = Ellipsoid(
            center=[0.0, 0.0, 3.0],
            radius_x=np.sqrt(8.0),
            radius_y=2.0,
            radius_z=1.0,
        )
        point = np.array([np.sqrt(2.0), 0.0, 3.0])

        closest = ellipsoid.closest_point(point)

        self.assertTrue(np.all(np.isfinite(closest)))
        self.assertAlmostEqual(
            float(np.sum(((closest - ellipsoid.center) / ellipsoid.radii) ** 2)),
            1.0,
            places=10,
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(point - closest)),
            abs(ellipsoid.signed_distance(point)),
            places=10,
        )

    def test_high_aspect_internal_projection_regression(self):
        ellipsoid = Ellipsoid(
            center=[0.0, 0.0, 7.724310944294375],
            radius_x=0.2158398687965686,
            radius_y=4.994807605965346,
            radius_z=6.724310944294375,
        )
        point = [
            -1.187837638188122e-05,
            0.0004166846805543608,
            7.724385444031308,
        ]

        self.assert_projection_consistent(ellipsoid, point, inside=True)

    def test_high_aspect_projection_fixed_cases(self):
        ellipsoid = Ellipsoid([0.0, 0.0, 8.0], 0.2, 5.0, 7.0)
        cases = (
            ([1e-7, 2e-4, 8.0003], True),
            ([0.0, 0.0, 8.0], True),
            ([0.0, 1.0, 8.0], True),
            ([0.5, 6.0, 16.0], False),
        )
        for point, inside in cases:
            with self.subTest(point=point):
                self.assert_projection_consistent(ellipsoid, point, inside=inside)

    def test_projection_consistency_for_fixed_seed_points(self):
        ellipsoid = Ellipsoid([0.0, 0.0, 8.0], 0.2, 5.0, 7.0)
        rng = np.random.default_rng(20260903)
        for index in range(32):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            scale = float(rng.uniform(0.05, 0.9) if index % 2 == 0 else rng.uniform(1.1, 2.0))
            point = ellipsoid.center + ellipsoid.radii * direction * scale
            with self.subTest(index=index, scale=scale):
                self.assert_projection_consistent(ellipsoid, point, inside=scale < 1.0)

    def test_segments_cover_crossing_tangent_miss_and_inside_start(self):
        crossing = self.ellipsoid.segment_intersection([-5.0, 0.0, 3.0], [5.0, 0.0, 3.0])
        self.assertIsNotNone(crossing)
        self.assertAlmostEqual(crossing.t, 0.2, places=10)
        np.testing.assert_allclose(crossing.point, [-3.0, 0.0, 3.0], atol=1e-10)

        tangent = self.ellipsoid.segment_intersection([-5.0, 2.0, 3.0], [5.0, 2.0, 3.0])
        self.assertIsNotNone(tangent)
        self.assertAlmostEqual(tangent.t, 0.5, places=10)
        self.assertEqual(self.ellipsoid.segment_clearance([-5.0, 2.0, 3.0], [5.0, 2.0, 3.0]), 0.0)

        self.assertIsNone(self.ellipsoid.segment_intersection([-5.0, 2.5, 3.0], [5.0, 2.5, 3.0]))
        self.assertAlmostEqual(
            self.ellipsoid.segment_clearance([-5.0, 2.5, 3.0], [5.0, 2.5, 3.0]), 0.5, places=8
        )

        inside = self.ellipsoid.segment_intersection([0.0, 0.0, 3.0], [1000.0, 0.0, 3.0])
        self.assertIsNotNone(inside)
        self.assertEqual(inside.t, 0.0)
        self.assertTrue(inside.starts_inside)

    def test_long_near_tangent_segment_remains_disjoint(self):
        ellipsoid = Ellipsoid(
            center=[0.0, 0.0, 1.0],
            radius_x=2.0,
            radius_y=1.0,
            radius_z=1.0,
        )
        start = [-1000.0, 1.000001, 1.0]
        end = [1000.0, 1.000001, 1.0]

        self.assertIsNone(ellipsoid.segment_intersection(start, end))
        self.assertAlmostEqual(ellipsoid.segment_clearance(start, end), 1e-6, places=9)

    def test_invalid_radii_and_queries_are_rejected(self):
        for name in ('radius_x', 'radius_y', 'radius_z'):
            values = {'radius_x': 3.0, 'radius_y': 2.0, 'radius_z': 1.0}
            values[name] = 0.0
            with self.subTest(name=name), self.assertRaises(ValueError):
                Ellipsoid(center=[0.0, 0.0, 3.0], **values)
        with self.assertRaises(ValueError):
            self.ellipsoid.closest_point([np.nan, 0.0, 3.0])


class TestBox(unittest.TestCase):
    def setUp(self) -> None:
        self.box = Box(center=[0.0, 0.0, 1.0], size_x=4.0, size_y=6.0, size_z=2.0)

    def test_face_edge_vertex_and_internal_distances(self):
        cases = (
            ([3.0, 0.0, 1.0], 1.0, [2.0, 0.0, 1.0]),
            ([3.0, 4.0, 1.0], np.sqrt(2.0), [2.0, 3.0, 1.0]),
            ([3.0, 4.0, 3.0], np.sqrt(3.0), [2.0, 3.0, 2.0]),
            ([0.0, 0.0, 1.0], -1.0, [0.0, 0.0, 2.0]),
            ([1.5, 0.0, 1.0], -0.5, [2.0, 0.0, 1.0]),
            ([2.0, 0.0, 1.0], 0.0, [2.0, 0.0, 1.0]),
        )
        for point, distance, closest in cases:
            with self.subTest(point=point):
                self.assertAlmostEqual(self.box.signed_distance(point), distance, places=10)
                np.testing.assert_allclose(self.box.closest_point(point), closest, atol=1e-10)
                self.assertAlmostEqual(
                    float(np.linalg.norm(np.asarray(point) - self.box.closest_point(point))),
                    abs(distance),
                    places=10,
                )
        np.testing.assert_allclose(
            self.box.surface_normal([3.0, 4.0, 3.0]),
            np.ones(3) / np.sqrt(3.0),
            atol=1e-10,
        )
        np.testing.assert_allclose(self.box.surface_normal([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
        np.testing.assert_allclose(self.box.surface_normal([1.5, 0.0, 1.0]), [1.0, 0.0, 0.0])

    def test_cube_rectangular_box_ground_and_suspended_construction(self):
        cube = Box(center=[1.0, 2.0, 1.0], size_x=2.0, size_y=2.0, size_z=2.0)
        np.testing.assert_allclose(cube.half_sizes, [1.0, 1.0, 1.0])
        suspended = Box(center=[0.0, 0.0, 5.0], size_x=2.0, size_y=4.0, size_z=2.0)
        np.testing.assert_allclose(suspended.bounding_box().min_corner, [-1.0, -2.0, 4.0])
        np.testing.assert_allclose(self.box.bounding_box().min_corner, [-2.0, -3.0, 0.0])
        np.testing.assert_allclose(self.box.bounding_box().max_corner, [2.0, 3.0, 2.0])
        with self.assertRaises(ValueError):
            Box(center=[0.0, 0.0, 0.5], size_x=2.0, size_y=2.0, size_z=2.0)

    def test_slab_intersection_tangent_miss_and_high_speed_crossing(self):
        crossing = self.box.segment_intersection([-5.0, 0.0, 1.0], [5.0, 0.0, 1.0])
        self.assertIsNotNone(crossing)
        self.assertAlmostEqual(crossing.t, 0.3, places=10)
        np.testing.assert_allclose(crossing.point, [-2.0, 0.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(crossing.normal, [-1.0, 0.0, 0.0], atol=1e-10)

        tangent = self.box.segment_intersection([-5.0, 3.0, 1.0], [5.0, 3.0, 1.0])
        self.assertIsNotNone(tangent)
        self.assertEqual(self.box.segment_clearance([-5.0, 3.0, 1.0], [5.0, 3.0, 1.0]), 0.0)

        self.assertIsNone(self.box.segment_intersection([-5.0, 3.5, 1.0], [5.0, 3.5, 1.0]))
        self.assertAlmostEqual(self.box.segment_clearance([-5.0, 3.5, 1.0], [5.0, 3.5, 1.0]), 0.5, places=8)

        high_speed = self.box.segment_intersection([-1000.0, 0.0, 1.0], [1000.0, 0.0, 1.0])
        self.assertIsNotNone(high_speed)
        self.assertAlmostEqual(high_speed.t, 0.499, places=10)

        inside = self.box.segment_intersection([0.0, 0.0, 1.0], [1000.0, 0.0, 1.0])
        self.assertIsNotNone(inside)
        self.assertEqual(inside.t, 0.0)
        self.assertTrue(inside.starts_inside)

    def test_invalid_sizes_and_queries_are_rejected(self):
        for name in ('size_x', 'size_y', 'size_z'):
            values = {'size_x': 4.0, 'size_y': 6.0, 'size_z': 2.0}
            values[name] = -1.0
            with self.subTest(name=name), self.assertRaises(ValueError):
                Box(center=[0.0, 0.0, 2.0], **values)
        with self.assertRaises(ValueError):
            self.box.segment_clearance([0.0, 0.0, 1.0], [np.inf, 0.0, 1.0])


if __name__ == '__main__':
    unittest.main()
