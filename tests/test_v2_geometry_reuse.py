"""Exact local geometry reuse checks; no CUDA or throughput claims."""

from unittest import TestCase, mock

import numpy as np

from brain_uav.geometry import Ellipsoid, NoFlyZone, TriangularPyramid, QuadrangularPyramid
from brain_uav.config import RewardConfig
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from test_v2_static_no_fly_env import _config, _scenario


class TestPyramidCombinedGeometry(TestCase):
    def test_exact_combination_one_solve_and_owned_normal(self):
        for shape_type in (TriangularPyramid, QuadrangularPyramid):
            shape = shape_type([0., 0., 0.], 8., 8., 12.)
            # Interior, base surface, outside, symmetric/tied apex and base edge.
            points = [[0., 0., 2.], [0., 0., 6.], [0., 0., 0.], [9., 3., 4.],
                      [0., 0., 12.], shape.vertices[0]]
            for point in points:
                with self.subTest(shape=shape_type.__name__, point=point):
                    with mock.patch.object(shape, '_closest_boundary',
                                           wraps=shape._closest_boundary) as solve:
                        distance = shape.signed_distance(point)
                        normal = shape.surface_normal(point)
                        self.assertEqual(solve.call_count, 2)
                        solve.reset_mock()
                        combined_distance, combined_normal = (
                            shape.signed_distance_and_surface_normal(point))
                        self.assertEqual(solve.call_count, 1)
                    self.assertEqual(distance, combined_distance)
                    np.testing.assert_array_equal(normal, combined_normal)
                    for face in shape._faces:
                        self.assertFalse(np.shares_memory(combined_normal, face.normal))
                    combined_normal[:] = 123.
                    np.testing.assert_array_equal(shape.surface_normal(point), normal)


class TestEllipsoidIntersectionReuse(TestCase):
    def test_raw_intersection_clearance_and_safety_are_exact(self):
        # Cross, miss near the safety boundary, tangent, zero length in/out.
        segments = [([-3., 0., 10.], [3., 0., 10.]),
                    ([-3., 1.1, 10.], [3., 1.1, 10.]),
                    ([-3., 1., 10.], [3., 1., 10.]),
                    ([0., 0., 10.], [0., 0., 10.]),
                    ([0., 1.1, 10.], [0., 1.1, 10.]),
                    ([-3., 60., 10.], [3., 60., 10.])]
        for cache in (False, True):
            for start, end in segments:
                shape = Ellipsoid([0., 0., 10.], 2., 1., 1.)
                shape.set_segment_clearance_cache_enabled(cache)
                hit = shape.segment_intersection(start, end)
                captured_hit, result = shape._segment_intersection_with_result(start, end)
                self.assertEqual(hit is None, captured_hit is None)
                if hit is not None:
                    self.assertEqual(hit.t, captured_hit.t)
                    np.testing.assert_array_equal(hit.point, captured_hit.point)
                    np.testing.assert_array_equal(hit.normal, captured_hit.normal)
                expected = shape.segment_clearance(start, end)
                shape.clear_segment_clearance_cache()
                with mock.patch.object(shape, 'segment_intersection',
                                       wraps=shape.segment_intersection) as query:
                    self.assertEqual(shape.segment_clearance(
                        start, end, _intersection_result=result), expected)
                    query.assert_not_called()
                zone = NoFlyZone('e', shape, safety_margin=.2)
                for extra in (0., 40.):
                    for radius in (0., .25):
                        shape.clear_segment_clearance_cache()
                        expected_clearance = zone.segment_clearance(start, end, radius + extra)
                        expected_safe = zone.violates_segment(start, end, radius + extra)
                        shape.clear_segment_clearance_cache()
                        self.assertEqual(zone.segment_clearance(start, end, radius + extra,
                            _intersection_result=result), expected_clearance)
                        self.assertEqual(zone.violates_segment(start, end, radius + extra,
                            _intersection_result=result), expected_safe)
                if start == [-3., 1.1, 10.]:
                    self.assertIsNone(hit)
                    self.assertTrue(zone.violates_segment(start, end, .25,
                        _intersection_result=result))

    def test_mismatches_fall_back_and_result_does_not_share_arrays(self):
        start = np.array([-3., 0., 10.]); end = np.array([3., 0., 10.])
        shape = Ellipsoid([0., 0., 10.], 2., 1., 1.)
        hit, result = shape._segment_intersection_with_result(start, end)
        for array in (hit.point, hit.normal):
            array.setflags(write=True)
            array[:] = 999.
        # Hints hold only immutable query snapshots and a boolean, never hit arrays.
        with mock.patch.object(shape, 'segment_intersection', wraps=shape.segment_intersection) as query:
            self.assertEqual(shape.segment_clearance(start, end, _intersection_result=result), 0.)
            query.assert_not_called()
        for change in ('start', 'end', 'shape', 'center', 'radii'):
            with self.subTest(change=change):
                shape = Ellipsoid([0., 0., 10.], 2., 1., 1.)
                a, b = start.copy(), end.copy()
                _, result = shape._segment_intersection_with_result(a, b)
                if change == 'start': a[0] += .1
                if change == 'end': b[0] += .1
                if change == 'shape': shape = Ellipsoid([0., 0., 10.], 2., 1., 1.)
                if change == 'center': shape._center[1] += .1
                if change == 'radii': shape._radii[1] += .1
                expected = shape.segment_clearance(a, b)
                with mock.patch.object(shape, 'segment_intersection', wraps=shape.segment_intersection) as query:
                    self.assertEqual(shape.segment_clearance(a, b, _intersection_result=result), expected)
                    query.assert_called_once()
        # Mutating caller-owned endpoints cannot mutate the snapshot.
        shape = Ellipsoid([0., 0., 10.], 2., 1., 1.)
        _, result = shape._segment_intersection_with_result(start, end)
        start[1] = 50.; end[1] = 50.
        self.assertGreater(shape.segment_clearance(start, end, _intersection_result=result), 0.)

    def test_invalid_hint_recomputes_even_with_a_prior_clearance_cache(self):
        shape = Ellipsoid([0., 0., 10.], 2., 1., 1.)
        shape.set_segment_clearance_cache_enabled(True)
        start, end = [-3., 0., 10.], [3., 0., 10.]
        _, result = shape._segment_intersection_with_result(start, end)
        self.assertEqual(shape.segment_clearance(start, end), 0.)
        shape._center[1] = 5.
        reference = Ellipsoid([0., 5., 10.], 2., 1., 1.)
        with mock.patch.object(shape, 'segment_intersection', wraps=shape.segment_intersection) as query:
            self.assertEqual(shape.segment_clearance(start, end, _intersection_result=result),
                             reference.segment_clearance(start, end))
            query.assert_called_once()


class TestEnvironmentIntersectionReuse(TestCase):
    def env(self, enabled):
        return V2StaticNoFlyTrajectoryEnv(_config(world_xy=1000.), RewardConfig(
            descent_three_band_enabled=True, boundary_directional_warning_enabled=True),
            seed=7, uav_collision_radius=.25, cpu_deduplication=enabled,
            fixed_scenarios=[_scenario([-5., 0., 10., 0., 0.], [330., 0., 10.],
                [NoFlyZone('e', Ellipsoid([10., 0., 10.], 2., 1., 1.), .2)],
                metadata={'direct_path_blocker_count': 1})])

    def assert_observation(self, a, b):
        for name in ('ego_features', 'goal_features', 'zone_features', 'presence_mask'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_real_lifecycle_call_reduction_and_no_residual_result(self):
        counts = []
        for enabled in (False, True):
            env = self.env(enabled)
            calls = []
            original = Ellipsoid.segment_intersection
            def query(shape, start, end):
                calls.append((np.array(start), np.array(end)))
                return original(shape, start, end)
            rows = []
            with mock.patch.object(Ellipsoid, 'segment_intersection', autospec=True,
                                   side_effect=query):
                for operation in (env.reset, lambda: env.step(np.zeros(2)),
                                  lambda: env.set_goal([340., 0., 10.]), env.reset):
                    calls.clear()
                    operation()
                    rows.append(sum(np.array_equal(a, env.state[:3]) and
                                    np.array_equal(b, env.goal) for a, b in calls))
                    self.assertFalse(any('intersection' in key for key in vars(env)))
                    # A later standalone info call has no result from the prior call.
                    env.zones[0].shape.clear_segment_clearance_cache()
                    calls.clear()
                    env._info(progress=0.)
                    self.assertEqual(len(calls), 1)
            counts.append(rows)
        self.assertEqual(counts, [[2, 2, 2, 2], [1, 1, 1, 1]])

    def test_fixed_actions_and_external_changes_are_exact(self):
        a, b = self.env(False), self.env(True)
        oa, ia = a.reset(); ob, ib = b.reset()
        self.assert_observation(oa, ob); self.assertEqual(ia, ib)
        old_zones = ob.zone_features.copy()
        for index in range(8):
            if index == 2:
                for env in (a, b):
                    env.zones[0].shape._center[1] = 1.5
                    env.zones[0].shape.clear_segment_clearance_cache()
                    env.uav_collision_radius = .5
                    env.zones[0]._safety_margin = .4
            if index == 4:
                for env in (a, b):
                    env.zones[0]._shape = Ellipsoid([10., 1., 10.], 1., 2., 1.)
            if index == 6:
                for env in (a, b):
                    env.zones = [NoFlyZone('new', Ellipsoid([12., 3., 10.], 1., 1., 2.))]
            self.assertEqual(a.line_to_goal_is_safe(clearance=40.),
                             b.line_to_goal_is_safe(clearance=40.))
            action = np.array([0., .01 * (index % 2)], dtype=np.float32)
            ra, rb = a.step(action), b.step(action)
            np.testing.assert_array_equal(a.state, b.state)
            self.assert_observation(ra[0], rb[0]); self.assertEqual(ra[1:], rb[1:])
            if ra[2] or ra[3]: break
        np.testing.assert_array_equal(ob.zone_features, old_zones)
        pa, pb = a.set_goal([40., 2., 11.]), b.set_goal([40., 2., 11.])
        self.assert_observation(pa[0], pb[0]); self.assertEqual(pa[1], pb[1])
        pa, pb = a.reset(), b.reset()
        self.assert_observation(pa[0], pb[0]); self.assertEqual(pa[1], pb[1])

    def test_info_rejects_changes_between_producer_and_consumer(self):
        for change in ('center', 'radii', 'shape', 'zone', 'goal', 'radius_margin'):
            with self.subTest(change=change):
                env = self.env(True)
                original_info = env._info
                checked = []
                def info(**kwargs):
                    shape = env.zones[0].shape
                    shape.set_segment_clearance_cache_enabled(False)
                    if change == 'center': shape._center[0] += .1
                    if change == 'radii': shape._radii[0] += .1
                    if change == 'shape':
                        env.zones[0]._shape = Ellipsoid([10., 0., 10.], 2., 1., 1.)
                    if change == 'zone':
                        env.zones = [NoFlyZone('replacement', Ellipsoid([10., 0., 10.], 2., 1., 1.), .2)]
                    if change == 'goal': env.goal[0] += 1.
                    if change == 'radius_margin':
                        env.uav_collision_radius += .1
                        env.zones[0]._safety_margin += .1
                    shape = env.zones[0].shape
                    expected = original_info(**{k: v for k, v in kwargs.items()
                                                if k != '_ellipsoid_intersections'})
                    with mock.patch.object(shape, 'segment_intersection',
                                           wraps=shape.segment_intersection) as query:
                        actual = original_info(**kwargs)
                        self.assertEqual(actual, expected)
                        self.assertEqual(query.call_count, 0 if change == 'radius_margin' else 1)
                    checked.append(True)
                    return actual
                with mock.patch.object(env, '_info', side_effect=info):
                    env.reset()
                self.assertEqual(checked, [True])
