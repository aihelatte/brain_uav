"""Tests for the standalone structured V2 observation contract and builder."""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import numpy as np

from brain_uav.geometry import (
    AABB,
    Box,
    Ellipsoid,
    GeometryShape,
    NoFlyZone,
    QuadrangularPyramid,
    Sphere,
    TriangularPyramid,
)
from brain_uav.observations import (
    EGO_FEATURE_DIM,
    EGO_FEATURE_INDEX,
    EGO_FEATURE_NAMES,
    GOAL_FEATURE_DIM,
    GOAL_FEATURE_INDEX,
    GOAL_FEATURE_NAMES,
    ZONE_FEATURE_DIM,
    ZONE_FEATURE_INDEX,
    ZONE_FEATURE_NAMES,
    V2Observation,
    V2ObservationScales,
    build_v2_observation,
)


class _UnknownShape(GeometryShape):
    """Minimal geometry implementation outside the builder's supported type set."""

    def signed_distance(self, point):
        del point
        return 1.0

    def closest_point(self, point):
        del point
        return np.zeros(3, dtype=np.float64)

    def surface_normal(self, point):
        del point
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def segment_intersection(self, start, end):
        del start, end
        return None

    def segment_clearance(self, start, end):
        del start, end
        return 1.0

    def bounding_box(self):
        return AABB(np.zeros(3), np.ones(3))


class TestV2ObservationContract(unittest.TestCase):
    def test_feature_names_dimensions_and_indexes_are_fixed_and_read_only(self):
        self.assertEqual(
            EGO_FEATURE_NAMES,
            (
                'uav_x_norm',
                'uav_y_norm',
                'uav_z_fraction',
                'gamma_fraction',
                'sin_psi',
                'cos_psi',
            ),
        )
        self.assertEqual(
            GOAL_FEATURE_NAMES,
            (
                'goal_forward_norm',
                'goal_right_norm',
                'goal_up_norm',
                'goal_distance_norm',
            ),
        )
        self.assertEqual(
            ZONE_FEATURE_NAMES,
            (
                'shape_sphere',
                'shape_ellipsoid',
                'shape_box',
                'shape_triangular_pyramid',
                'shape_quadrangular_pyramid',
                'zone_forward_norm',
                'zone_right_norm',
                'zone_up_norm',
                'extent_x_norm',
                'extent_y_norm',
                'extent_z_norm',
                'safety_margin_norm',
                'point_clearance_norm',
                'surface_normal_forward',
                'surface_normal_right',
                'surface_normal_up',
                'approach_cosine',
                'raw_goal_path_intersects',
                'raw_first_intersection_fraction',
            ),
        )
        self.assertEqual((EGO_FEATURE_DIM, GOAL_FEATURE_DIM, ZONE_FEATURE_DIM), (6, 4, 19))
        for names, indexes in (
            (EGO_FEATURE_NAMES, EGO_FEATURE_INDEX),
            (GOAL_FEATURE_NAMES, GOAL_FEATURE_INDEX),
            (ZONE_FEATURE_NAMES, ZONE_FEATURE_INDEX),
        ):
            self.assertEqual(dict(indexes), {name: index for index, name in enumerate(names)})
            with self.assertRaises(TypeError):
                indexes[names[0]] = 99

    def test_scales_validation_derived_values_and_immutability(self):
        scales = V2ObservationScales(100.0, 10.0, 60.0, math.pi / 4.0)
        self.assertEqual(scales.horizontal_span, 200.0)
        self.assertEqual(scales.vertical_span, 50.0)
        self.assertAlmostEqual(scales.world_diagonal, math.sqrt(200.0**2 * 2.0 + 50.0**2))
        with self.assertRaises(FrozenInstanceError):
            scales.world_xy = 200.0

        invalid = (
            (0.0, 0.0, 10.0, 1.0),
            (-1.0, 0.0, 10.0, 1.0),
            (1.0, 10.0, 10.0, 1.0),
            (1.0, 11.0, 10.0, 1.0),
            (1.0, 0.0, 10.0, 0.0),
            (np.nan, 0.0, 10.0, 1.0),
            (1.0, -np.inf, 10.0, 1.0),
            (1.0, 0.0, np.inf, 1.0),
            (1.0, 0.0, 10.0, np.nan),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                V2ObservationScales(*values)

    def test_observation_copies_casts_and_protects_arrays(self):
        ego = np.arange(EGO_FEATURE_DIM, dtype=np.float64)
        goal = np.arange(GOAL_FEATURE_DIM, dtype=np.int64)
        zones = np.arange(2 * ZONE_FEATURE_DIM, dtype=np.float64).reshape(2, ZONE_FEATURE_DIM)
        mask = np.ones(2, dtype=np.int8)
        observation = V2Observation(ego, goal, zones, mask)

        self.assertEqual(observation.ego_features.dtype, np.float32)
        self.assertEqual(observation.goal_features.dtype, np.float32)
        self.assertEqual(observation.zone_features.dtype, np.float32)
        self.assertEqual(observation.presence_mask.dtype, np.bool_)
        ego[:] = -1.0
        goal[:] = -1
        zones[:] = -1.0
        mask[:] = 0
        self.assertEqual(observation.ego_features[0], 0.0)
        self.assertEqual(observation.goal_features[1], 1.0)
        self.assertEqual(observation.zone_features[1, 0], float(ZONE_FEATURE_DIM))
        self.assertTrue(observation.presence_mask.all())
        for array in (
            observation.ego_features,
            observation.goal_features,
            observation.zone_features,
            observation.presence_mask,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 0
        with self.assertRaises(FrozenInstanceError):
            observation.ego_features = np.zeros(EGO_FEATURE_DIM, dtype=np.float32)

    def test_observation_rejects_bad_shapes_nonfinite_values_and_padding_mask(self):
        ego = np.zeros(EGO_FEATURE_DIM)
        goal = np.zeros(GOAL_FEATURE_DIM)
        zones = np.zeros((2, ZONE_FEATURE_DIM))
        mask = np.ones(2, dtype=bool)
        invalid_cases = (
            (np.zeros(EGO_FEATURE_DIM + 1), goal, zones, mask),
            (ego, np.zeros(GOAL_FEATURE_DIM + 1), zones, mask),
            (ego, goal, np.zeros((2, ZONE_FEATURE_DIM + 1)), mask),
            (ego, goal, zones, np.ones(1, dtype=bool)),
            (np.full(EGO_FEATURE_DIM, np.nan), goal, zones, mask),
            (ego, np.full(GOAL_FEATURE_DIM, np.inf), zones, mask),
            (ego, goal, np.full((2, ZONE_FEATURE_DIM), np.nan), mask),
            (ego, goal, zones, np.array([True, False])),
        )
        for values in invalid_cases:
            with self.subTest(shapes=tuple(np.asarray(value).shape for value in values)):
                with self.assertRaises(ValueError):
                    V2Observation(*values)


class TestV2ObservationBuilder(unittest.TestCase):
    def setUp(self):
        self.scales = V2ObservationScales(100.0, 0.0, 50.0, math.pi / 4.0)
        self.state = np.array([0.0, 0.0, 10.0, 0.0, 0.0], dtype=np.float64)
        self.goal = np.array([20.0, 0.0, 10.0], dtype=np.float64)

    def build(self, zones, *, state=None, goal=None, uav_radius=0.0):
        return build_v2_observation(
            self.state if state is None else state,
            self.goal if goal is None else goal,
            zones,
            self.scales,
            uav_radius=uav_radius,
        )

    @staticmethod
    def sphere_zone(zone_id, center, *, radius=1.0, margin=0.0, metadata=None):
        return NoFlyZone(zone_id, Sphere(center, radius), margin, metadata)

    def test_builder_rejects_invalid_state_goal_radius_scales_and_zones(self):
        zone = self.sphere_zone('valid', [10.0, 0.0, 10.0])
        for state in (
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, np.nan, 0.0, 0.0],
            [0.0, 0.0, 10.0, np.inf, 0.0],
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                build_v2_observation(state, self.goal, [zone], self.scales)
        for goal in (
            [0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, np.nan, 1.0],
            [0.0, 0.0, np.inf],
        ):
            with self.subTest(goal=goal), self.assertRaises(ValueError):
                build_v2_observation(self.state, goal, [zone], self.scales)
        for radius in (-1.0, np.nan, np.inf):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.build([zone], uav_radius=radius)
        with self.assertRaises(TypeError):
            build_v2_observation(self.state, self.goal, [zone], object())
        for zones in (None, 'zone', (item for item in [zone]), [object()]):
            with self.subTest(zones_type=type(zones).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    self.build(zones)

        corrupted_zone = self.sphere_zone('initially-valid', [10.0, 0.0, 10.0])
        corrupted_zone._zone_id = ''
        with self.assertRaisesRegex(ValueError, 'zone_id'):
            self.build([corrupted_zone])

    def test_ego_features_are_normalized_without_clipping(self):
        state = [125.0, -150.0, 55.0, math.pi / 2.0, math.pi / 6.0]
        observation = self.build([], state=state)
        expected = {
            'uav_x_norm': 1.25,
            'uav_y_norm': -1.5,
            'uav_z_fraction': 1.1,
            'gamma_fraction': 2.0,
            'sin_psi': 0.5,
            'cos_psi': math.sqrt(3.0) / 2.0,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(
                float(observation.ego_features[EGO_FEATURE_INDEX[name]]), value, places=6
            )

    def test_goal_features_transform_and_normalize_in_heading_frame(self):
        state = [10.0, -10.0, 5.0, 0.0, math.pi / 2.0]
        goal = [0.0, 30.0, 15.0]
        observation = self.build([], state=state, goal=goal)
        expected = {
            'goal_forward_norm': 40.0 / self.scales.horizontal_span,
            'goal_right_norm': 10.0 / self.scales.horizontal_span,
            'goal_up_norm': 10.0 / self.scales.vertical_span,
            'goal_distance_norm': math.sqrt(10.0**2 + 40.0**2 + 10.0**2)
            / self.scales.world_diagonal,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(
                float(observation.goal_features[GOAL_FEATURE_INDEX[name]]), value, places=7
            )

    def test_five_shape_one_hot_encodings_are_mutually_exclusive(self):
        zones = [
            NoFlyZone('sphere', Sphere([10.0, 0.0, 10.0], 1.0)),
            NoFlyZone('ellipsoid', Ellipsoid([10.0, 0.0, 10.0], 2.0, 3.0, 4.0)),
            NoFlyZone('box', Box([10.0, 0.0, 10.0], 2.0, 4.0, 6.0)),
            NoFlyZone('triangle', TriangularPyramid([10.0, 0.0, 0.0], 6.0, 3.0, 8.0)),
            NoFlyZone(
                'quadrangle', QuadrangularPyramid([10.0, 0.0, 0.0], 6.0, 4.0, 8.0)
            ),
        ]
        observation = self.build(zones)
        indices = [ZONE_FEATURE_INDEX[name] for name in ZONE_FEATURE_NAMES[:5]]
        one_hot = observation.zone_features[:, indices]
        np.testing.assert_array_equal(one_hot, np.eye(5, dtype=np.float32))
        np.testing.assert_array_equal(one_hot.sum(axis=1), np.ones(5, dtype=np.float32))

    def test_zone_reference_and_extents_are_derived_from_aabb(self):
        zone = NoFlyZone(
            'triangle', TriangularPyramid([10.0, 3.0, 0.0], 6.0, 3.0, 4.0)
        )
        observation = self.build([zone])
        expected = {
            'zone_forward_norm': 10.0 / self.scales.horizontal_span,
            'zone_right_norm': 3.5 / self.scales.horizontal_span,
            'zone_up_norm': -8.0 / self.scales.vertical_span,
            'extent_x_norm': 6.0 / self.scales.horizontal_span,
            'extent_y_norm': 3.0 / self.scales.horizontal_span,
            'extent_z_norm': 4.0 / self.scales.vertical_span,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(
                float(observation.zone_features[0, ZONE_FEATURE_INDEX[name]]), value, places=7
            )

    def test_heading_axes_at_zero_and_half_pi(self):
        zero = self.build([self.sphere_zone('zero', [10.0, 20.0, 10.0])])
        self.assertAlmostEqual(
            float(zero.zone_features[0, ZONE_FEATURE_INDEX['zone_forward_norm']]), 10.0 / 200.0
        )
        self.assertAlmostEqual(
            float(zero.zone_features[0, ZONE_FEATURE_INDEX['zone_right_norm']]), 20.0 / 200.0
        )

        half_pi = self.build(
            [self.sphere_zone('half-pi', [-20.0, 10.0, 10.0])],
            state=[0.0, 0.0, 10.0, 0.0, math.pi / 2.0],
        )
        self.assertAlmostEqual(
            float(half_pi.zone_features[0, ZONE_FEATURE_INDEX['zone_forward_norm']]),
            10.0 / 200.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(half_pi.zone_features[0, ZONE_FEATURE_INDEX['zone_right_norm']]),
            20.0 / 200.0,
            places=7,
        )

    def test_clearance_margin_radius_normal_and_approach_cosine(self):
        clearance_zone = self.sphere_zone(
            'clearance', [10.0, 0.0, 10.0], radius=2.0, margin=1.5
        )
        clearance = self.build([clearance_zone], uav_radius=0.5)
        self.assertAlmostEqual(
            float(clearance.zone_features[0, ZONE_FEATURE_INDEX['point_clearance_norm']]),
            (8.0 - 1.5 - 0.5) / self.scales.world_diagonal,
            places=7,
        )
        self.assertAlmostEqual(
            float(clearance.zone_features[0, ZONE_FEATURE_INDEX['safety_margin_norm']]),
            1.5 / self.scales.world_diagonal,
            places=7,
        )

        rotated = self.build(
            [self.sphere_zone('normal', [10.0, 0.0, 10.0])],
            state=[0.0, 0.0, 10.0, 0.0, math.pi / 2.0],
        )
        local_normal = np.array(
            [
                rotated.zone_features[0, ZONE_FEATURE_INDEX['surface_normal_forward']],
                rotated.zone_features[0, ZONE_FEATURE_INDEX['surface_normal_right']],
                rotated.zone_features[0, ZONE_FEATURE_INDEX['surface_normal_up']],
            ]
        )
        np.testing.assert_allclose(local_normal, [0.0, 1.0, 0.0], atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(local_normal)), 1.0, places=7)

        directions = self.build(
            [
                self.sphere_zone('ahead', [10.0, 0.0, 10.0]),
                self.sphere_zone('behind', [-10.0, 0.0, 10.0]),
                self.sphere_zone('side', [0.0, 10.0, 10.0]),
            ]
        )
        np.testing.assert_allclose(
            directions.zone_features[:, ZONE_FEATURE_INDEX['approach_cosine']],
            [1.0, -1.0, 0.0],
            atol=1e-7,
        )

    def test_precomputed_zone_clearances_are_strict_and_avoid_duplicate_queries(self):
        zones = [
            self.sphere_zone('first', [10.0, 0.0, 10.0], margin=0.5),
            self.sphere_zone('second', [0.0, 10.0, 10.0], radius=2.0),
        ]
        clearances = (8.25, 7.5)
        with mock.patch.object(
            NoFlyZone,
            'point_clearance',
            side_effect=AssertionError('precomputed clearances must be reused'),
        ):
            observation = build_v2_observation(
                self.state,
                self.goal,
                zones,
                self.scales,
                uav_radius=0.25,
                zone_point_clearances=clearances,
            )
        np.testing.assert_array_equal(
            observation.zone_features[:, ZONE_FEATURE_INDEX['point_clearance_norm']],
            (
                np.asarray(clearances, dtype=np.float64)
                / self.scales.world_diagonal
            ).astype(np.float32),
        )

        for invalid in ((1.0,), (1.0, float('nan')), (1.0, float('inf')), 'bad'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                (TypeError, ValueError), 'zone_point_clearances'
            ):
                build_v2_observation(
                    self.state,
                    self.goal,
                    zones,
                    self.scales,
                    zone_point_clearances=invalid,
                )

    def test_raw_goal_path_hit_and_fraction_ignore_effective_safety_boundary(self):
        crossing = self.sphere_zone(
            'crossing', [10.0, 0.0, 10.0], radius=2.0, margin=5.0
        )
        miss = self.sphere_zone('miss', [0.0, 10.0, 10.0], radius=2.0, margin=20.0)
        with mock.patch.object(
            Sphere,
            'segment_clearance',
            side_effect=AssertionError('builder must not call segment_clearance'),
        ):
            observation = self.build([crossing, miss], uav_radius=3.0)
        np.testing.assert_array_equal(
            observation.zone_features[:, ZONE_FEATURE_INDEX['raw_goal_path_intersects']],
            [1.0, 0.0],
        )
        np.testing.assert_allclose(
            observation.zone_features[:, ZONE_FEATURE_INDEX['raw_first_intersection_fraction']],
            [0.4, 1.0],
            atol=1e-7,
        )

    def test_dynamic_zone_counts_have_no_padding_truncation_or_flattening(self):
        for count in (0, 1, 5, 6, 10):
            zones = [
                self.sphere_zone(f'zone-{index}', [10.0 * (index + 1), 20.0, 10.0])
                for index in range(count)
            ]
            with self.subTest(count=count):
                observation = self.build(zones)
                self.assertEqual(observation.zone_features.shape, (count, ZONE_FEATURE_DIM))
                self.assertEqual(observation.presence_mask.shape, (count,))
                self.assertTrue(observation.presence_mask.all())
                if count >= 6:
                    self.assertAlmostEqual(
                        float(
                            observation.zone_features[
                                5, ZONE_FEATURE_INDEX['zone_forward_norm']
                            ]
                        ),
                        60.0 / self.scales.horizontal_span,
                        places=7,
                    )
                if count == 10:
                    self.assertAlmostEqual(
                        float(
                            observation.zone_features[
                                9, ZONE_FEATURE_INDEX['zone_forward_norm']
                            ]
                        ),
                        100.0 / self.scales.horizontal_span,
                        places=7,
                    )

    def test_order_is_preserved_and_zone_rows_are_permutation_equivariant(self):
        zones = [
            self.sphere_zone('sphere', [10.0, 0.0, 10.0]),
            NoFlyZone('box', Box([0.0, 20.0, 10.0], 2.0, 4.0, 6.0)),
            NoFlyZone('ellipsoid', Ellipsoid([-30.0, 0.0, 10.0], 2.0, 3.0, 4.0)),
        ]
        permutation = [2, 0, 1]
        original = self.build(zones)
        reordered = self.build([zones[index] for index in permutation])
        np.testing.assert_array_equal(reordered.ego_features, original.ego_features)
        np.testing.assert_array_equal(reordered.goal_features, original.goal_features)
        np.testing.assert_array_equal(reordered.zone_features, original.zone_features[permutation])
        np.testing.assert_array_equal(
            reordered.presence_mask, original.presence_mask[permutation]
        )

    def test_overlap_behind_duplicate_ids_and_nonpolicy_metadata(self):
        overlapping = [
            self.sphere_zone('first', [10.0, 0.0, 10.0], radius=3.0),
            NoFlyZone('second', Box([10.0, 0.0, 10.0], 4.0, 4.0, 4.0)),
        ]
        overlap_observation = self.build(overlapping)
        self.assertEqual(overlap_observation.zone_features.shape, (2, ZONE_FEATURE_DIM))
        np.testing.assert_array_equal(overlap_observation.presence_mask, [True, True])

        behind_observation = self.build(
            [self.sphere_zone('behind-zone', [-10.0, 0.0, 10.0])]
        )
        self.assertLess(
            float(
                behind_observation.zone_features[
                    0, ZONE_FEATURE_INDEX['zone_forward_norm']
                ]
            ),
            0.0,
        )
        self.assertTrue(behind_observation.presence_mask[0])

        duplicates = [
            self.sphere_zone('duplicate', [10.0, 0.0, 10.0]),
            self.sphere_zone('duplicate', [20.0, 0.0, 10.0]),
        ]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.build(duplicates)

        shape = Sphere([10.0, 0.0, 10.0], 2.0)
        first = self.build([NoFlyZone('alpha', shape, 0.5, {'label': 'first'})])
        second = self.build(
            [NoFlyZone('beta', shape, 0.5, {'nested': {'value': 99}})]
        )
        np.testing.assert_array_equal(first.zone_features, second.zone_features)

    def test_builder_does_not_modify_inputs_and_output_is_independent(self):
        state = self.state.copy()
        goal = self.goal.copy()
        zones = [
            self.sphere_zone('stable', [10.0, 0.0, 10.0], metadata={'value': 1})
        ]
        state_before = state.copy()
        goal_before = goal.copy()
        zone_before = zones[0].to_dict()
        zone_identity = id(zones[0])
        observation = self.build(zones, state=state, goal=goal)
        baseline = (
            observation.ego_features.copy(),
            observation.goal_features.copy(),
            observation.zone_features.copy(),
        )

        np.testing.assert_array_equal(state, state_before)
        np.testing.assert_array_equal(goal, goal_before)
        self.assertEqual(id(zones[0]), zone_identity)
        self.assertEqual(zones[0].to_dict(), zone_before)
        state[:] = 999.0
        goal[:] = -999.0
        zones.clear()
        np.testing.assert_array_equal(observation.ego_features, baseline[0])
        np.testing.assert_array_equal(observation.goal_features, baseline[1])
        np.testing.assert_array_equal(observation.zone_features, baseline[2])
        self.assertTrue(np.isfinite(observation.ego_features).all())
        self.assertTrue(np.isfinite(observation.goal_features).all())
        self.assertTrue(np.isfinite(observation.zone_features).all())
        self.assertEqual(observation.ego_features.dtype, np.float32)
        self.assertEqual(observation.goal_features.dtype, np.float32)
        self.assertEqual(observation.zone_features.dtype, np.float32)
        self.assertEqual(observation.presence_mask.dtype, np.bool_)

    def test_unknown_geometry_shape_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported GeometryShape.*_UnknownShape'):
            self.build([NoFlyZone('unknown', _UnknownShape())])


if __name__ == '__main__':
    unittest.main()
