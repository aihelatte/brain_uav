"""Tests for reproducible V2 random training-scenario generation."""

from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from brain_uav.config import ScenarioConfig
from brain_uav.envs.v2_scenario_generator import (
    DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES,
    DEFAULT_V2_SHAPE_PROBABILITIES,
    DEFAULT_V2_ZONE_COUNT_PROBABILITIES,
    V2_SCENARIO_GENERATOR_VERSION,
    V2ScenarioGenerator,
    V2ScenarioGeneratorConfig,
    V2ScenarioGenerationError,
)
from brain_uav.geometry import (
    Box,
    Ellipsoid,
    NoFlyZone,
    QuadrangularPyramid,
    Sphere,
    TriangularPyramid,
    no_fly_zone_from_dict,
)


SHAPE_TYPES = {
    'sphere': Sphere,
    'ellipsoid': Ellipsoid,
    'box': Box,
    'triangular_pyramid': TriangularPyramid,
    'quadrangular_pyramid': QuadrangularPyramid,
}


def _forced_counts(level: str, count: int):
    result = {
        key: dict(probabilities)
        for key, probabilities in DEFAULT_V2_ZONE_COUNT_PROBABILITIES.items()
    }
    result[level] = {count: 1.0}
    return result


def _forced_blocker(level: str, value: float):
    result = dict(DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES)
    result[level] = value
    return result


def _forced_shape(shape_type: str):
    return {name: float(name == shape_type) for name in SHAPE_TYPES}


def _generator(
    level: str,
    count: int,
    *,
    shape_type: str = 'sphere',
    seed: int = 7,
    ground_probability: float = 0.5,
    overlap_probability: float = 0.10,
    blocker_probability: float | None = None,
):
    if blocker_probability is None:
        blocker = dict(DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES)
        if count == 0 and level != 'hard':
            # Easy/medium targets are overall rates; hard keeps its 1.0
            # always-block sentinel even when this helper forces no zones.
            blocker[level] = 0.0
    else:
        blocker = _forced_blocker(level, blocker_probability)
    config = V2ScenarioGeneratorConfig(
        zone_count_probabilities=_forced_counts(level, count),
        shape_probabilities=_forced_shape(shape_type),
        ground_contact_probability=ground_probability,
        medium_overlap_probability=overlap_probability,
        direct_path_blocker_probability=blocker,
    )
    return V2ScenarioGenerator(
        ScenarioConfig(),
        curriculum_level=level,
        seed=seed,
        config=config,
    )


def _zones(payload):
    return [no_fly_zone_from_dict(item) for item in payload['zones']]


class TestV2ScenarioGeneratorConfiguration(unittest.TestCase):
    def test_default_count_and_shape_probabilities_are_exact(self):
        self.assertEqual(
            DEFAULT_V2_ZONE_COUNT_PROBABILITIES,
            {
                'easy': {0: 0.05, 1: 0.475, 2: 0.475},
                'medium': {0: 0.03, 2: 0.485, 3: 0.485},
                'hard': {0: 0.02, 2: 0.28, 3: 0.30, 4: 0.30, 5: 0.06, 6: 0.04},
            },
        )
        self.assertEqual(DEFAULT_V2_SHAPE_PROBABILITIES, {name: 0.2 for name in SHAPE_TYPES})
        self.assertEqual(
            DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES,
            {'easy': 0.50, 'medium': 0.80, 'hard': 1.00},
        )
        self.assertEqual(V2_SCENARIO_GENERATOR_VERSION, 2)
        config = V2ScenarioGeneratorConfig()
        self.assertAlmostEqual(
            config.direct_path_blocker_probability_nonzero['easy'],
            0.50 / 0.95,
        )
        self.assertAlmostEqual(
            config.direct_path_blocker_probability_nonzero['easy'] * 0.95,
            0.50,
        )
        self.assertAlmostEqual(
            config.direct_path_blocker_probability_nonzero['medium'],
            0.80 / 0.97,
        )
        self.assertAlmostEqual(
            config.direct_path_blocker_probability_nonzero['medium'] * 0.97,
            0.80,
        )
        self.assertEqual(config.direct_path_blocker_probability_nonzero['hard'], 1.0)

    def test_probability_and_scalar_validation_rejects_invalid_values(self):
        invalid_count_maps = (
            {'easy': {-1: 1.0}, 'medium': {0: 1.0}, 'hard': {0: 1.0}},
            {'easy': {1.5: 1.0}, 'medium': {0: 1.0}, 'hard': {0: 1.0}},
            {'easy': {0: -0.1, 1: 1.1}, 'medium': {0: 1.0}, 'hard': {0: 1.0}},
            {'easy': {0: float('nan')}, 'medium': {0: 1.0}, 'hard': {0: 1.0}},
            {'easy': {0: float('inf')}, 'medium': {0: 1.0}, 'hard': {0: 1.0}},
            {'easy': {0: 0.5}, 'medium': {0: 1.0}, 'hard': {0: 1.0}},
            {'easy': {0: 1.0}, 'medium': {0: 1.0}},
        )
        for probabilities in invalid_count_maps:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                V2ScenarioGeneratorConfig(zone_count_probabilities=probabilities)

        invalid_shapes = (
            {'sphere': 1.0},
            {**DEFAULT_V2_SHAPE_PROBABILITIES, 'unknown': 0.0},
            {**DEFAULT_V2_SHAPE_PROBABILITIES, 'sphere': -0.1, 'box': 0.3},
            {**DEFAULT_V2_SHAPE_PROBABILITIES, 'sphere': float('nan')},
        )
        for probabilities in invalid_shapes:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                V2ScenarioGeneratorConfig(shape_probabilities=probabilities)

        for values in (
            {'ground_contact_probability': -0.1},
            {'ground_contact_probability': 1.1},
            {'medium_overlap_probability': float('inf')},
            {'start_surface_clearance': 0.0},
            {'goal_surface_clearance': -1.0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                V2ScenarioGeneratorConfig(**values)

        invalid_blockers = (
            {'easy': 0.5, 'medium': 0.8},
            {**DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES, 'easy': 1.1},
            {**DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES, 'medium': -0.1},
            {**DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES, 'easy': float('nan')},
            {**DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES, 'hard': 0.9},
        )
        for probabilities in invalid_blockers:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                V2ScenarioGeneratorConfig(
                    direct_path_blocker_probability=probabilities
                )

        default_counts = {
            level: dict(values)
            for level, values in DEFAULT_V2_ZONE_COUNT_PROBABILITIES.items()
        }
        unreachable_combinations = (
            (
                {**default_counts, 'easy': {0: 0.6, 1: 0.4}},
                dict(DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES),
            ),
            (
                {'easy': {0: 1.0}, 'medium': {0: 1.0}, 'hard': {0: 1.0}},
                dict(DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES),
            ),
        )
        for counts, blockers in unreachable_combinations:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                V2ScenarioGeneratorConfig(
                    zone_count_probabilities=counts,
                    direct_path_blocker_probability=blockers,
                )

        zero_forced = V2ScenarioGeneratorConfig(
            zone_count_probabilities={
                'easy': {0: 1.0},
                'medium': {0: 1.0},
                'hard': {0: 1.0},
            },
            direct_path_blocker_probability={
                'easy': 0.0,
                'medium': 0.0,
                'hard': 1.0,
            },
        )
        self.assertEqual(
            zero_forced.direct_path_blocker_probability_nonzero,
            {'easy': 0.0, 'medium': 0.0, 'hard': 1.0},
        )

    def test_blocker_targets_use_the_full_reachable_overall_interval(self):
        counts = {
            **DEFAULT_V2_ZONE_COUNT_PROBABILITIES,
            'easy': {0: 0.6, 1: 0.4},
        }
        blockers = dict(DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES)
        blockers['easy'] = 0.4
        at_maximum = V2ScenarioGeneratorConfig(
            zone_count_probabilities=counts,
            direct_path_blocker_probability=blockers,
        )
        self.assertEqual(
            at_maximum.direct_path_blocker_probability_nonzero['easy'], 1.0
        )
        self.assertAlmostEqual(
            at_maximum.direct_path_blocker_probability_nonzero['easy'] * 0.4,
            0.4,
        )

        blockers['easy'] = 0.5
        with self.assertRaisesRegex(
            ValueError, r'achievable overall interval is \[0\.0, 0\.4\]'
        ):
            V2ScenarioGeneratorConfig(
                zone_count_probabilities=counts,
                direct_path_blocker_probability=blockers,
            )

        zero_targets = V2ScenarioGeneratorConfig(
            direct_path_blocker_probability={
                'easy': 0.0,
                'medium': 0.0,
                'hard': 1.0,
            },
        )
        self.assertEqual(
            zero_targets.direct_path_blocker_probability_nonzero,
            {'easy': 0.0, 'medium': 0.0, 'hard': 1.0},
        )


class TestV2ScenarioGenerator(unittest.TestCase):
    def test_all_zero_zone_distributions_generate_unblocked_scenarios(self):
        counts = {level: {0: 1.0} for level in ('easy', 'medium', 'hard')}
        config = V2ScenarioGeneratorConfig(
            zone_count_probabilities=counts,
            direct_path_blocker_probability={
                'easy': 0.0,
                'medium': 0.0,
                'hard': 1.0,
            },
        )
        for level in ('easy', 'medium', 'hard'):
            with self.subTest(level=level):
                payload = V2ScenarioGenerator(
                    ScenarioConfig(), level, seed=700, config=config
                ).generate()
                self.assertEqual(payload['zones'], [])
                self.assertEqual(
                    payload['metadata']['direct_path_blocker_count'], 0
                )
                self.assertIs(
                    payload['metadata']['requested_direct_path_blocker'], False
                )

    def test_hard_nonzero_blocker_branch_does_not_consume_rng(self):
        class RandomForbidden:
            def random(self):
                raise AssertionError('hard blocker selection must not draw RNG')

        generator = _generator('hard', 2)
        rng = RandomForbidden()
        self.assertIs(
            generator._sample_direct_path_blocker_branch(rng, zone_count=2),
            True,
        )
        self.assertIs(
            generator._sample_direct_path_blocker_branch(rng, zone_count=0),
            False,
        )

    def test_forced_counts_generate_zero_through_six_without_truncation(self):
        cases = ((0, 'easy'), (1, 'easy'), (2, 'easy'), (3, 'hard'), (4, 'hard'), (5, 'hard'), (6, 'hard'))
        for count, level in cases:
            with self.subTest(count=count, level=level):
                payload = _generator(level, count, seed=100 + count).generate()
                self.assertEqual(len(payload['zones']), count)
                self.assertEqual(payload['metadata']['requested_zone_count'], count)
                self.assertEqual(payload['metadata']['effective_zone_count'], count)
                self.assertEqual(payload['curriculum_level'], level)

    def test_all_five_shapes_have_valid_world_bounds_and_true_parameters(self):
        scenario = ScenarioConfig()
        for index, (shape_name, shape_class) in enumerate(SHAPE_TYPES.items()):
            with self.subTest(shape=shape_name):
                payload = _generator('easy', 1, shape_type=shape_name, seed=300 + index).generate()
                zone = _zones(payload)[0]
                self.assertIs(type(zone.shape), shape_class)
                bounds = zone.shape.bounding_box()
                self.assertGreaterEqual(bounds.min_corner[0], -scenario.world_xy)
                self.assertGreaterEqual(bounds.min_corner[1], -scenario.world_xy)
                self.assertGreaterEqual(bounds.min_corner[2], 0.0)
                self.assertLessEqual(bounds.max_corner[0], scenario.world_xy)
                self.assertLessEqual(bounds.max_corner[1], scenario.world_xy)
                self.assertLessEqual(bounds.max_corner[2], scenario.world_z_max)
                self.assertNotIn('orientation', zone.shape.to_dict())
                if isinstance(zone.shape, Box):
                    self.assertEqual(zone.shape.size_x, zone.shape.size_y)
                    self.assertEqual(zone.shape.size_y, zone.shape.size_z)
                if isinstance(zone.shape, (TriangularPyramid, QuadrangularPyramid)):
                    self.assertEqual(zone.shape.base_center[2], 0.0)

                reference = zone.metadata['requested_reference_scale']
                if isinstance(zone.shape, Sphere):
                    self.assertEqual(zone.shape.radius, reference)
                elif isinstance(zone.shape, Ellipsoid):
                    horizontal = sorted((zone.shape.radius_x, zone.shape.radius_y))
                    self.assertAlmostEqual(horizontal[1], reference)
                    self.assertGreaterEqual(horizontal[0] / reference, 0.70)
                    self.assertLessEqual(horizontal[0] / reference, 1.00)
                    self.assertGreaterEqual(zone.shape.radius_z / reference, 0.50)
                    self.assertLessEqual(zone.shape.radius_z / reference, 0.80)
                elif isinstance(zone.shape, Box):
                    self.assertEqual(zone.shape.size_x, 2.0 * reference)
                else:
                    self.assertGreaterEqual(zone.shape.base_size_x / reference, 1.50)
                    self.assertLessEqual(zone.shape.base_size_x / reference, 2.00)
                    self.assertGreaterEqual(zone.shape.base_size_y / reference, 1.50)
                    self.assertLessEqual(zone.shape.base_size_y / reference, 2.00)
                    self.assertGreaterEqual(zone.shape.height / reference, 0.75)
                    self.assertLessEqual(zone.shape.height / reference, 1.25)

    def test_primitives_can_be_forced_grounded_or_suspended(self):
        for shape_name in ('sphere', 'ellipsoid', 'box'):
            for probability, expected_grounded in ((1.0, True), (0.0, False)):
                with self.subTest(shape=shape_name, grounded=expected_grounded):
                    payload = _generator(
                        'easy',
                        1,
                        shape_type=shape_name,
                        seed=400,
                        ground_probability=probability,
                    ).generate()
                    bounds = _zones(payload)[0].shape.bounding_box()
                    if expected_grounded:
                        self.assertAlmostEqual(float(bounds.min_corner[2]), 0.0, places=9)
                    else:
                        self.assertGreater(float(bounds.min_corner[2]), 0.0)

    def test_real_shape_clearance_is_strict_for_start_and_goal(self):
        for index, shape_name in enumerate(SHAPE_TYPES):
            with self.subTest(shape=shape_name):
                payload = _generator('easy', 1, shape_type=shape_name, seed=500 + index).generate()
                state = np.asarray(payload['state'], dtype=np.float64)
                goal = np.asarray(payload['goal'], dtype=np.float64)
                zone = _zones(payload)[0]
                self.assertGreater(zone.point_clearance(state[:3]), 160.0)
                self.assertGreater(zone.point_clearance(goal), 106.0)

    def test_curriculum_blocking_rules_and_zero_zone_exceptions(self):
        easy_zero = _generator('easy', 0, seed=601).generate()
        medium_zero = _generator('medium', 0, seed=604).generate()
        hard_zero = _generator('hard', 0, seed=605).generate()
        easy_clear = _generator('easy', 2, seed=607, blocker_probability=0.0).generate()
        easy_blocked = _generator('easy', 2, seed=606, blocker_probability=1.0).generate()
        medium_blocked = _generator('medium', 2, seed=608, blocker_probability=1.0).generate()
        hard = _generator('hard', 2, seed=603).generate()

        for payload in (easy_zero, medium_zero, hard_zero, easy_clear):
            metadata = payload['metadata']
            self.assertEqual(metadata['direct_path_blocker_count'], 0)
            self.assertIs(metadata['requested_direct_path_blocker'], False)
            self.assertEqual(metadata['feasibility_check'], 'direct_safe_corridor')
        for payload in (easy_blocked, medium_blocked, hard):
            metadata = payload['metadata']
            self.assertGreaterEqual(metadata['direct_path_blocker_count'], 1)
            self.assertIs(metadata['requested_direct_path_blocker'], True)
            self.assertNotEqual(metadata['feasibility_check'], 'direct_safe_corridor')
            self.assertTrue(metadata['feasibility_passed'])
        self.assertFalse(easy_blocked['metadata']['overlap_allowed'])

    def test_start_goal_stage_ranges_preserve_confirmed_construction(self):
        expected = {
            'easy': ((0.55, 0.85), (0.16, 0.28), (0.16, 0.33), 0.12, (-0.12, 0.12)),
            'medium': ((0.80, 0.95), (0.18, 0.30), (0.18, 0.35), 0.14, (-0.15, 0.15)),
            'hard': ((0.90, 1.10), (0.18, 0.30), (0.18, 0.36), 0.15, (-0.20, 0.20)),
        }
        scenario = ScenarioConfig()
        for level, (distance_ratio, state_z_ratio, goal_z_ratio, gap_ratio, psi_range) in expected.items():
            with self.subTest(level=level):
                payload = _generator(level, 0, seed=650).generate()
                state = np.asarray(payload['state'], dtype=np.float64)
                goal = np.asarray(payload['goal'], dtype=np.float64)
                distance = float(np.linalg.norm(goal - state[:3]))
                self.assertGreaterEqual(distance, scenario.target_distance * distance_ratio[0] - 1e-4)
                self.assertLessEqual(distance, scenario.target_distance * distance_ratio[1] + 1e-4)
                self.assertGreaterEqual(state[2], scenario.world_z_max * state_z_ratio[0])
                self.assertLessEqual(state[2], scenario.world_z_max * state_z_ratio[1])
                self.assertGreaterEqual(goal[2], scenario.world_z_max * goal_z_ratio[0])
                self.assertLessEqual(goal[2], scenario.world_z_max * goal_z_ratio[1])
                self.assertLessEqual(abs(goal[2] - state[2]), scenario.world_z_max * gap_ratio + 1e-6)
                self.assertGreaterEqual(state[4], psi_range[0])
                self.assertLessEqual(state[4], psi_range[1])

    def test_overlap_modes_and_aabb_count_are_distinct_metadata(self):
        easy = _generator('easy', 2, seed=701).generate()
        medium_separate = _generator('medium', 2, seed=702, overlap_probability=0.0).generate()
        medium_allowed = _generator('medium', 2, seed=703, overlap_probability=1.0).generate()
        hard = _generator('hard', 2, seed=704).generate()

        self.assertFalse(easy['metadata']['overlap_allowed'])
        self.assertEqual(easy['metadata']['aabb_overlap_pair_count'], 0)
        self.assertFalse(medium_separate['metadata']['overlap_allowed'])
        self.assertEqual(medium_separate['metadata']['aabb_overlap_pair_count'], 0)
        self.assertTrue(medium_allowed['metadata']['overlap_allowed'])
        self.assertTrue(hard['metadata']['overlap_allowed'])
        for payload in (medium_allowed, hard):
            self.assertIsInstance(payload['metadata']['aabb_overlap_pair_count'], int)

        hard_overlap = _generator('hard', 6, seed=106).generate()
        self.assertGreater(hard_overlap['metadata']['aabb_overlap_pair_count'], 0)

    def test_medium_overlap_mode_controls_separation_without_forcing_overlap(self):
        state = np.array([-500.0, 0.0, 100.0, 0.0, 0.0], dtype=np.float32)
        goal = np.array([500.0, 0.0, 100.0], dtype=np.float32)
        first = NoFlyZone('first', Sphere([0.0, 0.0, 100.0], 80.0))
        overlapping = NoFlyZone('overlapping', Sphere([0.0, 0.0, 100.0], 80.0))
        separated = NoFlyZone('separated', Sphere([0.0, 600.0, 100.0], 80.0))
        generator = _generator('medium', 2, seed=720)

        with mock.patch.object(
            generator,
            '_sample_zone_candidate',
            side_effect=[first, overlapping],
        ):
            allowed = generator._sample_zones(
                np.random.default_rng(1),
                state,
                goal,
                2,
                requested_shape_types=('sphere', 'sphere'),
                requested_ground_contact=(True, True),
                overlap_allowed=True,
                require_direct_path_blocker=False,
            )
        self.assertEqual([zone.zone_id for zone in allowed[0]], ['first', 'overlapping'])

        with mock.patch.object(
            generator,
            '_sample_zone_candidate',
            side_effect=[first, overlapping, separated],
        ):
            separated_result = generator._sample_zones(
                np.random.default_rng(1),
                state,
                goal,
                2,
                requested_shape_types=('sphere', 'sphere'),
                requested_ground_contact=(True, True),
                overlap_allowed=False,
                require_direct_path_blocker=False,
            )
        self.assertEqual([zone.zone_id for zone in separated_result[0]], ['first', 'separated'])

    def test_candidate_retries_keep_requested_shape_and_ground_contact(self):
        state = np.array([-500.0, 0.0, 100.0, 0.0, 0.0], dtype=np.float32)
        goal = np.array([500.0, 0.0, 100.0], dtype=np.float32)
        accepted = NoFlyZone('accepted', Sphere([0.0, 600.0, 100.0], 40.0))
        generator = _generator('easy', 1, seed=730, ground_probability=0.0)
        requests = []

        def sample_candidate(
            rng,
            *,
            zone_id,
            shape_type,
            reference_scale,
            target_point,
            ground_contact,
        ):
            del rng, zone_id, reference_scale, target_point
            requests.append((shape_type, ground_contact))
            return accepted if len(requests) == 3 else None

        with mock.patch.object(
            generator,
            '_sample_zone_candidate',
            side_effect=sample_candidate,
        ):
            result = generator._sample_zones(
                np.random.default_rng(1),
                state,
                goal,
                1,
                requested_shape_types=('sphere',),
                requested_ground_contact=(False,),
                overlap_allowed=False,
                require_direct_path_blocker=False,
            )

        self.assertIsNotNone(result)
        self.assertEqual(requests, [('sphere', False)] * 3)

    def test_outer_scenario_retries_keep_requested_shape_plan(self):
        generator = _generator('easy', 1, shape_type='sphere', seed=740)
        recorded_shapes = []
        recorded_ground_contact = []
        accepted = NoFlyZone('zone-000', Sphere([0.0, 900.0, 100.0], 20.0))

        def sample_zones(
            rng,
            state,
            goal,
            zone_count,
            *,
            requested_shape_types,
            requested_ground_contact,
            overlap_allowed,
            require_direct_path_blocker,
            rejection_counts,
        ):
            del rng, state, goal, zone_count, overlap_allowed
            del require_direct_path_blocker, rejection_counts
            recorded_shapes.append(requested_shape_types)
            recorded_ground_contact.append(requested_ground_contact)
            if len(recorded_shapes) == 1:
                return None
            return [accepted], [150.0]

        with mock.patch.object(generator, '_sample_zones', side_effect=sample_zones):
            payload = generator.generate()

        self.assertEqual(len(recorded_shapes), 2)
        self.assertIs(recorded_shapes[0], recorded_shapes[1])
        self.assertIs(recorded_ground_contact[0], recorded_ground_contact[1])
        self.assertEqual(recorded_shapes[0], ('sphere',))
        self.assertEqual(payload['metadata']['requested_shape_types'], ['sphere'])
        self.assertEqual(
            payload['metadata']['requested_ground_contact'],
            [recorded_ground_contact[0][0]],
        )

    def test_pyramids_have_explicit_ground_contact_without_primitive_choice(self):
        for shape_name in ('triangular_pyramid', 'quadrangular_pyramid'):
            with self.subTest(shape=shape_name):
                payload = _generator(
                    'easy',
                    1,
                    shape_type=shape_name,
                    seed=750,
                    ground_probability=0.0,
                ).generate()
                self.assertEqual(payload['metadata']['requested_shape_types'], [shape_name])
                self.assertEqual(payload['metadata']['requested_ground_contact'], [True])

    def test_same_seed_reproduces_sequences_and_different_seed_changes_payload(self):
        first = _generator('easy', 1, seed=801)
        second = _generator('easy', 1, seed=801)
        other = _generator('easy', 1, seed=802)

        first_sequence = [first.generate() for _ in range(3)]
        second_sequence = [second.generate() for _ in range(3)]
        other_sequence = [other.generate() for _ in range(3)]

        self.assertEqual(first_sequence, second_sequence)
        self.assertNotEqual(first_sequence, other_sequence)
        json.loads(json.dumps(first_sequence, allow_nan=False))

    def test_medium_regression_seed_records_bounded_feasibility_structure(self):
        generator = V2ScenarioGenerator(
            ScenarioConfig(),
            'medium',
            seed=20260904,
            config=V2ScenarioGeneratorConfig(
                direct_path_blocker_probability=_forced_blocker('medium', 0.97),
            ),
        )

        payloads = [generator.generate() for _ in range(3)]

        for payload in payloads:
            metadata = payload['metadata']
            zone_count = metadata['effective_zone_count']
            self.assertEqual(metadata['feasibility_check'], 'sparse_visibility_graph')
            self.assertGreaterEqual(metadata['feasibility_examined_nodes'], 0)
            self.assertGreaterEqual(metadata['feasibility_edge_checks'], 0)
            max_nodes = 2 + 14 * zone_count
            self.assertLessEqual(metadata['feasibility_examined_nodes'], max_nodes)
            self.assertLessEqual(
                metadata['feasibility_edge_checks'],
                zone_count * (max_nodes * (max_nodes - 1) // 2 + max_nodes - 1),
            )

    def test_generation_does_not_consume_global_numpy_random_state(self):
        np.random.seed(12345)
        before = np.random.get_state()
        _generator('easy', 1, seed=850).generate()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_exhaustion_reports_stage_count_attempts_and_rejections(self):
        scenario = ScenarioConfig(world_z_max=100.0, scenario_max_sampling_attempts=3)
        config = V2ScenarioGeneratorConfig(
            zone_count_probabilities=_forced_counts('easy', 1),
            shape_probabilities=_forced_shape('sphere'),
            zone_candidate_attempts=2,
            direct_path_blocker_probability=_forced_blocker('easy', 0.0),
        )
        generator = V2ScenarioGenerator(
            scenario,
            curriculum_level='easy',
            seed=860,
            config=config,
        )

        with self.assertRaises(V2ScenarioGenerationError) as caught:
            generator.generate()

        error = caught.exception
        self.assertEqual(error.curriculum_level, 'easy')
        self.assertEqual(error.requested_zone_count, 1)
        self.assertEqual(error.attempts, 3)
        self.assertIsInstance(error.scenario_seed, int)
        self.assertEqual(error.rejection_counts['zone_candidate_sampling'], 3)
        self.assertEqual(error.rejection_counts['shape_or_position_invalid'], 6)
        self.assertIn('requested_zone_count=1', str(error))

    def test_metadata_records_generation_contract_and_shape_parameters(self):
        payload = _generator('easy', 1, shape_type='ellipsoid', seed=901).generate()
        metadata = payload['metadata']
        required = {
            'generator',
            'generator_version',
            'scenario_seed',
            'requested_curriculum_level',
            'effective_curriculum_level',
            'requested_zone_count',
            'effective_zone_count',
            'shape_counts',
            'requested_shape_types',
            'requested_ground_contact',
            'requested_reference_scales',
            'overlap_allowed',
            'aabb_overlap_pair_count',
            'requested_direct_path_blocker',
            'direct_path_blocker_count',
            'feasibility_check',
            'feasibility_passed',
            'feasibility_examined_nodes',
            'feasibility_edge_checks',
            'generation_attempts',
            'rejection_counts',
        }
        self.assertTrue(required.issubset(metadata))
        zone_metadata = payload['zones'][0]['metadata']
        self.assertEqual(zone_metadata['requested_reference_scale'], metadata['requested_reference_scales'][0])
        self.assertIn('actual_shape_parameters', zone_metadata)

    def test_branch_draws_match_metadata_and_true_geometry(self):
        scenario = ScenarioConfig()
        margin = scenario.corridor_blocking_margin
        for level, seed in (('easy', 900), ('medium', 910)):
            with self.subTest(level=level):
                generator = _generator(level, 2, seed=seed)
                branches = set()
                for _ in range(24):
                    payload = generator.generate()
                    metadata = payload['metadata']
                    state = np.asarray(payload['state'], dtype=np.float64)
                    goal = np.asarray(payload['goal'], dtype=np.float64)
                    actual = sum(
                        zone.violates_segment(state[:3], goal, uav_radius=margin)
                        for zone in _zones(payload)
                    )
                    self.assertEqual(
                        actual, metadata['direct_path_blocker_count']
                    )
                    requested = metadata['requested_direct_path_blocker']
                    self.assertIs(requested, actual >= 1)
                    if requested:
                        self.assertNotEqual(
                            metadata['feasibility_check'], 'direct_safe_corridor'
                        )
                        self.assertTrue(metadata['feasibility_passed'])
                    else:
                        self.assertEqual(
                            metadata['feasibility_check'], 'direct_safe_corridor'
                        )
                    branches.add(requested)
                self.assertEqual(len(branches), 2)

    def test_hard_nonzero_scenarios_always_block_the_corridor(self):
        generator = _generator('hard', 2, seed=950)
        for _ in range(8):
            metadata = generator.generate()['metadata']
            self.assertTrue(metadata['requested_direct_path_blocker'])
            self.assertGreaterEqual(metadata['direct_path_blocker_count'], 1)
            self.assertNotEqual(metadata['feasibility_check'], 'direct_safe_corridor')


if __name__ == '__main__':
    unittest.main()
