"""Contract and semantic-equivalence tests for the independent V2 environment."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import StaticNoFlyTrajectoryEnv
from brain_uav.geometry import (
    Box,
    Ellipsoid,
    NoFlyZone,
    QuadrangularPyramid,
    Sphere,
    TriangularPyramid,
)
from brain_uav.observations import V2Observation, ZONE_FEATURE_INDEX


def _v2_api():
    from brain_uav.envs import (
        V2_ENV_SCENARIO_FORMAT,
        V2_ENV_SCENARIO_VERSION,
        V2StaticNoFlyTrajectoryEnv,
    )

    return V2StaticNoFlyTrajectoryEnv, V2_ENV_SCENARIO_FORMAT, V2_ENV_SCENARIO_VERSION


def _scenario(state, goal, zones, *, level='test', metadata=None):
    _, scenario_format, scenario_version = _v2_api()
    payload = {
        'format': scenario_format,
        'format_version': scenario_version,
        'state': list(state),
        'goal': list(goal),
        'zones': [zone.to_dict() for zone in zones],
        'curriculum_level': level,
    }
    if metadata is not None:
        payload['metadata'] = metadata
    return payload


def _config(**overrides):
    values = {
        'world_xy': 100.0,
        'world_z_min': 0.1,
        'world_z_max': 50.0,
        'speed': 1.0,
        'dt': 1.0,
        'max_steps': 20,
        'goal_radius': 0.25,
    }
    values.update(overrides)
    return ScenarioConfig(**values)


class TestV2StaticNoFlyTrajectoryEnv(unittest.TestCase):
    def make_env(self, scenario_payload, *, config=None, radius=0.0):
        env_type, _, _ = _v2_api()
        return env_type(
            scenario=config or _config(),
            rewards=RewardConfig(),
            seed=17,
            fixed_scenarios=[scenario_payload],
            uav_collision_radius=radius,
        )

    def test_reset_without_explicit_or_fixed_scenario_is_rejected(self):
        env_type, _, _ = _v2_api()
        env = env_type(_config(), RewardConfig(), seed=1)
        with self.assertRaisesRegex(RuntimeError, 'random V2 scenario generation'):
            env.reset()

    def test_reward_secondary_zone_ratio_is_validated(self):
        self.assertEqual(RewardConfig().zone_secondary_penalty_ratio, 0.2)
        for value in (-0.1, 1.1, np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RewardConfig(zone_secondary_penalty_ratio=value)

    def test_zero_zones_reset_step_and_structured_contract(self):
        payload = _scenario([0.0, 0.0, 10.0, 0.0, 0.0], [20.0, 0.0, 10.0], [])
        env = self.make_env(payload)

        observation, reset_info = env.reset()
        next_observation, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )

        self.assertIsInstance(observation, V2Observation)
        self.assertIsInstance(next_observation, V2Observation)
        self.assertEqual(observation.zone_features.shape, (0, 19))
        self.assertEqual(next_observation.zone_features.shape, (0, 19))
        self.assertIsNone(env.observation_space)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(math.isfinite(reward))
        for current_info in (reset_info, info):
            self.assertTrue(current_info['line_to_goal_safe'])
            self.assertEqual(current_info['zone_warning_penalty'], 0.0)
            self.assertTrue(math.isinf(current_info['min_zone_clearance']))
        self.assertEqual(info['outcome'], 'running')

    def test_zone_counts_zero_one_five_six_and_ten_preserve_order(self):
        for count in (0, 1, 5, 6, 10):
            zones = [
                NoFlyZone(f'z-{index}', Sphere([10.0 + index, 5.0, 10.0], 0.25))
                for index in range(count)
            ]
            env = self.make_env(
                _scenario([0.0, 0.0, 10.0, 0.0, 0.0], [80.0, 0.0, 10.0], zones)
            )
            with self.subTest(count=count):
                observation, _ = env.reset()
                self.assertEqual(len(env.zones), count)
                self.assertEqual(observation.zone_features.shape, (count, 19))
                self.assertEqual([zone.zone_id for zone in env.zones], [f'z-{i}' for i in range(count)])
                if count:
                    expected = np.array([10.0 + i for i in range(count)]) / 200.0
                    np.testing.assert_allclose(
                        observation.zone_features[:, ZONE_FEATURE_INDEX['zone_forward_norm']],
                        expected,
                        atol=1e-7,
                    )

    def test_each_shape_detects_endpoint_entry_full_segment_crossing_and_miss(self):
        shapes = {
            'sphere': Sphere([0.0, 0.0, 2.0], 0.5),
            'ellipsoid': Ellipsoid([0.0, 0.0, 2.0], 0.5, 0.8, 1.0),
            'box': Box([0.0, 0.0, 2.0], 1.0, 1.0, 1.0),
            'triangular': TriangularPyramid([0.0, 0.0, 0.0], 2.0, 2.0, 4.0),
            'quadrangular': QuadrangularPyramid([0.0, 0.0, 0.0], 2.0, 2.0, 4.0),
        }
        for name, shape in shapes.items():
            zone = NoFlyZone(name, shape)
            with self.subTest(shape=name, case='endpoint'):
                endpoint_env = self.make_env(
                    _scenario([-1.0, 0.0, 2.0, 0.0, 0.0], [20.0, 0.0, 2.0], [zone]),
                    config=_config(speed=1.0),
                )
                endpoint_env.reset()
                _, _, terminated, _, info = endpoint_env.step(np.zeros(2, dtype=np.float32))
                self.assertTrue(terminated)
                self.assertEqual(info['outcome'], 'collision')

            with self.subTest(shape=name, case='crossing'):
                crossing_env = self.make_env(
                    _scenario([-2.0, 0.0, 2.0, 0.0, 0.0], [20.0, 0.0, 2.0], [zone]),
                    config=_config(speed=4.0),
                )
                crossing_env.reset()
                _, _, terminated, _, info = crossing_env.step(np.zeros(2, dtype=np.float32))
                self.assertTrue(terminated)
                self.assertEqual(info['outcome'], 'collision')
                self.assertFalse(zone.violates_point(crossing_env.state[:3]))

            with self.subTest(shape=name, case='miss'):
                miss_env = self.make_env(
                    _scenario([-2.0, 5.0, 2.0, 0.0, 0.0], [20.0, 5.0, 2.0], [zone]),
                    config=_config(speed=4.0),
                )
                miss_env.reset()
                _, _, terminated, truncated, info = miss_env.step(np.zeros(2, dtype=np.float32))
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertEqual(info['outcome'], 'running')

    def test_overlap_margin_and_uav_radius_use_unified_zone_queries(self):
        overlapping = [
            NoFlyZone('first', Sphere([0.0, 0.0, 2.0], 0.5)),
            NoFlyZone('second', Box([0.0, 0.0, 2.0], 1.0, 1.0, 1.0)),
        ]
        overlap_env = self.make_env(
            _scenario([-2.0, 0.0, 2.0, 0.0, 0.0], [20.0, 0.0, 2.0], overlapping),
            config=_config(speed=4.0),
        )
        overlap_env.reset()
        _, _, terminated, _, info = overlap_env.step(np.zeros(2, dtype=np.float32))
        self.assertEqual(len(overlap_env.zones), 2)
        self.assertTrue(terminated)
        self.assertEqual(info['outcome'], 'collision')

        expanded = NoFlyZone('expanded', Sphere([2.0, 0.0, 2.0], 0.5), safety_margin=0.25)
        expanded_env = self.make_env(
            _scenario([0.0, 0.0, 2.0, 0.0, 0.0], [20.0, 0.0, 2.0], [expanded]),
            config=_config(speed=1.0),
            radius=0.25,
        )
        _, reset_info = expanded_env.reset()
        self.assertAlmostEqual(reset_info['min_zone_clearance'], 1.0)
        _, _, terminated, _, info = expanded_env.step(np.zeros(2, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertEqual(info['outcome'], 'collision')

    def test_line_to_goal_is_safe_requires_all_zones_to_be_clear(self):
        blocking = NoFlyZone('blocking', Sphere([5.0, 0.0, 2.0], 1.0))
        clear = NoFlyZone('clear', Sphere([5.0, 5.0, 2.0], 1.0))
        state = [0.0, 0.0, 2.0, 0.0, 0.0]
        goal = [10.0, 0.0, 2.0]

        self.assertTrue(self.make_env(_scenario(state, goal, [])).reset()[1]['line_to_goal_safe'])
        self.assertTrue(self.make_env(_scenario(state, goal, [clear])).reset()[1]['line_to_goal_safe'])
        blocked = self.make_env(_scenario(state, goal, [clear, blocking]))
        blocked.reset()
        self.assertFalse(blocked.line_to_goal_is_safe())

    def test_export_round_trip_preserves_geometry_safety_and_observation(self):
        zones = [
            NoFlyZone('sphere', Sphere([5.0, 2.0, 4.0], 1.0), 0.3, {'tag': 'a'}),
            NoFlyZone('box', Box([8.0, -2.0, 3.0], 2.0, 4.0, 2.0), 0.2),
        ]
        env = self.make_env(_scenario([0.0, 0.0, 4.0, 0.1, 0.2], [20.0, 1.0, 5.0], zones))
        first_obs, _ = env.reset()
        exported = env.export_scenario()
        restored = self.make_env(exported)
        second_obs, _ = restored.reset()

        np.testing.assert_array_equal(restored.state, env.initial_state)
        np.testing.assert_array_equal(restored.goal, env.goal)
        self.assertEqual([zone.to_dict() for zone in restored.zones], [zone.to_dict() for zone in env.zones])
        np.testing.assert_allclose(second_obs.ego_features, first_obs.ego_features)
        np.testing.assert_allclose(second_obs.goal_features, first_obs.goal_features)
        np.testing.assert_allclose(second_obs.zone_features, first_obs.zone_features)

    def test_strict_v2_scenario_rejects_legacy_zone_payload(self):
        legacy = {
            'format': 'v2_static_no_fly_scenario',
            'format_version': 1,
            'state': [0.0, 0.0, 2.0, 0.0, 0.0],
            'goal': [10.0, 0.0, 2.0],
            'zones': [{'center_xy': [5.0, 0.0], 'radius': 1.0}],
            'curriculum_level': 'test',
        }
        with self.assertRaisesRegex(ValueError, 'NoFlyZone'):
            self.make_env(legacy).reset()

    def test_sphere_warning_penalty_matches_surface_clearance_formula(self):
        zone = NoFlyZone('sphere', Sphere([0.0, 0.0, 10.0], 2.0))
        env = self.make_env(
            _scenario([3.5, 0.0, 10.0, 0.0, 0.0], [20.0, 0.0, 10.0], [zone])
        )
        env.reset()
        clearance = np.linalg.norm(env.state[:3] - zone.shape.center) - zone.shape.radius
        ratio = np.clip((env.scenario.warning_distance - clearance) / env.scenario.warning_distance, 0.0, 1.0)
        expected = min(env.rewards.zone_penalty_weight * ratio**2, env.rewards.zone_penalty_cap)
        self.assertAlmostEqual(env._zone_warning_penalty(env.state[:3]), expected)

    def test_multi_zone_warning_penalty_uses_primary_and_bounded_secondary(self):
        state = [0.0, 0.0, 10.0, 0.0, 0.0]
        goal = [20.0, 0.0, 10.0]

        def penalty(count):
            zones = [NoFlyZone(f'danger-{index}', Sphere([0.0, 0.0, 10.0], 1.0)) for index in range(count)]
            env = self.make_env(_scenario(state, goal, zones))
            env.reset()
            return env._zone_warning_penalty(env.state[:3])

        self.assertAlmostEqual(penalty(0), 0.0)
        self.assertAlmostEqual(penalty(1), 450.0)
        self.assertAlmostEqual(penalty(2), 540.0)
        self.assertAlmostEqual(penalty(6), 540.0)

        main = NoFlyZone('main', Sphere([0.0, 0.0, 10.0], 1.0))
        far = NoFlyZone('far', Sphere([80.0, 80.0, 10.0], 1.0))
        env = self.make_env(_scenario(state, goal, [main, far]))
        env.reset()
        self.assertAlmostEqual(env._zone_warning_penalty(env.state[:3]), 450.0)
        self.assertLessEqual(env._zone_warning_penalty(env.state[:3]), env.rewards.zone_penalty_cap)

    def test_warning_penalty_keeps_safety_margin_and_uav_radius_semantics(self):
        zone = NoFlyZone('expanded', Sphere([111.0, 0.0, 10.0], 1.0), safety_margin=20.0)
        payload = _scenario([0.0, 0.0, 10.0, 0.0, 0.0], [20.0, 0.0, 10.0], [zone])
        point_env = self.make_env(payload, radius=0.0)
        radius_env = self.make_env(payload, radius=10.0)
        point_env.reset()
        radius_env.reset()

        self.assertAlmostEqual(point_env.min_zone_clearance(), 90.0)
        self.assertAlmostEqual(radius_env.min_zone_clearance(), 80.0)
        self.assertAlmostEqual(point_env._zone_warning_penalty(point_env.state[:3]), 4.5)
        self.assertAlmostEqual(radius_env._zone_warning_penalty(radius_env.state[:3]), 18.0)

    def test_lifecycle_reuses_one_exact_point_clearance_query_per_zone(self):
        zones = [
            NoFlyZone('sphere', Sphere([12.0, 16.0, 10.0], 2.0), 0.3),
            NoFlyZone('ellipsoid', Ellipsoid([18.0, -16.0, 10.0], 2.0, 3.0, 4.0), 0.2),
            NoFlyZone('box', Box([24.0, 16.0, 10.0], 4.0, 6.0, 8.0), 0.1),
            NoFlyZone('triangle', TriangularPyramid([30.0, -16.0, 0.0], 6.0, 5.0, 8.0), 0.4),
            NoFlyZone('quadrangle', QuadrangularPyramid([36.0, 16.0, 0.0], 7.0, 6.0, 9.0), 0.5),
        ]
        payload = _scenario(
            [0.0, 0.0, 10.0, 0.0, 0.0],
            [80.0, 0.0, 10.0],
            zones,
            metadata={'direct_path_blocker_count': 0},
        )
        radius = 0.75
        action = np.array([0.01, -0.02], dtype=np.float32)
        reference = self.make_env(payload, radius=radius)
        reference_reset = reference.reset()
        reference_step = reference.step(action)

        env = self.make_env(payload, radius=radius)
        original = NoFlyZone.point_clearance
        with mock.patch.object(
            NoFlyZone,
            'point_clearance',
            autospec=True,
            side_effect=original,
        ) as clearance:
            reset_observation, reset_info = env.reset()
            self.assertEqual(clearance.call_count, len(zones))
            clearance.reset_mock()

            step_result = env.step(action)
            self.assertEqual(clearance.call_count, len(zones))
            clearance.reset_mock()

            env.set_goal([75.0, 1.0, 11.0])
            self.assertEqual(clearance.call_count, len(zones))
            clearance.reset_mock()

            other_position = np.array([2.0, 3.0, 12.0], dtype=np.float64)
            env.min_zone_clearance(other_position)
            self.assertEqual(clearance.call_count, len(zones))
            clearance.reset_mock()
            env._zone_warning_penalty(other_position)
            self.assertEqual(clearance.call_count, len(zones))

        np.testing.assert_array_equal(
            reset_observation.zone_features,
            reference_reset[0].zone_features,
        )
        self.assertEqual(reset_info, reference_reset[1])
        np.testing.assert_array_equal(
            step_result[0].zone_features,
            reference_step[0].zone_features,
        )
        self.assertEqual(step_result[1:], reference_step[1:])

        direct_clearances = np.asarray(
            [
                original(zone, env.state[:3], uav_radius=radius)
                for zone in env.zones
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            step_result[0].zone_features[:, ZONE_FEATURE_INDEX['point_clearance_norm']],
            (direct_clearances / env.observation_scales.world_diagonal).astype(np.float32),
        )
        self.assertEqual(step_result[4]['min_zone_clearance'], float(direct_clearances.min()))
        warning_distance = max(env.scenario.warning_distance, 1e-6)
        individual = []
        for clearance_value in direct_clearances:
            intrusion = warning_distance - float(clearance_value)
            if intrusion > 0.0:
                ratio = float(np.clip(intrusion / warning_distance, 0.0, 1.0))
                individual.append(env.rewards.zone_penalty_weight * ratio**2)
        primary = max(individual)
        secondary = sum(individual) - primary
        expected_penalty = min(
            primary
            + env.rewards.zone_secondary_penalty_ratio * min(secondary, primary),
            env.rewards.zone_penalty_cap,
        )
        self.assertEqual(step_result[4]['zone_warning_penalty'], expected_penalty)

    def test_random_generator_is_optional_and_lower_priority_than_explicit_sources(self):
        from brain_uav.envs import V2ScenarioGenerator, V2ScenarioGeneratorConfig

        count_probabilities = {
            'easy': {0: 1.0},
            'medium': {0: 1.0},
            'hard': {0: 1.0},
        }
        generator = V2ScenarioGenerator(
            ScenarioConfig(),
            curriculum_level='easy',
            seed=123,
            config=V2ScenarioGeneratorConfig(zone_count_probabilities=count_probabilities),
        )
        env_type, _, _ = _v2_api()
        random_env = env_type(ScenarioConfig(), RewardConfig(), scenario_generator=generator)
        observation, info = random_env.reset()
        self.assertEqual(observation.zone_features.shape, (0, 19))
        self.assertEqual(info['curriculum_level'], 'easy')
        self.assertEqual(info['zone_count'], 0)
        self.assertIsInstance(info['scenario_seed'], int)
        self.assertEqual(info['requested_zone_count'], 0)
        self.assertEqual(info['direct_path_blocker_count'], 0)
        self.assertFalse(info['overlap_allowed'])

        fixed = _scenario([0.0, 0.0, 10.0, 0.0, 0.0], [20.0, 0.0, 10.0], [], level='fixed')
        fixed_env = env_type(
            ScenarioConfig(),
            RewardConfig(),
            fixed_scenarios=[fixed],
            scenario_generator=generator,
        )
        with mock.patch.object(generator, 'generate', wraps=generator.generate) as generate:
            _, fixed_info = fixed_env.reset()
            self.assertEqual(fixed_info['curriculum_level'], 'fixed')
            generate.assert_not_called()

            explicit = _scenario([1.0, 0.0, 10.0, 0.0, 0.0], [21.0, 0.0, 10.0], [], level='explicit')
            _, explicit_info = fixed_env.reset(options={'scenario': explicit})
            self.assertEqual(explicit_info['curriculum_level'], 'explicit')
            generate.assert_not_called()

    def test_overlap_allowed_info_preserves_declared_generation_semantics(self):
        from brain_uav.envs import V2ScenarioGenerator, V2ScenarioGeneratorConfig

        count_probabilities = {
            'easy': {0: 1.0},
            'medium': {0: 1.0},
            'hard': {0: 1.0},
        }
        generator_config = V2ScenarioGeneratorConfig(
            zone_count_probabilities=count_probabilities
        )
        env_type, _, _ = _v2_api()
        easy_env = env_type(
            ScenarioConfig(),
            RewardConfig(),
            scenario_generator=V2ScenarioGenerator(
                ScenarioConfig(),
                'easy',
                seed=124,
                config=generator_config,
            ),
        )
        hard_env = env_type(
            ScenarioConfig(),
            RewardConfig(),
            scenario_generator=V2ScenarioGenerator(
                ScenarioConfig(),
                'hard',
                seed=125,
                config=generator_config,
            ),
        )

        self.assertIs(easy_env.reset()[1]['overlap_allowed'], False)
        self.assertIs(hard_env.reset()[1]['overlap_allowed'], True)

        state = [0.0, 0.0, 10.0, 0.0, 0.0]
        goal = [20.0, 0.0, 10.0]
        fixed_without_metadata = self.make_env(_scenario(state, goal, []))
        self.assertIsNone(fixed_without_metadata.reset()[1]['overlap_allowed'])

        for declared in (False, True):
            with self.subTest(declared=declared):
                explicit = self.make_env(
                    _scenario(
                        state,
                        goal,
                        [],
                        metadata={'overlap_allowed': declared},
                    )
                )
                self.assertIs(explicit.reset()[1]['overlap_allowed'], declared)

        invalid = self.make_env(
            _scenario(
                state,
                goal,
                [],
                metadata={'overlap_allowed': 'yes'},
            )
        )
        self.assertIsNone(invalid.reset()[1]['overlap_allowed'])

    def test_scenario_metadata_is_strict_copied_and_round_trips(self):
        metadata = {
            'scenario_seed': 123,
            'requested_zone_count': 0,
            'direct_path_blocker_count': 0,
            'overlap_allowed': False,
            'nested': {'values': [1, True, None]},
        }
        payload = _scenario(
            [0.0, 0.0, 10.0, 0.0, 0.0],
            [20.0, 0.0, 10.0],
            [],
            metadata=metadata,
        )
        env = self.make_env(payload)
        env.reset()
        metadata['nested']['values'][0] = 99
        exported = env.export_scenario()
        self.assertEqual(exported['metadata']['nested']['values'][0], 1)

        restored = self.make_env(exported)
        restored.reset()
        self.assertEqual(restored.export_scenario(), exported)
        exported['metadata']['nested']['values'][0] = -1
        self.assertEqual(restored.export_scenario()['metadata']['nested']['values'][0], 1)

        invalid_values = (
            {'bad': (1, 2)},
            {'bad': {1, 2}},
            {'bad': np.array([1.0])},
            {1: 'bad-key'},
            {'bad': np.nan},
        )
        for invalid in invalid_values:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.make_env(_scenario([0.0, 0.0, 10.0, 0.0, 0.0], [20.0, 0.0, 10.0], [], metadata=invalid)).reset()

    def test_no_zone_dynamics_reward_and_termination_match_legacy_environment(self):
        cases = (
            ('running', _config(speed=1.0), [0.0, 0.0, 10.0, 0.0, 0.0], [20.0, 0.0, 10.0]),
            ('goal', _config(speed=25.0, goal_radius=5.0), [0.0, 0.0, 10.0, 0.0, 0.0], [10.0, 0.0, 10.0]),
            ('ground', _config(speed=1.0), [0.0, 0.0, 0.2, -0.6, 0.0], [20.0, 0.0, 10.0]),
            ('boundary', _config(world_xy=10.0, speed=1.0), [9.5, 0.0, 10.0, 0.0, 0.0], [-5.0, 0.0, 10.0]),
            ('timeout', _config(speed=1.0, max_steps=1), [0.0, 0.0, 10.0, 0.0, 0.0], [20.0, 0.0, 10.0]),
        )
        action = np.zeros(2, dtype=np.float32)
        for expected_outcome, config, state, goal in cases:
            with self.subTest(outcome=expected_outcome):
                legacy_payload = {'state': state, 'goal': goal, 'zones': [], 'curriculum_level': 'test'}
                legacy = StaticNoFlyTrajectoryEnv(config, RewardConfig(), fixed_scenarios=[legacy_payload])
                v2 = self.make_env(_scenario(state, goal, []), config=config)
                legacy.reset()
                v2.reset()
                _, old_reward, old_terminated, old_truncated, old_info = legacy.step(action)
                _, new_reward, new_terminated, new_truncated, new_info = v2.step(action)

                np.testing.assert_array_equal(v2.state, legacy.state)
                self.assertAlmostEqual(new_reward, old_reward, places=7)
                self.assertAlmostEqual(new_info['progress'], old_info['progress'], places=7)
                self.assertEqual((new_terminated, new_truncated), (old_terminated, old_truncated))
                self.assertEqual(new_info['outcome'], old_info['outcome'])
                self.assertEqual(new_info['outcome'], expected_outcome)


if __name__ == '__main__':
    unittest.main()
