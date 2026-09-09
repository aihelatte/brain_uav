from __future__ import annotations

import unittest

import numpy as np

from brain_uav.baselines import (
    ArtificialPotentialFieldPlanner,
    HeuristicPlanner,
    V2ArtificialPotentialFieldPlanner,
    V2HeuristicPlanner,
)
from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import (
    V2_ENV_SCENARIO_FORMAT,
    V2_ENV_SCENARIO_VERSION,
    V2StaticNoFlyTrajectoryEnv,
)
from brain_uav.geometry import (
    Box,
    Ellipsoid,
    NoFlyZone,
    QuadrangularPyramid,
    Sphere,
    TriangularPyramid,
)


def _scenario_config() -> ScenarioConfig:
    return ScenarioConfig(
        speed=2.0,
        world_xy=60.0,
        world_z_min=0.1,
        world_z_max=30.0,
        max_steps=8,
        goal_radius=0.5,
        warning_distance=12.0,
    )


def _payload(
    zones: list[NoFlyZone],
    *,
    state=(0.0, 0.0, 10.0, 0.0, 0.0),
    goal=(20.0, 0.0, 10.0),
) -> dict[str, object]:
    return {
        'format': V2_ENV_SCENARIO_FORMAT,
        'format_version': V2_ENV_SCENARIO_VERSION,
        'state': list(state),
        'goal': list(goal),
        'zones': [zone.to_dict() for zone in zones],
        'curriculum_level': 'easy',
    }


class TestV2BaselinePlanners(unittest.TestCase):
    def _assert_planners_return_legal_actions(self, zones: list[NoFlyZone]) -> None:
        env = V2StaticNoFlyTrajectoryEnv(
            _scenario_config(),
            RewardConfig(),
            fixed_scenarios=[_payload(zones)],
        )
        observation, _ = env.reset()
        for planner_type in (V2HeuristicPlanner, V2ArtificialPotentialFieldPlanner):
            with self.subTest(planner=planner_type.__name__):
                action = planner_type(env).act(observation)
                self.assertEqual(action.shape, (2,))
                self.assertEqual(action.dtype, np.float32)
                self.assertTrue(np.all(np.isfinite(action)))
                self.assertTrue(np.all(action >= env.action_space.low))
                self.assertTrue(np.all(action <= env.action_space.high))

    def test_zero_zone_actions_are_finite_and_bounded(self) -> None:
        self._assert_planners_return_legal_actions([])

    def test_v2_planners_are_distinct_from_legacy_planners(self) -> None:
        self.assertIsNot(V2HeuristicPlanner, HeuristicPlanner)
        self.assertIsNot(
            V2ArtificialPotentialFieldPlanner,
            ArtificialPotentialFieldPlanner,
        )

    def test_all_v2_shapes_use_the_unified_geometry_interface(self) -> None:
        shapes = (
            Sphere([0.0, 5.0, 10.0], 2.0),
            Ellipsoid([0.0, 5.0, 10.0], 2.0, 2.0, 2.0),
            Box([0.0, 5.0, 10.0], 4.0, 4.0, 4.0),
            TriangularPyramid([0.0, 4.0, 0.0], 4.0, 4.0, 12.0),
            QuadrangularPyramid([0.0, 4.0, 0.0], 4.0, 4.0, 12.0),
        )
        baseline_env = V2StaticNoFlyTrajectoryEnv(
            _scenario_config(),
            RewardConfig(),
            fixed_scenarios=[_payload([])],
        )
        baseline_observation, _ = baseline_env.reset()
        for index, shape in enumerate(shapes):
            with self.subTest(shape=type(shape).__name__):
                zone = NoFlyZone(f'zone-{index}', shape)
                self.assertFalse(hasattr(zone, 'center_xy'))
                self.assertFalse(hasattr(zone, 'radius'))
                clearance = zone.point_clearance(np.array([0.0, 0.0, 10.0]))
                self.assertGreater(clearance, 0.0)
                self.assertLess(clearance, _scenario_config().warning_distance)
                env = V2StaticNoFlyTrajectoryEnv(
                    _scenario_config(),
                    RewardConfig(),
                    fixed_scenarios=[_payload([zone])],
                )
                observation, _ = env.reset()
                for planner_type in (
                    V2HeuristicPlanner,
                    V2ArtificialPotentialFieldPlanner,
                ):
                    baseline_action = planner_type(baseline_env).act(
                        baseline_observation
                    )
                    action = planner_type(env).act(observation)
                    self.assertEqual(action.shape, (2,))
                    self.assertEqual(action.dtype, np.float32)
                    self.assertTrue(np.all(np.isfinite(action)))
                    self.assertTrue(np.all(action >= env.action_space.low))
                    self.assertTrue(np.all(action <= env.action_space.high))
                    self.assertFalse(np.allclose(action, baseline_action, atol=1e-7))

    def test_suspended_shapes_are_supported(self) -> None:
        zones = [
            NoFlyZone('sphere', Sphere([0.0, 7.0, 18.0], 2.0)),
            NoFlyZone('ellipsoid', Ellipsoid([4.0, 7.0, 18.0], 3.0, 2.0, 2.5)),
            NoFlyZone('box', Box([-4.0, 7.0, 18.0], 4.0, 6.0, 3.0)),
        ]
        self._assert_planners_return_legal_actions(zones)

    def test_action_is_explicitly_clipped_to_environment_bounds(self) -> None:
        zone = NoFlyZone('nearby', Sphere([-18.0, 0.0, 12.0], 1.0))
        env = V2StaticNoFlyTrajectoryEnv(
            _scenario_config(),
            RewardConfig(),
            fixed_scenarios=[_payload([zone])],
        )
        observation, _ = env.reset()
        for planner_type in (V2HeuristicPlanner, V2ArtificialPotentialFieldPlanner):
            action = planner_type(env).act(observation)
            np.testing.assert_array_less(
                action,
                env.action_space.high + np.finfo(np.float32).eps,
            )
            np.testing.assert_array_less(
                env.action_space.low - np.finfo(np.float32).eps,
                action,
            )


if __name__ == '__main__':
    unittest.main()
