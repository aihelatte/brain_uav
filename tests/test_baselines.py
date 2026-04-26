"""Small tests for baseline planner outputs."""

import unittest
from unittest import mock

import numpy as np

from brain_uav.baselines import AStarPlanner, ArtificialPotentialFieldPlanner, HeuristicPlanner
from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import StaticNoFlyTrajectoryEnv
from brain_uav.envs.static_no_fly_env_runtime import Zone


class TestBaselines(unittest.TestCase):
    """Baseline planners should all produce a 2D action."""

    def setUp(self) -> None:
        self.env = StaticNoFlyTrajectoryEnv(ScenarioConfig(), RewardConfig(), seed=11)
        self.obs, _ = self.env.reset(seed=11)

    def test_heuristic_action_shape(self):
        self.assertEqual(HeuristicPlanner(self.env).act(self.obs).shape, (2,))

    def test_apf_action_shape(self):
        self.assertEqual(ArtificialPotentialFieldPlanner(self.env).act(self.obs).shape, (2,))

    def test_astar_action_shape(self):
        self.assertEqual(AStarPlanner(self.env).act(self.obs).shape, (2,))

    def test_apf_uses_scaled_threshold_constant(self):
        planner = ArtificialPotentialFieldPlanner(self.env)
        self.env.state = np.array([10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.env.goal = self.env.state[:3].copy()
        self.env.zones = [Zone(center_xy=np.array([0.0, 0.0], dtype=np.float32), radius=1.0)]
        self.env.scenario.warning_distance = 1.0

        with mock.patch('brain_uav.baselines.apf.heading_to_action', side_effect=lambda gamma, psi, force, limits: force):
            force = planner.act(self.obs)

        np.testing.assert_allclose(force, np.zeros(3, dtype=np.float32))

    def test_heuristic_uses_scaled_influence_constant(self):
        planner = HeuristicPlanner(self.env)
        self.env.state = np.array([7.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.env.goal = self.env.state[:3].copy()
        self.env.zones = [Zone(center_xy=np.array([0.0, 0.0], dtype=np.float32), radius=1.0)]
        self.env.scenario.warning_distance = 1.0

        with mock.patch(
            'brain_uav.baselines.heuristic.heading_to_action',
            side_effect=lambda gamma, psi, direction, limits: direction,
        ):
            direction = planner.act(self.obs)

        np.testing.assert_allclose(direction, np.zeros(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
