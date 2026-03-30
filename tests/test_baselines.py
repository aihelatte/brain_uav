"""Small tests for baseline planner outputs."""

import unittest

from brain_uav.baselines import AStarPlanner, ArtificialPotentialFieldPlanner, HeuristicPlanner
from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import StaticNoFlyTrajectoryEnv


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


if __name__ == "__main__":
    unittest.main()
