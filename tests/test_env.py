import unittest

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import StaticNoFlyTrajectoryEnv


class TestStaticNoFlyEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = StaticNoFlyTrajectoryEnv(ScenarioConfig(), RewardConfig(), seed=7)

    def test_reset_and_shape(self):
        obs, info = self.env.reset(seed=7)
        self.assertEqual(obs.shape[0], self.env.observation_space.shape[0])
        self.assertIn("goal_distance", info)

    def test_step_contract(self):
        self.env.reset(seed=7)
        obs, reward, terminated, truncated, info = self.env.step(np.zeros(2, dtype=np.float32))
        self.assertEqual(obs.shape[0], self.env.observation_space.shape[0])
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("outcome", info)


if __name__ == "__main__":
    unittest.main()

