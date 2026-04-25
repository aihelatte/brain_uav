"""Small tests for environment API contract."""

import unittest

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import StaticNoFlyTrajectoryEnv


class TestStaticNoFlyEnv(unittest.TestCase):
    """Ensure reset/step follow the expected interface."""

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

    def test_config_distance_scales_are_reduced_tenfold(self):
        cfg = ScenarioConfig()
        rewards = RewardConfig()

        self.assertEqual(cfg.speed, 2.5)
        self.assertEqual(cfg.goal_radius, 4.5)
        self.assertEqual(cfg.world_xy, 80.0)
        self.assertEqual(cfg.world_z_min, 0.1)
        self.assertEqual(cfg.world_z_max, 40.0)
        self.assertEqual(cfg.no_fly_radius_range, (6.0, 14.0))
        self.assertEqual(cfg.warning_distance, 10.0)
        self.assertEqual(cfg.boundary_warning_distance, 10.0)
        self.assertEqual(cfg.ground_warning_height, 4.0)
        self.assertEqual(cfg.descent_penalty_height, 12.0)
        self.assertEqual(cfg.start_zone_clearance, 2.5)
        self.assertEqual(cfg.corridor_blocking_margin, 3.5)
        self.assertEqual(cfg.max_start_goal_height_gap, 11.0)
        self.assertEqual(cfg.dual_zone_min_margin, 13.0)
        self.assertEqual(cfg.easy_two_zone_min_gap, 22.0)
        self.assertEqual(rewards.min_progress_per_window, 2.0)
        self.assertEqual(rewards.breakthrough_reward_distance, 22.0)
        self.assertEqual(rewards.breakthrough_progress_threshold, 2.2)

    def test_easy_scenario_sampling_uses_scaled_ranges(self):
        scenario = self.env._sample_easy_scenario()
        self.assertIsNotNone(scenario)
        assert scenario is not None

        state = np.asarray(scenario['state'], dtype=np.float32)
        goal = np.asarray(scenario['goal'], dtype=np.float32)
        zone = scenario['zones'][0]

        self.assertGreaterEqual(state[2], 11.0)
        self.assertLessEqual(state[2], 15.5)
        self.assertGreaterEqual(goal[2], 10.5)
        self.assertLessEqual(goal[2], 16.5)
        self.assertLessEqual(abs(float(goal[2] - state[2])), 5.5)
        self.assertGreaterEqual(zone['radius'], 7.0)
        self.assertLessEqual(zone['radius'], 10.5)


if __name__ == "__main__":
    unittest.main()
