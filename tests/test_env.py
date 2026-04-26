"""Small tests for environment API contract."""

import unittest
from unittest import mock

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

    def test_progress_reward_uses_scale_compensation(self):
        prev_state = np.zeros(5, dtype=np.float32)
        prev_action = np.zeros(2, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        self.env.state = np.zeros(5, dtype=np.float32)

        with mock.patch.object(self.env, '_breakthrough_reward', return_value=0.0), \
             mock.patch.object(self.env, '_action_change_penalty', return_value=0.0), \
             mock.patch.object(self.env, '_zone_warning_penalty', return_value=0.0), \
             mock.patch.object(self.env, '_boundary_warning_penalty', return_value=0.0), \
             mock.patch.object(self.env, '_ground_warning_penalty', return_value=0.0), \
             mock.patch.object(self.env, '_descent_trend_penalty', return_value=0.0), \
             mock.patch.object(self.env, '_inefficiency_penalty', return_value=0.0):
            reward = self.env._compute_reward(
                prev_state=prev_state,
                prev_action=prev_action,
                prev_distance=10.0,
                new_distance=9.5,
                action=action,
                outcome='running',
                prev_best_goal_distance=10.0,
            )

        self.assertAlmostEqual(reward, self.env.rewards.progress_weight * 5.0 - self.env.rewards.step_penalty)

    def test_breakthrough_reward_uses_scale_compensation(self):
        self.env.recent_progress = [0.23] * self.env.rewards.progress_window_size
        self.env.state = np.zeros(5, dtype=np.float32)

        with mock.patch.object(self.env, '_nearest_zone_surface_clearance', return_value=0.0):
            reward = self.env._breakthrough_reward(
                new_distance=5.0,
                prev_best_goal_distance=6.0,
                outcome='running',
            )

        expected_window_progress = sum(self.env.recent_progress) * 10.0
        self.assertAlmostEqual(
            reward,
            min(
                self.env.rewards.breakthrough_reward_weight * expected_window_progress,
                self.env.rewards.breakthrough_reward_cap,
            ),
        )


if __name__ == "__main__":
    unittest.main()
