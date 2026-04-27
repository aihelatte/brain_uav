"""Small tests for environment API contract."""

import unittest
from unittest import mock

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import StaticNoFlyTrajectoryEnv
from brain_uav.envs.static_no_fly_env_runtime import Zone


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

    def test_config_uses_target_distance_derived_world_xy(self):
        cfg = ScenarioConfig()
        rewards = RewardConfig()

        self.assertEqual(cfg.target_distance, 700.0)
        self.assertAlmostEqual(cfg.world_xy, 525.0)
        self.assertAlmostEqual(cfg.world_z_max, 175.0)
        self.assertEqual(cfg.speed, 2.5)
        self.assertEqual(cfg.goal_radius, 5.0)
        self.assertEqual(cfg.world_z_min, 0.1)
        self.assertEqual(cfg.max_steps, 400)
        self.assertEqual(cfg.no_fly_radius_range, (100.0, 125.0))
        self.assertEqual(cfg.warning_distance, 40.0)
        self.assertEqual(cfg.boundary_warning_distance, 10.0)
        self.assertEqual(cfg.ground_warning_height, 4.0)
        self.assertEqual(cfg.descent_penalty_height, 12.0)
        self.assertEqual(cfg.start_zone_clearance, 30.0)
        self.assertEqual(cfg.corridor_blocking_margin, 20.0)
        self.assertEqual(cfg.max_start_goal_height_gap, 11.0)
        self.assertEqual(cfg.dual_zone_min_margin, 55.0)
        self.assertEqual(cfg.easy_two_zone_min_gap, 65.0)
        self.assertEqual(rewards.zone_penalty_weight, 240.0)
        self.assertEqual(rewards.zone_penalty_cap, 500.0)
        self.assertEqual(rewards.collision_penalty, 7500.0)
        self.assertEqual(rewards.min_progress_per_window, 2.0)
        self.assertEqual(rewards.breakthrough_reward_distance, 60.0)
        self.assertEqual(rewards.breakthrough_progress_threshold, 2.2)

    def test_curriculum_distance_ratios_include_benchmark_match(self):
        cfg = ScenarioConfig()

        self.assertEqual(cfg.distance_ratio_range_for_level('hard'), (0.90, 1.10))
        self.assertEqual(cfg.distance_ratio_range_for_level('benchmark'), (0.90, 1.10))

    def test_curriculum_radius_ranges(self):
        cfg = ScenarioConfig()

        self.assertEqual(cfg.radius_range_for_level('easy'), (60.0, 80.0))
        self.assertEqual(cfg.radius_range_for_level('easy_two_zone'), (75.0, 95.0))
        self.assertEqual(cfg.radius_range_for_level('medium'), (90.0, 110.0))
        self.assertEqual(cfg.radius_range_for_level('hard'), (100.0, 125.0))
        self.assertEqual(cfg.radius_range_for_level('benchmark'), (100.0, 125.0))

    def test_explicit_world_z_max_overrides_default_derivation(self):
        cfg = ScenarioConfig(world_z_max=40.0)

        self.assertEqual(cfg.world_xy, 525.0)
        self.assertEqual(cfg.world_z_max, 40.0)

    def test_curriculum_scenarios_follow_distance_ranges(self):
        for level in ('easy', 'easy_two_zone', 'medium', 'hard'):
            with self.subTest(level=level):
                env = StaticNoFlyTrajectoryEnv(ScenarioConfig(), RewardConfig(), seed=7)
                scenario = env._sample_curriculum_scenario(level)
                self.assertIsNotNone(scenario)
                assert scenario is not None

                state = np.asarray(scenario['state'], dtype=np.float32)
                goal = np.asarray(scenario['goal'], dtype=np.float32)
                sampled_distance = float(np.linalg.norm(goal - state[:3]))
                distance_min, distance_max = env.scenario.distance_range_for_level(level)
                radius_min, radius_max = env.scenario.radius_range_for_level(level)
                state_z_range, goal_z_range, max_height_gap = env._z_sampling_spec(level)

                self.assertGreaterEqual(sampled_distance, distance_min)
                self.assertLessEqual(sampled_distance, distance_max)
                self.assertGreaterEqual(float(state[2]), state_z_range[0])
                self.assertLessEqual(float(state[2]), state_z_range[1])
                self.assertGreaterEqual(float(goal[2]), goal_z_range[0])
                self.assertLessEqual(float(goal[2]), goal_z_range[1])
                self.assertGreaterEqual(float(state[2]), env.scenario.world_z_min)
                self.assertGreaterEqual(float(goal[2]), env.scenario.world_z_min)
                self.assertLessEqual(float(state[2]), env.scenario.world_z_max)
                self.assertLessEqual(float(goal[2]), env.scenario.world_z_max)
                self.assertLessEqual(abs(float(goal[2] - state[2])), max_height_gap)
                for zone in scenario['zones']:
                    self.assertGreaterEqual(float(zone['radius']), radius_min)
                    self.assertLessEqual(float(zone['radius']), radius_max)

                zones = [
                    Zone(center_xy=np.asarray(zone['center_xy'], dtype=np.float32), radius=float(zone['radius']))
                    for zone in scenario['zones']
                ]
                blockers = env._count_corridor_blockers(
                    state,
                    goal,
                    zones,
                    margin=env.scenario.corridor_blocking_margin,
                )
                if level == 'easy':
                    self.assertEqual(blockers, 0)
                if level == 'hard':
                    for idx, zone_a in enumerate(zones):
                        for zone_b in zones[idx + 1:]:
                            surface_gap = float(np.linalg.norm(zone_a.center_xy - zone_b.center_xy))
                            surface_gap -= zone_a.radius + zone_b.radius
                            self.assertGreaterEqual(surface_gap, env.scenario.dual_zone_min_margin)

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
