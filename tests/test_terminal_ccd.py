"""Tests for CCD-based goal detection."""

import unittest

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import StaticNoFlyTrajectoryEnv


class TestTerminalCCD(unittest.TestCase):
    def test_segment_crossing_goal_counts_as_success(self):
        scenario = ScenarioConfig(
            speed=25.0,
            dt=1.0,
            goal_radius=5.0,
            max_steps=5,
            min_no_fly_zones=0,
            max_no_fly_zones=0,
        )
        env = StaticNoFlyTrajectoryEnv(
            scenario=scenario,
            rewards=RewardConfig(),
            fixed_scenarios=[
                {
                    'state': [0.0, 0.0, 100.0, 0.0, 0.0],
                    'goal': [10.0, 0.0, 100.0],
                    'zones': [],
                    'curriculum_level': 'test',
                }
            ],
        )

        env.reset()
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['outcome'], 'goal')
        self.assertGreater(info['goal_distance'], scenario.goal_radius)
        self.assertAlmostEqual(info['segment_goal_distance'], 0.0, places=5)
        self.assertTrue(info['goal_reached_by_segment'])


if __name__ == '__main__':
    unittest.main()
