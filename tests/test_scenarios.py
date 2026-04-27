"""Tests for benchmark scenario generation ranges."""

import unittest

import numpy as np

from brain_uav.config import ScenarioConfig
from brain_uav.scenarios import BENCHMARK_CATEGORIES, generate_benchmark_suite


class TestBenchmarkScenarios(unittest.TestCase):
    def test_generate_benchmark_suite_uses_current_distance_and_height_ranges(self):
        cfg = ScenarioConfig()
        payload = generate_benchmark_suite(seed=123, count_per_category=2)
        distance_min, distance_max = cfg.distance_range_for_level('benchmark')
        start_z_range = (0.18 * cfg.world_z_max, 0.30 * cfg.world_z_max)
        goal_z_range = (0.18 * cfg.world_z_max, 0.36 * cfg.world_z_max)
        max_height_gap = 0.15 * cfg.world_z_max

        self.assertEqual(payload['categories'], list(BENCHMARK_CATEGORIES))
        self.assertEqual(payload['total_scenarios'], len(BENCHMARK_CATEGORIES) * 2)

        for item in payload['scenarios']:
            scenario = item['scenario']
            state = np.asarray(scenario['state'], dtype=np.float32)
            goal = np.asarray(scenario['goal'], dtype=np.float32)
            sampled_distance = float(np.linalg.norm(goal - state[:3]))

            self.assertGreaterEqual(sampled_distance, distance_min)
            self.assertLessEqual(sampled_distance, distance_max)
            self.assertGreaterEqual(float(state[2]), start_z_range[0])
            self.assertLessEqual(float(state[2]), start_z_range[1])
            self.assertGreaterEqual(float(goal[2]), goal_z_range[0])
            self.assertLessEqual(float(goal[2]), goal_z_range[1])
            self.assertLessEqual(abs(float(goal[2] - state[2])), max_height_gap)
            self.assertLessEqual(max(abs(float(state[0])), abs(float(goal[0]))), cfg.world_xy)
            self.assertLessEqual(max(abs(float(state[1])), abs(float(goal[1]))), cfg.world_xy)


if __name__ == '__main__':
    unittest.main()
