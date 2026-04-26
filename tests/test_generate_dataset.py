"""Tests for dataset generation helpers."""

import unittest
from pathlib import Path

from brain_uav.scripts.generate_dataset import build_dataset_log_prefix, build_planners


class TestGenerateDataset(unittest.TestCase):
    def test_build_planners_excludes_astar(self):
        env = object()
        planners = build_planners(env)
        planner_names = [planner.__class__.__name__ for planner in planners]
        self.assertEqual(planner_names, ['HeuristicPlanner', 'ArtificialPotentialFieldPlanner'])

    def test_build_dataset_log_prefix_uses_data_and_level(self):
        self.assertEqual(build_dataset_log_prefix('easy'), '[DATA easy]')

    def test_zone_candidate_safe_margin_buffer_is_one_km(self):
        source = Path(
            'E:/wurenji/my_project/src/brain_uav/envs/static_no_fly_env_runtime.py'
        ).read_text(encoding='utf-8')
        self.assertIn('safe_margin = radius + cfg.warning_distance + cfg.goal_radius + 1.0', source)
        self.assertNotIn('safe_margin = radius + cfg.warning_distance + cfg.goal_radius + 10.0', source)


if __name__ == '__main__':
    unittest.main()
