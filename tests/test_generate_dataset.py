"""Tests for dataset generation helpers."""

import unittest

from brain_uav.scripts.generate_dataset import build_planners


class TestGenerateDataset(unittest.TestCase):
    def test_build_planners_excludes_astar(self):
        env = object()
        planners = build_planners(env)
        planner_names = [planner.__class__.__name__ for planner in planners]
        self.assertEqual(planner_names, ['HeuristicPlanner', 'ArtificialPotentialFieldPlanner'])


if __name__ == '__main__':
    unittest.main()
