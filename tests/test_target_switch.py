"""Tests for target-switch sampling and evaluation helpers."""

import unittest

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import TargetSwitchTrajectoryEnv
from brain_uav.scripts.run_target_switch import build_parser
from brain_uav.target_switch import (
    TARGET_SWITCH_LEVELS,
    sample_valid_new_goal,
    target_switch_config_for_level,
)


def _small_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        target_distance=30.0,
        world_xy=100.0,
        world_z_max=100.0,
        max_steps=3,
        no_fly_radius_range=(5.0, 8.0),
        no_fly_radius_curriculum={
            'easy': (5.0, 8.0),
            'easy_two_zone': (5.0, 8.0),
            'medium': (5.0, 8.0),
            'hard': (5.0, 8.0),
            'benchmark': (5.0, 8.0),
        },
        start_zone_clearance=2.0,
    )


class TestTargetSwitchHelpers(unittest.TestCase):
    def test_target_switch_configs_create_all_levels(self):
        for level in TARGET_SWITCH_LEVELS:
            with self.subTest(level=level):
                cfg = target_switch_config_for_level(level)
                self.assertEqual(cfg.level, level)
                self.assertEqual(cfg.base_curriculum_level, 'hard')
                self.assertFalse(hasattr(cfg, 'old_goal_inertia_penalty'))

    def test_sample_valid_new_goal_stays_in_world_and_outside_clearance(self):
        env = TargetSwitchTrajectoryEnv(
            scenario=_small_scenario(),
            rewards=RewardConfig(),
            target_switch=target_switch_config_for_level('target_switch_easy'),
            seed=5,
            fixed_scenarios=[
                {
                    'state': [0.0, 0.0, 20.0, 0.0, 0.0],
                    'goal': [80.0, 0.0, 20.0],
                    'zones': [],
                    'curriculum_level': 'hard',
                }
            ],
        )
        env.reset(seed=5)
        rng = np.random.default_rng(5)

        goal = sample_valid_new_goal(env, rng, env.target_switch)

        self.assertLessEqual(abs(float(goal[0])), env.scenario.world_xy)
        self.assertLessEqual(abs(float(goal[1])), env.scenario.world_xy)
        self.assertGreater(float(goal[2]), env.scenario.world_z_min)
        self.assertLess(float(goal[2]), env.scenario.world_z_max)
        for zone in env.zones:
            self.assertFalse(env._inside_zone_with_clearance(goal, zone, env.scenario.start_zone_clearance))

    def test_forward_sampling_falls_back_when_heading_points_out_of_world(self):
        env = TargetSwitchTrajectoryEnv(
            scenario=ScenarioConfig(),
            rewards=RewardConfig(),
            target_switch=target_switch_config_for_level('target_switch_hard'),
            seed=19,
        )
        env.state = np.array([1220.0, 0.0, 140.0, 0.0, 0.0], dtype=np.float32)
        env.goal = np.array([0.0, 300.0, 140.0], dtype=np.float32)
        env.zones = []
        rng = np.random.default_rng(19)

        goal = sample_valid_new_goal(env, rng, env.target_switch, switch_mode='forward')

        self.assertLessEqual(abs(float(goal[0])), env.scenario.world_xy)
        self.assertLessEqual(abs(float(goal[1])), env.scenario.world_xy)

    def test_run_target_switch_parser_accepts_switch_mode(self):
        args = build_parser().parse_args(['--switch-mode', 'lateral'])

        self.assertEqual(args.switch_mode, 'lateral')


if __name__ == '__main__':
    unittest.main()
