"""Tests for the standalone baseline planner rollout script."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from brain_uav.baselines import ArtificialPotentialFieldPlanner, HeuristicPlanner
from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import StaticNoFlyTrajectoryEnv
from brain_uav.envs.static_no_fly_env_runtime import Zone
from brain_uav.scripts.test_baseline_planners import (
    build_parser,
    build_planner,
    episode_min_zone_clearance,
    rollout_planner,
    run_baseline_test,
)


class _DummyPlanner:
    def act(self, obs):
        del obs
        return np.zeros(2, dtype=np.float32)


class _DummyEnv:
    def __init__(self, outcomes: list[str], episode_length: int = 3) -> None:
        self.outcomes = outcomes
        self.episode_length = episode_length
        self.episode_idx = -1
        self.step_idx = 0
        self.current_outcome = 'goal'
        self.state = np.zeros(5, dtype=np.float32)
        self.goal = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        self.trajectory = [np.zeros(3, dtype=np.float32)]
        self.zones = [Zone(center_xy=np.array([0.0, 0.0], dtype=np.float32), radius=2.0)]
        self.scenario = ScenarioConfig()
        self.last_curriculum_level = 'hard'
        self.action_space = type('DummySpace', (), {'shape': (2,)})()

    def reset(self, seed: int | None = None, options: dict | None = None):
        del seed, options
        self.episode_idx += 1
        self.step_idx = 0
        self.current_outcome = self.outcomes[min(self.episode_idx, len(self.outcomes) - 1)]
        self.state = np.zeros(5, dtype=np.float32)
        self.trajectory = [np.zeros(3, dtype=np.float32)]
        return np.zeros(24, dtype=np.float32), {}

    def step(self, action):
        del action
        self.step_idx += 1
        self.state = np.array([float(self.step_idx), 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self.trajectory.append(self.state[:3].copy())
        done = self.step_idx >= self.episode_length
        outcome = self.current_outcome if done else 'running'
        info = {
            'goal_distance': 10.0 - self.step_idx,
            'segment_goal_distance': 10.0 - self.step_idx,
            'goal_reached_by_segment': bool(done and outcome == 'goal'),
            'progress': 1.0,
            'steps': self.step_idx,
            'curriculum_level': 'hard',
            'active_goal_radius': 5.0,
            'outcome': outcome,
        }
        return np.zeros(24, dtype=np.float32), 1.0, done, False, info

    @property
    def steps(self) -> int:
        return self.step_idx

    def export_scenario(self):
        return {
            'state': [0.0, 0.0, 0.0, 0.0, 0.0],
            'goal': self.goal.tolist(),
            'zones': [{'center_xy': [0.0, 0.0], 'radius': 2.0}],
            'curriculum_level': 'hard',
        }


class TestBaselinePlannerScript(unittest.TestCase):
    def test_parser_accepts_planner_and_save_artifacts(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                '--planner',
                'heuristic',
                '--evaluation-mode',
                'curriculum',
                '--curriculum-level',
                'hard',
                '--max-total-steps',
                '123',
                '--save-artifacts',
            ]
        )

        self.assertEqual(args.planner, 'heuristic')
        self.assertEqual(args.max_total_steps, 123)
        self.assertTrue(args.save_artifacts)

    def test_build_planner_returns_heuristic(self):
        env = StaticNoFlyTrajectoryEnv(ScenarioConfig(), RewardConfig(), seed=7)
        env.reset(seed=7)

        planner = build_planner('heuristic', env)

        self.assertIsInstance(planner, HeuristicPlanner)

    def test_build_planner_returns_apf(self):
        env = StaticNoFlyTrajectoryEnv(ScenarioConfig(), RewardConfig(), seed=7)
        env.reset(seed=7)

        planner = build_planner('apf', env)

        self.assertIsInstance(planner, ArtificialPotentialFieldPlanner)

    def test_episode_min_zone_clearance_matches_expected_geometry(self):
        env = _DummyEnv(['goal'])
        env.trajectory = [
            np.array([0.0, 0.0, 3.0], dtype=np.float32),
            np.array([3.0, 0.0, 4.0], dtype=np.float32),
        ]
        env.zones = [Zone(center_xy=np.array([0.0, 0.0], dtype=np.float32), radius=2.0)]

        clearance = episode_min_zone_clearance(env)

        self.assertAlmostEqual(clearance, 1.0)

    def test_rollout_collects_summary_friendly_records(self):
        env = _DummyEnv(['goal', 'collision'], episode_length=3)
        records, total_steps = rollout_planner(
            env,
            _DummyPlanner(),
            seed=7,
            episodes=2,
            max_total_steps=99,
            evaluation_mode='curriculum',
            config_payload={'scenario': {'goal_radius': 5.0}},
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(total_steps, 6)
        self.assertEqual(records[0]['outcome'], 'goal')
        self.assertEqual(records[1]['outcome'], 'collision')

    def test_max_total_steps_does_not_cut_episode_midway(self):
        env = _DummyEnv(['goal', 'goal'], episode_length=3)
        records, total_steps = rollout_planner(
            env,
            _DummyPlanner(),
            seed=7,
            episodes=None,
            max_total_steps=5,
            evaluation_mode='curriculum',
            config_payload={'scenario': {'goal_radius': 5.0}},
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(total_steps, 6)

    def test_save_artifacts_calls_export_helper(self):
        env = _DummyEnv(['goal'], episode_length=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)
            with mock.patch(
                'brain_uav.scripts.test_baseline_planners.export_episode_result',
                return_value={'json': 'a.json', 'png': 'a.png'},
            ) as export_mock:
                records, _total_steps = rollout_planner(
                    env,
                    _DummyPlanner(),
                    seed=7,
                    episodes=1,
                    max_total_steps=10,
                    evaluation_mode='curriculum',
                    config_payload={'scenario': {'goal_radius': 5.0}},
                    artifacts_dir=artifacts_dir,
                    save_artifacts=True,
                )

        export_mock.assert_called_once()
        self.assertEqual(records[0]['artifacts']['json'], 'a.json')

    def test_run_baseline_test_stops_and_summarizes(self):
        args = argparse.Namespace(
            planner='heuristic',
            evaluation_mode='curriculum',
            curriculum_level='hard',
            curriculum_mix=None,
            benchmark_suite=Path('unused.json'),
            max_total_steps=5,
            episodes=None,
            seed=7,
            output_root=Path(tempfile.mkdtemp()),
            save_artifacts=False,
        )
        env = _DummyEnv(['goal', 'boundary'], episode_length=3)

        with mock.patch('brain_uav.scripts.test_baseline_planners.make_env', return_value=env):
            with mock.patch('brain_uav.scripts.test_baseline_planners.build_planner', return_value=_DummyPlanner()):
                with mock.patch('brain_uav.scripts.test_baseline_planners.now_timestamp', return_value='20260506_123456'):
                    summary = run_baseline_test(args)

        self.assertEqual(summary['episodes'], 2)
        self.assertEqual(summary['total_steps'], 6)
        self.assertEqual(summary['success_count'], 1)
        self.assertAlmostEqual(summary['success_rate'], 0.5)
        self.assertTrue(Path(summary['episodes_path']).is_file())
        self.assertIn('20260506_123456', Path(summary['output_dir']).name)

    def test_benchmark_mode_uses_custom_suite_without_default_scenario_suite(self):
        args = argparse.Namespace(
            planner='apf',
            evaluation_mode='benchmark',
            curriculum_level='hard',
            curriculum_mix=None,
            benchmark_suite=Path('custom_suite.json'),
            max_total_steps=50,
            episodes=None,
            seed=7,
            output_root=Path(tempfile.mkdtemp()),
            save_artifacts=False,
        )
        env = _DummyEnv(['goal'], episode_length=2)
        fake_scenario = type(
            'FakeNamedScenario',
            (),
            {'scenario': {'state': [0, 0, 0, 0, 0], 'goal': [1, 0, 0], 'zones': [], 'curriculum_level': 'benchmark'}},
        )()

        with mock.patch('brain_uav.scripts.test_baseline_planners.load_benchmark_suite', return_value={'total_scenarios': 1}):
            with mock.patch('brain_uav.scripts.test_baseline_planners.build_benchmark_scenarios', return_value=[fake_scenario]):
                with mock.patch('brain_uav.scripts.test_baseline_planners.make_env', return_value=env) as make_env_mock:
                    with mock.patch('brain_uav.scripts.test_baseline_planners.build_planner', return_value=_DummyPlanner()):
                        with mock.patch('brain_uav.scripts.test_baseline_planners.now_timestamp', return_value='20260506_123456'):
                            summary = run_baseline_test(args)

        self.assertEqual(summary['episodes'], 1)
        self.assertEqual(make_env_mock.call_args.kwargs.get('scenario_suite'), None)


if __name__ == '__main__':
    unittest.main()
