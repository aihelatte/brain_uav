"""Tests for target-switch APF / Heuristic baseline evaluation."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import TargetSwitchTrajectoryEnv
from brain_uav.scripts.run_target_switch_baselines import (
    _build_output_dir,
    build_baseline_policy,
    build_parser,
    evaluate_target_switch_baselines,
)
from brain_uav.scripts.train_target_switch_td3 import _best_window_score
from brain_uav.target_switch import TargetSwitchConfig


def _small_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        target_distance=30.0,
        world_xy=100.0,
        world_z_max=100.0,
        max_steps=4,
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


def _switch_config() -> TargetSwitchConfig:
    return TargetSwitchConfig(
        level='target_switch_easy',
        switch_step_ratio_range=(0.34, 0.34),
        switch_angle_deg_range=(0.0, 10.0),
        new_goal_distance_ratio_range=(0.50, 0.60),
        new_goal_z_ratio_range=(0.20, 0.25),
        max_height_gap_ratio=0.20,
        lateral_offset_ratio=0.02,
    )


def _fixed_payload() -> dict:
    return {
        'state': [0.0, 0.0, 20.0, 0.0, 0.0],
        'goal': [80.0, 0.0, 20.0],
        'zones': [],
        'curriculum_level': 'hard',
    }


class _DummyEnv:
    def __init__(self, episode_specs: list[dict[str, object]]) -> None:
        self.episode_specs = episode_specs
        self.episode_idx = -1
        self.step_idx = 0
        self.switch_step = 2
        self.switch_index = 1
        self.switched = False
        self.pre_switch_done = False
        self.state = np.array([1.0, 0.0, 5.0, 0.0, 0.0], dtype=np.float32)
        self.goal = np.array([10.0, 0.0, 5.0], dtype=np.float32)
        self.old_goal = np.array([8.0, 0.0, 5.0], dtype=np.float32)
        self.new_goal = np.array([10.0, 0.0, 5.0], dtype=np.float32)
        self.trajectory = [np.array([0.0, 0.0, 5.0], dtype=np.float32)]
        self.action_space = SimpleNamespace(shape=(2,))
        self.scenario = _small_scenario()
        self.zones = []
        self.last_episode_summary = None
        self.current_spec: dict[str, object] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed, options
        self.episode_idx += 1
        self.current_spec = self.episode_specs[min(self.episode_idx, len(self.episode_specs) - 1)]
        self.step_idx = 0
        self.state = np.array([0.0, 0.0, 5.0, 0.0, 0.0], dtype=np.float32)
        self.old_goal = np.array([8.0, 0.0, 5.0], dtype=np.float32)
        self.new_goal = np.array([10.0, 0.0, 5.0], dtype=np.float32)
        self.goal = self.old_goal.copy()
        self.switch_index = 1
        self.switched = False
        self.pre_switch_done = False
        self.trajectory = [self.state[:3].copy()]
        return np.zeros(24, dtype=np.float32), {'switch_step': self.switch_step}

    def step(self, action):
        del action
        self.step_idx += 1
        self.state = np.array([2.0, 0.0, 5.0, 0.0, 0.0], dtype=np.float32)
        self.trajectory.append(self.state[:3].copy())
        switched = bool(self.current_spec.get('switched', True))
        if self.step_idx == 1 and switched:
            self.goal = self.new_goal.copy()
            self.switched = True
            self.last_episode_summary = {
                'switch_step': self.switch_step,
                'switched': True,
                'post_switch_steps': 0,
                'pre_switch_done': bool(self.current_spec.get('pre_switch_done', False)),
                'boundary_reason': None,
                'old_goal': self.old_goal.tolist(),
                'new_goal': self.new_goal.tolist(),
                'switch_position': [1.0, 0.0, 5.0],
                'final_to_new_distance': 4.0,
                'min_to_new_distance': 4.0,
                'distance_reduction': 1.0,
                'switch_alignment_reward_mean': 0.5,
                'ceiling_penalty_mean': -0.1,
            }
            return np.zeros(24, dtype=np.float32), 0.5, False, False, {'outcome': 'running', 'goal_distance': 4.0}

        outcome = str(self.current_spec.get('outcome', 'goal'))
        info = {'outcome': outcome, 'goal_distance': 3.0}
        self.last_episode_summary = {
            'switch_step': self.switch_step,
            'switched': switched,
            'post_switch_steps': 1 if switched else 0,
            'pre_switch_done': bool(self.current_spec.get('pre_switch_done', False)),
            'boundary_reason': self.current_spec.get('boundary_reason'),
            'old_goal': self.old_goal.tolist(),
            'new_goal': self.new_goal.tolist() if switched else None,
            'switch_position': [1.0, 0.0, 5.0] if switched else None,
            'final_to_new_distance': float(self.current_spec.get('final_to_new_distance', 3.0)),
            'min_to_new_distance': float(self.current_spec.get('min_to_new_distance', 3.0)),
            'distance_reduction': float(self.current_spec.get('distance_reduction', 2.0)),
            'switch_alignment_reward_mean': float(self.current_spec.get('switch_alignment_reward_mean', 1.0)),
            'ceiling_penalty_mean': float(self.current_spec.get('ceiling_penalty_mean', -0.2)),
        }
        return np.zeros(24, dtype=np.float32), 1.0, True, False, info

    def export_scenario(self):
        return {
            'state': _fixed_payload()['state'],
            'goal': self.goal.tolist(),
            'zones': [],
            'curriculum_level': 'hard',
        }

    def _active_goal_radius(self) -> float:
        return 5.0


class _DummyPolicy:
    def __init__(self, env) -> None:
        self.env = env
        self.goals_seen: list[list[float]] = []
        self.fail = False

    def act(self, obs):
        del obs
        self.goals_seen.append(self.env.goal.tolist())
        if self.fail:
            raise RuntimeError('planner failure')
        return np.zeros(2, dtype=np.float32)


class TestTargetSwitchBaselines(unittest.TestCase):
    def test_parser_accepts_apf_and_heuristic(self):
        parser = build_parser()
        args_apf = parser.parse_args(['--policy', 'apf'])
        args_heuristic = parser.parse_args(['--policy', 'heuristic'])

        self.assertEqual(args_apf.policy, 'apf')
        self.assertEqual(args_heuristic.policy, 'heuristic')

    def test_output_dir_naming_uses_timestamp_policy_and_level(self):
        args = argparse.Namespace(
            output=Path('outputs/target_switch_baselines'),
            policy='apf',
            target_switch_level='target_switch_hard',
        )
        with mock.patch('brain_uav.scripts.run_target_switch_baselines.now_timestamp', return_value='20260509_153000'):
            output_dir = _build_output_dir(args)

        self.assertIn('20260509_153000_apf_target_switch_hard', str(output_dir).replace('\\', '/'))

    def test_best_window_score_matches_td3_logic(self):
        better = {
            'switch_success_rate': 0.80,
            'switch_success_count': 40,
            'ground_count': 1,
            'boundary_count': 0,
            'collision_count': 0,
            'timeout_count': 0,
            'pre_switch_done_count': 0,
            'avg_final_to_new_distance': 50.0,
            'avg_return': 10.0,
        }
        worse = dict(better)
        worse['ground_count'] = 2

        self.assertGreater(_best_window_score(better), _best_window_score(worse))

    def test_apf_and_heuristic_can_step_after_goal_switch(self):
        for policy_name in ('apf', 'heuristic'):
            with self.subTest(policy=policy_name):
                env = TargetSwitchTrajectoryEnv(
                    scenario=_small_scenario(),
                    rewards=RewardConfig(),
                    target_switch=_switch_config(),
                    seed=5,
                    fixed_scenarios=[_fixed_payload()],
                )
                obs, _ = env.reset(seed=5)
                policy = build_baseline_policy(policy_name, env)

                policy.act(obs)
                env.step(np.zeros(2, dtype=np.float32))
                self.assertTrue(env.switched)
                self.assertIsNotNone(env.new_goal)
                action_after_switch = policy.act(env._get_obs())

                self.assertEqual(action_after_switch.shape, (2,))
                np.testing.assert_allclose(env.goal, env.new_goal)

    def test_evaluation_uses_new_goal_and_writes_no_checkpoint(self):
        args = argparse.Namespace(
            policy='heuristic',
            target_switch_level='target_switch_easy',
            episodes=2,
            seed=7,
            output=Path(tempfile.mkdtemp()),
            summary_every_episodes=1,
            allow_fallback=False,
            save_episode_json=False,
            save_failures_only=True,
            max_saved_episodes=100,
            save_snapshots=False,
            snapshot_every_window=5,
        )
        dummy_env = _DummyEnv(
            [
                {'outcome': 'goal', 'switched': True, 'pre_switch_done': False},
                {'outcome': 'collision', 'switched': True, 'pre_switch_done': False},
            ]
        )
        dummy_policy = _DummyPolicy(dummy_env)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / '20260509_153000_heuristic_target_switch_easy'
            with mock.patch('brain_uav.scripts.run_target_switch_baselines.TargetSwitchTrajectoryEnv', return_value=dummy_env):
                with mock.patch('brain_uav.scripts.run_target_switch_baselines.build_baseline_policy', return_value=dummy_policy):
                    summary = evaluate_target_switch_baselines(args, output_dir=output_dir)

            self.assertEqual(summary['episodes'], 2)
            self.assertTrue((output_dir / 'summary.json').is_file())
            self.assertTrue((output_dir / 'records.json').is_file())
            self.assertTrue((output_dir / 'best_window.json').is_file())
            self.assertEqual(list(output_dir.glob('*.pt')), [])
            self.assertEqual(summary['raw_goal_count'], 1)
            self.assertEqual(summary['switch_success_count'], 1)
            self.assertEqual(summary['records_path'], str(output_dir / 'records.json'))
            self.assertTrue(all(record['active_goal_matches_new_goal'] for record in json_load(output_dir / 'records.json')))
            self.assertIn(dummy_env.old_goal.tolist(), dummy_policy.goals_seen)
            self.assertIn(dummy_env.new_goal.tolist(), dummy_policy.goals_seen)

    def test_pre_switch_done_goal_counts_as_raw_goal_but_not_switch_success(self):
        args = argparse.Namespace(
            policy='heuristic',
            target_switch_level='target_switch_easy',
            episodes=2,
            seed=7,
            output=Path(tempfile.mkdtemp()),
            summary_every_episodes=1,
            allow_fallback=False,
            save_episode_json=False,
            save_failures_only=True,
            max_saved_episodes=100,
            save_snapshots=False,
            snapshot_every_window=5,
        )
        dummy_env = _DummyEnv(
            [
                {'outcome': 'goal', 'switched': False, 'pre_switch_done': True, 'final_to_new_distance': 30.0},
                {'outcome': 'goal', 'switched': True, 'pre_switch_done': False, 'final_to_new_distance': 3.0},
            ]
        )
        dummy_policy = _DummyPolicy(dummy_env)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'run'
            with mock.patch('brain_uav.scripts.run_target_switch_baselines.TargetSwitchTrajectoryEnv', return_value=dummy_env):
                with mock.patch('brain_uav.scripts.run_target_switch_baselines.build_baseline_policy', return_value=dummy_policy):
                    summary = evaluate_target_switch_baselines(args, output_dir=output_dir)

            records = json_load(output_dir / 'records.json')
            self.assertTrue(records[0]['raw_goal'])
            self.assertFalse(records[0]['switch_success'])
            self.assertTrue(records[1]['raw_goal'])
            self.assertTrue(records[1]['switch_success'])
            self.assertEqual(summary['success_count'], 1)
            self.assertEqual(summary['success_rate'], 0.5)
            self.assertEqual(summary['raw_goal_count'], 2)
            self.assertEqual(summary['raw_goal_rate'], 1.0)
            self.assertEqual(summary['switch_success_count'], 1)
            self.assertEqual(summary['switch_success_rate'], 0.5)
            self.assertEqual(summary['best_window']['switch_success_rate'], 1.0)
            self.assertEqual(summary['best_window']['raw_goal_rate'], 1.0)

    def test_best_window_prefers_switch_success_not_raw_goal(self):
        args = argparse.Namespace(
            policy='heuristic',
            target_switch_level='target_switch_easy',
            episodes=2,
            seed=7,
            output=Path(tempfile.mkdtemp()),
            summary_every_episodes=1,
            allow_fallback=False,
            save_episode_json=False,
            save_failures_only=True,
            max_saved_episodes=100,
            save_snapshots=False,
            snapshot_every_window=5,
        )
        dummy_env = _DummyEnv(
            [
                {'outcome': 'goal', 'switched': False, 'pre_switch_done': True, 'final_to_new_distance': 2.0},
                {'outcome': 'collision', 'switched': True, 'pre_switch_done': False, 'final_to_new_distance': 4.0},
            ]
        )
        dummy_policy = _DummyPolicy(dummy_env)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'run'
            with mock.patch('brain_uav.scripts.run_target_switch_baselines.TargetSwitchTrajectoryEnv', return_value=dummy_env):
                with mock.patch('brain_uav.scripts.run_target_switch_baselines.build_baseline_policy', return_value=dummy_policy):
                    summary = evaluate_target_switch_baselines(args, output_dir=output_dir)

            self.assertEqual(summary['best_window']['episode_start'], 1)
            self.assertEqual(summary['best_window']['switch_success_count'], 0)

    def test_fallback_disabled_raises(self):
        args = argparse.Namespace(
            policy='heuristic',
            target_switch_level='target_switch_easy',
            episodes=1,
            seed=7,
            output=Path(tempfile.mkdtemp()),
            summary_every_episodes=1,
            allow_fallback=False,
            save_episode_json=False,
            save_failures_only=True,
            max_saved_episodes=100,
            save_snapshots=False,
            snapshot_every_window=5,
        )
        dummy_env = _DummyEnv([{'outcome': 'goal', 'switched': True, 'pre_switch_done': False}])
        dummy_policy = _DummyPolicy(dummy_env)
        dummy_policy.fail = True
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'run'
            with mock.patch('brain_uav.scripts.run_target_switch_baselines.TargetSwitchTrajectoryEnv', return_value=dummy_env):
                with mock.patch('brain_uav.scripts.run_target_switch_baselines.build_baseline_policy', return_value=dummy_policy):
                    with self.assertRaises(RuntimeError):
                        evaluate_target_switch_baselines(args, output_dir=output_dir)

    def test_allow_fallback_records_errors(self):
        args = argparse.Namespace(
            policy='heuristic',
            target_switch_level='target_switch_easy',
            episodes=1,
            seed=7,
            output=Path(tempfile.mkdtemp()),
            summary_every_episodes=1,
            allow_fallback=True,
            save_episode_json=False,
            save_failures_only=True,
            max_saved_episodes=100,
            save_snapshots=False,
            snapshot_every_window=5,
        )
        dummy_env = _DummyEnv([{'outcome': 'goal', 'switched': True, 'pre_switch_done': False}])
        dummy_policy = _DummyPolicy(dummy_env)
        dummy_policy.fail = True
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'run'
            with mock.patch('brain_uav.scripts.run_target_switch_baselines.TargetSwitchTrajectoryEnv', return_value=dummy_env):
                with mock.patch('brain_uav.scripts.run_target_switch_baselines.build_baseline_policy', return_value=dummy_policy):
                    summary = evaluate_target_switch_baselines(args, output_dir=output_dir)

            self.assertGreater(summary['fallback_count'], 0)
            self.assertGreater(summary['fallback_error_count'], 0)
            self.assertTrue((output_dir / 'fallback_errors.json').is_file())

    def test_save_episode_json_persists_failure_records(self):
        args = argparse.Namespace(
            policy='heuristic',
            target_switch_level='target_switch_easy',
            episodes=2,
            seed=7,
            output=Path(tempfile.mkdtemp()),
            summary_every_episodes=1,
            allow_fallback=False,
            save_episode_json=True,
            save_failures_only=True,
            max_saved_episodes=100,
            save_snapshots=False,
            snapshot_every_window=5,
        )
        dummy_env = _DummyEnv(
            [
                {'outcome': 'goal', 'switched': True, 'pre_switch_done': False},
                {'outcome': 'collision', 'switched': True, 'pre_switch_done': False},
            ]
        )
        dummy_policy = _DummyPolicy(dummy_env)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'run'
            with mock.patch('brain_uav.scripts.run_target_switch_baselines.TargetSwitchTrajectoryEnv', return_value=dummy_env):
                with mock.patch('brain_uav.scripts.run_target_switch_baselines.build_baseline_policy', return_value=dummy_policy):
                    summary = evaluate_target_switch_baselines(args, output_dir=output_dir)

            records = json_load(output_dir / 'records.json')
            saved_paths = [record['episode_json_path'] for record in records if record['episode_json_path'] is not None]
            self.assertEqual(len(saved_paths), 1)
            self.assertIn('collision', Path(saved_paths[0]).name)
            self.assertEqual(summary['saved_episode_json_count'], 1)
            self.assertEqual(summary['episodes_dir'], str(output_dir / 'episodes'))

def json_load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
