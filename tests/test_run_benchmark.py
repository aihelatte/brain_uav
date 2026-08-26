"""Tests for the unified benchmark runner."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from brain_uav.config import ExperimentConfig
from brain_uav.scripts.run_benchmark import build_parser, run_benchmark, validate_args


class _DummyPlanner:
    def act(self, obs):
        del obs
        return np.zeros(2, dtype=np.float32)


class _DummyActor(torch.nn.Module):
    def __init__(self, obs_dim: int = 24, action_dim: int = 2) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(obs_dim, action_dim)

    def forward(self, obs):
        return torch.zeros((obs.shape[0], self.linear.out_features), dtype=torch.float32, device=obs.device)


class _DummyCudaGraph:
    def replay(self) -> None:
        return None


class _DummyEnv:
    def __init__(self, outcomes: list[str], episode_length: int = 3) -> None:
        self.outcomes = outcomes
        self.episode_length = episode_length
        self.episode_idx = -1
        self.step_idx = 0
        self.current_outcome = 'goal'
        self.state = np.zeros(5, dtype=np.float32)
        self.goal = np.array([10.0, 0.0, 5.0], dtype=np.float32)
        self.trajectory = [np.array([0.0, 0.0, 5.0], dtype=np.float32)]
        self.action_space = SimpleNamespace(shape=(2,))
        self.observation_space = SimpleNamespace(shape=(24,))
        self.current_scenario = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed
        self.episode_idx += 1
        self.step_idx = 0
        self.current_outcome = self.outcomes[min(self.episode_idx, len(self.outcomes) - 1)]
        self.state = np.array([0.0, 0.0, 5.0, 0.0, 0.0], dtype=np.float32)
        if options and 'scenario' in options:
            self.current_scenario = options['scenario']
            self.goal = np.asarray(self.current_scenario['goal'], dtype=np.float32)
        self.trajectory = [self.state[:3].copy()]
        return np.zeros(24, dtype=np.float32), {}

    def step(self, action):
        del action
        self.step_idx += 1
        self.state = np.array([float(self.step_idx), 0.0, 5.0, 0.0, 0.0], dtype=np.float32)
        self.trajectory.append(self.state[:3].copy())
        done = self.step_idx >= self.episode_length
        outcome = self.current_outcome if done else 'running'
        info = {
            'goal_distance': float(max(0.0, 10.0 - self.step_idx)),
            'segment_goal_distance': float(max(0.0, 10.0 - self.step_idx)),
            'goal_reached_by_segment': bool(done and outcome == 'goal'),
            'progress': 1.0,
            'steps': self.step_idx,
            'curriculum_level': 'benchmark',
            'active_goal_radius': 5.0,
            'outcome': outcome,
        }
        return np.zeros(24, dtype=np.float32), 1.0, done, False, info

    @property
    def steps(self) -> int:
        return self.step_idx

    def export_scenario(self):
        if self.current_scenario is None:
            raise RuntimeError('Scenario must be set before export.')
        return {
            'state': self.current_scenario['state'],
            'goal': self.current_scenario['goal'],
            'zones': self.current_scenario['zones'],
            'curriculum_level': self.current_scenario.get('curriculum_level', 'benchmark'),
        }


def _named_scenario(idx: int, category: str = 'single_detour'):
    return SimpleNamespace(
        scenario_id=f'S{idx:03d}',
        category=category,
        name=f'{category}_{idx:03d}',
        description='dummy',
        corridor_width=200.0 + idx,
        min_clearance_to_boundary=40.0 + idx,
        difficulty_score=2.5 + idx,
        scenario={
            'state': [0.0, 0.0, 5.0, 0.0, 0.0],
            'goal': [10.0, 0.0, 5.0],
            'zones': [{'center_xy': [2.0, 0.0], 'radius': 1.5}],
            'curriculum_level': 'benchmark',
            'scenario_id': f'S{idx:03d}',
            'category': category,
            'scenario_label': f'{category}_{idx:03d}',
        },
    )


class TestRunBenchmark(unittest.TestCase):
    def test_parser_requires_checkpoint_for_ann(self):
        parser = build_parser()
        args = parser.parse_args(['--method', 'ann'])
        with self.assertRaises(SystemExit):
            validate_args(parser, args)

    def test_parser_disallows_checkpoint_for_planner(self):
        parser = build_parser()
        args = parser.parse_args(['--method', 'apf', '--checkpoint', 'dummy.pt'])
        with self.assertRaises(SystemExit):
            validate_args(parser, args)

    def test_parser_defaults_reuse_obs_tensor_to_false(self):
        parser = build_parser()
        args = parser.parse_args(['--method', 'ann', '--checkpoint', 'model.pt'])
        self.assertFalse(args.reuse_obs_tensor)
        self.assertEqual(args.actor_execution, 'eager')

    def test_parser_accepts_cuda_graph_actor_execution(self):
        parser = build_parser()
        args = parser.parse_args(
            ['--method', 'snn', '--checkpoint', 'model.pt', '--actor-execution', 'cuda-graph']
        )
        self.assertEqual(args.actor_execution, 'cuda-graph')

    def test_benchmark_uses_full_suite_when_episodes_is_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                method='heuristic',
                checkpoint=None,
                benchmark_suite=Path('suite.json'),
                episodes=None,
                seed=7,
                output_root=Path(tmpdir),
                run_name='heuristic_run',
                device='cpu',
                snn_backend='torch',
                episode_artifacts='json',
            )
            env = _DummyEnv(['goal', 'collision'], episode_length=2)
            scenarios = [_named_scenario(1), _named_scenario(2, category='double_channel')]
            with mock.patch('brain_uav.scripts.run_benchmark.load_benchmark_suite', return_value={'suite_name': 'suite', 'seed': 1, 'count_per_category': 1, 'total_scenarios': 2, 'categories': ['single_detour', 'double_channel']}):
                with mock.patch('brain_uav.scripts.run_benchmark.build_benchmark_scenarios', return_value=scenarios):
                    with mock.patch('brain_uav.scripts.run_benchmark.make_env', return_value=env):
                        with mock.patch('brain_uav.scripts.run_benchmark.build_planner', return_value=_DummyPlanner()):
                            summary = run_benchmark(args)
            self.assertEqual(summary['episodes'], 2)
            self.assertEqual(summary['benchmark_total_scenarios'], 2)
            self.assertTrue(Path(summary['efficiency_summary_path']).is_file())

    def test_episodes_cannot_exceed_suite_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                method='heuristic',
                checkpoint=None,
                benchmark_suite=Path('suite.json'),
                episodes=3,
                seed=7,
                output_root=Path(tmpdir),
                run_name='heuristic_run',
                device='cpu',
                snn_backend='torch',
                episode_artifacts='json',
            )
            with mock.patch('brain_uav.scripts.run_benchmark.load_benchmark_suite', return_value={'suite_name': 'suite', 'seed': 1, 'count_per_category': 1, 'total_scenarios': 2, 'categories': ['single_detour']}):
                with mock.patch('brain_uav.scripts.run_benchmark.build_benchmark_scenarios', return_value=[_named_scenario(1), _named_scenario(2)]):
                    with self.assertRaises(ValueError):
                        run_benchmark(args)

    def test_episode_artifacts_json_writes_episode_json_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                method='heuristic',
                checkpoint=None,
                benchmark_suite=Path('suite.json'),
                episodes=1,
                seed=7,
                output_root=Path(tmpdir),
                run_name='heuristic_run',
                device='cpu',
                snn_backend='torch',
                episode_artifacts='json',
            )
            env = _DummyEnv(['goal'], episode_length=2)
            scenarios = [_named_scenario(1)]
            with mock.patch('brain_uav.scripts.run_benchmark.load_benchmark_suite', return_value={'suite_name': 'suite', 'seed': 1, 'count_per_category': 1, 'total_scenarios': 1, 'categories': ['single_detour']}):
                with mock.patch('brain_uav.scripts.run_benchmark.build_benchmark_scenarios', return_value=scenarios):
                    with mock.patch('brain_uav.scripts.run_benchmark.make_env', return_value=env):
                        with mock.patch('brain_uav.scripts.run_benchmark.build_planner', return_value=_DummyPlanner()):
                            summary = run_benchmark(args)
            episode_jsons = list((Path(summary['episodes_dir'])).glob('ep*.json'))
            pngs = list((Path(summary['episodes_dir'])).glob('*.png'))
            self.assertEqual(len(episode_jsons), 1)
            self.assertEqual(pngs, [])

    def test_episode_artifacts_none_skips_per_episode_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                method='heuristic',
                checkpoint=None,
                benchmark_suite=Path('suite.json'),
                episodes=1,
                seed=7,
                output_root=Path(tmpdir),
                run_name='heuristic_run',
                device='cpu',
                snn_backend='torch',
                episode_artifacts='none',
            )
            env = _DummyEnv(['goal'], episode_length=2)
            scenarios = [_named_scenario(1)]
            with mock.patch('brain_uav.scripts.run_benchmark.load_benchmark_suite', return_value={'suite_name': 'suite', 'seed': 1, 'count_per_category': 1, 'total_scenarios': 1, 'categories': ['single_detour']}):
                with mock.patch('brain_uav.scripts.run_benchmark.build_benchmark_scenarios', return_value=scenarios):
                    with mock.patch('brain_uav.scripts.run_benchmark.make_env', return_value=env):
                        with mock.patch('brain_uav.scripts.run_benchmark.build_planner', return_value=_DummyPlanner()):
                            summary = run_benchmark(args)
            episode_jsons = list((Path(summary['episodes_dir'])).glob('ep*.json'))
            self.assertEqual(episode_jsons, [])
            self.assertTrue((Path(summary['episodes_dir']) / 'index.json').is_file())

    def test_planner_benchmark_writes_efficiency_with_none_energy_and_cpu_device(self):
        for device in ('auto', 'cuda'):
            with self.subTest(device=device):
                with tempfile.TemporaryDirectory() as tmpdir:
                    args = argparse.Namespace(
                        method='apf',
                        checkpoint=None,
                        benchmark_suite=Path('suite.json'),
                        episodes=1,
                        seed=7,
                        output_root=Path(tmpdir),
                        run_name='apf_run',
                        device=device,
                        snn_backend='torch',
                        episode_artifacts='json',
                        actor_execution='cuda-graph',
                    )
                    env = _DummyEnv(['collision'], episode_length=2)
                    scenarios = [_named_scenario(1)]
                    with mock.patch('brain_uav.scripts.run_benchmark.load_benchmark_suite', return_value={'suite_name': 'suite', 'seed': 1, 'count_per_category': 1, 'total_scenarios': 1, 'categories': ['single_detour']}):
                        with mock.patch('brain_uav.scripts.run_benchmark.build_benchmark_scenarios', return_value=scenarios):
                            with mock.patch('brain_uav.scripts.run_benchmark.make_env', return_value=env):
                                with mock.patch('brain_uav.scripts.run_benchmark.build_planner', return_value=_DummyPlanner()):
                                    summary = run_benchmark(args)
                    efficiency = json.loads(Path(summary['efficiency_summary_path']).read_text(encoding='utf-8'))
                    self.assertIn('physics', summary)
                    self.assertIn('category_summary', summary)
                    self.assertIn('zone_count_summary', summary)
                    self.assertEqual(efficiency['planner_energy_pj'], None)
                    self.assertEqual(efficiency['device'], 'cpu')
                    self.assertNotIn('actor_execution', efficiency)
                    self.assertNotIn('actor_execution_requested', efficiency)
                    self.assertNotIn('actor_execution_effective', efficiency)
                    self.assertNotIn('cuda_graph_fallback_occurred', efficiency)
                    self.assertNotIn('cuda_graph_fallback_reason', efficiency)
                    self.assertGreaterEqual(summary['records'][0]['avg_decision_time_ms'], 0.0)

    def test_checkpoint_model_config_keeps_benchmark_physics_and_uses_model_fields(self):
        defaults = ExperimentConfig()
        for method in ('ann', 'snn'):
            with self.subTest(method=method):
                with tempfile.TemporaryDirectory() as tmpdir:
                    args = argparse.Namespace(
                        method=method,
                        checkpoint=Path('model.pt'),
                        benchmark_suite=Path('suite.json'),
                        episodes=1,
                        seed=7,
                        output_root=Path(tmpdir),
                        run_name=f'{method}_physics_run',
                        device='cpu',
                        snn_backend='torch',
                        episode_artifacts='json',
                    )
                    env = _DummyEnv(['goal'], episode_length=2)
                    scenarios = [_named_scenario(1)]
                    dummy_actor = _DummyActor()
                    checkpoint_payload = {
                        'state_dict': dummy_actor.state_dict(),
                        'config': {
                            'scenario': {
                                'max_steps': 777,
                                'world_xy': 9999.0,
                                'goal_radius': 123.0,
                            },
                            'rewards': {
                                'goal_reward': 42.0,
                            },
                            'training': {
                                'hidden_dim': 321,
                                'snn_time_window': 9,
                                'device': 'cuda',
                                'snn_backend': 'cupy',
                            },
                        },
                    }
                    captured_training: dict[str, object] = {}

                    def _make_actor_side_effect(cfg, model_type, state_dim, action_dim):
                        captured_training['model_type'] = model_type
                        captured_training['hidden_dim'] = cfg.training.hidden_dim
                        captured_training['snn_time_window'] = cfg.training.snn_time_window
                        captured_training['device'] = cfg.training.device
                        captured_training['snn_backend'] = cfg.training.snn_backend
                        return _DummyActor(obs_dim=state_dim, action_dim=action_dim)

                    with mock.patch('brain_uav.scripts.run_benchmark.load_benchmark_suite', return_value={'suite_name': 'suite', 'seed': 1, 'count_per_category': 1, 'total_scenarios': 1, 'categories': ['single_detour']}):
                        with mock.patch('brain_uav.scripts.run_benchmark.build_benchmark_scenarios', return_value=scenarios):
                            with mock.patch('brain_uav.scripts.run_benchmark.make_env', return_value=env):
                                with mock.patch('brain_uav.scripts.run_benchmark.make_actor', side_effect=_make_actor_side_effect):
                                    with mock.patch('brain_uav.scripts.run_benchmark.load_checkpoint', return_value=checkpoint_payload):
                                        summary = run_benchmark(args)

                    physics = summary['physics']
                    self.assertEqual(physics['max_steps_per_episode'], defaults.scenario.max_steps)
                    self.assertEqual(physics['world_xy'], float(defaults.scenario.world_xy))
                    self.assertEqual(physics['goal_radius'], defaults.scenario.goal_radius)
                    self.assertNotEqual(physics['max_steps_per_episode'], 777)
                    self.assertNotEqual(physics['world_xy'], 9999.0)
                    self.assertNotEqual(physics['goal_radius'], 123.0)
                    self.assertEqual(captured_training['hidden_dim'], 321)
                    self.assertEqual(captured_training['snn_time_window'], 9)
                    self.assertEqual(captured_training['device'], 'cpu')
                    if method == 'snn':
                        self.assertEqual(captured_training['snn_backend'], 'torch')
                    else:
                        self.assertEqual(summary['method'], 'ann')

    def test_ann_and_snn_benchmark_can_run_with_mock_actor(self):
        for method in ('ann', 'snn'):
            with self.subTest(method=method):
                with tempfile.TemporaryDirectory() as tmpdir:
                    args = argparse.Namespace(
                        method=method,
                        checkpoint=Path('model.pt'),
                        benchmark_suite=Path('suite.json'),
                        episodes=1,
                        seed=7,
                        output_root=Path(tmpdir),
                        run_name=f'{method}_run',
                        device='cpu',
                        snn_backend='torch',
                        episode_artifacts='json',
                        reuse_obs_tensor=False,
                        actor_execution='eager',
                    )
                    env = _DummyEnv(['goal'], episode_length=2)
                    scenarios = [_named_scenario(1)]
                    dummy_actor = _DummyActor()
                    checkpoint_payload = {'state_dict': dummy_actor.state_dict(), 'config': {}}
                    with mock.patch('brain_uav.scripts.run_benchmark.load_benchmark_suite', return_value={'suite_name': 'suite', 'seed': 1, 'count_per_category': 1, 'total_scenarios': 1, 'categories': ['single_detour']}):
                        with mock.patch('brain_uav.scripts.run_benchmark.build_benchmark_scenarios', return_value=scenarios):
                            with mock.patch('brain_uav.scripts.run_benchmark.make_env', return_value=env):
                                with mock.patch('brain_uav.scripts.run_benchmark.make_actor', return_value=_DummyActor()):
                                    with mock.patch('brain_uav.scripts.run_benchmark.load_checkpoint', return_value=checkpoint_payload):
                                        summary = run_benchmark(args)
                    efficiency = json.loads(Path(summary['efficiency_summary_path']).read_text(encoding='utf-8'))
                    self.assertEqual(summary['method'], method)
                    self.assertIn('param_count', efficiency)
                    self.assertEqual(efficiency['actor_execution'], 'eager')
                    self.assertEqual(efficiency['actor_execution_requested'], 'eager')
                    self.assertEqual(efficiency['actor_execution_effective'], 'eager')
                    self.assertFalse(efficiency['cuda_graph_fallback_occurred'])
                    self.assertIsNone(efficiency['cuda_graph_fallback_reason'])
                    self.assertFalse(efficiency['cuda_graph_available'])
                    self.assertEqual(efficiency['cuda_graph_error'], None)
                    self.assertEqual(efficiency['cuda_graph_warmup_steps'], 0)
                    self.assertEqual(efficiency['cuda_graph_sanity_max_abs_diff'], None)
                    self.assertEqual(efficiency['cuda_graph_diagnostics_sanity_max_abs_diff'], None)
                    for key in (
                        'obs_to_tensor_time_ms',
                        'actor_forward_time_ms',
                        'action_to_cpu_time_ms',
                        'decision_time_ms',
                    ):
                        self.assertIn(key, efficiency)
                        self.assertEqual(efficiency[key]['samples'], 2)
                        self.assertGreaterEqual(efficiency[key]['avg'], 0.0)
                    self.assertTrue((Path(summary['output_dir']) / 'summary.json').is_file())

    def test_policy_benchmark_can_reuse_obs_tensor_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                method='ann',
                checkpoint=Path('model.pt'),
                benchmark_suite=Path('suite.json'),
                episodes=1,
                seed=7,
                output_root=Path(tmpdir),
                run_name='ann_reuse_buffer_run',
                device='cpu',
                snn_backend='torch',
                episode_artifacts='json',
                reuse_obs_tensor=True,
                actor_execution='eager',
            )
            env = _DummyEnv(['goal'], episode_length=2)
            scenarios = [_named_scenario(1)]
            dummy_actor = _DummyActor()
            checkpoint_payload = {'state_dict': dummy_actor.state_dict(), 'config': {}}
            with mock.patch('brain_uav.scripts.run_benchmark.load_benchmark_suite', return_value={'suite_name': 'suite', 'seed': 1, 'count_per_category': 1, 'total_scenarios': 1, 'categories': ['single_detour']}):
                with mock.patch('brain_uav.scripts.run_benchmark.build_benchmark_scenarios', return_value=scenarios):
                    with mock.patch('brain_uav.scripts.run_benchmark.make_env', return_value=env):
                        with mock.patch('brain_uav.scripts.run_benchmark.make_actor', return_value=_DummyActor()):
                            with mock.patch('brain_uav.scripts.run_benchmark.load_checkpoint', return_value=checkpoint_payload):
                                summary = run_benchmark(args)
            efficiency = json.loads(Path(summary['efficiency_summary_path']).read_text(encoding='utf-8'))
            self.assertTrue(efficiency['reuse_obs_tensor'])
            self.assertEqual(efficiency['obs_to_tensor_time_ms']['samples'], 2)

    def test_policy_benchmark_cuda_graph_on_cpu_falls_back_to_eager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                method='ann',
                checkpoint=Path('model.pt'),
                benchmark_suite=Path('suite.json'),
                episodes=1,
                seed=7,
                output_root=Path(tmpdir),
                run_name='ann_cuda_graph_cpu_run',
                device='cpu',
                snn_backend='torch',
                episode_artifacts='json',
                reuse_obs_tensor=False,
                actor_execution='cuda-graph',
            )
            env = _DummyEnv(['goal'], episode_length=2)
            scenarios = [_named_scenario(1)]
            dummy_actor = _DummyActor()
            checkpoint_payload = {'state_dict': dummy_actor.state_dict(), 'config': {}}
            with mock.patch('brain_uav.scripts.run_benchmark.load_benchmark_suite', return_value={'suite_name': 'suite', 'seed': 1, 'count_per_category': 1, 'total_scenarios': 1, 'categories': ['single_detour']}):
                with mock.patch('brain_uav.scripts.run_benchmark.build_benchmark_scenarios', return_value=scenarios):
                    with mock.patch('brain_uav.scripts.run_benchmark.make_env', return_value=env):
                        with mock.patch('brain_uav.scripts.run_benchmark.make_actor', return_value=_DummyActor()):
                            with mock.patch('brain_uav.scripts.run_benchmark.load_checkpoint', return_value=checkpoint_payload):
                                stdout = io.StringIO()
                                with redirect_stdout(stdout):
                                    summary = run_benchmark(args)
            efficiency = json.loads(Path(summary['efficiency_summary_path']).read_text(encoding='utf-8'))
            self.assertEqual(efficiency['actor_execution'], 'cuda-graph')
            self.assertEqual(efficiency['actor_execution_requested'], 'cuda-graph')
            self.assertEqual(efficiency['actor_execution_effective'], 'eager')
            self.assertFalse(efficiency['cuda_graph_available'])
            self.assertIn('requires device=cuda', efficiency['cuda_graph_error'])
            self.assertTrue(efficiency['cuda_graph_fallback_occurred'])
            self.assertEqual(
                efficiency['cuda_graph_fallback_reason'],
                efficiency['cuda_graph_error'],
            )
            self.assertIn(
                '[benchmark] CUDA Graph unavailable; falling back to eager:',
                stdout.getvalue(),
            )
            self.assertEqual(efficiency['cuda_graph_diagnostics_sanity_max_abs_diff'], None)
            self.assertEqual(efficiency['actor_forward_time_ms']['samples'], 2)

    def test_policy_benchmark_cuda_graph_success_records_effective_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                method='ann',
                checkpoint=Path('model.pt'),
                benchmark_suite=Path('suite.json'),
                episodes=1,
                seed=7,
                output_root=Path(tmpdir),
                run_name='ann_cuda_graph_success_run',
                device='cpu',
                snn_backend='torch',
                episode_artifacts='json',
                reuse_obs_tensor=False,
                actor_execution='cuda-graph',
            )
            env = _DummyEnv(['goal'], episode_length=2)
            scenarios = [_named_scenario(1)]
            dummy_actor = _DummyActor()
            checkpoint_payload = {'state_dict': dummy_actor.state_dict(), 'config': {}}
            cuda_graph_state = {
                'available': True,
                'error': None,
                'warmup_steps': 3,
                'sanity_max_abs_diff': 0.0,
                'diagnostics_sanity_max_abs_diff': None,
                'graph': _DummyCudaGraph(),
                'static_obs_tensor': torch.zeros((1, 24), dtype=torch.float32),
                'static_action_tensor': torch.zeros((1, 2), dtype=torch.float32),
            }
            with mock.patch('brain_uav.scripts.run_benchmark.load_benchmark_suite', return_value={'suite_name': 'suite', 'seed': 1, 'count_per_category': 1, 'total_scenarios': 1, 'categories': ['single_detour']}):
                with mock.patch('brain_uav.scripts.run_benchmark.build_benchmark_scenarios', return_value=scenarios):
                    with mock.patch('brain_uav.scripts.run_benchmark.make_env', return_value=env):
                        with mock.patch('brain_uav.scripts.run_benchmark.make_actor', return_value=_DummyActor()):
                            with mock.patch('brain_uav.scripts.run_benchmark.load_checkpoint', return_value=checkpoint_payload):
                                with mock.patch('brain_uav.scripts.run_benchmark._build_cuda_graph_state', return_value=cuda_graph_state):
                                    stdout = io.StringIO()
                                    with redirect_stdout(stdout):
                                        summary = run_benchmark(args)
            efficiency = json.loads(Path(summary['efficiency_summary_path']).read_text(encoding='utf-8'))
            self.assertEqual(efficiency['actor_execution'], 'cuda-graph')
            self.assertEqual(efficiency['actor_execution_requested'], 'cuda-graph')
            self.assertEqual(efficiency['actor_execution_effective'], 'cuda-graph')
            self.assertTrue(efficiency['cuda_graph_available'])
            self.assertFalse(efficiency['cuda_graph_fallback_occurred'])
            self.assertIsNone(efficiency['cuda_graph_fallback_reason'])
            self.assertEqual(efficiency['cuda_graph_sanity_max_abs_diff'], 0.0)
            self.assertNotIn('falling back to eager', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
