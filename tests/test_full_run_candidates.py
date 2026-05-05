"""Tests for the parallel-candidate full-run pipeline helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from brain_uav.scripts.run_full_pipeline import FullRunStageError
from brain_uav.scripts.run_full_pipeline_candidates import (
    build_parser,
    candidate_seed,
    make_candidate_run,
    run_full_pipeline_candidates,
    run_candidate_stage,
    select_fallback_candidate,
)


class _FakePopen:
    early_seed: int | None = None
    hanging_seeds: set[int] = set()
    fail_seeds: set[int] = set()
    missing_metrics_seeds: set[int] = set()
    metrics_by_seed: dict[int, dict] = {}
    instances: list['_FakePopen'] = []

    def __init__(self, command, *, cwd, env) -> None:
        del cwd, env
        self.command = list(command)
        self.terminated = False
        self.killed = False
        self.seed = int(self.command[self.command.index('--seed') + 1])
        self.returncode = 1 if self.seed in self.fail_seeds else 0
        self.output = Path(self.command[self.command.index('--output') + 1])
        self.log_root = Path(self.command[self.command.index('--log-root') + 1])
        self.metrics_name = self.command[self.command.index('--metrics-out') + 1]
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(f'checkpoint seed={self.seed}', encoding='utf-8')
        metrics_dir = self.log_root / 'fake_ts'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if self.seed not in self.missing_metrics_seeds:
            payload = self.metrics_by_seed.get(
                self.seed,
                {
                    'stopped_early': self.seed == self.early_seed,
                    'stop_reason': 'qualified windows' if self.seed == self.early_seed else None,
                    'steps': 1234 + self.seed,
                    'episodes': 10,
                    'outcomes': {'goal': 9, 'collision': 1},
                    'episode_window_stats': [
                        {'goal_count': 2, 'collision_count': 1, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0}
                    ],
                },
            )
            (metrics_dir / self.metrics_name).write_text(json.dumps(payload), encoding='utf-8')
        self.instances.append(self)

    def poll(self):
        if self.seed in self.hanging_seeds and not self.terminated:
            return None
        return self.returncode if not self.terminated else None

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        del timeout
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class TestFullRunCandidates(unittest.TestCase):
    def setUp(self) -> None:
        _FakePopen.instances = []
        _FakePopen.early_seed = None
        _FakePopen.hanging_seeds = set()
        _FakePopen.fail_seeds = set()
        _FakePopen.missing_metrics_seeds = set()
        _FakePopen.metrics_by_seed = {}

    def test_parser_defaults_use_four_candidates(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'snn'])

        self.assertEqual(args.candidates, 4)
        self.assertEqual(args.candidate_workers, 4)
        self.assertFalse(args.keep_candidate_runs)
        self.assertFalse(args.skip_bc)
        self.assertFalse(args.hard_only)
        self.assertFalse(args.disable_terminal_guidance)
        self.assertFalse(args.continue_on_stage_failure)
        self.assertIsNone(args.td3_curriculum_mix)

    def test_candidate_seed_stride_is_100(self):
        self.assertEqual([candidate_seed(7, idx) for idx in range(4)], [7, 107, 207, 307])

    def test_candidate_command_uses_candidate_seed_and_output_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = make_candidate_run(
                model='ann',
                stage='medium',
                candidate_id=2,
                base_seed=7,
                init_checkpoint=root / 'init.pt',
                stage_candidate_root=root / 'candidates',
                summary_every_episodes=15,
                early_stop_min_steps=125000,
                device='cuda',
                snn_backend='torch',
            )

        self.assertEqual(candidate.seed, 207)
        self.assertIn('brain_uav.scripts.train_td3', candidate.command)
        self.assertIn('--curriculum-level', candidate.command)
        self.assertIn('medium', candidate.command)
        self.assertIn('--seed', candidate.command)
        self.assertIn('207', candidate.command)
        self.assertIn('--early-stop-min-steps', candidate.command)
        self.assertIn('125000', candidate.command)

    def test_candidate_command_can_skip_init_and_pass_ablation_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = make_candidate_run(
                model='snn',
                stage='hard',
                candidate_id=0,
                base_seed=7,
                init_checkpoint=None,
                stage_candidate_root=root / 'candidates',
                summary_every_episodes=15,
                early_stop_min_steps=125000,
                device='cuda',
                snn_backend='torch',
                td3_curriculum_mix='hard:1.0',
                disable_terminal_guidance=True,
            )

        self.assertNotIn('--init-checkpoint', candidate.command)
        self.assertIn('--curriculum-mix', candidate.command)
        self.assertIn('hard:1.0', candidate.command)
        self.assertIn('--disable-terminal-guidance', candidate.command)

    def test_stage_selects_early_stop_winner_and_terminates_other_running_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            init_checkpoint = root / 'init.pt'
            init_checkpoint.write_text('init', encoding='utf-8')
            _FakePopen.early_seed = 107
            _FakePopen.hanging_seeds = {207, 307}

            with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.subprocess.Popen', _FakePopen):
                with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.time.sleep'):
                    winner, summaries = run_candidate_stage(
                        project_root=root,
                        env={},
                        model='snn',
                        stage='easy_two_zone',
                        init_checkpoint=init_checkpoint,
                        stage_checkpoint=root / 'models' / 'td3_snn_easy_two_zone.pt',
                        stage_candidate_root=root / 'candidate_runs' / 'easy_two_zone',
                        winner_log_dir=root / 'logs' / 'td3' / 'easy_two_zone' / 'winner_run',
                        base_seed=7,
                        candidates=4,
                        candidate_workers=4,
                        keep_candidate_runs=False,
                        device='cuda',
                        snn_backend='torch',
                        poll_interval=0.0,
                    )

            self.assertEqual(winner.candidate_id, 1)
            self.assertEqual(winner.seed, 107)
            self.assertTrue((root / 'models' / 'td3_snn_easy_two_zone.pt').is_file())
            self.assertTrue((root / 'logs' / 'td3' / 'easy_two_zone' / 'winner_run').is_dir())
            self.assertEqual(len(summaries), 4)
            self.assertEqual(summaries[1]['status'], 'winner')
            self.assertEqual(summaries[2]['status'], 'terminated')
            self.assertEqual(summaries[3]['status'], 'terminated')
            self.assertFalse((root / 'candidate_runs' / 'easy_two_zone' / 'cand_02_seed207').exists())
            self.assertFalse((root / 'candidate_runs' / 'easy_two_zone' / 'cand_03_seed307').exists())

    def test_stage_failure_when_no_candidate_stops_early_preserves_candidate_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            init_checkpoint = root / 'init.pt'
            init_checkpoint.write_text('init', encoding='utf-8')
            _FakePopen.early_seed = None

            with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.subprocess.Popen', _FakePopen):
                with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.time.sleep'):
                    with self.assertRaises(FullRunStageError):
                        run_candidate_stage(
                            project_root=root,
                            env={},
                            model='ann',
                            stage='medium',
                            init_checkpoint=init_checkpoint,
                            stage_checkpoint=root / 'models' / 'td3_ann_medium.pt',
                            stage_candidate_root=root / 'candidate_runs' / 'medium',
                            winner_log_dir=root / 'logs' / 'td3' / 'medium' / 'winner_run',
                            base_seed=7,
                            candidates=4,
                            candidate_workers=4,
                            keep_candidate_runs=False,
                            device='cpu',
                            snn_backend='torch',
                            poll_interval=0.0,
                        )

            self.assertTrue((root / 'candidate_runs' / 'medium' / 'cand_00_seed7').exists())
            self.assertTrue((root / 'candidate_runs' / 'medium' / 'cand_01_seed107').exists())
            self.assertTrue((root / 'candidate_runs' / 'medium' / 'cand_02_seed207').exists())
            self.assertTrue((root / 'candidate_runs' / 'medium' / 'cand_03_seed307').exists())

    def test_stage_fallback_selects_candidate_when_continue_on_failure_is_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            init_checkpoint = root / 'init.pt'
            init_checkpoint.write_text('init', encoding='utf-8')
            _FakePopen.metrics_by_seed = {
                7: {
                    'stopped_early': False,
                    'steps': 5000,
                    'episodes': 50,
                    'outcomes': {'goal': 12},
                    'episode_window_stats': [
                        {'goal_count': 1, 'collision_count': 3, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0},
                        {'goal_count': 1, 'collision_count': 3, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0},
                        {'goal_count': 1, 'collision_count': 3, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0},
                        {'goal_count': 1, 'collision_count': 3, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0},
                    ],
                },
                107: {
                    'stopped_early': False,
                    'steps': 7000,
                    'episodes': 50,
                    'outcomes': {'goal': 18},
                    'episode_window_stats': [
                        {'goal_count': 3, 'collision_count': 1, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0},
                        {'goal_count': 3, 'collision_count': 1, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0},
                        {'goal_count': 3, 'collision_count': 1, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0},
                        {'goal_count': 3, 'collision_count': 1, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0},
                    ],
                },
            }

            with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.subprocess.Popen', _FakePopen):
                with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.time.sleep'):
                    selected, summaries = run_candidate_stage(
                        project_root=root,
                        env={},
                        model='snn',
                        stage='medium',
                        init_checkpoint=init_checkpoint,
                        stage_checkpoint=root / 'models' / 'td3_snn_medium.pt',
                        stage_candidate_root=root / 'candidate_runs' / 'medium',
                        winner_log_dir=root / 'logs' / 'td3' / 'medium' / 'winner_run',
                        base_seed=7,
                        candidates=2,
                        candidate_workers=2,
                        keep_candidate_runs=False,
                        device='cuda',
                        snn_backend='torch',
                        poll_interval=0.0,
                        continue_on_stage_failure=True,
                    )

        self.assertEqual(selected.seed, 107)
        self.assertEqual(selected.status, 'fallback_selected_no_early_stop')
        self.assertEqual(summaries[1]['status'], 'fallback_selected_no_early_stop')
        self.assertIn('last4_goal_count=12', selected.selection_reason)

    def test_select_fallback_ignores_failed_or_missing_metrics_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            good = make_candidate_run(
                model='ann',
                stage='hard',
                candidate_id=0,
                base_seed=7,
                init_checkpoint=None,
                stage_candidate_root=root / 'candidates',
                summary_every_episodes=15,
                early_stop_min_steps=125000,
                device='cpu',
                snn_backend='torch',
            )
            failed = make_candidate_run(
                model='ann',
                stage='hard',
                candidate_id=1,
                base_seed=7,
                init_checkpoint=None,
                stage_candidate_root=root / 'candidates',
                summary_every_episodes=15,
                early_stop_min_steps=125000,
                device='cpu',
                snn_backend='torch',
            )
            missing = make_candidate_run(
                model='ann',
                stage='hard',
                candidate_id=2,
                base_seed=7,
                init_checkpoint=None,
                stage_candidate_root=root / 'candidates',
                summary_every_episodes=15,
                early_stop_min_steps=125000,
                device='cpu',
                snn_backend='torch',
            )
            good.return_code = 0
            good.metrics_path = good.log_root / 'metrics.json'
            good.metrics_payload = {
                'steps': 100,
                'outcomes': {'goal': 5},
                'episode_window_stats': [
                    {'goal_count': 5, 'collision_count': 0, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0}
                ],
            }
            failed.return_code = 1
            failed.metrics_path = failed.log_root / 'metrics.json'
            failed.metrics_payload = {
                'steps': 10,
                'outcomes': {'goal': 99},
                'episode_window_stats': [
                    {'goal_count': 99, 'collision_count': 0, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0}
                ],
            }
            missing.return_code = 0
            missing.metrics_path = None

            selected, _reason = select_fallback_candidate([failed, missing, good])

        self.assertIs(selected, good)

    def test_select_fallback_prefers_safety_after_goal_tie(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unsafe = make_candidate_run(
                model='snn',
                stage='medium',
                candidate_id=0,
                base_seed=7,
                init_checkpoint=None,
                stage_candidate_root=root / 'candidates',
                summary_every_episodes=15,
                early_stop_min_steps=125000,
                device='cpu',
                snn_backend='torch',
            )
            safe = make_candidate_run(
                model='snn',
                stage='medium',
                candidate_id=1,
                base_seed=7,
                init_checkpoint=None,
                stage_candidate_root=root / 'candidates',
                summary_every_episodes=15,
                early_stop_min_steps=125000,
                device='cpu',
                snn_backend='torch',
            )
            for candidate, safety_failures in ((unsafe, 8), (safe, 2)):
                candidate.return_code = 0
                candidate.metrics_path = candidate.log_root / 'metrics.json'
                candidate.metrics_payload = {
                    'steps': 1000,
                    'outcomes': {'goal': 20},
                    'episode_window_stats': [
                        {
                            'goal_count': 5,
                            'collision_count': safety_failures,
                            'boundary_count': 0,
                            'ground_count': 0,
                            'timeout_count': 0,
                        }
                    ],
                }

            selected, reason = select_fallback_candidate([unsafe, safe])

        self.assertIs(selected, safe)
        self.assertIn('last4_safety_failures=2', reason)

    def test_skip_bc_uses_random_first_stage_and_does_not_run_data_or_bc_commands(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'snn', '--skip-bc', '--hard-only'])
        _FakePopen.early_seed = 7
        command_calls: list[list[str]] = []

        def fake_run_command(command, *, cwd, env):
            del cwd, env
            command_calls.append(command)

        with tempfile.TemporaryDirectory() as tmpdir:
            args.output_root = Path(tmpdir)
            with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.run_command', side_effect=fake_run_command):
                with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.subprocess.Popen', _FakePopen):
                    with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.time.sleep'):
                        report = run_full_pipeline_candidates(args)

        self.assertEqual(command_calls, [])
        self.assertTrue(report['skip_bc'])
        self.assertIsNone(report['dataset_path'])
        self.assertEqual(report['bc']['skipped'], True)
        self.assertEqual(report['stages'], ['hard'])
        self.assertNotIn('--init-checkpoint', _FakePopen.instances[0].command)

    def test_hard_only_mix_disable_terminal_and_continue_failure_are_reported(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                '--model',
                'snn',
                '--skip-bc',
                '--hard-only',
                '--td3-curriculum-mix',
                'hard:1.0',
                '--disable-terminal-guidance',
                '--continue-on-stage-failure',
            ]
        )
        _FakePopen.early_seed = None
        _FakePopen.metrics_by_seed = {
            7: {
                'stopped_early': False,
                'steps': 1000,
                'episodes': 20,
                'outcomes': {'goal': 8},
                'episode_window_stats': [
                    {'goal_count': 4, 'collision_count': 1, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0},
                    {'goal_count': 4, 'collision_count': 1, 'boundary_count': 0, 'ground_count': 0, 'timeout_count': 0},
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            args.output_root = Path(tmpdir)
            with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.run_command'):
                with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.subprocess.Popen', _FakePopen):
                    with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.time.sleep'):
                        report = run_full_pipeline_candidates(args)

            selection_summary = Path(report['run_root']) / 'reports' / 'selection_summary.json'
            self.assertTrue(selection_summary.is_file())

        command = _FakePopen.instances[0].command
        self.assertEqual(report['stages'], ['hard'])
        self.assertTrue(report['hard_only'])
        self.assertTrue(report['continue_on_stage_failure'])
        self.assertEqual(report['td3_curriculum_mix'], 'hard:1.0')
        self.assertTrue(report['disable_terminal_guidance'])
        self.assertIn('--curriculum-mix', command)
        self.assertIn('hard:1.0', command)
        self.assertIn('--disable-terminal-guidance', command)
        self.assertEqual(report['candidate_stages'][0]['status'], 'fallback_selected_no_early_stop')
        self.assertIn('fallback_score', report['candidate_stages'][0]['selection_reason'])

    def test_keep_candidate_runs_preserves_non_winner_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            init_checkpoint = root / 'init.pt'
            init_checkpoint.write_text('init', encoding='utf-8')
            _FakePopen.early_seed = 7

            with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.subprocess.Popen', _FakePopen):
                with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.time.sleep'):
                    run_candidate_stage(
                        project_root=root,
                        env={},
                        model='snn',
                        stage='easy',
                        init_checkpoint=init_checkpoint,
                        stage_checkpoint=root / 'models' / 'td3_snn_easy.pt',
                        stage_candidate_root=root / 'candidate_runs' / 'easy',
                        winner_log_dir=root / 'logs' / 'td3' / 'easy' / 'winner_run',
                        base_seed=7,
                        candidates=4,
                        candidate_workers=4,
                        keep_candidate_runs=True,
                        device='cuda',
                        snn_backend='torch',
                        poll_interval=0.0,
                    )

            self.assertTrue((root / 'candidate_runs' / 'easy' / 'cand_01_seed107').exists())


if __name__ == '__main__':
    unittest.main()
