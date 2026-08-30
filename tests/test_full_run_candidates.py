"""Tests for the parallel-candidate full-run pipeline helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from brain_uav.scripts.run_full_pipeline import FullRunStageError
from brain_uav.scripts.run_full_pipeline_candidates import (
    build_parser,
    candidate_seed,
    make_candidate_run,
    run_candidate_stage,
    run_full_pipeline_candidates,
)


class _FakePopen:
    early_seed: int | None = None
    hanging_seeds: set[int] = set()
    instances: list['_FakePopen'] = []

    def __init__(self, command, *, cwd, env) -> None:
        del cwd, env
        self.command = list(command)
        self.returncode = 0
        self.terminated = False
        self.killed = False
        self.seed = int(self.command[self.command.index('--seed') + 1])
        self.output = Path(self.command[self.command.index('--output') + 1])
        self.log_root = Path(self.command[self.command.index('--log-root') + 1])
        self.metrics_name = self.command[self.command.index('--metrics-out') + 1]
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(f'checkpoint seed={self.seed}', encoding='utf-8')
        metrics_dir = self.log_root / 'fake_ts'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'stopped_early': self.seed == self.early_seed,
            'stop_reason': 'qualified windows' if self.seed == self.early_seed else None,
            'steps': 1234 + self.seed,
            'episodes': 10,
            'outcomes': {'goal': 9, 'collision': 1},
        }
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

    def _capture_pipeline_commands(
        self,
        output_root: Path,
        extra_args: list[str] | None = None,
    ) -> tuple[dict, list[list[str]], dict]:
        parser = build_parser()
        argv = [
            '--model',
            'ann',
            '--max-stage',
            'easy',
            '--output-root',
            str(output_root),
            '--device',
            'cpu',
        ]
        args = parser.parse_args(argv + (extra_args or []))
        command_calls: list[list[str]] = []

        def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
            del cwd, env
            command_calls.append(command)

        winner = SimpleNamespace(
            candidate_id=0,
            seed=7,
            metrics_path=output_root / 'winner_metrics.json',
            stop_reason='qualified windows',
        )
        with mock.patch('brain_uav.scripts.run_full_pipeline.make_run_name', return_value='fixed_candidates'):
            with mock.patch(
                'brain_uav.scripts.run_full_pipeline_candidates.run_command',
                side_effect=fake_run_command,
            ):
                with mock.patch(
                    'brain_uav.scripts.run_full_pipeline_candidates.find_latest_metrics_file',
                    return_value=output_root / 'bc_metrics.json',
                ):
                    with mock.patch(
                        'brain_uav.scripts.run_full_pipeline_candidates.run_candidate_stage',
                        return_value=(winner, []),
                    ) as candidate_stage:
                        with mock.patch('brain_uav.scripts.run_full_pipeline_candidates.save_json'):
                            report = run_full_pipeline_candidates(args)
        return report, command_calls, dict(candidate_stage.call_args.kwargs)

    def test_parser_defaults_use_four_candidates(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'snn'])

        self.assertEqual(args.candidates, 4)
        self.assertEqual(args.candidate_workers, 4)
        self.assertFalse(args.keep_candidate_runs)
        self.assertEqual(args.tag, 'candidates')
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.max_stage, 'hard')
        self.assertEqual(args.output_root, Path('outputs/full_run'))
        self.assertEqual(args.device, 'auto')
        self.assertEqual(args.snn_backend, 'torch')
        self.assertEqual(args.poll_interval, 5.0)
        self.assertIsNone(args.bc_seed)

    def test_default_bc_command_matches_legacy_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            report, commands, _ = self._capture_pipeline_commands(output_root)
            run_root = output_root / 'fixed_candidates'
            expected_bc_command = [
                sys.executable,
                '-m',
                'brain_uav.scripts.train_bc',
                '--dataset',
                str(run_root / 'data' / 'bc_dataset_easy_v6.npz'),
                '--model',
                'ann',
                '--output',
                str(run_root / 'models' / 'bc_ann_final.pt'),
                '--best-output',
                str(run_root / 'models' / 'bc_ann_best.pt'),
                '--metrics-out',
                'bc_ann_metrics.json',
                '--log-root',
                str(run_root / 'logs' / 'bc'),
                '--device',
                'cpu',
                '--snn-backend',
                'torch',
            ]

        bc_command = next(cmd for cmd in commands if 'brain_uav.scripts.train_bc' in cmd)
        self.assertEqual(bc_command, expected_bc_command)
        self.assertIsNone(report['bc_seed_requested'])
        self.assertIsNone(report['bc_seed_effective'])

    def test_explicit_bc_seed_only_changes_bc_command_and_not_td3_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            default_report, default_commands, default_td3 = self._capture_pipeline_commands(output_root)
            seeded_report, seeded_commands, seeded_td3 = self._capture_pipeline_commands(
                output_root,
                ['--bc-seed', '4321'],
            )

        default_bc = next(cmd for cmd in default_commands if 'brain_uav.scripts.train_bc' in cmd)
        seeded_bc = next(cmd for cmd in seeded_commands if 'brain_uav.scripts.train_bc' in cmd)
        self.assertNotIn('--seed', default_bc)
        self.assertEqual(seeded_bc, [*default_bc, '--seed', '4321'])
        self.assertEqual(seeded_td3, default_td3)
        self.assertIsNone(default_report['bc_seed_requested'])
        self.assertIsNone(default_report['bc_seed_effective'])
        self.assertEqual(seeded_report['bc_seed_requested'], 4321)
        self.assertEqual(seeded_report['bc_seed_effective'], 4321)

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
        self.assertEqual(
            candidate.command,
            [
                sys.executable,
                '-m',
                'brain_uav.scripts.train_td3',
                '--model',
                'ann',
                '--curriculum-level',
                'medium',
                '--init-checkpoint',
                str(root / 'init.pt'),
                '--output',
                str(root / 'candidates' / 'cand_02_seed207' / 'td3_ann_medium.pt'),
                '--metrics-out',
                'td3_ann_medium_metrics.json',
                '--log-root',
                str(root / 'candidates' / 'cand_02_seed207' / 'logs'),
                '--seed',
                '207',
                '--early-stop-enabled',
                '--summary-every-episodes',
                '15',
                '--early-stop-windows',
                '4',
                '--early-stop-max-failures-per-window',
                '1',
                '--early-stop-goal-rate',
                '0.95',
                '--early-stop-min-steps',
                '125000',
                '--device',
                'cuda',
                '--snn-backend',
                'torch',
            ],
        )

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
