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
    run_candidate_stage,
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

    def test_parser_defaults_use_four_candidates(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'snn'])

        self.assertEqual(args.candidates, 4)
        self.assertEqual(args.candidate_workers, 4)
        self.assertFalse(args.keep_candidate_runs)

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

    def test_stage_failure_when_no_candidate_stops_early(self):
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

            self.assertFalse((root / 'candidate_runs' / 'medium' / 'cand_00_seed7').exists())

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
