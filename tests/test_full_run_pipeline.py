"""Tests for the full-run pipeline helpers."""

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from brain_uav.scripts.run_full_pipeline import (
    FullRunStageError,
    build_parser,
    create_full_run_layout,
    ensure_stage_stopped_early,
    run_full_pipeline,
    main,
)


class TestFullRunPipeline(unittest.TestCase):
    def _capture_pipeline_commands(
        self,
        output_root: Path,
        extra_args: list[str] | None = None,
    ) -> tuple[object, dict, list[list[str]]]:
        parser = build_parser()
        argv = [
            '--model',
            'snn',
            '--max-stage',
            'easy',
            '--output-root',
            str(output_root),
            '--device',
            'cuda',
            '--snn-backend',
            'cupy',
        ]
        args = parser.parse_args(argv + (extra_args or []))
        command_calls: list[list[str]] = []

        def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
            del cwd, env
            command_calls.append(command)

        with mock.patch('brain_uav.scripts.run_full_pipeline.make_run_name', return_value='fixed_run'):
            with mock.patch('brain_uav.scripts.run_full_pipeline.run_command', side_effect=fake_run_command):
                with mock.patch(
                    'brain_uav.scripts.run_full_pipeline.find_latest_metrics_file',
                    return_value=output_root / 'metrics.json',
                ):
                    with mock.patch(
                        'brain_uav.scripts.run_full_pipeline.ensure_stage_stopped_early',
                        return_value={'stopped_early': True, 'stop_reason': 'ok'},
                    ):
                        with mock.patch('brain_uav.scripts.run_full_pipeline.save_json'):
                            report = run_full_pipeline(args)
        return args, report, command_calls

    def test_layout_naming_and_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = create_full_run_layout(
                Path(tmpdir),
                'ann',
                'bugfix speedup',
                now=datetime(2026, 4, 24, 15, 30, 0),
            )
            self.assertEqual(layout.root.name, '0424_153000_ann_bugfix_speedup')
            self.assertTrue(layout.data_dir.is_dir())
            self.assertTrue(layout.models_dir.is_dir())
            self.assertTrue(layout.logs_dir.is_dir())
            self.assertTrue(layout.reports_dir.is_dir())

    def test_default_max_stage_is_hard(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'snn'])
        self.assertEqual(args.max_stage, 'hard')

    def test_parser_preserves_legacy_defaults_and_bc_seed_is_none(self):
        args = build_parser().parse_args(['--model', 'snn'])

        self.assertEqual(args.tag, 'run')
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.max_stage, 'hard')
        self.assertEqual(args.output_root, Path('outputs/full_run'))
        self.assertEqual(args.device, 'auto')
        self.assertEqual(args.snn_backend, 'torch')
        self.assertIsNone(args.bc_seed)

    def test_default_bc_and_td3_commands_match_legacy_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _, report, commands = self._capture_pipeline_commands(output_root)
            run_root = output_root / 'fixed_run'
            dataset_path = run_root / 'data' / 'bc_dataset_easy_v6.npz'
            expected_bc_command = [
                sys.executable,
                '-m',
                'brain_uav.scripts.train_bc',
                '--dataset',
                str(dataset_path),
                '--model',
                'snn',
                '--output',
                str(run_root / 'models' / 'bc_snn_final.pt'),
                '--best-output',
                str(run_root / 'models' / 'bc_snn_best.pt'),
                '--metrics-out',
                'bc_snn_metrics.json',
                '--log-root',
                str(run_root / 'logs' / 'bc'),
                '--device',
                'cuda',
                '--snn-backend',
                'cupy',
            ]
            expected_td3_command = [
                sys.executable,
                '-m',
                'brain_uav.scripts.train_td3',
                '--model',
                'snn',
                '--curriculum-level',
                'easy',
                '--init-checkpoint',
                str(run_root / 'models' / 'bc_snn_best.pt'),
                '--output',
                str(run_root / 'models' / 'td3_snn_easy.pt'),
                '--metrics-out',
                'td3_snn_easy_metrics.json',
                '--log-root',
                str(run_root / 'logs' / 'td3' / 'easy'),
                '--seed',
                '7',
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
                'cupy',
            ]

        bc_command = next(cmd for cmd in commands if 'brain_uav.scripts.train_bc' in cmd)
        td3_command = next(cmd for cmd in commands if 'brain_uav.scripts.train_td3' in cmd)
        self.assertEqual(bc_command, expected_bc_command)
        self.assertEqual(td3_command, expected_td3_command)
        self.assertIsNone(report['bc_seed_requested'])
        self.assertIsNone(report['bc_seed_effective'])

    def test_explicit_bc_seed_only_changes_bc_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _, default_report, default_commands = self._capture_pipeline_commands(output_root)
            _, seeded_report, seeded_commands = self._capture_pipeline_commands(output_root, ['--bc-seed', '1234'])

        default_bc = next(cmd for cmd in default_commands if 'brain_uav.scripts.train_bc' in cmd)
        seeded_bc = next(cmd for cmd in seeded_commands if 'brain_uav.scripts.train_bc' in cmd)
        default_td3 = next(cmd for cmd in default_commands if 'brain_uav.scripts.train_td3' in cmd)
        seeded_td3 = next(cmd for cmd in seeded_commands if 'brain_uav.scripts.train_td3' in cmd)
        self.assertNotIn('--seed', default_bc)
        self.assertEqual(seeded_bc, [*default_bc, '--seed', '1234'])
        self.assertEqual(seeded_td3, default_td3)
        self.assertIsNone(default_report['bc_seed_requested'])
        self.assertIsNone(default_report['bc_seed_effective'])
        self.assertEqual(seeded_report['bc_seed_requested'], 1234)
        self.assertEqual(seeded_report['bc_seed_effective'], 1234)

    def test_stage_failure_helper_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / 'metrics.json'
            metrics_path.write_text(json.dumps({'stopped_early': False, 'stop_reason': None}), encoding='utf-8')
            with self.assertRaises(FullRunStageError):
                ensure_stage_stopped_early(metrics_path, 'easy')

    def test_stage_helper_rejects_stop_reason_without_true_early_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / 'metrics.json'
            metrics_path.write_text(
                json.dumps({'stopped_early': False, 'stop_reason': 'final flush only'}),
                encoding='utf-8',
            )
            with self.assertRaises(FullRunStageError):
                ensure_stage_stopped_early(metrics_path, 'easy')

    def test_main_exits_nonzero_on_stage_failure(self):
        with mock.patch(
            'brain_uav.scripts.run_full_pipeline.run_full_pipeline',
            side_effect=FullRunStageError('Stage easy did not stop early'),
        ):
            with self.assertRaises(SystemExit) as ctx:
                main(['--model', 'ann'])
        self.assertEqual(ctx.exception.code, 1)

    def test_run_full_pipeline_passes_device_and_early_stop_args(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'snn', '--device', 'cuda', '--snn-backend', 'cupy'])
        command_calls: list[list[str]] = []

        def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
            command_calls.append(command)

        with tempfile.TemporaryDirectory() as tmpdir:
            args.output_root = Path(tmpdir)
            with mock.patch('brain_uav.scripts.run_full_pipeline.run_command', side_effect=fake_run_command):
                with mock.patch(
                    'brain_uav.scripts.run_full_pipeline.find_latest_metrics_file',
                    return_value=Path(tmpdir) / 'metrics.json',
                ):
                    with mock.patch(
                        'brain_uav.scripts.run_full_pipeline.ensure_stage_stopped_early',
                        return_value={'stopped_early': True, 'stop_reason': 'ok'},
                    ):
                        with mock.patch('brain_uav.scripts.run_full_pipeline.save_json'):
                            run_full_pipeline(args)

        bc_command = next(cmd for cmd in command_calls if 'brain_uav.scripts.train_bc' in cmd)
        td3_command = next(cmd for cmd in command_calls if 'brain_uav.scripts.train_td3' in cmd)
        self.assertIn('--device', bc_command)
        self.assertIn('cuda', bc_command)
        self.assertIn('--snn-backend', bc_command)
        self.assertIn('cupy', bc_command)
        self.assertIn('--device', td3_command)
        self.assertIn('--summary-every-episodes', td3_command)
        self.assertIn('15', td3_command)
        self.assertIn('--early-stop-max-failures-per-window', td3_command)
        self.assertIn('1', td3_command)
        self.assertIn('--early-stop-goal-rate', td3_command)
        self.assertIn('0.95', td3_command)
        self.assertIn('--early-stop-min-steps', td3_command)
        self.assertIn('125000', td3_command)


if __name__ == '__main__':
    unittest.main()
