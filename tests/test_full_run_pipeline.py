"""Tests for the full-run pipeline helpers."""

import json
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


if __name__ == '__main__':
    unittest.main()
