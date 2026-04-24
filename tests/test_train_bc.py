"""Tests for BC checkpoint helpers."""

import tempfile
import unittest
from pathlib import Path

import torch

from brain_uav.config import ExperimentConfig, ScenarioConfig
from brain_uav.scripts.train_bc import build_bc_checkpoint_payload, build_parser
from brain_uav.utils.io import load_checkpoint, save_checkpoint


class TestTrainBCHelpers(unittest.TestCase):
    def test_best_output_argument_is_parsed(self):
        parser = build_parser()
        args = parser.parse_args(
            ['--dataset', 'data/demo.npz', '--best-output', 'models/best.pt', '--device', 'cuda', '--snn-backend', 'cupy']
        )
        self.assertEqual(args.best_output, Path('models/best.pt'))
        self.assertEqual(args.device, 'cuda')
        self.assertEqual(args.snn_backend, 'cupy')

    def test_checkpoint_payload_records_best_fields(self):
        cfg = ExperimentConfig()
        state_dim = 5 + 3 + 4 + 4 * ScenarioConfig().nearest_zone_count
        actor = torch.nn.Linear(state_dim, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / 'best.pt'
            payload = build_bc_checkpoint_payload(
                model='ann',
                actor=actor,
                history=[0.5, 0.25],
                cfg=cfg,
                finished_at='20260424_120000',
                log_dir=Path(tmpdir),
                dataset_path=Path('data/demo.npz'),
                dataset_version='v_test',
                dataset_config=None,
                curriculum_level='easy',
                curriculum_mix={'easy': 1.0},
                init_checkpoint=None,
                best_loss=0.25,
                best_epoch=2,
                checkpoint_kind='best',
            )
            save_checkpoint(target, payload)
            saved = load_checkpoint(target)

        self.assertEqual(saved['best_loss'], 0.25)
        self.assertEqual(saved['best_epoch'], 2)
        self.assertEqual(saved['checkpoint_kind'], 'best')


if __name__ == '__main__':
    unittest.main()
