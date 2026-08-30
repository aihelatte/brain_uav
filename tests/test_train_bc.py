"""Tests for the BC training entry point and checkpoint helpers."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brain_uav.config import ExperimentConfig, ScenarioConfig
from brain_uav.scripts import train_bc
from brain_uav.scripts.train_bc import build_bc_checkpoint_payload, build_parser
from brain_uav.trainers import train_behavior_cloning
from brain_uav.utils.io import load_checkpoint, save_checkpoint


class TestTrainBCHelpers(unittest.TestCase):
    def test_seed_defaults_to_none(self):
        args = build_parser().parse_args(['--dataset', 'data/demo.npz'])

        self.assertIsNone(args.seed)

    def test_explicit_seed_is_parsed(self):
        args = build_parser().parse_args(['--dataset', 'data/demo.npz', '--seed', '1234'])

        self.assertEqual(args.seed, 1234)

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
                bc_seed_requested=17,
                bc_seed_effective=17,
            )
            save_checkpoint(target, payload)
            saved = load_checkpoint(target)

        self.assertEqual(saved['best_loss'], 0.25)
        self.assertEqual(saved['best_epoch'], 2)
        self.assertEqual(saved['checkpoint_kind'], 'best')
        self.assertEqual(saved['bc_seed_requested'], 17)
        self.assertEqual(saved['bc_seed_effective'], 17)

    def test_dataloader_generator_is_independent_from_global_torch_rng(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / 'tiny.npz'
            rng = np.random.default_rng(20260830)
            np.savez(
                dataset_path,
                observations=rng.normal(size=(11, 3)).astype(np.float32),
                actions=rng.normal(size=(11, 2)).astype(np.float32),
            )

            template = torch.nn.Linear(3, 2)
            initial_state = {name: tensor.detach().clone() for name, tensor in template.state_dict().items()}

            def run_once(global_seed: int) -> tuple[list[float], dict[str, torch.Tensor]]:
                actor = torch.nn.Linear(3, 2)
                actor.load_state_dict(initial_state)
                torch.manual_seed(global_seed)
                generator = torch.Generator()
                generator.manual_seed(31415)
                history = train_behavior_cloning(
                    actor,
                    dataset_path,
                    epochs=3,
                    batch_size=4,
                    lr=1e-3,
                    device='cpu',
                    verbose=False,
                    generator=generator,
                )
                return history, {name: tensor.detach().clone() for name, tensor in actor.state_dict().items()}

            first_history, first_state = run_once(1)
            second_history, second_state = run_once(999)

        self.assertEqual(first_history, second_history)
        self.assertEqual(first_state.keys(), second_state.keys())
        for name in first_state:
            self.assertTrue(torch.equal(first_state[name], second_state[name]), name)

    def test_same_seed_reproduces_cpu_ann_history_and_actor_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / 'tiny_ann.npz'
            scenario = ScenarioConfig()
            state_dim = 5 + 3 + 4 + 4 * scenario.nearest_zone_count
            rng = np.random.default_rng(20260830)
            np.savez(
                dataset_path,
                observations=rng.normal(size=(13, state_dim)).astype(np.float32),
                actions=rng.normal(scale=0.01, size=(13, 2)).astype(np.float32),
            )

            def run_once(run_name: str) -> tuple[dict, dict]:
                run_root = root / run_name
                output = run_root / 'final.pt'
                metrics_name = 'metrics.json'
                argv = [
                    'train_bc',
                    '--dataset',
                    str(dataset_path),
                    '--model',
                    'ann',
                    '--epochs',
                    '3',
                    '--batch-size',
                    '4',
                    '--output',
                    str(output),
                    '--best-output',
                    str(run_root / 'best.pt'),
                    '--metrics-out',
                    metrics_name,
                    '--log-root',
                    str(run_root / 'logs'),
                    '--device',
                    'cpu',
                    '--seed',
                    '2718',
                ]
                with mock.patch.object(sys, 'argv', argv):
                    with redirect_stdout(io.StringIO()):
                        train_bc.main()
                checkpoint = load_checkpoint(output)
                metrics_path = next((run_root / 'logs').rglob(metrics_name))
                metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
                return checkpoint, metrics

            first_checkpoint, first_metrics = run_once('first')
            second_checkpoint, second_metrics = run_once('second')

        self.assertEqual(first_checkpoint['loss_history'], second_checkpoint['loss_history'])
        self.assertEqual(first_checkpoint['state_dict'].keys(), second_checkpoint['state_dict'].keys())
        for name in first_checkpoint['state_dict']:
            self.assertTrue(
                torch.equal(first_checkpoint['state_dict'][name], second_checkpoint['state_dict'][name]),
                name,
            )
        self.assertEqual(first_checkpoint['bc_seed_requested'], 2718)
        self.assertEqual(first_checkpoint['bc_seed_effective'], 2718)
        self.assertEqual(first_metrics['bc_seed_requested'], 2718)
        self.assertEqual(first_metrics['bc_seed_effective'], 2718)
        self.assertEqual(first_metrics['loss_history'], second_metrics['loss_history'])


if __name__ == '__main__':
    unittest.main()
