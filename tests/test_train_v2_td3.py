"""Tests for the standalone formal V2 TD3 stage CLI."""

from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.models import V2ANNPolicyActor
from brain_uav.observations import V2ObservationScales
from brain_uav.scripts.common import resolve_training_device
from brain_uav.scripts.train_v2_td3 import build_parser, run_v2_td3_stage
from brain_uav.trainers.v2_formal_training import (
    V2BCFormalInitialization,
    V2PreparedStageInitialization,
)
from brain_uav.trainers.v2_reporting import V2ExperimentReporter
from brain_uav.v2_curriculum import derive_v2_component_seed


class TestTrainV2TD3CLI(unittest.TestCase):
    def test_report_finish_failure_preserves_core_td3_artifacts_and_closes_reporter(self):
        scenario = ScenarioConfig()
        rewards = RewardConfig()
        actor = V2ANNPolicyActor(
            V2ObservationScales(
                scenario.world_xy,
                scenario.world_z_min,
                scenario.world_z_max,
                scenario.gamma_max,
            ),
            action_dim=2,
            hidden_dim=8,
            action_limit=torch.tensor(
                [scenario.delta_gamma_max, scenario.delta_psi_max],
                dtype=torch.float32,
            ),
        )
        result = SimpleNamespace(
            passed_validation=True,
            stage_steps=1,
            global_steps_end=1,
            outcome_counts={'goal': 1},
            validation_records=[{'passed': True}],
            stop_reason='fixed_validation_passed',
            to_dict=lambda: {
                'passed_validation': True,
                'status': 'passed',
                'stop_reason': 'fixed_validation_passed',
                'stage_steps': 1,
                'global_steps_end': 1,
            },
        )
        pool = SimpleNamespace(
            scenario_count=100,
            curriculum_level='easy',
            master_seed=20260904,
            stage_seed=123,
            content_digest='digest',
        )
        model = SimpleNamespace(parameters=lambda: ())
        components = SimpleNamespace(
            engine=SimpleNamespace(
                actor=model,
                critic1=model,
                critic2=model,
                replay=SimpleNamespace(sampling_implementation='fenwick_ppswor_v1'),
            ),
            scenario_generators={'easy': object()},
            selector=object(),
            exploration_rng=object(),
            seed_manifest={'base_seed': 7},
            initialization_source={'kind': 'v2_bc_best'},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_source = root / 'bc.pt'
            output = root / 'easy.pt'
            metrics = root / 'metrics.json'
            prepared = V2PreparedStageInitialization(
                stage='easy',
                model_seed=derive_v2_component_seed(7, 'easy', 'model'),
                verified_initialization_source=checkpoint_source.resolve(),
                snn_time_window=None,
                scenario_config=scenario,
                reward_config=rewards,
                uav_collision_radius=0.0,
                model_type='ann',
                bc_initialization=V2BCFormalInitialization(
                    actor=actor,
                    scenario_config=scenario,
                    uav_collision_radius=0.0,
                    model_type='ann',
                ),
                formal_checkpoint=None,
                torch_rng_state=torch.get_rng_state().clone(),
            )
            finish_observations = []
            closed = []
            original_close = V2ExperimentReporter.close

            def save_marker(path, payload):
                Path(path).write_bytes(b'checkpoint')

            def fail_finish(reporter, payload):
                finish_observations.append((output.is_file(), metrics.is_file()))
                raise RuntimeError('injected TD3 report finish failure')

            def tracking_close(reporter):
                original_close(reporter)
                closed.append(reporter._closed)

            with mock.patch(
                'brain_uav.scripts.train_v2_td3.prepare_v2_stage_initialization',
                return_value=prepared,
            ), mock.patch(
                'brain_uav.scripts.train_v2_td3.load_v2_validation_pool',
                return_value=pool,
            ), mock.patch(
                'brain_uav.scripts.train_v2_td3.build_v2_stage_engine',
                return_value=components,
            ), mock.patch(
                'brain_uav.scripts.train_v2_td3.V2FormalStageTrainer'
            ) as trainer_type, mock.patch(
                'brain_uav.scripts.train_v2_td3.build_v2_formal_checkpoint',
                return_value={'checkpoint': True},
            ), mock.patch(
                'brain_uav.scripts.train_v2_td3.save_v2_formal_checkpoint',
                side_effect=save_marker,
            ), mock.patch.object(
                V2ExperimentReporter,
                'finish_stage',
                new=fail_finish,
            ), mock.patch.object(
                V2ExperimentReporter,
                'close',
                new=tracking_close,
            ):
                trainer_type.return_value.run.return_value = result
                with self.assertRaisesRegex(
                    RuntimeError, 'injected TD3 report finish failure'
                ):
                    run_v2_td3_stage(
                        stage='easy',
                        init_checkpoint=checkpoint_source,
                        output=output,
                        metrics_out=metrics,
                        validation_pool=root / 'validation.json',
                        device='cpu',
                    )

            self.assertEqual(finish_observations, [(True, True)])
            self.assertTrue(output.is_file())
            self.assertTrue(metrics.is_file())
            self.assertEqual(closed, [True])

    def test_parser_uses_formal_defaults_and_only_v2_stages(self):
        parser = build_parser()
        args = parser.parse_args([
            '--stage', 'easy',
            '--init-checkpoint', 'bc.pt',
            '--output', 'easy.pt',
            '--metrics-out', 'easy.json',
            '--validation-pool', 'easy_validation.json',
        ])
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.device, 'auto')
        self.assertEqual(args.model, 'ann')
        self.assertEqual(args.snn_time_window, 4)
        self.assertEqual(args.early_stop_min_steps, 125_000)
        self.assertEqual(args.window_episodes, 15)
        self.assertEqual(args.consecutive_windows, 4)
        self.assertEqual(args.max_failures_per_window, 1)
        self.assertEqual(args.validation_max_failures, 6)
        with self.assertRaises(SystemExit):
            parser.parse_args([
                '--stage', 'easy_two_zone',
                '--init-checkpoint', 'x',
                '--output', 'x',
                '--metrics-out', 'x',
                '--validation-pool', 'x',
            ])
        snn = parser.parse_args([
            '--stage', 'easy',
            '--init-checkpoint', 'x',
            '--output', 'x',
            '--metrics-out', 'x',
            '--validation-pool', 'x',
            '--model', 'snn',
            '--snn-time-window', '3',
        ])
        self.assertEqual((snn.model, snn.snn_time_window), ('snn', 3))

    def test_shared_device_resolution_is_strict(self):
        with mock.patch(
            'brain_uav.scripts.common.torch.cuda.is_available', return_value=True
        ):
            self.assertEqual(resolve_training_device('auto'), 'cuda')
        with mock.patch(
            'brain_uav.scripts.common.torch.cuda.is_available', return_value=False
        ):
            self.assertEqual(resolve_training_device('auto'), 'cpu')
            self.assertEqual(resolve_training_device('cpu'), 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                resolve_training_device('cuda')

    def test_stage_resolves_auto_before_engine_and_records_both_devices(self):
        scenario = ScenarioConfig(target_distance=1_701.0)
        rewards = RewardConfig(progress_weight=3.25)
        actor = V2ANNPolicyActor(
            V2ObservationScales(
                scenario.world_xy,
                scenario.world_z_min,
                scenario.world_z_max,
                scenario.gamma_max,
            ),
            action_dim=2,
            hidden_dim=8,
            action_limit=torch.tensor(
                [scenario.delta_gamma_max, scenario.delta_psi_max],
                dtype=torch.float32,
            ),
            uav_radius=0.75,
        )
        pool = SimpleNamespace(
            scenario_count=100,
            curriculum_level='easy',
            master_seed=20260904,
            stage_seed=123,
            content_digest='digest',
        )
        model = SimpleNamespace(parameters=lambda: ())
        components = SimpleNamespace(
            engine=SimpleNamespace(
                actor=model,
                critic1=model,
                critic2=model,
                replay=SimpleNamespace(
                    sampling_implementation='fenwick_ppswor_v1'
                ),
            ),
            scenario_generators={'easy': object()},
            selector=object(),
            exploration_rng=object(),
            seed_manifest={'base_seed': 7},
            initialization_source={'kind': 'v2_bc_best'},
        )
        result = SimpleNamespace(
            passed_validation=True,
            stage_steps=1,
            global_steps_end=1,
            outcome_counts={'goal': 1},
            validation_records=[{'passed': True}],
            stop_reason='fixed_validation_passed',
            to_dict=lambda: {
                'passed_validation': True,
                'status': 'passed',
                'stop_reason': 'fixed_validation_passed',
                'stage_steps': 1,
                'global_steps_end': 1,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'bc.pt'
            prepared = V2PreparedStageInitialization(
                stage='easy',
                model_seed=derive_v2_component_seed(7, 'easy', 'model'),
                verified_initialization_source=checkpoint.resolve(),
                snn_time_window=None,
                scenario_config=scenario,
                reward_config=rewards,
                uav_collision_radius=0.75,
                model_type='ann',
                bc_initialization=V2BCFormalInitialization(
                    actor=actor,
                    scenario_config=scenario,
                    uav_collision_radius=0.75,
                    model_type='ann',
                ),
                formal_checkpoint=None,
                torch_rng_state=torch.get_rng_state().clone(),
            )
            with mock.patch(
                'brain_uav.scripts.train_v2_td3.resolve_training_device',
                return_value='cuda',
            ) as resolver, mock.patch(
                'brain_uav.scripts.train_v2_td3.prepare_v2_stage_initialization',
                return_value=prepared,
            ) as initializer, mock.patch(
                'brain_uav.scripts.train_v2_td3.load_v2_validation_pool',
                return_value=pool,
            ), mock.patch(
                'brain_uav.scripts.train_v2_td3.build_v2_stage_engine',
                return_value=components,
            ) as builder, mock.patch(
                'brain_uav.scripts.train_v2_td3.V2FormalStageTrainer'
            ) as trainer_type, mock.patch(
                'brain_uav.scripts.train_v2_td3.build_v2_formal_checkpoint',
                return_value={'checkpoint': True},
            ), mock.patch(
                'brain_uav.scripts.train_v2_td3.save_v2_formal_checkpoint'
            ), mock.patch(
                'brain_uav.scripts.train_v2_td3._write_strict_json'
            ):
                trainer_type.return_value.run.return_value = result
                summary = run_v2_td3_stage(
                    stage='easy',
                    init_checkpoint=checkpoint,
                    output=root / 'easy.pt',
                    metrics_out=root / 'metrics.json',
                    validation_pool=root / 'validation.json',
                    device='auto',
                )

        resolver.assert_called_once_with('auto')
        self.assertEqual(initializer.call_args.kwargs['device'], 'cuda')
        self.assertIsNone(initializer.call_args.kwargs['scenario'])
        self.assertIsNone(initializer.call_args.kwargs['rewards'])
        self.assertIsNone(initializer.call_args.kwargs['uav_collision_radius'])
        self.assertIs(
            builder.call_args.kwargs['prepared_initialization'], prepared
        )
        self.assertEqual(builder.call_args.args[0], scenario)
        self.assertEqual(builder.call_args.kwargs['rewards'], rewards)
        self.assertEqual(
            builder.call_args.kwargs['uav_collision_radius'], 0.75
        )
        self.assertEqual(builder.call_args.kwargs['device'], 'cuda')
        self.assertNotEqual(builder.call_args.kwargs['device'], 'auto')
        self.assertEqual(summary['requested_device'], 'auto')
        self.assertEqual(summary['resolved_device'], 'cuda')
        self.assertEqual(
            summary['replay_sampling_implementation'],
            'fenwick_ppswor_v1',
        )


if __name__ == '__main__':
    unittest.main()
