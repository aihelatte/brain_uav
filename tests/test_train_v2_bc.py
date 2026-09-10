from __future__ import annotations

from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brain_uav.config import TrainingConfig
from brain_uav.models import V2ANNCritic, V2SNNPolicyActor
from brain_uav.observations import collate_v2_observations
from brain_uav.scripts.train_v2_bc import (
    build_parser,
    build_v2_bc_actor,
    train_v2_behavior_cloning,
)
from brain_uav.trainers.v2_bc import (
    V2BCTrainingResult,
    V2_BC_CHECKPOINT_FORMAT,
    V2_SNN_BC_CHECKPOINT_FORMAT,
    load_v2_bc_actor_checkpoint,
    load_v2_snn_bc_actor_checkpoint,
    load_v2_bc_trajectory_cluster,
)
from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer
from brain_uav.trainers.v2_reporting import V2BCTrainingReporter
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from brain_uav.utils.seeding import set_global_seed

from test_v2_bc import make_observation, write_cluster


class TestTrainV2BCScript(unittest.TestCase):
    def test_report_finish_failure_preserves_core_bc_artifacts_and_closes_reporter(self) -> None:
        def fake_train(actor, cluster, split, config, *, epoch_callback=None):
            state = {
                name: value.detach().cpu().clone()
                for name, value in actor.state_dict().items()
            }
            return V2BCTrainingResult(
                train_loss_history=(0.25,),
                validation_loss_history=(0.5,),
                best_epoch=1,
                best_validation_loss=0.5,
                best_state_dict=state,
                final_state_dict=state,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cluster_path = write_cluster(root / 'cluster', zone_counts=(0, 1, 2, 0))
            output = root / 'output'
            core_paths = (
                output / 'bc_v2_ann_best.pt',
                output / 'bc_v2_ann_final.pt',
                output / 'metrics.json',
                output / 'split.json',
            )
            finish_observations = []
            closed = []
            original_close = V2BCTrainingReporter.close

            def fail_finish(reporter):
                finish_observations.append(all(path.is_file() for path in core_paths))
                raise RuntimeError('injected BC report finish failure')

            def tracking_close(reporter):
                original_close(reporter)
                closed.append(reporter._closed)

            with mock.patch(
                'brain_uav.scripts.train_v2_bc.train_v2_bc_actor',
                side_effect=fake_train,
            ), mock.patch.object(
                V2BCTrainingReporter,
                'finish',
                new=fail_finish,
            ), mock.patch.object(
                V2BCTrainingReporter,
                'close',
                new=tracking_close,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, 'injected BC report finish failure'
                ):
                    train_v2_behavior_cloning(
                        trajectory_cluster=cluster_path,
                        output_dir=output,
                        seed=7,
                        validation_fraction=0.25,
                        epochs=1,
                        batch_size=2,
                        learning_rate=1e-3,
                        training_config=TrainingConfig(hidden_dim=8),
                        device='cpu',
                    )

            self.assertEqual(finish_observations, [True])
            self.assertTrue(all(path.is_file() for path in core_paths))
            self.assertEqual(closed, [True])

    def test_parser_supports_ann_and_strict_torch_snn_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            '--trajectory-cluster', 'cluster',
            '--output-dir', 'output',
            '--seed', '9',
            '--validation-fraction', '0.25',
            '--epochs', '3',
            '--batch-size', '4',
            '--lr', '0.0005',
        ])
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.validation_fraction, 0.25)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.lr, 0.0005)
        self.assertEqual(args.model, 'ann')
        self.assertEqual(args.snn_time_window, 4)
        self.assertEqual(args.device, 'auto')
        self.assertEqual(args.shard_cache_mb, 256.0)
        self.assertFalse(args.deduplicate_identical_trajectories)
        with self.assertRaises(SystemExit):
            parser.parse_args([
                '--trajectory-cluster', 'cluster',
                '--output-dir', 'output',
                '--device', 'mps',
            ])
        snn_args = parser.parse_args([
            '--trajectory-cluster', 'cluster',
            '--output-dir', 'output',
            '--model', 'snn',
            '--snn-time-window', '3',
        ])
        self.assertEqual(snn_args.model, 'snn')
        self.assertEqual(snn_args.snn_time_window, 3)

    def test_cache_and_deduplication_options_are_visible_in_metrics_and_split(self) -> None:
        def fake_train(actor, cluster, split, config, *, epoch_callback=None):
            state = {
                name: value.detach().cpu().clone()
                for name, value in actor.state_dict().items()
            }
            return V2BCTrainingResult(
                train_loss_history=(0.25,),
                validation_loss_history=(0.5,),
                best_epoch=1,
                best_validation_loss=0.5,
                best_state_dict=state,
                final_state_dict=state,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cluster_path = write_cluster(
                root / 'cluster', zone_counts=(0, 1, 2, 0),
                trajectories_per_scenario=2,
            )
            with mock.patch(
                'brain_uav.scripts.train_v2_bc.train_v2_bc_actor',
                side_effect=fake_train,
            ):
                metrics = train_v2_behavior_cloning(
                    trajectory_cluster=cluster_path,
                    output_dir=root / 'output',
                    seed=7,
                    validation_fraction=0.25,
                    epochs=1,
                    batch_size=128,
                    learning_rate=1e-3,
                    training_config=TrainingConfig(hidden_dim=8),
                    device='cpu',
                    shard_cache_mb=3.5,
                    deduplicate_identical_trajectories=True,
                )
            split = json.loads(
                (root / 'output' / 'split.json').read_text(encoding='utf-8')
            )

        self.assertEqual(metrics['batch_size'], 128)
        self.assertEqual(metrics['shard_cache_mb'], 3.5)
        self.assertTrue(metrics['deduplicate_identical_trajectories'])
        provenance = metrics['dataset_provenance']
        self.assertEqual(provenance['trajectory_count_before_deduplication'], 8)
        self.assertEqual(provenance['trajectory_count_after_deduplication'], 4)
        self.assertEqual(len(provenance['duplicate_trajectory_mappings']), 4)
        removed = sum(
            split[name]['deduplication']['removed_trajectory_count']
            for name in ('train_statistics', 'validation_statistics')
        )
        self.assertEqual(removed, 4)

    def test_auto_device_is_resolved_before_training_and_recorded(self) -> None:
        captured_configs = []

        def fake_train(actor, cluster, split, config, *, epoch_callback=None):
            captured_configs.append(config)
            state = {
                name: value.detach().cpu().clone()
                for name, value in actor.state_dict().items()
            }
            result = V2BCTrainingResult(
                train_loss_history=(0.25,),
                validation_loss_history=(0.5,),
                best_epoch=1,
                best_validation_loss=0.5,
                best_state_dict=state,
                final_state_dict=state,
            )
            if epoch_callback is not None:
                epoch_callback({
                    'epoch': 1,
                    'epochs': 1,
                    'train_loss': 0.25,
                    'validation_loss': 0.5,
                    'best_epoch': 1,
                    'best_validation_loss': 0.5,
                    'refreshed_best': True,
                })
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cluster_path = write_cluster(root / 'cluster', zone_counts=(0, 1, 2, 0))
            with mock.patch(
                'brain_uav.scripts.train_v2_bc.resolve_training_device',
                return_value='cuda',
            ) as resolver, mock.patch(
                'brain_uav.scripts.train_v2_bc.train_v2_bc_actor',
                side_effect=fake_train,
            ):
                metrics = train_v2_behavior_cloning(
                    trajectory_cluster=cluster_path,
                    output_dir=root / 'output',
                    seed=7,
                    validation_fraction=0.25,
                    epochs=1,
                    batch_size=2,
                    learning_rate=1e-3,
                    training_config=TrainingConfig(hidden_dim=8),
                    device='auto',
                )
            persisted = json.loads(
                (root / 'output' / 'metrics.json').read_text(encoding='utf-8')
            )

        resolver.assert_called_once_with('auto')
        self.assertEqual(str(captured_configs[0].device), 'cuda')
        self.assertEqual(metrics['requested_device'], 'auto')
        self.assertEqual(metrics['resolved_device'], 'cuda')
        self.assertEqual(persisted['requested_device'], 'auto')
        self.assertEqual(persisted['resolved_device'], 'cuda')

    def test_explicit_cpu_device_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cluster_path = write_cluster(root / 'cluster', zone_counts=(0, 1, 2, 0))
            with mock.patch(
                'brain_uav.scripts.train_v2_bc.resolve_training_device',
                wraps=lambda requested: requested,
            ) as resolver:
                metrics = train_v2_behavior_cloning(
                    trajectory_cluster=cluster_path,
                    output_dir=root / 'output',
                    seed=7,
                    validation_fraction=0.25,
                    epochs=1,
                    batch_size=2,
                    learning_rate=1e-3,
                    training_config=TrainingConfig(hidden_dim=8),
                    device='cpu',
                )
        resolver.assert_called_once_with('cpu')
        self.assertEqual(metrics['requested_device'], 'cpu')
        self.assertEqual(metrics['resolved_device'], 'cpu')

    def test_explicit_unavailable_cuda_fails_before_dataset_or_output_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            with mock.patch(
                'brain_uav.scripts.common.torch.cuda.is_available',
                return_value=False,
            ):
                with self.assertRaisesRegex(RuntimeError, 'CUDA was requested'):
                    train_v2_behavior_cloning(
                        trajectory_cluster=root / 'missing-cluster',
                        output_dir=output,
                        device='cuda',
                    )
            self.assertFalse(output.exists())

    def test_seed_controls_actor_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cluster = load_v2_bc_trajectory_cluster(write_cluster(Path(directory)))
            set_global_seed(7)
            first = build_v2_bc_actor(cluster, actor_hidden_dim=16)
            set_global_seed(7)
            second = build_v2_bc_actor(cluster, actor_hidden_dim=16)
            set_global_seed(8)
            different = build_v2_bc_actor(cluster, actor_hidden_dim=16)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name])
        self.assertTrue(any(
            not torch.equal(value, different.state_dict()[name])
            for name, value in first.state_dict().items()
            if value.is_floating_point()
        ))

    def test_seed_controls_snn_actor_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cluster = load_v2_bc_trajectory_cluster(write_cluster(Path(directory)))
            set_global_seed(7)
            first = build_v2_bc_actor(
                cluster, actor_hidden_dim=16, model='snn', snn_time_window=3
            )
            set_global_seed(7)
            second = build_v2_bc_actor(
                cluster, actor_hidden_dim=16, model='snn', snn_time_window=3
            )
        self.assertIsInstance(first, V2SNNPolicyActor)
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]))

    def test_snn_and_ann_share_dataset_split_and_snn_writes_distinct_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cluster_path = write_cluster(root / 'cluster', zone_counts=(0, 1, 2, 0))
            common = dict(
                trajectory_cluster=cluster_path,
                seed=19,
                validation_fraction=0.25,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                training_config=TrainingConfig(hidden_dim=8, bc_epochs=1, batch_size=2),
                device='cpu',
                shard_cache_mb=1.0,
                deduplicate_identical_trajectories=True,
            )
            train_v2_behavior_cloning(output_dir=root / 'ann', model='ann', **common)
            train_v2_behavior_cloning(
                output_dir=root / 'snn',
                model='snn',
                snn_time_window=2,
                **common,
            )
            self.assertEqual(
                json.loads((root / 'ann' / 'split.json').read_text(encoding='utf-8')),
                json.loads((root / 'snn' / 'split.json').read_text(encoding='utf-8')),
            )
            self.assertTrue((root / 'snn' / 'bc_v2_snn_best.pt').is_file())
            self.assertTrue((root / 'snn' / 'bc_v2_snn_final.pt').is_file())
            payload = torch.load(
                root / 'snn' / 'bc_v2_snn_best.pt',
                map_location='cpu',
                weights_only=False,
            )
            self.assertEqual(payload['format'], V2_SNN_BC_CHECKPOINT_FORMAT)
            self.assertEqual(payload['model_type'], 'snn')
            self.assertEqual(payload['architecture']['time_window'], 2)
            self.assertEqual(payload['architecture']['tau'], 2.0)
            self.assertEqual(payload['architecture']['surrogate'], 'atan')
            self.assertEqual(payload['architecture']['backend'], 'torch')

    def test_same_seed_reproduces_complete_v2_bc_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cluster_path = write_cluster(
                root / 'cluster',
                zone_counts=(0, 1, 2, 0),
                trajectories_per_scenario=2,
                shard_scenario_count=2,
            )
            common_arguments = {
                'trajectory_cluster': cluster_path,
                'seed': 73,
                'validation_fraction': 0.25,
                'epochs': 2,
                'batch_size': 2,
                'learning_rate': 1e-3,
                'training_config': TrainingConfig(
                    hidden_dim=16,
                    bc_epochs=2,
                    batch_size=2,
                ),
                'device': 'cpu',
            }
            first_output = root / 'first_output'
            second_output = root / 'second_output'
            train_v2_behavior_cloning(output_dir=first_output, **common_arguments)
            train_v2_behavior_cloning(output_dir=second_output, **common_arguments)

            first_split = json.loads(
                (first_output / 'split.json').read_text(encoding='utf-8')
            )
            second_split = json.loads(
                (second_output / 'split.json').read_text(encoding='utf-8')
            )
            for field in (
                'seed',
                'validation_fraction',
                'train_scenario_ids',
                'validation_scenario_ids',
                'train_statistics',
                'validation_statistics',
            ):
                with self.subTest(split_field=field):
                    self.assertEqual(first_split[field], second_split[field])

            first_metrics = json.loads(
                (first_output / 'metrics.json').read_text(encoding='utf-8')
            )
            second_metrics = json.loads(
                (second_output / 'metrics.json').read_text(encoding='utf-8')
            )
            for field in (
                'train_loss_history',
                'validation_loss_history',
                'best_epoch',
                'best_validation_loss',
            ):
                with self.subTest(metrics_field=field):
                    self.assertEqual(first_metrics[field], second_metrics[field])

            for checkpoint_name in ('bc_v2_ann_best.pt', 'bc_v2_ann_final.pt'):
                with self.subTest(checkpoint=checkpoint_name):
                    first_payload = torch.load(
                        first_output / checkpoint_name,
                        map_location='cpu',
                        weights_only=False,
                    )
                    second_payload = torch.load(
                        second_output / checkpoint_name,
                        map_location='cpu',
                        weights_only=False,
                    )
                    first_state = first_payload['actor_state_dict']
                    second_state = second_payload['actor_state_dict']
                    self.assertEqual(tuple(first_state), tuple(second_state))
                    for name, first_value in first_state.items():
                        self.assertTrue(
                            torch.equal(first_value, second_state[name]),
                            msg=f'{checkpoint_name} tensor {name!r} differs.',
                        )

            first_actor = load_v2_bc_actor_checkpoint(
                first_output / 'bc_v2_ann_best.pt'
            )
            second_actor = load_v2_bc_actor_checkpoint(
                second_output / 'bc_v2_ann_best.pt'
            )
            mixed_batch = collate_v2_observations([
                make_observation(0, 1.0),
                make_observation(1, 2.0),
                make_observation(2, 3.0),
            ])
            with torch.inference_mode():
                first_actions = first_actor(mixed_batch)
                second_actions = second_actor(mixed_batch)
            self.assertTrue(torch.equal(first_actions, second_actions))

    def test_tiny_training_writes_best_final_metrics_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cluster_path = write_cluster(root / 'cluster', zone_counts=(0, 1, 2, 0))
            output = root / 'output'
            metrics = train_v2_behavior_cloning(
                trajectory_cluster=cluster_path,
                output_dir=output,
                seed=7,
                validation_fraction=0.25,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                training_config=TrainingConfig(hidden_dim=16, bc_epochs=1, batch_size=2),
                device='cpu',
            )
            self.assertTrue((output / 'bc_v2_ann_best.pt').is_file())
            self.assertTrue((output / 'bc_v2_ann_final.pt').is_file())
            self.assertTrue((output / 'metrics.json').is_file())
            self.assertTrue((output / 'split.json').is_file())
            self.assertEqual(
                len((output / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()),
                1,
            )
            self.assertTrue((output / 'epochs.csv').is_file())
            self.assertTrue((output / 'mse_curve.png').is_file())
            self.assertEqual(len(metrics['train_loss_history']), 1)
            self.assertEqual(len(metrics['validation_loss_history']), 1)
            self.assertEqual(metrics['best_epoch'], 1)

    def test_nonempty_output_directory_is_rejected_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cluster_path = write_cluster(root / 'cluster')
            output = root / 'output'
            output.mkdir()
            marker = output / 'keep.txt'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaisesRegex(FileExistsError, 'non-empty'):
                train_v2_behavior_cloning(
                    trajectory_cluster=cluster_path,
                    output_dir=output,
                    seed=7,
                    epochs=1,
                    batch_size=2,
                    learning_rate=1e-3,
                    training_config=TrainingConfig(hidden_dim=16),
                )
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')


class TestV2BCCheckpoint(unittest.TestCase):
    def _train_checkpoint(self, root: Path) -> tuple[Path, object]:
        cluster_path = write_cluster(root / 'cluster', zone_counts=(0, 1, 2, 0))
        train_v2_behavior_cloning(
            trajectory_cluster=cluster_path,
            output_dir=root / 'output',
            seed=11,
            validation_fraction=0.25,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            training_config=TrainingConfig(hidden_dim=16, bc_epochs=1, batch_size=2),
            device='cpu',
        )
        return root / 'output' / 'bc_v2_ann_best.pt', load_v2_bc_trajectory_cluster(cluster_path)

    def test_checkpoint_round_trip_preserves_actor_output_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, cluster = self._train_checkpoint(Path(directory))
            payload = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(payload['format'], V2_BC_CHECKPOINT_FORMAT)
            self.assertEqual(payload['checkpoint_kind'], 'best')
            self.assertEqual(payload['dataset_provenance']['manifest'], cluster.manifest)
            restored = load_v2_bc_actor_checkpoint(path)
            direct = build_v2_bc_actor(cluster, actor_hidden_dim=16)
            direct.load_state_dict(payload['actor_state_dict'], strict=True)
            batch = collate_v2_observations([
                make_observation(0, 1.0),
                make_observation(2, 2.0),
            ])
            restored.eval()
            direct.eval()
            with torch.inference_mode():
                torch.testing.assert_close(restored(batch), direct(batch), atol=0.0, rtol=0.0)
            self.assertTrue(all(value.device.type == 'cpu' for value in payload['actor_state_dict'].values()))

    def test_old_flat_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            torch.save({'state_dict': {'weight': torch.zeros(1)}}, path)
            with self.assertRaisesRegex(ValueError, 'format'):
                load_v2_bc_actor_checkpoint(path)

    def test_tampered_architecture_and_action_contracts_are_rejected(self) -> None:
        cases = {
            'observation': lambda p: p['observation_contract'].__setitem__('id', 'flat_v1'),
            'scales': lambda p: p['architecture']['scales'].__setitem__('world_xy', 999.0),
            'action': lambda p: p['action_high'].__setitem__(0, 9.0),
            'uav_radius': lambda p: p['architecture'].__setitem__('uav_radius', 0.5),
            'hidden': lambda p: p['architecture'].__setitem__('actor_hidden_dim', 32),
            'encoder': lambda p: p['architecture']['encoder_config'].__setitem__('num_layers', 1),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self._train_checkpoint(root)
            original = torch.load(path, map_location='cpu', weights_only=False)
            for name, mutate in cases.items():
                with self.subTest(case=name):
                    payload = deepcopy(original)
                    mutate(payload)
                    altered = root / f'{name}.pt'
                    torch.save(payload, altered)
                    with self.assertRaisesRegex(ValueError, 'incompatible|mismatch|architecture|action'):
                        load_v2_bc_actor_checkpoint(altered)

    def test_loading_into_an_incompatible_existing_actor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, cluster = self._train_checkpoint(Path(directory))
            incompatible = build_v2_bc_actor(cluster, actor_hidden_dim=32)
            with self.assertRaisesRegex(ValueError, 'expected_actor.*incompatible'):
                load_v2_bc_actor_checkpoint(
                    checkpoint,
                    expected_actor=incompatible,
                )

    def test_loaded_actor_can_initialize_td3_and_serve_as_bc_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self._train_checkpoint(Path(directory))
            actor = load_v2_bc_actor_checkpoint(checkpoint)
            critic1 = V2ANNCritic(
                actor.scales, actor.action_dim, actor.hidden_dim,
                uav_radius=actor.uav_radius, encoder_config=actor.encoder_config,
            )
            critic2 = V2ANNCritic(
                actor.scales, actor.action_dim, actor.hidden_dim,
                uav_radius=actor.uav_radius, encoder_config=actor.encoder_config,
            )
            replay = V2ReplayBuffer(8, actor.action_dim, 10)
            limit = actor.action_limit.detach().cpu().numpy()
            engine = V2TD3UpdateEngine(
                actor, critic1, critic2, replay,
                actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005,
                policy_noise=0.01, noise_clip=0.02, policy_delay=2, batch_size=2,
                action_low=-limit, action_high=limit,
                bc_reference_actor=actor,
                terminal_geo_regularization_enabled=False,
                device='cpu',
            )
        self.assertIsNotNone(engine.bc_reference_actor)
        self.assertTrue(all(not parameter.requires_grad for parameter in engine.bc_reference_actor.parameters()))

    def test_snn_checkpoint_round_trip_and_cross_model_loaders_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cluster_path = write_cluster(root / 'cluster', zone_counts=(0, 1, 2, 0))
            output = root / 'snn'
            train_v2_behavior_cloning(
                trajectory_cluster=cluster_path,
                output_dir=output,
                seed=23,
                validation_fraction=0.25,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                training_config=TrainingConfig(hidden_dim=8, bc_epochs=1, batch_size=2),
                device='cpu',
                model='snn',
                snn_time_window=2,
            )
            best = output / 'bc_v2_snn_best.pt'
            final = output / 'bc_v2_snn_final.pt'
            restored = load_v2_snn_bc_actor_checkpoint(best)
            self.assertIsInstance(restored, V2SNNPolicyActor)
            batch = collate_v2_observations([
                make_observation(0, 1.0),
                make_observation(1, 2.0),
                make_observation(2, 3.0),
            ])
            payload = torch.load(best, map_location='cpu', weights_only=False)
            direct = build_v2_bc_actor(
                load_v2_bc_trajectory_cluster(cluster_path),
                actor_hidden_dim=8,
                model='snn',
                snn_time_window=2,
            )
            direct.load_state_dict(payload['actor_state_dict'], strict=True)
            with torch.inference_mode():
                self.assertTrue(torch.equal(restored(batch), direct(batch)))
            with self.assertRaisesRegex(ValueError, 'format'):
                load_v2_bc_actor_checkpoint(best)
            with self.assertRaisesRegex(ValueError, 'format|best'):
                load_v2_snn_bc_actor_checkpoint(final)

            ann_path, _ = self._train_checkpoint(root / 'ann_case')
            with self.assertRaisesRegex(ValueError, 'format'):
                load_v2_snn_bc_actor_checkpoint(ann_path)

    def test_snn_loader_rejects_expected_actor_architecture_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cluster_path = write_cluster(root / 'cluster', zone_counts=(0, 1, 2, 0))
            output = root / 'snn'
            train_v2_behavior_cloning(
                trajectory_cluster=cluster_path,
                output_dir=output,
                seed=29,
                validation_fraction=0.25,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                training_config=TrainingConfig(hidden_dim=8, bc_epochs=1, batch_size=2),
                model='snn',
                snn_time_window=2,
                device='cpu',
            )
            cluster = load_v2_bc_trajectory_cluster(cluster_path)
            mismatch = build_v2_bc_actor(
                cluster,
                actor_hidden_dim=8,
                model='snn',
                snn_time_window=3,
            )
            with self.assertRaisesRegex(ValueError, 'architecture'):
                load_v2_snn_bc_actor_checkpoint(
                    output / 'bc_v2_snn_best.pt', expected_actor=mismatch
                )


if __name__ == '__main__':
    unittest.main()
