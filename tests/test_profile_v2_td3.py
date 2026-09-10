from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.models import V2ANNPolicyActor
from brain_uav.observations import V2ObservationScales
from brain_uav.scripts.profile_v2_td3 import (
    DIAGNOSTIC_FORMAT,
    _prepare_diagnostic_pools,
    _run_diagnostic_level,
    build_parser,
    run_v2_td3_timing_diagnostic,
)
from brain_uav.scripts.train_v2_bc import build_v2_bc_actor
from brain_uav.trainers.v2_bc import (
    V2BCTrainingConfig,
    V2BCTrainingResult,
    build_v2_bc_checkpoint_payload,
    load_v2_bc_trajectory_cluster,
    split_v2_bc_scenarios,
)

from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig
from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer

from test_v2_bc import make_scenario_config, make_scenario_payload, write_cluster


class TestProfileV2TD3(unittest.TestCase):
    def run_small_level(self, *, warmup=0, steps=5, early_goal=False,
                        device='cpu', suppress_measured_actor=False):
        # Real environment and replay; only network work and CUDA are test doubles.
        scenario = make_scenario_config()
        scenario.max_steps = 100
        records = []
        for index, count in enumerate((0, 1, 2)):
            payload = make_scenario_payload(index, count)
            payload['goal'] = [30.0, 0.0, 10.0]
            if early_goal and index == 1:
                payload['goal'] = [-3.0, 0.0, 10.0]
            records.append({'scenario_id': f'fixed-{index}', 'payload': payload})
        replay = V2ReplayBuffer(64, 2, 2)
        engine = SimpleNamespace(
            actor=torch.nn.Linear(1, 1), replay=replay, batch_size=4,
            critic_update_count=0, actor_update_count=0,
            set_target_noise=lambda **kwargs: None,
        )
        submitted = []

        def select_action(observation, **kwargs):
            submitted.append(observation)
            return np.zeros(2, dtype=np.float32)

        def update_once(*, total_steps, bc_lambda):
            replay.sample(4)
            engine.critic_update_count += 1
            if total_steps % 2 == 0 and not (
                suppress_measured_actor and total_steps > 7
            ):
                engine.actor_update_count += 1

        engine.select_action = select_action
        engine.update_once = update_once
        synchronization_points = []
        event = mock.Mock()
        event.elapsed_time.return_value = 1.0
        with mock.patch(
            'brain_uav.scripts.profile_v2_td3.build_v2_stage_engine',
            return_value=SimpleNamespace(engine=engine, exploration_rng=np.random.default_rng(7)),
        ), mock.patch(
            'brain_uav.scripts.profile_v2_td3.perf_counter',
            side_effect=lambda: float(len(submitted)),
        ), mock.patch('torch.cuda.Event', return_value=event), mock.patch(
            'torch.cuda.synchronize',
            side_effect=lambda *args: synchronization_points.append(len(submitted)),
        ):
            result = _run_diagnostic_level(
                level='easy',
                pool=SimpleNamespace(stage_seed=101, scenarios=records, scenario_count=3),
                prepared=SimpleNamespace(
                    scenario_config=scenario, reward_config=RewardConfig(),
                    uav_collision_radius=0.0,
                ),
                formal_config=V2FormalTrainingConfig(
                    stage='easy', replay_capacity=64, batch_size=4, actor_freeze_steps=0,
                ),
                bc_checkpoint=Path('unused.pt'), model='ann', snn_time_window=4,
                device=torch.device(device), warmup_steps=warmup, measured_steps=steps,
            )
        return result, replay, synchronization_points

    def test_warmup_reaches_update_minima_and_is_excluded_from_measurement(self) -> None:
        for minimum, actual, critic, actor in ((0, 7, 4, 2), (12, 12, 9, 5)):
            with self.subTest(minimum=minimum):
                result, replay, syncs = self.run_small_level(warmup=minimum)
                self.assertEqual(result['warmup_steps'], actual)
                self.assertEqual(result['warmup_critic_updates'], critic)
                self.assertEqual(result['warmup_actor_updates'], actor)
                self.assertEqual(result['critic_updates'], 5)
                self.assertEqual(result['timing']['total_wall_seconds'], 5.0)
                self.assertEqual(result['actor_updates'], 3 if actual == 7 else 2)
                self.assertEqual(result['timing']['calls']['environment_step_wall_seconds'], 5)
                self.assertEqual(result['timing']['calls']['td3_update_wall_seconds'], 5)
                self.assertEqual(result['timing']['calls']['replay_sample_wall_seconds'], 5)
                self.assertEqual(len(replay), actual + 5)
                self.assertEqual(syncs, [])

    def test_fixed_scenario_fragments_cover_pool_without_fabricated_done_or_success(self) -> None:
        result, replay, _ = self.run_small_level()
        self.assertEqual(result['scenario_coverage'], [
            {'scenario_id': 'fixed-0', 'zone_count': 0, 'measured_steps': 2, 'episodes_completed': 0},
            {'scenario_id': 'fixed-1', 'zone_count': 1, 'measured_steps': 2, 'episodes_completed': 0},
            {'scenario_id': 'fixed-2', 'zone_count': 2, 'measured_steps': 1, 'episodes_completed': 0},
        ])
        np.testing.assert_array_equal(replay.zone_count[7:12], [0, 0, 1, 1, 2])
        self.assertFalse(replay.done[:len(replay)].any())
        self.assertFalse(replay.success[:len(replay)].any())
        self.assertEqual(replay.success_size, 0)
        self.assertEqual(result['timing']['calls']['scenario_reset_wall_seconds'], 3)

    def test_early_goal_repeats_same_scenario_and_drops_previous_partial_episode(self) -> None:
        result, replay, _ = self.run_small_level(early_goal=True)
        self.assertEqual(result['episodes_completed'], 2)
        self.assertEqual(result['scenario_coverage'][1]['episodes_completed'], 2)
        self.assertEqual(result['timing']['calls']['scenario_reset_wall_seconds'], 4)
        np.testing.assert_array_equal(replay.zone_count[7:12], [0, 0, 1, 1, 2])
        np.testing.assert_array_equal(replay.done[7:12, 0], [0, 0, 1, 1, 0])
        self.assertEqual(replay.success_size, 2)
        self.assertEqual(int(replay.success[:len(replay)].sum()), 2)
        np.testing.assert_array_equal(replay.success_zone_count[:2], [1, 1])

    def test_insufficient_scene_budget_fails_before_loading_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'steps_per_level.*scenario_count'):
                run_v2_td3_timing_diagnostic(
                    model='ann', bc_checkpoint=root / 'missing.pt',
                    output_dir=root / 'output', scenario_pool_dir=root / 'pools',
                    steps_per_level=2, scenario_count=3, device='cpu',
                )
            self.assertFalse((root / 'output').exists())
            self.assertFalse((root / 'pools').exists())

    def test_cuda_sync_boundaries_and_stream_interval_semantics(self) -> None:
        result, _, syncs = self.run_small_level(device='cuda')
        self.assertEqual(syncs, [7, 12])
        timing = result['timing']
        self.assertIn('cuda_stream_interval_seconds', timing)
        self.assertNotIn('gpu_event_seconds', timing)
        self.assertIn('host submission gaps and waits', timing['cuda_timing_note'])
        self.assertIn('not pure GPU compute', timing['cuda_timing_note'])
        self.assertEqual(timing['replay_sample_relation'], 'within_td3_update')
        self.assertIn('short fixed-scenario fragments', timing['measurement_note'])

    def test_measurement_without_actor_updates_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'measurement.*critic.*actor'):
            self.run_small_level(suppress_measured_actor=True)

    def test_parser_exposes_bounded_diagnostic_defaults(self) -> None:
        args = build_parser().parse_args([
            '--model', 'ann',
            '--bc-checkpoint', 'bc.pt',
            '--output-dir', 'diagnostic',
            '--scenario-pool-dir', 'pools',
        ])
        self.assertEqual(args.steps_per_level, 256)
        self.assertEqual(args.warmup_steps, 16)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.device, 'auto')
        self.assertEqual(args.snn_time_window, 4)

    def test_unavailable_explicit_cuda_fails_without_output_or_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            with mock.patch(
                'brain_uav.scripts.common.torch.cuda.is_available',
                return_value=False,
            ):
                with self.assertRaisesRegex(RuntimeError, 'CUDA was requested'):
                    run_v2_td3_timing_diagnostic(
                        model='ann',
                        bc_checkpoint=root / 'missing.pt',
                        output_dir=output,
                        scenario_pool_dir=root / 'pools',
                        device='cuda',
                    )
            self.assertFalse(output.exists())

    def test_summary_is_isolated_and_marks_nested_timing_and_nonformal_status(self) -> None:
        scenario = ScenarioConfig()
        actor = V2ANNPolicyActor(
            V2ObservationScales(
                scenario.world_xy,
                scenario.world_z_min,
                scenario.world_z_max,
                scenario.gamma_max,
            ),
            2,
            8,
            torch.tensor(
                [scenario.delta_gamma_max, scenario.delta_psi_max],
                dtype=torch.float32,
            ),
        )
        prepared = SimpleNamespace(
            scenario_config=scenario,
            reward_config=RewardConfig(),
            uav_collision_radius=0.0,
            bc_initialization=SimpleNamespace(actor=actor),
        )
        pools = {
            level: SimpleNamespace(
                content_digest=f'digest-{level}',
                master_seed=20260904,
                stage_seed=index + 1,
                scenario_count=2,
            )
            for index, level in enumerate(('easy', 'medium', 'hard'))
        }
        level_result = {
            'requested_minimum_warmup_steps': 2,
            'warmup_steps': 6,
            'warmup_critic_updates': 5,
            'warmup_actor_updates': 3,
            'measured_steps': 3,
            'scenario_coverage': [
                {'scenario_id': 'fixed-0', 'zone_count': 0, 'measured_steps': 2, 'episodes_completed': 1},
                {'scenario_id': 'fixed-1', 'zone_count': 1, 'measured_steps': 1, 'episodes_completed': 0},
            ],
            'episodes_completed': 1,
            'actor_updates': 1,
            'critic_updates': 2,
            'timing': {
                'total_wall_seconds': 1.0,
                'replay_sample_wall_seconds': 0.1,
                'td3_update_wall_seconds': 0.4,
                'replay_sample_relation': 'within_td3_update',
                'cuda_stream_interval_seconds': None,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'ann-output'
            with mock.patch(
                'brain_uav.scripts.profile_v2_td3._load_diagnostic_initialization',
                return_value=prepared,
            ), mock.patch(
                'brain_uav.scripts.profile_v2_td3._prepare_diagnostic_pools',
                return_value=pools,
            ) as pool_preparer, mock.patch(
                'brain_uav.scripts.profile_v2_td3._run_diagnostic_level',
                return_value=level_result,
            ) as level_runner:
                summary = run_v2_td3_timing_diagnostic(
                    model='ann',
                    bc_checkpoint=root / 'bc.pt',
                    output_dir=output,
                    scenario_pool_dir=root / 'pools',
                    device='cpu',
                    seed=7,
                    steps_per_level=3,
                    warmup_steps=2,
                    batch_size=2,
                    scenario_count=2,
                )
            persisted = json.loads(
                (output / 'diagnostic_summary.json').read_text(encoding='utf-8')
            )

        self.assertEqual(summary, persisted)
        self.assertEqual(summary['format'], DIAGNOSTIC_FORMAT)
        self.assertFalse(summary['formal_stage_passed'])
        self.assertEqual(tuple(summary['levels']), ('easy', 'medium', 'hard'))
        self.assertEqual(
            summary['levels']['easy']['timing']['replay_sample_relation'],
            'within_td3_update',
        )
        self.assertEqual(pool_preparer.call_count, 1)
        self.assertEqual(level_runner.call_count, 3)
        self.assertFalse(any(output.glob('*.pt')))

    def test_existing_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, 'fresh diagnostic'):
                run_v2_td3_timing_diagnostic(
                    model='ann',
                    bc_checkpoint=root / 'missing.pt',
                    output_dir=output,
                    scenario_pool_dir=root / 'pools',
                    device='cpu',
                )

    def test_two_step_cpu_diagnostic_connects_real_environment_replay_and_td3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = ScenarioConfig()
            cluster = load_v2_bc_trajectory_cluster(write_cluster(
                root / 'cluster',
                zone_counts=(0, 0, 0, 0),
                scenario_config=scenario,
            ))
            split = split_v2_bc_scenarios(
                cluster, validation_fraction=0.25, seed=7
            )
            actor = build_v2_bc_actor(cluster, actor_hidden_dim=8)
            state = {
                name: value.detach().cpu().clone()
                for name, value in actor.state_dict().items()
            }
            result = V2BCTrainingResult(
                train_loss_history=(0.0,),
                validation_loss_history=(0.0,),
                best_epoch=1,
                best_validation_loss=0.0,
                best_state_dict=state,
                final_state_dict=state,
            )
            checkpoint = root / 'bc.pt'
            torch.save(build_v2_bc_checkpoint_payload(
                checkpoint_kind='best',
                actor=actor,
                actor_state_dict=state,
                cluster=cluster,
                split=split,
                config=V2BCTrainingConfig(
                    epochs=1,
                    batch_size=2,
                    learning_rate=1e-3,
                    seed=7,
                    validation_fraction=0.25,
                ),
                result=result,
                finished_at='2026-09-10T00:00:00+00:00',
            ), checkpoint)

            summary = run_v2_td3_timing_diagnostic(
                model='ann',
                bc_checkpoint=checkpoint,
                output_dir=root / 'output',
                scenario_pool_dir=root / 'pools',
                device='cpu',
                seed=7,
                steps_per_level=2,
                warmup_steps=0,
                batch_size=1,
                replay_capacity=8,
                scenario_count=1,
            )
            persisted = json.loads(
                (root / 'output' / 'diagnostic_summary.json').read_text(encoding='utf-8')
            )
            with mock.patch(
                'brain_uav.scripts.profile_v2_td3.generate_v2_validation_pool',
                side_effect=AssertionError('existing pools must be reused'),
            ):
                reloaded = _prepare_diagnostic_pools(
                    root / 'pools',
                    scenario=scenario,
                    scenario_count=1,
                    master_seed=20260904,
                    uav_collision_radius=0.0,
                )

        for level in ('easy', 'medium', 'hard'):
            with self.subTest(level=level):
                self.assertEqual(persisted['levels'][level], summary['levels'][level])
                self.assertEqual(summary['levels'][level]['scenario_coverage'], [{
                    'scenario_id': reloaded[level].scenarios[0]['scenario_id'],
                    'zone_count': len(reloaded[level].scenarios[0]['payload']['zones']),
                    'measured_steps': 2,
                    'episodes_completed': 0,
                }])
                self.assertEqual(summary['levels'][level]['measured_steps'], 2)
                self.assertEqual(summary['levels'][level]['critic_updates'], 2)
                self.assertEqual(summary['levels'][level]['actor_updates'], 1)
                self.assertEqual(summary['levels'][level]['warmup_steps'], 4)
                self.assertEqual(summary['levels'][level]['warmup_critic_updates'], 4)
                self.assertEqual(summary['levels'][level]['warmup_actor_updates'], 2)
                self.assertGreater(
                    summary['levels'][level]['timing']['total_wall_seconds'], 0.0
                )
                self.assertEqual(
                    reloaded[level].content_digest,
                    summary['pools'][level]['content_digest'],
                )


if __name__ == '__main__':
    unittest.main()
