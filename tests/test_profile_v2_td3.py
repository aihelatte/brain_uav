from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
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
    UPDATE_TIMING_SECTION_NAMES,
    _UpdateTimingSummary,
    _DetailedUpdateProfiler,
    _compare_compiled_numeric_tensors,
    _compare_localization_stages,
    _compare_optional_numeric_tensor,
    _configure_group_compilation,
    _capture_critic_only_update_stages,
    _fixed_numeric_replay_batch,
    _terminal_numeric_observation_batch,
    _gradient_diagnostic_summary,
    _prepare_diagnostic_pools,
    _run_compiled_numerics_check,
    _run_actor_update_with_rl_gradient_capture,
    _report_and_validate_actor_regularizers,
    _report_and_validate_empty_token_update,
    _report_group_localization,
    _run_grouped_compiled_numerics_diagnostic,
    _zero_zone_sample_count,
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
import test_v2_td3 as v2_td3_tests
import test_v2_snn_td3 as v2_snn_td3_tests


class TestProfileV2TD3(unittest.TestCase):
    def run_small_level(self, *, warmup=0, steps=5, early_goal=False,
                        device='cpu', suppress_measured_actor=False,
                        detailed_profiler_updates=0, profiler_output_dir=None,
                        compile_critic_encoder=False,
                        compile_target_encoders=False,
                        compile_actors=False,
                        frozen_critic_strategy='eager',
                        compile_critic_block=False,
                        compile_target_block=False,
                        dynamo_graph_counts=(10, 10)):
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
        compile_calls = []
        compile_warmup_shapes = []
        target_compile_calls = []
        target_warmup_shapes = []

        def select_action(observation, **kwargs):
            submitted.append(observation)
            return np.zeros(2, dtype=np.float32)

        def update_once(*, total_steps, bc_lambda, timing_recorder=None,
                        profile_sections=False, reuse_shared_relations=True):
            del reuse_shared_relations
            replay.sample(4)
            engine.critic_update_count += 1
            actor_updated = total_steps % 2 == 0 and not (
                suppress_measured_actor and total_steps > 7
            )
            if actor_updated:
                engine.actor_update_count += 1
            if timing_recorder is not None:
                timing_recorder({
                    'replay_sample': 0.01,
                    'batch_preparation': 0.02,
                    'target_forward_and_td_target': 0.03,
                    'online_critic_forward_and_loss': 0.04,
                    'critic_backward': 0.02,
                    'critic_gradient_check_and_clip': 0.01,
                    'critic_optimizer_step': 0.02,
                    'actor_update': 0.06 if actor_updated else 0.0,
                    'target_soft_update': 0.07 if actor_updated else 0.02,
                })
            return SimpleNamespace(actor_updated=actor_updated)

        engine.select_action = select_action
        engine.update_once = update_once
        engine.enable_online_critic_encoder_compile = lambda **kwargs: (
            compile_calls.append(kwargs)
            or ('critic1.zone_set_encoder', 'critic2.zone_set_encoder')
        )
        engine.warmup_online_critic_encoder_compile = lambda batches: (
            compile_warmup_shapes.extend(
                (batch.batch_size, int(batch.zone_features.shape[1]))
                for batch in batches
            )
        )
        engine.enable_target_encoder_compile = lambda **kwargs: (
            target_compile_calls.append(kwargs)
            or (
                'actor_target.zone_set_encoder',
                'critic1_target.zone_set_encoder',
                'critic2_target.zone_set_encoder',
            )
        )
        engine.warmup_target_encoder_compile = lambda batches: (
            target_warmup_shapes.extend(
                (batch.batch_size, int(batch.zone_features.shape[1]))
                for batch in batches
            )
        )
        def configure_compilation(**kwargs):
            enabled_objects = []
            if kwargs['compile_critic_block']:
                enabled_objects.extend((
                    'critic1.full_forward', 'critic2.full_forward',
                    'twin_critic_loss',
                ))
            if kwargs['frozen_critic_strategy'] == 'compiled_no_grad_context':
                enabled_objects.append('critic1.actor_guidance_context')
            if kwargs['compile_target_block']:
                enabled_objects.extend((
                    'actor_target.full_forward', 'critic1_target.full_forward',
                    'critic2_target.full_forward', 'td_target',
                ))
            if kwargs['compile_actors']:
                enabled_objects.extend((
                    'actor.full_forward', 'bc_reference_actor.full_forward',
                ))
            return {
                'enabled_objects': enabled_objects,
                'critic_granularity': (
                    'full_forward_and_loss'
                    if kwargs['compile_critic_block'] else 'eager'
                ),
                'target_granularity': (
                    'full_tensor_block'
                    if kwargs['compile_target_block'] else 'eager'
                ),
                'actor_granularity': (
                    'ann_full_forward_or_snn_encoder'
                    if kwargs['compile_actors'] else 'eager'
                ),
                'frozen_critic_strategy': kwargs['frozen_critic_strategy'],
                'select_action_execution': 'eager',
                'cuda_graph': False,
            }

        engine.configure_compilation = configure_compilation
        engine.warmup_actor_compile = lambda batches: None
        engine.warmup_full_compile = lambda batches: None
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
        ), mock.patch(
            'brain_uav.scripts.profile_v2_td3._dynamo_unique_graph_count',
            side_effect=dynamo_graph_counts,
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
                detailed_profiler_updates=detailed_profiler_updates,
                profiler_output_dir=profiler_output_dir,
                compile_critic_encoder=compile_critic_encoder,
                compile_target_encoders=compile_target_encoders,
                compile_actors=compile_actors,
                frozen_critic_strategy=frozen_critic_strategy,
                compile_critic_block=compile_critic_block,
                compile_target_block=compile_target_block,
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
                self.assertFalse(
                    result['timing'][
                        'total_wall_seconds_includes_detailed_profiler_overhead'
                    ]
                )
                self.assertEqual(
                    result['timing']['throughput_environment_steps_per_second'],
                    1.0,
                )
                self.assertEqual(result['actor_updates'], 3 if actual == 7 else 2)
                self.assertEqual(result['timing']['calls']['environment_step_wall_seconds'], 5)
                self.assertEqual(result['timing']['calls']['td3_update_wall_seconds'], 5)
                self.assertEqual(result['timing']['calls']['replay_sample_wall_seconds'], 5)
                self.assertEqual(len(replay), actual + 5)
                self.assertEqual(syncs, [])

    def test_update_breakdown_classifies_actual_actor_result_and_excludes_warmup(self) -> None:
        result, _, _ = self.run_small_level(warmup=0)
        breakdown = result['timing']['td3_update_breakdown']

        self.assertEqual(breakdown['critic_only']['update_count'], 2)
        self.assertEqual(breakdown['actor_updated']['update_count'], 3)
        self.assertEqual(breakdown['overall_weighted']['update_count'], 5)
        self.assertEqual(
            breakdown['critic_only']['sections']['actor_update']['total_wall_seconds'],
            0.0,
        )
        self.assertAlmostEqual(
            breakdown['actor_updated']['sections']['actor_update']['total_wall_seconds'],
            0.18,
        )
        self.assertEqual(
            breakdown['overall_weighted']['update_count'],
            result['critic_updates'],
        )

    def test_update_summary_has_nonoverlapping_sections_and_weighted_overall(self) -> None:
        summary = _UpdateTimingSummary()
        summary.record(
            actor_updated=False,
            total_wall_seconds=20.0,
            sections=dict(zip(
                UPDATE_TIMING_SECTION_NAMES,
                (1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 2.0, 0.0, 1.0),
            )),
        )
        for multiplier in (1.0, 2.0):
            summary.record(
                actor_updated=True,
                total_wall_seconds=40.0 * multiplier,
                sections=dict(zip(
                    UPDATE_TIMING_SECTION_NAMES,
                    tuple(
                        value * multiplier
                        for value in (2, 3, 4, 5, 2, 1, 3, 7, 3)
                    ),
                )),
            )

        payload = summary.to_dict()
        self.assertEqual(payload['critic_only']['update_count'], 1)
        self.assertEqual(payload['actor_updated']['update_count'], 2)
        self.assertEqual(payload['overall_weighted']['update_count'], 3)
        self.assertEqual(payload['overall_weighted']['total_wall_seconds'], 140.0)
        self.assertAlmostEqual(
            payload['overall_weighted']['average_wall_seconds'], 140.0 / 3.0,
        )
        overall_sections = payload['overall_weighted']['sections']
        self.assertEqual(overall_sections['replay_sample']['total_wall_seconds'], 7.0)
        self.assertEqual(overall_sections['other_uncovered']['total_wall_seconds'], 34.0)
        self.assertAlmostEqual(
            overall_sections['replay_sample']['percent_of_update_wall_seconds'],
            5.0,
        )
        self.assertAlmostEqual(
            sum(
                section['total_wall_seconds']
                for section in overall_sections.values()
            ),
            payload['overall_weighted']['total_wall_seconds'],
        )

    def test_real_cpu_update_is_identical_with_internal_timing_enabled(self) -> None:
        fixture = v2_td3_tests.TestV2TD3(methodName='runTest')
        fixture.setUp()

        def make_engine():
            torch.manual_seed(1234)
            engine = fixture.make_engine(policy_delay=1, terminal_enabled=False)
            fixture.fill_replay(engine)
            engine.replay.rng = np.random.default_rng(4321)
            return engine

        untimed = make_engine()
        timed = make_engine()
        torch.manual_seed(9876)
        untimed_metrics = untimed.update_once(total_steps=1)
        captured = []
        torch.manual_seed(9876)
        with mock.patch(
            'brain_uav.trainers.v2_td3.perf_counter',
            side_effect=(float(index) for index in range(100)),
        ):
            timed_metrics = timed.update_once(
                total_steps=1,
                timing_recorder=captured.append,
            )

        self.assertEqual(timed_metrics, untimed_metrics)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], {
            'replay_sample': 1.0,
            'batch_preparation': 1.0,
            'target_forward_and_td_target': 1.0,
            'online_critic_forward_and_loss': 1.0,
            'critic_backward': 1.0,
            'critic_gradient_check_and_clip': 1.0,
            'critic_optimizer_step': 1.0,
            'actor_update': 3.0,
            'target_soft_update': 2.0,
        })
        for timed_model, untimed_model in (
            (timed.actor, untimed.actor),
            (timed.critic1, untimed.critic1),
            (timed.critic2, untimed.critic2),
            (timed.actor_target, untimed.actor_target),
            (timed.critic1_target, untimed.critic1_target),
            (timed.critic2_target, untimed.critic2_target),
        ):
            fixture.assert_state_dict_equal(
                timed_model.state_dict(),
                untimed_model.state_dict(),
            )
        self.assertEqual(timed.update_count, untimed.update_count)
        self.assertEqual(timed.critic_update_count, untimed.critic_update_count)
        self.assertEqual(timed.actor_update_count, untimed.actor_update_count)
        self.assertEqual(
            timed.critic_target_update_count,
            untimed.critic_target_update_count,
        )

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
        self.assertEqual(args.detailed_profiler_updates, 0)
        self.assertFalse(args.compile_critic_encoder)
        self.assertFalse(args.compile_target_encoders)
        self.assertFalse(args.compile_actors)
        self.assertEqual(args.frozen_critic_strategy, 'eager')
        self.assertFalse(args.compile_critic_block)
        self.assertFalse(args.compile_target_block)
        self.assertFalse(args.check_compiled_numerics)
        self.assertFalse(args.compiled_numerics_only)
        self.assertIsNone(args.compiled_numerics_group)
        enabled = build_parser().parse_args([
            '--model', 'ann', '--bc-checkpoint', 'bc.pt',
            '--output-dir', 'diagnostic', '--scenario-pool-dir', 'pools',
            '--detailed-profiler-updates',
        ])
        self.assertEqual(enabled.detailed_profiler_updates, 16)
        compiled = build_parser().parse_args([
            '--model', 'ann', '--bc-checkpoint', 'bc.pt',
            '--output-dir', 'diagnostic', '--scenario-pool-dir', 'pools',
            '--compile-critic-encoder',
        ])
        self.assertTrue(compiled.compile_critic_encoder)
        extended = build_parser().parse_args([
            '--model', 'ann', '--bc-checkpoint', 'bc.pt',
            '--output-dir', 'diagnostic', '--scenario-pool-dir', 'pools',
            '--compile-critic-encoder', '--compile-target-encoders',
            '--check-compiled-numerics', '--compiled-numerics-only',
        ])
        self.assertTrue(extended.compile_target_encoders)
        self.assertTrue(extended.check_compiled_numerics)
        self.assertTrue(extended.compiled_numerics_only)
        full = build_parser().parse_args([
            '--model', 'ann', '--bc-checkpoint', 'bc.pt',
            '--output-dir', 'diagnostic', '--scenario-pool-dir', 'pools',
            '--compile-actors', '--compile-critic-block',
            '--compile-target-block', '--frozen-critic-strategy',
            'compiled_no_grad_context',
        ])
        self.assertTrue(full.compile_actors)
        self.assertTrue(full.compile_critic_block)
        self.assertTrue(full.compile_target_block)
        self.assertEqual(
            full.frozen_critic_strategy,
            'compiled_no_grad_context',
        )
        grouped = build_parser().parse_args([
            '--model', 'ann', '--bc-checkpoint', 'bc.pt',
            '--output-dir', 'diagnostic', '--scenario-pool-dir', 'pools',
            '--compiled-numerics-group', 'B',
        ])
        self.assertEqual(grouped.compiled_numerics_group, 'B')

    def test_grouped_compile_modes_have_only_declared_warmup_differences(self) -> None:
        expected = {
            'A': ['enable_online', 'warmup_normal'],
            'B': ['enable_online', 'warmup_full'],
            'C': [
                'enable_online', 'warmup_full',
                'enable_target', 'warmup_target',
            ],
        }
        for group, expected_events in expected.items():
            with self.subTest(group=group):
                events = []
                engine = SimpleNamespace(
                    enable_online_critic_encoder_compile=(
                        lambda **kwargs: events.append('enable_online')
                    ),
                    warmup_online_critic_encoder_compile=(
                        lambda batches: events.append('warmup_full')
                    ),
                    enable_target_encoder_compile=(
                        lambda **kwargs: events.append('enable_target')
                    ),
                    warmup_target_encoder_compile=(
                        lambda batches: events.append('warmup_target')
                    ),
                )
                with mock.patch(
                    'brain_uav.scripts.profile_v2_td3.'
                    '_warmup_online_critic_normal_only',
                    side_effect=lambda engine, batches: events.append(
                        'warmup_normal'
                    ),
                ):
                    _configure_group_compilation(engine, group, ('batch',))
                self.assertEqual(events, expected_events)

    def test_compiled_critic_encoder_reports_separate_warmup_and_stable_timing(self) -> None:
        result, replay, _ = self.run_small_level(
            warmup=0,
            steps=5,
            compile_critic_encoder=True,
        )
        compile_info = result['critic_encoder_compile']

        self.assertTrue(compile_info['requested'])
        self.assertEqual(compile_info['enabled_objects'], [
            'critic1.zone_set_encoder',
            'critic2.zone_set_encoder',
        ])
        self.assertEqual(compile_info['backend'], 'inductor')
        self.assertEqual(compile_info['mode'], 'default')
        self.assertTrue(compile_info['fullgraph'])
        self.assertTrue(compile_info['dynamic'])
        self.assertEqual(compile_info['registration_wall_seconds'], 0.0)
        self.assertEqual(compile_info['warmup_wall_seconds'], 0.0)
        self.assertEqual(
            {tuple(shape) for shape in compile_info['warmup_batch_shapes']},
            {(4, 0), (4, 1), (4, 2)},
        )
        self.assertEqual(compile_info['measurement_new_graph_count'], 0)
        self.assertTrue(compile_info['stable_timing'])
        self.assertTrue(compile_info['valid_for_speed_comparison'])
        self.assertEqual(len(replay), result['warmup_steps'] + result['measured_steps'])

    def test_measurement_recompile_marks_compiled_timing_invalid(self) -> None:
        result, _, _ = self.run_small_level(
            warmup=0,
            steps=5,
            compile_critic_encoder=True,
            dynamo_graph_counts=(10, 12),
        )
        compile_info = result['critic_encoder_compile']

        self.assertEqual(compile_info['measurement_new_graph_count'], 2)
        self.assertFalse(compile_info['stable_timing'])
        self.assertFalse(compile_info['valid_for_speed_comparison'])
        self.assertIn('not stable timing', compile_info['measurement_note'])
        self.assertIsNone(
            result['timing']['throughput_environment_steps_per_second']
        )

    def test_target_encoder_compile_extends_objects_and_reuses_warmup_shapes(self) -> None:
        result, replay, _ = self.run_small_level(
            warmup=0,
            steps=5,
            compile_critic_encoder=True,
            compile_target_encoders=True,
        )
        compile_info = result['critic_encoder_compile']

        self.assertEqual(compile_info['enabled_objects'], [
            'critic1.zone_set_encoder',
            'critic2.zone_set_encoder',
            'actor_target.zone_set_encoder',
            'critic1_target.zone_set_encoder',
            'critic2_target.zone_set_encoder',
        ])
        self.assertTrue(compile_info['target_encoders_requested'])
        self.assertEqual(
            {tuple(shape) for shape in compile_info['target_warmup_batch_shapes']},
            {(4, 0), (4, 1), (4, 2)},
        )
        self.assertEqual(compile_info['target_registration_wall_seconds'], 0.0)
        self.assertEqual(compile_info['target_warmup_wall_seconds'], 0.0)
        self.assertEqual(len(replay), result['warmup_steps'] + result['measured_steps'])

    def test_full_compile_metadata_declares_granularity_and_no_cuda_graph(self) -> None:
        result, _, _ = self.run_small_level(
            compile_actors=True,
            frozen_critic_strategy='compiled_no_grad_context',
            compile_critic_block=True,
            compile_target_block=True,
        )
        compile_info = result['critic_encoder_compile']
        self.assertTrue(compile_info['requested'])
        self.assertEqual(
            compile_info['critic_granularity'],
            'full_forward_and_loss',
        )
        self.assertEqual(compile_info['target_granularity'], 'full_tensor_block')
        self.assertEqual(
            compile_info['frozen_critic_strategy'],
            'compiled_no_grad_context',
        )
        self.assertEqual(compile_info['select_action_execution'], 'eager')
        self.assertFalse(compile_info['cuda_graph'])
        self.assertTrue(compile_info['stable_timing'])

    def test_full_critic_only_diagnostic_does_not_enable_target_block(self) -> None:
        result, _, _ = self.run_small_level(compile_critic_block=True)
        compile_info = result['critic_encoder_compile']
        self.assertEqual(
            compile_info['critic_granularity'],
            'full_forward_and_loss',
        )
        self.assertEqual(compile_info['target_granularity'], 'eager')
        self.assertFalse(compile_info['target_block_requested'])
        self.assertNotIn('td_target', compile_info['enabled_objects'])

    def test_compiled_numeric_comparison_accepts_tolerance_and_rejects_mismatch(self) -> None:
        reference = {
            'critic_loss': torch.tensor(1.0),
            'critic1.weight': torch.tensor([1.0, -2.0]),
        }
        close = {
            'critic_loss': torch.tensor(1.0 + 1e-6),
            'critic1.weight': torch.tensor([1.0, -2.0 + 1e-6]),
        }
        result = _compare_compiled_numeric_tensors(
            reference,
            close,
            rtol=1e-4,
            atol=1e-5,
        )
        self.assertEqual(result['tensor_count'], 2)
        self.assertLessEqual(result['maximum_absolute_error'], 1e-5)
        self.assertEqual(result['rtol'], 1e-4)
        self.assertEqual(result['atol'], 1e-5)

        mismatched = dict(close)
        mismatched['critic1.weight'] = torch.tensor([1.0, -1.9])
        with self.assertRaisesRegex(AssertionError, 'critic1.weight'):
            _compare_compiled_numeric_tensors(
                reference,
                mismatched,
                rtol=1e-4,
                atol=1e-5,
            )
        with self.assertRaisesRegex(
            AssertionError,
            'compiled_only=.*gradients.critic1.empty_scene_token',
        ):
            _compare_compiled_numeric_tensors(
                {
                    'gradients.critic1.empty_scene_token.present': torch.tensor(False),
                },
                {
                    'gradients.critic1.empty_scene_token.present': torch.tensor(True),
                    'gradients.critic1.empty_scene_token': torch.zeros(1),
                },
                rtol=1e-4,
                atol=1e-5,
            )

    def test_actor_regularizer_prerequisites_report_before_rejecting_difference(self):
        metrics = SimpleNamespace(
            bc_lambda=1.5,
            bc_loss=0.25,
            terminal_geo_loss=0.5,
            terminal_geo_lambda=3000.0,
        )
        capture = {
            'bc_action_gradient': torch.tensor([[0.25, -0.5]]),
            'terminal_geo_action_gradient': torch.tensor([[0.75, 0.25]]),
        }
        output = StringIO()
        with redirect_stdout(output):
            record = _report_and_validate_actor_regularizers(
                metrics,
                metrics,
                capture,
                capture,
                expected_bc_lambda=1.5,
            )
        self.assertEqual(record['eager']['bc_action_gradient']['state'], 'nonzero')
        self.assertEqual(
            record['compiled']['terminal_geo_action_gradient']['state'],
            'nonzero',
        )

        invalid_cases = {
            'zero_action_gradient': {
                **capture,
                'bc_action_gradient': torch.zeros((1, 2)),
            },
            'disconnected_action_gradient': {
                **capture,
                'bc_action_gradient': None,
            },
            'different_action_gradient': {
                **capture,
                'bc_action_gradient': torch.tensor([[0.25, -0.75]]),
            },
        }
        for name, invalid in invalid_cases.items():
            with self.subTest(name=name):
                output = StringIO()
                with redirect_stdout(output), self.assertRaises(AssertionError):
                    _report_and_validate_actor_regularizers(
                        metrics,
                        metrics,
                        capture,
                        invalid,
                        expected_bc_lambda=1.5,
                    )
                self.assertIn(
                    'compiled_numeric_actor_regularizers',
                    output.getvalue(),
                )

    def test_regularizer_action_gradients_preserve_original_actor_backward(self):
        fixture = v2_td3_tests.TestV2TD3()
        fixture.setUp()
        engine = fixture.make_engine(
            policy_delay=2,
            terminal_enabled=True,
            bc_reference_actor=fixture.make_bc_reference(0.01),
        )
        observation = v2_td3_tests.collate_v2_observations([
            v2_td3_tests._observation(0, scales=fixture.scales),
            v2_td3_tests._observation(7, scales=fixture.scales),
        ])
        terminal_observation = _terminal_numeric_observation_batch(
            observation,
            engine,
        )
        batch = _fixed_numeric_replay_batch(
            terminal_observation,
            action_dim=engine.action_dim,
            device=torch.device('cpu'),
            line_to_goal_safe=True,
        )
        engine.replay.sample = lambda batch_size: batch
        engine.update_once(total_steps=2, bc_lambda=0.0)
        original_actor_terms = engine._compute_actor_loss_terms
        actor_hooks_before = tuple(engine.actor._forward_hooks)
        critic_head_hooks_before = tuple(engine.critic1.head._forward_hooks)
        actor_gradient_calls = [0 for _ in engine.actor.parameters()]
        gradient_handles = []
        for index, parameter in enumerate(engine.actor.parameters()):
            def count_gradient(gradient, slot=index):
                actor_gradient_calls[slot] += 1
                return gradient
            gradient_handles.append(parameter.register_hook(count_gradient))
        try:
            metrics, capture = _run_actor_update_with_rl_gradient_capture(
                engine,
                total_steps=4,
                bc_lambda=1.5,
                capture_regularizers=True,
            )
        finally:
            for handle in gradient_handles:
                handle.remove()

        self.assertGreater(metrics.bc_loss, 0.0)
        self.assertGreater(metrics.terminal_geo_loss, 0.0)
        self.assertTrue(bool(torch.count_nonzero(capture['bc_action_gradient'])))
        self.assertTrue(bool(torch.count_nonzero(
            capture['terminal_geo_action_gradient']
        )))
        self.assertTrue(any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in engine.actor.parameters()
        ))
        self.assertTrue(any(count == 1 for count in actor_gradient_calls))
        self.assertTrue(all(count <= 1 for count in actor_gradient_calls))
        restored_actor_terms = engine._compute_actor_loss_terms
        self.assertIs(restored_actor_terms.__func__, original_actor_terms.__func__)
        self.assertIs(restored_actor_terms.__self__, original_actor_terms.__self__)
        self.assertEqual(tuple(engine.actor._forward_hooks), actor_hooks_before)
        self.assertEqual(
            tuple(engine.critic1.head._forward_hooks),
            critic_head_hooks_before,
        )

        with mock.patch.object(
            engine,
            'update_once',
            side_effect=RuntimeError('controlled diagnostic failure'),
        ), self.assertRaisesRegex(RuntimeError, 'controlled diagnostic failure'):
            _run_actor_update_with_rl_gradient_capture(
                engine,
                total_steps=6,
                bc_lambda=1.5,
                capture_regularizers=True,
            )
        restored_after_error = engine._compute_actor_loss_terms
        self.assertIs(restored_after_error.__func__, original_actor_terms.__func__)
        self.assertEqual(tuple(engine.actor._forward_hooks), actor_hooks_before)
        self.assertEqual(
            tuple(engine.critic1.head._forward_hooks),
            critic_head_hooks_before,
        )

    def test_zero_zone_count_and_gradient_diagnostic_states(self) -> None:
        fixture = v2_td3_tests.TestV2TD3()
        fixture.setUp()
        observation = v2_td3_tests.collate_v2_observations([
            v2_td3_tests._observation(0, scales=fixture.scales),
            v2_td3_tests._observation(3, scales=fixture.scales),
            v2_td3_tests._observation(0, scales=fixture.scales),
        ])

        self.assertEqual(_zero_zone_sample_count(observation), 2)
        self.assertEqual(_gradient_diagnostic_summary(None), {
            'state': 'none',
            'maximum_absolute_value': None,
            'norm': None,
            'finite': None,
        })
        self.assertEqual(_gradient_diagnostic_summary(torch.zeros(3)), {
            'state': 'zero',
            'maximum_absolute_value': 0.0,
            'norm': 0.0,
            'finite': True,
        })
        nonzero = _gradient_diagnostic_summary(torch.tensor([3.0, 4.0]))
        self.assertEqual(nonzero['state'], 'nonzero')
        self.assertEqual(nonzero['maximum_absolute_value'], 4.0)
        self.assertEqual(nonzero['norm'], 5.0)
        self.assertTrue(nonzero['finite'])

    def test_optional_gradient_and_adam_tensor_differences_fail_with_field_name(self) -> None:
        with self.assertRaisesRegex(AssertionError, 'empty_scene_token.gradient'):
            _compare_optional_numeric_tensor(
                torch.zeros(2),
                None,
                name='critic1.empty_scene_token.gradient',
                rtol=1e-4,
                atol=1e-5,
            )

    def test_empty_token_adam_difference_is_reported_before_failure(self) -> None:
        fixture = v2_td3_tests.TestV2TD3()
        fixture.setUp()
        observation = v2_td3_tests.collate_v2_observations([
            v2_td3_tests._observation(0, scales=fixture.scales),
            v2_td3_tests._observation(3, scales=fixture.scales),
        ])
        batch = _fixed_numeric_replay_batch(
            observation,
            action_dim=2,
            device=torch.device('cpu'),
        )

        def state():
            return {
                'parameter': torch.zeros(2),
                'gradient': torch.ones(2),
                'adam_exists': True,
                'adam_step': torch.tensor(1.0),
                'exp_avg': torch.ones(2),
                'exp_avg_sq': torch.ones(2),
            }

        eager_before = {name: state() for name in ('critic1', 'critic2')}
        eager_after = {name: state() for name in ('critic1', 'critic2')}
        compiled_before = {name: state() for name in ('critic1', 'critic2')}
        compiled_after = {name: state() for name in ('critic1', 'critic2')}
        compiled_after['critic1']['exp_avg'] = torch.tensor([1.0, 1.1])
        output = StringIO()

        with redirect_stdout(output), self.assertRaisesRegex(
            AssertionError,
            'critic1.empty_scene_token.exp_avg',
        ):
            _report_and_validate_empty_token_update(
                update_index=2,
                update='injected_optimizer_difference',
                batch_construction='injected_test_batch',
                batch=batch,
                eager_before=eager_before,
                eager_after=eager_after,
                compiled_before=compiled_before,
                compiled_after=compiled_after,
                require_historical_momentum=False,
                rtol=1e-4,
                atol=1e-5,
            )

        record = json.loads(output.getvalue())['compiled_numeric_empty_token']
        self.assertAlmostEqual(
            record['critics']['critic1']['after_differences'][
                'exp_avg_maximum_absolute_difference'
            ],
            0.1,
            places=6,
        )

    def test_group_localization_reports_earliest_difference_before_failure(self) -> None:
        eager = {
            'target_forward': {'actor': torch.zeros(2), 'td_target': torch.ones(1)},
            'online_forward_and_loss': {
                'critic1': torch.zeros(1), 'critic_loss': torch.tensor(1.0),
            },
            'backward_pre_clip_gradients': {'critic1.weight': torch.ones(2)},
            'optimizer_step': {'parameters.critic1.weight': torch.ones(2)},
        }
        compiled = deepcopy(eager)
        compiled['online_forward_and_loss']['critic1'] = torch.ones(1)
        compiled['backward_pre_clip_gradients']['critic1.weight'] = torch.zeros(2)
        comparison = _compare_localization_stages(
            eager,
            compiled,
            rtol=1e-4,
            atol=1e-5,
        )
        self.assertEqual(
            comparison['earliest_difference_stage'],
            'online_forward_and_loss',
        )
        self.assertEqual(
            comparison['stages']['backward_pre_clip_gradients'][
                'difference_count'
            ],
            1,
        )

        output = StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(
            AssertionError,
            'online_forward_and_loss',
        ):
            _report_group_localization('A', eager, compiled)
        record = json.loads(output.getvalue())['compiled_numeric_group_localization']
        self.assertEqual(
            record['earliest_difference_stage'],
            'online_forward_and_loss',
        )
        self.assertEqual(
            record['frozen_critic_actor_encoder_execution'],
            'eager',
        )

    def test_capture_hooks_are_removed_after_success_and_exception(self) -> None:
        fixture = v2_td3_tests.TestV2TD3()
        fixture.setUp()
        engine = fixture.make_engine(policy_delay=2)
        observation = v2_td3_tests.collate_v2_observations([
            v2_td3_tests._observation(0, scales=fixture.scales),
            v2_td3_tests._observation(3, scales=fixture.scales),
        ])
        batch = _fixed_numeric_replay_batch(
            observation,
            action_dim=2,
            device=torch.device('cpu'),
        )
        hooked_modules = (
            engine.actor_target,
            engine.critic1_target,
            engine.critic2_target,
            engine.critic1,
            engine.critic2,
        )
        parameters = tuple(engine.critic1.parameters()) + tuple(
            engine.critic2.parameters()
        )

        captured = _capture_critic_only_update_stages(
            engine,
            batch,
            total_steps=1,
        )
        self.assertIn('td_target', captured['target_forward'])
        self.assertTrue(all(not module._forward_hooks for module in hooked_modules))
        self.assertTrue(all(not parameter._backward_hooks for parameter in parameters))

        with mock.patch.object(
            engine,
            'update_once',
            side_effect=RuntimeError('controlled update failure'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'controlled update failure'):
                _capture_critic_only_update_stages(
                    engine,
                    batch,
                    total_steps=1,
                )
        self.assertTrue(all(not module._forward_hooks for module in hooked_modules))
        self.assertTrue(all(not parameter._backward_hooks for parameter in parameters))

    def test_grouped_diagnostic_uses_independent_equalized_engines(self) -> None:
        fixture = v2_td3_tests.TestV2TD3()
        fixture.setUp()
        eager = fixture.make_engine(policy_delay=2)
        compiled = fixture.make_engine(policy_delay=2)
        observation = v2_td3_tests.collate_v2_observations([
            v2_td3_tests._observation(0, scales=fixture.scales),
            v2_td3_tests._observation(3, scales=fixture.scales),
        ])
        with mock.patch(
            'brain_uav.scripts.profile_v2_td3.build_v2_stage_engine',
            side_effect=(
                SimpleNamespace(engine=eager),
                SimpleNamespace(engine=compiled),
            ),
        ), mock.patch(
            'brain_uav.scripts.profile_v2_td3._compile_warmup_batches',
            return_value=(observation,),
        ), mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), redirect_stdout(StringIO()):
            result = _run_grouped_compiled_numerics_diagnostic(
                group='A',
                pool=SimpleNamespace(),
                prepared=SimpleNamespace(),
                formal_config=V2FormalTrainingConfig(
                    stage='easy', replay_capacity=32, batch_size=2,
                    actor_freeze_steps=0,
                ),
                bc_checkpoint=Path('unused.pt'),
                device=torch.device('cpu'),
                snn_time_window=4,
            )

        self.assertTrue(result['passed'])
        self.assertEqual(
            result['frozen_critic_actor_encoder_execution'],
            'eager',
        )
        self.assertIsNot(eager, compiled)
        self.assertEqual(eager.update_count, 1)
        self.assertEqual(compiled.update_count, 1)
        for eager_model, compiled_model in (
            (eager.actor, compiled.actor),
            (eager.critic1, compiled.critic1),
            (eager.critic2, compiled.critic2),
        ):
            fixture.assert_state_dict_equal(
                eager_model.state_dict(), compiled_model.state_dict()
            )
        with self.assertRaisesRegex(AssertionError, 'exp_avg'):
            _compare_optional_numeric_tensor(
                torch.zeros(2),
                torch.ones(2),
                name='critic1.empty_scene_token.adam.exp_avg',
                rtol=1e-4,
                atol=1e-5,
            )

    def test_grouped_mode_exits_before_regular_timing(self) -> None:
        prepared = SimpleNamespace(
            bc_initialization=SimpleNamespace(actor=torch.nn.Linear(1, 1)),
            scenario_config=make_scenario_config(),
            uav_collision_radius=0.0,
        )
        localization = {
            'requested': True,
            'group': 'B',
            'passed': True,
            'device': 'cuda',
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'group-b'
            with mock.patch(
                'brain_uav.scripts.profile_v2_td3.resolve_training_device',
                return_value='cuda',
            ), mock.patch(
                'brain_uav.scripts.profile_v2_td3._load_diagnostic_initialization',
                return_value=prepared,
            ), mock.patch(
                'brain_uav.scripts.profile_v2_td3._prepare_diagnostic_pools',
                return_value={'easy': SimpleNamespace()},
            ), mock.patch(
                'brain_uav.scripts.profile_v2_td3.'
                '_run_grouped_compiled_numerics_diagnostic',
                return_value=localization,
            ) as group_runner, mock.patch(
                'brain_uav.scripts.profile_v2_td3._run_diagnostic_level',
                side_effect=AssertionError('regular timing must not run'),
            ) as level_runner, redirect_stdout(StringIO()):
                summary = run_v2_td3_timing_diagnostic(
                    model='ann',
                    bc_checkpoint=root / 'bc.pt',
                    output_dir=output,
                    scenario_pool_dir=root / 'pools',
                    device='cuda',
                    steps_per_level=3,
                    scenario_count=3,
                    compiled_numerics_group='B',
                )

            persisted = json.loads(
                (output / 'diagnostic_summary.json').read_text(encoding='utf-8')
            )
        self.assertEqual(summary, persisted)
        self.assertEqual(summary['purpose'], 'compiled_numeric_group_localization_only')
        self.assertEqual(summary['timing_levels_executed'], 0)
        group_runner.assert_called_once()
        level_runner.assert_not_called()

        with self.assertRaisesRegex(ValueError, 'isolated mode'):
            run_v2_td3_timing_diagnostic(
                model='ann',
                bc_checkpoint=Path('unused.pt'),
                output_dir=Path('unused-output'),
                scenario_pool_dir=Path('unused-pools'),
                compiled_numerics_group='A',
                compile_critic_encoder=True,
            )

    def test_compiled_numerics_only_exits_before_regular_timing(self) -> None:
        prepared = SimpleNamespace(
            bc_initialization=SimpleNamespace(actor=torch.nn.Linear(1, 1)),
            scenario_config=make_scenario_config(),
            uav_collision_radius=0.0,
        )
        numerics = {'requested': True, 'passed': True, 'device': 'cuda'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'numerics-only'
            with mock.patch(
                'brain_uav.scripts.profile_v2_td3.resolve_training_device',
                return_value='cuda',
            ), mock.patch(
                'brain_uav.scripts.profile_v2_td3._load_diagnostic_initialization',
                return_value=prepared,
            ), mock.patch(
                'brain_uav.scripts.profile_v2_td3._prepare_diagnostic_pools',
                return_value={'easy': SimpleNamespace()},
            ), mock.patch(
                'brain_uav.scripts.profile_v2_td3._run_compiled_numerics_check',
                return_value=numerics,
            ) as numeric_runner, mock.patch(
                'brain_uav.scripts.profile_v2_td3._run_diagnostic_level',
                side_effect=AssertionError('regular timing must not run'),
            ) as level_runner, redirect_stdout(StringIO()):
                summary = run_v2_td3_timing_diagnostic(
                    model='ann',
                    bc_checkpoint=root / 'bc.pt',
                    output_dir=output,
                    scenario_pool_dir=root / 'pools',
                    device='cuda',
                    steps_per_level=3,
                    scenario_count=3,
                    compile_actors=True,
                    frozen_critic_strategy='compiled_no_grad_context',
                    compile_critic_block=True,
                    compile_target_block=True,
                    check_compiled_numerics=True,
                    compiled_numerics_only=True,
                )

            persisted = json.loads(
                (output / 'diagnostic_summary.json').read_text(encoding='utf-8')
            )
        self.assertEqual(summary, persisted)
        self.assertEqual(summary['purpose'], 'compiled_numeric_correctness_check_only')
        self.assertEqual(summary['timing_levels_executed'], 0)
        numeric_runner.assert_called_once()
        numeric_kwargs = numeric_runner.call_args.kwargs
        self.assertTrue(numeric_kwargs['compile_actors'])
        self.assertEqual(
            numeric_kwargs['frozen_critic_strategy'],
            'compiled_no_grad_context',
        )
        self.assertTrue(numeric_kwargs['compile_critic_block'])
        self.assertTrue(numeric_kwargs['compile_target_block'])
        level_runner.assert_not_called()

        with self.assertRaisesRegex(ValueError, 'requires check_compiled_numerics'):
            run_v2_td3_timing_diagnostic(
                model='ann',
                bc_checkpoint=Path('unused.pt'),
                output_dir=Path('unused-output'),
                scenario_pool_dir=Path('unused-pools'),
                compiled_numerics_only=True,
            )

    def test_compiled_numeric_check_runs_isolated_fixed_updates(self) -> None:
        fixture = v2_td3_tests.TestV2TD3()
        fixture.setUp()
        reference = fixture.make_engine(
            policy_delay=2,
            terminal_enabled=True,
            bc_reference_actor=fixture.make_bc_reference(0.01),
        )
        compiled = fixture.make_engine(
            policy_delay=2,
            terminal_enabled=True,
            bc_reference_actor=fixture.make_bc_reference(0.01),
        )
        batch = v2_td3_tests.collate_v2_observations([
            v2_td3_tests._observation(0, scales=fixture.scales),
            v2_td3_tests._observation(7, scales=fixture.scales),
        ])
        torch.manual_seed(1357)
        np.random.seed(2468)
        torch_rng_before = torch.random.get_rng_state().clone()
        numpy_rng_before = np.random.get_state()

        output = StringIO()
        with mock.patch(
            'brain_uav.scripts.profile_v2_td3.build_v2_stage_engine',
            side_effect=(
                SimpleNamespace(engine=reference),
                SimpleNamespace(engine=compiled),
            ),
        ), mock.patch(
            'brain_uav.scripts.profile_v2_td3._compile_warmup_batches',
            return_value=(batch,),
        ), mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.models.v2_ann.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), redirect_stdout(output):
            result = _run_compiled_numerics_check(
                pool=SimpleNamespace(),
                prepared=SimpleNamespace(),
                formal_config=V2FormalTrainingConfig(
                    stage='easy',
                    replay_capacity=32,
                    batch_size=2,
                    actor_freeze_steps=0,
                ),
                bc_checkpoint=Path('unused.pt'),
                device=torch.device('cpu'),
                snn_time_window=4,
                compile_critic_encoder=False,
                compile_target_encoders=False,
                compile_actors=True,
                frozen_critic_strategy='compiled_no_grad_context',
                compile_critic_block=True,
                compile_target_block=True,
            )

        self.assertTrue(result['passed'])
        self.assertEqual(tuple(result['updates']), (
            'critic_only',
            'actor_and_target_updated',
            'critic_only_after_actor',
            'actor_with_bc_and_terminal_geometry',
            'critic_only_nonempty_after_momentum',
        ))
        self.assertTrue(compiled.actor.compiled_full_forward_enabled)
        self.assertFalse(compiled.actor_target.compiled_full_forward_enabled)
        self.assertIsNotNone(compiled._compiled_target_block)
        self.assertEqual(
            compiled.frozen_critic_strategy,
            'compiled_no_grad_context',
        )
        self.assertEqual(reference.update_count, 5)
        self.assertEqual(compiled.update_count, 5)
        output_records = [
            json.loads(line) for line in output.getvalue().splitlines()
        ]
        records = [
            payload['compiled_numeric_empty_token']
            for payload in output_records
            if 'compiled_numeric_empty_token' in payload
        ]
        self.assertEqual(
            sum('compiled_numeric_actor_rl_gradient' in payload
                for payload in output_records),
            1,
        )
        self.assertEqual(
            [record['update'] for record in records],
            [
                'critic_only_with_empty_scene',
                'actor_and_target_updated_with_empty_scene',
                'critic_only_after_actor_with_empty_scene',
                'actor_with_nonzero_bc_and_terminal_geometry',
                'critic_only_all_nonempty_after_empty_scene_momentum',
            ],
        )
        self.assertEqual(
            [record['update_index'] for record in records],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(records[0]['batch_construction'], (
            'synthetic_from_fixed_pool_with_one_zero_zone_sample'
        ))
        self.assertEqual(
            [record['zero_zone_sample_count'] for record in records],
            [1, 1, 1, 1, 0],
        )
        self.assertTrue(records[0]['historical_momentum_prerequisite_met'])
        self.assertTrue(records[3]['historical_momentum_prerequisite_met'])
        self.assertEqual(
            result['regularizer_gradient_check']['bc_lambda'],
            1.5,
        )
        for execution in ('eager', 'compiled'):
            regularizers = result['regularizer_gradient_check'][execution]
            self.assertGreater(regularizers['bc_loss'], 0.0)
            self.assertEqual(regularizers['bc_action_gradient']['state'], 'nonzero')
            self.assertGreater(regularizers['terminal_geo_loss'], 0.0)
            self.assertEqual(
                regularizers['terminal_geo_action_gradient']['state'],
                'nonzero',
            )
        for execution in ('eager', 'compiled'):
            rl_gradient = result['actor_rl_gradient'][execution]
            self.assertEqual(rl_gradient['q_output_gradient']['state'], 'nonzero')
            self.assertGreater(rl_gradient['nonzero_actor_parameter_gradients'], 0)
        for critic_name in ('critic1', 'critic2'):
            for execution in ('eager', 'compiled'):
                self.assertEqual(
                    records[0]['critics'][critic_name][execution]['before'][
                        'gradient'
                    ]['state'],
                    'none',
                )
                self.assertEqual(
                    records[0]['critics'][critic_name][execution]['after'][
                        'gradient'
                    ]['state'],
                    'nonzero',
                )
                self.assertEqual(
                    records[0]['critics'][critic_name][execution]['after'][
                        'adam'
                    ]['exp_avg']['state'],
                    'nonzero',
                )
                self.assertEqual(
                    records[1]['critics'][critic_name][execution]['before'][
                        'gradient'
                    ]['state'],
                    'nonzero',
                )
                self.assertEqual(
                    records[1]['critics'][critic_name][execution]['after'][
                        'gradient'
                    ]['state'],
                    'nonzero',
                )
                self.assertEqual(
                    records[2]['critics'][critic_name][execution]['after'][
                        'gradient'
                    ]['state'],
                    'nonzero',
                )
                final_gradient_state = records[4]['critics'][critic_name][
                    execution
                ]['after']['gradient']['state']
                self.assertIn(final_gradient_state, ('none', 'zero'))
                other_execution = 'compiled' if execution == 'eager' else 'eager'
                self.assertEqual(
                    final_gradient_state,
                    records[4]['critics'][critic_name][other_execution][
                        'after'
                    ]['gradient']['state'],
                )
                self.assertEqual(
                    records[0]['critics'][critic_name][execution]['after'][
                        'adam'
                    ]['step'],
                    1.0,
                )
                self.assertEqual(
                    records[1]['critics'][critic_name][execution]['after'][
                        'adam'
                    ]['step'],
                    2.0,
                )
                self.assertEqual(
                    records[2]['critics'][critic_name][execution]['after'][
                        'adam'
                    ]['step'],
                    3.0,
                )
                self.assertEqual(
                    records[4]['critics'][critic_name][execution]['after'][
                        'adam'
                    ]['step'],
                    5.0,
                )
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng_before)
        numpy_rng_after = np.random.get_state()
        self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
        np.testing.assert_array_equal(numpy_rng_after[1], numpy_rng_before[1])
        self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])

    def test_snn_compiled_numeric_check_uses_encoder_and_ann_critic_blocks(self):
        fixture = v2_snn_td3_tests.TestV2SNNTD3()
        fixture.setUp()
        reference = fixture.make_engine(bc=fixture.make_actor())
        compiled = fixture.make_engine(bc=fixture.make_actor())
        for engine in (reference, compiled):
            engine.policy_delay = 2
            engine.terminal_geo_regularization_enabled = True
        batch = v2_snn_td3_tests.collate_v2_observations([
            v2_snn_td3_tests._observation(0, fixture.scales),
            v2_snn_td3_tests._observation(10, fixture.scales),
        ])
        with mock.patch(
            'brain_uav.scripts.profile_v2_td3.build_v2_stage_engine',
            side_effect=(
                SimpleNamespace(engine=reference),
                SimpleNamespace(engine=compiled),
            ),
        ), mock.patch(
            'brain_uav.scripts.profile_v2_td3._compile_warmup_batches',
            return_value=(batch,),
        ), mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), redirect_stdout(StringIO()):
            result = _run_compiled_numerics_check(
                pool=SimpleNamespace(),
                prepared=SimpleNamespace(),
                formal_config=V2FormalTrainingConfig(
                    stage='easy',
                    replay_capacity=32,
                    batch_size=2,
                    actor_freeze_steps=0,
                ),
                bc_checkpoint=Path('unused.pt'),
                device=torch.device('cpu'),
                snn_time_window=2,
                model='snn',
                compile_critic_encoder=False,
                compile_target_encoders=False,
                compile_actors=True,
                frozen_critic_strategy='compiled_no_grad_context',
                compile_critic_block=True,
                compile_target_block=True,
            )
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['updates']), 5)
        self.assertIn(
            'actor.zone_set_encoder',
            result['compilation']['enabled_objects'],
        )
        self.assertIn(
            'actor_target.eager_snn',
            result['compilation']['enabled_objects'],
        )
        self.assertEqual(reference.actor.snn_head.lif1.v, 0.0)
        self.assertEqual(compiled.actor_target.snn_head.lif2.v, 0.0)

    def test_compile_extension_rejects_unsupported_flag_combinations(self) -> None:
        common = {
            'model': 'ann',
            'bc_checkpoint': Path('missing.pt'),
            'output_dir': Path('unused-output'),
            'scenario_pool_dir': Path('missing-pools'),
            'device': 'cpu',
        }
        with self.assertRaisesRegex(ValueError, 'requires compile_critic_encoder'):
            run_v2_td3_timing_diagnostic(
                **common,
                compile_target_encoders=True,
            )
        with self.assertRaisesRegex(ValueError, 'requires compile_target_encoders'):
            run_v2_td3_timing_diagnostic(
                **common,
                compile_critic_encoder=True,
                check_compiled_numerics=True,
            )
        with self.assertRaisesRegex(ValueError, 'requires a CUDA diagnostic'):
            run_v2_td3_timing_diagnostic(
                **common,
                compile_critic_encoder=True,
                compile_target_encoders=True,
                check_compiled_numerics=True,
            )
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            run_v2_td3_timing_diagnostic(
                **common,
                compile_critic_encoder=True,
                compile_critic_block=True,
            )
        with self.assertRaisesRegex(ValueError, 'requires compile_critic_block'):
            run_v2_td3_timing_diagnostic(
                **common,
                compile_target_block=True,
            )
        with self.assertRaisesRegex(ValueError, 'compiled critic path'):
            run_v2_td3_timing_diagnostic(
                **common,
                frozen_critic_strategy='compiled_no_grad_context',
            )

    def test_compile_and_detailed_profiler_combination_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            with self.assertRaisesRegex(ValueError, 'cannot be combined'):
                run_v2_td3_timing_diagnostic(
                    model='ann',
                    bc_checkpoint=Path(directory) / 'missing.pt',
                    output_dir=output,
                    scenario_pool_dir=Path(directory) / 'pools',
                    device='cpu',
                    detailed_profiler_updates=1,
                    compile_critic_encoder=True,
                )
            self.assertFalse(output.exists())

    def test_detailed_profiler_excludes_warmup_caps_updates_and_isolates_outputs(self) -> None:
        class FakeAverages:
            def table(self, *, sort_by, row_limit):
                return f'{sort_by}:{row_limit}'

        class FakeProfiler:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def toggle_collection_dynamic(self, enabled, activities):
                pass

            def step(self):
                pass

            def key_averages(self):
                return FakeAverages()

            def export_chrome_trace(self, path):
                Path(path).write_text('{}', encoding='utf-8')

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            'brain_uav.scripts.profile_v2_td3.torch.profiler.profile',
            return_value=FakeProfiler(),
        ):
            output = Path(directory) / 'profiler' / 'easy'
            result, _, _ = self.run_small_level(
                warmup=0,
                steps=5,
                device='cuda',
                detailed_profiler_updates=2,
                profiler_output_dir=output,
            )
            details = result['detailed_profiler']
            self.assertTrue(details['enabled'])
            self.assertEqual(details['requested_updates'], 2)
            self.assertEqual(details['captured_updates'], 2)
            self.assertGreater(result['warmup_critic_updates'], 0)
            self.assertIn('profiler adds overhead', details['measurement_note'])
            timing = result['timing']
            self.assertEqual(timing['total_wall_seconds'], 5.0)
            self.assertTrue(
                timing['total_wall_seconds_includes_detailed_profiler_overhead']
            )
            self.assertIsNone(timing['throughput_environment_steps_per_second'])
            self.assertIn('detailed profiler', timing['total_wall_seconds_note'])
            self.assertIn('normal training throughput', timing['total_wall_seconds_note'])
            self.assertIn('optimization speedup comparisons', timing['total_wall_seconds_note'])
            for path in details['output_paths'].values():
                resolved = Path(path).resolve()
                self.assertTrue(resolved.is_file())
                self.assertTrue(resolved.is_relative_to(output.resolve()))

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
        update_breakdown = _UpdateTimingSummary()
        update_breakdown.record(
            actor_updated=False,
            total_wall_seconds=0.15,
            sections=dict(zip(
                UPDATE_TIMING_SECTION_NAMES,
                (0.01, 0.01, 0.02, 0.03, 0.02, 0.01, 0.01, 0.0, 0.01),
            )),
        )
        update_breakdown.record(
            actor_updated=True,
            total_wall_seconds=0.25,
            sections=dict(zip(
                UPDATE_TIMING_SECTION_NAMES,
                (0.01, 0.01, 0.03, 0.04, 0.02, 0.01, 0.02, 0.07, 0.02),
            )),
        )
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
            'detailed_profiler': {
                'enabled': False,
                'requested_updates': 0,
                'captured_updates': 0,
                'output_paths': {},
                'measurement_note': 'profiler adds overhead',
            },
            'timing': {
                'total_wall_seconds': 1.0,
                'replay_sample_wall_seconds': 0.1,
                'td3_update_wall_seconds': 0.4,
                'td3_update_breakdown': update_breakdown.to_dict(),
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
        self.assertEqual(
            summary['levels']['easy']['timing']['td3_update_breakdown'],
            update_breakdown.to_dict(),
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
