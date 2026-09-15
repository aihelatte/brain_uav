"""Tests for formal V2 easy-to-hard curriculum orchestration."""

from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.models import V2ANNPolicyActor, V2SNNPolicyActor
from brain_uav.observations import V2ObservationScales
from brain_uav.scripts.run_v2_td3_curriculum import (
    build_parser,
    prepare_v2_validation_pools,
    run_v2_curriculum,
)
from brain_uav.trainers.v2_validation import derive_validation_stage_seed
from brain_uav.trainers.v2_formal_training import (
    V2BCFormalInitialization,
    V2PreparedStageInitialization,
)
from brain_uav.v2_curriculum import derive_v2_component_seed


def _make_snn_actor(
    scenario: ScenarioConfig,
    *,
    time_window: int,
    tau: float,
) -> V2SNNPolicyActor:
    return V2SNNPolicyActor(
        V2ObservationScales(
            world_xy=scenario.world_xy,
            world_z_min=scenario.world_z_min,
            world_z_max=scenario.world_z_max,
            gamma_max=scenario.gamma_max,
        ),
        action_dim=2,
        hidden_dim=8,
        action_limit=torch.tensor(
            [scenario.delta_gamma_max, scenario.delta_psi_max],
            dtype=torch.float32,
        ),
        time_window=time_window,
        tau=tau,
    )


def _prepared_initialization(
    scenario: ScenarioConfig,
    *,
    actor=None,
    uav_collision_radius: float = 0.0,
    source: Path | None = None,
):
    if actor is None:
        actor = V2ANNPolicyActor(
            V2ObservationScales(
                world_xy=scenario.world_xy,
                world_z_min=scenario.world_z_min,
                world_z_max=scenario.world_z_max,
                gamma_max=scenario.gamma_max,
            ),
            action_dim=2,
            hidden_dim=8,
            action_limit=torch.tensor(
                [scenario.delta_gamma_max, scenario.delta_psi_max],
                dtype=torch.float32,
            ),
            uav_radius=uav_collision_radius,
        )
    model_type = 'snn' if isinstance(actor, V2SNNPolicyActor) else 'ann'
    return V2PreparedStageInitialization(
        stage='easy',
        model_seed=derive_v2_component_seed(7, 'easy', 'model'),
        verified_initialization_source=(source or Path('bc.pt')).resolve(),
        snn_time_window=actor.time_window if model_type == 'snn' else None,
        scenario_config=scenario,
        reward_config=RewardConfig(),
        uav_collision_radius=uav_collision_radius,
        model_type=model_type,
        bc_initialization=V2BCFormalInitialization(
            actor=actor,
            scenario_config=scenario,
            uav_collision_radius=uav_collision_radius,
            model_type=model_type,
        ),
        formal_checkpoint=None,
        torch_rng_state=torch.get_rng_state().clone(),
    )


class TestRunV2TD3CurriculumCLI(unittest.TestCase):
    def test_parser_defaults_to_one_easy_medium_hard_chain(self):
        args = build_parser().parse_args([
            '--bc-checkpoint', 'bc.pt',
            '--output-root', 'run',
            '--validation-pool-dir', 'validation',
        ])
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.validation_seed, 20260904)
        self.assertEqual(args.max_stage, 'hard')
        self.assertEqual(args.device, 'auto')
        self.assertEqual(args.model, 'ann')
        self.assertEqual(args.snn_time_window, 4)
        self.assertFalse(args.compile_actors)
        self.assertEqual(args.frozen_critic_strategy, 'eager')
        self.assertFalse(args.compile_critic_block)
        self.assertFalse(args.compile_target_block)
        self.assertFalse(args.compile_shared_relations)
        self.assertFalse(args.compile_snn_target_encoder)
        self.assertFalse(args.fused_adam)
        self.assertFalse(args.compile_actor_loss)
        self.assertFalse(args.aggregate_relation_values_first)

    def test_snn_curriculum_uses_distinct_outputs_and_forwards_model_contract(self):
        calls = []

        def fake_stage(**kwargs):
            calls.append(kwargs)
            return {
                'stage': kwargs['stage'],
                'passed': True,
                'checkpoint': str(kwargs['output']),
                'metrics': str(kwargs['metrics_out']),
                'steps': 1,
                'global_steps_end': 1,
                'outcome_counts': {'goal': 1},
                'validation_result': {'passed': True},
                'stop_reason': 'fixed_validation_passed',
            }

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkpoint = base / 'bc_snn.pt'
            checkpoint.write_bytes(b'test-double')
            scenario = ScenarioConfig()
            actor = _make_snn_actor(scenario, time_window=3, tau=3.25)
            prepared = _prepared_initialization(
                scenario, actor=actor, source=checkpoint
            )
            pool_path = base / 'validation' / 'easy.json'
            pool = SimpleNamespace(
                curriculum_level='easy',
                master_seed=20260904,
                stage_seed=derive_validation_stage_seed(20260904, 'easy'),
                scenario_count=100,
                content_digest='digest',
            )
            with mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.prepare_v2_stage_initialization',
                return_value=prepared,
            ) as initializer, mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.prepare_v2_validation_pools',
                return_value={'easy': pool_path, 'medium': pool_path, 'hard': pool_path},
            ), mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.load_v2_validation_pool',
                return_value=pool,
            ):
                result = run_v2_curriculum(
                    bc_checkpoint=checkpoint,
                    output_root=base / 'run',
                    validation_pool_dir=base / 'validation',
                    max_stage='easy',
                    device='cpu',
                    model='snn',
                    snn_time_window=3,
                    stage_runner=fake_stage,
                    compile_actors=True,
                    frozen_critic_strategy='compiled_no_grad_context',
                    compile_critic_block=True,
                    compile_target_block=True,
                    compile_shared_relations=True,
                    compile_snn_target_encoder=True,
                    fused_adam=True,
                    compile_actor_loss=True,
                    aggregate_relation_values_first=True,
                )
        initializer.assert_called_once()
        self.assertEqual(initializer.call_args.kwargs['init_checkpoint'], checkpoint)
        self.assertEqual(initializer.call_args.kwargs['device'], 'cpu')
        self.assertEqual(initializer.call_args.kwargs['model_type'], 'snn')
        self.assertEqual(calls[0]['model'], 'snn')
        self.assertEqual(calls[0]['snn_time_window'], 3)
        self.assertTrue(calls[0]['compile_actors'])
        self.assertEqual(
            calls[0]['frozen_critic_strategy'],
            'compiled_no_grad_context',
        )
        self.assertTrue(calls[0]['compile_critic_block'])
        self.assertTrue(calls[0]['compile_target_block'])
        self.assertTrue(calls[0]['compile_shared_relations'])
        self.assertTrue(calls[0]['compile_snn_target_encoder'])
        self.assertTrue(calls[0]['fused_adam'])
        self.assertTrue(calls[0]['compile_actor_loss'])
        self.assertTrue(calls[0]['aggregate_relation_values_first'])
        self.assertIs(calls[0]['prepared_initialization'], prepared)
        self.assertEqual(Path(calls[0]['output']).name, 'v2_snn_td3_easy.pt')
        self.assertEqual(
            Path(calls[0]['metrics_out']).name, 'v2_snn_td3_easy_metrics.json'
        )
        self.assertEqual(result['format'], 'v2_formal_snn_td3_curriculum_summary')
        self.assertEqual(result['model_type'], 'snn')
        self.assertTrue(result['compilation_request']['compile_shared_relations'])
        self.assertTrue(result['compilation_request']['compile_snn_target_encoder'])
        self.assertTrue(result['compilation_request']['fused_adam'])
        self.assertTrue(result['compilation_request']['compile_actor_loss'])
        self.assertTrue(result['compilation_request']['aggregate_relation_values_first'])
        self.assertEqual(result['snn'], {
            'time_window': actor.time_window,
            'tau': actor.tau,
            'surrogate': actor.surrogate_name,
            'backend': actor.backend,
        })

    def test_snn_time_window_mismatch_fails_before_any_output_or_validation_work(self):
        scenario = ScenarioConfig()
        actor = _make_snn_actor(scenario, time_window=5, tau=3.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'bc_snn.pt'
            checkpoint.write_bytes(b'test-double')
            output_root = root / 'run'
            pool_path = root / 'validation' / 'easy.json'
            pool = SimpleNamespace(
                curriculum_level='easy',
                master_seed=20260904,
                stage_seed=derive_validation_stage_seed(20260904, 'easy'),
                scenario_count=100,
                content_digest='digest',
            )
            with mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.prepare_v2_stage_initialization',
                return_value=_prepared_initialization(
                    scenario, actor=actor, source=checkpoint
                ),
            ), mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.prepare_v2_validation_pools',
                return_value={
                    'easy': pool_path,
                    'medium': pool_path,
                    'hard': pool_path,
                },
            ) as prepare, mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.load_v2_validation_pool',
                return_value=pool,
            ), mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.run_v2_td3_stage',
                return_value={
                    'stage': 'easy',
                    'passed': True,
                    'checkpoint': str(output_root / 'v2_snn_td3_easy.pt'),
                    'metrics': str(output_root / 'v2_snn_td3_easy_metrics.json'),
                    'steps': 1,
                    'global_steps_end': 1,
                    'outcome_counts': {'goal': 1},
                    'validation_result': {'passed': True},
                    'stop_reason': 'fixed_validation_passed',
                },
            ) as stage_runner:
                with self.assertRaisesRegex(
                    ValueError,
                    'checkpoint.*5.*requested.*3',
                ):
                    run_v2_curriculum(
                        bc_checkpoint=checkpoint,
                        output_root=output_root,
                        validation_pool_dir=root / 'validation',
                        max_stage='easy',
                        device='cpu',
                        model='snn',
                        snn_time_window=3,
                        stage_runner=stage_runner,
                    )
            self.assertFalse(output_root.exists())
            self.assertFalse((output_root / 'summary.json').exists())
        prepare.assert_not_called()
        stage_runner.assert_not_called()

    def test_curriculum_hands_checkpoint_forward_and_stops_on_failure(self):
        calls = []

        def fake_stage(**kwargs):
            calls.append(kwargs)
            stage = kwargs['stage']
            passed = stage != 'medium'
            return {
                'stage': stage,
                'passed': passed,
                'checkpoint': str(kwargs['output']),
                'metrics': str(kwargs['metrics_out']),
                'steps': 3,
                'outcome_counts': {'goal': 1},
                'validation_result': {'passed': passed},
                'stop_reason': 'fixed_validation_passed' if passed else 'max_steps_without_validation',
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            validation = Path(directory) / 'validation'
            bc_checkpoint = Path(directory) / 'bc.pt'
            bc_checkpoint.write_bytes(b'test-double')
            scenario = ScenarioConfig(max_steps=17, target_distance=900.0)
            actual_validation_seed = 20260905

            def loaded_pool(path, **kwargs):
                stage = Path(path).stem
                stage = next(level for level in ('easy', 'medium', 'hard') if level in stage)
                return SimpleNamespace(
                    curriculum_level=stage,
                    master_seed=actual_validation_seed,
                    stage_seed=derive_validation_stage_seed(actual_validation_seed, stage),
                    scenario_count=100,
                    content_digest=f'{stage}-digest',
                )

            with mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.prepare_v2_stage_initialization',
                return_value=_prepared_initialization(
                    scenario,
                    uav_collision_radius=0.25,
                    source=bc_checkpoint,
                ),
            ) as contract_loader, mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.prepare_v2_validation_pools',
                return_value={
                    'easy': validation / 'easy.json',
                    'medium': validation / 'medium.json',
                    'hard': validation / 'hard.json',
                },
            ) as prepare, mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.load_v2_validation_pool',
                side_effect=loaded_pool,
            ), mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.resolve_training_device',
                return_value='cpu',
            ):
                summary = run_v2_curriculum(
                    bc_checkpoint=bc_checkpoint,
                    output_root=root,
                    validation_pool_dir=validation,
                    seed=7,
                    validation_seed=20260904,
                    max_stage='hard',
                    device='cpu',
                    stage_runner=fake_stage,
                )

        contract_loader.assert_called_once()
        self.assertEqual(prepare.call_args.args[1], scenario)
        self.assertEqual(prepare.call_args.kwargs['uav_collision_radius'], 0.25)
        self.assertEqual([call['stage'] for call in calls], ['easy', 'medium'])
        self.assertEqual(calls[0]['init_checkpoint'], Path(directory) / 'bc.pt')
        self.assertEqual(calls[1]['init_checkpoint'], Path(calls[0]['output']))
        self.assertEqual(calls[0]['scenario'], scenario)
        self.assertEqual(calls[0]['uav_collision_radius'], 0.25)
        self.assertIsNotNone(calls[0]['prepared_initialization'])
        self.assertIsNone(calls[1]['prepared_initialization'])
        self.assertEqual(calls[0]['device'], 'cpu')
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['failed_stage'], 'medium')
        self.assertEqual(len(summary['stages']), 2)
        self.assertEqual(summary['validation_seed'], actual_validation_seed)
        self.assertEqual(summary['requested_device'], 'cpu')
        self.assertEqual(summary['resolved_device'], 'cpu')
        self.assertIsNone(summary['snn'])
        self.assertEqual(
            summary['validation_pools']['easy']['master_seed'],
            actual_validation_seed,
        )

    def test_existing_validation_pool_is_loaded_with_command_seed_contract(self):
        scenario = ScenarioConfig()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            easy_path = root / 'v2_validation_easy.json'
            easy_path.write_text('{}', encoding='utf-8')
            with mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.load_v2_validation_pool',
                side_effect=ValueError('Validation pool master_seed is incompatible.'),
            ) as loader:
                with self.assertRaisesRegex(ValueError, 'master_seed'):
                    prepare_v2_validation_pools(
                        root,
                        scenario,
                        validation_seed=81,
                        scenario_count=100,
                    )
        self.assertEqual(loader.call_args.kwargs['expected_master_seed'], 81)
        self.assertEqual(
            loader.call_args.kwargs['expected_stage_seed'],
            derive_validation_stage_seed(81, 'easy'),
        )

    def test_auto_device_is_resolved_before_stage_runner(self):
        calls = []

        def fake_stage(**kwargs):
            calls.append(kwargs)
            return {
                'stage': kwargs['stage'],
                'passed': True,
                'checkpoint': str(kwargs['output']),
                'metrics': str(kwargs['metrics_out']),
                'steps': 1,
                'global_steps_end': 1,
                'outcome_counts': {'goal': 1},
                'validation_result': {'passed': True},
                'stop_reason': 'fixed_validation_passed',
            }

        scenario = ScenarioConfig()
        master_seed = 20260904
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bc = root / 'bc.pt'
            bc.write_bytes(b'test-double')
            paths = {
                stage: root / f'{stage}.json'
                for stage in ('easy', 'medium', 'hard')
            }

            def load_pool(path, **kwargs):
                stage = Path(path).stem
                return SimpleNamespace(
                    curriculum_level=stage,
                    master_seed=master_seed,
                    stage_seed=derive_validation_stage_seed(master_seed, stage),
                    scenario_count=100,
                    content_digest=f'{stage}-digest',
                )

            with mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.prepare_v2_stage_initialization',
                return_value=_prepared_initialization(scenario, source=bc),
            ), mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.prepare_v2_validation_pools',
                return_value=paths,
            ), mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.load_v2_validation_pool',
                side_effect=load_pool,
            ), mock.patch(
                'brain_uav.scripts.run_v2_td3_curriculum.resolve_training_device',
                return_value='cuda',
            ):
                summary = run_v2_curriculum(
                    bc_checkpoint=bc,
                    output_root=root / 'run',
                    validation_pool_dir=root / 'pools',
                    max_stage='easy',
                    device='auto',
                    stage_runner=fake_stage,
                )

        self.assertEqual(calls[0]['device'], 'cuda')
        self.assertNotEqual(calls[0]['device'], 'auto')
        self.assertEqual(summary['requested_device'], 'auto')
        self.assertEqual(summary['resolved_device'], 'cuda')

    def test_curriculum_has_no_candidate_racing_or_automatic_seed_change(self):
        parser = build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn('--candidate-seeds', option_strings)
        self.assertNotIn('--num-candidates', option_strings)


if __name__ == '__main__':
    unittest.main()
