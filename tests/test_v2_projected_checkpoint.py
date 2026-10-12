"""Synthetic checkpoint round trips; no BC or RL training is executed."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch
from brain_uav.models.v2_policy_config import V2SpikeRegularizationConfig
from brain_uav.scripts.train_v2_bc import build_v2_bc_actor
from brain_uav.trainers import v2_bc as bc
from brain_uav.trainers import v2_formal_training as formal
from brain_uav.config import RewardConfig
from test_v2_bc import write_cluster
import test_v2_bc as bc_fixture
from test_diagnose_v2_policy import write_inputs
from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_inputs, run_policy_diagnostic


class TestProjectedCheckpoint(unittest.TestCase):
    def test_curriculum_forwards_one_config_to_preparation_and_all_stages(self):
        from types import SimpleNamespace
        from test_run_v2_td3_curriculum import _prepared_initialization
        from brain_uav.scripts import run_v2_td3_curriculum as curriculum
        from brain_uav.models import V2SNNPolicyActor
        from brain_uav.config import ScenarioConfig
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'bc.pt'
            path.write_bytes(b'mocked_input_only')
            scenario = ScenarioConfig()
            prepared = _prepared_initialization(scenario, source=path)
            old_actor = prepared.bc_initialization.actor
            actor = V2SNNPolicyActor(old_actor.scales, 2, 192, old_actor.action_limit, projection_dim=32)
            prepared = _prepared_initialization(scenario, source=path, actor=actor)
            spike = V2SpikeRegularizationConfig(beta_max=1.)
            paths = {stage:root/f'{stage}.json' for stage in ('easy', 'medium', 'hard')}
            def pool(path, **kwargs):
                return SimpleNamespace(master_seed=20260904, stage_seed=7,
                                       scenario_count=100, content_digest='mock')
            calls = []
            def runner(**kwargs):
                calls.append(kwargs)
                return {'passed':True, 'checkpoint':str(kwargs['output']), 'steps':1}
            with (mock.patch.object(curriculum, 'prepare_v2_stage_initialization', return_value=prepared) as preparation,
                 mock.patch.object(curriculum, 'prepare_v2_validation_pools', return_value=paths),
                 mock.patch.object(curriculum, 'load_v2_validation_pool', side_effect=pool)):
                summary = curriculum.run_v2_curriculum(bc_checkpoint=path, output_root=root/'out',
                    validation_pool_dir=root/'pools', model='snn', device='cpu', stage_runner=runner,
                    actor_projection_dim=32, actor_hidden_dim=192, spike_regularization=spike)
            self.assertEqual(preparation.call_args.args[0].spike_regularization, spike)
            self.assertEqual([call['stage'] for call in calls], ['easy', 'medium', 'hard'])
            self.assertTrue(all(call['spike_regularization'] == spike for call in calls))
            self.assertTrue(all(call['actor_hidden_dim'] == 192 for call in calls))
            self.assertEqual(calls[1]['init_checkpoint'], calls[0]['output'])
            self.assertEqual(calls[2]['init_checkpoint'], calls[1]['output'])
            self.assertEqual(summary['spike_regularization'], spike.to_dict())

    def test_bc_easy_medium_hard_structure_and_regularization_inheritance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_scenario_fixture = bc_fixture.make_scenario_payload
            def scenario_fixture(index, count):
                payload = original_scenario_fixture(index, count)
                payload['metadata'].update(requested_direct_path_blocker=False,
                    direct_path_blocker_count=0, feasibility_check='direct_safe_corridor')
                return payload
            with mock.patch('test_v2_bc.make_scenario_payload', side_effect=scenario_fixture):
                cluster = bc.load_v2_bc_trajectory_cluster(write_cluster(root/'cluster', zone_counts=(0, 1, 2, 0)))
            split = bc.split_v2_bc_scenarios(cluster, validation_fraction=.25, seed=7)
            for model in ('ann', 'snn'):
                for projection in (None, 32):
                    with self.subTest(model=model, projection=projection):
                        actor = build_v2_bc_actor(cluster, actor_hidden_dim=192 if projection else 8,
                                                  actor_projection_dim=projection, model=model)
                        state = actor.state_dict()
                        result = bc.V2BCTrainingResult((.1,), (.1,), 1, .1, state, state)
                        builder = bc.build_v2_bc_checkpoint_payload if model == 'ann' else bc.build_v2_snn_bc_checkpoint_payload
                        loader = bc.load_v2_bc_actor_checkpoint if model == 'ann' else bc.load_v2_snn_bc_actor_checkpoint
                        payload = builder(checkpoint_kind='best', actor=actor, actor_state_dict=state,
                            cluster=cluster, split=split, config=bc.V2BCTrainingConfig(epochs=1, batch_size=2, learning_rate=.001, seed=7, validation_fraction=.25),
                            result=result, finished_at='2026-10-12T00:00:00Z')
                        path = root/f'{model}_{projection}.pt'
                        torch.save(payload, path)
                        restored = loader(path)
                        self.assertEqual(restored.projection_dim, projection)
                        for name, value in state.items():
                            torch.testing.assert_close(restored.state_dict()[name], value)
                        if projection:
                            bad = deepcopy(payload)
                            bad['architecture'].pop('policy_head')
                            torch.save(bad, path)
                            with self.assertRaises(ValueError):
                                loader(path)
                            torch.save(payload, path)
                        spike = V2SpikeRegularizationConfig(beta_max=1. if model == 'snn' else 0.)
                        source = path
                        for stage in ('easy', 'medium', 'hard'):
                            config = formal.V2FormalTrainingConfig(stage=stage, max_steps=2,
                                batch_size=2, replay_capacity=8, spike_regularization=spike if stage=='easy' else None)
                            components = formal.build_v2_stage_engine(cluster.scenario_config, config,
                                init_checkpoint=source, model_type=model)
                            engine = components.engine
                            self.assertEqual(engine.actor.projection_dim, projection)
                            self.assertEqual(engine.critic1.hidden_dim, 128 if projection else 8)
                            self.assertEqual(engine.critic2.hidden_dim, engine.critic1.hidden_dim)
                            self.assertEqual(engine.spike_regularization, spike)
                            self.assertEqual(engine.actor_update_count, 0)
                            self.assertEqual(len(engine.replay), 0)
                            for name, value in engine.actor.state_dict().items():
                                torch.testing.assert_close(engine.bc_reference_actor.state_dict()[name], value)
                            result = formal.V2FormalTrainingResult.empty(stage, global_steps_start=0)
                            result.status = 'passed'
                            result.passed_validation = True
                            result.validation_records = [{'passed': True}]
                            source = formal.build_v2_formal_checkpoint(engine, result, config,
                                scenario=cluster.scenario_config, rewards=RewardConfig(), uav_collision_radius=0.,
                                seed_manifest={'model':7}, validation_pool_metadata={
                                    'path':'fixture_pool.json', 'format_version':1, 'curriculum_level':stage,
                                    'master_seed':20260904, 'stage_seed':7, 'scenario_count':100, 'content_digest':'synthetic_fixture'},
                                initialization_source=components.initialization_source)
                            self.assertEqual(source['formal_config']['spike_regularization'], spike.to_dict())
                            source = formal.load_v2_formal_checkpoint(source)
                            if stage == 'easy':
                                failed = deepcopy(source)
                                failed['status'] = 'failed'
                                failed['passed_validation'] = False
                                with self.assertRaises(ValueError):
                                    formal.build_v2_stage_engine(cluster.scenario_config,
                                        replace(config, stage='medium'), init_checkpoint=failed, model_type=model)
                                if model == 'snn':
                                    with self.assertRaises(ValueError):
                                        formal.build_v2_stage_engine(cluster.scenario_config,
                                            replace(config, stage='medium', spike_regularization=V2SpikeRegularizationConfig()),
                                            init_checkpoint=source, model_type=model)
                            snapshot = formal.build_v2_periodic_snapshot(engine, config, stage_steps=1,
                                scenario=cluster.scenario_config, rewards=RewardConfig(), uav_collision_radius=0.,
                                seed_manifest={'model':7}, initialization_source=components.initialization_source)
                            formal.load_v2_periodic_snapshot(snapshot)
                        bad = deepcopy(source)
                        bad['engine_checkpoint']['architecture']['policy_head'] = {'version': 2}
                        with self.assertRaises(ValueError):
                            formal.load_v2_formal_checkpoint(bad)

    def test_projected_diagnostic_reconstruction_and_smoke(self):
        from brain_uav.models import V2SNNPolicyActor
        from brain_uav.trainers import V2TD3UpdateEngine
        from test_v2_formal_training import _engine
        from brain_uav.config import ScenarioConfig
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, pool, payload = write_inputs(root)
            legacy = deepcopy(payload)
            legacy['engine_checkpoint']['algorithm_config'].pop('spike_regularization')
            legacy['formal_config'].pop('spike_regularization')
            normalized = formal.load_v2_formal_checkpoint(legacy)
            self.assertEqual(normalized['formal_config']['spike_regularization']['beta_max'], 0.)
            self.assertNotIn('spike_regularization', legacy['formal_config'])
            bad_legacy = deepcopy(legacy)
            bad_legacy['formal_config'].pop('gamma')
            with self.assertRaises(ValueError):
                formal.load_v2_formal_checkpoint(bad_legacy)
            old = _engine(ScenarioConfig(max_steps=2))
            periodic = formal.build_v2_periodic_snapshot(old, formal.V2FormalTrainingConfig(stage='easy'),
                stage_steps=1, scenario=ScenarioConfig(max_steps=2), rewards=RewardConfig(),
                uav_collision_radius=0., seed_manifest={'model':7},
                initialization_source=payload['initialization_source'])
            periodic['engine_checkpoint']['algorithm_config'].pop('spike_regularization')
            periodic['formal_config'].pop('spike_regularization')
            normalized_periodic = formal.load_v2_periodic_snapshot(periodic)
            self.assertEqual(normalized_periodic['formal_config']['spike_regularization']['beta_max'], 0.)
            actor = V2SNNPolicyActor(old.actor.scales, 2, 192, old.actor.action_limit, projection_dim=32)
            engine = V2TD3UpdateEngine(actor, old.critic1, old.critic2, old.replay,
                .001, .001, .99, .005, .01, .02, 2, 2, old.action_low, old.action_high,
                spike_regularization=V2SpikeRegularizationConfig(beta_max=1.))
            payload['format'] = formal.V2_SNN_FORMAL_CHECKPOINT_FORMAT
            payload['model_type'] = 'snn'
            payload['initialization_source']['model_type'] = 'snn'
            payload['engine_checkpoint'] = engine.checkpoint_state_dict()
            payload['formal_config']['spike_regularization'] = engine.spike_regularization.to_dict()
            torch.save(payload, checkpoint)
            loaded = load_diagnostic_inputs(model='snn', checkpoint=checkpoint, validation_pool=pool, device='cpu')
            self.assertEqual(loaded['actor'].projection_dim, 32)
            with mock.patch('brain_uav.scripts.diagnose_v2_policy.export_v2_trajectory_views', return_value={}):
                summary = run_policy_diagnostic(model='snn', checkpoint=checkpoint, validation_pool=pool,
                    output_dir=root/'diagnostic', device='cpu', max_scenes=1, count_operations=True)
            self.assertEqual(summary['spike_activity']['full_trajectories']['decision_count'], 1)
            with mock.patch('brain_uav.scripts.diagnose_v2_policy.export_v2_trajectory_views', return_value={}):
                other = run_policy_diagnostic(model='snn', checkpoint=checkpoint, validation_pool=pool,
                    output_dir=root/'activity_only', device='cpu', max_scenes=1)
            self.assertEqual(summary['spike_activity'], other['spike_activity'])
            # The existing second diagnostic also reconstructs projection strictly.
            from brain_uav.scripts.diagnose_discount_horizon import load_actor_and_critics
            models = load_actor_and_critics(checkpoint, model_type='snn')
            self.assertEqual(models['actor'].projection_dim, 32)
            self.assertEqual(models['actor_target'].projection_dim, 32)
