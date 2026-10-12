"""Controlled local optimization checks: no formal training or data generation."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from brain_uav.trainers.v2_td3 import V2TD3UpdateMetrics, _actor_loss_tensor_block
from brain_uav.observations import collate_v2_observations
import test_v2_td3 as ann_fixture
from test_v2_td3 import _observation
import test_v2_spike_regularization as spike_fixture


class TestLocalOptimizations(unittest.TestCase):
    def compare_tree(self, first, other):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, other, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), other.keys())
            for key in first:
                self.compare_tree(first[key], other[key])
        elif isinstance(first, (list, tuple)):
            for a, b in zip(first, other):
                self.compare_tree(a, b)
        else:
            self.assertEqual(first, other)

    def engine(self, model):
        if model == 'snn':
            engine = spike_fixture.TestSpikeRegularization().engine(freeze=25000, delay=2)
        else:
            fixture = ann_fixture.TestV2TD3(); fixture.setUp()
            engine = fixture.make_engine(actor_freeze_steps=25000, policy_delay=2)
            fixture.fill_replay(engine)
        engine.reduce_update_stat_syncs = True
        return engine

    def test_all_combinations_exact_updates_gradients_optimizers_and_rng(self):
        # No arithmetic changes: exact equality, rather than forgiving numerical drift.
        for model in ('ann', 'snn'):
            for compiled in (False, True):
                torch.manual_seed(9)
                engines = [self.engine(model) for _ in range(4)]
                initial = deepcopy(engines[0].checkpoint_state_dict())
                for engine, (window, buffers) in zip(engines, ((False,False),(True,False),(False,True),(True,True))):
                    engine.load_checkpoint_state_dict(initial)
                    engine.window_update_statistics = window
                    engine.reuse_update_buffers = buffers
                    if compiled:
                        engine._compiled_actor_loss = torch.compile(_actor_loss_tensor_block, backend='eager', fullgraph=True)
                for step, zones in zip((1,25000,25001,50000,75000,75001), (0,1,5,2,0,3)):
                    sampled = engines[0].replay.sample(2)
                    obs = collate_v2_observations([_observation(zones, scales=engines[0].actor.scales),
                                                   _observation(max(0,zones-1), scales=engines[0].actor.scales)])
                    batch = replace(sampled, obs=obs, next_obs=obs)
                    original = batch.obs.zone_features.clone()
                    noise = torch.full_like(batch.action, .03)
                    metrics = []
                    rng = torch.random.get_rng_state()
                    for engine in engines:
                        torch.random.set_rng_state(rng)
                        def fixed_noise(tensor, **kwargs):
                            return noise.clone()
                        with mock.patch.object(engine.replay, 'sample', return_value=batch), mock.patch('torch.randn_like', new=fixed_noise):
                            result = engine.update_once(total_steps=step, defer_statistics=engine.window_update_statistics)
                        engine.flush_update_statistics()
                        metrics.append(asdict(result))
                    for engine, result in zip(engines[1:], metrics[1:]):
                        self.compare_tree(metrics[0], result)
                        self.compare_tree(engines[0].checkpoint_state_dict(), engine.checkpoint_state_dict())
                        for first_model, other_model in zip(
                            (engines[0].actor, engines[0].critic1, engines[0].critic2),
                            (engine.actor, engine.critic1, engine.critic2)):
                            for left, right in zip(first_model.parameters(), other_model.parameters()):
                                self.compare_tree(left.grad, right.grad)
                        self.assertFalse(engine._pending_update_metrics)
                    torch.testing.assert_close(original, batch.obs.zone_features, rtol=0, atol=0)
                    if model == 'snn' and metrics[0]['actor_updated']:
                        self.assertEqual(metrics[0]['spike_beta_eff'], .5 if step == 50000 else 1.)
                self.assertEqual(engines[0].actor_update_count, 2)
                if model == 'snn':
                    self.assertEqual(metrics[0]['spike_beta_eff'], None) # last step critic-only

    def test_full_tail_windows_no_graph_no_overwrite_and_checkpoint_flush(self):
        engine = self.engine('ann'); engine.window_update_statistics = True; engine.reuse_update_buffers = True
        saved = []
        for i in range(65):
            engine._defer_statistics = True
            value = engine._read_update_statistics((torch.tensor(float(i), requires_grad=True),))[0]
            self.assertIsNone(value.grad_fn)
            metric = V2TD3UpdateMetrics(critic_loss=value)
            engine._pending_update_metrics.append(metric); saved.append(metric)
            if len(engine._pending_update_metrics) == 32:
                engine.flush_update_statistics()
            self.assertLess(len(engine._pending_update_metrics), 32)
        self.assertEqual(len(engine._pending_update_metrics), 1)
        engine.checkpoint_state_dict()
        self.assertEqual([m.critic_loss for m in saved], list(map(float, range(65))))
        self.assertEqual(np.mean([m.critic_loss for m in saved]), 32.)
        engine.flush_update_statistics() # idempotent
        self.assertFalse(engine._pending_update_metrics)
        engine._defer_statistics = False
        buffers = list(engine._statistic_buffers.values())
        for dtype in (torch.float32, torch.float64):
            values = engine._read_update_statistics((torch.tensor(2., dtype=dtype), torch.tensor(3.,dtype=dtype)))
            self.assertEqual(values, (2.,3.))
        self.assertEqual(buffers[0].numel(), 1)
        engine._read_update_statistics(tuple(torch.tensor(1.) for _ in range(10)))
        key = (10, torch.device('cpu'), torch.float32)
        address = engine._statistic_buffers[key].data_ptr()
        engine._read_update_statistics(tuple(torch.tensor(2.) for _ in range(16)))
        engine._read_update_statistics(tuple(torch.tensor(3.) for _ in range(10)))
        self.assertEqual(engine._statistic_buffers[key].data_ptr(), address)

    def test_actual_pending_updates_survive_following_updates(self):
        torch.manual_seed(11)
        first=self.engine('snn'); other=self.engine('snn')
        other.load_checkpoint_state_dict(deepcopy(first.checkpoint_state_dict()))
        other.window_update_statistics=True; other.reuse_update_buffers=True
        batch=first.replay.sample(2); expected=[]; pending=[]
        for step in range(75000,75005):
            rng=torch.random.get_rng_state()
            with mock.patch.object(first.replay,'sample',return_value=batch):
                expected.append(asdict(first.update_once(total_steps=step)))
            torch.random.set_rng_state(rng)
            with mock.patch.object(other.replay,'sample',return_value=batch):
                pending.append(other.update_once(total_steps=step,defer_statistics=True))
        self.assertEqual(len(other._pending_update_metrics),5)
        other.checkpoint_state_dict()
        self.assertEqual(expected,[asdict(value) for value in pending])
        self.compare_tree(first.checkpoint_state_dict(),other.checkpoint_state_dict())

    def test_curriculum_flags_reach_each_stage(self):
        from brain_uav.scripts import run_v2_td3_curriculum as curriculum
        from brain_uav.config import ScenarioConfig
        from test_run_v2_td3_curriculum import _prepared_initialization
        from types import SimpleNamespace
        from brain_uav.trainers.v2_validation import derive_validation_stage_seed
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'bc.pt'; checkpoint.write_bytes(b'mocked BC')
            calls=[]
            def runner(**kwargs):
                calls.append(kwargs)
                return {'stage':kwargs['stage'],'passed':True,'checkpoint':str(kwargs['output']),
                        'steps':1,'global_steps_end':kwargs['global_steps_start']+1}
            pools={stage:root/f'{stage}.json' for stage in ('easy','medium','hard')}
            def load(path,**kwargs):
                stage=Path(path).stem
                return SimpleNamespace(master_seed=20260904,stage_seed=derive_validation_stage_seed(20260904,stage),
                    scenario_count=100,content_digest=stage)
            with mock.patch.object(curriculum,'prepare_v2_stage_initialization',
                  return_value=_prepared_initialization(ScenarioConfig(),source=checkpoint)), \
                 mock.patch.object(curriculum,'prepare_v2_validation_pools',return_value=pools), \
                 mock.patch.object(curriculum,'load_v2_validation_pool',side_effect=load):
                summary=curriculum.run_v2_curriculum(bc_checkpoint=checkpoint,output_root=root/'out',
                    validation_pool_dir=root/'pools',device='cpu',stage_runner=runner,
                    window_update_statistics=True,reuse_update_buffers=True)
            self.assertEqual(len(calls),3)
            self.assertTrue(all(call['window_update_statistics'] and call['reuse_update_buffers'] for call in calls))

    def test_flags_defaults_and_stage_cli(self):
        from brain_uav.scripts import train_v2_td3 as stage, run_v2_td3_curriculum as course
        engine = self.engine('ann')
        self.assertFalse(engine.window_update_statistics)
        self.assertFalse(engine.reuse_update_buffers)
        from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
        for name in ('window_update_statistics','reuse_update_buffers'):
            with self.assertRaises(TypeError):
                V2TD3UpdateEngine(engine.actor,engine.critic1,engine.critic2,engine.replay,
                    .001,.001,.99,.005,.1,.05,2,2,engine.action_low,engine.action_high,**{name:1})
        with self.assertRaises(ValueError):
            engine.update_once(total_steps=1, defer_statistics=True)
        for parser in (stage.build_parser(), course.build_parser()):
            dests = {action.dest: action for action in parser._actions}
            self.assertFalse(dests['window_update_statistics'].default)
            self.assertFalse(dests['reuse_update_buffers'].default)
        self.assertTrue(stage.default_v2_cuda_graph_compilation(model='ann',resolved_device='cuda')['reduce_update_stat_syncs'])

    def test_short_actual_cpu_timing_loop_ann_snn_and_comparison_output(self):
        from brain_uav.config import ScenarioConfig
        from brain_uav.scripts import profile_v2_td3 as profile
        from brain_uav.scripts.train_v2_bc import build_v2_bc_actor
        from brain_uav.trainers import v2_bc as bc
        from brain_uav.trainers.v2_validation import save_v2_validation_pool
        from test_v2_validation import _pool
        from test_v2_bc import write_cluster
        import test_v2_bc as fixtures
        # Synthetic artifacts only in a TemporaryDirectory, no expert generation/training.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=fixtures.make_scenario_payload
            def payload(index,count):
                data=original(index,count)
                data['metadata'].update(requested_direct_path_blocker=False, direct_path_blocker_count=0, feasibility_check='direct_safe_corridor')
                return data
            with mock.patch.object(fixtures,'make_scenario_payload',side_effect=payload):
                cluster=bc.load_v2_bc_trajectory_cluster(write_cluster(root/'cluster',zone_counts=(0,0,0,0),scenario_config=ScenarioConfig(max_steps=4)))
            split=bc.split_v2_bc_scenarios(cluster,validation_fraction=.25,seed=7)
            pool=_pool('easy',cluster.scenario_config,(100.,100.,100.))
            pool_path=root/'pool.json';save_v2_validation_pool(pool_path,pool)
            for model in ('ann','snn'):
                actor=build_v2_bc_actor(cluster,actor_hidden_dim=8,actor_projection_dim=4,model=model)
                state=actor.state_dict();result=bc.V2BCTrainingResult((.1,),(.1,),1,.1,state,state)
                builder=bc.build_v2_bc_checkpoint_payload if model=='ann' else bc.build_v2_snn_bc_checkpoint_payload
                checkpoint=root/f'{model}.pt'
                torch.save(builder(checkpoint_kind='best',actor=actor,actor_state_dict=state,cluster=cluster,split=split,
                    config=bc.V2BCTrainingConfig(epochs=1,batch_size=2,learning_rate=.001,seed=7,validation_fraction=.25),
                    result=result,finished_at='synthetic'),checkpoint)
                args=profile.build_parser().parse_args(['--model',model,'--bc-checkpoint',str(checkpoint),
                    '--output-dir',str(root/model),'--validation-pool',str(pool_path),
                    '--local-optimization-comparison','--device','cpu','--comparison-eager-cpu',
                    '--batch-size','2','--replay-capacity','16','--steps-per-level','8','--warmup-steps','8',
                    '--repeats','1','--validation-seed',str(pool.master_seed)])
                summary=profile.run_local_optimization_comparison(args)
                self.assertEqual(summary['config']['failure_sample_bias'], 2.)
                self.assertFalse(summary['reward_config']['breakthrough_clearance_gate_enabled'])
                self.assertTrue(summary['reward_config']['descent_three_band_enabled'])
                self.assertTrue(summary['reward_config']['boundary_directional_warning_enabled'])
                self.assertEqual(set(summary['runs']), {'A','B','C'})
                self.assertEqual(summary['code_version']['returncode'],0)
                for group in summary['runs'].values():
                    run=group[0]
                    self.assertGreater(run['actor_updates'],0)
                    self.assertGreater(run['critic_updates'],run['actor_updates'])
                    self.assertEqual(run['controlled_stage_step_offset'],75000)
                    if model=='snn':
                        self.assertTrue(all(beta==1. for beta in run['measured_actor_spike_beta_eff']))
                import json
                saved=json.loads((root/model/'local_optimization_summary.json').read_text(encoding='utf-8'))
                self.assertTrue(saved['diagnostic_only'])
                with self.assertRaises(FileExistsError):
                    profile.run_local_optimization_comparison(args)

    def test_complete_failure_marked_but_budget_fragment_not_failure(self):
        from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
        from test_profile_v2_td3 import TestProfileV2TD3
        original = V2StaticNoFlyTrajectoryEnv.step
        calls = 0
        def failed_step(env, action):
            nonlocal calls
            observation, reward, terminated, truncated, info = original(env, action)
            calls += 1
            if calls % 2 == 0:
                terminated, truncated = True, False
                info = dict(info, outcome='collision')
            return observation, reward, terminated, truncated, info
        fixture = TestProfileV2TD3()
        with mock.patch.object(V2StaticNoFlyTrajectoryEnv, 'step', new=failed_step):
            result, replay, _ = fixture.run_small_level()
        self.assertGreater(result['episodes_completed'], 0)
        self.assertGreater(int(replay.failure[:len(replay)].sum()), 0)
        _, replay, _ = fixture.run_small_level()
        self.assertFalse(replay.failure[:len(replay)].any())

    def test_correctness_mode_refuses_cpu_before_artifact_loading(self):
        from brain_uav.scripts import profile_v2_td3 as profile
        args = profile.build_parser().parse_args(['--model','ann', '--bc-checkpoint','missing.pt',
            '--validation-pool','missing.json', '--output-dir','unused', '--device','cpu',
            '--local-optimization-correctness'])
        with self.assertRaisesRegex(ValueError, 'requires real CUDA'):
            profile.run_local_optimization_comparison(args)

if __name__ == '__main__':
    unittest.main()
