"""CPU/mock checks for the CUDA acceptance driver, never CUDA acceptance proof."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch
from types import SimpleNamespace

from brain_uav.scripts import diagnose_v2_mc_return_cuda as diagnostic
from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer
from test_v2_replay_buffer import _observation


class TestMCReturnCUDADiagnostic(unittest.TestCase):
    def test_plan_fixes_mc_algorithm_and_critic_lr(self):
        args = diagnostic.build_parser().parse_args(self.argv(Path('unused')))
        plan = diagnostic.build_plan(args)
        self.assertEqual((plan['target_q_aggregation'], plan['mc_return_weight'],
            plan['mc_return_max_age_steps'], plan['critic_lr']), ('mean', .1, 100000, .001))

    def test_real_cpu_update_passes_sampler_step_and_records_same_forward(self):
        from test_v2_mc_return import populated_engine
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig
        e = populated_engine()
        config = V2FormalTrainingConfig(stage='easy', target_q_aggregation='mean', mc_return_weight=.1,
                                       critic_lr=.001, batch_size=2)
        batch = e._sample_training_batch(2)
        original = e.replay.sample
        with mock.patch.object(e, '_sample_training_batch', wraps=e._sample_training_batch) as sample:
            metric, evidence, proof = diagnostic.run_update(e, batch, 2, torch.zeros_like(batch.action), config, eager=True)
        sample.assert_called_once_with(2)
        self.assertEqual(e.replay.sample, original)
        self.assertEqual(metric.mc_valid_count, 2.)
        self.assertGreater(metric.critic_mc_loss, 0.)
        self.assertIn('critic.q1', evidence)
        self.assertTrue(any(k.startswith('gradient.pre_clip.critic1.') for k in evidence))
        self.assertIsNone(proof)
        with self.assertRaisesRegex(AssertionError, 'contract'):
            with diagnostic.controlled_sample(e, batch, 2):
                e.replay.sample(2, current_stage_step=3, mc_return_max_age_steps=100000)
        self.assertEqual(e.replay.sample, original)

    def test_evidence_callback_once_and_update_failure_not_retried(self):
        from test_v2_mc_return import populated_engine
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig
        config = V2FormalTrainingConfig(stage='easy', target_q_aggregation='mean', mc_return_weight=.1)
        e = populated_engine(); batch = e._sample_training_batch(2)
        def proof(callback, device): callback(); return {'cuda_graph_launch_count': 1}
        with mock.patch.object(e, 'update_once', wraps=e.update_once) as update:
            _, _, evidence = diagnostic.run_update(e, batch, 2, torch.zeros_like(batch.action), config, eager=True, evidence=proof)
        self.assertEqual(update.call_count, 1); self.assertEqual(evidence['cuda_graph_launch_count'], 1)
        with mock.patch.object(e, 'update_once', side_effect=RuntimeError('update failed')) as update:
            with self.assertRaisesRegex(RuntimeError, 'update failed'):
                diagnostic.run_update(e, batch, 4, torch.zeros_like(batch.action), config, eager=True, evidence=proof)
        self.assertEqual(update.call_count, 1)

    def test_actual_source_hashes_include_protected_geometry_and_launcher(self):
        identity = diagnostic.source_identity()
        for name in ('src/brain_uav/envs/v2_static_no_fly_env.py',
                     'src/brain_uav/trainers/v2_mc_return.py', 'scripts/validate_v2_mc_return_cuda.sh'):
            self.assertEqual(len(identity['sha256'][name]), 64)
        self.assertEqual(len(identity['head']), 40)

    def test_snn_real_cpu_collectors_match_shared_gpu_arithmetic(self):
        from test_v2_spike_regularization import TestSpikeRegularization
        from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig
        from brain_uav.trainers.v2_gpu_compute import GPUCompute
        from brain_uav.scripts.v2_gpu_td3_diagnostic import complete_actor_evidence
        old=TestSpikeRegularization().engine(delay=2)
        e=V2TD3UpdateEngine(old.actor,old.critic1,old.critic2,old.replay,
            .001,.001,old.gamma,old.tau,old.policy_noise,old.noise_clip,2,2,old.action_low,old.action_high,
            target_q_aggregation='mean',mc_return_weight=.1,terminal_geo_regularization_enabled=False,
            spike_regularization=old.spike_regularization)
        for i in range(len(e.replay)): e.replay.collected_stage_step[i]=1
        e.replay.fill_episode_returns([(i,int(e.replay.write_id[i])) for i in range(len(e.replay))],[1.]*len(e.replay))
        other=deepcopy(e); batch=e._sample_training_batch(75000)
        config=V2FormalTrainingConfig(stage='easy',target_q_aggregation='mean',mc_return_weight=.1,
            spike_regularization=e.spike_regularization)
        u=GPUCompute.__new__(GPUCompute); u.e=other; u.diagnostic=True; u.compute_diagnostic=True; u.tensor_evidence={}
        u.actor_params=list(other.actor.parameters());u.critic_params=list(other.critic1.parameters())+list(other.critic2.parameters())
        u.initialize_actor_loss_scalars()
        class Adam:
            def __init__(self,opt): self.opt=opt
            def step(self,error):
                if not bool(error): self.opt.step()
        u.adams=(Adam(other.actor_optimizer),Adam(other.critic_optimizer));u.injections=torch.zeros(4)
        _,policy,clip=config.noise_schedule.values(74999,max_steps=config.max_steps)
        from brain_uav.v2_curriculum import v2_bc_lambda
        # No BC reference in this fixture: exercise the collector with zero BC
        # at the public update, without changing the tested production driver.
        normal=torch.zeros_like(batch.action)
        with mock.patch('brain_uav.trainers.v2_formal_training.v2_bc_lambda',return_value=0.):
            metric,left,_=diagnostic.run_update(e,batch,75000,normal,config,eager=True)
        u.coefficients=torch.tensor([0.,other.spike_regularization.beta(75000),policy,clip])
        u.noise=normal;u.failure_fraction=torch.tensor(batch.sample_failure_fraction)
        result=u.compute(batch,'actor')
        right=u.tensor_evidence['actor'];complete_actor_evidence(left,'snn')
        compared=diagnostic.compare_snapshots(left,right)
        self.assertTrue(compared['passed'],compared['differences'])
        self.assertAlmostEqual(metric.critic_mc_loss,float(result[23]),places=5)

    def test_successful_mock_cuda_driver_covers_all_cases_and_checkpoint_rebuild(self):
        # Real driver/config/batch construction/sample wrapper; CUDA/model work mocked.
        from brain_uav.scripts import profile_v2_td3 as p, train_v2_td3 as single
        from brain_uav.scripts import v2_gpu_compute_diagnostic as helper
        from brain_uav.trainers import v2_formal_training as formal, v2_validation as validation
        from brain_uav.observations import collate_v2_observations
        from brain_uav.trainers.v2_td3 import V2TD3UpdateMetrics
        from hashlib import sha256
        from dataclasses import asdict
        built = []
        class Engine:
            def __init__(self, config, compute):
                self.config = config; self.mc_return_max_age_steps = 100000
                self.action_dim = 2; self.device = torch.device('cpu'); self.compute = compute
                self.replay = V2ReplayBuffer(config.batch_size, 2, 0)
                self._gpu_update_executor = None; self.steps = []; self.loads = 0
            def checkpoint_state_dict(self): return {'same': 1}
            def load_checkpoint_state_dict(self, state):
                self.loads += 1
                if self._gpu_update_executor is not None: self._gpu_update_executor.graphs.clear()
                self._gpu_update_executor = None
            def set_target_noise(self, **kw): self.noise = kw
            def update_once(self, total_steps, bc_lambda, diagnostic_tensor_recorder):
                batch = self.replay.sample(self.config.batch_size, current_stage_step=total_steps, mc_return_max_age_steps=100000)
                self.steps.append(total_steps)
                mode = 'critic_only' if total_steps % 2 else 'frozen_targets' if total_steps <= 25000 else 'actor'
                if self.compute:
                    if self._gpu_update_executor is None:
                        self._gpu_update_executor = SimpleNamespace(captures=1, graphs={'one':1})
                q1 = torch.ones_like(batch.reward); q2 = 2 * q1
                g, mask = batch.mc_return, batch.mc_return_mask
                mc = float((((q1-g)**2 + (q2-g)**2)*mask).sum()/mask.sum().clamp_min(1))
                diagnostic_tensor_recorder('critic.q1', q1); diagnostic_tensor_recorder('critic.q2', q2)
                return V2TD3UpdateMetrics(critic_loss=3.+.1*mc, critic_td_loss=3., critic_mc_loss=mc,
                    critic_mc_weighted_loss=.1*mc, mc_valid_count=float(mask.sum()), mc_valid_fraction=float(mask.mean()),
                    critic_updated=True, critic_targets_updated=mode!='critic_only', actor_updated=mode=='actor')
            def local_optimization_metadata(self):
                return {'gpu_td3_compute': {'effective': True, 'replays': len(self.steps),
                    'replay_submissions_by_mode': dict(critic_only=1, frozen_targets=1, actor=1)}}
        def build(scenario, config, **kw):
            e = Engine(config, kw['gpu_td3_compute']); built.append(e)
            return SimpleNamespace(engine=e)
        def snap(engine, metric=None, tensors=None):
            return {**({'loss':torch.tensor(metric.critic_loss)} if metric else {}),
                    **({'tensor.'+k:v for k,v in tensors.items()} if tensors else {})}
        with TemporaryDirectory() as directory:
            root = Path(directory); bc=root/'bc'; bc.write_bytes(b'bc'); pool=root/'pool'; pool.write_bytes(b'pool')
            manifest=root/'source'; manifest.write_text(json.dumps(diagnostic.source_identity()), encoding='utf-8')
            args=diagnostic.build_parser().parse_args(self.argv(root/'output')+[
                '--bc-checkpoint',str(bc),'--expected-bc-sha256',sha256(b'bc').hexdigest(),
                '--validation-pool',str(pool),'--source-manifest',str(manifest),'--batch-size','4'])
            prepared=SimpleNamespace(scenario_config=None, uav_collision_radius=0.)
            mixed=collate_v2_observations([_observation(0)]*4)
            from contextlib import ExitStack, nullcontext
            with ExitStack() as stack:
                for obj,name,value in (
                    (__import__('brain_uav.trainers.v2_gpu_update',fromlist=['x']), 'require_cuda_abi', lambda:None),
                    (formal,'prepare_v2_stage_initialization',lambda *a,**k:prepared),
                    (formal,'build_v2_stage_engine',build),
                    (validation,'load_v2_validation_pool',lambda *a,**k:SimpleNamespace(content_digest='b'*64,scenario_count=100)),
                    (p,'_set_numeric_engine_modes',lambda *a:None),
                    (p,'_compile_warmup_batches',lambda **k:()),
                    (p,'_numeric_diagnostic_observation_batches',lambda *a,**k:(mixed,mixed)),
                    (single,'_configure_stage_compilation',lambda *a,**k:dict(mock=True)),
                    (torch.cuda,'get_device_name',lambda *a:'CPU mock'),
                    (diagnostic,'_snapshot',snap),
                    (diagnostic,'eager_evidence',lambda *a:nullcontext()),
                    (helper,'_mps_graph_replay_evidence',lambda callback,device:(callback() or dict(cuda_graph_launch_count=1)))):
                    stack.enter_context(mock.patch.object(obj,name,new=value))
                original_to=torch.Tensor.to
                def cpu_to(tensor,*a,**kw):
                    if a and isinstance(a[0],torch.device) and a[0].type=='cuda': a=(torch.device('cpu'),*a[1:])
                    return original_to(tensor,*a,**kw)
                stack.enter_context(mock.patch.object(torch.Tensor,'to',new=cpu_to))
                report=diagnostic.run_diagnostic(args,diagnostic.build_plan(args))
            self.assertTrue(report['passed'])
            self.assertEqual(len(report['checks']),len(diagnostic._jobs())+2)
            self.assertTrue(report['checks'][-1]['new_executor'])
            self.assertEqual(built[0].steps,built[1].steps)
            self.assertEqual(built[0].config.critic_lr,.001)
            self.assertEqual(built[0].config.mc_return_weight,.1)
            self.assertEqual(report['batch_cases']['age_100000']['valid_count'],4)
            self.assertEqual(report['batch_cases']['age_100001']['valid_count'],0)
    def argv(self, output):
        return [
            '--bc-checkpoint', 'not-loaded-bc.pt',
            '--validation-pool', 'not-loaded-pool.json',
            '--output-dir', str(output),
            '--expected-bc-sha256', 'a' * 64,
            '--expected-pool-digest', 'b' * 64,
        ]

    def test_default_plan_never_probes_cuda_loads_models_or_creates_output(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            args = diagnostic.build_parser().parse_args(self.argv(output))
            self.assertFalse(args.launch)
            with mock.patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA probe')), \
                    mock.patch.object(torch, 'load', side_effect=AssertionError('checkpoint loaded')), \
                    mock.patch.object(diagnostic, 'run_diagnostic', side_effect=AssertionError('launched')):
                plan = diagnostic.build_plan(args)
                self.assertIsInstance(plan, dict)
                with redirect_stdout(StringIO()):
                    self.assertEqual(diagnostic.main(self.argv(output)), 0)
            self.assertFalse(output.exists())

    def test_launch_exception_writes_failed_summary_and_returns_failure(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'failed'
            with mock.patch.object(diagnostic, 'run_diagnostic', side_effect=RuntimeError('fixture failure')) as run, \
                    redirect_stdout(StringIO()):
                self.assertEqual(diagnostic.main([*self.argv(output), '--launch']), 1)
            run.assert_called_once()
            report = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertIs(report['passed'], False)
            self.assertIs(report['execution_completed'], False)
            self.assertIn('fixture failure', json.dumps(report))

    def test_no_cuda_cannot_be_reported_as_success_or_load_checkpoint(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'no-cuda'
            with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                    mock.patch.object(torch, 'load', side_effect=AssertionError('must reject before loading')), \
                    redirect_stdout(StringIO()):
                self.assertEqual(diagnostic.main([*self.argv(output), '--launch']), 1)
            report = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertIs(report['passed'], False)
            self.assertIs(report['execution_completed'], False)
            self.assertIn('cuda', json.dumps(report).lower())

    def test_completed_comparison_failure_still_returns_nonzero(self):
        with TemporaryDirectory() as directory:
            report = {'passed': False, 'execution_completed': True, 'completed': True}
            with mock.patch.object(diagnostic, 'run_diagnostic', return_value=report), \
                    redirect_stdout(StringIO()):
                self.assertEqual(diagnostic.main([*self.argv(Path(directory) / 'comparison'), '--launch']), 1)

    def snapshot(self):
        return {
            'actor.weight': torch.tensor([1., 2.]),
            'critic1.weight': torch.tensor([3., 4.]),
            'critic2.weight': torch.tensor([5., 6.]),
            'actor_target.weight': torch.tensor([7., 8.]),
            'critic1_target.weight': torch.tensor([9., 10.]),
            'critic2_target.weight': torch.tensor([11., 12.]),
            'bc_reference.weight': torch.tensor([13., 14.]),
            'optimizer.critic.state.0.exp_avg': torch.tensor([.1, .2]),
            'optimizer.critic.state.0.exp_avg_sq': torch.tensor([.01, .04]),
            'optimizer.critic.state.0.step': torch.tensor(75000.),
            'counts.actor_update_count': torch.tensor(25000),
            'gradient.critic1.weight': torch.tensor([.3, -.2]),
            'gradient.actor.weight.none': torch.tensor(True),
            'rng.cpu': torch.tensor([1, 2, 3], dtype=torch.uint8),
        }

    def test_compare_covers_models_targets_adam_gradients_and_counters(self):
        expected = self.snapshot()
        self.assertTrue(diagnostic.compare_snapshots(expected, deepcopy(expected))['passed'])
        for field in expected:
            actual = deepcopy(expected)
            value = actual[field]
            if value.dtype == torch.bool:
                actual[field] = ~value
            else:
                actual[field] = value + 1
            with self.subTest(field=field):
                result = diagnostic.compare_snapshots(expected, actual)
                self.assertFalse(result['passed'])
                self.assertTrue(result['differences'])

    def test_compare_rejects_structure_shape_dtype_and_matching_nonfinite(self):
        expected = self.snapshot()
        missing = deepcopy(expected)
        del missing['critic2_target.weight']
        extra = deepcopy(expected)
        extra['unexpected'] = torch.tensor(1.)
        wrong_shape = deepcopy(expected)
        wrong_shape['critic1.weight'] = wrong_shape['critic1.weight'].reshape(1, 2)
        wrong_dtype = deepcopy(expected)
        wrong_dtype['actor.weight'] = wrong_dtype['actor.weight'].double()
        for name, actual in [('missing', missing), ('extra', extra),
                             ('shape', wrong_shape), ('dtype', wrong_dtype)]:
            with self.subTest(case=name):
                self.assertFalse(diagnostic.compare_snapshots(expected, actual)['passed'])
        for nonfinite in (float('nan'), float('inf'), -float('inf')):
            actual = deepcopy(expected)
            actual['critic1.weight'][0] = nonfinite
            with self.subTest(nonfinite=nonfinite):
                self.assertFalse(diagnostic.compare_snapshots(expected, actual)['passed'])
                self.assertFalse(diagnostic.compare_snapshots(actual, deepcopy(actual))['passed'])

    def test_compare_keeps_steps_counts_and_rng_exact_despite_float_tolerance(self):
        expected = self.snapshot()
        nearby = deepcopy(expected)
        nearby['actor.weight'][0] += 1e-6
        self.assertTrue(diagnostic.compare_snapshots(expected, nearby)['passed'])
        # +1 at this scale lies within rtol=1e-4, but Adam step must remain exact.
        changed = deepcopy(expected)
        changed['optimizer.critic.state.0.step'] += 1
        self.assertFalse(diagnostic.compare_snapshots(expected, changed)['passed'])

    def test_mc_batch_cases_refresh_age_and_runtime_values_without_changing_td(self):
        replay = V2ReplayBuffer(4, 2, 0, success_replay_fraction=0., success_batch_fraction=0., seed=7)
        replay.enable_mc_returns()
        observation = _observation(0)
        refs = [replay.add(observation, [0., 0.], float(i), observation, i == 3,
                           collected_stage_step=1) for i in range(4)]
        replay.fill_episode_returns(refs, [0., -2., 3., 4.])
        source_returns = replay.mc_return.copy()
        source_rng = deepcopy(replay.rng.bit_generator.state)
        cases = diagnostic.mc_batch_cases(replay, 4)
        for name, count in [('active', 4), ('age_100000', 4), ('age_100001', 0), ('empty', 0)]:
            with self.subTest(case=name):
                batch = cases[name]
                self.assertEqual(batch.mc_return_mask.shape, (4, 1))
                self.assertEqual(batch.mc_return_mask.sum().item(), count)
                self.assertEqual(batch.mc_return_mask.dtype, torch.float32)
                self.assertEqual(batch.reward.shape, (4, 1))
        active, changed = cases['active'], cases['changed']
        self.assertEqual(active.mc_return.shape, changed.mc_return.shape)
        self.assertEqual(active.mc_return_mask.shape, changed.mc_return_mask.shape)
        self.assertFalse(torch.equal(active.mc_return, changed.mc_return))
        self.assertFalse(torch.equal(active.mc_return_mask, changed.mc_return_mask))
        for name in ('action', 'reward', 'done', 'success', 'near_goal', 'line_to_goal_safe'):
            for case, batch in cases.items():
                self.assertTrue(torch.equal(getattr(active, name), getattr(batch, name)), (case, name))
        self.assertEqual(replay.rng.bit_generator.state, source_rng)
        self.assertEqual(replay.mc_return.tolist(), source_returns.tolist())
        self.assertTrue(replay.mc_valid[:len(replay)].all())


if __name__ == '__main__':
    unittest.main()
