"""CPU wiring/mechanism checks only, not manual CUDA capture evidence."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock
import inspect
import json
import numpy as np
import torch

from brain_uav.trainers.v2_gpu_compute import GPUCompute, clone_batch, preserving_clip, tensors
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from test_v2_td3 import TestV2TD3 as Fixture, _observation


class TestGPUCompute(TestCase):
    def test_error_state_is_loaded_relative_to_gpu_compute_reference(self):
        from brain_uav.scripts import v2_gpu_compute_diagnostic as diagnostic
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'baseline'
            (source/'errors').mkdir(parents=True)
            expected = {'before': {'values': {'weight': torch.tensor([1., 2.])}},
                        'after': {'values': {'weight': torch.tensor([3., 4.])}}}
            state_path = source/'errors'/'error_state_15.pt'
            torch.save(expected, state_path)
            reference_path = source/'reference.pt'
            torch.save({'error_states': {15: 'errors/error_state_15.pt'}}, reference_path)
            args = SimpleNamespace(gpu_compute_reference=reference_path, td3_block_reference=None)
            reference = torch.load(args.gpu_compute_reference, map_location='cpu', weights_only=False)
            actual = diagnostic._load_reference_error_state(args, reference, 15)
            for phase in ('before', 'after'):
                torch.testing.assert_close(actual[phase]['values']['weight'],
                    expected[phase]['values']['weight'], rtol=0, atol=0)
            state_path.unlink()
            with self.assertRaises(FileNotFoundError):
                diagnostic._load_reference_error_state(args, reference, 15)

    def test_environment_diagnostic_cli_is_default_off_and_timing_only(self):
        from brain_uav.scripts import profile_v2_td3 as p, v2_gpu_compute_diagnostic as d
        base = ['--model','snn','--device','cuda','--bc-checkpoint','bc.pt',
            '--validation-pool','pool.json','--output-dir','unused']
        self.assertFalse(p.build_parser().parse_args(base).gpu_compute_environment_diagnostic)
        for mode in ([], ['--gpu-compute-worker','correctness'], ['--gpu-compute-comparison'],
                     ['--gpu-compute-worker','timing']):
            with self.subTest(mode=mode), mock.patch.object(p,'run_local_optimization_comparison') as run:
                with self.assertRaisesRegex(ValueError, 'timing.*gpu-td3-compute'):
                    p.main(base+mode+['--gpu-compute-environment-diagnostic'])
                run.assert_not_called()
        for profiler in ('--detailed-profiler-updates','--compiled-performance-diagnostic-updates'):
            with self.assertRaisesRegex(ValueError,'profiler'):
                d.run(p.build_parser().parse_args(base+['--gpu-compute-worker','timing',
                    '--gpu-td3-compute','--gpu-compute-environment-diagnostic',profiler,'2']))

    def test_environment_only_recorder_does_not_instrument_gpu_update(self):
        from brain_uav.scripts import profile_v2_td3 as p
        from brain_uav.config import RewardConfig
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig
        from test_v2_bc import make_scenario_config, make_scenario_payload
        e = self.engine()
        e.bc_reference_actor = e._frozen_target(e.actor,device=e.device)
        e.gpu_td3_compute = True
        e._gpu_update_executor = SimpleNamespace(begin_measurement=lambda: None,invalidate=lambda: None,
            metadata=lambda: {'requested': True})
        original = e.update_once
        def update(**kwargs):
            self.assertNotIn('diagnostic_timing_recorder',kwargs)
            self.assertFalse(kwargs.get('profile_sections',False))
            e.gpu_td3_compute = False  # CPU orchestration test, no CUDA executor evidence.
            try:
                return original(**kwargs)
            finally:
                e.gpu_td3_compute = True
        e.update_once = update
        e.select_action = lambda *a,**kw: [0.,0.]
        scenario = make_scenario_config()
        with mock.patch.object(p,'build_v2_stage_engine',return_value=SimpleNamespace(
                engine=e, exploration_rng=np.random.default_rng(7))), \
                mock.patch('torch.profiler.profile',side_effect=AssertionError('profiler enabled')):
            result = p._run_diagnostic_level(level='easy',
                pool=SimpleNamespace(stage_seed=7, scenarios=[
                    dict(scenario_id='fixed',payload=make_scenario_payload(0,1))]),
                prepared=SimpleNamespace(scenario_config=scenario,reward_config=RewardConfig(),uav_collision_radius=0.),
                formal_config=V2FormalTrainingConfig(stage='easy',batch_size=2,replay_capacity=64,actor_freeze_steps=0),
                bc_checkpoint=Path('unused'), model='ann',snn_time_window=4,device=torch.device('cpu'),
                warmup_steps=8,measured_steps=5,gpu_td3_compute=True,environment_performance_diagnostic=True)
        env = result['environment_geometry_diagnostic']
        self.assertEqual(env['step_count'],5)
        self.assertEqual(env['duplicate_query_audit']['audited_step_count'],5)
        self.assertTrue(env['geometry_queries'])
        for name in p._EnvironmentGeometryDiagnostic._STEP_SECTIONS:
            self.assertEqual(env['step_sections'][name]['calls'],5)
        self.assertAlmostEqual(env['step_total_wall_seconds'],
            sum(section['total_wall_seconds'] for section in env['step_sections'].values()))
        self.assertIsNone(result['timing']['td3_nested_update_breakdown'])
        self.assertFalse(result['detailed_profiler']['enabled'])

    def test_environment_timing_wiring_and_compact_review_output(self):
        from brain_uav.scripts import profile_v2_td3 as p, v2_gpu_compute_diagnostic as d
        from brain_uav.config import RewardConfig
        from test_v2_bc import make_scenario_payload
        for enabled in (False,True):
            with self.subTest(enabled=enabled), TemporaryDirectory() as directory:
                root = Path(directory); bc=root/'bc.pt'; bc.write_bytes(b'fixture')
                pool_file=root/'pool.json'; pool_file.write_text('{}')
                output=root/'output'
                pool=SimpleNamespace(content_digest='digest',scenario_count=1,master_seed=7,stage_seed=7,
                    scenario_config={},uav_collision_radius=0.,curriculum_level='easy',
                    scenarios=[dict(scenario_id='fixed',payload=make_scenario_payload(0,1))])
                recorder=p._EnvironmentGeometryDiagnostic()
                env=recorder.to_dict(); recorder.close()
                result=dict(timing={'total_wall_seconds':1.,'environment_step_wall_seconds':.5,
                        'scenario_reset_wall_seconds':.01,'pre_action_geometry_wall_seconds':.02},measured_steps=256,
                    scenario_coverage=[dict(scenario_id='fixed',zone_count=1,measured_steps=256)],
                    environment_geometry_diagnostic=env,critic_encoder_compile={'valid_for_speed_comparison':True})
                args=p.build_parser().parse_args(['--model','snn','--device','cuda','--gpu-compute-worker','timing',
                    '--gpu-td3-compute','--bc-checkpoint',str(bc),'--validation-pool',str(pool_file),
                    '--output-dir',str(output),'--repeats','2','--steps-per-level','256','--warmup-steps','128']+
                    (['--gpu-compute-environment-diagnostic'] if enabled else []))
                with mock.patch.object(p,'resolve_training_device',return_value='cuda'), \
                        mock.patch.object(p,'prepare_v2_stage_initialization',return_value=SimpleNamespace(
                            reward_config=RewardConfig(),scenario_config={},uav_collision_radius=0.)), \
                        mock.patch.object(p,'load_v2_validation_pool',return_value=pool), \
                        mock.patch.object(p,'V2ValidationPool',side_effect=lambda **kw: pool), \
                        mock.patch.object(p,'_run_diagnostic_level',side_effect=lambda **kw: deepcopy(result)) as run, \
                        mock.patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')), \
                        mock.patch('importlib.metadata.version',return_value='fixture'), \
                        mock.patch('torch.cuda.empty_cache'):
                    summary=d.run(args)
                self.assertEqual(run.call_count,2)
                for call in run.call_args_list:
                    self.assertEqual(call.kwargs['environment_performance_diagnostic'],enabled)
                    self.assertTrue(call.kwargs['gpu_td3_compute'])
                    self.assertEqual(call.kwargs['compiled_path_profiler_updates'],0)
                    self.assertTrue(call.kwargs['cache_ellipsoid_segment_clearance'])
                if enabled:
                    report=json.loads((output/'review/environment_summary.json').read_text(encoding='utf-8'))
                    self.assertEqual(len(report['runs']),2)
                    self.assertEqual(report['runs'][0]['environment_geometry_diagnostic'],env)
                    self.assertEqual(report['scenarios'][0]['shape_types'],['sphere'])
                    self.assertEqual(report['runs'][0]['measured_steps'],256)
                    self.assertFalse(report['valid_for_speed_comparison'])
                    self.assertFalse(summary['valid_for_speed_comparison'])
                    self.assertFalse(summary['final_speed_acceptance'])
                    self.assertIsNone(summary['comparison']['compute']['speedup'])
                    self.assertEqual(set(f.name for f in (output/'review').iterdir()),{'environment_summary.json'})
                    self.assertTrue((output/'other/local_optimization_summary.json').exists())
                    self.assertTrue((output/'other/compute_0.json').exists())
                    self.assertFalse(list(output.rglob('*.pt')))
                    self.assertFalse((output/'local_optimization_summary.json').exists())
                else:
                    self.assertTrue((output/'local_optimization_summary.json').exists())
                    self.assertFalse((output/'review').exists())

    def test_eval_bc_initialization_modes_before_first_compile_and_error_report(self):
        from brain_uav.scripts import profile_v2_td3 as p, train_v2_td3 as single
        from brain_uav.scripts import v2_gpu_compute_diagnostic as d
        from brain_uav.config import RewardConfig
        from test_v2_spike_regularization import TestSpikeRegularization
        config = SimpleNamespace(to_dict=lambda: {})
        prepared = SimpleNamespace(reward_config=RewardConfig())
        pool = SimpleNamespace(content_digest='test-pool')
        for model in ('ann', 'snn'):
            with self.subTest(model=model), TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root/'bc.pt'; checkpoint.write_bytes(b'local fixture')
                pool_file = root/'pool.json'; pool_file.write_text('{}')
                contract = None
                for enabled in (False, True):
                    e = self.engine() if model == 'ann' else TestSpikeRegularization().engine(delay=2)
                    e.actor.eval()  # Actual BC initialization mode, not default train.
                    e.critic1.eval(); e.critic2.eval()
                    e.bc_reference_actor = e._frozen_target(e.actor, device=e.device)
                    initial = deepcopy(e.checkpoint_state_dict())
                    output = root/str(enabled); output.mkdir()
                    reference = root/'reference.pt'
                    if enabled:
                        torch.save(dict(contract=contract, initial=initial), reference)
                    args = SimpleNamespace(model=model, seed=7, bc_checkpoint=checkpoint,
                        validation_pool=pool_file, gpu_td3_compute=enabled, snn_time_window=4,
                        non_blocking_batch_transfer=False,
                        correctness_updates=8, comparison_bc_schedule='smooth-final')
                    error = RuntimeError('stop at first compile')
                    def configure(engine, *a, **kw):
                        self.assertTrue(all(getattr(engine,n).training for n in ('actor','critic1','critic2')))
                        for n in ('actor_target','critic1_target','critic2_target','bc_reference_actor'):
                            module = getattr(engine,n)
                            self.assertFalse(module.training)
                            self.assertTrue(all(not x.requires_grad for x in module.parameters()))
                        raise error
                    with mock.patch.object(p, 'build_v2_stage_engine', return_value=SimpleNamespace(engine=e)), \
                            mock.patch.object(p, '_td3_block_correctness_steps', return_value=[1]), \
                            mock.patch.object(single, 'default_v2_cuda_graph_compilation',
                                return_value=dict(fused_adam=False,reduce_update_stat_syncs=False)), \
                            mock.patch.object(single, '_configure_stage_compilation', side_effect=configure), \
                            mock.patch.object(d.subprocess, 'run', return_value=SimpleNamespace(stdout='test')):
                        with self.assertRaises(RuntimeError) as caught:
                            d.correctness(args,prepared=prepared,config=config,pool=pool,
                                device='cpu',output=output,full_pool_digest='test-pool')
                    self.assertIs(caught.exception,error)
                    report = json.loads((output/'correctness.json').read_text())
                    self.assertFalse(report['passed']); self.assertFalse(report['completed'])
                    self.assertEqual(report['error']['type'],'RuntimeError')
                    self.assertEqual(report['error']['message'],str(error))
                    self.assertIn('Traceback',report['error']['traceback'])
                    self.assertEqual(report['error'],report['failure'])
                    contract = report['inputs']
                    self.assertEqual(contract['version'],8)

    def test_limited_residual_values_do_not_hide_structure_nonfinite_or_active_changes(self):
        from brain_uav.scripts.v2_gpu_compute_diagnostic import limited_retained_comparison as compare
        def at_loss(a,b,**kwargs):
            return compare(a,b,phase='critic_loss',**kwargs)
        key = 'gradient.actor.zone_set_encoder.empty_scene_token'
        a = {key:torch.ones(3)}
        result = at_loss(a,{key:torch.zeros(3)})
        self.assertTrue(result['passed']); self.assertFalse(result['raw']['passed'])
        self.assertEqual(len(result['waived_fields']),1)
        for actual in ({key+'.none':torch.tensor(True)}, {key:torch.zeros(4)},
                {key:torch.zeros(3,dtype=torch.float64)}):
            self.assertFalse(at_loss(a,actual)['passed'])
        for value in (float('nan'),float('inf'),float('-inf')):
            result = at_loss(a,{key:torch.tensor([value,0.,0.])})
            self.assertTrue(result['passed']); self.assertFalse(result['raw']['passed'])
        self.assertFalse(compare(a,{key:torch.zeros(3)})['passed'])
        self.assertTrue(at_loss({key:torch.tensor([float('inf'),1.,0.])},
            {key:torch.tensor([1.,float('inf'),0.])})['passed'])
        from brain_uav.scripts.v2_gpu_compute_diagnostic import state_difference
        # Compute's own protection is exact and never calls the waiver helper.
        self.assertFalse(state_difference(a,{key:torch.zeros(3)},exact=True)['passed'])
        for key in ('actor.weight','optimizer.actor.state.0.exp_avg','counts.update_count',
                    'gradient.actor_target.head.weight','gradient.bc_reference_actor.head.weight'):
            self.assertFalse(at_loss({key:torch.ones(3)},{key:torch.zeros(3)})['passed'])
        key = 'grad.actor.0'
        mapping = {key:dict(parameter_name='actor.zone_set_encoder.empty_scene_token')}
        self.assertTrue(at_loss({key:torch.ones(3)},{key:torch.zeros(3)},gradients=mapping)['passed'])
        self.assertFalse(compare({key:torch.ones(3)},{key:torch.zeros(3)},gradients=mapping)['passed'])

    def test_actor_and_critic_residuals_are_classified_by_consumption_not_parameter_whitelist(self):
        from brain_uav.scripts.v2_gpu_compute_diagnostic import limited_retained_comparison as compare
        for module,name in (('actor','head.0.weight'),('actor','zone_set_encoder.empty_scene_token'),
                            ('critic1','head.0.weight'),('critic2','zone_set_encoder.empty_scene_token')):
            for indexed in (False,True):
                key = 'grad.'+module+'.123' if indexed else 'gradient.'+module+'.'+name
                mapping = {key:dict(parameter_name=module+'.'+name)} if indexed else {}
                expected,actual = {key:torch.ones(3)},{key:torch.zeros(3)}
                with self.subTest(module=module,indexed=indexed):
                    loss = compare(expected,actual,phase='critic_loss',gradients=mapping)
                    self.assertTrue(loss['passed']); self.assertFalse(loss['raw']['passed'])
                    self.assertEqual(loss['waived_fields'][0]['parameter_name'],module+'.'+name)
                    recovery = compare(expected,actual,phase='critic_only_recovery',gradients=mapping)
                    self.assertEqual(recovery['passed'],module=='actor')
                    for phase in ('critic_loss','critic_only_recovery'):
                        self.assertFalse(compare(expected,actual,phase=phase,actor_backward=True,gradients=mapping)['passed'])
                    for phase in ('critic_gradient','actor_loss'):
                        self.assertEqual(compare(expected,actual,phase=phase,gradients=mapping)['passed'],module=='actor')
                    for phase in ('actor_gradient',None):
                        self.assertFalse(compare(expected,actual,phase=phase,gradients=mapping)['passed'])
                    for changed in (torch.zeros(4),torch.zeros(3,dtype=torch.float64)):
                        self.assertFalse(compare(expected,{key:changed},phase='critic_loss',gradients=mapping)['passed'])
                    for phase in ('critic_gradient','actor_loss','actor_gradient',None):
                        changed = {key:torch.tensor([float('nan'),0.,0.])}
                        self.assertEqual(compare(expected,changed,phase=phase,gradients=mapping)['passed'],
                            module=='actor' and phase in ('critic_gradient','actor_loss'))
                    self.assertFalse(compare(expected,{key+'.none':torch.tensor(True)},
                        phase='critic_loss',gradients=mapping)['passed'])
                    if indexed:
                        self.assertFalse(compare(expected,actual,phase='critic_loss')['passed'])
                        self.assertFalse(compare(expected,actual,phase='critic_loss',
                            gradients={key:dict(parameter_name='actor_target.'+name)})['passed'])

    def test_recovery_gate_requires_critic_and_full_actor_success(self):
        from brain_uav.scripts.v2_gpu_compute_diagnostic import recovery_acceptance
        updates = [dict(actor_backward=False,active_update=dict(passed=True),all_fields=dict(passed=False)),
            dict(actor_backward=True,active_update=dict(passed=True),all_fields=dict(passed=True))]
        self.assertTrue(recovery_acceptance(updates))
        for index,field in ((0,'active_update'),(1,'active_update'),(1,'all_fields')):
            bad = deepcopy(updates); bad[index][field]['passed'] = False
            self.assertFalse(recovery_acceptance(bad))
        self.assertFalse(recovery_acceptance(updates[:1]))
        self.assertFalse(recovery_acceptance([dict(actor_backward=False),dict(actor_backward=True)]))

    def test_exception_own_protection_is_exact_and_current_gradient_nan_positions_remain_strict(self):
        from brain_uav.scripts.v2_gpu_compute_diagnostic import exception_self_protection, current_gradient_comparison
        from brain_uav.scripts.v2_gpu_compute_diagnostic import EXCEPTION_GRADIENTS
        before = {name:torch.ones(2) for name in ('actor.weight','actor_target.weight',
            'critic1.weight','critic1_target.weight','optimizer.actor.m','optimizer.critic.m',
            'counts.update_count','gradient.actor.weight','gradient.critic1.weight','gradient.critic2.weight')}
        for failure,plan in EXCEPTION_GRADIENTS.items():
            for name in before:
                after = deepcopy(before); after[name].add_(1)
                protected = exception_self_protection(before,after,failure)
                is_current_critic = name.startswith(('gradient.critic1.','gradient.critic2.'))
                if failure=='critic_loss': expected = False
                elif failure=='critic_gradient': expected = is_current_critic
                else:
                    expected = name.startswith(('critic1.','critic1_target.','optimizer.critic.',
                        'gradient.critic1.','gradient.critic2.')) or (failure=='actor_gradient' and name.startswith('gradient.actor.'))
                self.assertEqual(protected['passed'],expected,(failure,name))
            for module in plan['current']:
                key = 'gradient.'+module+'.weight'
                a,b = {key:torch.tensor([float('inf'),1.])},{key:torch.tensor([1.,float('inf')])}
                self.assertFalse(current_gradient_comparison(a,b,plan['current'])['passed'])
        a = deepcopy(before); a['gradient.actor.weight'][0] = float('nan')
        self.assertTrue(exception_self_protection(a,deepcopy(a),'actor_loss')['passed'])
        self.assertFalse(exception_self_protection(a,before,'actor_loss')['passed'])

    def test_full_driver_only_starts_timing_after_both_correctness_workers(self):
        from brain_uav.scripts.profile_v2_td3 import build_parser
        from brain_uav.scripts.v2_gpu_compute_diagnostic import run
        with TemporaryDirectory() as directory:
            args = build_parser().parse_args(['--model','snn','--device','cuda',
                '--gpu-compute-comparison','--bc-checkpoint','bc.pt','--validation-pool','pool.json',
                '--output-dir',directory])
            calls = []
            def child(command,**kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0 if len(calls)<3 else 1)
            with mock.patch('brain_uav.scripts.v2_gpu_compute_diagnostic.subprocess.run',side_effect=child):
                with self.assertRaisesRegex(RuntimeError,'Timing worker failed'): run(args)
            self.assertEqual([c[c.index('--gpu-compute-worker')+1] for c in calls],
                ['correctness','correctness','timing'])
            self.assertTrue(all('--gpu-compute-first-critic-loss-only' not in c for c in calls))

    def test_retained_trace_is_optional_owned_and_restores_wrappers(self):
        from brain_uav.scripts.v2_gpu_compute_diagnostic import trace_retained_gradient, retained_trace_summary
        e = self.engine(); parameter = e.actor.zone_set_encoder.empty_scene_token
        parameter.grad = torch.ones_like(parameter)
        original_grad = parameter.grad
        original_mark = e._mark_cuda_graph_step_begin
        events = []
        with mock.patch.object(torch.Tensor,'cpu',side_effect=AssertionError('disabled trace sampled')):
            with trace_retained_gradient(e,events,enabled=False,context={}):
                pass
        self.assertFalse(events)
        # Artificial mutation only in this unit test; the observer must not
        # restore/rebind/clear the live gradient to hide it.
        def mark(): parameter.grad.add_(1.)
        with mock.patch.object(e,'_mark_cuda_graph_step_begin',new=mark):
            with trace_retained_gradient(e,events,enabled=True,context=dict(index=15,failure='critic_loss')):
                e._mark_cuda_graph_step_begin()
            self.assertIs(e._mark_cuda_graph_step_begin,mark)
        self.assertEqual(e._mark_cuda_graph_step_begin,original_mark)
        self.assertIs(parameter.grad,original_grad)
        torch.testing.assert_close(events[0]['value'],torch.ones_like(parameter))
        self.assertNotEqual(events[0]['value'].data_ptr(),original_grad.data_ptr())
        summary = retained_trace_summary(events)
        self.assertEqual(summary['first_observed_change']['boundary'],'_mark_cuda_graph_step_begin.after')
        self.assertEqual(summary['first_observed_change']['parameter'],'actor.zone_set_encoder.empty_scene_token')
        anchor = dict(events[0],failure=None,boundary='update.after',value=events[0]['value']-1.)
        self.assertEqual(retained_trace_summary([anchor]+events)['first_observed_change']['boundary'],'update.before')

    def test_error_recovery_worker_reports_failure_without_terminal_traceback(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from brain_uav.scripts import profile_v2_td3 as p
        from brain_uav.scripts.v2_gpu_compute_diagnostic import run
        args = p.build_parser().parse_args(['--model','snn','--device','cuda',
            '--gpu-compute-worker','correctness','--gpu-compute-first-critic-loss-only',
            '--gpu-compute-error-recovery','--bc-checkpoint','bc.pt','--validation-pool','pool.json',
            '--output-dir','unused-mock-output'])
        terminal = StringIO()
        with mock.patch.object(p,'run_local_optimization_comparison',side_effect=AssertionError('mismatch')), \
                redirect_stdout(terminal),self.assertRaises(SystemExit) as caught:
            run(args)
        self.assertEqual(caught.exception.code,1)
        self.assertNotIn('Traceback',terminal.getvalue())
        self.assertFalse(json.loads(terminal.getvalue())['passed'])

    def test_error_recovery_finishes_both_steps_before_failing_cpu(self):
        from brain_uav.scripts import profile_v2_td3 as p, train_v2_td3 as single
        from brain_uav.scripts import v2_gpu_compute_diagnostic as d
        from brain_uav.config import RewardConfig
        for case in ('success','protected_error','checkpoint_abort'):
            inject_protected_error = case == 'protected_error'
            with self.subTest(case=case), TemporaryDirectory() as directory:
                e = self.engine(); e.actor_freeze_steps = 25000
                e.bc_reference_actor = e._frozen_target(e.actor,device=e.device)
                config = SimpleNamespace(to_dict=lambda: {},policy_delay=2,actor_freeze_steps=25000,
                    spike_regularization=e.spike_regularization,batch_size=e.batch_size,max_steps=500000,
                    noise_schedule=SimpleNamespace(values=lambda *a,**kw: (0.,.01,.02)))
                observation = e.replay.sample(2).obs
                root = Path(directory)
                checkpoint = root/'bc.pt'; checkpoint.write_bytes(b'fixture')
                pool_file = root/'pool.json'; pool_file.write_text('{}')
                args = SimpleNamespace(model='ann',seed=7,bc_checkpoint=checkpoint,
                    validation_pool=pool_file,gpu_td3_compute=False,snn_time_window=4,
                    non_blocking_batch_transfer=False,
                    correctness_updates=8,comparison_bc_schedule='smooth-final',
                    gpu_compute_first_critic_loss_only=False,gpu_compute_error_recovery=True,
                    gpu_compute_trace_retained_gradient=True)
                original = e._require_finite_loss
                original_reload = d._prepare_loaded_bc_reference
                def reload(*a,**kw):
                    if case == 'checkpoint_abort':
                        raise RuntimeError('checkpoint abort')
                    return original_reload(*a,**kw)
                def loss_check(loss,**kwargs):
                    if inject_protected_error and not bool(torch.isfinite(loss).all()):
                        e.update_count += 1  # Diagnostic test: prove protection failure remains fatal.
                        e.actor_update_count += 1
                    return original(loss,**kwargs)
                with mock.patch.object(p,'build_v2_stage_engine',return_value=SimpleNamespace(engine=e)), \
                        mock.patch('torch.compile',new=lambda fn,**kw: fn), \
                        mock.patch.object(single,'_configure_stage_compilation',return_value={}), \
                        mock.patch.object(p,'_compile_warmup_batches',return_value=[observation]), \
                        mock.patch.object(p,'_numeric_diagnostic_observation_batches',return_value=(observation,observation)), \
                        mock.patch.object(d.subprocess,'run',return_value=SimpleNamespace(stdout='fixture')), \
                        mock.patch.object(torch.cuda,'get_rng_state',return_value=torch.tensor([1],dtype=torch.uint8)), \
                        mock.patch.object(d,'_prepare_loaded_bc_reference',side_effect=reload), \
                        mock.patch.object(e,'_require_finite_loss',new=loss_check):
                    if case == 'checkpoint_abort':
                        with self.assertRaisesRegex(RuntimeError,'checkpoint abort'):
                            d.correctness(args,prepared=SimpleNamespace(reward_config=RewardConfig()),
                                config=config,pool=SimpleNamespace(content_digest='fixture'),device='cpu',output=root,
                                full_pool_digest='fixture')
                    elif inject_protected_error:
                        with self.assertRaisesRegex(AssertionError,'comparisons completed but inconsistent'):
                            d.correctness(args,prepared=SimpleNamespace(reward_config=RewardConfig()),
                                config=config,pool=SimpleNamespace(content_digest='fixture'),device='cpu',output=root,
                                full_pool_digest='fixture')
                    else:
                        d.correctness(args,prepared=SimpleNamespace(reward_config=RewardConfig()),
                            config=config,pool=SimpleNamespace(content_digest='fixture'),device='cpu',output=root,
                            full_pool_digest='fixture')
                report = json.loads((root/'correctness.json').read_text())
                self.assertTrue(report['recovery_completed'])
                if case == 'checkpoint_abort':
                    self.assertFalse(report['completed']); self.assertFalse(report['passed'])
                    self.assertIsNot(report.get('load_rebuild_verified'),True)
                    self.assertEqual(len(report['checks']),28)
                    self.assertFalse(report['acceptance']['semantic_passed'])
                    continue
                self.assertTrue(report['completed'])
                self.assertEqual(report['passed'],not inject_protected_error)
                self.assertEqual(len(report['checks']),29)
                self.assertEqual([x['actor_backward'] for x in report['recovery_updates']],[False,True]*3)
                self.assertEqual([x['stage_steps'] for x in report['recovery_updates']],
                    [75013,75014,75019,75020,75025,75026])
                self.assertTrue(report['load_rebuild_verified'])
                self.assertTrue((root/'retained_gradient_trace.pt').exists())
                compact = json.loads((root/'error_recovery_summary.json').read_text())
                self.assertEqual(compact['protected_state'][0]['passed'],not inject_protected_error)
                if inject_protected_error:
                    self.assertIn('Traceback',report['error']['traceback'])
                    self.assertEqual(report['first_error_state_failure']['failure'],'critic_loss')
                    self.assertGreaterEqual(len(report['deferred_failures']),2)
                else:
                    self.assertTrue(report['load_rebuild_verified'])
                    self.assertEqual([v['failure'] for v in report['checks'] if v['failure']],
                        ['critic_loss','critic_gradient','actor_loss','actor_gradient'])
                    steps = [v['total_steps'] for v in report['checks']]
                    self.assertEqual(steps,sorted(set(steps)))
                    self.assertEqual(report['acceptance']['semantic_passed'],True)

    def test_cpu_unused_actor_gradient_is_not_consumed_by_later_updates(self):
        from brain_uav.scripts.v2_gpu_compute_diagnostic import error_state, state_difference
        e = self.engine(); batch = e.replay.sample(e.batch_size)
        def noise(t,**kw): return torch.zeros_like(t)
        with mock.patch('torch.randn_like',new=noise):
            e.update_once(total_steps=2)
        other = self.engine(); other.load_checkpoint_state_dict(deepcopy(e.checkpoint_state_dict()))
        for engine,value in ((e,1000.),(other,-1000.)):
            # Synthetic differing stale values, not a diagnostic repair/writeback.
            p = engine.actor.zone_set_encoder.empty_scene_token
            p.grad = torch.full_like(p,value)
            for critic in (engine.critic1,engine.critic2):
                parameter = next(critic.parameters())
                parameter.grad = torch.full_like(parameter,value)
            engine.replay.sample = lambda size: batch
        for total in (3,4):
            with mock.patch('torch.randn_like',new=noise):
                e.update_once(total_steps=total); other.update_once(total_steps=total)
            a,b = error_state(e)['values'],error_state(other)['values']
            whole = state_difference(a,b)
            if total == 3:
                self.assertFalse(whole['passed'])  # Residual difference remains a difference.
                self.assertTrue(all(x['field'].startswith('gradient.actor.') for x in whole['differences']))
            else:
                self.assertTrue(whole['passed'],whole['first_difference'])

    def test_error_state_named_owned_snapshots_preserve_none_and_live_gradients(self):
        from brain_uav.scripts.v2_gpu_compute_diagnostic import error_state, state_difference
        e = self.engine()
        name, parameter = next(e.critic1.named_parameters())
        parameter.grad = torch.ones_like(parameter)
        reference = parameter.grad
        before = error_state(e)
        parameter.grad.add_(2.)
        after = error_state(e)
        key = 'gradient.critic1.' + name
        self.assertIs(parameter.grad,reference)
        torch.testing.assert_close(before['values'][key],torch.ones_like(parameter))
        self.assertNotEqual(before['values'][key].data_ptr(),reference.data_ptr())
        self.assertFalse(before['values'][key].requires_grad)
        self.assertIn('gradient.critic2.'+next(e.critic2.named_parameters())[0]+'.none',before['values'])
        difference = state_difference(before['values'],after['values'],exact=True)
        self.assertEqual(difference['first_difference']['field'],key)
        self.assertEqual(difference['first_difference']['maximum_finite_absolute_error'],2.)
        self.assertTrue(any(n.startswith('optimizer.') for n in before['values']))
        self.assertIn('counts.update_count',before['values'])

    def test_error_state_collects_all_differences_and_saves_both_sides(self):
        from brain_uav.scripts.v2_gpu_compute_diagnostic import error_state, record_error_state
        e = self.engine()
        e.update_once(total_steps=2)  # Populate real Adam state, one synthetic CPU update.
        e.critic1.head[0].weight.grad = torch.ones_like(e.critic1.head[0].weight)
        before = error_state(e); after = error_state(e)
        expected = dict(before=deepcopy(before),after=deepcopy(after))
        key = 'gradient.critic1.head.0.weight'
        after['values'][key].flatten()[0] = float('inf')
        after['values']['counts.update_count'].add_(1)
        for prefix in ('critic1.','critic1_target.','optimizer.critic.state.'):
            field = next(n for n,t in after['values'].items() if n.startswith(prefix) and t.is_floating_point())
            after['values'][field].add_(1.)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = record_error_state(root,15,before,after,expected)
            self.assertFalse(report['cross_after']['passed'])
            self.assertEqual(len(report['cross_after']['differences']),5)
            item = next(x for x in report['cross_after']['differences'] if x['field']==key)
            self.assertEqual(item['actual_nonfinite']['positive_inf'],1)
            self.assertEqual(item['expected_nonfinite']['positive_inf'],0)
            saved = torch.load(report['snapshot_path'],weights_only=False)
            self.assertEqual(saved['before']['values'][key].flatten()[0],1.)
            self.assertTrue(torch.isinf(saved['after']['values'][key].flatten()[0]))
            self.assertLessEqual(len(torch.load(report['difference_evidence_path'],weights_only=False)),4)
            self.assertTrue((root/'error_state_15.json').exists())

    def test_first_critic_loss_driver_rebuilds_reference_and_never_times(self):
        from brain_uav.scripts.profile_v2_td3 import build_parser
        from brain_uav.scripts.v2_gpu_compute_diagnostic import run
        with TemporaryDirectory() as directory:
            args = build_parser().parse_args(['--model','snn','--device','cuda',
                '--gpu-compute-comparison','--gpu-compute-first-critic-loss-only',
                '--gpu-compute-error-recovery','--gpu-compute-trace-retained-gradient',
                '--bc-checkpoint','bc.pt','--validation-pool','pool.json','--output-dir',directory])
            calls = []
            def child(command,**kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0)
            with mock.patch('brain_uav.scripts.v2_gpu_compute_diagnostic.subprocess.run',side_effect=child):
                report = run(args)
            self.assertTrue(report['passed'])
            self.assertEqual(len(calls),2)
            self.assertTrue(all('--gpu-compute-first-critic-loss-only' in c and 'correctness' in c for c in calls))
            self.assertTrue(all('--gpu-compute-error-recovery' in c and '--gpu-compute-trace-retained-gradient' in c for c in calls))
            self.assertNotIn('--gpu-compute-reference',calls[0])
            self.assertIn('--gpu-compute-reference',calls[1])

    def test_first_critic_loss_preserves_full_preceding_sequence_cpu(self):
        from brain_uav.scripts import profile_v2_td3 as p, train_v2_td3 as single
        from brain_uav.scripts import v2_gpu_compute_diagnostic as d
        from brain_uav.config import RewardConfig
        e = self.engine(); e.actor_freeze_steps = 25000
        e.bc_reference_actor = e._frozen_target(e.actor,device=e.device)
        config = SimpleNamespace(to_dict=lambda: {},policy_delay=2,actor_freeze_steps=25000,
            spike_regularization=e.spike_regularization,batch_size=e.batch_size,max_steps=500000,
            noise_schedule=SimpleNamespace(values=lambda *a,**kw: (0.,.01,.02)))
        observation = e.replay.sample(2).obs
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'bc.pt'; checkpoint.write_bytes(b'fixture')
            pool_file = root/'pool.json'; pool_file.write_text('{}')
            args = SimpleNamespace(model='ann',seed=7,bc_checkpoint=checkpoint,
                validation_pool=pool_file,gpu_td3_compute=False,snn_time_window=4,
                non_blocking_batch_transfer=False,
                correctness_updates=8,comparison_bc_schedule='smooth-final',
                gpu_compute_first_critic_loss_only=True)
            with mock.patch.object(p,'build_v2_stage_engine',return_value=SimpleNamespace(engine=e)), \
                    mock.patch.object(single,'_configure_stage_compilation',return_value={}), \
                    mock.patch.object(p,'_compile_warmup_batches',return_value=[observation]), \
                    mock.patch.object(p,'_numeric_diagnostic_observation_batches',return_value=(observation,observation)), \
                    mock.patch.object(d.subprocess,'run',return_value=SimpleNamespace(stdout='fixture')), \
                    mock.patch.object(torch.cuda,'get_rng_state',return_value=torch.tensor([1],dtype=torch.uint8)):
                report = d.correctness(args,prepared=SimpleNamespace(reward_config=RewardConfig()),
                    config=config,pool=SimpleNamespace(content_digest='fixture'),device='cpu',output=root,
                    full_pool_digest='fixture')
            self.assertTrue(report['passed'])
            self.assertEqual(len(report['checks']),16)
            self.assertEqual(report['checks'][-1]['failure'],'critic_loss')
            self.assertEqual(report['checks'][-1]['total_steps'],75012)
            self.assertIsNone(report['load_rebuild_verified'])
            reference = torch.load(root/'reference.pt',weights_only=False)
            self.assertTrue(reference['contract']['first_critic_loss_only'])
            self.assertEqual(set(reference['error_states']),{15})
            self.assertTrue((root/reference['error_states'][15]).exists())
            reference['contract']['version'] = 3
            torch.save(reference,root/'obsolete.pt')
            args.gpu_td3_compute = True; args.gpu_compute_reference = root/'obsolete.pt'
            with mock.patch.object(d.subprocess,'run',return_value=SimpleNamespace(stdout='fixture')):
                with self.assertRaisesRegex(ValueError,'contract mismatch'):
                    d.correctness(args,prepared=SimpleNamespace(reward_config=RewardConfig()),
                        config=config,pool=SimpleNamespace(content_digest='fixture'),device='cpu',output=root,
                        full_pool_digest='fixture')

    def test_checkpoint_reload_reuses_compilation_and_rebinds_only_bc(self):
        from brain_uav.scripts.v2_gpu_compute_diagnostic import _prepare_loaded_bc_reference
        from test_v2_spike_regularization import TestSpikeRegularization
        for model in ('ann', 'snn'):
            with self.subTest(model=model):
                e = self.engine() if model == 'ann' else TestSpikeRegularization().engine(delay=2)
                e.bc_reference_actor = e._frozen_target(e.actor, device=e.device)
                initial = deepcopy(e.checkpoint_state_dict())
                # Real registration guards and loader, without CPU Inductor/CUDA.
                with mock.patch('torch.compile', new=lambda fn, **kwargs: fn):
                    e.configure_compilation(compile_actors=True, compile_shared_relations=True)
                    models = tuple(getattr(e, name) for name in ('actor', 'actor_target',
                        'critic1', 'critic2', 'critic1_target', 'critic2_target'))
                    online_wrapper = (e.actor._compiled_full_forward if model == 'ann'
                        else e.actor.zone_set_encoder._compiled_tensor_forward)
                    relations = e.critic1.zone_set_encoder._compiled_shared_relations
                    old_bc = e.bc_reference_actor
                    e.load_checkpoint_state_dict(initial)
                    self.assertIsNot(e.bc_reference_actor, old_bc)
                    e.actor.eval(); e.critic1.eval(); e.critic2.eval()
                    batch = e.replay.sample(e.batch_size).obs
                    with mock.patch.object(e, 'configure_compilation', side_effect=AssertionError('duplicate registration')):
                        _prepare_loaded_bc_reference(e, (batch,), dict(compile_actors=True,
                            cuda_graph_actor_update=False))
                    for name, obj in zip(('actor', 'actor_target', 'critic1', 'critic2',
                            'critic1_target', 'critic2_target'), models):
                        self.assertIs(getattr(e, name), obj)
                    self.assertIs(e.critic1.zone_set_encoder._compiled_shared_relations, relations)
                    self.assertIs(e.actor._compiled_full_forward if model == 'ann'
                        else e.actor.zone_set_encoder._compiled_tensor_forward, online_wrapper)
                    bc = e.bc_reference_actor
                    wrapper = (bc._compiled_full_forward if model == 'ann'
                        else bc.zone_set_encoder._compiled_tensor_forward)
                    self.assertIs(wrapper.__self__, bc if model == 'ann' else bc.zone_set_encoder)
                    self.assertEqual((bc.compiled_full_forward_config if model == 'ann'
                        else bc.zone_set_encoder.compiled_tensor_forward_config)['backend'], 'inductor')
                    self.assertFalse(bc.training)
                    self.assertTrue(all(getattr(e,n).training for n in ('actor','critic1','critic2')))
                    for name in ('actor_target','critic1_target','critic2_target','bc_reference_actor'):
                        self.assertFalse(getattr(e,name).training)
                        self.assertTrue(all(not p.requires_grad for p in getattr(e,name).parameters()))
                    self.assertTrue(all(not p.requires_grad for p in bc.parameters()))
                    self.assertIsNotNone(e.update_once(total_steps=2, bc_lambda=.2))
                    e.load_checkpoint_state_dict(initial)
                    e.gpu_td3_compute = True
                    e.actor.eval(); e.critic1.eval(); e.critic2.eval()
                    with mock.patch('torch.compile', side_effect=AssertionError('unused BC compile')):
                        _prepare_loaded_bc_reference(e, (batch,), dict(compile_actors=True,
                            cuda_graph_actor_update=True))
                    bc = e.bc_reference_actor
                    self.assertTrue(all(getattr(e,n).training for n in ('actor','critic1','critic2')))
                    self.assertFalse(bc.training)
                    self.assertFalse(bc.zone_set_encoder.compiled_tensor_forward_enabled)
                    if model == 'ann':
                        self.assertFalse(bc.compiled_full_forward_enabled)

    def engine(self):
        f = Fixture(); f.setUp()
        e = f.make_engine()
        for n in (0,2):
            observation = _observation(n,scales=e.actor.scales)
            e.replay.add(observation,[0.,0.],.1,observation,False)
        return e

    def test_default_conflicts_cpu_and_checkpoint_schema(self):
        e = self.engine()
        self.assertFalse(e.gpu_td3_compute)
        self.assertNotIn('gpu_td3_compute',e.checkpoint_state_dict()['algorithm_config'])
        kw = dict(actor=e.actor,critic1=e.critic1,critic2=e.critic2,replay=e.replay,
            actor_lr=.001,critic_lr=.001,gamma=.99,tau=.1,policy_noise=.01,noise_clip=.02,
            policy_delay=2,batch_size=2,action_low=e.action_low,action_high=e.action_high)
        for extra, message in ((dict(gpu_td3_compute=1),'must be a bool'),
                (dict(gpu_td3_compute=True),'requires CUDA'),
                (dict(gpu_td3_compute=True,gpu_td3_update=True),'conflicts'),
                (dict(gpu_td3_compute=True,non_blocking_batch_transfer=True),'requires CUDA')):
            with self.subTest(extra=extra), self.assertRaisesRegex((TypeError,ValueError),message):
                V2TD3UpdateEngine(**kw,**extra)

    def test_cli_and_api_defaults_explicit(self):
        from brain_uav.scripts import train_v2_td3 as single, run_v2_td3_curriculum as curriculum, profile_v2_td3 as profile
        from brain_uav.trainers.v2_formal_training import build_v2_stage_engine
        for function in (single.run_v2_td3_stage,curriculum.run_v2_curriculum):
            self.assertIsNone(inspect.signature(function).parameters['gpu_td3_compute'].default)
        self.assertIs(inspect.signature(build_v2_stage_engine).parameters['gpu_td3_compute'].default,False)
        args = profile.build_parser().parse_args(['--model','snn','--output-dir','unused','--gpu-td3-compute'])
        self.assertTrue(args.gpu_td3_compute)
        self.assertFalse(profile.build_parser().parse_args(['--model','ann','--output-dir','unused']).gpu_td3_compute)

    def test_clone_independent_dynamic_shapes_and_masks(self):
        e = self.engine()
        batch = e.replay.sample(2)
        cloned = clone_batch(batch)
        self.assertEqual(batch.sample_failure_fraction,cloned.sample_failure_fraction)
        for a,b in zip(tensors(batch),tensors(cloned)):
            self.assertEqual(a.shape,b.shape); self.assertEqual(a.dtype,b.dtype)
            self.assertFalse(b.requires_grad)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            if a.numel(): self.assertNotEqual(a.data_ptr(),b.data_ptr())

    def test_nonfinite_gradient_unchanged_and_priority(self):
        p = torch.nn.Parameter(torch.ones(2)); p.grad = torch.tensor([float('inf'),3.])
        for loss,code in ((1.,2),(float('nan'),1)):
            before = p.grad.clone()
            error,_ = preserving_clip([p],torch.tensor(loss),torch.tensor(0),1,2,1.)
            self.assertEqual(int(error),code); self.assertTrue(torch.equal(p.grad,before))

    def test_real_common_compute_cpu_same_shapes_loss_gradient_and_update(self):
        # Exactly the same RAW arithmetic boundary, no replay adapter at all.
        from brain_uav.trainers.v2_gpu_update import DeviceAdam
        e = self.engine(); other = self.engine()
        other.load_checkpoint_state_dict(deepcopy(e.checkpoint_state_dict()))
        batch = e.replay.sample(2)
        e.replay.sample = lambda size: batch
        u = GPUCompute.__new__(GPUCompute)
        u.e = other; u.diagnostic = False; u.compute_diagnostic = False
        u.actor_params = list(other.actor.parameters())
        u.critic_params = list(other.critic1.parameters())+list(other.critic2.parameters())
        u.initialize_actor_loss_scalars()
        class CPUAdam:
            def __init__(self,opt): self.opt=opt
            def step(self,error):
                if not bool(error): self.opt.step()
        u.adams = (CPUAdam(other.actor_optimizer),CPUAdam(other.critic_optimizer))
        u.injections = torch.zeros(4)
        u.coefficients = torch.tensor([0.,0.,other.policy_noise,other.noise_clip])
        u.noise = torch.zeros_like(batch.action)
        u.failure_fraction = torch.tensor(batch.sample_failure_fraction)
        def noise(t,**kw): return torch.zeros_like(t)
        with mock.patch('torch.randn_like',new=noise):
            expected = e.update_once(total_steps=2)
        with mock.patch.object(torch.Tensor, 'new_tensor', side_effect=AssertionError('CPU scalar inside compute')):
            values = u.compute(batch,'actor')
        self.assertEqual(int(values[0]),0)
        self.assertAlmostEqual(expected.critic_loss,float(values[1]),places=5)
        self.assertAlmostEqual(expected.actor_loss,float(values[2]),places=5)
        for name in ('actor','critic1','critic2','actor_target','critic1_target','critic2_target'):
            for a,b in zip(getattr(e,name).parameters(),getattr(other,name).parameters()):
                torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
                self.assertEqual(a.grad is None,b.grad is None)
                if a.grad is not None: torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-5)
        self.assertFalse(hasattr(u,'replay'))

    def test_snn_regularization_uses_real_same_forward_and_resets(self):
        from test_v2_spike_regularization import TestSpikeRegularization
        e = TestSpikeRegularization().engine(delay=2)
        other = TestSpikeRegularization().engine(delay=2)
        options = GPUCompute.static_options(other)
        other.actor.snn_head.time_window += 1
        self.assertNotEqual(GPUCompute.static_options(other),options)
        other.actor.snn_head.time_window -= 1
        other.load_checkpoint_state_dict(deepcopy(e.checkpoint_state_dict()))
        batch = e.replay.sample(2); e.replay.sample = lambda size: batch
        u = GPUCompute.__new__(GPUCompute)
        u.e=other; u.diagnostic=False; u.compute_diagnostic=False
        u.actor_params=list(other.actor.parameters()); u.critic_params=list(other.critic1.parameters())+list(other.critic2.parameters())
        u.initialize_actor_loss_scalars()
        class CPUAdam:
            def __init__(self,opt): self.opt=opt
            def step(self,error):
                if not bool(error): self.opt.step()
        u.adams=(CPUAdam(other.actor_optimizer),CPUAdam(other.critic_optimizer))
        u.injections=torch.zeros(4); u.noise=torch.zeros_like(batch.action)
        u.coefficients=torch.tensor([0.,1.,other.policy_noise,other.noise_clip]); u.failure_fraction=torch.tensor(batch.sample_failure_fraction)
        def noise(t,**kw): return torch.zeros_like(t)
        with mock.patch('torch.randn_like',new=noise): expected=e.update_once(total_steps=75002)
        with mock.patch.object(torch.Tensor, 'new_tensor', side_effect=AssertionError('CPU scalar inside compute')):
            values=u.compute(batch,'actor')
        self.assertAlmostEqual(expected.actor_batch_spike_rate_l1,float(values[10]),places=6)
        self.assertAlmostEqual(expected.spike_regularization_loss,float(values[8]),places=6)
        for a,b in zip(e.actor.parameters(),other.actor.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-5)
        self.assertEqual(other.actor.snn_head.lif1.v,0.)
        self.assertEqual(other.actor.snn_head.lif2.v,0.)

    def test_preallocated_actor_loss_scalars_and_static_configuration_guard(self):
        from brain_uav.trainers.v2_gpu_update import GPUUpdate
        e = self.engine()
        for executor in (GPUUpdate, GPUCompute):
            for enabled in (False, True):
                with self.subTest(executor=executor.__name__, terminal=enabled):
                    e.terminal_geo_regularization_enabled = enabled
                    u = executor.__new__(executor); u.e = e
                    u.actor_params = list(e.actor.parameters())
                    u.initialize_actor_loss_scalars()
                    expected = (e.actor_rl_scale_alpha,e.terminal_geo_lambda if enabled else 0.)
                    for tensor, value in zip((u.actor_loss_alpha,u.actor_loss_terminal_lambda), expected):
                        self.assertEqual(tensor.shape, torch.Size([]))
                        self.assertEqual(tensor.dtype, u.actor_params[0].dtype)
                        self.assertEqual(tensor.device, u.actor_params[0].device)
                        torch.testing.assert_close(tensor,torch.tensor(value),rtol=0,atol=0)
                        self.assertFalse(tensor.requires_grad)
                    before = executor.static_options(e)
                    for field in ('actor_rl_scale_alpha','terminal_geo_lambda'):
                        original = getattr(e,field)
                        setattr(e,field,original+1.)
                        self.assertNotEqual(executor.static_options(e),before)
                        setattr(e,field,original)

    def test_load_invalidates_without_serializing_executor(self):
        e=self.engine(); state=deepcopy(e.checkpoint_state_dict())
        execution=mock.Mock(); e._gpu_update_executor=execution
        generation=e._gpu_update_generation
        e.load_checkpoint_state_dict(state)
        execution.invalidate.assert_called_once()
        self.assertIsNone(e._gpu_update_executor)
        self.assertGreater(e._gpu_update_generation,generation)

    def test_stage_dispatch_samples_once_then_uploads_and_generates_noise(self):
        e=self.engine(); batch=e.replay.sample(2)
        e.gpu_td3_compute=True; e._gpu_update_formal_configured=True
        calls=[]
        def sample(size): calls.append('sample'); return batch
        e.replay.sample=sample
        def noise(t,**kw): calls.append('noise'); return torch.zeros_like(t)
        class Executor:
            diagnostic=False
            last_sections={'gpu_compute_input_copy':0.,'gpu_compute_to_result':0.}
            def step_batch(self,actual,normal,total,bc,**kw):
                calls.append('compute'); self.actual=actual
                return SimpleNamespace(actor_updated=False)
        executor=Executor(); e._gpu_update_executor=executor
        with mock.patch('torch.randn_like',new=noise): e.update_once(total_steps=1)
        self.assertEqual(calls,['sample','noise','compute'])
        torch.testing.assert_close(executor.actual.reward,batch.reward,rtol=0,atol=0)

    def test_actual_capture_restore_cpu_mock_preserves_values_none_and_masks(self):
        from contextlib import ExitStack, nullcontext
        e=self.engine(); batch=e.replay.sample(2)
        u=GPUCompute.__new__(GPUCompute)
        u.e=e; u.diagnostic=False; u.compute_diagnostic=False
        u.models=(e.actor,e.critic1,e.critic2,e.actor_target,e.critic1_target,e.critic2_target,e.bc_reference_actor)
        u.actor_params=list(e.actor.parameters()); u.critic_params=list(e.critic1.parameters())+list(e.critic2.parameters())
        u.params=(u.actor_params,u.critic_params)
        u.banks=[[[torch.ones_like(p)*3 for p in params] for _ in range(2)] for params in u.params]
        class Adam:
            def __init__(self,params):
                self.params=params
                self.values=[torch.zeros_like(p) for p in params]
            def state_tensors(self): return self.values
            def step(self,error):
                with torch.no_grad():
                    for p,v in zip(self.params,self.values):
                        if p.grad is not None: p.add_(.001); v.add_(1.)
        u.adams=(Adam(u.actor_params),Adam(u.critic_params))
        u.initialize_actor_loss_scalars()
        u.graphs={}; u.tensor_evidence={}; u.graph_limit=8*(e.replay.zone_storage_capacity+1)**2
        u.zone_limit=e.replay.zone_storage_capacity; u.bound_device=torch.device('cpu')
        u.published=[0,0]; u.injections=torch.zeros(4)
        u.noise=torch.zeros_like(batch.action); u.coefficients=torch.tensor([0.,0.,.01,.02])
        u.capture_seconds=0.; u.captures=0
        u.stream=SimpleNamespace(wait_stream=lambda other: None)
        u.actor_params[0].grad=u.banks[0][0][0]
        original_ref=u.actor_params[0].grad
        before=[p.detach().clone() for p in u.persistent()]
        original_mask=batch.obs.presence_mask.clone()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(torch.cuda,'mem_get_info',return_value=(1024**3,1024**3)))
            stack.enter_context(mock.patch.object(torch.cuda,'current_stream',return_value=u.stream))
            stack.enter_context(mock.patch.object(torch.cuda,'stream',return_value=nullcontext()))
            stack.enter_context(mock.patch.object(torch.cuda,'graph',return_value=nullcontext()))
            stack.enter_context(mock.patch.object(torch.cuda,'CUDAGraph',return_value=object()))
            stack.enter_context(mock.patch.object(torch.cuda,'synchronize'))
            stack.enter_context(mock.patch.object(torch.cuda,'get_rng_state',return_value=torch.tensor([1],dtype=torch.uint8)))
            stack.enter_context(mock.patch.object(torch.cuda,'set_rng_state'))
            u.capture_shape(u.key(batch,'actor'),batch)
        for p,value in zip(u.persistent(),before): torch.testing.assert_close(p,value,rtol=0,atol=0)
        self.assertIs(u.actor_params[0].grad,original_ref)
        torch.testing.assert_close(original_ref,torch.full_like(original_ref,3.),rtol=0,atol=0)
        self.assertIsNone(u.actor_params[1].grad)
        torch.testing.assert_close(batch.obs.presence_mask,original_mask,rtol=0,atol=0)
        self.assertEqual(u.captures,1)

    def test_curriculum_passes_compute_all_three_stages(self):
        from brain_uav.config import ScenarioConfig
        from brain_uav.scripts.run_v2_td3_curriculum import run_v2_curriculum
        from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation
        from test_run_v2_td3_curriculum import _prepared_initialization, derive_validation_stage_seed
        calls=[]
        def stage(**kw):
            calls.append(kw)
            return dict(stage=kw['stage'],passed=True,checkpoint=str(kw['output']),metrics=str(kw['metrics_out']),
                steps=1,outcome_counts={'goal':1},validation_result={'passed':True},stop_reason='fixed_validation_passed')
        def pool(path,**kw):
            level=Path(path).stem
            return SimpleNamespace(curriculum_level=level,master_seed=20260904,
                stage_seed=derive_validation_stage_seed(20260904,level),scenario_count=100,content_digest=level)
        with TemporaryDirectory() as directory:
            root=Path(directory); bc=root/'bc.pt'; bc.write_bytes(b'never loaded; mock only')
            prefix='brain_uav.scripts.run_v2_td3_curriculum.'
            with mock.patch(prefix+'prepare_v2_stage_initialization',return_value=_prepared_initialization(ScenarioConfig(),source=bc)), \
                 mock.patch(prefix+'prepare_v2_validation_pools',return_value={s:root/(s+'.json') for s in ('easy','medium','hard')}), \
                 mock.patch(prefix+'load_v2_validation_pool',side_effect=pool), \
                 mock.patch(prefix+'resolve_training_device',return_value='cuda'):
                run_v2_curriculum(bc_checkpoint=bc,output_root=root/'run',validation_pool_dir=root/'pools',device='cuda',
                    seed=7,max_stage='hard',gpu_td3_compute=True,stage_runner=stage,
                    **default_v2_cuda_graph_compilation(model='ann',resolved_device='cuda'))
        self.assertEqual([c['stage'] for c in calls],['easy','medium','hard'])
        self.assertTrue(all(c['gpu_td3_compute'] and not c['gpu_td3_update'] for c in calls))
        self.assertEqual(calls[1]['init_checkpoint'],calls[0]['output'])
        self.assertEqual(calls[2]['init_checkpoint'],calls[1]['output'])

    def test_cached_binding_checks_reject_parameter_and_optimizer_replacement(self):
        e=self.engine(); u=GPUCompute.__new__(GPUCompute)
        u.e=e; u.engine_device=e.device; u.bound_device=e.device; u.generation=e._gpu_update_generation
        u.models=(e.actor,e.critic1,e.critic2,e.actor_target,e.critic1_target,e.critic2_target,e.bc_reference_actor)
        u.optimizers=(e.actor_optimizer,e.critic_optimizer)
        u.params=(list(e.actor.parameters()),list(e.critic1.parameters())+list(e.critic2.parameters()))
        u.module_bindings=[]; u.buffer_bindings=[]; u.encoder_bindings=[]; u.action_bindings=[]; u.registrations=[]
        for model in u.models:
            if model is not None:
                for parent in model.modules():
                    for name,p in parent._parameters.items():
                        if p is not None: u.registrations.append((parent,name,p,p.data_ptr(),p.shape,p.stride(),p.requires_grad))
        u.optimizer_groups=tuple(o.param_groups[0] for o in u.optimizers)
        u.optimizer_states=tuple(o.state for o in u.optimizers)
        u.optimizer_options=tuple(u.options(o) for o in u.optimizers)
        u.static_configuration=u.static_options(e)
        u.native_states=[[o.state.get(p) for p in params] for o,params in zip(u.optimizers,u.params)]
        u.state_handles=[[None]*len(params) for params in u.params]
        u.validate_binding()
        parent,name,p,*_=u.registrations[0]
        parent._parameters[name]=torch.nn.Parameter(p.detach().clone())
        with self.assertRaisesRegex(RuntimeError,'parameter/address'): u.validate_binding()
        parent._parameters[name]=p
        e.actor_optimizer.state[p]={'step':torch.tensor(0.)}
        with self.assertRaisesRegex(RuntimeError,'Adam state replaced'): u.validate_binding()

    def test_shape_keys_use_bank_addresses_not_mask_values(self):
        e = self.engine(); batch = e.replay.sample(2)
        u = GPUCompute.__new__(GPUCompute)
        u.e=e; u.bound_device=torch.device('cpu'); u.zone_limit=e.replay.zone_storage_capacity
        u.published=[0,1]
        key=u.key(batch,'actor')
        changed=replace(batch,obs=replace(batch.obs,presence_mask=~batch.obs.presence_mask))
        self.assertEqual(u.key(changed,'actor'),key)
        u.published=[1,0]
        self.assertNotEqual(u.key(batch,'actor'),key)
        self.assertNotEqual(u.key(batch,'critic_only'),key)

    def test_serial_correctness_failure_skips_timing(self):
        from brain_uav.scripts.profile_v2_td3 import build_parser
        from brain_uav.scripts.v2_gpu_compute_diagnostic import run
        with TemporaryDirectory() as directory:
            args=build_parser().parse_args(['--model','snn','--device','cuda','--gpu-compute-comparison',
                '--bc-checkpoint','bc.pt','--validation-pool','pool.json','--output-dir',directory])
            calls=[]
            def subprocess(command,**kw):
                calls.append(command)
                return SimpleNamespace(returncode=0 if len(calls)==1 else 1)
            with mock.patch('brain_uav.scripts.v2_gpu_compute_diagnostic.subprocess.run',side_effect=subprocess):
                with self.assertRaisesRegex(RuntimeError,'correctness failed'): run(args)
            self.assertEqual(len(calls),2)
            self.assertTrue(all('correctness' in call for call in calls))
            report=json.loads((Path(directory)/'summary.json').read_text())
            self.assertFalse(report['passed'])
