"""CPU/mock only: coefficient experiment wiring, ownership and wall timers."""
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock
import gc
import weakref
import inspect
import os
import subprocess
import torch

from brain_uav.trainers.v2_gpu_compute import GPUCompute
from brain_uav.scripts import profile_v2_td3 as p
from brain_uav.scripts import v2_gpu_compute_diagnostic as d
from brain_uav.scripts import v2_concurrent_diagnostic as c
from brain_uav.trainers.v2_td3 import V2SpikeRegularizationConfig
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from brain_uav.v2_curriculum import v2_bc_lambda
from test_v2_concurrent_diagnostic import arguments, worker_summary
import test_v2_gpu_compute as fixtures


class TestCoefficients(TestCase):
    def test_defaults_invalid_combinations_and_no_checkpoint_fields(self):
        from brain_uav.scripts import train_v2_td3 as single
        engine=fixtures.TestGPUCompute().engine()
        kwargs=dict(actor=engine.actor,critic1=engine.critic1,critic2=engine.critic2,replay=engine.replay,
            actor_lr=.001,critic_lr=.001,gamma=.99,tau=.1,policy_noise=.01,noise_clip=.02,
            policy_delay=2,batch_size=2,action_low=engine.action_low,action_high=engine.action_high)
        for name in ('gpu_compute_non_blocking_coefficients','gpu_compute_lightweight_timing'):
            self.assertFalse(getattr(engine,name))
            self.assertNotIn(name,engine.checkpoint_state_dict()['algorithm_config'])
            self.assertNotIn(name,inspect.signature(single.run_v2_td3_stage).parameters)
            with self.assertRaisesRegex(ValueError,'requires GPUCompute'):
                V2TD3UpdateEngine(**kwargs,**{name:True})
        base=['--device','cuda','--model','ann','--bc-checkpoint','bc.pt','--validation-pool','pool.json',
              '--output-dir','unused','--gpu-compute-worker','correctness','--gpu-td3-compute',
              '--non-blocking-batch-transfer']
        for extra in (['--gpu-compute-non-blocking-coefficients'],['--gpu-compute-lightweight-timing'],
                      ['--gpu-compute-coefficient-comparison','--gpu-compute-write-reference','--gpu-compute-non-blocking-coefficients']):
            with self.assertRaises(ValueError): d.run(p.build_parser().parse_args(base+extra))
        for extra in (['--gpu-compute-coefficient-comparison','--gpu-compute-write-reference'],
                      ['--gpu-compute-coefficient-comparison','--gpu-compute-reference','reference.pt','--gpu-compute-non-blocking-coefficients']):
            with mock.patch.object(p,'run_local_optimization_comparison',return_value={}):
                d.run(p.build_parser().parse_args(base+extra))
    def test_correctness_public_entry_reference_write_read_and_rejections(self):
        from test_v2_bc import make_scenario_config
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root/'reference.pt'
            contract = {'test': 'coefficient entry'}
            base = ['--device','cuda','--model','ann','--bc-checkpoint','bc.pt',
                    '--validation-pool','pool.json','--gpu-compute-worker','correctness',
                    '--gpu-td3-compute','--non-blocking-batch-transfer',
                    '--gpu-compute-coefficient-comparison']
            prepared = SimpleNamespace(scenario_config=make_scenario_config(),
                uav_collision_radius=0.)
            pool = SimpleNamespace(content_digest='digest', scenario_count=3,
                master_seed=20260904, stage_seed=1, scenario_config=prepared.scenario_config,
                uav_collision_radius=0., scenarios=[{}]*3)
            def correctness(args, **kwargs):
                if args.gpu_compute_write_reference:
                    self.assertFalse(args.gpu_compute_non_blocking_coefficients)
                    torch.save(dict(contract=contract,
                        coefficient_reference=dict(gpu_td3_compute=True,
                            non_blocking_batch_transfer=True,gpu_compute_non_blocking_coefficients=False),
                        transfer_reference=dict(gpu_td3_compute=True,non_blocking_batch_transfer=True)),
                        reference_path)
                else:
                    self.assertTrue(args.gpu_compute_non_blocking_coefficients)
                    d._load_compute_reference(args, contract)
                return {'passed': True}
            def parse(extra, name):
                return p.build_parser().parse_args(base+['--output-dir',str(root/name)]+extra)
            with mock.patch.object(p,'prepare_v2_stage_initialization',return_value=prepared), \
                    mock.patch.object(p,'resolve_training_device',return_value='cuda'), \
                    mock.patch.object(p,'load_v2_validation_pool',return_value=pool), \
                    mock.patch.object(p,'V2ValidationPool',side_effect=lambda **kw:SimpleNamespace(**kw)), \
                    mock.patch.object(d,'correctness',side_effect=correctness) as reached:
                self.assertTrue(d.run(parse(['--gpu-compute-write-reference'],'a'))['passed'])
                b = ['--gpu-compute-reference',str(reference_path),'--gpu-compute-non-blocking-coefficients']
                self.assertTrue(d.run(parse(b,'b'))['passed'])
                self.assertEqual(reached.call_count,2)
                # Reference provenance remains checked after the real public entry dispatch.
                torch.save(dict(contract=contract,transfer_reference=dict(
                    gpu_td3_compute=True,non_blocking_batch_transfer=False)),reference_path)
                with self.assertRaisesRegex(ValueError,'Coefficient'):
                    d.run(parse(b,'old_reference'))
                for name, changes in (
                        ('missing_contract',dict(gpu_compute_coefficient_comparison=False)),
                        ('blocking_batch',dict(non_blocking_batch_transfer=False)),
                        ('no_compute',dict(gpu_td3_compute=False)),
                        ('no_reference',dict(gpu_compute_reference=None)),
                        ('direct_reference_write',dict(gpu_compute_write_reference=True,gpu_compute_reference=None)),
                        ('light_correctness',dict(gpu_compute_lightweight_timing=True))):
                    args = parse(b,name)
                    for key,value in changes.items(): setattr(args,key,value)
                    before = reached.call_count
                    with self.subTest(name=name), self.assertRaises(ValueError): d.run(args)
                    self.assertEqual(reached.call_count,before)
                args = parse([], 'light_without_window')
                args.gpu_compute_worker='timing'
                args.gpu_compute_coefficient_comparison=False
                args.gpu_compute_lightweight_timing=True
                with self.assertRaisesRegex(ValueError,'common concurrent measurement window'):
                    d.run(args)

    def executor(self, *, direct=False, timing=False):
        @dataclass
        class Metric:
            sample_failure_fraction: float = 0.
        engine = fixtures.TestGPUCompute().engine()
        engine.gpu_td3_compute = engine.non_blocking_batch_transfer = True
        engine.actor_freeze_steps = 25000
        engine.gpu_compute_non_blocking_coefficients = direct
        engine.gpu_compute_lightweight_timing = timing
        engine.spike_regularization = V2SpikeRegularizationConfig(beta_max=1.)
        executor = GPUCompute.__new__(GPUCompute)
        executor.e = engine
        executor.validate_binding = lambda: None
        executor.key = lambda batch,mode: mode
        executor.noise = torch.zeros(2,2)
        executor.coefficients = torch.zeros(4)
        batch = engine.replay.sample(2)
        graph = mock.Mock()
        output = torch.zeros(10)
        executor.graphs = {mode:(graph,batch,torch.zeros(()),output,([],[]),{})
                           for mode in ('critic_only','frozen_targets','actor')}
        executor.copy_seconds = executor.compute_seconds = executor.ready_to_result_seconds = 0.
        executor.captures = executor.replays = executor.handoffs = 0
        executor.mode_replays = dict(critic_only=0,frozen_targets=0,actor=0)
        executor.adams = [SimpleNamespace(params=[],states=[]),SimpleNamespace(params=[],states=[])]
        executor.metrics = lambda *args: Metric()
        executor.begin_measurement()
        return executor,batch,graph

    def test_four_fp32_values_modes_schedules_and_unchanged_gpu_address(self):
        for direct in (False,True):
            executor,batch,_ = self.executor(direct=direct)
            pointer = executor.coefficients.data_ptr()
            for step in (1,25000,25001,25002,74999,75000,75001,150000,150001,200001):
                bc = v2_bc_lambda(step-1,schedule='smooth-mid')
                expected = torch.tensor([bc,executor.e.spike_regularization.beta(step),
                                        executor.e.policy_noise,executor.e.noise_clip],dtype=torch.float32)
                with mock.patch('torch.cuda.current_stream') as stream:
                    executor.step_batch(batch,torch.zeros_like(batch.action),step,bc)
                    stream.return_value.synchronize.assert_not_called()
                self.assertTrue(torch.equal(executor.coefficients,expected))
                self.assertEqual(executor.coefficients.data_ptr(),pointer)
            self.assertEqual(getattr(executor.e,'_gpu_compute_coefficient_uploads',0),10 if direct else 0)
            self.assertEqual(set(k for k,v in executor.mode_replays.items() if v),
                             {'critic_only','frozen_targets','actor'})
            self.assertEqual(executor.e.local_optimization_metadata()['gpu_compute_non_blocking_coefficients']['effective'],direct)

    def test_source_owned_through_original_barrier_and_failed_nested_drain(self):
        for failure in (None,'capture','interrupt','drain'):
            executor,batch,graph = self.executor(direct=True)
            refs = []
            class Destination:
                def copy_(self, source, **kwargs):
                    self_outer.assertEqual(kwargs,{'non_blocking':True})
                    self_outer.assertEqual(source.dtype,torch.float32)
                    self_outer.assertEqual(source.device.type,'cpu')
                    refs.append(weakref.ref(source))
            self_outer = self
            executor.coefficients = Destination()
            original_graphs = executor.graphs
            if failure=='capture':
                executor.graphs = {}
                def capture(*args):
                    self.assertIsNotNone(refs[0]())
                    raise ValueError('capture failure')
                executor.capture_shape = capture
            def replay():
                self.assertIsNotNone(refs[0]())
                if failure in ('interrupt','drain'): raise KeyboardInterrupt('interrupted')
            graph.replay.side_effect = replay
            class Output:
                def cpu(self):
                    self_outer.assertIsNotNone(refs[0]())
                    return self
                def tolist(self):
                    self_outer.assertIsNotNone(refs[0]())
                    return [0.]*10
            for mode, data in list(original_graphs.items()):
                original_graphs[mode] = (*data[:3],Output(),*data[4:])
            stream = mock.Mock()
            def drain():
                self.assertIsNotNone(refs[0]())
                if failure=='drain': raise RuntimeError('drain failure')
            stream.synchronize.side_effect = drain
            with mock.patch('torch.cuda.current_stream',return_value=stream):
                try:
                    with executor.e._gpu_compute_source_lifetime(batch):
                        executor.step_batch(batch,torch.zeros_like(batch.action),75001,30.)
                except BaseException as exc:
                    self.assertIsNotNone(failure)
                    self.assertIn(type(exc),(ValueError,KeyboardInterrupt,RuntimeError))
            if failure=='drain':
                gc.collect()
                self.assertIsNotNone(refs[0]())
                self.assertIsNotNone(executor.e._pending_gpu_compute_transfer)
                with self.assertRaisesRegex(RuntimeError,'source retained'):
                    executor.step_batch(batch,torch.zeros_like(batch.action),75001,30.)
            else:
                gc.collect()
                self.assertIsNone(refs[0]())
            if failure is None: stream.synchronize.assert_not_called()

    def test_lightweight_modes_reset_tail_and_default_no_new_timer_reads(self):
        executor,batch,_ = self.executor(timing=True)
        clock = iter(float(i) for i in range(1000))
        with mock.patch('brain_uav.trainers.v2_gpu_compute.perf_counter',side_effect=lambda:next(clock)):
            for step in (1,2,75002,75003):
                executor.step_batch(batch,torch.zeros_like(batch.action),step,30.)
                executor.record_lightweight_step(step!=75003)
        self.assertEqual(set(executor.lightweight_timing),{'critic_only','frozen_targets','actor'})
        for row in executor.lightweight_timing.values():
            self.assertEqual(row,dict(count=1,coefficients_seconds=1.,graph_replay_call_seconds=1.,result_completion_seconds=1.))
        executor.record_lightweight_step(True)
        self.assertEqual(sum(r['count'] for r in executor.lightweight_timing.values()),3)
        executor.begin_measurement()
        self.assertEqual(executor.lightweight_timing,{})
        self.assertIsNone(executor.last_lightweight_timing)
        # Baseline perf_counter sequence remains exactly the existing six reads.
        baseline,batch,_ = self.executor()
        with mock.patch('brain_uav.trainers.v2_gpu_compute.perf_counter',side_effect=range(6)) as clock:
            baseline.step_batch(batch,torch.zeros_like(batch.action),1,30.)
        self.assertEqual(clock.call_count,6)
        self.assertEqual(set(baseline.last_sections),{'gpu_compute_input_copy','gpu_compute_to_result'})
        self.assertFalse(hasattr(baseline,'lightweight_timing'))

    def test_cli_forwarding_and_distinct_reference_provenance(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); args=arguments(root)
            self.assertFalse(args.gpu_compute_non_blocking_coefficients)
            args.gpu_compute_non_blocking_coefficients=True
            with self.assertRaisesRegex(ValueError,'nonblocking'): c.build_plan(args)
            args.non_blocking_batch_transfer=True; args.gpu_compute_lightweight_timing=True
            parsed=p.build_parser().parse_args(c.worker_command(args,'snn','snn_0',root)[4:])
            self.assertTrue(parsed.gpu_compute_non_blocking_coefficients)
            self.assertTrue(parsed.gpu_compute_lightweight_timing)
            path=root/'reference.pt'
            parsed.gpu_compute_reference=path; parsed.gpu_compute_coefficient_comparison=True
            reference={'contract':{'test':1},'coefficient_reference':dict(gpu_td3_compute=True,
                non_blocking_batch_transfer=True,gpu_compute_non_blocking_coefficients=False),
                'transfer_reference':dict(gpu_td3_compute=True,non_blocking_batch_transfer=True)}
            torch.save(reference,path)
            d._load_compute_reference(parsed,{'test':1})
            reference['coefficient_reference']['gpu_compute_non_blocking_coefficients']=True
            torch.save(reference,path)
            with self.assertRaisesRegex(ValueError,'Coefficient'): d._load_compute_reference(parsed,{'test':1})
            torch.save({'contract':{'test':1},'transfer_reference':dict(gpu_td3_compute=True,non_blocking_batch_transfer=False)},path)
            with self.assertRaisesRegex(ValueError,'Coefficient'): d._load_compute_reference(parsed,{'test':1})
            # Original batch transfer reference checking is unchanged.
            parsed.gpu_compute_coefficient_comparison=False
            d._load_compute_reference(parsed,{'test':1})
            window={'start':7.,'end':17.}
            report=worker_summary(window)
            report['runs']['compute'][0]['local_optimizations']['gpu_td3_compute']['lightweight_timing']={
                'requested':True,'by_mode':{'actor':{'count':1}}}
            result=c.summarize_worker(report,window)
            self.assertTrue(result['diagnostic_valid'])
            self.assertFalse(result['valid_for_speed_comparison'])

    def test_worker_reaches_actual_engine_configuration(self):
        from test_v2_bc import make_scenario_config
        class ReachedLoop(Exception): pass
        with TemporaryDirectory() as directory:
            root=Path(directory); args=arguments(root)
            args.non_blocking_batch_transfer=True
            args.gpu_compute_non_blocking_coefficients=True
            args.gpu_compute_lightweight_timing=True
            parsed=p.build_parser().parse_args(c.worker_command(args,'snn','snn_0',root)[4:])
            prepared=SimpleNamespace(scenario_config=make_scenario_config(),uav_collision_radius=0.,reward_config=SimpleNamespace())
            pool=SimpleNamespace(content_digest='digest',scenario_count=3,master_seed=20260904,
                stage_seed=1,scenario_config=prepared.scenario_config,uav_collision_radius=0.,scenarios=[{}]*3)
            with mock.patch.object(p,'prepare_v2_stage_initialization',return_value=prepared), \
                    mock.patch.object(p,'resolve_training_device',return_value='cuda'), \
                    mock.patch.object(p,'load_v2_validation_pool',return_value=pool), \
                    mock.patch.object(p,'V2ValidationPool',side_effect=lambda **kw:SimpleNamespace(**kw)), \
                    mock.patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')), \
                    mock.patch.object(p,'_run_diagnostic_level',side_effect=ReachedLoop) as loop:
                with self.assertRaises(ReachedLoop): p.run_local_optimization_comparison(parsed)
            for flag in ('gpu_compute_non_blocking_coefficients','gpu_compute_lightweight_timing',
                         'non_blocking_batch_transfer','gpu_td3_compute','cuda_graph_action_inference','fused_adam'):
                self.assertTrue(loop.call_args.kwargs[flag])

    def test_documented_commands_parse_plan_only_and_stop_on_failure(self):
        import shutil
        git=shutil.which('git')
        bash=Path(git).parent.parent/'usr/bin/bash.exe' if os.name=='nt' and git else shutil.which('bash')
        if not bash or not Path(bash).is_file(): self.skipTest('Local Bash unavailable')
        doc=(Path(__file__).resolve().parents[1]/'docs/v2_gpu_compute_transfer_ab.md').read_text(encoding='utf-8')
        function='uav_coeff_ab() {'+doc.split('uav_coeff_ab() {',1)[1].split('\nuav_coeff_ab\n',1)[0]
        function=function.replace('py=/home/zj/envs/uav_torch/bin/python','py=mock_driver')
        setup='''
cd() { builtin cd -- "$MOCK_CWD"; }
env() { shift; "$@"; }
mock_driver() {
  printf '%s\\0' "$@" >> "$MOCK_ARGS"
  printf '\\0' >> "$MOCK_ARGS"
  n=$(cat "$MOCK_COUNTER")
  n=$(( n + 1 )); printf '%s' "$n" > "$MOCK_COUNTER"
  if [ "$n" -eq "$FAIL_CALL" ]; then return 17; fi
  return 0
}
'''
        for launch,fail in ((False,0),(True,0),(True,3)):
            with self.subTest(launch=launch,fail=fail), TemporaryDirectory() as directory:
                root=Path(directory); counter=root/'counter'; counter.write_text('0')
                capture=root/'args'
                env=dict(os.environ,MOCK_CWD=root.as_posix(),MOCK_ARGS=capture.as_posix(),
                         MOCK_COUNTER=counter.as_posix(),FAIL_CALL=str(fail))
                env['PATH']=str(Path(bash).parent)+os.pathsep+env['PATH']
                result=subprocess.run([str(bash),'--noprofile','--norc','-c',setup+function+
                    ('\nuav_coeff_ab --launch\n' if launch else '\nuav_coeff_ab\n')],
                    env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,17 if fail else 0,result.stdout+result.stderr)
                commands=[row.decode().split('\0') for row in capture.read_bytes().split(b'\0\0') if row]
                self.assertEqual(len(commands),3 if fail else 14 if launch else 2)
                for command in commands:
                    if command[1]=='-c': continue
                    module=command[2]; argv=command[3:]
                    parsed=(c.build_parser() if module.endswith('v2_concurrent_diagnostic') else p.build_parser()).parse_args(argv)
                    self.assertTrue(parsed.non_blocking_batch_transfer)
                    if module.endswith('v2_concurrent_diagnostic'):
                        self.assertEqual(parsed.concurrency,[2,8])
                        self.assertEqual(parsed.repeats,2)
                        self.assertEqual(parsed.measurement_seconds,30)
                        if not launch: self.assertFalse(parsed.launch)
                    else:
                        self.assertTrue(parsed.gpu_compute_coefficient_comparison)
                        self.assertEqual(parsed.gpu_compute_worker,'correctness')
                if fail:
                    self.assertFalse(any('--launch' in cmd for cmd in commands))
