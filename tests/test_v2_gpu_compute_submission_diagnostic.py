"""CPU/mock observer checks; no CUDA, training or throughput conclusions."""
from contextlib import contextmanager
from contextlib import redirect_stdout
from io import StringIO
from dataclasses import dataclass
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from brain_uav.scripts import v2_batch_transfer_diagnostic as d
from brain_uav.scripts import v2_concurrent_diagnostic as c
from brain_uav.scripts import profile_v2_td3 as p
from brain_uav.trainers.v2_gpu_compute import GPUCompute
from test_v2_concurrent_diagnostic import arguments, worker_summary
import test_v2_gpu_compute as fixtures


class TestSubmissionDiagnostic(unittest.TestCase):
    def recorder(self, root, updates=2):
        recorder = d.GPUComputeSubmissionDiagnostic(updates=updates, output_dir=root,
                                                    worker_id='ann_0', model='ann')
        profiler = mock.MagicMock()
        def start():
            recorder.profiler = profiler
            recorder.activities = [torch.profiler.ProfilerActivity.CPU]
        def export(path):
            Path(path).write_text(json.dumps({'traceEvents': self.trace_events(recorder.records)}))
        recorder._start_trace = start
        profiler.export_chrome_trace.side_effect = export
        return recorder, profiler

    def operation(self, observer, *, actor=False):
        if observer is not None:
            obs = SimpleNamespace(ego_features=torch.zeros(2,3), goal_features=torch.zeros(2,3),
                                  zone_features=torch.zeros(2,1,4), presence_mask=torch.ones(2,1,dtype=torch.bool))
            observer.observe_batch(SimpleNamespace(obs=obs, next_obs=obs, action=torch.zeros(2,2),
                reward=torch.zeros(2,1), **{n:torch.zeros(2,1,dtype=torch.bool) for n in
                    ('done','success','near_goal','line_to_goal_safe')}))
            for name in observer.required_scopes:
                with observer.scope(name):
                    pass
        return SimpleNamespace(actor_updated=actor)

    @staticmethod
    def trace_events(records):
        events = []
        for i, record in enumerate(records):
            base = i * 100000
            events.append(dict(ph='X',cat='user_annotation',name=f"batch_diagnostic.update.{record['update_index']}",
                               ts=base,dur=90000,pid=1,tid=2))
            for j, name in enumerate(('batch_upload','coefficients_construct_and_upload')):
                start = base + 1000 + j*30000
                events.append(dict(ph='X',cat='user_annotation',name='gpu_compute_submission.'+name,
                                   ts=start,dur=20000,pid=1,tid=2))
                sizes = ([t['bytes'] for t in record['upload_sources'] if t['source_device']=='cpu' and t['bytes']]
                         if j == 0 else [record['coefficient_upload_bytes']])
                for k, size in enumerate(sizes):
                    correlation = i*100+j*20+k
                    events += [dict(ph='X',cat='cuda_runtime',name='cudaMemcpyAsync',ts=start+100+k*200,
                                    dur=50,pid=1,tid=2,args={'correlation':correlation}),
                               dict(ph='X',cat='gpu_memcpy',name='Memcpy HtoD',ts=start+200+k*200,
                                    dur=10,pid=0,tid=7,args={'bytes':size,'correlation':correlation})]
        return events

    def test_upload_attribution_requires_real_scoped_links_for_both_uploads(self):
        from copy import deepcopy
        with TemporaryDirectory() as root:
            recorder, _ = self.recorder(root)
            for i in range(2):
                recorder.run(lambda observer: self.operation(observer,actor=i==1),update_index=75001+i)
            events = self.trace_events(recorder.records)
            self.assertFalse(any('Synchronize' in e.get('name','') for e in events))
            self.assertTrue(recorder._upload_attribution(events,recorder.records)['complete'])
            cases = {}
            cases['no_memcpy'] = [e for e in events if e.get('cat')!='gpu_memcpy']
            cases['external_only'] = deepcopy(events)
            for event in cases['external_only']:
                if 'correlation' in event.get('args',{}):
                    event['args']['External id'] = event['args'].pop('correlation')
            cases['unmatched'] = deepcopy(events)
            for event in cases['unmatched']:
                if event.get('cat')=='gpu_memcpy': event['args']['correlation'] += 10000
            cases['coefficient_missing'] = [e for e in events if not
                (e.get('cat')=='gpu_memcpy' and e['args']['correlation']%100>=20)]
            cases['outside_scope'] = deepcopy(events)
            for event in cases['outside_scope']:
                if event.get('cat')=='cuda_runtime': event['tid']=999
            for name, bad in cases.items():
                with self.subTest(case=name):
                    report = recorder._upload_attribution(bad,recorder.records)
                    self.assertFalse(report['complete'])
                    self.assertTrue(any(row['status']=='unknown' for row in report['updates']))
                    check, profiler = self.recorder(Path(root)/name)
                    check.records = deepcopy(recorder.records)
                    check.profiler = profiler
                    profiler.export_chrome_trace.side_effect = lambda path: Path(path).write_text(
                        json.dumps({'traceEvents':bad}))
                    self.assertFalse(check.finish()['diagnostic_valid'])
            # Count derives from nonempty input metadata, not a hardcoded 14 copies.
            empty = deepcopy(recorder.records)
            for record in empty:
                for tensor in record['upload_sources']:
                    if tensor['name'].endswith(('zone_features','presence_mask')): tensor['bytes']=0
            report = recorder._upload_attribution(self.trace_events(empty),empty)
            self.assertTrue(report['complete'])
            self.assertEqual(report['updates'][0]['expected_nonempty_uploads'],10)
            # Real ac2g flow endpoints also establish linkage without correlation args.
            flow = deepcopy(events)
            runtimes = [e for e in flow if e.get('cat')=='cuda_runtime']
            copies = [e for e in flow if e.get('cat')=='gpu_memcpy']
            for rt,cp in zip(runtimes,copies):
                key = rt['args'].pop('correlation'); cp['args'].pop('correlation')
                flow += [dict(ph='s',cat='ac2g',id=key,pid=rt['pid'],tid=rt['tid'],ts=rt['ts']),
                         dict(ph='f',cat='ac2g',id=key,pid=cp['pid'],tid=cp['tid'],ts=cp['ts'])]
            self.assertTrue(recorder._upload_attribution(flow,recorder.records)['complete'])
            # Explicit mismatched correlation must not be rescued by a matching flow.
            for rt, cp in zip(runtimes, copies):
                rt['args']['correlation'] = 1
                cp['args']['correlation'] = 2
            self.assertFalse(recorder._upload_attribution(flow,recorder.records)['complete'])

    def test_cpu_annotations_disambiguate_identical_gpu_annotations(self):
        with TemporaryDirectory() as root:
            recorder, _ = self.recorder(root)
            recorder.run(lambda observer: self.operation(observer),update_index=75001)
            events = self.trace_events(recorder.records)
            gpu = [dict(event, cat='gpu_user_annotation') for event in events
                   if event.get('cat')=='user_annotation']
            # GPU annotations come first, with even the same timestamps/thread ids.
            self.assertTrue(recorder._upload_attribution(gpu+events,recorder.records)['complete'])
            self.assertFalse(recorder._upload_attribution(gpu+[e for e in events
                if e.get('cat')!='user_annotation'],recorder.records)['complete'])

    def test_flow_candidates_are_indexed_without_cartesian_endpoint_scan(self):
        with TemporaryDirectory() as root:
            recorder, _ = self.recorder(root)
            recorder.run(lambda observer: self.operation(observer),update_index=75001)
            events = self.trace_events(recorder.records)
            runtime = [e for e in events if e.get('cat')=='cuda_runtime']
            copies = [e for e in events if e.get('cat')=='gpu_memcpy']
            for rt, cp in zip(runtime,copies):
                key = rt['args'].pop('correlation'); cp['args'].pop('correlation')
                events += [dict(ph='s',cat='ac2g',id=key,pid=rt['pid'],tid=rt['tid'],ts=rt['ts']),
                           dict(ph='f',cat='ac2g',id=key,pid=cp['pid'],tid=cp['tid'],ts=cp['ts'])]
            # Endpoints overlap real runtime/copy intervals but have unrelated keys.
            # A start x end search visits O(n^2) pairs; indexed key intersection does not.
            for i in range(400):
                events += [dict(ph='s',cat='ac2g',id=f'start_{i}',pid=1,tid=2,ts=runtime[0]['ts']),
                           dict(ph='f',cat='ac2g',id=f'end_{i}',pid=0,tid=7,ts=copies[0]['ts'])]
            for i in range(400):
                events.append(dict(ph='X',cat='gpu_memcpy',name='Memcpy HtoD',pid=0,tid=7,
                                   ts=1000000+i,dur=1,args={'bytes':16}))
            accesses = [0]
            outside_owner_reads = [0]
            class CountedEvent(dict):
                def get(self, key, default=None):
                    accesses[0] += 1
                    if key=='args' and dict.get(self,'ts',0) >= 1000000:
                        outside_owner_reads[0] += 1
                    return super().get(key,default)
            events = [CountedEvent(e) for e in events]
            self.assertTrue(recorder._upload_attribution(events,recorder.records)['complete'])
            self.assertEqual(outside_owner_reads[0],0)
            # Count structural event accesses, not a machine-dependent timing threshold.
            self.assertLess(accesses[0],len(events)*80)

    def test_documented_shell_function_returns_real_driver_and_setup_status(self):
        git = shutil.which('git')
        bash = (Path(git).parent.parent/'usr/bin/bash.exe') if os.name == 'nt' and git else shutil.which('bash')
        if not bash or not Path(bash).is_file():
            self.skipTest('Local Bash unavailable; documented shell execution not verified.')
        doc = (Path(__file__).resolve().parents[1]/'docs/v2_gpu_compute_transfer_ab.md').read_text(encoding='utf-8')
        function = 'uav_submission_trace() {' + doc.split('uav_submission_trace() {',1)[1].split('\nuav_submission_trace\n',1)[0]
        syntax = subprocess.run([str(bash),'--noprofile','--norc','-n'],input=function,text=True,capture_output=True)
        self.assertEqual(syntax.returncode,0,syntax.stderr)
        function = function.replace('py=/home/zj/envs/uav_torch/bin/python','py=mock_driver')
        for failure, driver_rc, expected in (('none',0,0),('none',7,7),('cd',0,1),('mkdir',0,1)):
            with self.subTest(failure=failure,driver_rc=driver_rc), TemporaryDirectory() as directory:
                env = dict(os.environ, MOCK_CWD=Path(directory).as_posix(), MOCK_RC=str(driver_rc))
                env['PATH'] = str(Path(bash).parent) + os.pathsep + env['PATH']
                setup = ('cd() { builtin cd -- "$MOCK_CWD"; }\n'
                         'mock_driver() { printf "simulated driver\\n"; return "$MOCK_RC"; }\n')
                if failure == 'cd': setup += 'cd() { return 3; }\n'
                if failure == 'mkdir': setup += 'mkdir() { return 9; }\n'
                run = subprocess.run([str(bash),'--noprofile','--norc','-c',setup+function+'\nuav_submission_trace --launch\n'],
                                     env=env,text=True,capture_output=True,timeout=20)
                self.assertEqual(run.returncode,expected,run.stdout+run.stderr)
                logs = list(Path(directory).rglob('driver.log'))
                self.assertEqual(len(logs),int(failure=='none'))
                if logs:
                    log = logs[0].read_text()
                    self.assertIn(f'exit_code={driver_rc}',log)
                    self.assertIn('process_wall_seconds=',log)

    def test_limits_deadlines_crossing_and_idempotent_finish(self):
        with TemporaryDirectory() as root:
            recorder, profiler = self.recorder(root)
            clock = [8.]
            calls = []
            def update(observer):
                calls.append(observer)
                result = self.operation(observer, actor=len(calls) == 2)
                clock[0] += 1.5
                return result
            with mock.patch.object(d, 'perf_counter', side_effect=lambda: clock[0]):
                for i in range(3):
                    recorder.run(update, update_index=75001+i, window={'start':7., 'end':10.})
            self.assertEqual([o is not None for o in calls], [True,True,False])
            self.assertEqual([r['within_common_window'] for r in recorder.records], [True,False])
            self.assertEqual(recorder.records[0]['total_steps'], 75001)
            report = recorder.finish()
            self.assertFalse(report['diagnostic_valid'])
            self.assertEqual(report['valid_samples_in_window'], 1)
            self.assertIs(recorder.finish(), report)
            profiler.__exit__.assert_called_once()
            profiler.export_chrome_trace.assert_called_once()

    def test_limit_stops_profiler_before_later_updates_and_late_start(self):
        with TemporaryDirectory() as root:
            recorder, profiler = self.recorder(root)
            for index in range(4):
                recorder.run(lambda obs: self.operation(obs, actor=index % 2 == 1), update_index=index)
                if index >= 1:
                    profiler.__exit__.assert_called_once()
            self.assertEqual(len(recorder.records), 2)
            report = recorder.finish()
            self.assertTrue(report['diagnostic_valid'])
            self.assertFalse(report['valid_for_speed_comparison'])
            self.assertTrue(report['execution']['effective'])
        with TemporaryDirectory() as root:
            recorder, profiler = self.recorder(root)
            with mock.patch.object(d, 'perf_counter', return_value=11.):
                recorder.run(lambda obs: self.assertIsNone(obs), update_index=1, window={'start':7.,'end':10.})
            self.assertFalse(recorder.finish()['diagnostic_valid'])
            profiler.__enter__.assert_not_called()

    def test_exception_stop_failure_and_trace_unavailable_preserve_update_error(self):
        with TemporaryDirectory() as root:
            recorder, profiler = self.recorder(root)
            error = ValueError('original update failure')
            profiler.__exit__.side_effect = RuntimeError('stop failure')
            with self.assertRaises(ValueError) as caught:
                recorder.run(lambda obs: (_ for _ in ()).throw(error), update_index=1)
            self.assertIs(caught.exception, error)
            report = recorder.finish()
            self.assertFalse(report['diagnostic_valid'])
            self.assertIn('stop failure', report['error']['message'])
            profiler.__exit__.assert_called_once()
        with TemporaryDirectory() as root:
            recorder = d.GPUComputeSubmissionDiagnostic(updates=2, output_dir=root, worker_id='ann_0', model='ann')
            with mock.patch('torch.profiler.supported_activities', return_value={torch.profiler.ProfilerActivity.CPU}):
                with self.assertRaisesRegex(RuntimeError, 'CUDA activity'):
                    recorder.run(lambda obs: self.fail('must not run'), update_index=1)
            self.assertFalse(recorder.finish()['diagnostic_valid'])

    def test_original_nonblocking_upload_and_rng_unchanged(self):
        for enabled in (False, True):
            engine = fixtures.TestGPUCompute().engine()
            batch = engine.replay.sample(2)
            engine.gpu_td3_compute = engine.non_blocking_batch_transfer = True
            engine._gpu_update_formal_configured = True
            calls = []
            class Source:
                def to(self, device, **kw):
                    calls.append(('original_upload',kw))
                    return batch
            engine.replay.sample = lambda _: Source()
            def step(*args, **kw):
                calls.append(('compute', kw))
                return SimpleNamespace(actor_updated=False)
            engine._gpu_update_executor = SimpleNamespace(diagnostic=False, step_batch=step)
            observer = mock.Mock()
            observer.scope.side_effect = lambda name: mock.MagicMock()
            with mock.patch('torch.cuda.current_stream'), mock.patch('torch.randn_like', wraps=torch.randn_like) as noise:
                engine.update_once(total_steps=75001, **({'gpu_compute_submission_diagnostic':observer} if enabled else {}))
                noise.assert_called_once_with(batch.action)
            self.assertEqual(calls[0], ('original_upload', {'non_blocking':True}))
            self.assertEqual('submission_diagnostic' in calls[1][1], enabled)
            self.assertEqual(observer.observe_batch.call_count, int(enabled))

    def test_real_step_batch_scopes_preserve_cpu_operation_order(self):
        @dataclass
        class Metric:
            sample_failure_fraction: float = 0.
        for enabled in (False, True):
            engine = fixtures.TestGPUCompute().engine()
            batch = engine.replay.sample(2)
            executor = GPUCompute.__new__(GPUCompute)
            executor.e = engine
            executor.validate_binding = lambda: None
            executor.key = lambda batch, mode: 'existing'
            executor.noise = torch.zeros_like(batch.action)
            executor.coefficients = torch.zeros(4)
            graph = mock.Mock()
            executor.graphs = {'existing': (graph,batch,torch.zeros(()),torch.zeros(10),([],[]),{})}
            executor.copy_seconds = executor.compute_seconds = executor.ready_to_result_seconds = 0.
            executor.replays = executor.handoffs = 0
            executor.mode_replays = dict(critic_only=0, frozen_targets=0, actor=0)
            executor.adams = [SimpleNamespace(params=[],states=[]),SimpleNamespace(params=[],states=[])]
            executor.metrics = lambda *args: Metric()
            names = []
            class Observer:
                @contextmanager
                def scope(self, name):
                    names.append(name)
                    yield
            result = executor.step_batch(batch, torch.ones_like(batch.action),75001,30.,
                                          **({'submission_diagnostic':Observer()} if enabled else {}))
            self.assertEqual(names, list(d.GPUComputeSubmissionDiagnostic.required_scopes[2:]) if enabled else [])
            self.assertTrue(torch.equal(executor.coefficients,
                torch.tensor([30.,0.,engine.policy_noise,engine.noise_clip])))
            self.assertEqual(executor.replays, 1)
            graph.replay.assert_called_once()
            self.assertEqual(result.sample_failure_fraction, batch.sample_failure_fraction)

    def test_cli_plan_forwarding_and_independent_validity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            self.assertEqual(args.gpu_compute_submission_trace_updates,0)
            args.gpu_compute_submission_trace_updates = 4
            with self.assertRaisesRegex(ValueError,'real nonblocking'):
                c.build_plan(args)
            args.non_blocking_batch_transfer = True
            for model in ('ann','snn'):
                (root/f'{model}.pt').write_bytes(b'fixture')
            pool = SimpleNamespace(content_digest='digest', scenario_count=3,
                                   scenarios=[{'scenario_id':str(i),'payload':{'zones':[]}} for i in range(3)])
            with mock.patch('brain_uav.trainers.v2_validation.load_v2_validation_pool', return_value=pool):
                self.assertEqual(c.build_plan(args)['batch_preparation_diagnostic']['mode'],'gpu_compute_submission_trace')
            for worker in ('ann_0','snn_0','ann_1'):
                parsed = p.build_parser().parse_args(c.worker_command(args,worker.split('_')[0],worker,root)[4:])
                self.assertEqual(parsed.gpu_compute_submission_trace_updates,4 if worker.endswith('_0') else 0)
                self.assertTrue(parsed.non_blocking_batch_transfer)
            window = {'start':7.,'end':17.}
            summary = worker_summary(window)
            summary['runs']['compute'][0]['batch_preparation_diagnostic'] = {
                'diagnostic_valid':True, 'invalid_reasons':[], 'instrumented':True,
                'valid_for_speed_comparison':False, 'mode':'gpu_compute_submission_trace'}
            result = c.summarize_worker(summary,window)
            self.assertTrue(result['diagnostic_valid'])
            self.assertFalse(result['valid_for_speed_comparison'])

    def test_real_cpu_profiler_stops_before_outside_allocations(self):
        with TemporaryDirectory() as root:
            recorder = d.GPUComputeSubmissionDiagnostic(updates=2, output_dir=root, worker_id='ann_0', model='ann')
            def start_cpu():
                recorder.activities = [torch.profiler.ProfilerActivity.CPU]
                recorder.profiler = torch.profiler.profile(activities=recorder.activities, profile_memory=True)
                recorder.profiler.__enter__()
            recorder._start_trace = start_cpu
            for index in range(2):
                recorder.run(lambda obs: self.operation(obs, actor=index == 1), update_index=index)
            profiler = recorder.profiler
            self.assertTrue(recorder.trace_stopped)
            before = [e for e in profiler.profiler.kineto_results.events() if e.name() == '[memory]']
            self.assertTrue(before)
            with torch.profiler.record_function('outside_submission_samples'):
                allocations = [torch.empty(2048+i) for i in range(8)]
            del allocations
            after = [e for e in profiler.profiler.kineto_results.events() if e.name() == '[memory]']
            self.assertEqual(len(before), len(after))
            self.assertFalse(recorder.finish()['diagnostic_valid'])  # CPU trace lacks CUDA evidence
            events = json.loads((Path(root)/'other/gpu_compute_submission_trace.json').read_text())['traceEvents']
            self.assertFalse(any(e.get('name') == 'outside_submission_samples' for e in events))

    def test_worker_instantiates_observer_only_after_warmup_loop_boundary(self):
        from test_v2_bc import make_scenario_config
        class ReachedLoop(Exception): pass
        with TemporaryDirectory() as root:
            args = arguments(Path(root))
            args.non_blocking_batch_transfer = True
            args.gpu_compute_submission_trace_updates = 4
            parsed = p.build_parser().parse_args(c.worker_command(args,'ann','ann_0',root)[4:])
            prepared = SimpleNamespace(scenario_config=make_scenario_config(),
                reward_config=SimpleNamespace(), uav_collision_radius=0.)
            pool = SimpleNamespace(content_digest='digest', scenario_count=3, master_seed=20260904,
                stage_seed=1, scenario_config=prepared.scenario_config, uav_collision_radius=0., scenarios=[{}]*3)
            with mock.patch.object(p,'prepare_v2_stage_initialization',return_value=prepared), \
                    mock.patch.object(p,'resolve_training_device',return_value='cuda'), \
                    mock.patch.object(p,'load_v2_validation_pool',return_value=pool), \
                    mock.patch.object(p,'V2ValidationPool',side_effect=lambda **kw: SimpleNamespace(**kw)), \
                    mock.patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')), \
                    mock.patch.object(p,'_run_diagnostic_level',side_effect=ReachedLoop) as loop:
                with self.assertRaises(ReachedLoop):
                    p.run_local_optimization_comparison(parsed)
            self.assertTrue(loop.call_args.kwargs['gpu_td3_compute'])
            self.assertTrue(loop.call_args.kwargs['non_blocking_batch_transfer'])
            self.assertIsInstance(loop.call_args.kwargs['batch_preparation_diagnostic'],d.GPUComputeSubmissionDiagnostic)
            self.assertIsNone(loop.call_args.kwargs['batch_preparation_diagnostic'].profiler)

    def test_default_plan_never_launches_and_instrumented_conditions_continue(self):
        for launch in (False, True):
            with TemporaryDirectory() as root:
                args = arguments(Path(root)); args.launch = launch; args.repeats = 1
                args.non_blocking_batch_transfer = True
                args.gpu_compute_submission_trace_updates = 4
                counts = []
                def batch(args,count,repeat,root):
                    counts.append(count)
                    return dict(concurrency=count,repeat=repeat,instrumented=True,
                                diagnostic_valid=True,valid_for_speed_comparison=False)
                with mock.patch.object(c,'build_parser') as parser, \
                        mock.patch.object(c,'build_plan',return_value={}), \
                        mock.patch.object(c,'run_batch',side_effect=batch), redirect_stdout(StringIO()):
                    parser.return_value.parse_args.return_value = args
                    self.assertEqual(c.main([]),0)
                self.assertEqual(counts,[2,8] if launch else [])
