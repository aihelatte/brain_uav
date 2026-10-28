"""CPU/mock Event ordering/window tests; no CUDA execution or timing proof."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest import mock

import torch
from brain_uav.scripts import v2_gpu_compute_event_diagnostic as d
from brain_uav.scripts import v2_concurrent_diagnostic as c
from brain_uav.scripts import profile_v2_td3 as p
import test_v2_gpu_compute_coefficients as fixtures
from test_v2_concurrent_diagnostic import arguments, worker_summary


class TestEvents(unittest.TestCase):
    def observer(self, root, *, limit=32):
        return d.GPUComputeEventDiagnostic(updates=limit,output_dir=root,worker_id='ann_0',
            model='ann',scene_ids=['scene'])

    def prime(self, observer, log):
        class Event:
            def __init__(self, index): self.index=index; self.ready=True
            def record(self, stream): log.append(('record',self.index,stream))
            def query(self): log.append(('query',self.index)); return self.ready
            def elapsed_time(self, other):
                log.append(('elapsed',self.index,other.index))
                return {(0,1):1.25,(1,2):3.75}[(self.index,other.index)]
        events=[Event(i) for i in range(3)]
        with mock.patch.object(torch.cuda,'Event',side_effect=events) as create, \
                mock.patch.object(torch.cuda,'current_stream',return_value='stream'):
            observer.prime('cuda')
            self.assertEqual(create.call_count,3)
            for call in create.call_args_list: self.assertEqual(call.kwargs,dict(enable_timing=True))
            observer.arm(dict(start=7.,end=11.5))
        observer.set_context('scene')
        log.clear()
        return events

    def test_default_off_and_order_same_stream_reuse_real_step(self):
        with TemporaryDirectory() as tmp:
            observer=self.observer(Path(tmp)); log=[]
            events=self.prime(observer,log)
            executor,batch,graph=fixtures.TestCoefficients().executor(timing=True)
            executor.e._gpu_compute_event_diagnostic=observer
            graph.replay.side_effect=lambda:log.append(('replay',))
            original=torch._foreach_copy_
            def copy(*args,**kwargs): log.append(('copy',)); return original(*args,**kwargs)
            class Output:
                def cpu(self): log.append(('cpu',)); return self
                def tolist(self): log.append(('tolist',)); return [0.]*10
            for mode,values in list(executor.graphs.items()):
                executor.graphs[mode]=(*values[:3],Output(),*values[4:])
            with mock.patch.object(d,'perf_counter',return_value=8.), \
                    mock.patch.object(torch.cuda,'current_stream',return_value='stream'), \
                    mock.patch.object(torch,'_foreach_copy_',side_effect=copy), \
                    mock.patch.object(torch.cuda,'synchronize') as sync:
                for step in (75001,75002):
                    executor.step_batch(batch,torch.zeros_like(batch.action),step,30.)
                    observer.step_completed(True)
                sync.assert_not_called()
            records=[x for x in log if x[0]=='record']
            self.assertEqual(records,[('record',i,'stream') for _ in range(2) for i in range(3)])
            first_query=next(i for i,x in enumerate(log) if x[0]=='query')
            self.assertLess(log.index(('tolist',)),first_query)
            self.assertLess(log.index(('record',0,'stream')),log.index(('copy',)))
            self.assertLess(log.index(('record',1,'stream')),log.index(('replay',)))
            self.assertLess(log.index(('replay',)),log.index(('record',2,'stream')))
            self.assertLess(log.index(('record',2,'stream')),log.index(('cpu',)))
            self.assertEqual(observer.events,events)
            self.assertEqual([x for x in log if x[0]=='elapsed'],[('elapsed',0,1),('elapsed',1,2)]*2)
            for row in observer.rows:
                self.assertEqual(row['gpu_timeline_span_ms'],dict(fixed_input=1.25,graph_replay=3.75))
            self.assertTrue(observer.finish()['diagnostic_valid'])
            observer.finish() # idempotent, no record/query/wait after export
            baseline,batch,_=fixtures.TestCoefficients().executor()
            with mock.patch.object(torch.cuda,'Event') as create:
                baseline.step_batch(batch,torch.zeros_like(batch.action),75001,30.)
            create.assert_not_called()
            self.assertFalse(hasattr(baseline.e,'_gpu_compute_event_diagnostic'))

    def test_warmup_deadline_unready_tail_limit_and_exception(self):
        with TemporaryDirectory() as tmp:
            observer=self.observer(Path(tmp),limit=2)
            _,batch,_=fixtures.TestCoefficients().executor()
            kwargs=dict(device='cuda',mode='critic_only',total_steps=75001,batch=batch,update_started=7.)
            with mock.patch.object(d,'perf_counter',return_value=8.):
                self.assertFalse(observer.begin(**kwargs)) # warmup never samples
            log=[]; events=self.prime(observer,log)
            with mock.patch.object(d,'perf_counter',return_value=12.):
                self.assertFalse(observer.begin(**kwargs)) # after action/environment crosses deadline
            with mock.patch.object(d,'perf_counter',return_value=8.), \
                    mock.patch.object(torch.cuda,'current_stream',return_value='stream'):
                self.assertTrue(observer.begin(**kwargs))
                self.assertFalse(observer.begin(**kwargs)) # no overwrite before read
            with mock.patch.object(d,'perf_counter',return_value=12.): observer.complete((.1,.2,.3))
            observer.step_completed(False)
            self.assertFalse(observer.rows[0]['valid'])
            kwargs.update(mode='actor')
            with mock.patch.object(d,'perf_counter',return_value=9.), \
                    mock.patch.object(torch.cuda,'current_stream',return_value='stream'):
                self.assertTrue(observer.begin(**kwargs))
                events[2].ready=False
                observer.complete((.1,.2,.3))
                observer.step_completed(True)
                self.assertFalse(observer.begin(**kwargs))
            self.assertNotIn('gpu_timeline_span_ms',observer.rows[-1])
            self.assertFalse(observer.finish()['diagnostic_valid'])
            self.assertEqual(len(observer.rows),2)
            failed=self.observer(Path(tmp)/'failed'); self.prime(failed,[])
            executor,batch,graph=fixtures.TestCoefficients().executor(timing=True)
            executor.e._gpu_compute_event_diagnostic=failed
            graph.replay.side_effect=KeyboardInterrupt('original')
            with mock.patch.object(d,'perf_counter',return_value=8.), \
                    mock.patch.object(torch.cuda,'current_stream',return_value='stream'):
                with self.assertRaisesRegex(KeyboardInterrupt,'original'):
                    executor.step_batch(batch,torch.zeros_like(batch.action),75001,30.)
            self.assertFalse(failed.finish()['diagnostic_valid'])
            self.assertEqual(len(failed.rows),1)

    def test_mode_time_scene_sampling_and_parent_validity(self):
        with TemporaryDirectory() as tmp:
            observer=self.observer(Path(tmp)); self.prime(observer,[])
            _,batch,_=fixtures.TestCoefficients().executor()
            with mock.patch.object(torch.cuda,'current_stream',return_value='stream'):
                for i in range(20):
                    with mock.patch.object(d,'perf_counter',return_value=7.1+i*.2):
                        if observer.begin(device='cuda',mode='actor' if i%2 else 'critic_only',
                                total_steps=75000+i,batch=batch,update_started=7.1+i*.2):
                            observer.complete((.1,.2,.3)); observer.step_completed(True)
            self.assertEqual({r['mode'] for r in observer.rows},{'actor','critic_only'})
            self.assertGreater(len(observer.rows),2)
            self.assertLessEqual(len(observer.rows),32)
            report=worker_summary(dict(start=7.,end=11.5))
            compute=report['runs']['compute'][0]['local_optimizations']['gpu_td3_compute']
            compute['event_diagnostic']=observer.finish()
            result=c.summarize_worker(report,dict(start=7.,end=11.5))
            self.assertTrue(result['diagnostic_valid']); self.assertTrue(result['instrumented'])
            self.assertFalse(result['valid_for_speed_comparison'])
            compute['event_diagnostic'].update(diagnostic_valid=False,invalid_reasons=['not ready'])
            self.assertFalse(c.summarize_worker(report,dict(start=7.,end=11.5))['diagnostic_valid'])

    def test_parser_worker_forwarding_default_plan_and_invalid_combinations(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=arguments(root)
            self.assertEqual(args.gpu_compute_event_updates,0)
            args.gpu_compute_event_updates=32
            with self.assertRaisesRegex(ValueError,'Event diagnosis'): c.build_plan(args)
            args.non_blocking_batch_transfer=True; args.gpu_compute_lightweight_timing=True
            for worker in ('ann_0','snn_0','ann_1','snn_3'):
                parsed=p.build_parser().parse_args(c.worker_command(args,worker.split('_')[0],worker,root)[4:])
                self.assertEqual(parsed.gpu_compute_event_updates,32 if worker.endswith('_0') else 0)
                self.assertTrue(parsed.gpu_compute_lightweight_timing)
                self.assertFalse(parsed.gpu_compute_non_blocking_coefficients)
            with mock.patch.object(c,'build_plan',return_value={}),mock.patch.object(c,'run_batch') as batch:
                from contextlib import redirect_stdout
                from io import StringIO
                with redirect_stdout(StringIO()):
                    c.main(['--output-dir',str(root),'--ann-bc-checkpoint','ann','--snn-bc-checkpoint','snn',
                        '--validation-pool','pool','--expected-pool-digest','digest',
                        '--ann-actor-lr','.000155','--snn-actor-lr','.000825','--bc-schedule','smooth-mid'])
                batch.assert_not_called()

    def test_event_observer_forwarded_through_real_public_worker(self):
        from test_v2_bc import make_scenario_config
        from types import SimpleNamespace
        from brain_uav.scripts import v2_gpu_compute_diagnostic as diagnostic
        class ReachedLoop(Exception): pass
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=arguments(root)
            args.gpu_compute_event_updates=32; args.non_blocking_batch_transfer=True
            args.gpu_compute_lightweight_timing=True
            parsed=p.build_parser().parse_args(c.worker_command(args,'snn','snn_0',root)[4:])
            prepared=SimpleNamespace(scenario_config=make_scenario_config(),uav_collision_radius=0.)
            pool=SimpleNamespace(content_digest='digest',scenario_count=3,master_seed=20260904,
                stage_seed=1,scenario_config=prepared.scenario_config,uav_collision_radius=0.,
                scenarios=[dict(scenario_id=str(i)) for i in range(3)])
            with mock.patch.object(p,'prepare_v2_stage_initialization',return_value=prepared), \
                    mock.patch.object(p,'resolve_training_device',return_value='cuda'), \
                    mock.patch.object(p,'load_v2_validation_pool',return_value=pool), \
                    mock.patch.object(p,'V2ValidationPool',side_effect=lambda **kw:SimpleNamespace(**kw)), \
                    mock.patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')), \
                    mock.patch.object(p,'_run_diagnostic_level',side_effect=ReachedLoop) as loop:
                with self.assertRaises(ReachedLoop): diagnostic.run(parsed)
            observer=loop.call_args.kwargs['gpu_compute_event_diagnostic']
            self.assertEqual(observer.limit,32); self.assertIsNone(observer.events)
            self.assertTrue(loop.call_args.kwargs['non_blocking_batch_transfer'])

    def test_documented_command_parses_and_instrumented_batches_continue(self):
        import shlex
        from contextlib import redirect_stdout
        from io import StringIO
        text=(Path(__file__).resolve().parents[1]/'docs/v2_gpu_compute_events.md').read_text(encoding='utf-8')
        command=text.split('PYTHONPATH=src "$py" -u -m brain_uav.scripts.v2_concurrent_diagnostic',1)[1].split('2>&1 | tee',1)[0]
        argv=shlex.split(command.replace('\\\n',' ').replace('"${launch[@]}"',''))
        args=c.build_parser().parse_args(argv)
        self.assertFalse(args.launch)
        self.assertEqual(args.concurrency,[2,8]); self.assertEqual(args.repeats,2)
        self.assertEqual(args.gpu_compute_event_updates,32)
        self.assertEqual(args.measurement_seconds,30)
        self.assertFalse(args.gpu_compute_non_blocking_coefficients)
        with TemporaryDirectory() as tmp:
            argv[argv.index('--output-dir')+1]=str(Path(tmp)/'new')
            report=dict(instrumented=True,diagnostic_valid=True,valid_for_speed_comparison=False)
            with mock.patch.object(c,'build_plan',return_value={}), \
                    mock.patch.object(c,'run_batch',return_value=report) as run, \
                    mock.patch.object(c,'summarize_batches',return_value={}), \
                    redirect_stdout(StringIO()):
                self.assertEqual(c.main(argv+['--launch']),0)
            self.assertEqual([(call.args[1],call.args[2]) for call in run.call_args_list],
                             [(2,0),(2,1),(8,0),(8,1)])

    def test_graph_cpu_timer_excludes_event_record_overhead(self):
        from brain_uav.trainers import v2_gpu_compute as gpu
        with TemporaryDirectory() as tmp:
            observer=self.observer(Path(tmp)); events=self.prime(observer,[])
            executor,batch,graph=fixtures.TestCoefficients().executor(timing=True)
            executor.e._gpu_compute_event_diagnostic=observer
            clock=[8.]
            def record(stream): clock[0]+=100.
            for event in events: event.record=record
            def replay(): clock[0]+=3.
            graph.replay.side_effect=replay
            # Diagnostic clock stays inside the shared window; CPU clock independently
            # accounts for artificial record overhead on both sides of replay.
            with mock.patch.object(d,'perf_counter',return_value=8.), \
                    mock.patch.object(gpu,'perf_counter',side_effect=lambda:clock[0]), \
                    mock.patch.object(torch.cuda,'current_stream',return_value='stream'):
                executor.step_batch(batch,torch.zeros_like(batch.action),75002,30.)
            self.assertEqual(executor.last_lightweight_timing[2],3.)
            self.assertEqual(observer.rows[0]['cpu_wall_seconds']['graph_replay_call'],3.)
            self.assertGreater(clock[0],300.)

    def test_post_submit_event_errors_finish_publication_and_preserve_update_error(self):
        from types import SimpleNamespace
        for failure in ('record','query','elapsed_time'):
            for status in (0,1,2,3,4):
                with self.subTest(failure=failure,status=status),TemporaryDirectory() as tmp:
                    observer=self.observer(Path(tmp)); events=self.prime(observer,[])
                    executor,batch,graph=fixtures.TestCoefficients().executor(timing=True)
                    executor.e._gpu_compute_event_diagnostic=observer
                    e=executor.e
                    actor_p=next(e.actor.parameters()); critic_p=next(e.critic1.parameters())
                    parameters=[actor_p,critic_p]
                    old_gradients=[torch.zeros_like(p) for p in parameters]
                    for p,g in zip(parameters,old_gradients): p.grad=g
                    banks=[torch.ones_like(p) for p in parameters]
                    old_states=[{'step':torch.tensor(1.)} for _ in range(2)]
                    new_states=[{'step':torch.tensor(2.)} for _ in range(2)]
                    optimizers=[e.actor_optimizer,e.critic_optimizer]
                    for opt,p,state in zip(optimizers,parameters,old_states): opt.state[p]=state
                    executor.params=[[p] for p in parameters]
                    executor.banks=[[[g]] for g in banks]
                    executor.published=[None,None]
                    executor.native_states=[[s] for s in old_states]
                    executor.adams=[SimpleNamespace(params=[p],states=[s],optimizer=opt)
                        for p,s,opt in zip(parameters,new_states,optimizers)]
                    key=(2,2,'actor',0,0)
                    executor.key=lambda *args:key
                    calls=[]
                    class Output:
                        def cpu(self): calls.append('cpu'); return self
                        def tolist(self):
                            calls.append('tolist')
                            values=[0.]*10; values[0]=float(status)
                            values[-1]=float(status==0); values[-2]=float(status in (0,3,4))
                            return values
                    executor.graphs={key:(graph,batch,torch.zeros(()),Output(),([True],[True]),{})}
                    def fail(*args,**kwargs): raise ValueError('Event-only failure')
                    if failure=='record': events[2].record=fail
                    elif failure=='query': events[0].query=fail
                    else: events[0].elapsed_time=fail
                    before=(e.update_count,e.critic_update_count,e.actor_update_count,e.critic_target_update_count)
                    expected_exception=RuntimeError if status==0 else FloatingPointError
                    expected_message='after original update cleanup' if status==0 else 'Non-finite'
                    with mock.patch.object(d,'perf_counter',return_value=8.), \
                            mock.patch.object(torch.cuda,'current_stream',return_value='stream'), \
                            mock.patch.object(torch.cuda,'synchronize') as sync:
                        with self.assertRaisesRegex(expected_exception,expected_message):
                            executor.step_batch(batch,torch.zeros_like(batch.action),75002,30.)
                        sync.assert_not_called()
                        replay_count=graph.replay.call_count
                        with self.assertRaisesRegex(RuntimeError,'Prior Event diagnostic failed'):
                            executor.step_batch(batch,torch.zeros_like(batch.action),75004,30.)
                        self.assertEqual(graph.replay.call_count,replay_count)
                    self.assertEqual(calls,['cpu','tolist'])
                    self.assertEqual(executor.replays,1)
                    self.assertEqual(executor.mode_replays['actor'],1)
                    self.assertEqual(executor.handoffs,2)
                    self.assertEqual(e.update_count,before[0]+int(status==0))
                    self.assertEqual(e.critic_update_count,before[1]+int(status==0))
                    self.assertEqual(e.actor_update_count,before[2]+int(status==0))
                    self.assertEqual(e.critic_target_update_count,before[3]+int(status in (0,3,4)))
                    for group,(p,opt) in enumerate(zip(parameters,optimizers)):
                        committed=status==0 if group==0 else status in (0,3,4)
                        self.assertIs(p.grad,banks[group] if committed else old_gradients[group])
                        self.assertIs(opt.state[p],new_states[group] if committed else old_states[group])
                        self.assertIs(executor.native_states[group][0],new_states[group] if committed else old_states[group])
                    self.assertTrue(observer.failed)
                    self.assertTrue(all(not r['valid'] for r in observer.rows))
                    self.assertFalse(observer.finish()['diagnostic_valid'])

    def test_real_observer_attaches_to_real_engine_via_public_worker(self):
        from test_v2_bc import make_scenario_config
        from types import SimpleNamespace
        from brain_uav.scripts import v2_gpu_compute_diagnostic as diagnostic
        class Attached(Exception): pass
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=arguments(root)
            args.gpu_compute_event_updates=32; args.non_blocking_batch_transfer=True
            args.gpu_compute_lightweight_timing=True
            parsed=p.build_parser().parse_args(c.worker_command(args,'snn','snn_0',root)[4:])
            prepared=SimpleNamespace(scenario_config=make_scenario_config(),uav_collision_radius=0.,reward_config=object())
            pool=SimpleNamespace(content_digest='digest',scenario_count=3,master_seed=20260904,
                stage_seed=1,scenario_config=prepared.scenario_config,uav_collision_radius=0.,
                scenarios=[dict(scenario_id=str(i)) for i in range(3)])
            executor,_,_=fixtures.TestCoefficients().executor(timing=True)
            engine=executor.e  # Real V2TD3UpdateEngine, not a namespace/test-double.
            for record in pool.scenarios: record['payload'] = {'zones': []}
            self.assertFalse(hasattr(engine,'_gpu_compute_event_diagnostic'))
            def environment(*args,**kwargs):
                observer=engine._gpu_compute_event_diagnostic
                self.assertIsInstance(observer,d.GPUComputeEventDiagnostic)
                self.assertEqual(observer.scene_ids,{'0','1','2'})
                self.assertEqual(observer.limit,32)
                self.assertIsNone(observer.events)
                raise Attached
            with mock.patch.object(p,'prepare_v2_stage_initialization',return_value=prepared), \
                    mock.patch.object(p,'resolve_training_device',return_value='cuda'), \
                    mock.patch.object(p,'load_v2_validation_pool',return_value=pool), \
                    mock.patch.object(p,'V2ValidationPool',side_effect=lambda **kw:SimpleNamespace(**kw)), \
                    mock.patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')), \
                    mock.patch.object(p,'build_v2_stage_engine',return_value=SimpleNamespace(engine=engine)), \
                    mock.patch.object(p,'V2StaticNoFlyTrajectoryEnv',side_effect=environment):
                # Neither the observer constructor nor _run_diagnostic_level is mocked.
                with self.assertRaises(Attached): diagnostic.run(parsed)

    def test_missing_one_of_multiple_scenes_invalidates_event_diagnosis(self):
        with TemporaryDirectory() as tmp:
            observer=d.GPUComputeEventDiagnostic(updates=32,output_dir=Path(tmp),worker_id='ann_0',
                                                model='ann',scene_ids=['scene','missing'])
            self.prime(observer,[])
            _,batch,_=fixtures.TestCoefficients().executor()
            with mock.patch.object(torch.cuda,'current_stream',return_value='stream'), \
                    mock.patch.object(d,'perf_counter',return_value=8.):
                for mode in ('actor','critic_only'):
                    self.assertTrue(observer.begin(device='cuda',mode=mode,total_steps=75002,batch=batch,update_started=8.))
                    observer.complete((.1,.2,.3)); observer.step_completed(True)
            result=observer.finish()
            self.assertEqual(result['valid_samples'],2)
            self.assertFalse(result['diagnostic_valid'])
            self.assertTrue(any('all selected scenes' in reason for reason in result['invalid_reasons']))
