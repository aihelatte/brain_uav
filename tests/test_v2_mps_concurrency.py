"""CPU/mock MPS size comparison; no real workers, services or CUDA."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
from types import SimpleNamespace

from brain_uav.scripts import v2_mps_diagnostic as m
from brain_uav.scripts import v2_concurrent_diagnostic as c
from brain_uav.scripts import profile_v2_td3 as p
import test_v2_mps_diagnostic as mps_fixtures
import test_v2_concurrent_diagnostic as fixtures


def arguments(root):
    args=mps_fixtures.TestMPS().args(root)
    args.concurrency=[2,8]; args.mps_concurrency_only=True
    return args


def batch_result(count,light):
    window=dict(start=7.,end=37.)
    workers={}
    for model in ('ann','snn'):
        for i in range(count//2):
            summary=fixtures.worker_summary(window)
            run=summary['runs']['compute'][0]
            run['timing']['td3_update_breakdown']={'overall_weighted':dict(update_count=90,
                total_wall_seconds=1.,sections={key:dict(total_wall_seconds=value)
                    for key,value in dict(replay_sample=.4,batch_preparation=.5,gpu_td3_sample_to_result=.6).items()})}
            run['batch_transfer_window_completion']['td3_wall_including_final_window_wait_seconds']=1.01
            light_data=dict(by_mode={mode:dict(count=45,coefficients_seconds=.1,
                graph_replay_call_seconds=.2,result_completion_seconds=.3) for mode in ('actor','critic_only')}) if light else None
            if light: run['local_optimizations']['gpu_td3_compute']['lightweight_timing']=light_data
            worker=c.summarize_worker(summary,window)
            worker['execution']['gpu_compute_lightweight_timing']=light_data
            worker['model']=model
            workers[f'{model}_{i}']=worker
    return dict(concurrency=count,repeat=0,completed=True,diagnostic_valid=True,
        valid_for_speed_comparison=not light,instrumented=light,common_window=window,
        batch_steps_per_second=sum(w['window_steps_per_second'] for w in workers.values()),workers=workers)


class TestMPSConcurrency(TestCase):
    def test_documented_command_real_parser_and_mutually_exclusive_modes(self):
        import shlex
        text=(Path(__file__).resolve().parents[1]/'docs/v2_mps_diagnostic.md').read_text(encoding='utf-8')
        text=text.split('uav_mps_concurrency() {',1)[1]
        command=text.split('PYTHONPATH=src "$py" -u -m brain_uav.scripts.v2_mps_diagnostic',1)[1].split('2>&1 | tee',1)[0]
        argv=shlex.split(command.replace('\\\n',' ').replace('"$@"',''))
        args=m.build_parser().parse_args(argv)
        m.validate(args)
        self.assertTrue(args.mps_concurrency_only); self.assertFalse(args.launch)
        self.assertEqual(args.concurrency,[2,8]); self.assertEqual(args.repeats,2)
        self.assertEqual(args.bc_schedule,'smooth-mid'); self.assertEqual(args.measurement_seconds,30)
        for mode in ('--capability-only','--correctness-only'):
            with self.subTest(mode=mode),redirect_stderr(StringIO()),self.assertRaises(SystemExit):
                m.build_parser().parse_args(argv+[mode])

    def test_plan_only_real_build_plan_and_old_mode_restrictions(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=arguments(root)
            for model in ('ann','snn'): (root/f'{model}.pt').write_bytes(b'BC fixture')
            pool=SimpleNamespace(content_digest='digest',scenario_count=3,scenarios=[
                dict(scenario_id=str(i),payload=dict(zones=[])) for i in range(3)])
            with mock.patch.object(m,'build_parser') as parser, \
                    mock.patch('brain_uav.trainers.v2_validation.load_v2_validation_pool',return_value=pool), \
                    mock.patch.object(m,'Session') as session, mock.patch.object(c,'run_batch') as run, \
                    redirect_stdout(StringIO()) as output:
                parser.return_value.parse_args.return_value=args
                self.assertEqual(m.main([]),0)
                session.assert_not_called(); run.assert_not_called()
            plan=json.loads(output.getvalue())['plan']
            self.assertEqual(plan['mps_comparison']['conditions'],['mps'])
            self.assertEqual(plan['mps_comparison']['concurrency_order_per_phase'],[2,8,8,2])
            self.assertTrue(plan['execution_requested']['cpu_deduplication'])
            self.assertFalse(args.output_dir.exists())
            for mode in ('default','capability_only','correctness_only'):
                invalid=deepcopy(args); invalid.mps_concurrency_only=False
                setattr(invalid,mode,True)
                with self.subTest(mode=mode),self.assertRaises(ValueError): m.validate(invalid)
                invalid.concurrency=[8]; m.validate(invalid)
            for sizes in ([8],[2],[8,2],[2,8,8]):
                invalid=deepcopy(args); invalid.concurrency=sizes
                with self.subTest(sizes=sizes),self.assertRaises(ValueError): m.validate(invalid)

    def test_serial_batch_order_configuration_validity_and_failure_gates(self):
        for failure in (None,'pid','invalid_speed','cleanup'):
            with self.subTest(failure=failure),TemporaryDirectory() as tmp:
                root=Path(tmp); args=arguments(root); args.launch=True
                calls=[]; lifecycle=[]; sessions=[]
                def session_factory(where):
                    session=mock.Mock(env={'private_pipe':str(where)})
                    session.start.side_effect=lambda:lifecycle.append('start')
                    def finish():
                        lifecycle.append('finish')
                        return dict(stopped=not (failure=='cleanup' and len(sessions)>1),cleanup_error='cleanup blocked')
                    session.finish.side_effect=finish
                    sessions.append(session)
                    return session
                def run(actual,count,repeat,where,**kw):
                    self.assertEqual(lifecycle[-1],'start')
                    self.assertEqual(actual.concurrency,[count])
                    self.assertEqual(actual.seed,args.seed); self.assertEqual(actual.batch_size,args.batch_size)
                    self.assertEqual(actual.bc_schedule,args.bc_schedule)
                    self.assertEqual(actual.ann_actor_lr,args.ann_actor_lr); self.assertEqual(actual.snn_actor_lr,args.snn_actor_lr)
                    self.assertTrue(actual.non_blocking_batch_transfer); self.assertFalse(actual.gpu_compute_non_blocking_coefficients)
                    self.assertEqual(kw['worker_env'],sessions[-1].env)
                    kw['access_check']({'ready':mock.Mock(pid=12)})
                    self.assertEqual(kw['before_terminate'],sessions[-1].before_terminate)
                    self.assertEqual(kw['register_process'],sessions[-1].register)
                    calls.append((actual.gpu_compute_lightweight_timing,count,repeat))
                    result=batch_result(count,actual.gpu_compute_lightweight_timing)
                    if failure=='pid': result['diagnostic_valid']=False
                    if failure=='invalid_speed' and not actual.gpu_compute_lightweight_timing:
                        result['valid_for_speed_comparison']=False
                    return result
                with mock.patch.object(m,'build_parser') as parser, \
                        mock.patch.object(c,'build_plan',return_value={}), \
                        mock.patch.object(m,'Session',side_effect=session_factory), \
                        mock.patch.object(m,'capability',return_value=dict(passed=True)) as capability, \
                        mock.patch.object(m,'mps_processes',return_value=[]), \
                        mock.patch.object(m,'gpu_processes',return_value=set()), \
                        mock.patch.object(m,'command',return_value='GPU'), \
                        mock.patch.object(m.sys,'platform','linux'), \
                        mock.patch.object(c,'run_batch',side_effect=run), \
                        mock.patch.object(m,'ordinary_access') as ordinary, \
                        mock.patch.object(m,'run_correctness_worker') as correctness, \
                        mock.patch.dict(m.os.environ,{},clear=True),redirect_stderr(StringIO()):
                    parser.return_value.parse_args.return_value=args
                    self.assertEqual(m.main([]),0 if failure is None else 1)
                    ordinary.assert_not_called(); correctness.assert_not_called()
                    self.assertEqual(capability.call_count,2)
                expected=[(light,count,slot//2) for light in (True,False) for slot,count in enumerate((2,8,8,2))]
                self.assertEqual(calls,expected[:len(calls)])
                self.assertEqual(len(calls),8 if failure is None else 5 if failure=='invalid_speed' else 1)
                self.assertEqual(lifecycle,['start','finish']*(1+len(calls)))
                for session in sessions[1:]: session.verify.assert_called_once()
                report=c.read_json(args.output_dir/'review'/'summary.json')
                self.assertEqual(report['completed'],failure is None)
                if failure is None:
                    self.assertEqual([b['instrumented'] for b in report['batches']],[True]*4+[False]*4)
                    self.assertEqual([b['valid_for_speed_comparison'] for b in report['batches']],[False]*4+[True]*4)

    def test_model_units_weighting_and_tail_exclusion(self):
        result=batch_result(8,True)
        summary=m.concurrency_model_summary(result)
        ann=summary['models']['ann']
        self.assertEqual(ann['worker_count'],4)
        self.assertEqual(ann['completed_steps_in_window'],360)
        self.assertEqual(ann['measured_steps_including_tail'],364)
        self.assertEqual(ann['window_steps_per_second'],12.)
        self.assertEqual(summary['batch_steps_per_second'],24.)
        self.assertAlmostEqual(ann['full_step_ms_per_step_including_final_drain'],10000/91)
        self.assertAlmostEqual(ann['td3_children']['batch_preparation']['ms_per_update'],500/90)
        self.assertAlmostEqual(ann['td3_children']['batch_preparation']['ms_per_step'],500/91)
        self.assertAlmostEqual(ann['full_td3_ms_per_step_including_final_wait'],1010/91)
        self.assertAlmostEqual(ann['final_wait_ms_per_step'],10/91)
        self.assertAlmostEqual(ann['lightweight_by_mode']['actor']['ms_per_update']['graph_replay_call_seconds'],200/45)
        self.assertIsNone(m.concurrency_model_summary(batch_result(2,False))['models']['snn']['lightweight_by_mode'])

    def test_real_dispatch_worker_counts_pid_gate_and_safe_timeout_cleanup(self):
        for count in (2,8):
            for failure in ('timeout','pid','cleanup'):
                with self.subTest(count=count,failure=failure),TemporaryDirectory() as tmp:
                    root=Path(tmp); args=arguments(root)
                    session=mps_fixtures.TestMPS().session(root); session.started=True
                    workers=[]; commands=[]; waits=[]
                    def spawn(command,**kw):
                        commands.append(command)
                        alive=[True]; worker=mock.Mock(pid=100+len(workers),returncode=-15)
                        worker.poll.side_effect=lambda:None if alive[0] else -15
                        worker.wait.side_effect=lambda **kw:alive.__setitem__(0,False) or -15
                        workers.append(worker); return worker
                    def wait(*a,**kw):
                        waits.append(1)
                        if len(waits)==2: raise TimeoutError('common window timeout')
                    def clients():
                        return {**{worker.pid:99 for worker in workers},**({999:99} if failure=='pid' else {})}
                    with mock.patch.object(c.subprocess,'Popen',side_effect=spawn), \
                            mock.patch.object(c,'wait_workers',side_effect=wait), \
                            mock.patch.object(session,'clients',side_effect=clients), \
                            mock.patch.object(session,'control',return_value='700' if failure=='cleanup' else '0'):
                        report=c.run_batch(args,count,0,root/'batch',worker_env=session.env,
                            access_check=session.verify,before_terminate=session.before_terminate,register_process=session.register)
                    self.assertEqual(len(workers),count); self.assertEqual(len(session.owned),count)
                    parsed=[p.build_parser().parse_args(command[4:]) for command in commands]
                    self.assertEqual([arg.model for arg in parsed],['ann']*(count//2)+['snn']*(count//2))
                    self.assertTrue(all(arg.cpu_deduplication and arg.gpu_td3_compute and arg.non_blocking_batch_transfer for arg in parsed))
                    self.assertTrue(all(not arg.gpu_compute_non_blocking_coefficients for arg in parsed))
                    self.assertFalse(report['diagnostic_valid'])
                    self.assertEqual((root/'batch'/'other'/'control'/'window.json').exists(),failure!='pid')
                    for worker in workers:
                        if failure=='timeout': worker.terminate.assert_called_once()
                        else: worker.terminate.assert_not_called(); worker.kill.assert_not_called()
                    if failure!='timeout':
                        with mock.patch.object(session,'control') as quit_control:
                            self.assertFalse(session.finish()['stopped']); quit_control.assert_not_called()
