"""CPU/mock only; no service, CUDA context, server access or real workers."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from brain_uav.scripts import v2_mps_diagnostic as m
from brain_uav.scripts import v2_concurrent_diagnostic as c
from test_v2_concurrent_diagnostic import arguments


class TestMPS(unittest.TestCase):
    def session(self, root):
        pipe=root/'private_pipe'; pipe.mkdir()
        with mock.patch.object(m.tempfile,'mkdtemp',return_value=str(pipe)):
            return m.Session(root/'session')

    def args(self,root):
        args=arguments(root)
        args.concurrency=[8]; args.repeats=2; args.measurement_seconds=30.
        args.non_blocking_batch_transfer=True
        args.capability_only=False
        return args

    def test_plan_only_has_no_runtime_side_effects_and_rejects_conflicts(self):
        with TemporaryDirectory() as tmp:
            args=self.args(Path(tmp))
            with mock.patch.object(m,'build_parser') as parser, \
                    mock.patch.object(c,'build_plan',return_value={}), \
                    mock.patch.object(m,'Session') as session, \
                    mock.patch.object(m.subprocess,'Popen') as spawn, \
                    mock.patch.object(m,'command') as command, \
                    redirect_stdout(StringIO()) as output:
                parser.return_value.parse_args.return_value=args
                with mock.patch.dict(m.os.environ,{},clear=True): self.assertEqual(m.main([]),0)
                self.assertIn('plan_only',output.getvalue())
                session.assert_not_called(); spawn.assert_not_called(); command.assert_not_called()
            self.assertFalse(args.output_dir.exists())
            args.gpu_compute_event_updates=32
            with self.assertRaises(ValueError): m.validate(args)

    def test_pid_lists_fail_closed_and_real_membership_not_environment(self):
        self.assertEqual(m.pid_list('12\n34'),{12,34})
        for text in ('error 12','12 other','Server 12'):
            with self.assertRaises(RuntimeError): m.pid_list(text)
        with TemporaryDirectory() as tmp:
            session=self.session(Path(tmp))
            worker=mock.Mock(pid=12)
            with mock.patch.object(session,'control',side_effect=['99','12']):
                evidence=session.verify({'ann_0':worker})
            self.assertEqual(evidence['client_server_pid'],{'12':99})
            with mock.patch.object(session,'control',side_effect=['99','13']):
                with self.assertRaisesRegex(RuntimeError,'actual clients'): session.verify({'ann_0':worker})
            with mock.patch.object(m,'mps_processes',return_value=[]),mock.patch.object(m,'gpu_processes',return_value={12}):
                self.assertEqual(m.ordinary_access({'ann_0':worker})['effective'],'ordinary')
            with mock.patch.object(m,'mps_processes',return_value=['99 nvidia-cuda-mps']):
                with self.assertRaises(RuntimeError): m.ordinary_access({'ann_0':worker})

    def test_cleanup_terminates_cuda_context_before_signals_and_blocks_failure(self):
        with TemporaryDirectory() as tmp:
            session=self.session(Path(tmp)); log=[]
            process=mock.Mock(pid=12)
            process.poll.return_value=None
            process.terminate.side_effect=lambda:log.append('terminate')
            session.register({'owned':process})
            with mock.patch.object(session,'clients',return_value={12:99}), \
                    mock.patch.object(session,'control',side_effect=lambda text:log.append(text) or '0'):
                m.stop_probe(process,session)
            self.assertEqual(log,['terminate_client 99 12','terminate'])
            for clients in ({12:99,44:99},{},):
                process.reset_mock(); process.poll.return_value=None
                with mock.patch.object(session,'clients',return_value=clients),mock.patch.object(session,'control') as control:
                    with self.assertRaises(RuntimeError): m.stop_probe(process,session)
                    process.terminate.assert_not_called(); process.kill.assert_not_called(); control.assert_not_called()
            process.reset_mock(); process.poll.return_value=None
            with mock.patch.object(session,'clients',return_value={12:99}), \
                    mock.patch.object(session,'control',side_effect=RuntimeError('unsupported terminate_client')):
                with self.assertRaises(RuntimeError): m.stop_probe(process,session)
                process.terminate.assert_not_called(); process.kill.assert_not_called()
            session.started=True
            with mock.patch.object(session,'control') as control:
                self.assertFalse(session.finish()['stopped']); control.assert_not_called()

    def test_batch_timeout_registers_partial_workers_and_honors_cleanup_hook(self):
        for safe in (True,False):
            with self.subTest(safe=safe),TemporaryDirectory() as tmp:
                root=Path(tmp); args=self.args(root)
                owned=mock.Mock(pid=12,returncode=-15); owned.poll.return_value=None
                events=[]
                owned.terminate.side_effect=lambda:events.append('OS terminate')
                def cleanup(process):
                    self.assertIs(process,owned); events.append('context terminate')
                    if not safe: raise RuntimeError('context termination rejected')
                with mock.patch.object(c.subprocess,'Popen',side_effect=[owned,OSError('dispatch failure')]), \
                        mock.patch.object(c,'wait_workers',side_effect=TimeoutError('timeout')), \
                        mock.patch.object(c,'worker_command',return_value=['mock']):
                    report=c.run_batch(args,8,0,root/'batch',worker_env={'private':'env'},
                        register_process=lambda ps:events.append(('register',list(ps))),before_terminate=cleanup)
                self.assertFalse(report['diagnostic_valid'])
                self.assertEqual(events[0],('register',['ann_0']))
                if safe: self.assertEqual(events[-2:],['context terminate','OS terminate'])
                else:
                    owned.terminate.assert_not_called(); owned.kill.assert_not_called()
                    self.assertIn('cleanup_errors',report)

    def test_capability_requires_matching_graph_ready_client_and_normal_exit(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); session=self.session(root)
            process=mock.Mock(pid=12); process.poll.side_effect=[None,0,0]; process.wait.return_value=0
            def ready(processes,predicate,**kwargs):
                c.write_json(root/'probe'/'ready.json',dict(pid=12,graph_replays=8,synchronized=True))
                self.assertTrue(predicate('probe'))
            with mock.patch.object(m.subprocess,'Popen',return_value=process) as spawn, \
                    mock.patch.object(c,'wait_workers',side_effect=ready), \
                    mock.patch.object(session,'verify',return_value=dict(client_server_pid={'12':99})) as verify:
                result=m.capability(root/'probe',session)
            self.assertTrue(result['passed']); self.assertEqual(result['exit_code'],0)
            verify.assert_called_once_with({'probe':process})
            self.assertTrue((root/'probe'/'release.json').exists())
            self.assertIn('brain_uav.scripts.v2_mps_diagnostic',spawn.call_args.args[0])
            process.terminate.assert_not_called()

    def test_driver_gates_capability_and_runs_same_eight_workers_two_phases(self):
        for pass_capability in (True,False):
            with self.subTest(passed=pass_capability),TemporaryDirectory() as tmp:
                root=Path(tmp); args=self.args(root); args.launch=True
                batches=[]
                def batch(args,count,repeat,where,**kw):
                    batches.append((args.gpu_compute_lightweight_timing,count,repeat,kw))
                    return dict(diagnostic_valid=True,valid_for_speed_comparison=not args.gpu_compute_lightweight_timing)
                session=mock.Mock(env={'private':'env'})
                session.finish.return_value=dict(stopped=True)
                with mock.patch.object(m,'build_parser') as parser, \
                        mock.patch.object(c,'build_plan',return_value={}), \
                        mock.patch.object(m,'Session',return_value=session), \
                        mock.patch.object(m,'capability',side_effect=(lambda *a,**kw: {'passed':True}) if pass_capability else RuntimeError('capability failed')), \
                        mock.patch.object(m,'mps_processes',return_value=[]), \
                        mock.patch.object(m,'gpu_processes',return_value=set()), \
                        mock.patch.object(m,'command',return_value='mock GPU'), \
                        mock.patch.object(m.sys,'platform','linux'), \
                        mock.patch.object(c,'run_batch',side_effect=batch), \
                        mock.patch.object(m.tempfile,'mkdtemp',return_value='private'), \
                        mock.patch.dict(m.os.environ,{},clear=True):
                    parser.return_value.parse_args.return_value=args
                    self.assertEqual(m.main([]),0 if pass_capability else 1)
                if pass_capability:
                    self.assertEqual([(a,b,c) for a,b,c,_ in batches],
                        [(True,8,0),(True,8,1),(True,8,0),(True,8,1),
                         (False,8,0),(False,8,1),(False,8,0),(False,8,1)])
                    for index,(_,_,_,kw) in enumerate(batches):
                        self.assertEqual(kw['before_terminate'] is not None,index in (2,3,6,7))
                else: self.assertFalse(batches)
                self.assertTrue((args.output_dir/'review'/'summary.json').exists())

    def test_control_failure_is_not_accepted_as_attachment(self):
        for rc,text in ((1,''),(0,'Error: command unavailable'),(0,'Client not found')):
            with mock.patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=rc,stdout=text,stderr='')):
                with self.assertRaises(RuntimeError): m.command([m.CONTROL],text='get_client_list 1\n')

    def test_capability_timeout_retains_error_and_safe_context_cleanup(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); session=self.session(root)
            process=mock.Mock(pid=12,returncode=-15); process.poll.return_value=None
            sequence=[]
            process.terminate.side_effect=lambda:sequence.append('os terminate')
            with mock.patch.object(m.subprocess,'Popen',return_value=process), \
                    mock.patch.object(c,'wait_workers',side_effect=TimeoutError('ready timeout')), \
                    mock.patch.object(session,'before_terminate',side_effect=lambda p:sequence.append('terminate_client')):
                with self.assertRaisesRegex(TimeoutError,'ready timeout'):
                    m.capability(root/'probe',session)
            self.assertEqual(sequence,['terminate_client','os terminate'])
            result=c.read_json(root/'probe'/'summary.json')
            self.assertFalse(result['passed']); self.assertIn('ready timeout',result['error'])

    def test_service_stop_refuses_foreign_clients_and_unsupported_control(self):
        with TemporaryDirectory() as tmp:
            session=self.session(Path(tmp)); session.started=True
            with mock.patch.object(session,'clients',return_value={44:99}), \
                    mock.patch.object(session,'control') as control:
                result=session.finish()
            self.assertFalse(result['stopped']); control.assert_not_called()
            self.assertIn('Clients still connected',result['cleanup_error'])
            with mock.patch.object(session,'clients',return_value={}), \
                    mock.patch.object(session,'control',return_value='') as control:
                self.assertTrue(session.finish()['stopped'])
                control.assert_called_once_with('quit')

    def test_start_uses_real_pid_query_without_legacy_help_and_capability_only(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); session=self.session(root)
            (session.pipe/'nvidia-cuda-mps-control.pid').write_text('99')
            with mock.patch.object(m,'mps_processes',return_value=[]), \
                    mock.patch.object(m.Path,'is_file',return_value=True), \
                    mock.patch.object(m.subprocess,'run',side_effect=lambda argv,**kw: SimpleNamespace(
                        returncode=0,stdout='Unknown command "help"' if kw['input']=='help\n' else '',stderr='')) as run:
                session.start()
                self.assertEqual([call.kwargs['input'] for call in run.call_args_list],[None,'get_server_list\n'])
                for response in ('Unknown command "get_server_list"','not a PID'):
                    with mock.patch.object(session,'control',return_value=response):
                        with self.assertRaises(RuntimeError): session.start()
            args=self.args(root); args.launch=True; args.capability_only=True
            owned=mock.Mock(); owned.finish.return_value=dict(stopped=True)
            with mock.patch.object(m,'build_parser') as parser, \
                    mock.patch.object(c,'build_plan',return_value={}), \
                    mock.patch.object(m,'Session',return_value=owned), \
                    mock.patch.object(m,'capability',return_value=dict(passed=True)), \
                    mock.patch.object(m,'mps_processes',return_value=[]), \
                    mock.patch.object(m,'gpu_processes',return_value=set()), \
                    mock.patch.object(m,'command',return_value='GPU'), \
                    mock.patch.object(m.sys,'platform','linux'), \
                    mock.patch.object(c,'run_batch') as batch, \
                    mock.patch.dict(m.os.environ,{},clear=True):
                parser.return_value.parse_args.return_value=args
                self.assertEqual(m.main([]),0)
                batch.assert_not_called()

    def test_independent_termination_probe_requires_sync_pid_and_exact_zero(self):
        for response in ('0','700','','unknown',m.subprocess.TimeoutExpired('control',30)):
            with self.subTest(response=response),TemporaryDirectory() as tmp:
                root=Path(tmp); session=self.session(root); session.started=True
                process=mock.Mock(pid=12); alive=[True]
                process.poll.side_effect=lambda: None if alive[0] else -15
                process.wait.side_effect=lambda **kw: alive.__setitem__(0,False) or -15
                def ready(processes,predicate,**kwargs):
                    c.write_json(root/'termination'/'ready.json',dict(pid=12,graph_replays=8,synchronized=True))
                    self.assertTrue(predicate('probe'))
                def control(argv,**kw):
                    self.assertEqual(kw['input'],'terminate_client 99 12\n')
                    if isinstance(response,Exception): raise response
                    return SimpleNamespace(returncode=0,stdout=response,stderr='')
                with mock.patch.object(m.subprocess,'Popen',return_value=process), \
                        mock.patch.object(c,'wait_workers',side_effect=ready), \
                        mock.patch.object(session,'clients',return_value={12:99}), \
                        mock.patch.object(m.subprocess,'run',side_effect=control) as run:
                    if response=='0':
                        result=m.capability(root/'termination',session,terminate_client=True)
                        self.assertTrue(result['passed']); self.assertEqual(result['terminate_client_response'],'0')
                        process.terminate.assert_called_once()
                    else:
                        with self.assertRaises((RuntimeError,m.subprocess.TimeoutExpired)):
                            m.capability(root/'termination',session,terminate_client=True)
                        process.terminate.assert_not_called(); process.kill.assert_not_called()
                        with mock.patch.object(session,'control') as quit_control:
                            self.assertFalse(session.finish()['stopped']); quit_control.assert_not_called()
                    run.assert_called_once()  # Never retry unknown termination status.
                self.assertFalse((root/'termination'/'release.json').exists())
                self.assertEqual(c.read_json(root/'termination'/'summary.json')['passed'],response=='0')

    def test_termination_probe_never_terminates_unverified_or_unsynchronized_child(self):
        for failure in ('timeout','sync','pid','exited'):
            with self.subTest(failure=failure),TemporaryDirectory() as tmp:
                root=Path(tmp); session=self.session(root)
                process=mock.Mock(pid=12); process.poll.return_value=None
                if failure=='exited': process.poll.side_effect=[None,0,0]
                def ready(*a,**kw):
                    if failure=='timeout': raise TimeoutError('ready timeout')
                    c.write_json(root/'termination'/'ready.json',dict(pid=12,graph_replays=8,synchronized=failure!='sync'))
                with mock.patch.object(m.subprocess,'Popen',return_value=process), \
                        mock.patch.object(c,'wait_workers',side_effect=ready), \
                        mock.patch.object(session,'clients',return_value={13:99} if failure=='pid' else {12:99}), \
                        mock.patch.object(session,'before_terminate') as terminate:
                    with self.assertRaises((RuntimeError,TimeoutError)):
                        m.capability(root/'termination',session,terminate_client=True)
                    terminate.assert_not_called(); process.terminate.assert_not_called(); process.kill.assert_not_called()
                result=c.read_json(root/'termination'/'summary.json')
                self.assertFalse(result['passed'])
                if failure=='exited': self.assertIn('exited before terminate_client',result['error'])
                else: self.assertIn('no OS kill',result['cleanup_error'])

    def test_termination_probe_failure_stops_driver_and_logs_error(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=self.args(root); args.launch=True
            session=mock.Mock(env={}); session.finish.return_value=dict(stopped=False,cleanup_error='service retained')
            with mock.patch.object(m,'build_parser') as parser, \
                    mock.patch.object(c,'build_plan',return_value={}), \
                    mock.patch.object(m,'Session',return_value=session), \
                    mock.patch.object(m,'capability',side_effect=[dict(passed=True),RuntimeError('terminate_client response 700')]) as probe, \
                    mock.patch.object(m,'mps_processes',return_value=[]), \
                    mock.patch.object(m,'gpu_processes',return_value=set()), \
                    mock.patch.object(m,'command',return_value='GPU'), \
                    mock.patch.object(m.sys,'platform','linux'), \
                    mock.patch.object(c,'run_batch') as batch, \
                    mock.patch.dict(m.os.environ,{},clear=True),redirect_stderr(StringIO()) as output:
                parser.return_value.parse_args.return_value=args
                self.assertEqual(m.main([]),1)
                batch.assert_not_called()
                self.assertTrue(probe.call_args.kwargs['terminate_client'])
                self.assertIn('terminate_client response 700',output.getvalue())
                self.assertIn('service retained',output.getvalue())
            summary=c.read_json(args.output_dir/'review'/'summary.json')
            self.assertFalse(summary['completed']); self.assertFalse(summary['capability_passed'])

    def test_access_failure_prevents_common_window_release(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=self.args(root)
            worker=mock.Mock(pid=12,returncode=-15); worker.poll.return_value=None
            with mock.patch.object(c.subprocess,'Popen',return_value=worker), \
                    mock.patch.object(c,'wait_workers'), \
                    mock.patch.object(c,'worker_command',return_value=['mock']):
                report=c.run_batch(args,2,0,root/'batch',
                    access_check=mock.Mock(side_effect=RuntimeError('PID mismatch')),before_terminate=lambda p:None)
            self.assertFalse(report['completed'])
            self.assertFalse((root/'batch'/'other'/'control'/'window.json').exists())
            self.assertIn('PID mismatch',report['error']['message'])

    def test_documented_command_uses_real_cli_and_defaults_plan_only(self):
        import shlex
        text=(Path(__file__).resolve().parents[1]/'docs/v2_mps_diagnostic.md').read_text(encoding='utf-8')
        text=text.split('uav_mps_compare() {',1)[1]
        command=text.split('PYTHONPATH=src "$py" -u -m brain_uav.scripts.v2_mps_diagnostic',1)[1].split('2>&1 | tee',1)[0]
        argv=shlex.split(command.replace('\\\n',' ').replace('"$@"',''))
        args=m.build_parser().parse_args(argv)
        with mock.patch.dict(m.os.environ,{},clear=True): m.validate(args)
        self.assertFalse(args.launch); self.assertEqual(args.concurrency,[8])
        self.assertEqual(args.ann_actor_lr,1.55e-4); self.assertEqual(args.snn_actor_lr,8.25e-4)
        self.assertEqual(args.bc_schedule,'smooth-mid')

    def test_terminate_response_zero_only_gates_batch_os_signals(self):
        for response in ('0','700','','unknown response'):
            with self.subTest(response=response),TemporaryDirectory() as tmp:
                root=Path(tmp); args=self.args(root); session=self.session(root)
                session.started=True
                worker=mock.Mock(pid=12,returncode=None); worker.poll.return_value=None
                with mock.patch.object(c.subprocess,'Popen',side_effect=[worker,OSError('dispatch failed')]), \
                        mock.patch.object(c,'worker_command',return_value=['mock']), \
                        mock.patch.object(session,'clients',return_value={12:99}), \
                        mock.patch.object(m.subprocess,'run',return_value=SimpleNamespace(
                            returncode=0,stdout=response,stderr='')) as control:
                    report=c.run_batch(args,8,0,root/'batch',register_process=session.register,
                                       before_terminate=session.before_terminate)
                control.assert_called_once()
                self.assertEqual(control.call_args.kwargs['input'],'terminate_client 99 12\n')
                if response=='0':
                    worker.terminate.assert_called_once(); worker.wait.assert_called_once()
                    self.assertNotIn('cleanup_errors',report)
                else:
                    worker.terminate.assert_not_called(); worker.kill.assert_not_called()
                    self.assertIn('terminate_client',report['cleanup_errors']['ann_0'])
                    self.assertFalse(report['diagnostic_valid'])
                    with mock.patch.object(session,'control') as quit_control:
                        self.assertFalse(session.finish()['stopped'])
                        quit_control.assert_not_called()

    def test_bad_terminate_response_blocks_probe_driver_and_preserves_service(self):
        for response in ('700','','unknown response'):
            with self.subTest(response=response),TemporaryDirectory() as tmp:
                root=Path(tmp); args=self.args(root); args.launch=True
                session=self.session(root); session.started=True
                process=mock.Mock(pid=12,returncode=None); process.poll.return_value=None
                with mock.patch.object(m,'build_parser') as parser, \
                        mock.patch.object(c,'build_plan',return_value={}), \
                        mock.patch.object(m,'Session',return_value=session), \
                        mock.patch.object(session,'start'), \
                        mock.patch.object(session,'clients',return_value={12:99}), \
                        mock.patch.object(m,'mps_processes',return_value=[]), \
                        mock.patch.object(m,'gpu_processes',return_value=set()), \
                        mock.patch.object(m.sys,'platform','linux'), \
                        mock.patch.object(m.subprocess,'Popen',return_value=process), \
                        mock.patch.object(m.subprocess,'run',return_value=SimpleNamespace(
                            returncode=0,stdout=response,stderr='')), \
                        mock.patch.object(c,'wait_workers',side_effect=TimeoutError('probe ready timeout')), \
                        mock.patch.object(c,'run_batch') as batch, \
                        mock.patch.dict(m.os.environ,{},clear=True):
                    parser.return_value.parse_args.return_value=args
                    self.assertEqual(m.main([]),1)
                    batch.assert_not_called()
                process.terminate.assert_not_called(); process.kill.assert_not_called()
                summary=c.read_json(args.output_dir/'review'/'summary.json')
                probe=c.read_json(args.output_dir/'other'/'capability'/'summary.json')
                self.assertIn('probe ready timeout',probe['error'])
                self.assertIn('terminate_client',probe['cleanup_error'])
                self.assertFalse(summary['completed']); self.assertFalse(summary['sessions'][0]['stopped'])
                self.assertTrue(session.started)
                self.assertFalse(any(action['command']=='quit' for action in session.actions))
