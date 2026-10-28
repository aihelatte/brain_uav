"""CPU/mock only: new ordinary/MPS correctness entry, provenance and PID gate."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from brain_uav.scripts import profile_v2_td3 as p
from brain_uav.scripts import v2_gpu_compute_diagnostic as d
from brain_uav.scripts import v2_mps_diagnostic as m
from brain_uav.scripts import v2_concurrent_diagnostic as c
from test_v2_concurrent_diagnostic import arguments
from test_v2_bc import make_scenario_config
import test_v2_mps_diagnostic as fixtures


class TestMPSCorrectness(unittest.TestCase):
    def args(self, root):
        args = arguments(root)
        args.concurrency=[8]; args.repeats=2; args.measurement_seconds=30.
        args.non_blocking_batch_transfer=True
        args.capability_only=False; args.correctness_only=True
        return args

    def reference(self, contract):
        return dict(contract=contract,mps_reference=dict(d.MPS_REFERENCE),
            transfer_reference=dict(gpu_td3_compute=True,non_blocking_batch_transfer=True),
            cuda_access_evidence=dict(effective='ordinary',worker_pid=12,
                compute_pids=[12],verified_before_update=True,comparison='ordinary_vs_mps'))

    def test_real_public_worker_entry_write_read_and_provenance_rejection(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=self.args(root); contract={'comparison':'ordinary_vs_mps'}
            reference=root/'ref.pt'
            prepared=SimpleNamespace(scenario_config=make_scenario_config(),uav_collision_radius=0.)
            pool=SimpleNamespace(content_digest='digest',scenario_count=3,master_seed=20260904,
                stage_seed=1,scenario_config=prepared.scenario_config,uav_collision_radius=0.,scenarios=[{}]*3)
            reached=[]
            def correctness(parsed,**kw):
                reached.append(parsed)
                self.assertTrue(parsed.gpu_td3_compute and parsed.non_blocking_batch_transfer)
                self.assertFalse(parsed.gpu_compute_non_blocking_coefficients)
                self.assertFalse(parsed.gpu_compute_coefficient_comparison)
                if parsed.gpu_compute_write_reference: torch.save(self.reference(contract),reference)
                else: d._load_compute_reference(parsed,contract)
                c.write_json(kw['output']/'error_recovery_summary.json',dict(passed=True))
                return dict(passed=True)
            with mock.patch.object(p,'prepare_v2_stage_initialization',return_value=prepared), \
                    mock.patch.object(p,'resolve_training_device',return_value='cuda'), \
                    mock.patch.object(p,'load_v2_validation_pool',return_value=pool), \
                    mock.patch.object(p,'V2ValidationPool',side_effect=lambda **kw:SimpleNamespace(**kw)), \
                    mock.patch.object(d,'correctness',side_effect=correctness):
                for role in ('ordinary','mps'):
                    argv=m.correctness_command(args,'snn',role,root/role,reference)
                    self.assertTrue(p.main(argv[argv.index('brain_uav.scripts.profile_v2_td3')+1:])['passed'])
                self.assertEqual([a.gpu_compute_mps_comparison for a in reached],['ordinary','mps'])
                for change in ('old_blocking','coefficient','mps_source','no_pid','wrong_pid','no_initial_contract'):
                    value=self.reference(contract)
                    if change=='old_blocking': value['transfer_reference']['non_blocking_batch_transfer']=False
                    if change=='coefficient': value['coefficient_reference']={'gpu_td3_compute':True}
                    if change=='mps_source': value['mps_reference']['source']='mps'
                    if change=='no_pid': value.pop('cuda_access_evidence')
                    if change=='wrong_pid': value['cuda_access_evidence']['compute_pids']=[13]
                    if change=='no_initial_contract': value.pop('mps_reference')
                    torch.save(value,reference)
                    argv=m.correctness_command(args,'snn','mps',root/change,reference)
                    with self.subTest(change=change),self.assertRaises(SystemExit):
                        p.main(argv[argv.index('brain_uav.scripts.profile_v2_td3')+1:])
            torch.save(self.reference(contract),reference)
            parsed=reached[-1]
            parsed.gpu_compute_mps_comparison=None
            with self.assertRaisesRegex(ValueError,'cannot be used'): d._load_compute_reference(parsed,contract)

    def test_invalid_worker_combinations_fail_before_initialization(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=self.args(root)
            argv=m.correctness_command(args,'ann','ordinary',root/'a')
            argv=argv[argv.index('brain_uav.scripts.profile_v2_td3')+1:]
            for changes in (dict(non_blocking_batch_transfer=False),dict(gpu_td3_compute=False),
                    dict(gpu_compute_coefficient_comparison=True),dict(gpu_compute_non_blocking_coefficients=True),
                    dict(gpu_compute_lightweight_timing=True),dict(gpu_compute_write_reference=False),
                    dict(gpu_compute_correctness_control_dir=None),dict(gpu_compute_first_critic_loss_only=True)):
                parsed=p.build_parser().parse_args(argv)
                for key,value in changes.items(): setattr(parsed,key,value)
                with mock.patch.object(p,'prepare_v2_stage_initialization') as init:
                    with self.subTest(changes=changes),self.assertRaises(ValueError): d.run(parsed)
                    init.assert_not_called()

    def test_worker_gate_synchronizes_then_waits_for_matching_release(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=self.args(root)
            command=m.correctness_command(args,'ann','ordinary',root)
            parsed=p.build_parser().parse_args(command[command.index('brain_uav.scripts.profile_v2_td3')+1:])
            control=parsed.gpu_compute_correctness_control_dir; control.mkdir(parents=True)
            access=dict(worker_pid=os.getpid(),effective='ordinary',verified_before_update=True,comparison='ordinary_vs_mps')
            def release(_):
                self.assertTrue(c.read_json(control/'ready.json')['synchronized'])
                c.write_json(control/'release.json',access)
            with mock.patch.object(torch.cuda,'synchronize') as sync,mock.patch('time.sleep',side_effect=release):
                self.assertEqual(d._mps_correctness_ready(parsed,'cuda'),access)
                sync.assert_called_once_with('cuda')
            access['worker_pid']=-1; c.write_json(control/'release.json',access)
            with mock.patch.object(torch.cuda,'synchronize'),self.assertRaisesRegex(ValueError,'PID release'):
                d._mps_correctness_ready(parsed,'cuda')
            c.write_json(control/'abort.json',{})
            with mock.patch.object(torch.cuda,'synchronize'),self.assertRaisesRegex(RuntimeError,'aborted'):
                d._mps_correctness_ready(parsed,'cuda')
            (control/'abort.json').unlink()
            with mock.patch.object(torch.cuda,'synchronize'),mock.patch('time.monotonic',side_effect=[0,parsed.concurrent_timeout+1]), \
                    self.assertRaisesRegex(TimeoutError,'PID release timeout'):
                d._mps_correctness_ready(parsed,'cuda')

    def run_worker(self, root, *, failure=None, session=None):
        args=self.args(root); worker_root=root/'worker'; role='mps' if session else 'ordinary'
        process=mock.Mock(pid=12); alive=[True]
        process.poll.side_effect=lambda: None if alive[0] else 0
        process.wait.side_effect=lambda **kw: alive.__setitem__(0,False) or 0
        calls=[]
        def wait(processes,predicate,**kw):
            calls.append(1)
            if len(calls)==1:
                c.write_json(worker_root/'other'/'control'/'ready.json',
                    dict(pid=12,role=role,synchronized=True,comparison='ordinary_vs_mps'))
            else:
                if failure=='timeout': raise TimeoutError('controlled update timeout')
                alive[0]=False
                detail=worker_root/'other'/'ann_0'; detail.mkdir()
                c.write_json(detail/'correctness.json',dict(comparison='ordinary_vs_mps',
                    passed=failure!='numeric',completed=True,load_rebuild_verified=True,recovery_completed=True,
                    manual_replay_evidence={key:{} for key in ('actor','frozen_targets','critic_only')}))
            self.assertTrue(predicate('correctness'))
        with mock.patch.object(m.subprocess,'Popen',return_value=process), \
                mock.patch.object(c,'wait_workers',side_effect=wait), \
                mock.patch.object(m,'ordinary_access',side_effect=RuntimeError('PID mismatch') if failure=='pid' else None,
                    return_value=dict(effective='ordinary')):
            result=m.run_correctness_worker(args,'ann',role,worker_root,reference=root/'ref.pt',session=session)
        return result,process,worker_root

    def test_parent_pid_and_numeric_failure_are_not_success(self):
        for failure in (None,'pid','numeric'):
            with self.subTest(failure=failure),TemporaryDirectory() as tmp:
                result,process,root=self.run_worker(Path(tmp),failure=failure)
                self.assertEqual(result['passed'],failure is None)
                self.assertEqual((root/'other'/'control'/'release.json').exists(),failure!='pid')
                if failure: self.assertIn('error',result)

    def test_correctness_cleanup_failure_blocks_signals_and_retains_service(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); session=fixtures.TestMPS().session(root); session.started=True
            with mock.patch.object(session,'verify',return_value=dict(effective='mps')), \
                    mock.patch.object(session,'clients',return_value={12:99}), \
                    mock.patch.object(session,'control',return_value='700') as control:
                result,process,_=self.run_worker(root,failure='timeout',session=session)
                process.terminate.assert_not_called(); process.kill.assert_not_called()
                self.assertFalse(result['passed']); self.assertIn('700',result['cleanup_error'])
                self.assertFalse(session.finish()['stopped'])
                control.assert_called_once_with('terminate_client 99 12')

    def test_new_mode_plan_and_serial_driver_no_timing(self):
        for launch,failure in ((False,None),(True,None),(True,'numeric'),(True,'cleanup')):
            with self.subTest(launch=launch,failure=failure),TemporaryDirectory() as tmp:
                root=Path(tmp); args=self.args(root); args.launch=launch
                session=mock.Mock(env={})
                session.finish.side_effect=[dict(stopped=True),dict(stopped=failure!='cleanup'),dict(stopped=True)]
                runs=[]
                def run(args,model,role,where,**kw):
                    runs.append((model,role))
                    return dict(passed=failure!='numeric',comparison='ordinary_vs_mps')
                with mock.patch.object(m,'build_parser') as parser, \
                        mock.patch.object(c,'build_plan',return_value={}), \
                        mock.patch.object(m,'Session',return_value=session), \
                        mock.patch.object(m,'capability',return_value=dict(passed=True)), \
                        mock.patch.object(m,'mps_processes',return_value=[]), \
                        mock.patch.object(m,'gpu_processes',return_value=set()), \
                        mock.patch.object(m,'command',return_value='GPU'), \
                        mock.patch.object(m.sys,'platform','linux'), \
                        mock.patch.object(m,'run_correctness_worker',side_effect=run), \
                        mock.patch.object(c,'run_batch') as timing, \
                        mock.patch.dict(os.environ,{},clear=True),redirect_stdout(StringIO()) as stdout,redirect_stderr(StringIO()):
                    parser.return_value.parse_args.return_value=args
                    self.assertEqual(m.main([]),1 if failure else 0)
                    timing.assert_not_called()
                    if not launch:
                        self.assertIn('correctness_only',stdout.getvalue()); session.start.assert_not_called()
                    elif failure=='numeric': self.assertEqual(runs,[('ann','ordinary')])
                    elif failure=='cleanup': self.assertEqual(runs,[('ann','ordinary'),('ann','mps')])
                    else: self.assertEqual(runs,[('ann','ordinary'),('ann','mps'),('snn','ordinary'),('snn','mps')])

    def test_real_parser_exposes_new_mode_and_rejects_capability_combination(self):
        base=['--output-dir','unused','--ann-bc-checkpoint','ann.pt','--snn-bc-checkpoint','snn.pt',
              '--validation-pool','pool.json','--expected-pool-digest','digest',
              '--ann-actor-lr','0.000155','--snn-actor-lr','0.000825',
              '--bc-schedule','smooth-mid','--correctness-only']
        parsed=m.build_parser().parse_args(base)
        self.assertTrue(parsed.correctness_only); self.assertFalse(parsed.launch)
        with mock.patch.object(c,'build_plan',return_value={}),mock.patch.object(m,'Session') as session, \
                mock.patch.dict(os.environ,{},clear=True),redirect_stdout(StringIO()) as output:
            self.assertEqual(m.main(base+['--concurrency','8','--measurement-seconds','30','--repeats','2',
                                         '--non-blocking-batch-transfer']),0)
            self.assertIn('ordinary_vs_mps',output.getvalue()); session.assert_not_called()
        with redirect_stderr(StringIO()),self.assertRaises(SystemExit): m.build_parser().parse_args(base+['--capability-only'])
