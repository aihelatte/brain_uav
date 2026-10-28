"""CPU/mock only: new ordinary/MPS correctness entry, provenance and PID gate."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
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

    def reuse_fixture(self, root):
        args=self.args(root); args.validation_pool.write_bytes(b'fixed pool')
        detail=root/'ordinary'/'other'/'ann_0'; detail.mkdir(parents=True)
        (root/'ordinary'/'review').mkdir()
        plan={'models':{'ann':{'config':{'batch_size':64},'checkpoint_sha256':'checkpoint hash'}}}
        contract=dict(version=d.CONTRACT_VERSION,model='ann',comparison='ordinary_vs_mps',
            config=plan['models']['ann']['config'],seed=args.seed,schedule=args.bc_schedule,
            checkpoint_sha256='checkpoint hash',full_pool_digest=args.expected_pool_digest,
            pool_file_sha256=c._hash(args.validation_pool),exception_gradient_lifecycle=d.EXCEPTION_GRADIENTS,
            first_critic_loss_only=False,error_recovery=True,trace_retained_gradient=False)
        ref=self.reference(contract)
        ref.update(initial={},snapshots=[{}]*4,contexts=[dict(failure=name) for name in d.EXCEPTION_GRADIENTS],
            error_states={i:f'error_state_{i}.pt' for i in range(4)})
        torch.save(ref,detail/'reference.pt')
        for name in ref['error_states'].values():
            torch.save(dict(before=dict(values={}),after=dict(values={})),detail/name)
        acceptance=dict(comparison='ordinary_vs_mps',semantic_passed=True,completed=True,
            recovery_required=True,recovery_completed=True,checkpoint_reload=True)
        c.write_json(root/'ordinary'/'review'/'summary.json',dict(passed=True,comparison='ordinary_vs_mps',
            role='ordinary',model='ann',exit_code=0,acceptance=acceptance))
        c.write_json(detail/'correctness.json',dict(passed=True,completed=True,role='ordinary',comparison='ordinary_vs_mps',
            recovery_completed=True,load_rebuild_verified=True,inputs=contract,snapshot_count=4,
            acceptance=acceptance,checks=[dict(passed=True)]*4,
            error_states={str(i):dict(state_checks_completed=True,protected_state_exact=dict(passed=True),
                own_protected_state=dict(passed=True)) for i in range(4)},
            manual_replay_evidence={mode:dict(cuda_graph_launch_count=1) for mode in ('actor','frozen_targets','critic_only')}))
        for i in range(4):
            c.write_json(detail/f'error_state_{i}_checks.json',dict(location=dict(index=i),protected_state_exact=dict(passed=True)))
            c.write_json(detail/f'error_state_{i}.json',{})
        c.write_json(detail/'error_recovery_summary.json',dict(passed=True,completed=True,recovery_completed=True))
        return args,plan,root/'ordinary',detail,ref

    def test_reuse_reference_reads_real_contract_reports_and_error_files(self):
        for change in (None,'failed_report','contract','source','missing_error','bad_error','incomplete_checks','config'):
            with self.subTest(change=change),TemporaryDirectory() as tmp:
                args,plan,root,detail,ref=self.reuse_fixture(Path(tmp))
                if change=='failed_report':
                    report=c.read_json(root/'review'/'summary.json'); report['passed']=False
                    c.write_json(root/'review'/'summary.json',report)
                if change=='contract': ref['contract']=dict(ref['contract'],seed=-1)
                if change=='source': ref['mps_reference']=dict(d.MPS_REFERENCE,source='mps')
                if change in ('contract','source'): torch.save(ref,detail/'reference.pt')
                if change=='missing_error': (detail/'error_state_0.pt').unlink()
                if change=='bad_error': torch.save({},detail/'error_state_0.pt')
                if change=='incomplete_checks':
                    report=c.read_json(detail/'correctness.json'); report['checks'][0]['passed']=False
                    c.write_json(detail/'correctness.json',report)
                if change=='config': plan=deepcopy(plan); plan['models']['ann']['config']['batch_size']=32
                if change:
                    with self.assertRaises((ValueError,FileNotFoundError)): m.reuse_ann_reference(root,args,plan)
                else:
                    report=m.reuse_ann_reference(root,args,plan)
                    self.assertTrue(report['reused']); self.assertFalse(report['new_comparison_passed'])
                    self.assertEqual(len(report['files_sha256']),16)

    def test_progress_is_mps_only_flushes_last_stage_and_cancels_watchdog(self):
        with TemporaryDirectory() as tmp:
            args=SimpleNamespace(gpu_compute_correctness_control_dir=Path(tmp),gpu_compute_mps_comparison=None)
            with mock.patch.object(d.faulthandler,'dump_traceback_later') as arm, \
                    mock.patch.object(d.faulthandler,'cancel_dump_traceback_later') as cancel, \
                    redirect_stderr(StringIO()) as log:
                progress=d._MPSProgress(args); progress.stage('unused',0); progress.arm(); progress.finish()
                arm.assert_not_called(); cancel.assert_not_called(); self.assertEqual(log.getvalue(),'')
                args.gpu_compute_mps_comparison='mps'; progress=d._MPSProgress(args)
                progress.stage('pid_released',0); progress.arm()
                try:
                    progress.stage('profiler_enter',0); raise RuntimeError('update error')
                except RuntimeError:
                    pass
                finally: progress.finish()
                progress.finish()
                arm.assert_called_once_with(60,repeat=True,file=log); cancel.assert_called_once()
                stage=c.read_json(Path(tmp)/'last_stage.json')
                self.assertEqual(stage['phase'],'profiler_enter'); self.assertEqual(stage['index'],0)
                self.assertGreater(stage['unix_time'],0); self.assertIn('MPS_STAGE',log.getvalue())

    def test_correctness_finalizer_keeps_watchdog_until_invalidate_and_report_finish(self):
        # Real correctness/finalizer and CPU engine; stub only CUDA setup and watchdog timer.
        from brain_uav.scripts import train_v2_td3 as single
        from brain_uav.config import RewardConfig
        import test_v2_gpu_compute as gpu_fixtures
        for failure in (None,'body','invalidate','report','recovery_report'):
            with self.subTest(failure=failure),TemporaryDirectory() as tmp:
                root=Path(tmp); (root/'control').mkdir()
                e=gpu_fixtures.TestGPUCompute().engine(); e.actor_freeze_steps=25000
                e.bc_reference_actor=e._frozen_target(e.actor,device=e.device)
                observation=e.replay.sample(2).obs
                checkpoint=root/'bc.pt'; checkpoint.write_bytes(b'fixture')
                pool_file=root/'pool.json'; pool_file.write_text('{}')
                config=SimpleNamespace(to_dict=lambda:{},policy_delay=2,actor_freeze_steps=25000,
                    spike_regularization=e.spike_regularization,batch_size=e.batch_size,max_steps=500000,
                    noise_schedule=SimpleNamespace(values=lambda *a,**kw:(0.,.01,.02)))
                args=SimpleNamespace(model='ann',seed=7,bc_checkpoint=checkpoint,validation_pool=pool_file,
                    gpu_td3_compute=False,non_blocking_batch_transfer=False,snn_time_window=4,correctness_updates=8,
                    comparison_bc_schedule='smooth-final',gpu_compute_first_critic_loss_only=True,
                    gpu_compute_error_recovery=failure=='recovery_report',gpu_compute_trace_retained_gradient=False)
                progress=d._MPSProgress(SimpleNamespace(gpu_compute_mps_comparison='ordinary',
                    gpu_compute_correctness_control_dir=root/'control'))
                active=[False]; events=[]
                original_stage=progress.stage; original_invalidate=e.invalidate_gpu_update
                original_write=p._strict_json_write
                def stage(phase,*a,**kw):
                    events.append(phase); original_stage(phase,*a,**kw)
                def arm(*a,**kw): active[0]=True; events.append('armed')
                def cancel():
                    self.assertTrue(active[0]); active[0]=False; events.append('cancelled')
                def make_progress(_): progress.arm(); return progress
                def invalidate():
                    self.assertTrue(active[0])
                    if events[-1]=='executor_invalidate_begin':
                        events.append('invalidate_call')
                        if failure=='invalidate': raise RuntimeError('invalidate failure')
                    return original_invalidate()
                def write(path,value):
                    self.assertTrue(active[0])
                    if path.name=='error_recovery_summary.json' and failure=='recovery_report':
                        self.assertEqual(events[-1],'recovery_report_write_begin')
                        raise OSError('recovery_report failure')
                    if path.name=='correctness.json':
                        self.assertEqual(events[-1],'final_report_write_begin'); events.append('report_call')
                        if failure=='report': raise OSError('report failure')
                    return original_write(path,value)
                with mock.patch.object(d,'_MPSProgress',side_effect=make_progress), \
                        mock.patch.object(progress,'stage',side_effect=stage), \
                        mock.patch.object(d.faulthandler,'dump_traceback_later',side_effect=arm), \
                        mock.patch.object(d.faulthandler,'cancel_dump_traceback_later',side_effect=cancel) as cancelled, \
                        mock.patch.object(e,'invalidate_gpu_update',side_effect=invalidate), \
                        mock.patch.object(p,'_strict_json_write',side_effect=write), \
                        mock.patch.object(p,'build_v2_stage_engine',return_value=SimpleNamespace(engine=e)), \
                        mock.patch('torch.compile',new=lambda fn,**kw:fn), \
                        mock.patch.object(single,'_configure_stage_compilation',return_value={},
                            side_effect=RuntimeError('body failure') if failure=='body' else None), \
                        mock.patch.object(p,'_compile_warmup_batches',return_value=[observation]), \
                        mock.patch.object(p,'_numeric_diagnostic_observation_batches',return_value=(observation,observation)), \
                        mock.patch.object(d.subprocess,'run',return_value=SimpleNamespace(stdout='fixture')), \
                        mock.patch.object(torch.cuda,'get_rng_state',return_value=torch.tensor([1],dtype=torch.uint8)), \
                        redirect_stderr(StringIO()):
                    def run():
                        return d.correctness(args,prepared=SimpleNamespace(reward_config=RewardConfig()),
                            config=config,pool=SimpleNamespace(content_digest='fixture'),device='cpu',output=root,
                            full_pool_digest='fixture')
                    if failure:
                        with self.assertRaisesRegex((RuntimeError,OSError),failure+' failure'): run()
                    else: self.assertTrue(run()['passed'])
                    cancelled.assert_called_once(); self.assertFalse(active[0])
                self.assertEqual(events[-1],'cancelled')
                if failure=='recovery_report':
                    self.assertNotIn('executor_invalidate_begin',events)
                    self.assertNotIn('recovery_report_write_complete',events)
                    continue
                self.assertLess(events.index('executor_invalidate_begin'),events.index('invalidate_call'))
                if failure=='invalidate':
                    self.assertNotIn('executor_invalidate_complete',events)
                    self.assertNotIn('final_report_write_begin',events)
                else:
                    self.assertLess(events.index('executor_invalidate_complete'),events.index('report_call'))
                    if failure=='report': self.assertNotIn('final_report_write_complete',events)
                    else: self.assertEqual(events[-2],'final_report_write_complete')

    def test_reuse_driver_skips_only_ann_ordinary_and_keeps_new_mps_gate(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args,plan,source,_,_=self.reuse_fixture(root)
            args.launch=True; args.reuse_ann_ordinary_reference=source
            plan['execution_requested']={'cpu_deduplication':True}
            runs=[]
            session=mock.Mock(env={}); session.finish.return_value=dict(stopped=True)
            def run(args,model,role,where,**kw):
                runs.append((model,role,kw))
                return dict(passed=True,comparison='ordinary_vs_mps')
            with mock.patch.object(m,'build_parser') as parser, \
                    mock.patch.object(c,'build_plan',return_value=plan), \
                    mock.patch.object(m,'Session',return_value=session), \
                    mock.patch.object(m,'capability',return_value=dict(passed=True)), \
                    mock.patch.object(m,'mps_processes',return_value=[]), \
                    mock.patch.object(m,'gpu_processes',return_value=set()), \
                    mock.patch.object(m,'command',return_value='GPU'), \
                    mock.patch.object(m.sys,'platform','linux'), \
                    mock.patch.object(m,'run_correctness_worker',side_effect=run), \
                    mock.patch.object(c,'run_batch') as batch, \
                    mock.patch.dict(os.environ,{},clear=True):
                parser.return_value.parse_args.return_value=args
                self.assertEqual(m.main([]),0); batch.assert_not_called()
            self.assertEqual([(model,role) for model,role,_ in runs],[('ann','mps'),('snn','ordinary'),('snn','mps')])
            self.assertIs(runs[0][2]['session'],session)
            self.assertEqual(runs[0][2]['reference'],source/'other'/'ann_0'/'reference.pt')
            report=c.read_json(args.output_dir/'review'/'ann_ordinary_reused.json')
            self.assertFalse(report['new_comparison_passed']); self.assertTrue(report['original_passed'])

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

    def test_public_main_rejects_partial_mps_arguments_before_any_dispatch(self):
        for argv in (['--gpu-compute-mps-comparison','ordinary'],
                     ['--gpu-compute-correctness-control-dir','control']):
            with self.subTest(argv=argv), \
                    mock.patch.object(p,'prepare_v2_stage_initialization') as init, \
                    mock.patch.object(p,'resolve_training_device') as device, \
                    mock.patch.object(d,'run') as run, \
                    mock.patch.object(p,'run_v2_td3_timing_diagnostic') as timing:
                with self.assertRaises(ValueError):
                    p.main(['--model','ann','--output-dir','unused']+argv)
                init.assert_not_called(); device.assert_not_called()
                run.assert_not_called(); timing.assert_not_called()

    def test_real_plan_matches_correctness_and_timing_worker_cpu_deduplication(self):
        import json
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=self.args(root)
            for model in ('ann','snn'): (root/f'{model}.pt').write_bytes(b'checkpoint fixture')
            pool=SimpleNamespace(content_digest='digest',scenario_count=3,scenarios=[
                dict(scenario_id=str(i),payload=dict(zones=[])) for i in range(3)])
            for correctness_only in (True,False):
                args.correctness_only=correctness_only
                with self.subTest(correctness_only=correctness_only), \
                        mock.patch.object(m,'build_parser') as parser, \
                        mock.patch('brain_uav.trainers.v2_validation.load_v2_validation_pool',return_value=pool), \
                        mock.patch.object(m,'Session') as session, \
                        mock.patch.object(m,'run_correctness_worker') as worker_run, \
                        mock.patch.object(c,'run_batch') as batch, \
                        redirect_stdout(StringIO()) as output:
                    parser.return_value.parse_args.return_value=args
                    self.assertEqual(m.main([]),0)
                    plan=json.loads(output.getvalue())['plan']
                    session.assert_not_called(); worker_run.assert_not_called(); batch.assert_not_called()
                for model in ('ann','snn'):
                    for role in ('ordinary','mps'):
                        command=(m.correctness_command(args,model,role,root/role,root/'reference.pt')
                            if correctness_only else c.worker_command(args,model,model+'_0',root/role))
                        worker=p.build_parser().parse_args(command[command.index('brain_uav.scripts.profile_v2_td3')+1:])
                        self.assertEqual(plan['execution_requested']['cpu_deduplication'],worker.cpu_deduplication)
                        self.assertEqual(worker.cpu_deduplication,not correctness_only)

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
                if failure=='timeout':
                    c.write_json(worker_root/'other'/'control'/'last_stage.json',dict(phase='profiler_enter',index=0))
                    raise TimeoutError('controlled update timeout')
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
                self.assertEqual(result['last_stage'],dict(phase='profiler_enter',index=0))
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
                        mock.patch.object(c,'build_plan',return_value={'execution_requested':{'cpu_deduplication':True}}), \
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
        with mock.patch.object(c,'build_plan',return_value={'execution_requested':{'cpu_deduplication':True}}),mock.patch.object(m,'Session') as session, \
                mock.patch.dict(os.environ,{},clear=True),redirect_stdout(StringIO()) as output:
            self.assertEqual(m.main(base+['--concurrency','8','--measurement-seconds','30','--repeats','2',
                                         '--non-blocking-batch-transfer']),0)
            self.assertIn('ordinary_vs_mps',output.getvalue()); session.assert_not_called()
        with redirect_stderr(StringIO()),self.assertRaises(SystemExit): m.build_parser().parse_args(base+['--capability-only'])
