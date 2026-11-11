"""Review regressions. Tiny CPU data and mocked child processes only."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
import json
import sys
import subprocess
import unittest
from contextlib import ExitStack,contextmanager
import numpy as np
import torch

from brain_uav.scripts import run_v2_formal_easy_baseline as run,v2_formal_easy_cuda_gate as gate
from brain_uav.trainers import v2_formal_training as formal
from brain_uav.trainers.v2_formal_acc import FormalACC,load_state
from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash
import test_v2_formal_training as f


class Review(unittest.TestCase):
    def args(self,root,gate_only=False):
        argv=['launch','--root',str(root),'--ann-bc','ann','--snn-bc','snn','--validation-pool','pool','--launch']
        return run.parser().parse_args(argv+(['--gate-only'] if gate_only else []))

    @contextmanager
    def terminal_batch(self,root):
        from brain_uav.trainers.v2_online_research import atomic_json
        a=self.args(root);raws={}
        pool=SimpleNamespace(content_digest=run.provenance.DIGEST,scenarios=[{'scenario_id':str(i)} for i in range(100)])
        paths={'pool':str(root/'pool')}
        for model in ('ann','snn'):
            bc=root/(model+'_best.pt');bc.write_bytes(model.encode());paths[model+'_bc']=str(bc)
        evidence={'paths':paths,'hashes':{k:run.file_hash(p) for k,p in paths.items() if k!='pool'}}
        atomic_json(root/'manifest.json',dict(inputs=evidence,plan=run.plan(a),batch_id='test'))
        for job in run.jobs():
            spec,cfg=run.resolved(job);directory=root/job['id'];directory.mkdir()
            cp=directory/'model_failed.pt';cp.write_bytes(job['id'].encode())
            stop='max_steps_without_validation';result=dict(stage_steps=500000,status='failed',stop_reason=stop)
            raws[str(cp)]=dict(formal_config=cfg.to_dict(),training_result=result,passed_validation=False,
                reward_config=spec['reward_config'],scenario_config=spec['scenario_config'],
                validation_pool={'content_digest':pool.content_digest},uav_collision_radius=0.,
                initialization_source={'kind':'v2_bc_best','checkpoint_kind':'best','path':paths[job['model']+'_bc']},
                bc_schedule=formal.v2_bc_schedule_metadata('easy',schedule='smooth-mid-strong'))
            evaluation=dict(goal_count=0,scenarios=[dict(scenario_id=str(i),episode_return=1.,episode_length=2,outcome='timeout') for i in range(100)])
            atomic_json(directory/'terminal_evaluation.json',dict(checkpoint=str(cp),checkpoint_sha256=run.file_hash(cp),
                pool_sha256=run.POOL_SHA,pool_digest=pool.content_digest,evaluation=evaluation,
                formal_passed=False,actual_steps=500000,reward_config=spec['reward_config']))
            atomic_json(directory/'worker_summary.json',dict(execution_completed=True,formal_passed=False,job=job,pid=100,batch_id='test',
                result=dict(checkpoint=str(cp),steps=500000,stop_reason=stop,passed=False)))
            atomic_json(directory/'process.json',dict(exit_code=0,pid=100))
            atomic_json(directory/'ready.json',dict(pid=100,job=job))
        with patch.object(run,'check_inputs'),patch('brain_uav.trainers.v2_validation.load_v2_validation_pool',return_value=pool), \
             patch.object(formal,'load_v2_formal_checkpoint',side_effect=lambda p,**kw:deepcopy(raws[str(p)])):
            yield a,raws

    def test_collect_validates_all_four_before_publishing_and_second_collect_preserves_success(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.terminal_batch(root) as (a,raws):
                result=run.collect(a);self.assertTrue(result['execution_completed'])
                before=(root/'selection_summary.json').read_bytes()
                try:again=run.collect(a)
                except FileExistsError:again=None
                self.assertEqual((root/'selection_summary.json').read_bytes(),before)
                if again:self.assertEqual(again,result)
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.terminal_batch(root) as (a,raws):
                (root/'snn_seed23/terminal_evaluation.json').unlink()
                with self.assertRaises(FileNotFoundError):run.collect(a)
                self.assertFalse((root/'selected').exists())

    def test_terminal_identity_errors_and_atomic_copy_failure(self):
        for mutation in ('external','job','stop','reward','copy'):
            with self.subTest(mutation=mutation),TemporaryDirectory() as tmp:
                root=Path(tmp)
                with self.terminal_batch(root) as (a,raws):
                    where=root/'snn_seed23'
                    if mutation=='external':
                        path=where/'terminal_evaluation.json';e=json.loads(path.read_text())
                        external=root/'outside.pt';external.write_bytes((where/'model_failed.pt').read_bytes())
                        e['checkpoint']=str(external);run.atomic_json(path,e)
                    elif mutation=='job':
                        path=where/'worker_summary.json';s=json.loads(path.read_text());s['job']=run.jobs()[0];run.atomic_json(path,s)
                    elif mutation=='stop':raws[str(where/'model_failed.pt')]['training_result']['stop_reason']='research_hard_limit'
                    elif mutation=='reward':raws[str(where/'model_failed.pt')]['reward_config']={}
                    if mutation=='copy':
                        with patch.object(run.shutil,'copyfile',side_effect=OSError('copy failure')):
                            with self.assertRaisesRegex(OSError,'copy failure'):run.collect(a)
                    else:
                        with self.assertRaises(ValueError):run.collect(a)
                    self.assertFalse((root/'selected').exists())
                    self.assertFalse(json.loads((root/'selection_summary.json').read_text())['execution_completed'])

    def test_real_isolated_scoring_does_not_pollute_training_state(self):
        from test_v2_td3 import _observation
        engine=f._engine(f.ScenarioConfig())
        cfg=formal.V2FormalTrainingConfig(stage='easy',acc_eta=.01,replay_capacity=16,batch_size=2)
        hook=FormalACC(engine,cfg)
        rows=[dict(observation=_observation(0,scales=engine.actor.scales),action=np.zeros(2,dtype=np.float32))]
        q=gate.checked_bank_score(engine,hook.scorer,rows)
        self.assertEqual(q.shape,(1,2))
        for live,copy in zip((engine.critic1,engine.critic2),hook.scorer.copies):
            self.assertTrue(all(torch.equal(a,b) and a.data_ptr()!=b.data_ptr() for a,b in zip(live.parameters(),copy.parameters())))
        class PollutingScorer:
            copies=hook.scorer.copies
            def __call__(self,engine,rows):
                torch.rand(1)
                return hook.scorer(engine,rows)
        with self.assertRaisesRegex(AssertionError,'RNG'):gate.checked_bank_score(engine,PollutingScorer(),rows)

    def test_missing_gate_refuses_before_session_or_children(self):
        from brain_uav.scripts import v2_mps_diagnostic as mps
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp)/'batch')
            with patch.object(run.provenance,'PYTHON',sys.executable),patch.object(run,'inputs',return_value={}), \
                 patch.object(mps,'Session',side_effect=RuntimeError('must not create')) as session,patch.object(run.subprocess,'Popen') as spawn:
                with self.assertRaises(ValueError):run.launch(a)
                session.assert_not_called();spawn.assert_not_called()

    def test_failed_restore_keeps_min_engine_unmodified(self):
        scenario=f.ScenarioConfig();source=f._engine(scenario)
        cfg=formal.V2FormalTrainingConfig(stage='easy',acc_eta=.01,replay_capacity=16,batch_size=2)
        FormalACC(source,cfg)
        payload=formal.build_v2_periodic_snapshot(source,cfg,stage_steps=1,scenario=scenario,rewards=f.RewardConfig(),
            uav_collision_radius=0.,seed_manifest={'base_seed':7},initialization_source={'kind':'test'})
        target=f._engine(scenario);target.gamma=.9
        before=state_hash(target.checkpoint_state_dict())
        with self.assertRaises(ValueError):load_state(target,cfg,payload)
        self.assertEqual(state_hash(target.checkpoint_state_dict()),before)
        self.assertFalse(hasattr(target,'formal_acc'))
        self.assertIsNone(getattr(target,'research_target_config',None))
        target.load_checkpoint_state_dict(deepcopy(target.checkpoint_state_dict()))
        target=f._engine(scenario);hook=FormalACC(target,cfg)
        before=state_hash(target.checkpoint_state_dict());controller=state_hash(hook.controller.state())
        bad=deepcopy(payload);bad['engine_checkpoint']['critic_optimizer_state_dict']['param_groups'][0]['params']=[]
        with self.assertRaises(ValueError):load_state(target,cfg,bad)
        self.assertIs(target.formal_acc,hook)
        self.assertEqual(state_hash(target.checkpoint_state_dict()),before)
        self.assertEqual(state_hash(hook.controller.state()),controller)
        from test_v2_td3 import _observation
        transition=SimpleNamespace(observation=_observation(0,scales=target.actor.scales),action=np.zeros(2,dtype=np.float32),
            reward=1.,done=True,collected_stage_step=25001)
        hook.episode(SimpleNamespace(engine=target,result=SimpleNamespace(stage_steps=25001)),[transition])
        self.assertEqual(len(hook.controller.controls),1)

    def gate_fixture(self,root,evidence):
        from brain_uav.trainers.v2_online_research import atomic_json
        manifest=dict(plan={'gate_only':True},inputs=evidence,batch_id='gate-batch')
        atomic_json(root/'manifest.json',manifest)
        for i,job in enumerate(run.jobs()):
            pid=100+i;where=root/job['id']
            atomic_json(where/'ready.json',dict(pid=pid,job=job))
            atomic_json(where/'process.json',dict(pid=pid,exit_code=0))
            atomic_json(where/'worker_summary.json',dict(pid=pid,job=job,batch_id='gate-batch',
                execution_completed=True,strict_passed=True))
            identity=dict(job=job,pid=pid,batch_id='gate-batch',resolved=run.resolved(job)[0],
                source_sha256=evidence['source']['sha256'],input_hashes=evidence['hashes'])
            atomic_json(where/'cuda_gate.json',dict(identity=identity,strict_passed=True,execution_completed=True))
        bindings=run.gate_bindings(root,manifest)
        atomic_json(root/'batch_summary.json',dict(execution_completed=True,gate_bindings=bindings))

    def test_gate_report_identity_hash_inputs_exit_and_swaps_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.args(root/'training');a.gate_report=root/'gate'
            evidence=dict(source={'sha256':{'file':'hash'}},hashes={'BC':'hash'})
            self.gate_fixture(a.gate_report,evidence)
            proof=run.validate_gate(a,evidence);self.assertEqual(len(proof['bindings']),4)
            for change in ('swap','pid','exit','hash','source'):
                self.gate_fixture(a.gate_report,evidence)
                path=a.gate_report/'ann_seed7/cuda_gate.json'
                value=json.loads(path.read_text())
                if change=='swap':value=json.loads((a.gate_report/'ann_seed23/cuda_gate.json').read_text())
                elif change=='pid':value['identity']['pid']=999
                elif change=='exit':
                    run.atomic_json(a.gate_report/'ann_seed7/process.json',{'pid':100,'exit_code':1})
                elif change=='hash':value['extra']='changed'
                else:value['identity']['source_sha256']={'file':'wrong'}
                run.atomic_json(path,value)
                with self.subTest(change=change),self.assertRaises(ValueError):run.validate_gate(a,evidence)

    def test_gate_timeout_after_release_cleans_owned_workers(self):
        from brain_uav.scripts import v2_mps_diagnostic as mps
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp)/'batch',True);a.gate_timeout=1.
            session=Mock(env={});session.finish.return_value={'stopped':True}
            processes=[Mock(pid=i+10) for i in range(4)]
            for p in processes:p.poll.return_value=None
            with patch.object(run.provenance,'PYTHON',sys.executable),patch.object(run,'inputs',return_value={}), \
                 patch.object(mps,'Session',return_value=session),patch.object(mps,'capability'), \
                 patch.object(run.subprocess,'Popen',side_effect=processes),patch.object(run,'wait_ready'), \
                 patch.object(run,'release'),patch.object(run.time,'monotonic',side_effect=(0.,2.)):
                with self.assertRaisesRegex(TimeoutError,'after release'):run.launch(a)
            self.assertEqual(session.before_terminate.call_count,4);session.finish.assert_called_once()

    def test_probe_process_group_isolation(self):
        from brain_uav.scripts import v2_mps_diagnostic as mps,v2_concurrent_diagnostic as concurrent
        with TemporaryDirectory() as tmp:
            root=Path(tmp);session=Mock(env={});process=Mock(pid=12)
            session.verify.return_value={'client_server_pid':{'12':99}}
            process.poll.side_effect=[None,0,0];process.wait.return_value=0
            def ready(*args,**kw):run.atomic_json(root/'probe/ready.json',dict(pid=12,graph_replays=8,synchronized=True))
            with patch.object(mps.subprocess,'Popen',return_value=process) as spawn,patch.object(concurrent,'wait_workers',side_effect=ready):
                mps.capability(root/'probe',session)
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])

    def test_mask_evidence_and_valid_single_observation_and_beta_types(self):
        from test_v2_td3 import _observation
        from brain_uav.observations import collate_v2_observations
        scales=f._engine(f.ScenarioConfig()).actor.scales
        batch=collate_v2_observations([_observation(0,scales=scales),_observation(1,scales=scales)])
        from brain_uav.observations import V2Observation
        from brain_uav.scripts.diagnose_v2_mc_return_cuda import compare_snapshots
        with self.assertRaises(ValueError):
            V2Observation(batch.ego_features[0].numpy(),batch.goal_features[0].numpy(),
                batch.zone_features[0].numpy(),batch.presence_mask[0].numpy())
        with self.assertRaises(AttributeError):compare_snapshots({'beta':.5},{'beta':.5})
        observation=gate.bank_observation(batch)
        self.assertEqual(len(observation.zone_features),int(batch.presence_mask[0].sum()))
        evidence=gate.batch_evidence(SimpleNamespace(obs=batch,next_obs=batch))
        self.assertTrue(torch.equal(evidence['replay.mask'],batch.presence_mask))
        altered=deepcopy(evidence);altered['replay.mask'].logical_not_()
        self.assertFalse(compare_snapshots(evidence,altered)['passed'])
        def state(beta):return gate.controller_evidence(dict(beta=beta,controls=[{'C':1.}]))
        self.assertTrue(compare_snapshots(state(.5),state(.5))['passed'])
        self.assertFalse(compare_snapshots(state(.5),state(.6))['passed'])

    def test_cached_beta_update_must_reuse_executor(self):
        old=SimpleNamespace(captures=3,replays=5)
        proof=gate.executor_evidence(old);old.replays+=1
        gate.require_cached_replay(old,proof)
        old.captures+=1
        with self.assertRaises(AssertionError):gate.require_cached_replay(old,proof)
        with self.assertRaises(AssertionError):gate.require_cached_replay(SimpleNamespace(captures=3,replays=7),proof)
        from brain_uav.trainers.v2_td3 import research_target_pair
        batch=SimpleNamespace(reward=torch.zeros(2,1),done=torch.tensor([[0.],[1.]]))
        q1=torch.tensor([[2.],[4.]]);q2=torch.tensor([[8.],[9.]])
        target=batch.reward+(1-batch.done)*.99*research_target_pair(q1,q2,'acc',.6)
        proof=gate.beta_target_evidence(batch,.99,q1,q2,target,.6,.5)
        self.assertTrue(all(proof.values()))
        stale=batch.reward+(1-batch.done)*.99*research_target_pair(q1,q2,'acc',.5)
        self.assertFalse(gate.beta_target_evidence(batch,.99,q1,q2,stale,.6,.5)['runtime_target_matches'])

    def test_full_gate_sequence_uses_real_keys_and_published_banks(self):
        from brain_uav.trainers.v2_gpu_compute import GPUCompute
        from brain_uav.observations import collate_v2_observations
        from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch
        from test_v2_td3 import _observation
        engine=f._engine(f.ScenarioConfig());engine.batch_size=2
        obs=collate_v2_observations([_observation(1,scales=engine.actor.scales)]*2)
        batch=V2ReplayBatch(obs,torch.zeros(2,2),torch.zeros(2,1),obs,torch.zeros(2,1),torch.zeros(2,1),torch.zeros(2,1),torch.ones(2,1))
        executor=object.__new__(GPUCompute)
        executor.e=engine;executor.zone_limit=6;executor.bound_device=torch.device('cpu');executor.published=[None,None]
        seen=set();keys={}
        self.assertEqual(gate.GATE_STEPS,(1,3,4,25001,25002,26002,26003,26004))
        for index,step in enumerate(gate.GATE_STEPS):
            mode='actor' if step%2==0 and step>25000 else 'frozen_targets' if step%2==0 else 'critic_only'
            key=executor.key(batch,mode);keys[step]=key
            if step in gate.BETA_REFRESH_STEPS:self.assertIn(key,seen,(step,executor.published,key))
            seen.add(key)
            # Successful compute publishes the critic bank, and actor bank only on actor updates.
            executor.published[1]=key[4]
            if mode=='actor':executor.published[0]=key[3]
            if index==6:
                # Checkpoint load invalidates the old executor; rebuilt banks start unpublished.
                seen.clear();executor.published=[None,None]
        self.assertEqual(keys[25002][-2:],(0,0))
        self.assertEqual(keys[26002][-2:],(1,1))
        self.assertNotEqual(keys[25002],keys[26002])
        self.assertEqual(keys[26003],keys[1])
        self.assertEqual(keys[26004][-2:],(0,0))
        self.assertEqual({k[2] for k in keys.values()},{'critic_only','frozen_targets','actor'})

    def test_collect_rejects_batch_and_pid_mismatch_preserving_selection(self):
        for field in ('batch_id','worker_pid','process_pid'):
            for existing in (False,True):
                with self.subTest(field=field,existing=existing),TemporaryDirectory() as tmp:
                    root=Path(tmp)
                    with self.terminal_batch(root) as (a,_):
                        before=None
                        if existing:
                            run.collect(a);before=(root/'selection_summary.json').read_bytes()
                            selected={str(p):p.read_bytes() for p in (root/'selected').rglob('*') if p.is_file()}
                        path=root/'snn_seed23'/('process.json' if field=='process_pid' else 'worker_summary.json')
                        value=json.loads(path.read_text());value['batch_id' if field=='batch_id' else 'pid']='wrong' if field=='batch_id' else 999
                        run.atomic_json(path,value)
                        with self.assertRaises(ValueError):run.collect(a)
                        if existing:
                            self.assertEqual((root/'selection_summary.json').read_bytes(),before)
                            self.assertEqual({str(p):p.read_bytes() for p in (root/'selected').rglob('*') if p.is_file()},selected)
                        else:self.assertFalse((root/'selected').exists())

    def test_real_sigterm_enters_cleanup_and_restores_handler(self):
        code='''
import os,signal,sys
from pathlib import Path
from unittest.mock import Mock,patch
from brain_uav.scripts import run_v2_formal_easy_baseline as r,v2_mps_diagnostic as m
root=Path(sys.argv[1]);a=r.parser().parse_args(['launch','--root',str(root),'--ann-bc','ann','--snn-bc','snn','--validation-pool','pool','--gate-only','--launch'])
old=signal.getsignal(signal.SIGTERM)
session=Mock(env={});session.before_terminate.side_effect=lambda *a:(root/'stopped').write_text('yes')
session.finish.side_effect=lambda:((root/'finished').write_text('yes') and {'stopped':True})
workers=[Mock(pid=i+10) for i in range(4)]
for p in workers:p.poll.return_value=None
def terminate(*a):
 if os.name=='nt':signal.raise_signal(signal.SIGTERM)
 else:os.kill(os.getpid(),signal.SIGTERM)
try:
 with patch.object(r.provenance,'PYTHON',sys.executable),patch.object(r,'inputs',return_value={}),patch.object(m,'Session',return_value=session),patch.object(m,'capability'),patch.object(r.subprocess,'Popen',side_effect=workers),patch.object(r,'wait_ready',side_effect=terminate):r.launch(a)
except BaseException as exc:
 (root/'exception').write_text(str(exc))
 (root/'restored').write_text(str(signal.getsignal(signal.SIGTERM)==old))
 raise SystemExit(1)
'''
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'batch'
            result=subprocess.run([sys.executable,'-B','-c',code,str(root)],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertTrue((root/'stopped').exists(),result.stderr)
            self.assertTrue((root/'finished').exists(),result.stderr)
            self.assertEqual((root/'restored').read_text(),'True')
            self.assertIn('SIGTERM',(root/'exception').read_text())
            self.assertIn('SIGTERM',json.loads((root/'abort.json').read_text())['error'])

    def test_abort_write_failure_still_cleans_and_preserves_interrupt(self):
        from brain_uav.scripts import v2_mps_diagnostic as mps
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp)/'batch',True);session=Mock();session.env={};session.finish.return_value={'stopped':True}
            processes=[Mock(pid=i+10) for i in range(4)]
            for p in processes:p.poll.return_value=None
            original=run.atomic_json
            def write(path,value):
                if Path(path).name in ('abort.json','process.json'):raise OSError('report failure')
                return original(path,value)
            with patch.object(run.provenance,'PYTHON',sys.executable),patch.object(run,'inputs',return_value={}), \
                 patch.object(mps,'Session',return_value=session),patch.object(mps,'capability'), \
                 patch.object(run.subprocess,'Popen',side_effect=processes) as spawn,patch.object(run,'wait_ready',side_effect=KeyboardInterrupt), \
                 patch.object(run,'atomic_json',side_effect=write):
                with self.assertRaises(KeyboardInterrupt):run.launch(a)
                self.assertTrue(all(c.kwargs['start_new_session'] for c in spawn.call_args_list))
            session.finish.assert_called_once()
            self.assertEqual(session.before_terminate.call_count,4)
            for p in processes:p.terminate.assert_called_once()


if __name__=='__main__':unittest.main()
