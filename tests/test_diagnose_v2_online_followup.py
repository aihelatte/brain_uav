"""CPU/tiny fixtures only; never historical model flight or CUDA."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brain_uav.scripts import diagnose_v2_online_followup as d
from brain_uav.trainers.v2_online_research import OnlineResearch,target_config,training_identity,predict_bank
import test_v2_gpu_compute as compute_fixture


class TestFollowup(unittest.TestCase):
    def args(self,root,mode='snn-inference'):
        root=Path(root).resolve()
        return d.parser().parse_args([mode,'--history',str(root/'history'),'--historical-repo',str(root/'old'),
            '--validation-pool',str(root/'pool'),'--ann-bc',str(root/'ann'),'--snn-bc',str(root/'snn'),
            '--output-dir',str(root/'output'),'--source-manifest',str(root/'source')])

    def test_real_cli_plan_only_no_io_models_cuda_or_jobs(self):
        with TemporaryDirectory() as root:
            for mode in ('snn-inference','acc-bank'):
                args=self.args(root,mode)
                argv=[mode,'--history',str(args.history),'--historical-repo',str(args.historical_repo),
                    '--validation-pool',str(args.validation_pool),'--ann-bc',str(args.ann_bc),'--snn-bc',str(args.snn_bc),
                    '--output-dir',str(args.output_dir),'--source-manifest',str(args.source_manifest)]
                with patch.object(d,'execute',side_effect=AssertionError('execute')),patch.object(d.cal.base,'load_models',side_effect=AssertionError('load')), \
                        patch.object(d.research,'identity',side_effect=AssertionError('identity')),patch.object(torch.cuda,'init',side_effect=AssertionError('CUDA')), \
                        patch.object(d.subprocess,'run',side_effect=AssertionError('process')),redirect_stdout(StringIO()) as out:
                    self.assertEqual(d.main(argv),0)
                self.assertFalse(json.loads(out.getvalue())['launch'])
                self.assertEqual(list(Path(root).iterdir()),[])
            args.launch=True
            with patch.object(d,'execute',return_value={'strict_passed':True}) as execute:
                self.assertEqual(d.main(argv+['--launch']),0);execute.assert_called_once()
            args.updates_per_worker=201
            with self.assertRaises(ValueError):d.plan(args)

    def test_location_guard_protects_original_repo_and_history(self):
        with TemporaryDirectory() as root:
            args=self.args(root)
            args.output_dir=args.history/'bad'
            with self.assertRaises(ValueError):d.guard_locations(args)
            args.output_dir=Path(root)/'safe'
            with patch.object(d.research,'REPO',args.historical_repo):
                with self.assertRaises(ValueError):d.guard_locations(args)

    def test_actual_action_entries_independent_snn_and_sequences(self):
        from test_v2_spike_regularization import TestSpikeRegularization
        engine=TestSpikeRegularization().engine(delay=2)
        actor=deepcopy(engine.actor).eval().requires_grad_(False)
        data=dict(actor=actor,device=torch.device('cpu'))
        batch=engine.replay.sample(2)
        # Obtain the public observations directly from the fixture's real replay.
        from test_v2_td3 import _observation
        states=[{'observation':_observation(n,scales=engine.actor.scales)} for n in (0,2)]
        identities={p.data_ptr() for p in engine.actor.parameters()}
        self.assertFalse(identities & {p.data_ptr() for p in actor.parameters()})
        before=training_identity(engine)
        with patch.object(engine,'select_action',wraps=engine.select_action) as train, \
                patch.object(d.cal.base,'actor_action',wraps=d.cal.base.actor_action) as evaluate, \
                d.action_observer(engine,data) as current:
            result=d.action_sequences(engine,data,states,current)
        self.assertEqual(train.call_count,4);self.assertEqual(evaluate.call_count,4)
        self.assertTrue(result['passed'],result)
        self.assertEqual(training_identity(engine),before)
        for row in result['sequence']:
            self.assertTrue(row['context']['train_context']['inference'])
            self.assertTrue(row['context']['eval_context']['inference'])
            self.assertEqual(row['pulses']['status'],'partial')
            self.assertEqual(set(row['pulses']['evaluate']),{'l1','l2'})

    def test_isolated_actual_scoring_does_not_mutate_live_state_or_rng(self):
        engine=compute_fixture.TestGPUCompute().engine()
        from test_v2_td3 import _observation
        rows=[dict(observation=_observation(i%3,scales=engine.actor.scales),action=np.zeros(2,dtype=np.float32)) for i in range(65)]
        scorer=d.IsolatedScorer(deepcopy((engine.critic1,engine.critic2)))
        before=training_identity(engine)
        expected=d.protected_score(predict_bank,engine,rows)
        actual=d.protected_score(scorer,engine,rows)
        np.testing.assert_allclose(expected,actual,rtol=d.RTOL,atol=d.ATOL)
        self.assertEqual(training_identity(engine),before)
        def corrupt(e,rows):
            with torch.no_grad():next(e.critic1.parameters()).add_(1)
            return np.zeros((len(rows),2))
        with self.assertRaisesRegex(RuntimeError,'polluted'):d.protected_score(corrupt,engine,rows)

    def test_real_hook_bank_controller_then_real_update_uses_beta(self):
        from test_v2_td3 import _observation
        engines=[compute_fixture.TestGPUCompute().engine(),compute_fixture.TestGPUCompute().engine()]
        for e in engines:e.configure_research_targets(target_config('acc_001'))
        engines[1].load_checkpoint_state_dict(deepcopy(engines[0].checkpoint_state_dict()))
        episode=[dict(observation=_observation(0,scales=engines[0].actor.scales),action=np.zeros(2,dtype=np.float32),
            reward=100.,step=1+i,done=i==1) for i in range(2)]
        with TemporaryDirectory() as root:
            records=[];targets=[]
            for i,e in enumerate(engines):
                hook=OnlineResearch('acc_001',Path(root)/str(i))
                scorer=predict_bank if i==0 else d.IsolatedScorer(deepcopy((e.critic1,e.critic2)))
                with patch.object(hook.acc,'adjust',wraps=hook.acc.adjust) as adjust:
                    result=d.bank_episode(hook,e,episode,25000,lambda live,rows:d.protected_score(scorer,live,rows))
                self.assertEqual(adjust.call_count,1);self.assertGreater(e.research_beta,.5)
                self.assertEqual(len(result['scores']),1);self.assertEqual(result['scores'][0]['rows'],2)
                records.append(result)
                batch=d.replay_batch(e,episode);evidence={}
                with patch.object(e.replay,'sample',return_value=batch),patch('torch.randn_like',new=torch.zeros_like):
                    e.update_once(total_steps=25002,diagnostic_tensor_recorder=lambda n,t:evidence.update({n:t.detach().clone()}))
                target=evidence['target.td'];q1=evidence['target.q1'];q2=evidence['target.q2']
                expected=batch.reward+(1-batch.done)*e.gamma*e._aggregate_target_q(q1,q2)
                torch.testing.assert_close(target,expected,rtol=d.RTOL,atol=d.ATOL)
                checked=d.beta_target_check(evidence,batch,e.gamma,.5,e.research_beta)
                self.assertEqual(checked['runtime_target_matches'],[True,True])
                e.set_research_beta(.5)
                old=batch.reward+(1-batch.done)*e.gamma*e._aggregate_target_q(q1,q2)
                self.assertFalse(torch.equal(target,old));targets.append(target)
            self.assertEqual(records[0]['control'],records[1]['control'])
            torch.testing.assert_close(targets[0],targets[1],rtol=d.RTOL,atol=d.ATOL)
            for a,b in zip(engines[0].critic1.parameters(),engines[1].critic1.parameters()):
                torch.testing.assert_close(a,b,rtol=d.RTOL,atol=d.ATOL)

    def test_input_replacement_and_pool_identity_fail_closed(self):
        with TemporaryDirectory() as root:
            args=self.args(root)
            for name in ('ann_bc','snn_bc','validation_pool'):getattr(args,name).write_bytes(name.encode())
            args.source_manifest.write_text(json.dumps({'sha256':{'a':'reviewed'}}))
            args.history.mkdir();(args.history/'manifest.json').write_text(json.dumps({'preflight':{'source':{'sha256':{'old':'history'}}}}))
            bc={m:(None,d.file_sha256(getattr(args,m+'_bc'))) for m in ('ann','snn')}
            with patch.object(d.research,'identity',return_value={'sha256':{'a':'reviewed'}}),patch.object(d.research,'BC',bc):
                inputs=d.Inputs(args);inputs.check()
                with patch.object(d.cal.base,'load_models',return_value={'checkpoint_hash':inputs.files[str(args.ann_bc.resolve())],
                        'pool':SimpleNamespace(content_digest='other legitimate pool')}):
                    with self.assertRaisesRegex(ValueError,'pool identity'):d.load_data(args.ann_bc,args,inputs,'cpu')
                args.validation_pool.write_bytes(b'replacement')
                with self.assertRaisesRegex(ValueError,'changed'):inputs.check()

    def test_snapshot_metadata_wrong_hash_or_actual_rejected(self):
        with TemporaryDirectory() as root:
            args=self.args(root);job=args.history/'snn_min'
            (job/'review').mkdir(parents=True);(job/'other').mkdir()
            torch.save({'arm':'min','requested_step':200000,'actual_step':200010},job/'other/not_prefix.pt')
            sidecar=job/'review/snapshot_x.json'
            sidecar.write_text(json.dumps({'requested_step':200000,'actual_step':200011,
                'sha256':d.file_sha256(job/'other/not_prefix.pt'),'pool_digest':d.research.DIGEST}))
            tracker=SimpleNamespace(add=lambda p:Path(p))
            with patch.object(d,'validate_snapshot') as validate:
                with self.assertRaisesRegex(ValueError,'metadata'):d.discover(args,tracker,'snn_min',200000)
                validate.assert_called_once()
            doc=json.loads(sidecar.read_text());doc['sha256']='wrong';sidecar.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'identity'):d.discover(args,tracker,'snn_min',200000)

    def test_discover_real_strict_validator_binds_model_both_directions(self):
        import test_v2_formal_training as fixture
        import test_v2_spike_regularization as snn_fixture
        from brain_uav.config import ScenarioConfig,RewardConfig
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig,build_v2_periodic_snapshot
        from brain_uav.trainers.v2_online_research import FORMAT
        for model in ('ann','snn'):
            with self.subTest(model=model),TemporaryDirectory() as root:
                args=self.args(root);scenario=ScenarioConfig()
                engine=fixture._engine(scenario) if model=='ann' else snn_fixture.TestSpikeRegularization().engine()
                saved=build_v2_periodic_snapshot(engine,V2FormalTrainingConfig(stage='easy'),stage_steps=200010,
                    scenario=scenario,rewards=RewardConfig(),uav_collision_radius=0.,
                    seed_manifest={'base_seed':7},initialization_source={'kind':'test'})
                saved['research_only']=True
                raw=dict(format=FORMAT,format_version=1,diagnostic_only=True,formal_passed=False,
                    arm='min',target_config=None,beta=.5,controller=None,requested_step=200000,actual_step=200010,snapshot=saved)
                for directory_model in ('ann','snn'):
                    job=args.history/(directory_model+'_min')
                    (job/'review').mkdir(parents=True);(job/'other').mkdir()
                    path=job/'other/arbitrary.pt';torch.save(raw,path)
                    (job/'review/snapshot_x.json').write_text(json.dumps(dict(requested_step=200000,actual_step=200010,
                        sha256=d.file_sha256(path),pool_digest=d.research.DIGEST)))
                    tracker=SimpleNamespace(add=lambda p:Path(p))
                    if model==directory_model:d.discover(args,tracker,directory_model+'_min',200000)
                    else:
                        with self.assertRaisesRegex(ValueError,'model type'):d.discover(args,tracker,directory_model+'_min',200000)

    def test_beta_same_update_targets_new_old_and_information_insufficient(self):
        from brain_uav.trainers.v2_td3 import research_target_pair
        batch=SimpleNamespace(reward=torch.tensor([[1.],[2.],[3.]]),done=torch.tensor([[0.],[0.],[1.]]))
        q1=torch.tensor([[1.],[8.],[5.]]);q2=torch.tensor([[7.],[2.],[4.]])
        def evidence(beta,a=q1,b=q2):
            return {'target.q1':a,'target.q2':b,'target.td':batch.reward+(1-batch.done)*.99*research_target_pair(a,b,'acc',beta)}
        good=d.beta_target_check(evidence(.51),batch,.99,.5,.51)
        self.assertTrue(good['strict_passed']);self.assertEqual(good['runtime_target_matches'],[True,True])
        self.assertEqual(good['beta_refresh_distinguishable_on_same_target_q'],[True,True])
        stale=d.beta_target_check(evidence(.5),batch,.99,.5,.51)
        self.assertFalse(stale['strict_passed']);self.assertEqual(stale['status'],'failed')
        for ev,old,new in ((evidence(.5),.5,.5),(evidence(.51,q1,q1),.5,.51)):
            result=d.beta_target_check(ev,batch,.99,old,new)
            self.assertFalse(result['strict_passed']);self.assertEqual(result['status'],'information_insufficient')
        bad=evidence(.51);bad['target.q1'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):d.beta_target_check(bad,batch,.99,.5,.51)

    def test_exception_summary_is_not_a_success(self):
        with TemporaryDirectory() as root:
            args=self.args(root);args.launch=True
            tracker=SimpleNamespace(evidence=lambda:{'files':{},'diagnostic_source':'new','historical_source':'old'},check=lambda:None)
            with patch.object(d,'Inputs',return_value=tracker),patch.object(d,'inference',side_effect=RuntimeError('synthetic failure')):
                with self.assertRaises(RuntimeError):d.execute(args)
            summary=json.loads((args.output_dir/'summary.json').read_text())
            self.assertFalse(summary['execution_completed']);self.assertIsNone(summary['strict_passed'])
            self.assertEqual(summary['exit_code'],1)

    def test_strict_failure_is_distinct_from_incomplete(self):
        with TemporaryDirectory() as root:
            args=self.args(root);args.launch=True
            tracker=SimpleNamespace(evidence=lambda:{},check=lambda:None)
            with patch.object(d,'Inputs',return_value=tracker),patch.object(d,'inference',return_value={'strict_passed':False}):
                result=d.execute(args)
            self.assertTrue(result['execution_completed']);self.assertFalse(result['strict_passed']);self.assertEqual(result['exit_code'],1)

    def test_beta_worker_uses_real_hook_update_and_same_update_recorder_cpu(self):
        from test_v2_td3 import _observation
        from brain_uav.v2_curriculum import V2NoiseSchedule
        e=compute_fixture.TestGPUCompute().engine();e.configure_research_targets(target_config('acc_001'))
        e.bc_reference_actor=deepcopy(e.actor).eval().requires_grad_(False)
        # Tiny CPU fixture only: distinct constant target critics, never edit historical G.
        for n,c in enumerate((e.critic1_target,e.critic2_target)):
            with torch.no_grad():
                for p in c.parameters():p.zero_()
                c.head[-1].bias.fill_(1.+4*n)
        episode=[dict(observation=_observation(0,scales=e.actor.scales),action=np.zeros(2,dtype=np.float32),
            reward=0.,step=1+i,done=i==1) for i in range(2)]
        with TemporaryDirectory() as root:
            args=self.args(root,'acc-bank');args.worker_model='ann';args.bank_path='beta'
            out=args.output_dir;(out/'other').mkdir(parents=True)
            path=Path(root)/'bank.pt';path.write_bytes(b'CPU fixture')
            raw=dict(arm='acc_001',actual_step=75001,controller={'episodes':[episode]},
                snapshot={'engine_checkpoint':{},'scenario_config':{},'reward_config':{},'uav_collision_radius':0.})
            inputs=SimpleNamespace(files={str(path.resolve()):d.file_sha256(path)},check=lambda:None)
            # CPU cannot execute Graph; only mock its post-update metadata, not the hook/update/recorder.
            executor=SimpleNamespace(captures=0,replays=1,metadata=lambda:{'mock_cpu_only':True})
            e._gpu_update_executor=executor
            with patch.object(d,'discover',return_value=(path,raw,{})),patch.object(d,'load_data',return_value={}), \
                    patch.object(d,'make_engine',return_value=(e,deepcopy((e.critic1,e.critic2)),
                        SimpleNamespace(noise_schedule=V2NoiseSchedule(),gamma=e.gamma),{})),patch.object(e,'invalidate_gpu_update'):
                result=d.bank_worker(args,inputs,out)
            self.assertEqual(e.update_count,1);self.assertEqual(result['physical_updates_upper_bound'],1)
            check=result['beta_check'];self.assertEqual(check['runtime_target_matches'],[True,True])
            self.assertTrue(check['strict_passed'])
            self.assertEqual(check['beta_new'],check['control']['beta_after'])
            self.assertTrue((out/'other/beta_target_check.json').exists())

    def test_beta_failure_blocks_top_level_acceptance_and_budget_is_shared(self):
        with TemporaryDirectory() as root:
            args=self.args(root,'acc-bank');args.launch=True
            description=d.plan(args)
            self.assertEqual(description['controlled_updates'],26)
            self.assertEqual(4*description['per_main_process_budget']+2*description['per_beta_process_budget'],1000)
            tracker=SimpleNamespace(evidence=lambda:{},check=lambda:None)
            with patch.object(d,'Inputs',return_value=tracker),patch.object(d,'compare_bank_workers',
                    side_effect=d.StrictMismatch('beta information insufficient')):
                with self.assertRaises(d.StrictMismatch):d.execute(args)
            summary=json.loads((args.output_dir/'summary.json').read_text())
            self.assertFalse(summary['execution_completed']);self.assertIs(summary['strict_passed'],False)
            self.assertEqual(summary['physical_update_budgets'],dict(main_reproduction=800,beta_check=200,total=1000))

    def test_acc_children_are_independent_bounded_and_fail_closed(self):
        import subprocess
        for condition in ('success','rc','timeout','coverage','numerical','identity','beta_failed','beta_insufficient'):
            with self.subTest(condition=condition),TemporaryDirectory() as root:
                args=self.args(root,'acc-bank');out=Path(root)/'results'
                (out/'review').mkdir(parents=True);(out/'other').mkdir()
                source=Path(root)/'historical_input';source.write_bytes(b'original')
                expected_hash=d.file_sha256(source)
                def check():
                    if d.file_sha256(source)!=expected_hash:raise ValueError('Historical input changed.')
                inputs=SimpleNamespace(files={},check=check) if condition=='identity' else None
                calls=[]
                def child(cmd,**kwargs):
                    calls.append(cmd)
                    if condition=='timeout':raise subprocess.TimeoutExpired(cmd,1)
                    if condition=='rc':return SimpleNamespace(returncode=1)
                    directory=Path(cmd[cmd.index('--output-dir')+1]);(directory/'other').mkdir(parents=True)
                    (directory/'review').mkdir()
                    (directory/'review/used_inputs_final.json').write_text(json.dumps({'files':{str(source):expected_hash}}))
                    if condition=='identity' and len(calls)==2:source.write_bytes(b'replaced')
                    value=2. if condition=='numerical' and len(calls)==2 else 1.
                    torch.save({'parameter':torch.tensor(value)},directory/'other/state_0.pt')
                    record=dict(repeat_Q=[[1.,1.]],scores=[{'Q':[[1.,1.]]}],control={'beta_after':.5},
                        batch_sha256='batch',noise_sha256='noise')
                    result=dict(diagnostic_valid=condition!='coverage',initial_state_sha256='same',bank_sha256='same-bank',
                        records=[record],physical_updates_upper_bound=1)
                    if cmd[cmd.index('--bank-path')+1]=='beta':
                        result.update(strict_passed=condition not in ('beta_failed','beta_insufficient'),
                            beta_check={'status':'information_insufficient' if condition=='beta_insufficient' else 'failed' if condition=='beta_failed' else 'passed',
                                'strict_passed':condition not in ('beta_failed','beta_insufficient'),
                                'information_sufficient':condition!='beta_insufficient'})
                    negative=cmd[cmd.index('--bank-path')+1]=='beta' and condition in ('beta_failed','beta_insufficient')
                    if negative:result['diagnostic_valid']=False
                    (directory/'summary.json').write_text(json.dumps({'diagnostic_only':True,'execution_completed':True,
                        'strict_passed':not negative,'exit_code':1 if negative else 0,'result':result}))
                    return SimpleNamespace(returncode=1 if negative else 0)
                with patch.object(d.subprocess,'run',side_effect=child):
                    if condition=='success':self.assertTrue(d.compare_bank_workers(args,out,inputs)['strict_passed'])
                    else:
                        with self.assertRaises((RuntimeError,ValueError,subprocess.TimeoutExpired,d.StrictMismatch)):d.compare_bank_workers(args,out,inputs)
                expected=6 if condition=='success' else 3 if condition in ('numerical','beta_failed','beta_insufficient') else 2 if condition=='identity' else 1
                self.assertEqual(len(calls),expected)
                for cmd in calls:
                    self.assertIn('brain_uav.scripts.diagnose_v2_online_followup',cmd)
                    self.assertIn('--launch',cmd);self.assertIn('--worker-model',cmd);self.assertIn('--bank-path',cmd)
                    self.assertEqual(cmd[cmd.index('--history')+1],str(args.history))
                if condition=='timeout':
                    status=json.loads((out/'review/ann_original_process.json').read_text())
                    self.assertTrue(status['timed_out']);self.assertIsNone(status['exit_code'])

    def test_real_parent_classifies_beta_negative_and_runtime_failures(self):
        import subprocess
        cases=('failed','information_insufficient','cuda','incomplete','missing','damaged','bad_structure',
            'unexpected_rc','zero_rc_negative','wrong_summary_rc','missing_contract','timeout')
        for case in cases:
            with self.subTest(case=case),TemporaryDirectory() as root:
                args=self.args(root,'acc-bank');args.launch=True;calls=[]
                tracker=SimpleNamespace(files={},evidence=lambda:{},check=lambda:None)
                def child(cmd,**kwargs):
                    calls.append(cmd);path=cmd[cmd.index('--bank-path')+1]
                    directory=Path(cmd[cmd.index('--output-dir')+1]);directory.mkdir(parents=True)
                    (directory/'review').mkdir()
                    (directory/'review/used_inputs_final.json').write_text(json.dumps({'files':{}}))
                    if path!='beta':
                        document={'execution_completed':True,'result':{'diagnostic_valid':True}}
                        (directory/'summary.json').write_text(json.dumps(document))
                        return SimpleNamespace(returncode=0)
                    if case=='timeout':raise subprocess.TimeoutExpired(cmd,1)
                    reason='information_insufficient' if case=='information_insufficient' else 'failed'
                    document=dict(diagnostic_only=True,execution_completed=True,strict_passed=False,exit_code=1,
                        result=dict(strict_passed=False,diagnostic_valid=False,
                            beta_check=dict(strict_passed=False,status=reason,information_sufficient=reason=='failed')))
                    rc=1
                    if case=='cuda':document.update(execution_completed=False,strict_passed=None,error='CUDA illegal memory access')
                    if case=='incomplete':document['execution_completed']=False
                    if case=='bad_structure':document['result']=None
                    if case=='unexpected_rc':rc=2
                    if case=='zero_rc_negative':rc=0
                    if case=='wrong_summary_rc':document['exit_code']=0
                    if case=='missing_contract':document['result']['beta_check'].pop('strict_passed')
                    if case!='missing':(directory/'summary.json').write_text('{' if case=='damaged' else json.dumps(document))
                    return SimpleNamespace(returncode=rc)
                with patch.object(d,'Inputs',return_value=tracker),patch.object(d.subprocess,'run',side_effect=child):
                    with self.assertRaises(Exception) as caught:d.execute(args)
                summary=json.loads((args.output_dir/'summary.json').read_text())
                self.assertFalse(summary['execution_completed']);self.assertEqual(summary['exit_code'],1)
                self.assertEqual(len(calls),3)
                self.assertEqual([c[c.index('--bank-path')+1] for c in calls],['original','isolated','beta'])
                if case in ('failed','information_insufficient'):
                    self.assertIsInstance(caught.exception,d.StrictMismatch)
                    self.assertIs(summary['strict_passed'],False);self.assertIn(case,summary['error'])
                else:
                    self.assertNotIsInstance(caught.exception,d.StrictMismatch)
                    self.assertIsNone(summary['strict_passed'])

    def test_common_trajectory_identity_rejected_before_replay(self):
        with TemporaryDirectory() as root:
            args=self.args(root)
            directory=args.history/'snn_min/other/eval_200000_200001';directory.mkdir(parents=True)
            (directory/'manifest.json').write_text(json.dumps({'pool_digest':'wrong','checkpoint_sha256':'model'}))
            inputs=SimpleNamespace(add=lambda p:Path(p))
            with patch.object(d.cal,'replay_episode') as replay:
                with self.assertRaisesRegex(ValueError,'identity'):d.common_observations(args,inputs,{'checkpoint_hash':'model'},200001)
                replay.assert_not_called()

    def test_prewarm_host_counts_do_not_change_call_returns_or_persist(self):
        engine=SimpleNamespace(_action_inference_tensor=lambda x:x+1,
            _prepare_action_inference_cuda_graph_replay=lambda x:lambda:x+2)
        original=engine._action_inference_tensor
        with d.warmup_call_counts(engine) as counts:
            self.assertEqual(engine._action_inference_tensor(1),2)
            self.assertEqual(engine._action_inference_tensor(2),3)
            replay=engine._prepare_action_inference_cuda_graph_replay(3)
            self.assertEqual(replay(),5)
        self.assertIs(engine._action_inference_tensor,original)
        self.assertEqual(counts,{'action_tensor_calls':2,'compiled_replay_probe_calls':1})

    def test_explicit_closed_loop_natural_end_and_same_forward_evidence(self):
        from test_v2_spike_regularization import TestSpikeRegularization
        from test_v2_td3 import _observation
        engine=TestSpikeRegularization().engine(delay=2)
        obs=_observation(0,scales=engine.actor.scales)
        data=dict(actor=deepcopy(engine.actor).eval().requires_grad_(False),device=torch.device('cpu'),
            gamma=.99,scenario=SimpleNamespace(max_steps=2))
        class TinyEnv:
            def reset(self,**kwargs):
                self.state=np.zeros(5);self.steps=0
                return obs,{}
            def step(self,action):
                self.steps+=1;self.state[:2]+=action
                return obs,1.,self.steps==2,False,{'outcome':'goal' if self.steps==2 else 'running'}
        scenes=[{'scenario_id':'empty','payload':{'zones':[]}}, {'scenario_id':'nonempty','payload':{'zones':['fixture']}}]
        with patch.object(d.cal.base,'make_env',side_effect=lambda data:TinyEnv()),d.action_observer(engine,data) as current:
            result=d.closed_loop(engine,data,scenes,current)
        self.assertEqual(len(result),2)
        for scene in result:
            self.assertTrue(scene['outcomes_equal'])
            for branch in scene['branches']:
                self.assertEqual(branch['steps'],2);self.assertTrue(branch['terminated'])
                self.assertEqual(len(branch['raw_actions']),2)
                self.assertEqual(branch['episode_return'],2.)
                self.assertAlmostEqual(branch['discounted_return'],1.99)
                if branch['path']=='eval':self.assertIn('sha256',branch['pulses'][0])
                else:self.assertEqual(branch['pulses'],[None,None])


if __name__=='__main__':unittest.main()
