"""Bounded CPU research-contract checks; never launch source-model training."""
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
from brain_uav.trainers import v2_td3 as td
import test_v2_formal_training as f
from brain_uav.config import ScenarioConfig,RewardConfig


class TestOnlineTargets(unittest.TestCase):
    def test_research_target_endpoints_and_done(self):
        self.assertTrue(hasattr(td, 'research_target_pair'))
        a=torch.tensor([[2.],[8.]]); b=torch.tensor([[6.],[4.]])
        for mode,weight,expected in [('wd3',0.,(a+b)/2),('wd3',1.,torch.minimum(a,b)),('acc',0.,torch.minimum(a,b))]:
            y=td.research_target_pair(a,b,mode,weight)
            self.assertTrue(torch.equal(y[:,0:1],expected));self.assertTrue(torch.equal(y[:,1:2],expected))
        y=td.research_target_pair(a,b,'acc',torch.tensor(1.))
        self.assertTrue(torch.equal(y,torch.cat((a,b),-1)))
        self.assertTrue(torch.equal(7+0*y,torch.full_like(y,7)))

    def test_controller_complete_episode_and_interval(self):
        from brain_uav.trainers.v2_online_research import ACCController
        c=ACCController(.1, gamma=.5, capacity=3)
        rows=[dict(observation='obs',action=np.array([.1,.2]),reward=2.,step=i+1,done=i==1) for i in range(2)]
        c.complete_episode(rows,2000,'training')
        self.assertEqual([r['G'] for r in c.episodes[0]], [3.,2.])
        self.assertIsNone(c.adjust(24999,lambda rows: np.zeros((len(rows),2))))
        result=c.adjust(25000,lambda rows: np.zeros((len(rows),2)))
        self.assertEqual(result['beta_after'],.6);self.assertEqual(result['EMA'],2.5)
        self.assertIsNone(c.adjust(25999,lambda rows: self.fail('too early')))
        c.adjust(26000,lambda rows: np.full((len(rows),2),10.))
        self.assertAlmostEqual(c.ema,.95*2.5+.05*7.5)
        c.complete_episode(rows,27000,'training')
        self.assertEqual(len(c.episodes),1)
        with self.assertRaises(ValueError): c.complete_episode(rows,28000,'validation')
        self.assertEqual(rows[0]['action'][0],.1)

    def test_plan_no_io_and_matrix(self):
        from brain_uav.scripts import run_v2_online_wd3_acc as r
        with TemporaryDirectory() as root:
            args=r.parser().parse_args(['plan','--root',root])
            with patch.object(r,'preflight',side_effect=AssertionError('preflight')):
                plan=r.plan(args)
            self.assertEqual(len(plan['jobs']),16)
            self.assertEqual(len(list(Path(root).iterdir())),0)
            for model in ('ann','snn'):
                self.assertEqual(len([j for j in plan['jobs'] if j['model']==model]),8)
            self.assertTrue(all(j['schedule_horizon_steps']==500000 and j['mc_return_weight']==0 for j in plan['jobs']))
            self.assertEqual(r.parser().parse_args([]).mode,'plan')
            from contextlib import redirect_stdout
            from io import StringIO
            for mode in ('launch','evaluate','probe','confirmation','cuda-check'):
                with patch.object(r,'launch',side_effect=AssertionError('no launch')), \
                        patch.object(r,'preflight',side_effect=AssertionError('no inputs')),redirect_stdout(StringIO()) as out:
                    self.assertEqual(r.main([mode,'--root',str(Path(root)/mode)]),0)
                self.assertFalse(__import__('json').loads(out.getvalue())['launch'])
                self.assertFalse((Path(root)/mode).exists())

    def test_shared_targets_compile_beta_binding_and_strict_engine_load(self):
        from brain_uav.trainers.v2_online_research import target_config
        from brain_uav.trainers.v2_gpu_update import GPUUpdate
        engine=f._engine(ScenarioConfig());engine.configure_research_targets(target_config('acc_001'))
        args=[torch.zeros(1)]*7;q1=torch.tensor([[2.],[8.]]);q2=torch.tensor([[6.],[4.]])
        with patch.object(engine.critic1_target,'_compute_full_forward_tensors',return_value=q1), \
                patch.object(engine.critic2_target,'_compute_full_forward_tensors',return_value=q2):
            call=lambda:engine._compute_target_critics_td_tensors(*args,torch.zeros(2,2),torch.ones(2,1),torch.tensor([[1.],[0.]]))[-1]
            initial=call();binding=GPUUpdate.static_options(engine);address=engine._research_beta_tensor.data_ptr()
            engine.set_research_beta(1.)
            following=call()
            self.assertEqual(GPUUpdate.static_options(engine),binding);self.assertEqual(address,engine._research_beta_tensor.data_ptr())
            self.assertTrue(torch.equal(initial[0],torch.ones(2)));self.assertFalse(torch.equal(initial[1],following[1]))
            self.assertTrue(torch.equal(following[1],1+engine.gamma*torch.tensor([8.,4.])))
        saved=deepcopy(engine.checkpoint_state_dict());engine.set_research_beta(.2);engine.load_checkpoint_state_dict(saved)
        self.assertEqual(engine.research_beta,1.)
        bad=deepcopy(saved);bad['algorithm_config']['online_research_target']['method']='unknown'
        with self.assertRaises(ValueError):engine.load_checkpoint_state_dict(bad)
        old=f._engine(ScenarioConfig())
        with self.assertRaises(ValueError):old.load_checkpoint_state_dict(saved)
        compiled=torch.compile(lambda a,b,w:td.research_target_pair(a,b,'acc',w),backend='eager',fullgraph=True)
        self.assertFalse(torch.equal(compiled(q1,q2,torch.tensor(0.)),compiled(q1,q2,torch.tensor(1.))))

    def test_real_trainer_pause_stack_budget_and_non_acc_skip(self):
        from brain_uav.trainers.v2_online_research import OnlineResearch,atomic_json
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig,V2FormalStageTrainer
        from brain_uav.trainers.v2_mc_return import discounted_episode_returns
        scenario=ScenarioConfig(max_steps=2);engine=f._engine(scenario)
        config=V2FormalTrainingConfig(stage='easy',max_steps=500000,batch_size=2)
        source=f._Source(f._scenario_payload(0));calls=[]
        with TemporaryDirectory() as directory:
            hook=OnlineResearch('min',directory);hook.pause_step=2;hook.hard_limit=5
            def sleep(seconds):
                before=(source.calls,len(engine.replay),engine.update_count,deepcopy(engine.replay.rng.bit_generator.state))
                self.assertEqual(before[0],1);self.assertEqual(before[1],2)
                self.assertTrue(Path(directory,'review/progress.json').exists())
                atomic_json(Path(directory)/'continue_to_400k.json',dict(pid=__import__('os').getpid(),arm='min',actual_step=2,continue_to=400000))
                self.assertEqual(before,(source.calls,len(engine.replay),engine.update_count,engine.replay.rng.bit_generator.state))
            hook.sleep=sleep
            trainer=V2FormalStageTrainer(scenario,RewardConfig(),config,engine,scenario_sources={'easy':source},
                validation_runner=lambda a:self.fail('candidate validation'),online_research=hook,
                weight_snapshot_steps=(2,4),weight_snapshot_sink=lambda r,a:calls.append((r,a)))
            with patch.object(engine,'update_once',return_value=td.V2TD3UpdateMetrics()) as updates, \
                    patch('brain_uav.trainers.v2_mc_return.discounted_episode_returns',side_effect=AssertionError('non ACC/MC must skip G')):
                result=trainer.run()
            self.assertEqual(result.stage_steps,5);self.assertEqual(len(result.episodes),2)
            self.assertEqual(result.status,'research_completed_budget')
            self.assertEqual(result.candidate_validation_count,0)
            self.assertEqual(source.calls,3);self.assertEqual(calls,[(2,2),(4,4)])
            self.assertEqual(updates.call_count,4);self.assertIsNone(hook.acc)
        noise=config.noise_schedule.values(100000,max_steps=config.max_steps)
        for actual,expected in zip(noise,(.014,.0114,.0228)):self.assertAlmostEqual(actual,expected)

    def test_snapshot_explicit_diagnostic_load_and_tamper_rejection(self):
        from brain_uav.trainers.v2_online_research import FORMAT,validate_snapshot,target_config
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig,build_v2_periodic_snapshot,load_v2_periodic_snapshot
        from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_checkpoint
        scenario=ScenarioConfig();e=f._engine(scenario);config=V2FormalTrainingConfig(stage='easy',max_steps=500000,batch_size=2)
        e.configure_research_targets(target_config('wd3_045'))
        saved=build_v2_periodic_snapshot(e,config,stage_steps=75010,scenario=scenario,rewards=RewardConfig(),
            uav_collision_radius=0.,seed_manifest={},initialization_source={},bc_schedule='smooth-mid-strong')
        saved['research_only']=True
        raw=dict(format=FORMAT,format_version=1,diagnostic_only=True,formal_passed=False,arm='wd3_045',
            target_config=target_config('wd3_045'),beta=.5,controller=None,requested_step=75000,actual_step=75010,snapshot=saved)
        # Fill strict legacy provenance with the fixture's documented seeds.
        from brain_uav.v2_curriculum import derive_v2_component_seed
        raw['snapshot']['seed_manifest']={'base_seed':config.seed,**{f'{k}_seed':derive_v2_component_seed(config.seed,'easy',k)
            for k in ('model','replay','curriculum','exploration')}}
        raw['snapshot']['initialization_source']={'kind':'test'}
        with TemporaryDirectory() as root:
            path=Path(root)/'research.pt';torch.save(raw,path)
            loaded=load_diagnostic_checkpoint(path,expected_stage='easy')
            self.assertEqual(loaded['diagnostic_research']['arm'],'wd3_045')
            with self.assertRaises(ValueError):load_v2_periodic_snapshot(raw)
            with self.assertRaises(ValueError):load_v2_periodic_snapshot(raw['snapshot'])
            unwrapped=deepcopy(raw['snapshot']);unwrapped.pop('research_only')
            with self.assertRaisesRegex(ValueError,'research'):load_v2_periodic_snapshot(unwrapped)
        for key,value in [('beta',float('nan')),('arm','unknown'),('actual_step',1)]:
            bad=deepcopy(raw);bad[key]=value
            with self.assertRaises(ValueError):validate_snapshot(bad)

    def test_preflight_uses_real_paths_and_strict_bc_pool_load(self):
        from brain_uav.scripts import run_v2_online_wd3_acc as r
        from brain_uav.trainers.v2_formal_training import prepare_v2_stage_initialization
        from brain_uav.trainers.v2_validation import load_v2_validation_pool
        local=Path('E:/wurenji/train_result')
        bc={'ann':local/'ann_20261012_102839/bc/bc_v2_ann_best.pt','snn':local/'snn_20261012_102848/bc/bc_v2_snn_best.pt'}
        pool=local/'v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json'
        if not all(p.exists() for p in [*bc.values(),pool]):self.skipTest('read-only source fixtures not mounted')
        with TemporaryDirectory() as root:
            manifest=Path(root)/'source.json';manifest.write_text(__import__('json').dumps(r.identity()),encoding='utf-8')
            args=r.parser().parse_args(['plan','--root',root,'--source-manifest',str(manifest)])
            original=prepare_v2_stage_initialization
            def prepared(*a,**kw):
                self.assertIsInstance(kw['init_checkpoint'],Path)
                return original(*a,**kw)
            with patch.object(r,'BC',{m:(p,r.BC[m][1]) for m,p in bc.items()}),patch.object(r,'POOL',pool), \
                    patch('brain_uav.trainers.v2_formal_training.prepare_v2_stage_initialization',side_effect=prepared) as real_load:
                result=r.preflight(args,cuda=False)
            self.assertEqual(real_load.call_count,2);self.assertEqual(set(result['inputs']),{'ann','snn'})

    def test_initial_gate_and_dispatch_failure_prevents_release(self):
        from brain_uav.scripts import run_v2_online_wd3_acc as r
        from brain_uav.trainers.v2_online_research import atomic_json
        with TemporaryDirectory() as root:
            root=Path(root);jobs=r.matrix()
            for job in jobs:atomic_json(root/job['id']/'review/initialized.json',dict(job=job,initial_state_sha256={'all':job['model']}))
            self.assertEqual(len(r.verify_initializations(root,jobs)),16)
            atomic_json(root/jobs[-1]['id']/'review/initialized.json',dict(job=jobs[-1],initial_state_sha256={'all':'wrong'}))
            with self.assertRaises(ValueError):r.verify_initializations(root,jobs)

    def test_real_shared_compute_matches_eager_all_three_modes_cpu(self):
        from brain_uav.trainers.v2_gpu_compute import GPUCompute
        from brain_uav.trainers.v2_online_research import target_config
        import test_v2_gpu_compute as fixture
        for arm,beta in [('wd3_045',None),('fixed_own',None),('acc_001',0.),('acc_010',.7)]:
            for mode,step in [('critic_only',1),('frozen_targets',2),('actor',25002)]:
                with self.subTest(arm=arm,mode=mode):
                    e=fixture.TestGPUCompute().engine();other=fixture.TestGPUCompute().engine()
                    for engine in (e,other):
                        engine.actor_freeze_steps=25000
                        engine.bc_reference_actor=deepcopy(engine.actor).eval().requires_grad_(False)
                        engine.configure_research_targets(target_config(arm))
                        if beta is not None:engine.set_research_beta(beta)
                    other.load_checkpoint_state_dict(deepcopy(e.checkpoint_state_dict()))
                    batch=e.replay.sample(2)
                    u=GPUCompute.__new__(GPUCompute);u.e=other;u.diagnostic=True;u.compute_diagnostic=False;u.tensor_evidence={}
                    u.actor_params=list(other.actor.parameters());u.critic_params=list(other.critic1.parameters())+list(other.critic2.parameters())
                    u.initialize_actor_loss_scalars()
                    class CPUAdam:
                        def __init__(self,opt):self.opt=opt
                        def step(self,error):
                            if not bool(error):self.opt.step()
                    u.adams=(CPUAdam(other.actor_optimizer),CPUAdam(other.critic_optimizer))
                    u.injections=torch.zeros(4);u.noise=torch.full_like(batch.action,.3)
                    u.coefficients=torch.tensor([5.,0.,other.policy_noise,other.noise_clip,other.research_beta])
                    u.failure_fraction=torch.tensor(batch.sample_failure_fraction)
                    expected={}
                    with patch.object(e.replay,'sample',return_value=batch),patch('torch.randn_like',new=lambda value:torch.full_like(value,.3)):
                        metric=e.update_once(total_steps=step,bc_lambda=5.,diagnostic_tensor_recorder=lambda n,t:expected.update({n:t.detach().clone()}))
                    output=u.compute(batch,mode)
                    self.assertEqual(int(output[0]),0)
                    self.assertAlmostEqual(metric.critic_loss,float(output[1]),places=5)
                    for name in ('target.td','target.noise','target.action'):
                        torch.testing.assert_close(expected[name],u.tensor_evidence[mode][name],rtol=1e-4,atol=1e-5)
                    self.assertEqual(u.tensor_evidence[mode]['target.td'].shape,(2,2))
                    for name in ('actor','critic1','critic2','actor_target','critic1_target','critic2_target'):
                        for a,b in zip(getattr(e,name).parameters(),getattr(other,name).parameters()):
                            torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
                    # Joint Adam moments and parameter ordering remain identical.
                    for left,right in zip(e.critic_optimizer.state.values(),other.critic_optimizer.state.values()):
                        for key in left:torch.testing.assert_close(left[key],right[key],rtol=1e-4,atol=1e-5)

    def test_controller_bank_long_episode_bounds_and_no_loss_g(self):
        from brain_uav.trainers.v2_online_research import ACCController,OnlineResearch
        c=ACCController(.1,capacity=2)
        episode=[dict(observation='s',action=np.array([i,.2]),reward=1.,step=25000+i,done=i==3) for i in range(4)]
        c.complete_episode(episode,25003,'training')
        self.assertEqual(sum(map(len,c.episodes)),4)
        episode[0]['action'][0]=999
        self.assertEqual(c.episodes[0][0]['action'][0],0)
        r=c.adjust(25003,lambda rows:np.full((len(rows),2),-100.))
        self.assertEqual(r['age_min'],0);self.assertEqual(r['age_max'],3)
        for step in range(26003,36003,1000):c.adjust(step,lambda rows:np.full((len(rows),2),-100.))
        self.assertEqual(c.beta,1.);self.assertIsNotNone(c.first_boundary)
        with self.assertRaises(ValueError):c.complete_episode(episode[:-1],30000,'training')
        with TemporaryDirectory() as directory:
            for arm in ('min','mean','wd3_015','fixed_own'):
                hook=OnlineResearch(arm,directory)
                with patch('brain_uav.trainers.v2_online_research.predict_bank',side_effect=AssertionError('no calibration')):
                    hook.episode(None,None)
                self.assertIsNone(hook.acc)

    def test_dispatch_all_or_none_training_release_and_custom_socket(self):
        import json,subprocess
        from types import SimpleNamespace
        from brain_uav.scripts import run_v2_online_wd3_acc as r
        from brain_uav.trainers.v2_online_research import atomic_json
        for fail in (False,True):
            with TemporaryDirectory() as temp:
                root=Path(temp)/'batch';args=r.parser().parse_args(['launch','--root',str(root),'--socket','private','--launch'])
                calls=[]
                def tmux(args,*cmd,check=True):
                    calls.append(cmd)
                    if cmd[0]=='has-session':return SimpleNamespace(returncode=1)
                    count=sum(c[0] in ('new-session','new-window') for c in calls)
                    if fail and count==5:raise subprocess.CalledProcessError(1,cmd)
                    job=r.matrix()[count-1]
                    atomic_json(root/job['id']/'review/initialized.json',dict(job=job,pid=count,initial_state_sha256={'all':job['model']}))
                    return SimpleNamespace(returncode=0)
                with patch.object(r,'preflight',return_value={}),patch.object(r,'tmux',side_effect=tmux),patch.object(r.os,'kill',return_value=None):
                    if fail:
                        with self.assertRaises(subprocess.CalledProcessError):r.launch(args)
                        self.assertFalse((root/'release.json').exists())
                        abort=json.loads((root/'abort.json').read_text());self.assertEqual(len(abort['started']),4)
                    else:
                        r.launch(args);self.assertTrue((root/'release.json').exists())
                        self.assertEqual(sum(c[0]=='new-session' for c in calls),1)
                        self.assertEqual(sum(c[0]=='new-window' for c in calls),15)
                self.assertFalse(any('send-keys' in c for c in calls))

    def test_evaluation_metrics_and_training_state_guard(self):
        from brain_uav.scripts.diagnose_v2_online_research import grouped_statistics
        from brain_uav.trainers.v2_online_research import training_identity
        rows=[dict(scenario_id='a',G=2.,done=False,empty_scene=True,outcome='goal',remaining_steps=2),
              dict(scenario_id='a',G=0.,done=True,empty_scene=True,outcome='goal',remaining_steps=1),
              dict(scenario_id='b',G=0.,done=True,empty_scene=False,outcome='collision',remaining_steps=1)]
        result=grouped_statistics(rows,np.array([[4.,2.],[0.,0.],[6.,-2.]]),{'a':{'shape_types':[]},'b':{'shape_types':['sphere']}})
        self.assertEqual(result['groups']['all']['transition_weighted']['Q1']['bias'],8/3)
        self.assertEqual(result['groups']['all']['scene_equal']['Q1']['bias'],3.5)
        self.assertEqual(result['groups']['failure_terminal']['transition_weighted']['Q1']['bias'],6.)
        engine=f._engine(ScenarioConfig());before=training_identity(engine)
        self.assertEqual(training_identity(engine),before)
        engine.replay.rng.random()
        self.assertNotEqual(training_identity(engine),before)

    def test_real_compiled_shared_blocks_use_distinct_labels_and_runtime_beta(self):
        from brain_uav.trainers.v2_online_research import target_config
        import test_v2_gpu_compute as fixture
        eager=fixture.TestGPUCompute().engine();compiled=fixture.TestGPUCompute().engine()
        for engine in (eager,compiled):engine.configure_research_targets(target_config('acc_001'))
        compiled.load_checkpoint_state_dict(deepcopy(eager.checkpoint_state_dict()))
        batch=eager.replay.sample(2)
        with patch('torch.compile',side_effect=lambda function,**kwargs:function):
            compiled.enable_critic_loss_compile();compiled.enable_target_block_compile()
        for beta,step in ((0.,1),(.5,2),(1.,4)):
            records=[]
            for engine in (eager,compiled):
                engine.set_research_beta(beta);tensors={}
                with patch.object(engine.replay,'sample',return_value=batch),patch('torch.randn_like',new=lambda x:torch.full_like(x,.3)):
                    metric=engine.update_once(total_steps=step,diagnostic_tensor_recorder=lambda n,t:tensors.update({n:t.detach().clone()}))
                records.append((metric,tensors))
            self.assertAlmostEqual(records[0][0].critic_loss,records[1][0].critic_loss,places=5)
            torch.testing.assert_close(records[0][1]['target.td'],records[1][1]['target.td'],rtol=1e-4,atol=1e-5)
            for a,b in zip(eager.critic1.parameters(),compiled.critic1.parameters()):torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)

    def test_worker_actual_setup_snapshot_evaluation_and_status_mock(self):
        from types import SimpleNamespace
        import json
        from brain_uav.scripts import run_v2_online_wd3_acc as r
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig,V2FormalTrainingResult
        from brain_uav.trainers.v2_online_research import atomic_json
        scenario=ScenarioConfig();engine=f._engine(scenario)
        cfg=V2FormalTrainingConfig(stage='easy',max_steps=500000,batch_size=2)
        prepared=SimpleNamespace(scenario_config=scenario,reward_config=r.research_rewards(),uav_collision_radius=0.)
        components=SimpleNamespace(engine=engine,scenario_generators={},selector=None,exploration_rng=None,
            seed_manifest={'base_seed':23},initialization_source={'kind':'mock_only'})
        observed={}
        class Trainer:
            def __init__(self,*a,**kw):observed.update(kw)
            def run(self):
                observed['weight_snapshot_sink'](75000,75001)
                result=V2FormalTrainingResult.empty('easy',global_steps_start=0);result.stage_steps=400000
                return result
        with TemporaryDirectory() as directory:
            root=Path(directory);bc=root/'best.pt';bc.write_bytes(b'mock');pool=root/'pool.json';pool.write_bytes(b'pool')
            job='ann_wd3_045';(root/job/'review').mkdir(parents=True)
            atomic_json(root/'manifest.json',{'preflight':{'source':r.identity(),
                'inputs':{'ann':{'pool_file_sha256':__import__('hashlib').sha256(pool.read_bytes()).hexdigest()}}}})
            atomic_json(root/'release.json',{'mock_only':True})
            args=r.parser().parse_args(['worker','--root',str(root),'--job',job,'--launch'])
            with patch.object(r,'BC',{'ann':(bc,__import__('hashlib').sha256(bc.read_bytes()).hexdigest())}), \
                    patch.object(r,'POOL',pool), \
                    patch.object(r,'make_config',return_value=cfg), \
                    patch('brain_uav.trainers.v2_formal_training.prepare_v2_stage_initialization',return_value=prepared), \
                    patch('brain_uav.trainers.v2_formal_training.build_v2_stage_engine',return_value=components) as build, \
                    patch('brain_uav.trainers.v2_validation.load_v2_validation_pool',return_value=SimpleNamespace(content_digest=r.DIGEST,scenarios=[])), \
                    patch('brain_uav.scripts.train_v2_td3._configure_stage_compilation',return_value={'mock_only':True}) as compile_, \
                    patch('brain_uav.trainers.v2_reporting.V2ExperimentReporter') as reporter, \
                    patch('brain_uav.trainers.v2_formal_training.V2FormalStageTrainer',Trainer), \
                    patch.object(engine,'local_optimization_metadata',return_value={'gpu_td3_compute':{'effective':True,'replays':1}}), \
                    patch.object(r.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as evaluate:
                # identity uses subprocess too; keep actual source evidence narrowly fixed here.
                with patch.object(r,'identity',return_value=json.loads((root/'manifest.json').read_text())['preflight']['source']):r.worker(args)
            self.assertTrue(build.call_args.kwargs['gpu_td3_compute'])
            self.assertFalse(build.call_args.kwargs['non_blocking_batch_transfer'])
            self.assertTrue(compile_.call_args.kwargs['compile_actor_loss'])
            self.assertTrue(reporter.return_value.start_stage.called)
            self.assertEqual(evaluate.call_count,2)
            self.assertEqual(observed['bc_schedule'],'smooth-mid-strong')
            result=json.loads((root/job/'review/summary.json').read_text())
            self.assertTrue(result['execution_completed']);self.assertFalse(result['formal_passed'])
            self.assertEqual(json.loads((root/job/'review/progress.json').read_text())['state'],'completed_budget')

    def test_acc_tail_never_becomes_complete_g_bank(self):
        from brain_uav.trainers.v2_online_research import OnlineResearch,target_config
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig,V2FormalStageTrainer
        scenario=ScenarioConfig(max_steps=2);engine=f._engine(scenario)
        engine.configure_research_targets(target_config('acc_001'))
        cfg=V2FormalTrainingConfig(stage='easy',max_steps=500000,batch_size=2)
        source=f._Source(f._scenario_payload(0))
        with TemporaryDirectory() as root:
            hook=OnlineResearch('acc_001',root);hook.pause_step=200000;hook.hard_limit=5
            trainer=V2FormalStageTrainer(scenario,RewardConfig(),cfg,engine,scenario_sources={'easy':source},
                validation_runner=lambda a:self.fail('validation'),online_research=hook)
            with patch.object(engine,'update_once',return_value=td.V2TD3UpdateMetrics()):result=trainer.run()
            self.assertEqual(result.stage_steps,5);self.assertEqual(len(result.episodes),2)
            self.assertEqual(len(hook.acc.episodes),2);self.assertEqual(sum(map(len,hook.acc.episodes)),4)
            self.assertEqual(hook.acc.controls,[])
            self.assertTrue(all(e[-1]['done'] and not e[0]['done'] for e in hook.acc.episodes))
            self.assertFalse(hasattr(engine.replay,'mc_return') and engine.replay.mc_return is not None)

    def test_small_real_environment_evaluation_and_changed_input_failure(self):
        import json
        from types import SimpleNamespace
        from brain_uav.scripts import diagnose_v2_online_research as e,run_v2_online_wd3_acc as r
        import test_diagnose_v2_wd3_acc as fixture
        for changed in (False,True):
            data,_=fixture.fixture();data['model']='ann'
            scenes=[dict(scenario_id=str(i),sequence_index=i,payload=fixture.fixtures._scenario([0,0,10,0,0],[60,0,10],[])) for i in range(2)]
            with TemporaryDirectory() as root:
                root=Path(root);checkpoint=root/'snapshot';checkpoint.write_bytes(b'fixture')
                pool=root/'pool';pool.write_bytes(b'pool')
                data.update(checkpoint_hash=e.file_sha256(checkpoint),pool_hash=e.file_sha256(pool))
                data['pool']=SimpleNamespace(uav_collision_radius=0.,content_digest=r.DIGEST)
                args=SimpleNamespace(checkpoint=checkpoint,root=root/'eval',mode='evaluate')
                original=e.cal.base.fly
                def fly(*a,**kw):
                    self.assertTrue((args.root/'manifest.json').exists())
                    result=original(*a,**kw)
                    if changed:checkpoint.write_bytes(b'replaced')
                    return result
                with patch.object(r,'POOL',pool),patch.object(e.cal.base,'load_models',return_value=data), \
                        patch.object(e.cal.base,'select_scenarios',return_value=scenes),patch.object(e.cal.base,'fly',side_effect=fly):
                    if changed:
                        with self.assertRaisesRegex(RuntimeError,'changed'):e.run(args)
                    else:
                        e.run(args)
                        rows=[json.loads(x) for x in (args.root/'transitions.jsonl').read_text().splitlines()]
                        self.assertEqual(len(rows),4);self.assertTrue(all('reward' in x and 'G' in x and 'action' in x for x in rows))
                summary=json.loads((args.root/'summary.json').read_text())
                self.assertEqual(summary['execution_completed'],not changed)

    def test_common_probe_candidates_and_real_prefix_replay_small_cpu(self):
        import json
        from types import SimpleNamespace
        from brain_uav.scripts import diagnose_v2_online_research as e,run_v2_online_wd3_acc as r
        import test_diagnose_v2_wd3_acc as fixture
        data,_=fixture.fixture();data['model']='ann'
        data['pool'].content_digest=r.DIGEST
        scenes=[dict(scenario_id=str(i),sequence_index=i,payload=fixture.fixtures._scenario([0,0,10,0,0],[60,0,10],[])) for i in range(12)]
        with TemporaryDirectory() as root:
            root=Path(root);checkpoint=root/'source';checkpoint.write_bytes(b'mock');pool=root/'pool';pool.write_bytes(b'pool')
            data.update(checkpoint_hash=e.file_sha256(checkpoint),pool_hash=e.file_sha256(pool))
            args=SimpleNamespace(checkpoint=checkpoint,root=root/'probe',mode='probe')
            prepared=SimpleNamespace(bc_initialization=SimpleNamespace(actor=deepcopy(data['actor'])))
            with patch.object(r,'POOL',pool),patch.object(r,'BC',{'ann':(checkpoint,e.file_sha256(checkpoint))}), \
                    patch.object(e.cal.base,'load_models',return_value=data),patch.object(e.cal.base,'select_scenarios',return_value=scenes), \
                    patch('brain_uav.trainers.v2_formal_training.prepare_v2_stage_initialization',return_value=prepared):e.run(args)
            plans=json.loads((args.root/'probe_plan.json').read_text());result=json.loads((args.root/'probes.json').read_text())
            self.assertEqual(len(plans),12);self.assertTrue(all(len(v['prefix'])==0 for v in plans))
            self.assertEqual(len(result),12)
            for value in result:
                self.assertEqual(len(value['candidates']),5)
                self.assertIn('Q2',value['rankings'])
                self.assertTrue(all(len(c['branch']['rewards'])==2 for c in value['candidates']))


    def test_source_identity_normalizes_line_endings_but_detects_code_changes(self):
        from types import SimpleNamespace
        from brain_uav.scripts import run_v2_online_wd3_acc as r
        with TemporaryDirectory() as directory:
            repo=Path(directory);(repo/'src').mkdir();(repo/'scripts').mkdir()
            source=repo/'src/example.py';launcher=repo/'scripts/online_v2_wd3_acc.sh'
            source.write_bytes(b'value = 1\n');launcher.write_bytes(b'#!/bin/bash\necho ok\n')
            with patch.object(r,'REPO',repo),patch.object(r.subprocess,'run',return_value=SimpleNamespace(stdout='fixture')):
                lf=r.identity()['sha256']
                source.write_bytes(b'value = 1\r\n');launcher.write_bytes(b'#!/bin/bash\r\necho ok\r\n')
                self.assertEqual(r.identity()['sha256'],lf)
                source.write_bytes(b'value = 2\r\n')
                self.assertNotEqual(r.identity()['sha256']['src/example.py'],lf['src/example.py'])
                launcher.write_bytes(b'#!/bin/bash\r\necho changed\r\n')
                self.assertNotEqual(r.identity()['sha256']['scripts/online_v2_wd3_acc.sh'],lf['scripts/online_v2_wd3_acc.sh'])

    def test_snn_actor_real_evidence_completion_and_strict_comparison(self):
        from brain_uav.scripts import diagnose_v2_online_research as d
        from brain_uav.scripts import diagnose_v2_mc_return_cuda as helper
        from brain_uav.trainers.v2_gpu_compute import GPUCompute
        from test_v2_spike_regularization import TestSpikeRegularization
        eager=TestSpikeRegularization().engine(delay=2)
        compute=TestSpikeRegularization().engine(delay=2)
        compute.load_checkpoint_state_dict(deepcopy(eager.checkpoint_state_dict()))
        batch=eager.replay.sample(2)
        recorded={'replay.mask':batch.obs.presence_mask.clone(),'replay.next_mask':batch.next_obs.presence_mask.clone()}
        record=lambda n,t:recorded.update({n:t.detach().clone()})
        with patch.object(eager.replay,'sample',return_value=batch),patch('torch.randn_like',new=torch.zeros_like),helper.eager_evidence(eager,record):
            metric=eager.update_once(total_steps=75002,diagnostic_tensor_recorder=record)
        self.assertTrue(metric.actor_updated)
        self.assertNotIn('actor.weighted_spike',recorded)
        original=deepcopy(recorded)
        u=GPUCompute.__new__(GPUCompute);u.e=compute;u.diagnostic=True;u.compute_diagnostic=True;u.tensor_evidence={}
        u.actor_params=list(compute.actor.parameters());u.critic_params=list(compute.critic1.parameters())+list(compute.critic2.parameters())
        u.initialize_actor_loss_scalars()
        class CPUAdam:
            def __init__(self,opt):self.opt=opt
            def step(self,error):
                if not bool(error):self.opt.step()
        u.adams=(CPUAdam(compute.actor_optimizer),CPUAdam(compute.critic_optimizer))
        u.injections=torch.zeros(4);u.noise=torch.zeros_like(batch.action)
        u.coefficients=torch.tensor([0.,1.,compute.policy_noise,compute.noise_clip])
        u.failure_fraction=torch.tensor(batch.sample_failure_fraction)
        u.compute(batch,'actor')
        actual=u.tensor_evidence['actor']
        self.assertIn('actor.weighted_spike',actual)
        left={};right={}
        d._append_update_evidence(left,recorded,'snn');d._append_update_evidence(right,actual,'snn')
        for name,value in original.items():self.assertTrue(torch.equal(recorded[name],value),name)
        result=helper.compare_snapshots(left,right)
        self.assertTrue(result['passed'],result)
        right['tensor.actor.weighted_spike']=right['tensor.actor.weighted_spike']+1
        self.assertFalse(helper.compare_snapshots(left,right)['passed'])
        ann={'actor.action':torch.ones(2)};view={}
        d._append_update_evidence(view,ann,'ann')
        self.assertEqual(set(ann),{'actor.action'})

    def test_probe_wrong_pool_digest_rejected_before_flight(self):
        import json
        from types import SimpleNamespace
        from brain_uav.scripts import diagnose_v2_online_research as d
        with TemporaryDirectory() as root:
            args=SimpleNamespace(checkpoint=Path(root)/'source',root=Path(root)/'output',mode='probe')
            with patch.object(d.cal.base,'load_models',return_value={'pool':SimpleNamespace(content_digest='wrong')}), \
                    patch.object(d.cal.base,'fly') as fly,patch.object(d.cal.base,'select_scenarios') as select:
                with self.assertRaisesRegex(ValueError,'digest'):d.run(args)
                fly.assert_not_called();select.assert_not_called()
            self.assertFalse(json.loads((args.root/'summary.json').read_text())['execution_completed'])

    def test_acc_snapshot_bank_returns_and_chronology(self):
        from brain_uav.trainers.v2_online_research import ACCController,FORMAT,target_config,validate_snapshot
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig,build_v2_periodic_snapshot
        from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_checkpoint
        from brain_uav.v2_curriculum import derive_v2_component_seed
        scenario=ScenarioConfig();engine=f._engine(scenario);engine.configure_research_targets(target_config('acc_001'))
        config=V2FormalTrainingConfig(stage='easy',max_steps=500000,batch_size=2)
        seeds={'base_seed':config.seed,**{f'{k}_seed':derive_v2_component_seed(config.seed,'easy',k) for k in ('model','replay','curriculum','exploration')}}
        saved=build_v2_periodic_snapshot(engine,config,stage_steps=75000,scenario=scenario,rewards=RewardConfig(),
            uav_collision_radius=0.,seed_manifest=seeds,initialization_source={'kind':'test'},bc_schedule='smooth-mid-strong')
        saved['research_only']=True
        controller=ACCController(.01)
        def episode(first,length):
            rows=[dict(observation='fixture',action=np.zeros(2,dtype=np.float32),reward=1.,step=first+i,done=i==length-1) for i in range(length)]
            controller.complete_episode(rows,first+length-1,'training')
        episode(1,2);episode(3,2)
        raw=dict(format=FORMAT,format_version=1,diagnostic_only=True,formal_passed=False,arm='acc_001',
            target_config=target_config('acc_001'),beta=.5,controller=controller.state(),requested_step=75000,actual_step=75000,snapshot=saved)
        with TemporaryDirectory() as root:
            path=Path(root)/'acc.pt';torch.save(raw,path)
            self.assertEqual(load_diagnostic_checkpoint(path)['diagnostic_research']['arm'],'acc_001')
        for change in ('G','completed','future','gap','noninteger','order'):
            bad=deepcopy(raw);episodes=bad['controller']['episodes']
            if change=='G':episodes[0][-1]['G']=999.
            elif change=='completed':episodes[0][0]['completed_step']=1
            elif change=='future':
                for row in episodes[0]:row['completed_step']=500000
            elif change=='gap':episodes[0][-1]['step']=3
            elif change=='noninteger':episodes[0][0]['step']=0.5
            else:episodes.reverse()
            with self.subTest(change=change),self.assertRaises(ValueError):validate_snapshot(bad)
        # Eviction preserves whole episodes, including an oversized newest episode.
        episode(100,5001)
        self.assertEqual(len(controller.episodes),1)
        raw['controller']=controller.state();validate_snapshot(raw)
        episode(6000,2)
        self.assertEqual(len(controller.episodes),1)
        raw['controller']=controller.state();validate_snapshot(raw)


if __name__=='__main__': unittest.main()
