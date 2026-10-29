"""CPU checks only; no CUDA acceptance or real fitting."""
from copy import deepcopy
import unittest
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock
from contextlib import redirect_stdout
from io import StringIO
import json
import numpy as np
import test_v2_formal_training as fixtures
import test_run_v2_td3_curriculum as curriculum_fixtures
import test_diagnose_v2_critic_calibration as calibration_fixtures
import test_v2_gpu_compute as gpu_fixtures
from brain_uav.scripts import diagnose_v2_target_aggregation as d
from brain_uav.scripts import train_v2_td3 as single, run_v2_td3_curriculum as course
from brain_uav.trainers import v2_formal_training as formal
from brain_uav.trainers.v2_gpu_compute import GPUCompute
from brain_uav.trainers.v2_gpu_update import GPUUpdate
from brain_uav.config import ScenarioConfig, RewardConfig
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig


class TestTargetAggregation(unittest.TestCase):
    def test_mean_eager_matches_real_gpu_shared_compute_on_cpu(self):
        original = gpu_fixtures.Fixture.make_engine
        def mean_engine(fixture, *args, **kwargs):
            e = original(fixture, *args, **kwargs)
            e.target_q_aggregation = 'mean'
            return e
        with mock.patch.object(gpu_fixtures.Fixture, 'make_engine', new=mean_engine):
            gpu_fixtures.TestGPUCompute().test_real_common_compute_cpu_same_shapes_loss_gradient_and_update()

    def test_compiled_shared_target_respects_mode(self):
        e=gpu_fixtures.TestGPUCompute().engine()
        batch=e.replay.sample(2)
        executor=GPUCompute.__new__(GPUCompute);executor.e=e
        args=executor.arguments(batch.next_obs)
        compiled=torch.compile(e._compute_target_critics_td_tensors,backend='eager',fullgraph=True)
        for mode in ('min','mean','min'):
            e.target_q_aggregation=mode
            with torch.no_grad():
                raw=e._compute_target_critics_td_tensors(*args,batch.action,batch.reward,batch.done)
                result=compiled(*args,batch.action,batch.reward,batch.done)
            for a,b in zip(raw,result): self.assertTrue(torch.equal(a,b))

    def test_formula_and_validation(self):
        engine = object.__new__(V2TD3UpdateEngine)
        q1, q2 = torch.tensor([2., 8.]), torch.tensor([6., 4.])
        for mode, expected in [('min', [2., 4.]), ('mean', [4., 6.])]:
            engine.target_q_aggregation = mode
            value = engine._aggregate_target_q(q1, q2)
            self.assertTrue(torch.equal(value, torch.tensor(expected)))
            self.assertTrue(torch.equal(torch.tensor([3., 3.]) + .5*(1-torch.tensor([1., 0.]))*value,
                                        torch.tensor([3., 3.+.5*expected[1]])))
        self.assertEqual(V2FormalTrainingConfig(stage='easy').target_q_aggregation, 'min')
        for invalid in ('max', '', None, True):
            with self.assertRaises(ValueError):
                V2FormalTrainingConfig(stage='easy', target_q_aggregation=invalid)

    def test_shared_gpu_target_static_binding_and_engine_checkpoint(self):
        e = fixtures._engine(ScenarioConfig())
        q1, q2 = torch.tensor([[2.],[8.]]), torch.tensor([[6.],[4.]])
        args = [torch.zeros(1)]*7
        for mode, expected in [('min',[2.,4.]),('mean',[4.,6.])]:
            e.target_q_aggregation = mode
            with mock.patch.object(e.critic1_target,'_compute_full_forward_tensors',return_value=q1), \
                 mock.patch.object(e.critic2_target,'_compute_full_forward_tensors',return_value=q2):
                result = e._compute_target_critics_td_tensors(*args, torch.zeros(2,2), torch.ones(2,1),torch.tensor([[1.],[0.]]))[-1]
            self.assertTrue(torch.equal(result,torch.tensor([[1.],[1.+e.gamma*expected[1]]])))
            for executor in (GPUUpdate,GPUCompute):
                before = executor.static_options(e)
                e.target_q_aggregation = 'mean' if mode=='min' else 'min'
                self.assertNotEqual(before,executor.static_options(e))
                e.target_q_aggregation = mode
            saved = e.checkpoint_state_dict()
            e.load_checkpoint_state_dict(saved)
            bad = deepcopy(saved); bad['algorithm_config']['target_q_aggregation']='max'
            with self.assertRaisesRegex(ValueError,'algorithm_config'): e.load_checkpoint_state_dict(bad)
        old = deepcopy(saved); old['algorithm_config'].pop('target_q_aggregation')
        with self.assertRaisesRegex(ValueError,'algorithm_config'): e.load_checkpoint_state_dict(old)
        e.target_q_aggregation='min'; e.load_checkpoint_state_dict(old)

    def test_formal_snapshot_legacy_and_tamper(self):
        scenario = ScenarioConfig()
        e = fixtures._engine(scenario)
        self._check_snapshot(e, scenario)

    def test_snn_formal_snapshot_legacy_and_tamper(self):
        import test_v2_spike_regularization as snn_fixtures
        self._check_snapshot(snn_fixtures.TestSpikeRegularization().engine(), ScenarioConfig())

    def _check_snapshot(self, e, scenario):
        for mode in ('min','mean'):
            e.target_q_aggregation=mode
            cfg=V2FormalTrainingConfig(stage='easy',target_q_aggregation=mode)
            result=formal.V2FormalTrainingResult.empty('easy',global_steps_start=0)
            result.status='failed'; result.stop_reason='max_steps'
            common=dict(scenario=scenario,rewards=RewardConfig(),uav_collision_radius=0.,
                seed_manifest={'base_seed':7},initialization_source={'kind':'v2_bc_best','path':'bc.pt'})
            if e.model_type == 'snn': common['initialization_source']['model_type']='snn'
            p=formal.build_v2_formal_checkpoint(e,result,cfg,validation_pool_metadata=dict(path='pool.json',
                format_version=1,curriculum_level='easy',master_seed=20260904,stage_seed=11,
                scenario_count=100,content_digest='digest'),**common)
            snapshot=formal.build_v2_periodic_snapshot(e,cfg,stage_steps=200123,**common)
            for value,save,load in ((p,formal.save_v2_formal_checkpoint,formal.load_v2_formal_checkpoint),
                                     (snapshot,formal.save_v2_periodic_snapshot,formal.load_v2_periodic_snapshot)):
                with TemporaryDirectory() as tmp:
                    path=Path(tmp)/'state.pt'; save(path,value)
                    restored=load(path)
                    self.assertEqual(restored['formal_config']['target_q_aggregation'],mode)
                    # Actual shared diagnostic dispatch accepts terminal AND snapshot.
                    from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_checkpoint
                    self.assertEqual(load_diagnostic_checkpoint(path)['formal_config']['target_q_aggregation'],mode)
                for field in ('formal_config','engine_checkpoint'):
                    bad=deepcopy(value)
                    config=bad[field] if field=='formal_config' else bad[field]['algorithm_config']
                    config['target_q_aggregation']='mean' if mode=='min' else 'min'
                    with self.assertRaisesRegex(ValueError,'target_q_aggregation'): load(bad)
                    config['target_q_aggregation']='max'
                    with self.assertRaisesRegex(ValueError,'target_q_aggregation'): load(bad)
                legacy=deepcopy(value)
                legacy['formal_config'].pop('target_q_aggregation')
                legacy['engine_checkpoint']['algorithm_config'].pop('target_q_aggregation')
                self.assertEqual(load(legacy)['formal_config']['target_q_aggregation'],'min')

    def test_cli_and_real_curriculum_forwarding(self):
        for mode in ('min','mean'):
            argv=['--stage','easy','--init-checkpoint','bc.pt','--output','o.pt',
                '--metrics-out','m.json','--validation-pool','pool.json','--device','cpu',
                '--target-q-aggregation',mode]
            with redirect_stdout(StringIO()),mock.patch.object(single,'run_v2_td3_stage',return_value={'passed':True}) as run:
                single.main(argv)
            self.assertEqual(run.call_args.kwargs['target_q_aggregation'],mode)
            with TemporaryDirectory() as tmp:
                root=Path(tmp); bc=root/'bc.pt';bc.write_bytes(b'fixture')
                prepared=curriculum_fixtures._prepared_initialization(ScenarioConfig(),source=bc)
                calls=[]
                def stage(**kw):
                    calls.append(kw)
                    return dict(stage=kw['stage'],passed=True,checkpoint=str(kw['output']),steps=1,global_steps_end=1)
                with mock.patch.object(course,'prepare_v2_stage_initialization',return_value=prepared), \
                     mock.patch.object(course,'prepare_v2_validation_pools',return_value={s:root/f'{s}.json' for s in ('easy','medium','hard')}), \
                     mock.patch.object(course,'load_v2_validation_pool',return_value=SimpleNamespace(master_seed=20260904,
                         stage_seed=1,scenario_count=100,content_digest='digest')):
                    course.run_v2_curriculum(bc_checkpoint=bc,output_root=root/'run',validation_pool_dir=root/'pool',
                        target_q_aggregation=mode,device='cpu',stage_runner=stage)
                self.assertEqual([c['target_q_aggregation'] for c in calls],[mode]*3)
        with mock.patch.object(course,'run_v2_curriculum',return_value={'passed':True}) as runner,redirect_stdout(StringIO()):
            # CLI wrapper uses the same parameter; return shape is only printed.
            course.main(['--bc-checkpoint','bc.pt','--output-root','out','--validation-pool-dir','pool',
                         '--target-q-aggregation','mean','--device','cpu'])
        self.assertEqual(runner.call_args.kwargs['target_q_aggregation'],'mean')

    def test_documented_commands_compile_and_diagnostic_cli(self):
        import re, ast, inspect
        text=(Path(__file__).resolve().parents[1]/'docs/v2_target_q_aggregation.md').read_text(encoding='utf-8')
        snippets=re.findall(r"<<'PY'[^\n]*\n(.*?)^PY$",text,re.MULTILINE|re.DOTALL)
        self.assertEqual(len(snippets),2)
        for snippet in snippets:
            tree=ast.parse(snippet)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='run_v2_td3_stage':
                    for keyword in node.keywords:
                        if keyword.arg is not None:
                            self.assertIn(keyword.arg,inspect.signature(single.run_v2_td3_stage).parameters)
        for mode, extra in (('prepare',['--history-dir','history']),
                            ('fit',['--prepared-dir','data','--aggregation','min']),
                            ('fit',['--prepared-dir','data','--aggregation','mean']),
                            ('evaluate',['--history-dir','history'])):
            with mock.patch.object(d,'run') as run:
                d.main(['--mode',mode,'--checkpoint','b.pt','--validation-pool','pool.json',
                        '--output-dir','out','--device','cuda',*extra])
                self.assertEqual(run.call_args.args[0].mode,mode)

    def test_documented_formal_bc_hash_precheck_stops_before_training(self):
        import re
        from brain_uav.scripts import diagnose_v2_policy as policy
        from brain_uav.trainers import v2_validation as validation
        text=(Path(__file__).resolve().parents[1]/'docs/v2_target_q_aggregation.md').read_text(encoding='utf-8')
        snippet=re.findall(r"<<'PY'[^\n]*\n(.*?)^PY$",text,re.MULTILINE|re.DOTALL)[0]
        expected='7b2b266fdf32cdb8b4eb220ef26a45008717a0af1538f8d478019f37ddc9adbc'
        for actual in (expected,'wrong hash'):
            output=StringIO(); order=[]
            with mock.patch('sys.argv',['-','--seed','23','--target-q-aggregation','mean','--root','unused']), \
                 mock.patch.object(policy,'file_sha256',side_effect=lambda p: order.append('hash') or actual) as hash_file, \
                 mock.patch.object(validation,'load_v2_validation_pool',side_effect=lambda *a,**kw:
                     order.append('pool') or SimpleNamespace(content_digest=d.cal.base.POOL_DIGEST)) as pool, \
                 mock.patch.object(single,'run_v2_td3_stage',side_effect=lambda **kw:
                     order.append('train') or {'passed':True}) as train,redirect_stdout(output):
                if actual==expected:
                    with self.assertRaises(SystemExit) as result: exec(compile(snippet,'documented-formal-command','exec'),{})
                    self.assertEqual(result.exception.code,0)
                    self.assertEqual(order,['hash','pool','train'])
                else:
                    with self.assertRaisesRegex(ValueError,'BC best SHA256 mismatch'):
                        exec(compile(snippet,'documented-formal-command','exec'),{})
                    pool.assert_not_called();train.assert_not_called()
                hash_file.assert_called_once_with(Path('outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt'))
            self.assertIn('expected='+expected,output.getvalue())
            self.assertIn('actual='+actual,output.getvalue())

    def test_documented_running_initial_state_check_uses_stage_start(self):
        import re
        text=(Path(__file__).resolve().parents[1]/'docs/v2_target_q_aggregation.md').read_text(encoding='utf-8')
        self.assertIn('23 /absolute/min/other/metrics_reports/stage_start.json /absolute/mean/other/metrics_reports/stage_start.json',text)
        snippet=re.findall(r"<<'PY'[^\n]*\n(.*?)^PY$",text,re.MULTILINE|re.DOTALL)[1]
        fields=('actor','critic1','critic2','actor_target','critic1_target','critic2_target','actor_optimizer','critic_optimizer')
        a=dict(formal_config=dict(seed=23,target_q_aggregation='min'),initial_state_sha256={k:k for k in fields})
        b=deepcopy(a);b['formal_config']['target_q_aggregation']='mean'
        with TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[root/'min'/'other/metrics_reports/stage_start.json',root/'mean'/'other/metrics_reports/stage_start.json']
            for case in ('normal','seed','mode','state','missing'):
                changed=deepcopy(b)
                if case=='seed': changed['formal_config']['seed']=11
                elif case=='mode': changed['formal_config']['target_q_aggregation']='min'
                elif case=='state': changed['initial_state_sha256']['critic1']='different'
                elif case=='missing': changed['initial_state_sha256'].pop('critic1')
                for path,value in zip(paths,(a,changed)):
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
                with mock.patch('sys.argv',['-','23',*[str(p) for p in paths]]),redirect_stdout(StringIO()):
                    if case=='normal': exec(compile(snippet,'documented-initial-check','exec'),{})
                    else:
                        with self.assertRaises(AssertionError): exec(compile(snippet,'documented-initial-check','exec'),{})

    def test_dynamic_arms_frozen_actor_live_targets_and_G_not_in_loss(self):
        fixture=calibration_fixtures.TestCalibration()
        with mock.patch.object(d.cal.base,'actor_action',return_value=np.zeros(2,dtype=np.float32)):
            data,scene,record=fixture.fixture()
        rows=d.cal.replay_episode(data,scene,record)
        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(2))
            def forward(self,obs): return self.weight.expand(len(obs.ego_features),2)
        critics=[calibration_fixtures.TinyCritic(),calibration_fixtures.TinyCritic()]
        opt=torch.optim.Adam([p for c in critics for p in c.parameters()],lr=.001)
        sum(p.square().sum() for c in critics for p in c.parameters()).backward(); opt.step()
        actor=Actor().requires_grad_(False)
        data.update(actor=actor,critics=critics,device='cpu')
        data['checkpoint_payload']['engine_checkpoint']=dict(critic_optimizer_state_dict=deepcopy(opt.state_dict()),
            algorithm_config=dict(critic_grad_clip_norm=1.,tau=.3,policy_delay=2),
            update_counts=dict(last_total_steps=500000),actor_target_state_dict=deepcopy(actor.state_dict()),
            critic1_target_state_dict=deepcopy(critics[0].state_dict()),critic2_target_state_dict=deepcopy(critics[1].state_dict()))
        artifact=dict(rows=rows,batches=np.tile([0,1],(2,32)),split=dict(train=['one'],holdout=[]))
        before=d.cal.state_hash([actor.state_dict(),*[c.state_dict() for c in critics],data['checkpoint_payload']])
        a=d.fit_arm(data,artifact,'min',budget=2,evaluations=(0,2))
        b=d.fit_arm(data,artifact,'mean',budget=2,evaluations=(0,2))
        self.assertEqual(a['initial'],b['initial']);self.assertEqual(a['batch_sha256'],b['batch_sha256'])
        self.assertEqual(a['target_updates'],1);self.assertNotEqual(a['initial']['targets'],a['final_targets'])
        self.assertEqual(before,d.cal.state_hash([actor.state_dict(),*[c.state_dict() for c in critics],data['checkpoint_payload']]))
        changed=deepcopy(artifact)
        for r in changed['rows']: r['G'] += 100000
        c=d.fit_arm(data,changed,'min',budget=2,evaluations=(0,2))
        self.assertEqual(a['final_online'],c['final_online'])
        self.assertEqual(a['final_targets'],c['final_targets'])
        self.assertNotEqual(a['curve'],c['curve'])
        targets=d.cal.load_targets(data)[1:]
        self.assertFalse(torch.equal(d.dynamic_labels(data,targets,rows,[0],'min'),
                                     d.dynamic_labels(data,targets,rows,[0],'mean')))
        for mode in ('min','mean'):
            y=d.dynamic_labels(data,targets,rows,[1],mode)
            self.assertEqual(y.item(),rows[1]['reward'])
        targets[0].linear.weight.data.fill_(float('nan'))
        with self.assertRaises(ValueError): d.dynamic_labels(data,targets,rows,[0],'mean')

    def test_formal_builder_same_seed_initialization_and_mode(self):
        from brain_uav.v2_curriculum import derive_v2_component_seed
        scenario=ScenarioConfig(); source={}
        prepared=formal.V2PreparedStageInitialization(stage='easy',
            model_seed=derive_v2_component_seed(7,'easy','model'),verified_initialization_source=source,
            snn_time_window=None,scenario_config=scenario,reward_config=RewardConfig(),
            uav_collision_radius=0.,model_type='ann',
            bc_initialization=formal.V2BCFormalInitialization(actor=fixtures._engine(scenario).actor,
                scenario_config=scenario,uav_collision_radius=0.,model_type='ann'),
            formal_checkpoint=None,torch_rng_state=torch.get_rng_state().clone())
        summaries=[]
        for mode in ('min','mean'):
            cfg=V2FormalTrainingConfig(stage='easy',batch_size=2,replay_capacity=16,target_q_aggregation=mode)
            e=formal.build_v2_stage_engine(scenario,cfg,init_checkpoint=source,prepared_initialization=prepared).engine
            self.assertEqual(e.target_q_aggregation,mode)
            summaries.append(formal.v2_initial_state_summary(e))
        self.assertEqual(summaries[0],summaries[1])

    def test_prepared_artifact_hash_leakage_and_run_fit_success(self):
        records=[dict(scenario_id=str(i),empty_scene=i<7,sequence_index=i) for i in range(20)]
        split=d.cal.split_scenes(records)
        with TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'b.pt';checkpoint.write_bytes(b'fixture')
            pool=root/'pool.json';pool.write_text('{}');history=root/'history';history.mkdir()
            (history/'manifest.json').write_text('{}');(history/'baseline_episodes.jsonl').write_text('{}')
            args=SimpleNamespace(mode='prepare',checkpoint=checkpoint,validation_pool=pool,
                history_dir=history,device='cpu',output_dir=root/'prepared')
            data=dict(pool=SimpleNamespace(content_digest=d.cal.base.POOL_DIGEST))
            def replay(data,scene,record):
                self.assertTrue((args.output_dir/'manifest.json').is_file())
                return [dict(scenario_id=record['scenario_id'],empty_scene=record['empty_scene'],G=3.,done=True)]
            with mock.patch.object(d,'source_data',return_value=data),mock.patch.object(d,'history_selection',
                return_value=(records,split,{r['scenario_id']:{} for r in records})),mock.patch.object(d.cal,'replay_episode',side_effect=replay):
                self.assertTrue(d.run(args)['execution_completed'])
            fit_args=SimpleNamespace(mode='fit',checkpoint=checkpoint,validation_pool=pool,
                prepared_dir=args.output_dir,device='cpu',output_dir=root/'arm',aggregation='mean')
            artifact,evidence=d.load_prepared(fit_args)
            self.assertFalse(any(artifact['rows'][int(i)]['scenario_id'] in split['holdout'] for i in np.unique(artifact['batches'])))
            with mock.patch.object(d,'source_data',return_value=dict(checkpoint_payload=dict(engine_checkpoint=dict(algorithm_config={})))), \
                 mock.patch.object(d,'fit_arm',return_value=dict(actual_updates=2000,curve=[],gradient_norms=[])) as fit:
                self.assertTrue(d.run(fit_args)['execution_completed'])
                self.assertEqual(fit.call_args.args[-1],'mean')
            self.assertTrue((fit_args.output_dir/'fit_curves.json').is_file())
            self.assertTrue(json.loads((fit_args.output_dir/'summary.json').read_text())['execution_completed'])
            path=args.output_dir/'data.pt'
            artifact['batches'][0,0]=next(i for i,r in enumerate(artifact['rows']) if r['scenario_id'] in split['holdout'])
            torch.save(artifact,path)
            with self.assertRaisesRegex(ValueError,'SHA256'): d.load_prepared(fit_args)
            evidence['data_sha256']=d.file_sha256(path)
            (args.output_dir/'data_evidence.json').write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError,'Holdout leakage'): d.load_prepared(fit_args)

    def test_history_identity_and_pool_digest_are_strict(self):
        records=[dict(model='B',scenario_id=str(i),empty_scene=i<7,sequence_index=i) for i in range(20)]
        pool=SimpleNamespace(content_digest=d.cal.base.POOL_DIGEST,scenarios=[
            dict(scenario_id=r['scenario_id'],sequence_index=r['sequence_index'],
                 payload=dict(zones=[] if r['empty_scene'] else [{}])) for r in records])
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            prior=dict(inputs=dict(B=dict(sha256=d.cal.CHECKPOINT_SHA256)),
                       pool=dict(content_digest=pool.content_digest),selection=dict(scenarios=records))
            (root/'manifest.json').write_text(json.dumps(prior))
            (root/'baseline_episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            args=SimpleNamespace(history_dir=root)
            rs,split,_=d.history_selection(args,pool)
            self.assertEqual(len(rs),20);self.assertEqual(len(split['holdout']),4)
            pool.scenarios[0]['sequence_index']=-1
            with self.assertRaisesRegex(ValueError,'identity'): d.history_selection(args,pool)
            pool.content_digest='wrong'
            with self.assertRaisesRegex(ValueError,'provenance'): d.history_selection(args,pool)

    def test_evaluation_metrics_actual_return_and_current_td(self):
        class Critic(torch.nn.Module):
            def __init__(self,value): super().__init__(); self.value=value
            def forward(self,obs,actions):return torch.full((len(actions),),self.value)
        rows=[dict(scenario_id='a',empty_scene=True,done=True,G=3.),
              dict(scenario_id='b',empty_scene=False,done=False,G=7.)]
        with mock.patch.object(d.cal,'tensor_batch',side_effect=lambda rows,ids,*a,**kw:(None,torch.zeros(len(ids),2))), \
             mock.patch.object(d,'dynamic_labels',side_effect=lambda data,targets,rows,ids,agg:torch.tensor([2.,4.])[ids]):
            result=d.evaluate_fixed(dict(device='cpu'),[Critic(1.),Critic(5.)],[],rows,dict(train=['a'],holdout=['b']),'mean')
        a=result['per_scene']['a']['critic1']['transition_weighted']
        self.assertEqual((a['Q_minus_G'],a['MAE'],a['RMSE'],a['TD_MAE']),(-2.,2.,2.,1.))
        self.assertEqual(result['per_scene']['b']['critic2']['transition_weighted']['Q_minus_G'],-2.)

    def test_evaluate_run_saves_actual_q_g_and_goal_rate_without_fit(self):
        self._check_evaluate_run()

    def test_evaluate_rejects_checkpoint_replaced_during_evaluation(self):
        self._check_evaluate_run('checkpoint')

    def test_evaluate_rejects_pool_replaced_during_evaluation(self):
        self._check_evaluate_run('pool')

    def test_evaluate_rejects_history_manifest_changed_during_evaluation(self):
        self._check_evaluate_run('history_manifest')

    def test_evaluate_rejects_history_episodes_changed_during_evaluation(self):
        self._check_evaluate_run('episodes')

    def _check_evaluate_run(self, changed_input=None):
        fixture=calibration_fixtures.TestCalibration()
        with mock.patch.object(d.cal.base,'actor_action',return_value=np.zeros(2,dtype=np.float32)):
            data,scene,record=fixture.fixture()
        data.update(device='cpu',actor=torch.nn.Linear(1,1).requires_grad_(False),
            critics=[calibration_fixtures.TinyCritic().requires_grad_(False) for _ in range(2)])
        data['checkpoint_payload']['engine_checkpoint']=dict(algorithm_config=dict(target_q_aggregation='mean'))
        data['pool'].content_digest=d.cal.base.POOL_DIGEST
        with TemporaryDirectory() as tmp:
            root=Path(tmp); history=root/'history';history.mkdir()
            (history/'manifest.json').write_text(json.dumps(dict(selection=dict(scenarios=[dict(scenario_id='one')]))))
            (history/'baseline_episodes.jsonl').write_text('fixture history\n')
            checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'fixture')
            pool=root/'pool';pool.write_bytes(b'fixture pool')
            paths=dict(checkpoint=checkpoint,pool=pool,history_manifest=history/'manifest.json',
                       episodes=history/'baseline_episodes.jsonl')
            hashes={k:d.file_sha256(p) for k,p in paths.items()}
            data.update(checkpoint_hash=hashes['checkpoint'],pool_hash=hashes['pool'])
            args=SimpleNamespace(mode='evaluate',checkpoint=checkpoint,validation_pool=root/'pool',
                history_dir=history,device='cpu',output_dir=root/'out')
            scene.update(scenario_id='one')
            original_fly=d.cal.base.fly
            def fly(*args,**kwargs):
                result=original_fly(*args,**kwargs)
                if changed_input is not None: paths[changed_input].write_bytes(b'replaced during evaluation')
                return result
            with mock.patch.object(d.cal.base,'load_models',return_value=data), \
                 mock.patch.object(d,'history_selection',return_value=([],{},dict(one=scene))), \
                 mock.patch.object(d.cal.base,'fly',side_effect=fly), \
                 mock.patch.object(d.cal.base,'actor_action',return_value=np.zeros(2,dtype=np.float32)), \
                 mock.patch.object(d,'fit_arm',side_effect=AssertionError('no fitting')):
                if changed_input is not None:
                    with self.assertRaisesRegex(RuntimeError,'Evaluation input changed: '+changed_input): d.run(args)
                    summary=json.loads((args.output_dir/'summary.json').read_text())
                    self.assertFalse(summary['execution_completed'])
                    self.assertNotIn('result',summary)
                    self.assertFalse((args.output_dir/'evaluation.json').exists())
                    manifest=json.loads((args.output_dir/'manifest.json').read_text())
                    self.assertEqual(manifest['inputs'][changed_input]['sha256'],hashes[changed_input])
                    return
                result=d.run(args)
            self.assertTrue(result['execution_completed'])
            report=json.loads((args.output_dir/'evaluation.json').read_text())
            transitions=[json.loads(l) for l in (args.output_dir/'transitions.jsonl').read_text().splitlines()]
            self.assertEqual(len(transitions),2)
            self.assertEqual(transitions[0]['G'],record['discounted_return'])
            self.assertEqual(report['goal_rate'],float(record['outcome']=='goal'))
            self.assertEqual(report['aggregation'],'mean')
            self.assertEqual(report['checkpoint_sha256'],data['checkpoint_hash'])
            self.assertEqual({k:v['sha256'] for k,v in report['inputs'].items()},hashes)
            self.assertEqual(report['per_scene']['one']['partition'],'evaluation')
