"""Tiny CPU fixtures only; no real dataset replay or CUDA fitting."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.scripts import diagnose_v2_wd3_acc as d
import test_diagnose_v2_critic_calibration as fixtures


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear=torch.nn.Linear(6,2)
        self.register_buffer('action_limit',torch.tensor([.14,.2]))
        with torch.no_grad(): self.linear.weight.zero_(); self.linear.bias.zero_()
    def forward(self, obs): return self.linear(obs.ego_features)


def fixture(budget=2):
    with mock.patch.object(d.cal.base,'actor_action',return_value=np.zeros(2,dtype=np.float32)):
        data,scene,record=fixtures.TestCalibration().fixture()
    base=d.cal.replay_episode(data,scene,record)
    critics=[fixtures.TinyCritic(),fixtures.TinyCritic()]
    opt=torch.optim.Adam([p for c in critics for p in c.parameters()],lr=.001)
    sum(p.square().sum() for c in critics for p in c.parameters()).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    actor=Actor().requires_grad_(False)
    e=dict(critic_optimizer_state_dict=deepcopy(opt.state_dict()),
        algorithm_config=dict(tau=.25,policy_delay=2,critic_grad_clip_norm=.1),
        update_counts=dict(last_total_steps=0),actor_target_state_dict=deepcopy(actor.state_dict()),
        critic1_target_state_dict=deepcopy(critics[0].state_dict()),critic2_target_state_dict=deepcopy(critics[1].state_dict()))
    data.update(actor=actor,critics=[c.requires_grad_(False) for c in critics],device=torch.device('cpu'),
        checkpoint_payload=dict(engine_checkpoint=e,scenario_config={},formal_config=dict(seed=23)))
    rows=[]
    for i in range(60):
        for row in base:
            rows.append(dict(deepcopy(row),scenario_id=str(i),empty_scene=i<2 or i==30 or 40<=i<47,
                             outcome='goal' if i%2 else 'collision'))
    split=dict(train=[str(i) for i in range(30)],calibration=[str(i) for i in range(30,40)],score=[str(i) for i in range(40,60)])
    train=[i for i,r in enumerate(rows) if r['scenario_id'] in split['train']]
    batches=np.random.default_rng(d.SEED).choice(train,size=(budget,64),replace=True)
    cs,adam,_=d.cal.make_fit(data['critics'],e['critic_optimizer_state_dict'])
    artifact=dict(protocol=d.protocol(budget),rows=rows,split=split,batches=batches,generation=d.generation_config(data),
                  initial=d.initial_state(data,cs,d.cal.load_targets(data)[1:],adam))
    return data,artifact


def scene_fixture():
    scenes=[]
    for i in range(60):
        empty=i<2 or i==30 or 40<=i<47
        payload=dict(curriculum_level='easy',state=[i,0,10,0,0],goal=[i+60,0,10],zones=[] if empty else
            [dict(shape=dict(shape_type='sphere',center=[100+i,0,10],radius=2.),safety_margin=0.)])
        scenes.append(dict(scenario_id=str(i),sequence_index=i,payload=payload,
            partition='train' if i<30 else 'calibration' if i<40 else 'score'))
    return scenes


class TestWD3ACC(unittest.TestCase):
    def test_formulas_done_mask_and_g_never_in_td(self):
        a,b=torch.tensor([2.,8.]),torch.tensor([6.,4.])
        for arm,expected in [('min',([2.,4.],[2.,4.])),('mean',([4.,6.],[4.,6.])),
                              ('wd3',([3.,5.],[3.,5.])),('acc',([2.,6.],[4.,4.]))]:
            pair=d.target_pair(a,b,arm,.5)
            for actual,want in zip(pair,expected): self.assertTrue(torch.equal(actual,torch.tensor(want)))
        data,artifact=fixture()
        targets=d.cal.load_targets(data)[1:]
        ids=[0,1]
        for arm in d.ARMS:
            old=d.labels(data,targets,artifact['rows'],ids,arm,.5)
            self.assertTrue(all(x[1].item()==artifact['rows'][1]['reward'] for x in old))
            changed=deepcopy(artifact['rows'])
            for row in changed: row.pop('G')
            current=d.labels(data,targets,changed,ids,arm,.5)
            for x,y in zip(old,current): self.assertTrue(torch.equal(x,y))

    def test_scene_isolation_fixed_pool_physical_identity_and_order(self):
        scenes=scene_fixture()
        split=d.split_scenes(scenes,set(),set())
        self.assertEqual([len(split[k]) for k in ('train','calibration','score')],[30,10,20])
        self.assertEqual(split,d.split_scenes(list(reversed(scenes)),set(),set()))
        self.assertEqual(sum(int(sid)==30 for sid in split['calibration']),1)
        score_ids=set(split['score']); keys={d.scene_key(s['payload']) for s in scenes if s['scenario_id'] in score_ids}
        actual=d.split_scenes(scenes,score_ids,keys)
        self.assertEqual(set(actual['score']),score_ids)
        renamed=deepcopy(scenes)
        for s in renamed: s['scenario_id']='renamed_'+s['scenario_id']
        self.assertEqual({d.scene_key(s['payload']) for s in renamed if s['scenario_id'] in d.split_scenes(renamed,set(),keys)['score']},keys)
        with self.assertRaisesRegex(ValueError,'leaked'): d.split_scenes(scenes,score_ids|{split['train'][0]},keys)
        with self.assertRaisesRegex(ValueError,'leaked'):
            d.split_scenes(scenes,set(),keys|{d.scene_key(scenes[30]['payload'])})
        duplicate=deepcopy(scenes); duplicate[1]['payload']=deepcopy(duplicate[0]['payload'])
        with self.assertRaisesRegex(ValueError,'duplicated'): d.split_scenes(duplicate,set(),set())

    def test_acc_direction_normalization_scene_equal_and_bounds(self):
        rows=[dict(scenario_id='long',G=10.,empty_scene=True)]*10+[dict(scenario_id='short',G=-10.,empty_scene=False)]
        prediction=np.array([[0.,0.]]*10+[[10.,10.]])
        beta,record=d.acc_control(.5,prediction,rows,list(range(11)),.1)
        self.assertEqual(record['normalizer'],10.)
        self.assertEqual(record['residual'],-17.)
        self.assertEqual(beta,.45)
        # Within the nonempty group, a long scene must not dominate a short one.
        rs=[dict(scenario_id='e',empty_scene=True,G=10.)]+[
            dict(scenario_id='n_long',empty_scene=False,G=-10.)]*10+[
            dict(scenario_id='n_short',empty_scene=False,G=10.)]
        qs=np.array([[0.,0.]]+[[10.,10.]]*10+[[10.,10.]])
        _,weighted=d.acc_control(.5,qs,rs,list(range(len(rs))),.1)
        self.assertEqual(weighted['residual'],-8.)
        self.assertEqual(weighted['normalizer'],10.)
        for g,q,start,expected in ((10.,-100.,.99,1.),(-10.,100.,.01,0.),(0.,0.,.5,.5)):
            rs=[dict(scenario_id=str(i),G=g,empty_scene=i==0) for i in range(2)]
            value,_=d.acc_control(start,np.array([[q,q]]*2),rs,[0,1],.05)
            self.assertEqual(value,expected)
        with self.assertRaises(ValueError): d.acc_control(.5,np.array([[np.nan,0.]]),rows,[0],.05)

    def test_four_arms_same_initial_adam_batches_frozen_and_target_updates(self):
        torch.manual_seed(7)
        data,artifact=fixture()
        original=d.cal.state_hash([m.state_dict() for m in [data['actor'],*data['critics']]])
        allocations=[]; original_make=d.cal.make_fit
        def make(*a,**kw):
            value=original_make(*a,**kw); allocations.append(value)
            return value
        with mock.patch.object(d.cal,'make_fit',new=make):
            results=[d.fit_arm(data,deepcopy(artifact),arm) for arm in d.ARMS]
        for result in results:
            self.assertEqual(result['initial'],artifact['initial'])
            self.assertEqual(result['batch_sha256'],results[0]['batch_sha256'])
            self.assertEqual(result['actual_updates'],2)
            self.assertEqual(result['target_updates'],1)
            self.assertTrue(result['frozen_unchanged'])
            self.assertNotEqual(result['final_targets'],result['initial']['targets'])
            self.assertEqual(len(result['gradient']['pre_clip_norms']),2)
            self.assertFalse(result['formal_passed'])
        self.assertEqual(original,d.cal.state_hash([m.state_dict() for m in [data['actor'],*data['critics']]]))
        self.assertEqual([r['arm'] for r in results],list(d.ARMS))
        for i,(_,opt,params) in enumerate(allocations):
            self.assertEqual(len(opt.param_groups),1)
            for _,other_opt,other_params in allocations[i+1:]:
                self.assertTrue({p.data_ptr() for p in params}.isdisjoint({p.data_ptr() for p in other_params}))
                self.assertIsNot(opt,other_opt)
        bad=deepcopy(artifact); bad['batches'][0,0]=len(bad['rows'])-1
        with self.assertRaisesRegex(ValueError,'leakage'): d.fit_arm(data,bad,'acc')
        bad=deepcopy(artifact); bad['initial']['adam']='wrong'
        with self.assertRaisesRegex(ValueError,'identity'): d.fit_arm(data,bad,'min')

    def test_training_and_scoring_g_do_not_change_critic_updates_or_acc(self):
        torch.manual_seed(7)
        data,artifact=fixture(101)
        e=data['checkpoint_payload']['engine_checkpoint']
        e['algorithm_config'].update(tau=.005,critic_grad_clip_norm=None)
        e['critic2_target_state_dict']['linear.bias']+=100.
        cs,opt,_=d.cal.make_fit(data['critics'],e['critic_optimizer_state_dict'])
        artifact['initial']=d.initial_state(data,cs,d.cal.load_targets(data)[1:],opt)
        for row in artifact['rows']:
            if row['scenario_id'] in artifact['split']['calibration']: row['G']=-1000000.
        # Control update actually happens at 100; score G must never affect it.
        changed=deepcopy(artifact)
        for row in changed['rows']:
            if row['scenario_id'] not in changed['split']['calibration']: row['G']+=10000.
        first=d.fit_arm(data,artifact,'acc'); second=d.fit_arm(data,changed,'acc')
        self.assertEqual(first['controls'],second['controls'])
        self.assertEqual(first['final_online'],second['final_online'])
        self.assertEqual(len(first['controls']),2)
        changed=deepcopy(artifact)
        for row in changed['rows']:
            if row['scenario_id'] in changed['split']['calibration']: row['G']=1000000.
        third=d.fit_arm(data,changed,'acc')
        self.assertNotEqual(first['controls'][-1],third['controls'][-1])
        # At 100 beta changes; update 101 must consume the new scalar.
        self.assertNotEqual(first['final_online'],third['final_online'])
        before=deepcopy(artifact); before['protocol']=d.protocol(100); before['batches']=before['batches'][:100]
        altered=deepcopy(before)
        for row in altered['rows']:
            if row['scenario_id'] in altered['split']['calibration']: row['G']=1000000.
        self.assertEqual(d.fit_arm(data,before,'acc')['final_online'],d.fit_arm(data,altered,'acc')['final_online'])

    def test_metrics_scene_equal_failure_overestimation_and_empty(self):
        rows=[dict(scenario_id='a',G=0.),dict(scenario_id='a',G=0.),dict(scenario_id='b',G=0.)]
        q=np.array([[2.,-2.],[2.,-2.],[8.,4.]])
        value=d.metrics(rows,q,np.zeros_like(q),[0,1,2])['critic1']
        self.assertEqual(value['transition_weighted']['bias'],4.)
        self.assertEqual(value['scene_equal']['bias'],5.)
        self.assertAlmostEqual(value['scene_equal']['RMSE'],np.sqrt(34))
        self.assertEqual(d.metrics(rows,q,q,[])['information'],'insufficient')
        data,artifact=fixture()
        result=d.evaluate(data,data['critics'],d.cal.load_targets(data)[1:],artifact['rows'],artifact['split'],'min',.5)
        failures=[x for x in result['grouped'] if x['partition']=='score' and x['geometry']=='all' and x['slice']=='failure_path']
        self.assertEqual(failures[0]['transitions'],20)
        self.assertIn('overestimate_fraction',failures[0]['critic1']['transition_weighted'])

    def test_default_plan_no_io_cuda_or_fit_and_budget(self):
        args=['--mode','fit','--model','snn','--checkpoint','not-loaded','--validation-pool','not-loaded',
              '--prepared-dir','not-loaded','--arm','acc','--output-dir','not-created']
        with mock.patch.object(d,'run',side_effect=AssertionError('execution')), \
                mock.patch('torch.load',side_effect=AssertionError('load')),redirect_stdout(StringIO()) as out:
            self.assertEqual(d.main(args),0)
        value=json.loads(out.getvalue()); self.assertFalse(value['launch'])
        self.assertEqual(value['protocol']['updates'],10000)
        for value in (0,10001,True,1.5):
            with self.assertRaises(ValueError): d.protocol(value)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'not-created'
            args=['--mode','collect','--model','snn','--checkpoint','absent','--validation-pool','absent',
                  '--score-history','absent','--output-dir',str(path)]
            with mock.patch.object(d,'run',side_effect=AssertionError('collect')), \
                    mock.patch.object(d,'collection_plan',side_effect=AssertionError('generate')), \
                    mock.patch.object(d.cal.base,'fly',side_effect=AssertionError('flight')),redirect_stdout(StringIO()):
                self.assertEqual(d.main(args),0)
            self.assertFalse(path.exists())

    def test_launcher_preflight_uses_path_inputs_before_cuda_probe(self):
        import sys
        launcher = Path(__file__).resolve().parents[1] / 'docs/v2_wd3_acc_launch.sh'
        snippet = launcher.read_text(encoding='utf-8').split("<<'PREFLIGHT'\n", 1)[1].split('\nPREFLIGHT', 1)[0]
        digest = '2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b'
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ('pool.json', 'ann.pt', 'snn.pt')]
            for path in paths:
                path.write_bytes(b'preflight fixture')
            def source(args):
                self.assertIsInstance(args.checkpoint, Path)
                self.assertIsInstance(args.validation_pool, Path)
                self.assertEqual(d.file_sha256(args.checkpoint), d.file_sha256(paths[1]))
            with mock.patch.object(sys, 'argv', ['-', *map(str, paths)]), \
                    mock.patch('brain_uav.trainers.v2_validation.load_v2_validation_pool',
                               return_value=SimpleNamespace(content_digest=digest)), \
                    mock.patch.object(d, 'source_data', side_effect=source) as load, \
                    mock.patch.object(torch.cuda, 'is_available', return_value=True) as available, \
                    mock.patch.object(torch.cuda, 'get_device_name', return_value='mock CUDA'), \
                    mock.patch.object(torch, 'ones') as ones, redirect_stdout(StringIO()):
                ones.return_value.sum.return_value.item.return_value = 1
                exec(compile(snippet, str(launcher), 'exec'), {})
                self.assertEqual([call.args[0].model for call in load.call_args_list], ['ann', 'snn'])
                self.assertEqual(load.call_count, 2)
                load.side_effect = ValueError('source rejected')
                available.reset_mock()
                with self.assertRaisesRegex(ValueError, 'source rejected'):
                    exec(compile(snippet, str(launcher), 'exec'), {})
                available.assert_not_called()

    def test_collection_selection_unstratified_train_then_one_nine_and_no_flight(self):
        data,_=fixture(); scenes=scene_fixture(); data['pool']=SimpleNamespace(scenarios=scenes[40:])
        # Geometry-only proposals, with one validation duplicate rejected first.
        candidates=[scenes[40]['payload']]+[s['payload'] for s in scenes[:40]]
        generator=mock.Mock(); generator.generate.side_effect=candidates
        with mock.patch.object(d,'V2ScenarioGenerator',return_value=generator) as factory, \
                mock.patch.object(d.cal.base,'select_scenarios',return_value=scenes[40:]), \
                mock.patch.object(d.cal.base,'fly',side_effect=AssertionError('flight')):
            plan,config=d.collection_plan(SimpleNamespace(),data)
        self.assertEqual(factory.call_args.kwargs['seed'],d.SEED)
        self.assertEqual(len(plan),60); self.assertEqual(config['candidate_draws'],41)
        self.assertEqual([s['payload'] for s in plan[:30]],[s['payload'] for s in scenes[:30]])
        self.assertEqual(sum(not s['payload']['zones'] for s in plan[30:40]),1)
        self.assertEqual(config['p_empty'],d.V2ScenarioGeneratorConfig().zone_count_probabilities['easy'][0])

    def test_all_twenty_snn_score_records_reused_without_flight(self):
        data,_=fixture(); scenes=scene_fixture()[40:]
        data['checkpoint_hash']='source';data['pool']=SimpleNamespace(content_digest='pool',scenarios=scenes)
        records=[dict(model='B',scenario_id=s['scenario_id'],sequence_index=s['sequence_index'],
                      empty_scene=not s['payload']['zones']) for s in scenes]
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifest.json').write_text(json.dumps(dict(inputs=dict(B=dict(sha256='source')),
                pool=dict(content_digest='pool'),execution=dict(device='cuda:0'))),encoding='utf-8')
            (root/'baseline_episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
            with mock.patch.object(d.cal.base,'select_scenarios',return_value=scenes), \
                    mock.patch.object(d.cal.base,'fly',side_effect=AssertionError('must reuse')):
                actual,selected,backend=d.history_score(SimpleNamespace(score_history=root),data)
                self.assertEqual(actual,records);self.assertEqual(len(selected),20)
                self.assertEqual(backend['device'],'cuda:0')
                (root/'baseline_episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in records[:-1]),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'twenty'): d.history_score(SimpleNamespace(score_history=root),data)

    def test_snn_collection_validates_twenty_before_forty_new_flights(self):
        data,_=fixture();scenes=scene_fixture()
        data['checkpoint_payload']['reward_config']={}
        prior=[dict(scenario_id=s['scenario_id'],reused=True) for s in scenes[40:]]
        order=[]
        def verify(data,scene,record): order.append('reuse' if record.get('reused') else 'new');return []
        def fly(*a,**kw):
            self.assertEqual(order[:20],['reuse']*20)
            return dict(outcome='goal')
        with TemporaryDirectory() as directory:
            root=Path(directory);history=root/'prior';history.mkdir()
            checkpoint=root/'source';checkpoint.write_bytes(b'source');pool=root/'pool';pool.write_bytes(b'pool')
            for name in ('manifest.json','baseline_episodes.jsonl'): (history/name).write_text('{}',encoding='utf-8')
            data.update(checkpoint_hash=d.file_sha256(checkpoint),pool_hash=d.file_sha256(pool))
            args=SimpleNamespace(mode='collect',model='snn',checkpoint=checkpoint,validation_pool=pool,
                independent_history=None,score_history=history,output_dir=root/'new',updates=2)
            with mock.patch.object(d,'source_data',return_value=data), \
                    mock.patch.object(d,'collection_plan',return_value=(scenes,d.generation_config(data))), \
                    mock.patch.object(d,'history_score',return_value=(prior,scenes[40:],{'device':'cuda:0'})), \
                    mock.patch.object(d.cal.base,'make_env',return_value=mock.Mock(reset=mock.Mock(return_value=(None,None)))), \
                    mock.patch.object(d,'verify_record',side_effect=verify), \
                    mock.patch.object(d.cal.base,'fly',side_effect=fly) as flights:
                result=d.run(args)
            self.assertEqual(flights.call_count,40)
            self.assertEqual(result['result']['reused'],20)
            self.assertEqual(result['result']['collected'],40)
            self.assertEqual(len((args.output_dir/'baseline_episodes.jsonl').read_text(encoding='utf-8').splitlines()),60)

    def test_real_small_environment_collect_and_strict_replay(self):
        data,artifact=fixture()
        # Use the genuine two-step environment and actor_action; no source-model flight.
        scene=dict(scenario_id='tiny',sequence_index=0,partition='train',
                   payload=fixtures._scenario([0,0,10,0,0],[60,0,10],[]))
        with TemporaryDirectory() as directory:
            root=Path(directory); checkpoint=root/'source';checkpoint.write_bytes(b'fixture')
            pool=root/'pool';pool.write_bytes(b'fixture pool')
            data.update(checkpoint_hash=d.file_sha256(checkpoint),pool_hash=d.file_sha256(pool))
            data['checkpoint_payload']['reward_config']={}
            args=SimpleNamespace(mode='collect',model='ann',checkpoint=checkpoint,validation_pool=pool,
                independent_history=None,score_history=None,output_dir=root/'collected',updates=2)
            original_fly=d.cal.base.fly
            def fly(*a,**kw):
                self.assertTrue((args.output_dir/'manifest.json').exists())
                return original_fly(*a,**kw)
            with mock.patch.object(d,'source_data',return_value=data), \
                    mock.patch.object(d,'collection_plan',return_value=([scene],d.generation_config(data))), \
                    mock.patch.object(d.cal.base,'fly',side_effect=fly) as flights:
                result=d.run(args)
            self.assertTrue(result['execution_completed']);self.assertEqual(flights.call_count,1)
            record=json.loads((args.output_dir/'baseline_episodes.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(record['episode_length'],2);self.assertEqual(len(record['states']),3)
            self.assertTrue(record['truncated'])
            self.assertEqual(len(d.verify_record(data,scene,record)),2)
            for key in ('actions','rewards','states','pre_action_hashes','final_info','discounted_return'):
                missing=deepcopy(record);missing.pop(key)
                with self.subTest(missing=key),self.assertRaises((KeyError,ValueError)):
                    d.verify_record(data,scene,missing)
            bad=deepcopy(record);bad['rewards'][0]+=1
            with self.assertRaisesRegex(ValueError,'mismatch'): d.verify_record(data,scene,bad)
            bad=deepcopy(record);bad['pre_action_hashes'][0]='tampered'
            with self.assertRaisesRegex(ValueError,'hash'): d.verify_record(data,scene,bad)

    def test_prepare_load_fit_artifact_hashes_and_failure_summary(self):
        data,artifact=fixture()
        scenes=scene_fixture()
        data['pool']=SimpleNamespace(scenarios=scenes[40:])
        with TemporaryDirectory() as directory:
            root=Path(directory); history=root/'history';history.mkdir()
            checkpoint=root/'source.pt'; checkpoint.write_bytes(b'checkpoint fixture')
            pool=root/'pool.json'; pool.write_text('{}',encoding='utf-8')
            data['checkpoint_hash']=d.file_sha256(checkpoint); data['pool_hash']=d.file_sha256(pool)
            records=[dict(scenario_id=s['scenario_id'],sequence_index=s['sequence_index'],
                          empty_scene=not s['payload']['zones'],outcome='goal') for s in scenes]
            manifest=dict(format='v2_frozen_actor_episodes_v2',exploration_noise=0.,generation=d.generation_config(data),
                code=d.code_identity(),
                backend=d.execution_backend(data),checkpoint_sha256=data['checkpoint_hash'],
                actor_sha256=d.cal.state_hash(data['actor'].state_dict()),scenarios=scenes)
            (history/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            (history/'baseline_episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
            d._write_json(history/'evidence.json',dict(manifest_sha256=d.file_sha256(history/'manifest.json'),
                episodes_sha256=d.file_sha256(history/'baseline_episodes.jsonl')))
            args=SimpleNamespace(mode='prepare',model='snn',checkpoint=checkpoint,validation_pool=pool,
                independent_history=history,score_history=None,updates=2,device='cpu',output_dir=root/'prepared')
            original_rows=artifact['rows'][:2]
            def replay(data,scene,record):
                self.assertTrue((root/'prepared/manifest.json').exists())
                return [dict(deepcopy(r),scenario_id=record['scenario_id'],empty_scene=record['empty_scene']) for r in original_rows]
            with mock.patch.object(d.cal.base,'select_scenarios',return_value=scenes[40:]), \
                    mock.patch.object(d,'source_data',return_value=data),mock.patch.object(d.cal,'replay_episode',side_effect=replay) as repeated:
                result=d.run(args)
            self.assertTrue(result['execution_completed']);self.assertEqual(repeated.call_count,60)
            d._write_json(history/'summary.json',dict(execution_completed=True))
            existing=deepcopy(args);existing.mode='collect';existing.output_dir=history
            with mock.patch.object(d.cal.base,'select_scenarios',return_value=scenes[40:]), \
                    mock.patch.object(d,'source_data',return_value=data), \
                    mock.patch.object(d,'verify_record',return_value=[]) as verify, \
                    mock.patch.object(d.cal.base,'fly',side_effect=AssertionError('existing data must not refly')):
                self.assertTrue(d.run(existing)['existing_validated'])
                self.assertEqual(verify.call_count,60)
            args.mode='fit';args.prepared_dir=root/'prepared';args.arm='acc';args.output_dir=root/'arm'
            # Histories no longer need to be mounted for a fit; sealed prepared
            # evidence retains their hashes. Source/pool still checked live.
            args.independent_history=None
            with mock.patch.object(d.cal.base,'select_scenarios',return_value=scenes[40:]), \
                    mock.patch.object(d,'source_data',return_value=data): summary=d.run(args)
            self.assertTrue(summary['execution_completed'])
            self.assertTrue((root/'arm/fit_curves.json').exists())
            loaded=json.loads((root/'arm/manifest.json').read_text(encoding='utf-8'))
            self.assertIn('independent_history_episodes',loaded['prepared_source_input_hashes'])
            for kind in ('input','source'):
                args.output_dir=root/kind
                real_code=d.code_identity()
                changed=deepcopy(real_code);changed['sha256']={'changed':'source'}
                def mutate(*a):
                    if kind=='input': checkpoint.write_bytes(b'changed during fit')
                    return {}
                with mock.patch.object(d.cal.base,'select_scenarios',return_value=scenes[40:]), \
                        mock.patch.object(d,'source_data',return_value=data), \
                        mock.patch.object(d,'fit_arm',side_effect=mutate), \
                        mock.patch.object(d,'code_identity',side_effect=[real_code,real_code,changed] if kind=='source' else None,
                                          return_value=real_code), \
                        self.assertRaisesRegex(RuntimeError,'changed'):
                    d.run(args)
                self.assertFalse(json.loads((args.output_dir/'summary.json').read_text(encoding='utf-8'))['execution_completed'])
                self.assertFalse((args.output_dir/'fit_curves.json').exists())
                checkpoint.write_bytes(b'checkpoint fixture')
            args.output_dir=root/'bad'
            (root/'prepared/data.pt').write_bytes(b'replaced')
            with mock.patch.object(d,'source_data',return_value=data),self.assertRaisesRegex(ValueError,'SHA256'): d.run(args)
            self.assertFalse(json.loads((root/'bad/summary.json').read_text(encoding='utf-8'))['execution_completed'])
            self.assertFalse((root/'bad/fit_curves.json').exists())

    def test_replay_and_actor_identity_failures_block_prepare(self):
        # Shared replay rejection remains real (hash/state/reward checks), not
        # reimplemented with weaker comparisons by the new diagnostic.
        fixtures.TestCalibration().test_real_linear_replay_exact_and_mismatch_rejected()
        data,artifact=fixture()
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifest.json').write_text(json.dumps(dict(format='v2_frozen_actor_episodes_v2',
                exploration_noise=0.,checkpoint_sha256='wrong',actor_sha256='wrong',scenarios=[])),encoding='utf-8')
            (root/'baseline_episodes.jsonl').write_text('',encoding='utf-8')
            data['checkpoint_hash']='source'
            with self.assertRaisesRegex(ValueError,'identity'):
                d.read_histories(SimpleNamespace(independent_history=root,score_history=None),data)
        artifact['rows'][0]['reward']=float('nan')
        with self.assertRaisesRegex(ValueError,'Non-finite'): d.fit_arm(data,artifact,'min')


if __name__=='__main__': unittest.main()
