import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from contextlib import contextmanager, ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brain_uav.scripts import diagnose_v2_online_confirmation as d


def payload(i=0, shape=None):
    return dict(state=[float(i),0,1],goal=[100,0,1],zones=[] if shape is None else
        [dict(shape=dict(shape_type=shape,center=[5,0,1]),safety_margin=0)])


def scenes():
    return [dict(scenario_id=str(i),payload=payload(i,None if i<8 else sorted(d.SHAPES)[i%5])) for i in range(40)]


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@contextmanager
def complete_bundle(root, args):
    """Small JSON-only bundle; no source models, scene generation or flight."""
    with ExitStack() as stack:
        source={'sha256':{'fixture.py':'f'*64}}
        stack.enter_context(patch.object(d.research,'identity',return_value=source))
        for key in ('validation_pool','ann_bc','snn_bc'):
            dump(getattr(args,key),{'fixture':key})
        args.source_manifest=root/'source.json';dump(args.source_manifest,source)
        stack.enter_context(patch.object(d.research,'BC',{m:('',d.file_sha256(getattr(args,m+'_bc'))) for m in ('ann','snn')}))
        lock=[]
        for job in d.IDS:
            m,arm=job.split('_',1);path=root/(job+'.pt');path.write_bytes(job.encode())
            lock.append(dict(id=job,model=m,arm=arm,requested_step=400000,actual_step=400000,
                local_checkpoint=str(path),sha256=d.file_sha256(path)))
        dump(args.candidate_lock,dict(candidates=lock))
        dump(args.completion_audit,dict(jobs=[dict(job=j,execution_completed=True,stage_steps=400000) for j in d.IDS]))
        for batch,ids in ((args.original_batch,[j['id'] for j in d.research.matrix()]),
                (args.acc_batch,[f'{m}_acc_{e}' for m in ('ann','snn') for e in ('001','010')])):
            dump(batch/'manifest.json',dict(plan=dict(jobs=[dict(id=j) for j in ids]),preflight=dict(source=source)))
            for job in ids:
                p=payload(999)
                path=batch/job/'other/training_scenes.jsonl';path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(start_step=0,physical_sha256=d.offline.scene_key(p),payload=p)),encoding='utf-8')
                path=batch/job/'other/metrics_reports/episodes.jsonl';path.parent.mkdir()
                path.write_text(json.dumps(dict(stage_steps=2)),encoding='utf-8')
        for m in ('ann','snn'):
            ss=[dict(scenario_id=str(i),payload=payload(i+1000),partition='train' if i<30 else 'calibration') for i in range(40)]
            keys={s['scenario_id']:d.offline.scene_key(s['payload']) for s in ss}
            dump(args.offline_root/f'{m}_collect/other/manifest.json',dict(scenarios=ss,scene_keys=keys))
            dump(args.offline_root/f'{m}_prepare/other/manifest.json',dict(scene_keys=keys,
                split={p:[s['scenario_id'] for s in ss if s['partition']==p] for p in ('train','calibration')}))
        cfg=dict(max_steps=2,delta_gamma_max=1.,delta_psi_max=1.)
        hs=[dict(scenario_id=f'old_{i}',sequence_index=i,payload=payload(10000+i,None if i<7 else 'sphere')) for i in range(100)]
        historical=SimpleNamespace(scenarios=hs,content_digest=d.research.DIGEST,scenario_config=cfg,uav_collision_radius=0.)
        independent=SimpleNamespace(scenarios=scenes(),content_digest='independent',scenario_config=cfg,
            uav_collision_radius=0.,master_seed=d.SEED)
        stack.enter_context(patch.object(d,'load_v2_validation_pool',side_effect=lambda p,**kw:
            historical if Path(p)==args.validation_pool else independent))
        out=args.output_dir;out.mkdir()
        tracked=d.Inputs();tracked.add(args.source_manifest)
        locked=d.candidate_records(args.candidate_lock,'local_checkpoint',tracked)
        for p in (args.completion_audit,args.validation_pool,args.ann_bc,args.snn_bc):tracked.add(p)
        excluded,evidence=d.exclusions(args,historical,tracked)
        # A legal pre-existing BC bundle has no per-scene files in this output.
        selected=d.cal.base.select_scenarios(historical)[:12]
        def record(sid=None,empty=True,hash_='a'*64,action=(0.,0.)):
            r=dict(episode_length=2,actions=[list(action),[0.,0.]],rewards=[1.,1.],
                pre_action_hashes=[hash_,'b'*64],states=[[0,0,1,0,0]]*3,
                terminated=False,truncated=True,final_steps=2,outcome='timeout',final_info={'outcome':'timeout'},
                episode_return=2.,discounted_return=1.99)
            if sid is not None:r.update(scenario_id=sid,empty_scene=empty)
            return r
        shared={};bc_saved={}
        for m in ('ann','snn'):
            episodes=[record(s['scenario_id'],not s['payload']['zones']) for s in selected]
            shared[m]=[dict(scene=s,prefix=[],hash=r['pre_action_hashes'][0]) for s,r in zip(selected,episodes)]
            anchor=dict(identity=dict(actor='c'*64,historical_pool=d.research.DIGEST,backend={'path':'fixture'},
                scenario_ids=[s['scenario_id'] for s in selected]),episodes=episodes)
            supplied=root/f'{m}_supplied_anchors.json';dump(supplied,anchor);setattr(args,m+'_anchors',supplied)
            tracked.add(supplied);bc_saved[m]=anchor
        spec=json.loads(json.dumps(d.plan(args)))
        dump(out/'other/excluded_scene_keys.json',sorted(excluded))
        dump(out/'review/manifest.json',dict(plan=spec,candidates=locked,inputs=tracked.files.copy(),
            diagnostic_source=source,exclusions=evidence,excluded_unique=len(excluded),
            excluded_keys=dict(path='other/excluded_scene_keys.json',sha256=d.file_sha256(out/'other/excluded_scene_keys.json'))))
        pool_path=out/'other/independent_pool.json';dump(pool_path,{'fixture_pool':40});tracked.add(pool_path)
        dump(out/'review/scene_plan.json',dict(path=str(pool_path),sha256=d.file_sha256(pool_path),
            content_digest=independent.content_digest,scene_keys=[d.offline.scene_key(s['payload']) for s in independent.scenarios],
            historical_pool=dict(path=str(args.validation_pool),sha256=d.file_sha256(args.validation_pool),digest=d.research.DIGEST)))
        dump(out/'review/probe_plan.json',shared);dump(out/'review/sealed_inputs.json',tracked.files)
        for m,anchor in bc_saved.items():dump(out/f'other/BC_{m}/anchors.json',anchor)
        results={}
        for c in locked:
            base=out/'other'/c['id'];flights=[];scores=[];per={}
            for s in independent.scenarios:
                sid=s['scenario_id'];r=record(sid,not s['payload']['zones']);dump(base/f'independent/{sid}.json',r)
                flights.append(dict(scenario_id=sid,outcome='timeout',length=2,episode_return=2.,G0=1.99))
                scores.extend(dict(scenario_id=sid,step=i,G=g,reward=1.,done=i==1,Q1=0.,Q2=0.) for i,g in enumerate((1.99,1.)))
            rows=[dict(r,empty_scene=not independent.scenarios[int(r['scenario_id'])]['payload']['zones'],
                outcome='timeout',remaining_steps=2-r['step']) for r in scores]
            scene_info={s['scenario_id']:dict(outcome='timeout',shape_types=[z['shape']['shape_type'] for z in s['payload']['zones']])
                for s in independent.scenarios}
            stats=d.metrics.grouped_statistics(rows,np.zeros((len(rows),2)),scene_info)
            stats.update(flights=flights,transition_scores=scores,outcomes={'timeout':40},goal_rate=0.)
            dump(base/'independent_scores.json',stats);probes=[];branches=0
            for i,p in enumerate(shared[c['model']]):
                aa,ss=d.actions(np.zeros(2),np.zeros(2),np.ones(2));directory=base/'development_probes'
                dump(directory/f'plan_{i:02d}.json',dict(prefix=p,sources=ss,actions=[a.tolist() for a in aa]))
                rows=[]
                for j,a in enumerate(aa):
                    branch=record(hash_=p['hash'],action=a.tolist());dump(directory/f'branch_{i:02d}_{j}.json',branch)
                    rows.append(dict(candidate_index=j,action=a.tolist(),Q1=float(j),Q2=float(j+1),G=1.99,
                        critic_disagreement=1.,outcome='timeout',remaining_initial_steps=2,consumed_initial_steps=0,
                        Q1_minus_G=j-1.99,Q2_minus_G=j+1-1.99))
                branches+=len(rows)
                probes.append(dict(scenario_id=p['scene']['scenario_id'],state_hash=p['hash'],sources=ss,
                    rows=rows,rankings=d.rankings(rows,ss)))
            dump(base/'probe_results.json',probes)
            results[c['id']]=dict(per_scene=flights,checkpoint_sha256=c['sha256'],frozen_unchanged=True,
                branches=branches,probe_count=12,outcomes={'timeout':40},goal_rate=0.)
        result=dict(execution_completed=True,results=results,inputs=tracked.files,diagnostic_source=source,
            independent_pool_digest=independent.content_digest,
            artifacts={p.relative_to(out).as_posix():d.file_sha256(p) for p in out.rglob('*') if p.is_file()})
        dump(out/'review/summary.json',result)
        yield result


class Confirmation(unittest.TestCase):
    def args(self,root):
        return d.parser().parse_args([v for k in ('candidate-lock','completion-audit','original-batch',
            'acc-batch','offline-root','validation-pool','ann-bc','snn-bc','output-dir','historical-repo')
            for v in ('--'+k,str(root/k))])

    def test_plan_has_no_io_models_generation_or_flight(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); args=self.args(root)
            with patch.object(d.cal.base,'load_models',side_effect=AssertionError),patch.object(d,'select_scenes',side_effect=AssertionError):
                p=d.execute(args)
            self.assertFalse(p['launch']); self.assertEqual(list(root.iterdir()),[])
            self.assertEqual((p['independent_flights'],p['optional_BC_anchor_flights'],p['branch_limit']),(240,24,432))

    def test_scene_physical_identity_isolation_strata_and_coverage(self):
        ss=scenes(); d.check_scenes(ss,set())
        with self.assertRaises(ValueError):d.check_scenes(ss,{d.offline.scene_key(ss[0]['payload'])})
        for kind in ('duplicate','strata','coverage'):
            other=json.loads(json.dumps(ss))
            if kind=='duplicate':other[-1]['payload']=other[-2]['payload']
            if kind=='strata':other[8]['payload']['zones']=[]
            if kind=='coverage':
                for s in other:
                    for z in s['payload']['zones']:z['shape']['shape_type']='sphere'
            with self.subTest(kind=kind),self.assertRaises(ValueError):d.check_scenes(other,set())
        changed=json.loads(json.dumps(ss[0]['payload']));changed['metadata']={'ignored':'id'}
        self.assertEqual(d.offline.scene_key(changed),d.offline.scene_key(ss[0]['payload']))

    def test_clipped_float32_dedup_source_mapping_and_references(self):
        a,s=d.actions(np.ones(2,dtype=np.float32),np.array([-.5,-.5]),np.ones(2))
        self.assertLessEqual(len(a),6);self.assertEqual(s['BC'],s['BC_0_+1']);self.assertEqual(a[0].dtype,np.float32)
        rows=[dict(candidate_index=i,Q1=float(i),Q2=float(i+1),G=float(-i),
            critic_disagreement=1.,outcome='goal') for i in range(len(a))]
        ranked=d.rankings(rows,s)
        common=ranked['Q1']['common'];own=ranked['Q1']['including_own']
        self.assertNotIn(s['own'],[r['candidate_index'] for r in common['relative_to_own']])
        self.assertEqual(common['own_actor_candidate'],0)
        self.assertEqual(own['own_actor_candidate'],s['own'])
        self.assertEqual(common['effective_non_tied_pairs'],3)
        for r in rows:r['Q1']=1.;r['Q2']=1.
        self.assertEqual(d.rankings(rows,s)['Q1']['common']['ranking_information'],'insufficient')

    def test_frozen_state_and_nonfinite_rejection(self):
        models=[torch.nn.Linear(2,1).eval().requires_grad_(False) for _ in range(3)]
        data=dict(actor=models[0],critics=models[1:]); before=d.frozen_identity(data)
        with torch.no_grad():models[0].weight.add_(1)
        self.assertNotEqual(before,d.frozen_identity(data))
        models[0].train()
        with self.assertRaises(ValueError):d.frozen_identity(data)
        with self.assertRaises(ValueError):d.actions([np.nan,0],[0,0],np.ones(2))

    def test_candidate_lock_tamper_and_strict_validator(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); records=[]
            for job in d.IDS:
                model,arm=job.split('_',1);p=root/(job+'.pt')
                torch.save(dict(arm=arm,requested_step=400000,actual_step=400000,model=model),p)
                records.append(dict(id=job,model=model,arm=arm,requested_step=400000,
                    actual_step=400000,local_checkpoint=str(p),sha256=d.file_sha256(p)))
            lock=root/'lock.json';dump(lock,dict(candidates=records))
            class Guard:
                def add(self,p):return Path(p)
            with patch.object(d,'validate_snapshot') as validate:
                result=d.candidates(lock,'local_checkpoint',Guard());self.assertEqual(len(result),6)
                self.assertEqual(validate.call_args.kwargs['expected_model_type'],'snn')
                records[0]['sha256']='wrong';dump(lock,dict(candidates=records))
                with self.assertRaisesRegex(ValueError,'SHA256'):d.candidates(lock,'local_checkpoint',Guard())
                records[0]['actual_step']=399999;dump(lock,dict(candidates=records))
                with self.assertRaisesRegex(ValueError,'metadata'):d.candidates(lock,'local_checkpoint',Guard())

    def test_two_pools_never_rebind_historical_checkpoint(self):
        c=dict(sha256='checkpoint',model='ann')
        data=dict(checkpoint_hash='checkpoint',model='ann',pool=SimpleNamespace(content_digest=d.research.DIGEST))
        d.validate_loaded(data,c)
        data['pool']=SimpleNamespace(content_digest='independent40')
        with self.assertRaises(ValueError):d.validate_loaded(data,c)

    def test_complete_episode_rejects_missing_and_nonfinite(self):
        r=dict(episode_length=1,actions=[[0,0]],rewards=[1.],pre_action_hashes=['h'],states=[[0],[1]],terminated=True,truncated=False)
        d.complete_episode(r)
        r['pre_action_hashes']=[]
        with self.assertRaises(ValueError):d.complete_episode(r)
        r['pre_action_hashes']=['h'];r['rewards']=[float('nan')]
        with self.assertRaises(ValueError):d.complete_episode(r)

    def test_reuse_rejects_incomplete_or_changed_artifacts(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);summary=root/'review/summary.json'
            dump(summary,dict(execution_completed=False))
            with self.assertRaises(ValueError):d.reuse(root)
            artifact=root/'other/record.json';dump(artifact,{'x':1})
            result=dict(execution_completed=True,results={j:dict(per_scene=[{}]*40,probe_count=12,
                branches=60,frozen_unchanged=True) for j in d.IDS},inputs={},
                diagnostic_source={'sha256':{}},artifacts={'other/record.json':d.file_sha256(artifact)})
            dump(summary,result)
            with patch.object(d.research,'identity',return_value={'sha256':{}}):
                with self.assertRaises(ValueError):d.reuse(root)

    def test_branch_uses_candidate_first_reward_own_continuation_remaining_time(self):
        # Actual fly/reset_replay helpers, a deterministic two-control tiny environment.
        class Env:
            def reset(self,options):
                self.steps=0;self.state=np.array([0.]);return self.state.copy(),{'outcome':'flying'}
            def step(self,action):
                self.steps+=1;self.state+=float(action[0]);done=self.steps==3
                return self.state.copy(),float(action[0]),done,False,{'outcome':'timeout' if done else 'flying'}
        env=Env();scene={'payload':{}}
        data=dict(gamma=.5,scenario=SimpleNamespace(max_steps=3))
        with patch.object(d.cal.base,'probe_state',side_effect=lambda e,o,i:dict(steps=e.steps)), \
             patch.object(d.cal.base,'actor_action',return_value=np.array([2.,0.],dtype=np.float32)) as actor:
            obs,state=d.cal.base.reset_replay(env,scene,[[.5,0]],d.cal.base.digest({'steps':1}))
            r=d.cal.base.fly(data,env,obs,first_action=[1.,0.],initial_info=state,record_probe_hashes=True)
        self.assertEqual(r['rewards'],[1.,2.]);self.assertEqual(r['discounted_return'],2.)
        self.assertEqual(r['final_steps'],3);self.assertEqual(r['episode_length'],2);actor.assert_called_once()

    def test_exclusion_reads_twenty_jobs_all_stages_and_offline(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.args(root)
            for batch,ids in ((a.original_batch,[j['id'] for j in d.research.matrix()]),
                (a.acc_batch,[f'{m}_acc_{e}' for m in ('ann','snn') for e in ('001','010')])):
                dump(batch/'manifest.json',dict(plan=dict(jobs=[dict(id=j) for j in ids]),preflight=dict(source={})))
                for j in ids:
                    p=batch/j/'other/training_scenes.jsonl';p.parent.mkdir(parents=True)
                    p.write_text('\n'.join(json.dumps(dict(start_step=t,physical_sha256=d.offline.scene_key(payload(t)),
                        payload=payload(t))) for t in (0,200001,399900)),encoding='utf-8')
                    e=batch/j/'other/metrics_reports/episodes.jsonl';e.parent.mkdir()
                    e.write_text('\n'.join(json.dumps(dict(stage_steps=t)) for t in (200001,399900)),encoding='utf-8')
            for m in ('ann','snn'):
                ss=[dict(scenario_id=str(i),payload=payload(i+100),partition='train' if i<30 else 'calibration') for i in range(40)]
                keys={s['scenario_id']:d.offline.scene_key(s['payload']) for s in ss}
                dump(a.offline_root/f'{m}_collect/other/manifest.json',dict(scenarios=ss,scene_keys=keys))
                dump(a.offline_root/f'{m}_prepare/other/manifest.json',dict(scene_keys=keys,
                    split={p:[s['scenario_id'] for s in ss if s['partition']==p] for p in ('train','calibration')}))
            class Guard:
                def add(self,p):
                    if not p.is_file():raise FileNotFoundError(p)
                    return p
            keys,evidence=d.exclusions(a,SimpleNamespace(scenarios=[dict(payload=payload(999))]),Guard())
            self.assertIn(d.offline.scene_key(payload(399900)),keys);self.assertEqual(len(evidence),23)
            log=a.original_batch/'ann_min/other/training_scenes.jsonl'
            original=log.read_text();log.write_text(original.splitlines()[0],encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'full episode history'):
                d.exclusions(a,SimpleNamespace(scenarios=[]),Guard())
            log.write_text(original,encoding='utf-8')
            (a.original_batch/'snn_acc_010/other/training_scenes.jsonl').unlink()
            with self.assertRaises(FileNotFoundError):d.exclusions(a,SimpleNamespace(scenarios=[]),Guard())

    def test_input_replacement_and_normalized_source_guard(self):
        with TemporaryDirectory() as tmp,patch.object(d.research,'identity',return_value={'sha256':{'x':'1'}}):
            path=Path(tmp)/'input';path.write_text('a');inputs=d.Inputs();inputs.add(path);inputs.check()
            path.write_text('b')
            with self.assertRaises(ValueError):inputs.check()

    def test_failed_launch_writes_incomplete_summary_not_success(self):
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp));a.launch=True
            with patch.object(d,'candidates',side_effect=ValueError('tampered')),patch.object(d.Inputs,'add',return_value=Path(tmp)/'source.json'), \
                 patch.object(d,'read',return_value={'sha256':{}}),patch.object(d.research,'identity',return_value={'sha256':{}}):
                with self.assertRaisesRegex(ValueError,'tampered'):d.execute(a)
            summary=json.loads((a.output_dir/'review/summary.json').read_text())
            self.assertFalse(summary['execution_completed']);self.assertIn('tampered',summary['error'])

    def test_real_tiny_environment_episode_replay_no_second_flight(self):
        from test_diagnose_v2_wd3_acc import fixture, fixtures
        data,_=fixture()
        scene=dict(scenario_id='tiny',payload=fixtures._scenario([0,0,10,0,0],[60,0,10],[]))
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'episode.json'
            record,rows=d.episode(data,scene,path)
            self.assertEqual(record['episode_length'],2);self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['G'],rows[0]['reward']+data['gamma']*rows[1]['reward'])
            with patch.object(d.cal.base,'fly',side_effect=AssertionError('no repeat flight')):
                same,replayed=d.episode(data,scene,path)
            self.assertEqual(same,record);self.assertEqual(replayed[0]['G'],rows[0]['G'])
            record['pre_action_hashes'][0]='wrong';dump(path,record)
            with self.assertRaisesRegex(ValueError,'hash'):d.episode(data,scene,path)

    def test_bc_anchor_shared_prefix_identity_and_no_recollection(self):
        from test_diagnose_v2_wd3_acc import fixture, fixtures
        data,_=fixture(); data['actor'].eval()
        scene=dict(scenario_id='tiny',payload=fixtures._scenario([0,0,10,0,0],[60,0,10],[]))
        data['pool']=SimpleNamespace(content_digest='historical',uav_collision_radius=data['pool'].uav_collision_radius)
        with TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            with patch.object(d.cal.base,'select_scenarios',return_value=[scene]*12):
                first=d.anchors(data,source,None,None)
                target=root/'target';target.mkdir()
                class Guard:
                    def add(self,p):return p
                with patch.object(d.cal.base,'fly',side_effect=AssertionError('no repeat')):
                    second=d.anchors(data,target,source/'anchors.json',Guard())
                self.assertEqual(first,second);self.assertEqual(first[0]['prefix'],[])
                saved=d.read(source/'anchors.json');saved['identity']['actor']='wrong';dump(source/'anchors.json',saved)
                with self.assertRaisesRegex(ValueError,'identity'):d.anchors(data,target,source/'anchors.json',Guard())

    def test_execute_final_hash_failure_never_marks_completed(self):
        from contextlib import ExitStack
        with TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);a=self.args(root);a.launch=True;a.source_manifest=root/'source.json'
            for k in ('candidate_lock','completion_audit','validation_pool','ann_bc','snn_bc','source_manifest'):
                dump(getattr(a,k),{})
            dump(a.source_manifest,dict(sha256={}))
            dump(a.completion_audit,dict(jobs=[dict(job=j,execution_completed=True,stage_steps=400000) for j in d.IDS]))
            checkpoint=root/'checkpoint';checkpoint.write_bytes(b'fixture')
            locked=[dict(id=j,model=j.split('_')[0],sha256=d.file_sha256(checkpoint),path=str(checkpoint)) for j in d.IDS]
            models=[torch.nn.Linear(2,1).eval().requires_grad_(False) for _ in range(3)]
            pool=SimpleNamespace(content_digest=d.research.DIGEST,uav_collision_radius=0.,scenarios=scenes())
            data=dict(actor=models[0],critics=models[1:],model='ann',checkpoint_hash=locked[0]['sha256'],pool=pool,
                device='cpu',scenario=SimpleNamespace())
            stack.enter_context(patch.object(d.research,'identity',return_value=dict(sha256={})))
            stack.enter_context(patch.object(d.research,'BC',{m:('',d.file_sha256(getattr(a,m+'_bc'))) for m in ('ann','snn')}))
            stack.enter_context(patch.object(d,'candidates',return_value=locked))
            stack.enter_context(patch.object(d,'exclusions',return_value=(set(),[])))
            stack.enter_context(patch.object(d.cal.base,'load_models',return_value=data))
            stack.enter_context(patch.object(d,'validate_loaded'))
            newpool=SimpleNamespace(content_digest='independent',scenarios=scenes())
            def generate(*args):
                self.assertTrue((a.output_dir/'review/manifest.json').is_file())
                return newpool
            stack.enter_context(patch.object(d,'select_scenes',side_effect=generate))
            stack.enter_context(patch.object(d,'save_v2_validation_pool',side_effect=lambda p,v:dump(p,{'new':40})))
            loader=stack.enter_context(patch.object(d,'load_v2_validation_pool',return_value=newpool))
            stack.enter_context(patch.object(d,'bc_data',return_value=data))
            stack.enter_context(patch.object(d,'anchors',return_value=[{'hash':'same'}]*12))
            def flights(*args):
                self.assertTrue((a.output_dir/'review/probe_plan.json').is_file())
                return dict(goal_rate=.5,outcomes={'goal':20,'timeout':20},flights=[{'scenario_id':str(i)} for i in range(40)])
            stack.enter_context(patch.object(d,'flight_metrics',side_effect=flights))
            stack.enter_context(patch.object(d,'probe',return_value=[{'rows':[{}]*5}]*12))
            original_hash=d.file_sha256
            def fail_final_hash(path):
                if (Path(path)==a.output_dir/'other/ann_min/independent_scores.json'
                        and (a.output_dir/'other/snn_acc_001/probe_results.json').exists()):
                    raise OSError('injected final hash failure')
                return original_hash(path)
            with patch.object(d,'file_sha256',side_effect=fail_final_hash):
                with self.assertRaisesRegex(OSError,'final hash failure'):d.execute(a)
            summary=json.loads((a.output_dir/'review/summary.json').read_text())
            self.assertFalse(summary['execution_completed']);self.assertIn('injected final hash failure',summary['error'])
            self.assertEqual(loader.call_args.args[0],a.output_dir/'other/independent_pool.json')
            self.assertNotEqual(loader.call_args.args[0],a.validation_pool)

    def test_legal_complete_reuse_is_read_only_and_checks_request(self):
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp));a.launch=True
            with complete_bundle(Path(tmp),a),patch.object(d.cal.base,'load_models',side_effect=AssertionError('model')), \
                 patch.object(d.cal.base,'fly',side_effect=AssertionError('flight')), \
                 patch.object(d,'select_scenes',side_effect=AssertionError('generation')),patch.object(torch,'load',side_effect=AssertionError('checkpoint load')):
                before={str(p):d.file_sha256(p) for p in a.output_dir.rglob('*') if p.is_file()}
                self.assertTrue(d.execute(a)['execution_completed'])
                self.assertFalse(any((a.output_dir/'other/BC_ann').glob('old_*.json')))
                self.assertEqual(before,{str(p):d.file_sha256(p) for p in a.output_dir.rglob('*') if p.is_file()})
                a.device='cpu'
                with self.assertRaisesRegex(ValueError,'sealed plan'):d.execute(a)

    def test_required_evidence_inventory_and_files_cannot_be_omitted(self):
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp));a.launch=True
            with complete_bundle(Path(tmp),a) as result:
                summary=a.output_dir/'review/summary.json'
                for field in ('inputs','artifacts'):
                    changed=json.loads(json.dumps(result));changed[field]={};dump(summary,changed)
                    with self.subTest(empty=field),self.assertRaises(ValueError):d.execute(a)
                for rel in ('other/ann_min/independent/0.json','other/BC_ann/anchors.json',
                        'other/ann_min/independent_scores.json','other/ann_min/development_probes/plan_00.json',
                        'other/ann_min/probe_results.json','other/ann_min/development_probes/branch_00_0.json',
                        'other/independent_pool.json','review/probe_plan.json','review/sealed_inputs.json'):
                    path=a.output_dir/rel;content=path.read_bytes()
                    changed=json.loads(json.dumps(result));changed['artifacts'].pop(rel);dump(summary,changed)
                    with self.subTest(unsealed=rel),self.assertRaises(ValueError):d.execute(a)
                    path.unlink();dump(summary,result)
                    with self.subTest(missing=rel),self.assertRaises((ValueError,FileNotFoundError)):d.execute(a)
                    path.write_bytes(content)
                for path in (a.candidate_lock,a.completion_audit,a.ann_bc,a.validation_pool):
                    changed=json.loads(json.dumps(result));changed['inputs'].pop(str(path));dump(summary,changed)
                    with self.subTest(input=path),self.assertRaises(ValueError):d.execute(a)

    def test_resealed_empty_duplicate_or_mismatched_records_rejected(self):
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp));a.launch=True
            with complete_bundle(Path(tmp),a) as result:
                mutations=[('other/ann_min/independent/0.json',lambda r:r.update(scenario_id='wrong')),
                    ('other/ann_min/independent_scores.json',lambda r:r['flights'].__setitem__(0,{})),
                    ('other/ann_min/independent_scores.json',lambda r:r['flights'].__setitem__(1,r['flights'][0])),
                    ('other/ann_min/independent_scores.json',lambda r:r['per_scene']['0'].update(Q1={})),
                    ('other/ann_min/development_probes/plan_00.json',lambda r:r.update(actions=[])),
                    ('other/ann_min/probe_results.json',lambda r:r.__setitem__(1,r[0])),
                    ('other/ann_min/development_probes/branch_00_0.json',lambda r:r.update(actions=[])),
                    ('other/BC_ann/anchors.json',lambda r:r['episodes'].__setitem__(1,r['episodes'][0]))]
                for rel,mutate in mutations:
                    path=a.output_dir/rel;original=path.read_bytes();value=d.read(path);mutate(value);dump(path,value)
                    changed=json.loads(json.dumps(result));changed['artifacts'][rel]=d.file_sha256(path)
                    dump(a.output_dir/'review/summary.json',changed)
                    with self.subTest(record=rel),self.assertRaises((ValueError,KeyError,IndexError)):d.execute(a)
                    path.write_bytes(original)

    def test_probe_nonzero_prefix_remaining_metadata(self):
        p=dict(scene={'scenario_id':'probe','payload':{}},prefix=[[0.,0.]]*10,hash='a'*64)
        actor=SimpleNamespace(action_limit=torch.ones(2));data=dict(actor=actor,scenario=SimpleNamespace(max_steps=100))
        branch=dict(episode_length=1,actions=[[0.,0.]],rewards=[1.],states=[[0]*5,[0]*5],pre_action_hashes=[p['hash']],
            terminated=True,truncated=False,outcome='goal',discounted_return=1.,final_steps=11)
        with TemporaryDirectory() as tmp,patch.object(d.cal.base,'make_env'), \
             patch.object(d.cal.base,'reset_replay',return_value=('obs',{'info':{}})), \
             patch.object(d.cal.base,'actor_action',return_value=np.zeros(2)), \
             patch.object(d.cal.base,'score',return_value={'Q1':1.,'Q2':1.,'critic_disagreement':0.}), \
             patch.object(d.cal.base,'fly',return_value=branch):
            result=d.probe(data,data,[p],Path(tmp))
        self.assertTrue(all(r['remaining_initial_steps']==90 and r['consumed_initial_steps']==10 for r in result[0]['rows']))


if __name__=='__main__':unittest.main()
