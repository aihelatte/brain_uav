"""Independent frozen snapshot evaluation, common-action probes and CUDA gate."""
from copy import deepcopy
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from brain_uav.scripts import diagnose_v2_critic_calibration as cal
from brain_uav.scripts.diagnose_v2_policy import _write_json, file_sha256


def statistics(rows,q):
    """Q1 (the actor's score) and Q2 kept separate. No overall pass threshold."""
    if not rows:return dict(information='insufficient',count=0)
    g=np.asarray([r['G'] for r in rows]);result=dict(count=len(rows))
    for i in range(2):
        error=q[:,i]-g;positive=error[error>0]
        result[f'Q{i+1}']=dict(Q_mean=float(q[:,i].mean()),G_mean=float(g.mean()),bias=float(error.mean()),
            MAE=float(abs(error).mean()),RMSE=float(np.sqrt((error**2).mean())),
            positive_error_mean=float(np.maximum(error,0).mean()),
            positive_only_mean=None if not len(positive) else float(positive.mean()),
            error_quantiles=dict(zip(('p0','p50','p90','p95','p99','p100'),np.quantile(error,[0,.5,.9,.95,.99,1]).tolist())),
            positive_tail_p95=None if not len(positive) else float(np.quantile(positive,.95)))
    cal.finite_tree(result);return result


def grouped_statistics(rows,q,scenes):
    per={sid:statistics([r for r in rows if r['scenario_id']==sid],q[[i for i,r in enumerate(rows) if r['scenario_id']==sid]])
         for sid in scenes}
    groups={}
    selections={'all':list(range(len(rows))),
        'empty':[i for i,r in enumerate(rows) if r['empty_scene']],
        'nonempty':[i for i,r in enumerate(rows) if not r['empty_scene']],
        'terminal':[i for i,r in enumerate(rows) if r['done']],
        'nonterminal':[i for i,r in enumerate(rows) if not r['done']],
        'failure_terminal':[i for i,r in enumerate(rows) if r['done'] and r['outcome']!='goal'],
        'failure_last5':[i for i,r in enumerate(rows) if r['outcome']!='goal' and r['remaining_steps']<=5]}
    for outcome in ('goal','collision','boundary','ground','timeout'):
        selections[outcome]=[i for i,r in enumerate(rows) if r['outcome']==outcome]
    for shape in ('sphere','ellipsoid','box','triangular_pyramid','quadrangular_pyramid'):
        selections[shape]=[i for i,r in enumerate(rows) if shape in scenes[r['scenario_id']]['shape_types']]
    for name,ids in selections.items():
        value=statistics([rows[i] for i in ids],q[ids])
        scene_values=[]
        for sid in sorted({rows[i]['scenario_id'] for i in ids}):
            subset=[i for i in ids if rows[i]['scenario_id']==sid]
            scene_values.append(statistics([rows[i] for i in subset],q[subset]))
        equal={}
        for c in ('Q1','Q2'):
            if scene_values:
                equal[c]={k:float(np.mean([v[c][k] for v in scene_values])) for k in ('Q_mean','G_mean','bias','MAE','positive_error_mean')}
                equal[c]['RMSE']=float(np.sqrt(np.mean([v[c]['RMSE']**2 for v in scene_values])))
        groups[name]=dict(transition_weighted=value,scene_equal=equal,scenes=len(scene_values))
    return dict(per_scene=per,groups=groups,note='overlapping subgroups; development subset, not blind or population validation')


def evaluate(args,output):
    from brain_uav.scripts.run_v2_online_wd3_acc import REPO,POOL,DIGEST,identity
    paths=dict(checkpoint=Path(args.checkpoint),pool=getattr(args,'validation_pool',None) or REPO/POOL)
    hashes={k:file_sha256(p) for k,p in paths.items()};code=identity()
    expected=getattr(args,'expected_pool_file_sha256',None)
    if expected is not None and hashes['pool']!=expected:raise ValueError('Evaluation pool differs from worker/preflight.')
    data=cal.base.load_models(paths['checkpoint'],paths['pool'],'cuda')
    if data['checkpoint_hash']!=hashes['checkpoint'] or data['pool'].content_digest!=DIGEST: raise ValueError('Evaluation input mismatch.')
    selected=cal.base.select_scenarios(data['pool'])
    _write_json(output/'manifest.json',dict(diagnostic_only=True,formal_passed=False,
        checkpoint_sha256=data['checkpoint_hash'],inputs=hashes,input_paths={k:str(p) for k,p in paths.items()},code=code,
        research=data['checkpoint_payload'].get('diagnostic_research'),selection=selected,
        gamma=data['gamma'],pool_digest=DIGEST,backend='eager frozen cuda actor/critics, no exploration'))
    before=cal.state_hash([m.state_dict() for m in [data['actor'],*data['critics']]])
    rows=[];scenes={}
    with (output/'episodes.jsonl').open('x',encoding='utf-8') as stream:
        for scene in selected:
            env=cal.base.make_env(data);obs,info=env.reset(options={'scenario':scene['payload']})
            record=dict(cal.base.fly(data,env,obs,record_probe_hashes=True,initial_info=info),
                scenario_id=scene['scenario_id'],sequence_index=scene['sequence_index'],empty_scene=not scene['payload']['zones'])
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            transitions=cal.replay_episode(data,scene,record)
            for row in transitions:
                row.update(outcome=record['outcome'],remaining_steps=len(transitions)-row['step'])
            rows.extend(transitions)
            scenes[scene['scenario_id']]=dict(outcome=record['outcome'],shape_types=sorted({z['shape']['shape_type'] for z in scene['payload']['zones']}))
    q=[]
    with torch.no_grad():
        for start in range(0,len(rows),64):
            obs,actions=cal.tensor_batch(rows,list(range(start,min(start+64,len(rows)))),data['device'])
            value=torch.stack([c(obs,actions).reshape(-1) for c in data['critics']],1)
            cal.finite_tree(value);q.extend(value.cpu().tolist())
    q=np.asarray(q)
    with (output/'transitions.jsonl').open('x',encoding='utf-8') as stream:
        for row,values in zip(rows,q):
            stream.write(json.dumps(dict(scenario_id=row['scenario_id'],step=row['step'],action=row['action'].tolist(),
                reward=row['reward'],G=row['G'],done=row['done'],outcome=row['outcome'],Q1=values[0],Q2=values[1]),allow_nan=False)+'\n')
    result=grouped_statistics(rows,q,scenes)
    if before!=cal.state_hash([m.state_dict() for m in [data['actor'],*data['critics']]]): raise RuntimeError('Frozen evaluation models changed.')
    if any(file_sha256(paths[k])!=h for k,h in hashes.items()) or identity()['sha256']!=code['sha256']:
        raise RuntimeError('Evaluation input/source changed.')
    _write_json(output/'evaluation.json',result)
    return dict(scenarios=len(selected),checkpoint_sha256=data['checkpoint_hash'],frozen_unchanged=True,
        goal_rate=sum(s['outcome']=='goal' for s in scenes.values())/len(selected))


def probe(args,output):
    from brain_uav.scripts.run_v2_online_wd3_acc import REPO,POOL,BC,DIGEST,identity
    from brain_uav.observations.v2_batch import collate_v2_observations
    data=cal.base.load_models(args.checkpoint,REPO/POOL,'cuda')
    if data['pool'].content_digest!=DIGEST: raise ValueError('Probe pool content digest mismatch.')
    source=identity();checkpoint_hash=data['checkpoint_hash'];pool_hash=data['pool_hash']
    # Shared probes derive from the fixed BC policy, not any tested arm outcome/Q.
    from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig,prepare_v2_stage_initialization
    bc_path=REPO/BC[data['model']][0]
    if file_sha256(bc_path)!=BC[data['model']][1]: raise ValueError('Probe BC identity mismatch.')
    prepared=prepare_v2_stage_initialization(V2FormalTrainingConfig(stage='easy',seed=23),
        init_checkpoint=bc_path,device=data['device'],model_type=data['model'])
    bc_data=dict(data,actor=prepared.bc_initialization.actor.to(data['device']).eval().requires_grad_(False))
    selected=cal.base.select_scenarios(data['pool'])[:12]
    _write_json(output/'manifest.json',dict(source=source,checkpoint_sha256=checkpoint_hash,
        pool_sha256=pool_hash,BC_sha256=BC[data['model']][1],scenes=selected,
        prefix_rule='BC natural flight then floor((T-1)/2) prefix',selection='IDs/sequence only; no outcome/Q selection',
        diagnostic_only=True,formal_passed=False))
    plans=[]
    for scene in selected:
        env=cal.base.make_env(bc_data);obs,info=env.reset(options={'scenario':scene['payload']})
        natural=cal.base.fly(bc_data,env,obs,record_probe_hashes=True,initial_info=info)
        index=(natural['episode_length']-1)//2
        plans.append(dict(scene=scene,prefix=natural['actions'][:index],hash=natural['pre_action_hashes'][index]))
    _write_json(output/'probe_plan.json',plans)  # all probes sealed before candidate scoring
    results=[]
    for plan in plans:
        env=cal.base.make_env(data)
        obs,_=cal.base.reset_replay(env,plan['scene'],plan['prefix'],plan['hash'])
        bc=cal.base.actor_action(bc_data,obs);own=cal.base.actor_action(data,obs)
        limit=data['actor'].action_limit.detach().cpu().numpy()
        candidates=[bc]
        for dim in range(2):
            for sign in (-1,1):
                value=bc.copy();value[dim]+=sign*.2*limit[dim];candidates.append(value)
        candidates.append(own);unique=[]
        for a in candidates:
            a=np.clip(a,-limit,limit).astype(np.float32)
            if not any(np.array_equal(a,b) for b in unique):unique.append(a)
        own_index=next(i for i,a in enumerate(unique) if np.array_equal(a,own))
        rows=[]
        for action in unique:
            obs,_=cal.base.reset_replay(env,plan['scene'],plan['prefix'],plan['hash'])
            scores=cal.base.score(data,obs,action)
            branch=cal.base.fly(data,env,obs,first_action=action)
            rows.append(dict(action=action.tolist(),Q1=scores['Q1'],Q2=scores['Q2'],
                G=branch['discounted_return'],outcome=branch['outcome'],
                critic_disagreement=scores['critic_disagreement'],candidate_index=len(rows),branch=branch))
        rankings={}
        for c in ('Q1','Q2'):
            view=[dict(r,Q1=r[c]) for r in rows]
            rankings[c]=cal.base.ranking(view,own_index)
        results.append(dict(scenario_id=plan['scene']['scenario_id'],candidates=rows,rankings=rankings,
            delta_Q1=[r['Q1']-rows[own_index]['Q1'] for r in rows],delta_G=[r['G']-rows[own_index]['G'] for r in rows]))
    if (file_sha256(args.checkpoint)!=checkpoint_hash or file_sha256(REPO/POOL)!=pool_hash
            or file_sha256(bc_path)!=BC[data['model']][1] or identity()['sha256']!=source['sha256']):
        raise RuntimeError('Probe inputs/source changed.')
    _write_json(output/'probes.json',results)
    return dict(probes=12,diagnostic_only=True)


def run(args):
    args.root.mkdir(parents=True,exist_ok=False)
    result=dict(diagnostic_only=True,formal_passed=False,execution_completed=False)
    started=time.perf_counter()
    try:
        result['result']=evaluate(args,args.root) if args.mode=='evaluate' else probe(args,args.root)
        result['execution_completed']=True
    except BaseException as exc:
        result['error']=str(exc);raise
    finally:
        result['process_wall_seconds']=time.perf_counter()-started
        _write_json(args.root/'summary.json',result)


def _append_update_evidence(snapshot, tensors, model):
    from brain_uav.scripts.v2_gpu_td3_diagnostic import complete_actor_evidence
    complete_actor_evidence(tensors, model)
    for name,tensor in tensors.items():snapshot['tensor.'+name]=tensor


def cuda_check(args):
    """Finite actual eager vs GPUCompute check; no training loop or data generation."""
    from brain_uav.scripts import run_v2_online_wd3_acc as run
    from brain_uav.scripts import diagnose_v2_mc_return_cuda as helper
    from brain_uav.scripts import profile_v2_td3 as profile
    from brain_uav.scripts.v2_gpu_compute_diagnostic import _mps_graph_replay_evidence
    from brain_uav.scripts.v2_gpu_td3_diagnostic import snapshot, flatten
    from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation,_configure_stage_compilation
    from brain_uav.trainers.v2_formal_training import prepare_v2_stage_initialization,build_v2_stage_engine,v2_bc_lambda
    from brain_uav.trainers.v2_validation import load_v2_validation_pool
    from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch
    from brain_uav.trainers.v2_gpu_update import require_cuda_abi
    args.root.mkdir(parents=True,exist_ok=False)
    report=dict(diagnostic_only=True,strict_passed=False,engineering_accepted=False,execution_completed=False,checks=[])
    started=time.perf_counter()
    try:
        # Same strict per-file/input provenance gate. tmux isn't needed for this finite check.
        evidence=run.preflight(args,cuda=False);require_cuda_abi()
        report['inputs']=evidence
        jobs=[j for j in run.matrix() if j['model']==args.model and (args.job is None or j['id']==args.job)]
        if not jobs: raise ValueError('Model/job mismatch.')
        for job in jobs:
            config=run.make_config(job);bc=run.REPO/run.BC[args.model][0]
            prepared=prepare_v2_stage_initialization(config,init_checkpoint=bc,device='cuda',model_type=args.model,rewards=run.research_rewards())
            pool=load_v2_validation_pool(run.REPO/run.POOL,expected_level='easy',expected_count=100,
                expected_scenario=prepared.scenario_config,expected_uav_collision_radius=prepared.uav_collision_radius)
            engines=[build_v2_stage_engine(None,config,init_checkpoint=bc,device='cuda',model_type=args.model,
                prepared_initialization=prepared,fused_adam=True,reduce_update_stat_syncs=True,
                gpu_td3_compute=enabled,non_blocking_batch_transfer=job['non_blocking_batch_transfer']).engine for enabled in (False,True)]
            eager,compute=engines
            for e in engines:
                if job['target_config'] is not None:e.configure_research_targets(job['target_config'])
                profile._set_numeric_engine_modes(e)
            compute.load_checkpoint_state_dict(deepcopy(eager.checkpoint_state_dict()))
            defaults=default_v2_cuda_graph_compilation(model=args.model,resolved_device='cuda')
            defaults.pop('fused_adam');defaults.pop('reduce_update_stat_syncs')
            _configure_stage_compilation(compute,pool,prepared,compile_critic_encoder=False,compile_target_encoders=False,**defaults)
            initial=helper.compare_snapshots(snapshot(eager)[0],snapshot(compute)[0])
            report['checks'].append(dict(arm=job['arm'],phase='initial_state',**initial))
            if not initial['passed']:raise AssertionError('Initial eager/GPUCompute state differs after prewarm.')
            warm=profile._compile_warmup_batches(pool=pool,prepared=prepared,batch_size=64,device=torch.device('cpu'),cache_ellipsoid_segment_clearance=True)
            mixed,_=profile._numeric_diagnostic_observation_batches(warm,device=torch.device('cpu'))
            empty=deepcopy(mixed);empty.presence_mask.zero_()
            # Warm both gradient banks in all three modes/shapes; then beta changes reuse them.
            steps=[1,3,4,6,8,10,25001,25003,25004,25006,25008,25010]
            cases=[('mixed',mixed),('empty',empty)]
            total=0
            previous_beta={}
            for case,obs in cases:
                batch=V2ReplayBatch(obs,torch.zeros(64,2),torch.linspace(-2,2,64).reshape(-1,1),deepcopy(obs),
                    torch.arange(64).remainder(2).float().reshape(-1,1),torch.zeros(64,1),torch.zeros(64,1),torch.ones(64,1))
                noise=torch.randn(64,2,generator=torch.Generator().manual_seed(23)).cuda()
                for n,step in enumerate(steps):
                    beta=(.0,.5,1.)[n%3] if job['arm'].startswith('acc_') else None
                    states=[];before=0 if compute._gpu_update_executor is None else compute._gpu_update_executor.captures
                    for engine in engines:
                        if beta is not None:engine.set_research_beta(beta)
                        tensors={'replay.mask':batch.obs.presence_mask.clone(),'replay.next_mask':batch.next_obs.presence_mask.clone()}
                        def record(name,value):tensors[name]=value.detach().cpu().clone()
                        _,pn,nc=config.noise_schedule.values(step-1,max_steps=500000)
                        engine.set_target_noise(policy_noise=pn,noise_clip=nc)
                        def update():
                            return engine.update_once(total_steps=step,bc_lambda=v2_bc_lambda(step-1,schedule='smooth-mid-strong'),diagnostic_tensor_recorder=record)
                        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(engine.device).clone()
                        with patch.object(engine.replay,'sample',return_value=batch),patch.object(torch,'randn_like',new=lambda x:noise.to(x.device)), \
                                (helper.eager_evidence(engine,record) if engine is eager else nullcontext()):
                            if engine is compute and n in (0,2,6,8):
                                metrics=[];proof=_mps_graph_replay_evidence(lambda:metrics.append(update()),engine.device);metric=metrics[0]
                                report.setdefault('graph_evidence',[]).append(proof)
                            else:metric=update()
                        if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state(engine.device)):
                            raise AssertionError('Controlled update/capture changed RNG despite fixed noise and batch.')
                        snap,_=snapshot(engine)
                        for module in ('actor','critic1','critic2'):
                            for name,p in getattr(engine,module).named_parameters():flatten(p.grad,'gradient.'+module+'.'+name,snap)
                        _append_update_evidence(snap,tensors,engine.model_type)
                        flatten({k:getattr(metric,k) for k in ('critic_loss','actor_loss','critic_target_q_mean','critic_td_error_mean')},'metric',snap)
                        states.append(snap)
                    check=helper.compare_snapshots(*states)
                    # Both physical banks are warm after n=1,3,7,9 respectively.
                    cached=n in (4,5,10,11)
                    check.update(arm=job['arm'],shape=case,step=step,beta=beta,captures=compute._gpu_update_executor.captures,rng_unchanged=True)
                    if cached and compute._gpu_update_executor.captures!=before:check['passed']=False
                    if job['target_config'] is not None:
                        target=states[1]['tensor.target.td'];q1=states[1]['tensor.target.q1'];q2=states[1]['tensor.target.q2']
                        expected=batch.reward+(1-batch.done)*config.gamma*compute._aggregate_target_q(q1.cuda(),q2.cuda()).cpu()
                        check['runtime_target_matches']=bool(torch.allclose(target,expected,rtol=1e-4,atol=1e-5))
                        check['passed'] &=check['runtime_target_matches']
                        mode='critic_only' if step%2 else 'frozen_targets' if step<=25000 else 'actor'
                        if beta is not None and mode in previous_beta and beta!=previous_beta[mode]:
                            from brain_uav.trainers.v2_td3 import research_target_pair
                            old_target=batch.reward+(1-batch.done)*config.gamma*research_target_pair(q1,q2,'acc',previous_beta[mode])
                            distinguishable=not bool(torch.allclose(target,old_target,rtol=1e-4,atol=1e-5))
                            check['beta_refresh_distinguishable_on_same_target_q']=distinguishable
                            check['passed'] &=distinguishable
                        if beta is not None:previous_beta[mode]=beta
                    report['checks'].append(check);total+=1
                    if not check['passed']:raise AssertionError('Strict research CUDA mismatch; no new ANN exemption.')
            old=compute._gpu_update_executor
            for e in engines:e.load_checkpoint_state_dict(deepcopy(e.checkpoint_state_dict()))
            if compute._gpu_update_executor is not None or old.graphs:raise AssertionError('Load did not invalidate graphs.')
            rebuilt=[]
            for engine in engines:
                _,pn,nc=config.noise_schedule.values(100001,max_steps=500000)
                engine.set_target_noise(policy_noise=pn,noise_clip=nc)
                with patch.object(engine.replay,'sample',return_value=batch),patch.object(torch,'randn_like',new=lambda x:noise.to(x.device)):
                    metric=engine.update_once(total_steps=100002,bc_lambda=v2_bc_lambda(100001,schedule='smooth-mid-strong'))
                state,_=snapshot(engine,metric);rebuilt.append(state)
            compared=helper.compare_snapshots(*rebuilt)
            if not compared['passed'] or compute._gpu_update_executor is old or compute._gpu_update_executor.captures==0:
                raise AssertionError('Research checkpoint Graph rebuild failed strict comparison.')
            report['checks'].append(dict(arm=job['arm'],phase='load_rebuild',**compared))
            report.setdefault('execution',[]).append(dict(arm=job['arm'],updates_per_path=total+1,metadata=compute.local_optimization_metadata(),load_invalidated=True))
            compute.invalidate_gpu_update()
            del engines,eager,compute,old
        if run.identity()['sha256']!=evidence['source']['sha256']:raise RuntimeError('Source changed during CUDA gate.')
        if sha256(Path(args.source_manifest).read_bytes()).hexdigest()!=evidence['source_manifest_sha256']:
            raise RuntimeError('Reviewed source manifest changed during CUDA gate.')
        for entry in evidence['inputs'].values():
            if file_sha256(Path(entry['path']))!=entry['sha256'] or file_sha256(run.REPO/run.POOL)!=entry['pool_file_sha256']:
                raise RuntimeError('Input changed during CUDA gate.')
        report.update(strict_passed=True,execution_completed=True)
    except BaseException as exc:
        report['error']=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        report['process_wall_seconds']=time.perf_counter()-started
        _write_json(args.root/'summary.json',report)
    return report
