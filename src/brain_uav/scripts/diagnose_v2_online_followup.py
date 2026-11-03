"""Bounded, independent follow-up; never resume or modify historical workers."""
import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from brain_uav.scripts import diagnose_v2_critic_calibration as cal
from brain_uav.scripts import run_v2_online_wd3_acc as research
from brain_uav.scripts.diagnose_v2_policy import file_sha256, _write_json
from brain_uav.trainers.v2_online_research import OnlineResearch, predict_bank, training_identity, validate_snapshot

SNN_ARMS=('min','mean','fixed_own','wd3_045','wd3_015')
RTOL,ATOL=1e-4,1e-5


class StrictMismatch(AssertionError):
    pass


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('snn-inference','acc-bank'))
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--historical-repo',type=Path,required=True)
    p.add_argument('--validation-pool',type=Path,required=True)
    p.add_argument('--ann-bc',type=Path,required=True)
    p.add_argument('--snn-bc',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-manifest',type=Path,default=research.REPO/'scripts/v2_online_wd3_acc_source.json')
    p.add_argument('--launch',action='store_true')
    p.add_argument('--closed-loop',action='store_true')
    p.add_argument('--updates-per-worker',type=int,default=6)
    p.add_argument('--worker-timeout',type=float,default=600.)
    # Internal subprocess selection; every CUDA trial owns a new context.
    p.add_argument('--worker-model',choices=('ann','snn'),help=argparse.SUPPRESS)
    p.add_argument('--bank-path',choices=('original','isolated','beta'),help=argparse.SUPPRESS)
    return p


def plan(args):
    if not 3<=args.updates_per_worker<=200 or not np.isfinite(args.worker_timeout) or args.worker_timeout<=0:
        raise ValueError('Require 3..200 updates/worker and a finite positive timeout.')
    if args.closed_loop and args.mode!='snn-inference':raise ValueError('Closed loop is inference-only.')
    if bool(args.worker_model)!=bool(args.bank_path):raise ValueError('Incomplete private worker selection.')
    if args.worker_model and args.mode!='acc-bank':raise ValueError('Private workers are acc-bank only.')
    for key in ('history','historical_repo','validation_pool','ann_bc','snn_bc','output_dir','source_manifest'):
        if not getattr(args,key).is_absolute():raise ValueError(f'{key} must be absolute.')
    return dict(diagnostic_only=True,launch=args.launch,mode=args.mode,
        inputs={k:str(getattr(args,k)) for k in ('history','historical_repo','validation_pool','ann_bc','snn_bc')},
        output=str(args.output_dir),single_action_limit=500,common_states=32,
        closed_loop_episodes=8 if args.closed_loop else 0,
        controlled_updates=4*args.updates_per_worker+2 if args.mode=='acc-bank' else 0,
        physical_update_budget=1000,main_reproduction_budget=800,beta_check_budget=200,
        per_main_process_budget=200,per_beta_process_budget=100,
        source_policy='separate diagnostic checkout; historical input identities are not current source identity',
        candidate='isolated eager critic copies, never replace formal predict_bank',rtol=RTOL,atol=ATOL)


def guard_locations(args):
    old=args.historical_repo.resolve();out=args.output_dir.resolve()
    if research.REPO.resolve()==old or out.is_relative_to(old) or out.is_relative_to(args.history.resolve()):
        raise ValueError('Use an independent code checkout and output outside historical repo/experiment.')


class Inputs:
    def __init__(self,args):
        self.files={};self.code=research.identity()
        expected=json.loads(self.add(args.source_manifest).read_text(encoding='utf-8'))
        if expected['sha256']!=self.code['sha256']:raise ValueError('Reviewed diagnostic source manifest mismatch.')
        self.history=json.loads(self.add(args.history/'manifest.json').read_text(encoding='utf-8'))
        for model in ('ann','snn'):
            path=self.add(getattr(args,model+'_bc'))
            if self.files[str(path)]!=research.BC[model][1]:raise ValueError('BC best identity mismatch.')
        self.add(args.validation_pool)

    def add(self,path):
        path=Path(path).resolve();value=file_sha256(path)
        if str(path) in self.files and self.files[str(path)]!=value:raise ValueError('Input replaced during loading.')
        self.files[str(path)]=value;return path

    def check(self):
        if any(file_sha256(Path(p))!=h for p,h in self.files.items()):raise ValueError('Historical/input file changed.')
        if research.identity()['sha256']!=self.code['sha256']:raise ValueError('Diagnostic source changed.')

    def evidence(self):
        return dict(files=self.files,diagnostic_source=self.code,
            historical_source=self.history['preflight']['source'],
            historical_source_is_not_diagnostic_source=True,
            runtime=dict(python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,threads=torch.get_num_threads()))


def discover(args,inputs,job,requested):
    """Discover via small sidecars, then validate actual checkpoint metadata/hash."""
    directory=args.history/job;matches=[]
    for path in sorted((directory/'review').glob('snapshot_*.json')):
        record=json.loads(inputs.add(path).read_text(encoding='utf-8'))
        if record['requested_step']==requested:matches.append(record)
    if len(matches)!=1:raise ValueError(f'Require unique {job} requested={requested} snapshot metadata.')
    record=matches[0];found=[]
    for path in (directory/'other').glob('*.pt'):
        # Hash identity, not a filename prefix, binds sidecar to snapshot.
        if file_sha256(path)==record['sha256']:found.append(path)
    if len(found)!=1:raise ValueError('Snapshot file identity missing/ambiguous.')
    path=inputs.add(found[0]);raw=torch.load(path,map_location='cpu',weights_only=False)
    model=job.split('_',1)[0]
    if model not in ('ann','snn'):raise ValueError('Unknown snapshot model prefix.')
    validate_snapshot(raw,expected_stage='easy',expected_model_type=model)
    if (raw['arm']!=job.split('_',1)[1] or raw['requested_step']!=requested
            or raw['actual_step']!=record['actual_step'] or record['pool_digest']!=research.DIGEST):
        raise ValueError('Snapshot metadata identity mismatch.')
    return path,raw,record


def load_data(path,args,inputs,device='cuda'):
    inputs.add(path)
    data=cal.base.load_models(path,args.validation_pool,device)
    if data['checkpoint_hash']!=inputs.files[str(Path(path).resolve())] or data['pool'].content_digest!=research.DIGEST:
        raise ValueError('Checkpoint/fixed pool identity mismatch.')
    inputs.check();return data


@contextmanager
def warmup_call_counts(engine):
    """Count host submissions outside compiled callables, not compiler tracing."""
    counts=dict(action_tensor_calls=0,compiled_replay_probe_calls=0)
    action=engine._action_inference_tensor;prepare=engine._prepare_action_inference_cuda_graph_replay
    def counted_action(*a,**kw):
        counts['action_tensor_calls']+=1
        return action(*a,**kw)
    def counted_prepare(*a,**kw):
        replay=prepare(*a,**kw)
        def counted_replay():
            counts['compiled_replay_probe_calls']+=1
            return replay()
        return counted_replay
    with patch.object(engine,'_action_inference_tensor',counted_action), \
            patch.object(engine,'_prepare_action_inference_cuda_graph_replay',counted_prepare):yield counts


def make_engine(args,data,raw,*,compile=True):
    from brain_uav.trainers.v2_formal_training import prepare_v2_stage_initialization,build_v2_stage_engine
    from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation,_configure_stage_compilation
    job=next(j for j in research.matrix() if j['model']==data['model'] and j['arm']==raw['arm'])
    config=research.make_config(job);bc=getattr(args,data['model']+'_bc')
    prepared=prepare_v2_stage_initialization(config,init_checkpoint=bc,device=data['device'],model_type=data['model'],rewards=research.research_rewards())
    defaults=default_v2_cuda_graph_compilation(model=data['model'],resolved_device=data['device'].type)
    engine=build_v2_stage_engine(None,config,init_checkpoint=bc,device=data['device'],model_type=data['model'],
        prepared_initialization=prepared,fused_adam=defaults['fused_adam'],reduce_update_stat_syncs=defaults['reduce_update_stat_syncs'],
        gpu_td3_compute=True,non_blocking_batch_transfer=False).engine
    engine.online_research_statistics=True  # same graph output contract as the historical online worker
    if job['target_config'] is not None:engine.configure_research_targets(job['target_config'])
    algorithm=raw['snapshot']['engine_checkpoint']['algorithm_config']
    engine.set_target_noise(policy_noise=algorithm['policy_noise'],noise_clip=algorithm['noise_clip'])
    engine.load_checkpoint_state_dict(deepcopy(raw['snapshot']['engine_checkpoint']))
    # Copies are created BEFORE any independent compilation; no live flag mutation.
    copies=deepcopy((engine.critic1,engine.critic2))
    for c in copies:c.requires_grad_(False)
    options=dict(defaults);options.pop('fused_adam');options.pop('reduce_update_stat_syncs')
    before=engine.update_count;weights=cal.state_hash(engine.checkpoint_state_dict());started=time.perf_counter()
    with warmup_call_counts(engine) as calls:
        metadata=_configure_stage_compilation(engine,data['pool'],prepared,compile_critic_encoder=False,
            compile_target_encoders=False,**options) if compile else {}
    compile_seconds=time.perf_counter()-started
    # GPUCompute's existing compile helper performs exactly one raw actor call
    # per warmup batch before its separately observed action/probe calls.
    calls['raw_actor_warmup_calls']=len(metadata.get('warmup_batch_shapes',[])) if compile else 0
    if engine.update_count!=before:raise RuntimeError('Unexpected state-changing compile warmup update.')
    if cal.state_hash(engine.checkpoint_state_dict())!=weights:raise RuntimeError('Compile prewarm changed network/target/Adam state.')
    return engine,copies,config,dict(configuration=metadata,device=str(engine.device),
        gpu=torch.cuda.get_device_name(engine.device),seconds=compile_seconds,
        effective_formal_config=config.to_dict(),actual_algorithm=engine._algorithm_config(),prewarm_state_unchanged=True,
        state_changing_warmup_updates=engine.update_count-before,
        pure_forward_host_calls=calls,compiler_internal_tracing_calls='unknown')


def common_observations(args,inputs,data,actual):
    directory=args.history/'snn_min'/'other'/f'eval_200000_{actual}'
    manifest=json.loads(inputs.add(directory/'manifest.json').read_text(encoding='utf-8'))
    if (manifest['pool_digest']!=research.DIGEST or manifest['checkpoint_sha256']!=data['checkpoint_hash']):
        raise ValueError('Common trajectory source identity mismatch.')
    records=[json.loads(line) for line in inputs.add(directory/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=manifest['selection'];by_id={r['scenario_id']:r for r in records}
    if len(records)!=20 or len(by_id)!=20 or set(by_id)!={s['scenario_id'] for s in selected}:
        raise ValueError('Require complete identity-matched historical twenty trajectories.')
    # 11 predeclared scenes: first 2 empty + first 9 nonempty, IDs/order only.
    scenes=([s for s in selected if not s['payload']['zones']][:2]
            +[s for s in selected if s['payload']['zones']][:9])
    if len(scenes)!=11:raise ValueError('Insufficient common empty/nonempty scenes.')
    result=[]
    for scene in scenes:
        rows=cal.replay_episode(data,scene,by_id[scene['scenario_id']])
        for phase,i in zip(('start','middle','late'),(0,len(rows)//2,len(rows)-1)):
            result.append(dict(scene=scene,phase=phase,index=i,observation=deepcopy(rows[i]['observation'])))
    # Fixed 32 of 33 before looking at any tested actions.
    return result[:32],scenes


@contextmanager
def action_observer(engine,data):
    """Observe returned tensor of the real call, without a second forward."""
    current={};original=engine._action_inference_tensor
    def train_tensor(batch):
        tensor=original(batch)
        current['train_raw']=tensor.detach().cpu().numpy()[0].copy()
        current['train_context']=dict(inference=torch.is_inference_mode_enabled(),grad=torch.is_grad_enabled())
        return tensor
    def evaluate_hook(module,args,tensor):
        current['eval_raw']=tensor.detach().cpu().numpy()[0].copy()
        current['eval_context']=dict(inference=torch.is_inference_mode_enabled(),grad=torch.is_grad_enabled())
    handle=data['actor'].register_forward_hook(evaluate_hook)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(engine,'_action_inference_tensor',train_tensor))
            if hasattr(data['actor'],'snn_head'):
                head=data['actor'].snn_head;original_head=head._forward_impl
                def observed_head(*a,**kw):
                    raw,first,second=original_head(*a,**kw)
                    current['eval_pulses']={'l1':first.detach().cpu().bool().tolist(),'l2':second.detach().cpu().bool().tolist()}
                    return raw,first,second
                stack.enter_context(patch.object(head,'_forward_impl',observed_head))
            yield current
    finally:handle.remove()


def compare_action(engine,data,obs,current):
    observation_hash=cal.base.digest(cal.base.observation_value(obs))
    current.clear();begin=time.perf_counter()
    a=engine.select_action(obs,exploration_noise=0.,cpu_deduplication=True)
    train_seconds=time.perf_counter()-begin
    if cal.base.digest(cal.base.observation_value(obs))!=observation_hash:raise RuntimeError('Training action mutated common observation.')
    begin=time.perf_counter()
    b=cal.base.actor_action(data,obs);eval_seconds=time.perf_counter()-begin
    if cal.base.digest(cal.base.observation_value(obs))!=observation_hash:raise RuntimeError('Frozen action mutated common observation.')
    if not all(k in current for k in ('train_raw','eval_raw')):raise RuntimeError('Missing same-forward action evidence.')
    return dict(train_raw=current['train_raw'].tolist(),eval_raw=current['eval_raw'].tolist(),
        train_clipped=a.tolist(),eval_clipped=b.tolist(),bitwise=bool(np.array_equal(a,b)),
        raw_bitwise=bool(np.array_equal(current['train_raw'],current['eval_raw'])),
        observer_instrumented_seconds=dict(train=train_seconds,evaluate=eval_seconds),
        passed=bool(np.allclose(a,b,rtol=RTOL,atol=ATOL) and np.allclose(current['train_raw'],current['eval_raw'],rtol=RTOL,atol=ATOL)),
        context={k:current[k] for k in ('train_context','eval_context')},
        training_modes=dict(train=engine.actor.training,evaluate=data['actor'].training),
        pulses=dict(status='partial' if 'eval_pulses' in current else 'missing',evaluate=current.get('eval_pulses'),
            train='missing: compiled action callable exposes no same-forward pulse tensor; no extra forward performed'))


def action_sequences(engine,data,states,current):
    records=[]
    for label,i in zip(('A','A','B','A'),(0,0,1,0)):
        records.append(dict(label=label,**compare_action(engine,data,states[i]['observation'],current)))
    repeat={}
    for side in ('train','eval'):
        values=[np.asarray(records[i][side+'_raw']) for i in (0,1,3)]
        repeat[side]=dict(bitwise=all(np.array_equal(values[0],v) for v in values[1:]),
            passed=all(np.allclose(values[0],v,rtol=RTOL,atol=ATOL) for v in values[1:]))
    return dict(sequence=records,repeat=repeat,passed=all(v['passed'] for v in repeat.values()) and all(v['passed'] for v in records))


def closed_loop(engine,data,scenes,current):
    outputs=[]
    for scene in (next(s for s in scenes if not s['payload']['zones']),next(s for s in scenes if s['payload']['zones'])):
        branches=[]
        for path in ('train','eval'):
            env=cal.base.make_env(data);obs,info=env.reset(options={'scenario':scene['payload']})
            actions=[];raw_actions=[];pulses=[];states=[cal.base.serial(env.state)];rewards=[];started=time.perf_counter()
            while True:
                current.clear()
                action=(engine.select_action(obs,exploration_noise=0.,cpu_deduplication=True) if path=='train' else cal.base.actor_action(data,obs))
                raw_actions.append(current[path+'_raw'].tolist())
                activity=current.get('eval_pulses') if path=='eval' else None
                pulses.append(None if activity is None else dict(sha256=cal.base.digest(activity),
                    active_counts={k:int(np.asarray(v).sum()) for k,v in activity.items()}))
                obs,reward,terminated,truncated,info=env.step(action)
                actions.append(action.tolist());rewards.append(float(reward));states.append(cal.base.serial(env.state))
                if terminated or truncated:break
                if len(actions)>=data['scenario'].max_steps:raise RuntimeError('Missing natural environment termination.')
            branches.append(dict(path=path,actions=actions,raw_actions=raw_actions,pulses=pulses,states=states,outcome=info['outcome'],steps=len(actions),
                terminated=bool(terminated),truncated=bool(truncated),final_info=cal.base.serial(info),
                episode_return=sum(rewards),discounted_return=cal.base.discounted_return(rewards,data['gamma']),seconds=time.perf_counter()-started))
        first=None
        for i,(a,b) in enumerate(zip(branches[0]['actions'],branches[1]['actions'])):
            if a!=b or branches[0]['states'][i+1]!=branches[1]['states'][i+1]:first=i;break
        if first is None and branches[0]['steps']!=branches[1]['steps']:first=min(b['steps'] for b in branches)
        outputs.append(dict(scenario_id=scene['scenario_id'],branches=branches,first_action_or_state_difference=first,
            pulse_difference='unknown: same-forward pulse evidence unavailable',outcomes_equal=branches[0]['outcome']==branches[1]['outcome']))
    return outputs


def inference(args,inputs,output):
    source,raw,_=discover(args,inputs,'snn_min',200000)
    source_data=load_data(source,args,inputs,device='cpu')
    states,scenes=common_observations(args,inputs,source_data,raw['actual_step'])
    _write_json(output/'review/common_states.json',[dict(scenario_id=s['scene']['scenario_id'],phase=s['phase'],step=s['index'],
        observation_sha256=cal.base.digest(cal.base.observation_value(s['observation']))) for s in states])
    results=[];calls=0
    for arm in SNN_ARMS:
        path,raw,sidecar=discover(args,inputs,'snn_'+arm,200000)
        data=load_data(path,args,inputs)
        cal.base.compare_inputs(source_data,data)
        engine,_,_,compiled=make_engine(args,data,raw)
        if cal.state_hash(engine.actor.state_dict())!=cal.state_hash(data['actor'].state_dict()):raise ValueError('Actor weights disagree.')
        if {p.data_ptr() for p in engine.actor.parameters()} & {p.data_ptr() for p in data['actor'].parameters()}:
            raise ValueError('Action paths share mutable model parameters.')
        records=[];weights=cal.state_hash(engine.actor.state_dict())
        with action_observer(engine,data) as current:
            for state in states:
                calls+=2
                if calls>500:raise RuntimeError('Single action budget exceeded.')
                records.append(dict(scene=state['scene']['scenario_id'],phase=state['phase'],**compare_action(engine,data,state['observation'],current)))
            sequence=action_sequences(engine,data,states,current);calls+=8
            loop=closed_loop(engine,data,scenes,current) if args.closed_loop and arm in ('min','wd3_015') else []
        if cal.state_hash(engine.actor.state_dict())!=weights or cal.state_hash(data['actor'].state_dict())!=weights:
            raise RuntimeError('Inference path changed actor weights/buffers.')
        result=dict(arm=arm,requested=raw['requested_step'],actual=sidecar['actual_step'],records=records,sequence=sequence,
            compilation=compiled,closed_loop=loop,passed=all(v['passed'] for v in records) and sequence['passed'] and all(v['outcomes_equal'] for v in loop))
        _write_json(output/f'other/{arm}.json',result);results.append(result)
        inputs.check();engine.invalidate_gpu_update()
        del engine,data
    return dict(strict_passed=all(r['passed'] for r in results),action_calls=calls,closed_loop_requested=args.closed_loop,
        strict_scope='action values/repeat-state tests only; training pulse equivalence unknown',
        first_difference=next((dict(arm=r['arm'],comparison=v) for r in results for v in r['records'] if not v['passed']),None),
        first_repeat_difference=next((dict(arm=r['arm'],repeat=r['sequence']['repeat']) for r in results if not r['sequence']['passed']),None),
        first_closed_loop_outcome_difference=next((dict(arm=r['arm'],scene=v['scenario_id']) for r in results for v in r['closed_loop'] if not v['outcomes_equal']),None),
        coverage='32 fixed historical states, A-A/A-B-A; not a long-term protocol exclusion')


class IsolatedScorer:
    """Eager copies created pre-compile; never toggle live module state/addresses."""
    def __init__(self,copies):
        if any(c.zone_set_encoder.compiled_tensor_forward_enabled for c in copies):
            raise ValueError('Isolated critics must be created before compilation.')
        self.copies=copies

    def __call__(self,engine,rows):
        for live,copy in zip((engine.critic1,engine.critic2),self.copies):
            if {p.data_ptr() for p in live.parameters()} & {p.data_ptr() for p in copy.parameters()}:
                raise ValueError('Isolated scorer shares live parameter storage.')
            copy.load_state_dict(live.state_dict(),strict=True)
        return predict_bank(SimpleNamespace(critic1=self.copies[0],critic2=self.copies[1],device=engine.device),rows)


def bank_episode(hook,engine,episode,stage,scorer):
    """Real OnlineResearch.episode -> complete_episode -> predict_bank -> adjust."""
    scores=[];started=time.perf_counter()
    def observed(live,rows):
        from brain_uav.scripts.profile_v2_td3 import _dynamo_unique_graph_count
        graphs=_dynamo_unique_graph_count()
        begin=time.perf_counter()
        q=scorer(live,rows);scores.append(dict(rows=len(rows),tail=len(rows)%64,Q=q.tolist(),
            wall_seconds_including_state_audit=time.perf_counter()-begin,
            dynamo_new_graphs=_dynamo_unique_graph_count()-graphs,
            lazy_compile_seconds='unknown: included in this score wall time, not separately instrumented'))
        return q
    rows=[SimpleNamespace(observation=deepcopy(r['observation']),action=r['action'].copy(),reward=r['reward'],
        collected_stage_step=stage-len(episode)+1+i,done=r['done']) for i,r in enumerate(episode)]
    trainer=SimpleNamespace(engine=engine,result=SimpleNamespace(stage_steps=stage))
    # Only dispatch the diagnostic scorer; the real hook and controller remain intact.
    with patch('brain_uav.trainers.v2_online_research.predict_bank',side_effect=observed):hook.episode(trainer,rows)
    return dict(scores=scores,control=deepcopy(hook.acc.controls[-1]) if hook.acc.controls else None,
        beta=engine.research_beta,seconds=time.perf_counter()-started,
        synthetic_stage_steps=stage,source_episode_steps=[episode[0]['step'],episode[-1]['step']])


def protected_score(scorer,engine,rows):
    before=training_identity(engine)
    q=scorer(engine,rows)
    if training_identity(engine)!=before:raise RuntimeError('Bank scoring polluted live state/gradients/Adam/targets/RNG.')
    if np.shape(q)!=(len(rows),2) or not np.isfinite(q).all():raise ValueError('Nonfinite/invalid bank Q evidence.')
    return q


def replay_batch(engine,episode):
    from brain_uav.observations.v2_batch import collate_v2_observations
    from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch
    # Derive nonterminal next obs from adjacent bank entries. Terminal next obs
    # was not saved: use a declared placeholder (done=1 masks its target).
    rows=[episode[i%len(episode)] for i in range(64)]
    following=[episode[min(i%len(episode)+1,len(episode)-1)] for i in range(64)]
    obs=collate_v2_observations([r['observation'] for r in rows])
    return V2ReplayBatch(obs,torch.as_tensor(np.stack([r['action'] for r in rows])),
        torch.tensor([[r['reward']] for r in rows],dtype=torch.float32),
        collate_v2_observations([r['observation'] for r in following]),torch.tensor([[float(r['done'])] for r in rows],dtype=torch.float32),
        torch.zeros(64,1),torch.zeros(64,1),torch.ones(64,1))


def beta_target_check(evidence,batch,gamma,beta_old,beta_new):
    """Same-update recorded Q/TD only; no extra forward or relaxed tolerance."""
    from brain_uav.trainers.v2_td3 import research_target_pair
    q1,q2,target=(evidence[name] for name in ('target.q1','target.q2','target.td'))
    if not all(torch.isfinite(t).all() for t in (q1,q2,target)):
        raise ValueError('Nonfinite beta target evidence.')
    reward,done=batch.reward.cpu(),batch.done.cpu()
    expected=reward+(1-done)*gamma*research_target_pair(q1,q2,'acc',beta_new)
    old=reward+(1-done)*gamma*research_target_pair(q1,q2,'acc',beta_old)
    matches=[bool(torch.allclose(target[:,k],expected[:,k],rtol=RTOL,atol=ATOL)) for k in range(2)]
    # The smaller-Q critic is beta-invariant by definition. Both must match,
    # but at least one critic must distinguish old/new on the same target Q.
    distinguishable=[not bool(torch.allclose(expected[:,k],old[:,k],rtol=RTOL,atol=ATOL)) for k in range(2)]
    sufficient=beta_old!=beta_new and any(distinguishable)
    return dict(strict_passed=sufficient and all(matches),information_sufficient=sufficient,
        status='information_insufficient' if not sufficient else 'passed' if all(matches) else 'failed',
        runtime_target_matches=matches,beta_refresh_distinguishable_on_same_target_q=distinguishable,
        beta_old=beta_old,beta_new=beta_new,gamma=gamma,reward=reward.tolist(),done=done.tolist(),
        target_Q1=q1.tolist(),target_Q2=q2.tolist(),
        actual_td_target=target.tolist(),expected_new=expected.tolist(),expected_old=old.tolist())


def bank_worker(args,inputs,output):
    from brain_uav.scripts.profile_v2_td3 import _dynamo_unique_graph_count
    from brain_uav.trainers.v2_gpu_compute import GPUCompute
    from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
    from brain_uav.scripts.v2_gpu_td3_diagnostic import snapshot,flatten
    # Historical ANN bank is a traceable independent fixture for SNN, not SNN history.
    bank_path,bank_raw,_=discover(args,inputs,'ann_acc_001',75000)
    model=args.worker_model
    path,raw,_=(bank_path,deepcopy(bank_raw),None) if model=='ann' else discover(args,inputs,'snn_min',200000)
    data=load_data(path,args,inputs)
    original_raw=deepcopy(raw)
    if model=='snn':
        raw=deepcopy(raw);raw['arm']='acc_001';raw['target_config']=research.target_config('acc_001')
        raw['snapshot']['engine_checkpoint']['algorithm_config']['online_research_target']=raw['target_config']
        raw['snapshot']['engine_checkpoint']['online_research_beta']=.5
    # Independent, preregistered controller starts beta=.5 in both arms. Not resume.
    raw['snapshot']['engine_checkpoint']['online_research_beta']=.5
    beta_trial=args.bank_path=='beta'
    limit=100 if beta_trial else 200
    physical=[0]
    original_compute=GPUCompute.compute
    original_update=V2TD3UpdateEngine.update_once
    def reserve():
        if physical[0]>=limit:raise RuntimeError('Per-process physical update budget exceeded.')
        physical[0]+=1
    def counted(self,*a,**kw):
        reserve()
        return original_compute(self,*a,**kw)
    def counted_update(self,*a,**kw):
        reserve()  # includes any state-changing prewarm, public update/replay
        return original_update(self,*a,**kw)
    with patch.object(GPUCompute,'compute',counted),patch.object(V2TD3UpdateEngine,'update_once',counted_update):
        engine,copies,config,compiled=make_engine(args,data,raw)
        initial=cal.state_hash(engine.checkpoint_state_dict())
        hook=OnlineResearch('acc_001',output/'other/hook')
        candidate=IsolatedScorer(copies)
        scorer=candidate if args.bank_path=='isolated' else predict_bank
        episodes=bank_raw['controller']['episodes']
        if not episodes:raise ValueError('Historical complete ACC bank missing.')
        for key in ('scenario_config','reward_config','uav_collision_radius'):
            if bank_raw['snapshot'][key]!=original_raw['snapshot'][key]:raise ValueError('Bank/source observation semantics mismatch.')
        records=[];stage=25000-max(5001,len(episodes[1%len(episodes)]))
        if stage%2==0:stage-=1
        if stage<len(episodes[0]):raise ValueError('Bank too long for pre-freeze diagnostic chronology.')
        for i in range(1 if beta_trial else args.updates_per_worker):
            episode=episodes[i%len(episodes)]
            if beta_trial or i==1:stage=25000
            elif i>1:stage+=max(1000,len(episode));stage+=int(i%2!=stage%2)
            _write_json(output/f'other/pre_bank_{i}.json',dict(index=i,synthetic_stage=stage,
                beta=engine.research_beta,source_episode_steps=[episode[0]['step'],episode[-1]['step']],
                initial_state_sha256=initial,physical_update_upper_bound=physical[0]))
            beta_before=engine.research_beta
            record=bank_episode(hook,engine,episode,stage,lambda live,rows:protected_score(scorer,live,rows))
            # Repeated real scores before update test graph/encoder alternation.
            if hook.acc.episodes and not beta_trial:
                flat=[r for e in hook.acc.episodes for r in e]
                graphs=_dynamo_unique_graph_count();score_started=time.perf_counter()
                repeated=protected_score(scorer,engine,flat)
                record['repeat_score_wall_seconds_including_audit']=time.perf_counter()-score_started
                record['repeat_score_dynamo_new_graphs']=_dynamo_unique_graph_count()-graphs
                record['repeat_Q_sha256']=cal.state_hash(torch.as_tensor(repeated))
                record['repeat_Q']=repeated.tolist()
                record['repeated_score_rows']=len(flat)
                record['repeated_score_tail']=len(flat)%64
            batch=replay_batch(engine,episode)
            noise=torch.randn(batch.action.shape,generator=torch.Generator().manual_seed(23+i)).to(engine.device)
            _,pn,nc=config.noise_schedule.values(stage-1,max_steps=500000)
            engine.set_target_noise(policy_noise=pn,noise_clip=nc)
            captures=0 if engine._gpu_update_executor is None else engine._gpu_update_executor.captures
            _write_json(output/f'other/pre_update_{i}.json',dict(index=i,stage=stage,beta=engine.research_beta,
                bank_rows=sum(map(len,hook.acc.episodes)),captures=captures))
            update_started=time.perf_counter()
            with patch.object(engine.replay,'sample',return_value=batch),patch('torch.randn_like',new=lambda x:noise.to(x.device)):
                if beta_trial:
                    evidence={}
                    def record_target(name,value):
                        if name in ('target.q1','target.q2','target.td'):evidence[name]=value.detach().cpu().clone()
                    metric=engine.update_once(total_steps=stage,bc_lambda=_bc(stage),diagnostic_tensor_recorder=record_target)
                else:metric=engine.update_once(total_steps=stage,bc_lambda=_bc(stage))
            record['update_wall_seconds']=time.perf_counter()-update_started
            if beta_trial:
                from brain_uav.trainers.v2_gpu_compute import tensors
                check=beta_target_check(evidence,batch,config.gamma,beta_before,engine.research_beta)
                check.update(control=record['control'],bank_sha256=inputs.files[str(bank_path.resolve())],
                    bank_episode_sha256=cal.state_hash(episode),bank_source=str(bank_path),
                    batch_sha256=cal.state_hash(tuple(tensors(batch))),noise_sha256=cal.state_hash(noise),
                    synthetic_stage_steps=stage,graph=engine._gpu_update_executor.metadata(),
                    update_wall_seconds=record['update_wall_seconds'],
                    physical_updates_upper_bound=physical[0],physical_update_budget=limit,
                    evidence_scope='independent diagnostic Graph; same-update Q/target; not main crash reproduction')
                if record['control'] is None or engine._gpu_update_executor.replays<=0:
                    raise RuntimeError('Missing real controller adjustment or Graph replay.')
                _write_json(output/'other/beta_target_check.json',check)
                inputs.check();engine.invalidate_gpu_update()
                return dict(strict_passed=check['strict_passed'],diagnostic_valid=check['strict_passed'],
                    beta_check=check,controlled_updates=1,physical_updates_upper_bound=physical[0],
                    compilation=compiled,source_actual_step=original_raw['actual_step'],
                    fixture_is_snn_history=False,controller_reinitialized=True)
            values,_=snapshot(engine)
            flatten(asdict(metric),'metrics',values)
            for name in ('actor','critic1','critic2'):
                for key,p in getattr(engine,name).named_parameters():flatten(p.grad,'gradient.'+name+'.'+key,values)
            torch.save(values,output/f'other/state_{i}.pt')
            record.update(index=i,metric=asdict(metric),mode='critic_only' if stage%2 else 'frozen_targets' if stage<=25000 else 'actor',
                graph=engine._gpu_update_executor.metadata(),state_path=f'other/state_{i}.pt',
                real_compute_statistics=dict(engine._online_research_last_statistics))
            from brain_uav.trainers.v2_gpu_compute import tensors
            record.update(batch_sha256=cal.state_hash(tuple(tensors(batch))),noise_sha256=cal.state_hash(noise))
            records.append(record)
            _write_json(output/f'other/record_{i}.json',record)
        inputs.check()
        if engine._gpu_update_executor.replays<=0:raise RuntimeError('No actual GPUCompute Graph replay evidence.')
        coverage=dict(modes=sorted({r['mode'] for r in records}),bank_sizes=sorted({r['repeated_score_rows'] for r in records}),
            has_tail=any(r['repeated_score_tail'] for r in records))
        diagnostic_valid=(len(coverage['modes'])==3 and len(coverage['bank_sizes'])>1 and coverage['has_tail'])
        result=dict(strict_passed=None,diagnostic_valid=diagnostic_valid,reproduction='no CUDA exception within this budget',candidate_not_accepted=True,
            bank_source=str(bank_path),bank_sha256=inputs.files[str(bank_path.resolve())],
            source_actual_step=original_raw['actual_step'],synthetic_counter_and_replay=True,
            fixture_is_snn_history=False,source_arm=original_raw['arm'],treatment='acc_001',controller_reinitialized=True,
            initial_state_sha256=initial,records=records,compilation=compiled,coverage=coverage,
            controlled_updates=len(records),physical_updates_upper_bound=physical[0],physical_update_budget=limit,
            graph=engine._gpu_update_executor.metadata())
        engine.invalidate_gpu_update();return result


def _bc(stage):
    from brain_uav.v2_curriculum import v2_bc_lambda
    return v2_bc_lambda(stage-1,schedule='smooth-mid-strong')


def compare_bank_workers(args,output,inputs=None):
    from brain_uav.scripts.diagnose_v2_mc_return_cuda import compare_snapshots
    results=[]
    for model in ('ann','snn'):
        summaries=[]
        for path in ('original','isolated','beta'):
            root=output/f'other/{model}_{path}'
            cmd=[sys.executable,'-u','-B','-m','brain_uav.scripts.diagnose_v2_online_followup','acc-bank','--launch',
                '--worker-model',model,'--bank-path',path]
            for option in ('history','historical_repo','validation_pool','ann_bc','snn_bc','source_manifest','updates_per_worker'):
                cmd.extend(['--'+option.replace('_','-'),str(getattr(args,option))])
            cmd.extend(['--output-dir',str(root)])
            with (output/f'review/{model}_{path}.log').open('x',encoding='utf-8') as log:
                # No recovery in a poisoned CUDA context; subprocess exits on the first exception.
                started=time.perf_counter();status=dict(command=cmd,exit_code=None,timed_out=False,
                    evidence_scope='beta target' if path=='beta' else 'main reproduction',
                    physical_update_budget=100 if path=='beta' else 200)
                try:
                    proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=args.worker_timeout)
                    status['exit_code']=proc.returncode
                except subprocess.TimeoutExpired:
                    status['timed_out']=True;raise
                finally:
                    status['process_wall_seconds']=time.perf_counter()-started
                    _write_json(output/f'review/{model}_{path}_process.json',status)
            document=json.loads((root/'summary.json').read_text()) if path=='beta' else None
            if path=='beta':
                result=document.get('result',{})
                check=result.get('beta_check',{})
                reason=check.get('status')
                # rc=1 alone is not numerical evidence. Require the complete,
                # internally consistent negative-result contract before classifying.
                negative=(type(proc.returncode) is int and proc.returncode==1
                    and type(document.get('exit_code')) is int and document['exit_code']==1
                    and document.get('diagnostic_only') is True and not document.get('error')
                    and document.get('execution_completed') is True and document.get('strict_passed') is False
                    and result.get('strict_passed') is False and result.get('diagnostic_valid') is False
                    and check.get('strict_passed') is False
                    and reason in ('failed','information_insufficient')
                    and check.get('information_sufficient') is (reason=='failed'))
                if negative:
                    raise StrictMismatch(f'{model}/beta {reason}; acceptance blocked. Evidence: {root}/summary.json')
                if result.get('strict_passed') is not True:
                    raise RuntimeError(f'{model}/beta invalid/incomplete negative-result contract rc={proc.returncode}; further trials blocked.')
            if proc.returncode:raise RuntimeError(f'{model}/{path} failed rc={proc.returncode}; further trials blocked.')
            if document is None:document=json.loads((root/'summary.json').read_text())
            if not document['execution_completed'] or not document['result']['diagnostic_valid']:
                raise RuntimeError('Worker incomplete or required bank/mode/tail coverage missing.')
            if inputs is not None:
                used=json.loads((root/'review/used_inputs_final.json').read_text())
                for name,h in used['files'].items():
                    if name in inputs.files and inputs.files[name]!=h:raise ValueError('Workers used different input identities.')
                    inputs.files[name]=h
                inputs.check()
            summaries.append(document['result'])
        a,b,beta=summaries
        if beta.get('strict_passed') is not True:raise StrictMismatch('Independent beta target check did not pass.')
        if a['initial_state_sha256']!=b['initial_state_sha256'] or a['bank_sha256']!=b['bank_sha256']:
            raise ValueError('Worker initial state/bank mismatch.')
        checks=[]
        for i,(ra,rb) in enumerate(zip(a['records'],b['records'])):
            q=bool(np.allclose(ra['repeat_Q'],rb['repeat_Q'],rtol=RTOL,atol=ATOL))
            if ra['scores']:
                q &=bool(np.allclose(ra['scores'][0]['Q'],rb['scores'][0]['Q'],rtol=RTOL,atol=ATOL))
            if ra['batch_sha256']!=rb['batch_sha256'] or ra['noise_sha256']!=rb['noise_sha256']:
                raise ValueError('Controlled batch/noise identity mismatch.')
            from brain_uav.scripts.v2_gpu_td3_diagnostic import flatten
            ca={};cb={};flatten(ra['control'],'control',ca);flatten(rb['control'],'control',cb)
            control=compare_snapshots(ca,cb)['passed']
            state=compare_snapshots(torch.load(output/f'other/{model}_original/other/state_{i}.pt',weights_only=False),
                torch.load(output/f'other/{model}_isolated/other/state_{i}.pt',weights_only=False))
            checks.append({**state,'index':i,'Q_passed':q,'controller_passed':control,'passed':q and control and state['passed']})
        results.append(dict(model=model,checks=checks,beta_check=beta,
            main_reproduction_completed=True,numerical_comparison_passed=all(c['passed'] for c in checks),
            physical_updates_upper_bound=sum(s['physical_updates_upper_bound'] for s in summaries),
            strict_passed=all(c['passed'] for c in checks)))
        _write_json(output/f'review/{model}_comparison.json',results[-1])
        if not results[-1]['strict_passed']:raise StrictMismatch('Strict original/candidate mismatch; subsequent model blocked.')
    return dict(strict_passed=all(r['strict_passed'] for r in results),results=results,
        interpretation='本预算内未复现不等于排除根因；候选未取得正式修复验收',controlled_updates=4*args.updates_per_worker+2,
        budgets=dict(main_reproduction=800,beta_check=200,total=1000),
        evidence_scopes=['main reproduction','original/candidate numerical comparison','independent beta target check'])


def execute(args):
    plan(args);guard_locations(args)
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    (output/'review').mkdir();(output/'other').mkdir()
    summary=dict(diagnostic_only=True,execution_completed=False,strict_passed=None,exit_code=1,
        physical_update_budgets=dict(main_reproduction=800,beta_check=200,total=1000) if args.mode=='acc-bank' else None)
    inputs=None
    started=time.perf_counter()
    try:
        inputs=Inputs(args)
        _write_json(output/'manifest.json',dict(plan=plan(args),**inputs.evidence()))
        if args.mode=='snn-inference':result=inference(args,inputs,output)
        elif args.worker_model:result=bank_worker(args,inputs,output)
        else:result=compare_bank_workers(args,output,inputs)
        inputs.check();_write_json(output/'review/used_inputs.json',inputs.evidence())
        summary.update(result=result,execution_completed=True,strict_passed=result['strict_passed'],
            exit_code=1 if result['strict_passed'] is False else 0)
    except BaseException as exc:
        if isinstance(exc,StrictMismatch):summary['strict_passed']=False
        summary.update(error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if inputs is not None:_write_json(output/'review/used_inputs_final.json',inputs.evidence())
        summary['process_wall_seconds']=time.perf_counter()-started
        _write_json(output/'summary.json',summary)
    return summary


def main(argv=None):
    args=parser().parse_args(argv);description=plan(args)
    if not args.launch:print(json.dumps(description,indent=2));return 0
    try:
        result=execute(args)
        return 0 if result['strict_passed'] is not False else 1
    except BaseException:
        import traceback
        traceback.print_exc();return 1


if __name__=='__main__':raise SystemExit(main())
