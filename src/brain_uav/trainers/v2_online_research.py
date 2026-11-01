"""Explicit online research hooks; absent from all default training calls."""
from collections import deque
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

ARMS = ('min','mean','wd3_015','wd3_045','wd3_075','fixed_own','acc_001','acc_010')
FORMAT = 'v2_online_wd3_acc_research_snapshot'


def target_config(arm):
    if arm not in ARMS: raise ValueError('Unknown predeclared research arm.')
    if arm in ('min','mean'): return None
    if arm.startswith('wd3_'):
        return dict(method='wd3',weight={'wd3_015':.15,'wd3_045':.45,'wd3_075':.75}[arm],initial_beta=.5)
    return dict(method='acc',weight=None,initial_beta=1. if arm=='fixed_own' else .5)


def validate_target_config(config):
    if not isinstance(config,dict) or set(config)!={'method','weight','initial_beta'}:
        raise ValueError('Malformed research target configuration.')
    if config['method'] not in ('wd3','acc'): raise ValueError('Unknown research target method.')
    if type(config['initial_beta']) not in (int,float) or not math.isfinite(config['initial_beta']) or not 0<=config['initial_beta']<=1:
        raise ValueError('Invalid initial beta.')
    if config['method']=='wd3':
        if type(config['weight']) not in (int,float) or not math.isfinite(config['weight']) or not 0<=config['weight']<=1:
            raise ValueError('Invalid WD3 weight.')
    elif config['weight'] is not None: raise ValueError('ACC does not use a WD3 weight.')
    return dict(config)


class ACCController:
    def __init__(self, eta, *, gamma=.99, capacity=5000):
        if eta not in (.01,.1) or not 0<gamma<=1 or capacity<1: raise ValueError('Invalid ACC controller contract.')
        self.eta,self.gamma,self.capacity=eta,gamma,capacity
        self.beta=.5;self.ema=None;self.last_adjust=None
        self.episodes=deque();self.controls=[];self.first_boundary=None

    def complete_episode(self, rows, completed_step, source):
        if source!='training' or not rows: raise ValueError('ACC accepts complete training episodes only.')
        if not rows[-1]['done'] or any(r['done'] for r in rows[:-1]): raise ValueError('Incomplete/invalid episode done boundary.')
        from .v2_mc_return import discounted_episode_returns
        gs=discounted_episode_returns([r['reward'] for r in rows],self.gamma)
        episode=[dict(deepcopy(r),G=float(g),completed_step=completed_step) for r,g in zip(rows,gs)]
        self.episodes.append(episode)
        while len(self.episodes)>1 and sum(map(len,self.episodes))>self.capacity: self.episodes.popleft()

    def adjust(self, stage_steps, predict):
        if stage_steps<25000 or not self.episodes or (self.last_adjust is not None and stage_steps-self.last_adjust<1000): return None
        rows=[r for episode in self.episodes for r in episode]
        q=np.asarray(predict(rows),dtype=np.float64)
        g=np.asarray([r['G'] for r in rows],dtype=np.float64)
        if q.shape!=(len(rows),2) or not np.isfinite(q).all() or not np.isfinite(g).all():
            raise FloatingPointError('Invalid ACC feedback values.')
        residual=g-q.mean(axis=1);c=float(residual.mean());a=float(abs(residual).mean())
        ema=a if self.ema is None else .95*self.ema+.05*a
        scale=max(1.,ema);u=float(np.clip(c/scale,-1,1))
        beta=float(np.clip(self.beta+self.eta*u,0,1))
        ages=[stage_steps-r['step'] for r in rows]
        record=dict(stage_steps=stage_steps,beta_before=self.beta,beta_after=beta,C=c,A=a,EMA=ema,S=scale,u=u,
            interval=None if self.last_adjust is None else stage_steps-self.last_adjust,
            transitions=len(rows),episodes=len(self.episodes),age_min=min(ages),age_max=max(ages),
            episode_boundaries=[dict(first=e[0]['step'],last=e[-1]['step'],length=len(e)) for e in self.episodes])
        self.beta,self.ema,self.last_adjust=beta,ema,stage_steps
        self.controls.append(record)
        if beta in (0.,1.) and self.first_boundary is None: self.first_boundary=stage_steps
        return record

    def state(self):
        return dict(eta=self.eta,gamma=self.gamma,capacity=self.capacity,beta=self.beta,ema=self.ema,
            last_adjust=self.last_adjust,first_boundary=self.first_boundary,controls=self.controls,
            boundary_fraction=sum(c['beta_after'] in (0.,1.) for c in self.controls)/max(1,len(self.controls)),
            episodes=list(self.episodes))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def predict_bank(engine, rows):
    """No actor forward, backward, optimizer step or sampling; actual stored (s,a)."""
    from brain_uav.observations.v2_batch import collate_v2_observations
    result=[]
    with torch.no_grad():
        for start in range(0,len(rows),64):
            selected=rows[start:start+64]
            obs=collate_v2_observations([r['observation'] for r in selected],device=engine.device)
            action=torch.as_tensor(np.stack([r['action'] for r in selected]),device=engine.device)
            result.extend(torch.stack([c(obs,action).reshape(-1) for c in (engine.critic1,engine.critic2)],1).cpu().tolist())
    return np.asarray(result)


class OnlineResearch:
    """In-process pause keeps the trainer stack/replay/RNG/Adam/CUDA alive."""
    hard_limit=400000
    pause_step=200000

    def __init__(self, arm, directory, *, sleep=time.sleep, clock=time.perf_counter):
        if arm not in ARMS: raise ValueError('Unknown arm.')
        self.arm=arm;self.directory=Path(directory);self.sleep=sleep;self.clock=clock
        self.acc=ACCController(.01 if arm=='acc_001' else .1) if arm.startswith('acc_') else None
        self.paused_once=False;self.times=dict(acc_seconds=0.,pause_seconds=0.,evaluation_seconds=0.)
        self.forbidden_scene_keys=set()
        self.update_statistics=[]

    def status(self,state,**extra):
        record=dict(state=state,pid=os.getpid(),arm=self.arm,time=time.time(),times=self.times,**extra)
        atomic_json(self.directory/'review/progress.json',record)
        path=self.directory/'other/progress_history.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')

    def episode(self, trainer, transitions):
        if self.acc is None: return  # no bank allocation, G or extra forward for other arms
        started=self.clock()
        rows=[dict(observation=t.observation,action=t.action.copy(),reward=t.reward,step=t.collected_stage_step,done=t.done) for t in transitions]
        self.acc.complete_episode(rows,trainer.result.stage_steps,'training')
        record=self.acc.adjust(trainer.result.stage_steps,lambda bank:predict_bank(trainer.engine,bank))
        if record is not None:
            trainer.engine.set_research_beta(self.acc.beta)
            path=self.directory/'other/controls.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a',encoding='utf-8') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
        self.times['acc_seconds']+=self.clock()-started

    def scene(self,payload,step):
        from brain_uav.scripts.diagnose_v2_wd3_acc import scene_key
        from brain_uav.scripts.diagnose_v2_critic import serial
        key=scene_key(payload)
        if key in self.forbidden_scene_keys: raise ValueError('Development/validation scene cannot enter online training or ACC.')
        path=self.directory/'other/training_scenes.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(start_step=step,physical_sha256=key,payload=serial(payload)),allow_nan=False)+'\n')

    def episode_metrics(self,metrics,step):
        if not metrics:return
        # Compact episode aggregates, no extra network forward/backward.
        fields=('critic_loss','critic_td_error_mean','critic_target_q_mean','actor_rl_scale',
            'rl_actor_loss','scaled_rl_actor_loss','bc_loss','actor_grad_norm')
        record={k:float(np.mean([getattr(m,k) for m in metrics if m.actor_updated or not k.startswith(('actor','rl_','scaled_','bc_'))]))
                if any(m.actor_updated or not k.startswith(('actor','rl_','scaled_','bc_')) for m in metrics) else None for k in fields}
        if self.update_statistics:
            record['real_compute_statistics']={k:dict(mean=float(np.mean([r[k] for r in self.update_statistics])),
                min=float(min(r[k] for r in self.update_statistics)),max=float(max(r[k] for r in self.update_statistics))) for k in self.update_statistics[0]}
            record['critic_clip_trigger_fraction']=float(np.mean([r['critic_clipped'] for r in self.update_statistics]))
        self.update_statistics.clear()
        path=self.directory/'other/update_statistics.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(stage_steps=step,**record),allow_nan=False)+'\n')

    def update(self,metric,engine):
        values=getattr(engine,'_online_research_last_statistics',None)
        if values is not None:
            if not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('Nonfinite research update statistics.')
            self.update_statistics.append(dict(values,critic_clipped=float(engine.critic_grad_clip_norm is not None and values['critic_grad_norm']>engine.critic_grad_clip_norm)))

    def boundary(self,trainer):
        step=trainer.result.stage_steps
        if self.paused_once or step<self.pause_step or step>=self.hard_limit:
            self.status('running',stage_steps=step)
            return
        self.paused_once=True
        self.status('paused_at_200k',requested_step=self.pause_step,actual_step=step)
        started=self.clock(); token=self.directory/'continue_to_400k.json'
        while not token.exists(): self.sleep(1.)
        approval=json.loads(token.read_text(encoding='utf-8'))
        if approval!=dict(pid=os.getpid(),arm=self.arm,actual_step=step,continue_to=400000):
            raise ValueError('Continuation approval must identify this live paused process.')
        self.times['pause_seconds']+=self.clock()-started
        self.status('running',stage_steps=step,continued=True)


def validate_snapshot(raw, **expected):
    """Explicit research format; return strict policy view only for diagnostic loaders."""
    from .v2_formal_training import load_v2_periodic_snapshot
    if set(raw)!={'format','format_version','diagnostic_only','formal_passed','arm','target_config','beta','controller','snapshot','requested_step','actual_step'}:
        raise ValueError('Unexpected research snapshot fields.')
    if raw['format']!=FORMAT or raw['format_version']!=1 or raw['diagnostic_only'] is not True or raw['formal_passed'] is not False:
        raise ValueError('Invalid research snapshot identity.')
    config=target_config(raw['arm'])
    if config!=raw['target_config']: raise ValueError('Research target arm/config mismatch.')
    beta=raw['beta']
    if type(beta) not in (int,float) or not math.isfinite(beta) or not 0<=beta<=1: raise ValueError('Invalid research beta.')
    if config is None or raw['arm']=='fixed_own' or raw['arm'].startswith('wd3_'):
        if raw['controller'] is not None: raise ValueError('Nonadaptive arm cannot have ACC state.')
        if beta!=(1. if raw['arm']=='fixed_own' else .5): raise ValueError('Fixed beta mismatch.')
    else:
        state=raw['controller']
        if (not isinstance(state,dict) or set(state)!={'eta','gamma','capacity','beta','ema','last_adjust','first_boundary','controls','boundary_fraction','episodes'}
                or state['beta']!=beta or state['eta']!=(.01 if raw['arm']=='acc_001' else .1)
                or state['gamma']!=.99 or state['capacity']!=5000):
            raise ValueError('Controller/beta/arm mismatch.')
        from brain_uav.scripts.diagnose_v2_critic_calibration import finite_tree
        finite_tree(state)
        previous_beta=.5;previous_step=None;previous_ema=None
        for c in state['controls']:
            step=c['stage_steps']
            if step<25000 or step>raw['actual_step'] or (previous_step is not None and step-previous_step<1000):
                raise ValueError('Invalid ACC update interval.')
            ema=c['A'] if previous_ema is None else .95*previous_ema+.05*c['A']
            scale=max(1.,ema);u=float(np.clip(c['C']/scale,-1,1))
            if (c['beta_before']!=previous_beta or c['EMA']!=ema or c['S']!=scale or c['u']!=u
                    or c['beta_after']!=float(np.clip(previous_beta+state['eta']*u,0,1))):
                raise ValueError('ACC control trajectory algebra mismatch.')
            previous_beta,previous_step,previous_ema=c['beta_after'],step,ema
        if state['beta']!=previous_beta or state['last_adjust']!=previous_step or state['ema']!=previous_ema:
            raise ValueError('ACC final state/trajectory mismatch.')
        first=next((c['stage_steps'] for c in state['controls'] if c['beta_after'] in (0.,1.)),None)
        fraction=sum(c['beta_after'] in (0.,1.) for c in state['controls'])/max(1,len(state['controls']))
        if state['first_boundary']!=first or state['boundary_fraction']!=fraction:raise ValueError('ACC boundary evidence mismatch.')
        episodes=state['episodes']
        from .v2_mc_return import discounted_episode_returns
        previous_end=None
        for episode in episodes:
            if not episode or not episode[-1]['done'] or any(r['done'] for r in episode[:-1]):raise ValueError('Invalid ACC bank episode.')
            steps=[r['step'] for r in episode]
            if (any(type(step) is not int for step in steps)
                    or any(b!=a+1 for a,b in zip(steps,steps[1:]))
                    or (previous_end is not None and steps[0]<=previous_end)):
                raise ValueError('Invalid ACC bank episode chronology.')
            previous_end=steps[-1]
            # Same Python-float recurrence as bank creation; no storage rounding occurs.
            returns=discounted_episode_returns([r['reward'] for r in episode],state['gamma'])
            for r,g in zip(episode,returns):
                if not 0<r['step']<=raw['actual_step'] or not np.isfinite([r['reward'],r['G']]).all():
                    raise ValueError('Invalid ACC bank values/age.')
                if (type(r['completed_step']) is not int or r['completed_step']!=steps[-1]
                        or r['completed_step']>raw['actual_step'] or r['G']!=g):
                    raise ValueError('ACC bank return/completion mismatch.')
                if r['action'].dtype!=np.float32 or r['action'].shape!=(2,) or not np.isfinite(r['action']).all():
                    raise ValueError('Invalid actual bank action.')
        if len(episodes)>1 and sum(map(len,episodes))>5000:raise ValueError('ACC bank exceeds complete-episode capacity policy.')
    payload=deepcopy(raw['snapshot'])
    if payload.pop('research_only',None) is not True:raise ValueError('Missing explicit inner research snapshot identity.')
    algorithm=payload['engine_checkpoint']['algorithm_config']
    if algorithm.pop('online_research_target',None)!=config: raise ValueError('Engine research target mismatch.')
    if config is None and 'online_research_beta' in payload['engine_checkpoint']:raise ValueError('Unexpected beta on min/mean engine.')
    saved_beta=payload['engine_checkpoint'].pop('online_research_beta',.5)
    if saved_beta!=beta: raise ValueError('Engine research beta mismatch.')
    if algorithm.get('mc_return_weight',0.)!=0: raise ValueError('MC cannot be used in this experiment.')
    if algorithm.get('target_q_aggregation','min')!=('mean' if raw['arm']=='mean' else 'min'):
        raise ValueError('Underlying research aggregation/arm mismatch.')
    if payload['stage_steps']!=raw['actual_step'] or not 0<=raw['requested_step']<=raw['actual_step']<=400000:
        raise ValueError('Invalid research snapshot step.')
    validated=load_v2_periodic_snapshot(payload,**expected)
    validated['diagnostic_research']=dict(arm=raw['arm'],beta=beta,target_config=config)
    return validated


def training_identity(engine, exploration_rng=None, selector=None):
    """Snapshot evidence at evaluation boundaries, not in steady-state updates."""
    from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash
    models=[getattr(engine,n) for n in ('actor','critic1','critic2','actor_target','critic1_target','critic2_target','bc_reference_actor')]
    value=dict(checkpoint=engine.checkpoint_state_dict(),
        gradients=[p.grad for m in models if m is not None for p in m.parameters()],
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if engine.device.type=='cuda' else [],
                 python=random.getstate(),numpy=(lambda s:(s[0],torch.from_numpy(s[1].astype(np.int64)),*s[2:]))(np.random.get_state()),
                 replay=engine.replay.rng.bit_generator.state),
        replay=dict(length=len(engine.replay),success=engine.replay.success_size),
        exploration=None if exploration_rng is None else exploration_rng.bit_generator.state,
        selector=None if selector is None else selector.rng.bit_generator.state,
        snn=[getattr(node,'v',None) for m in models if m is not None for node in m.modules() if hasattr(node,'v')])
    return state_hash(value)
