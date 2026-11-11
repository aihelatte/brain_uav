"""Explicit easy ACC feedback, independent of research pause/budget hooks."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import math
import torch

from .v2_online_research import ACCController, IsolatedScorer, target_config, validate_acc_controller


class FormalACC:
    def __init__(self, engine, config, directory=None):
        if config.acc_eta != .01 or config.stage != 'easy':raise ValueError('Only explicit easy ACC(.01) is supported.')
        self.controller=ACCController(config.acc_eta,gamma=config.gamma)
        copies=deepcopy((engine.critic1,engine.critic2))
        for c in copies:c.requires_grad_(False)
        self.scorer=IsolatedScorer(copies)
        self.directory=None if directory is None else Path(directory)
        engine.configure_research_targets(target_config('acc_001'))
        engine.formal_acc=self

    def episode(self, trainer, transitions):
        rows=[dict(observation=t.observation,action=t.action.copy(),reward=t.reward,
                   step=t.collected_stage_step,done=t.done) for t in transitions]
        self.controller.complete_episode(rows,trainer.result.stage_steps,'training')
        record=self.controller.adjust(trainer.result.stage_steps,lambda bank:self.scorer(trainer.engine,bank))
        if record is not None:
            trainer.engine.set_research_beta(self.controller.beta)
            if self.directory is not None:
                self.directory.mkdir(parents=True,exist_ok=True)
                with (self.directory/'acc_controls.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record,allow_nan=False)+'\n')


def save_state(engine,config):
    hook=getattr(engine,'formal_acc',None)
    if config.acc_eta is None:
        if hook is not None:raise ValueError('Unexpected formal ACC hook.')
        return None
    if hook is None or hook.controller.beta!=engine.research_beta:raise ValueError('ACC hook/beta mismatch.')
    return dict(method='acc',scoring='isolated',controller=deepcopy(hook.controller.state()))


def validate_state(payload):
    formal=payload['formal_config'];engine=payload['engine_checkpoint'];algorithm=engine['algorithm_config']
    eta=formal.get('acc_eta')
    if eta is None:
        if 'acc_state' in payload or 'online_research_target' in algorithm or 'online_research_beta' in engine:
            raise ValueError('Online research/ACC targets require an explicit compatible formal contract.')
        return
    state=payload.get('acc_state')
    if (eta!=.01 or formal['stage']!='easy' or formal['gamma']!=.99 or formal['mc_return_weight']!=0
            or formal['target_q_aggregation']!='min' or not isinstance(state,dict)
            or set(state)!={'method','scoring','controller'} or state['method']!='acc' or state['scoring']!='isolated'
            or algorithm.get('online_research_target')!=target_config('acc_001')):
        raise ValueError('Invalid formal ACC contract.')
    for name in ('gamma','tau','policy_delay','batch_size','actor_freeze_steps','actor_grad_clip_norm',
                 'critic_grad_clip_norm','actor_rl_scale_alpha','terminal_geo_regularization_enabled',
                 'terminal_geo_radius','terminal_geo_lambda','mc_return_weight','target_q_aggregation','spike_regularization'):
        if formal[name]!=algorithm[name]:
            raise ValueError(f'Formal ACC and engine {name} disagree.')
    step=payload.get('stage_steps',payload.get('training_result',{}).get('stage_steps'))
    if type(step) is not int or not 0<=step<=formal['max_steps']:
        raise ValueError('Invalid formal ACC step.')
    if (not isinstance(state['controller']['episodes'],list) or not isinstance(state['controller']['controls'],list)
            or any(not isinstance(episode,list) for episode in state['controller']['episodes'])):
        raise ValueError('Invalid formal ACC bank/control container.')
    validate_acc_controller(state['controller'],engine.get('online_research_beta'),step,eta)
    from brain_uav.observations import V2Observation
    for episode in state['controller']['episodes']:
        for row in episode:
            obs=row.get('observation')
            if (not isinstance(obs,V2Observation)
                    or any(not np.isfinite(getattr(obs,key)).all() for key in ('ego_features','goal_features','zone_features'))
                    or np.any(row['action']<engine['action_low']) or np.any(row['action']>engine['action_high'])):
                raise ValueError('Invalid formal ACC bank observation/action.')
            V2Observation(obs.ego_features,obs.goal_features,obs.zone_features,obs.presence_mask)


def load_state(engine,config,payload):
    """Explicit strict diagnostic load; not a replay/RNG training resume."""
    from collections import deque
    validate_state(payload)
    if config.acc_eta is None or payload['formal_config']!=config.to_dict():
        raise ValueError('Formal ACC restore configuration mismatch.')
    checkpoint=payload['engine_checkpoint']
    if getattr(engine,'research_target_config',None) not in (None,target_config('acc_001')):
        raise ValueError('Target engine uses a different target method.')
    for name in ('policy_noise','noise_clip'):
        value=checkpoint['algorithm_config'][name]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid saved target noise.')
    expected=engine._algorithm_config()
    expected.update(online_research_target=target_config('acc_001'),
        policy_noise=checkpoint['algorithm_config']['policy_noise'],noise_clip=checkpoint['algorithm_config']['noise_clip'])
    if expected!=checkpoint['algorithm_config']:
        raise ValueError('Target engine algorithm is incompatible with formal ACC checkpoint.')
    # These validators read the live model metadata, without loading or invalidating graphs.
    engine._validated_network_state_targets(checkpoint)
    bc=checkpoint['bc_reference_actor_state_dict']
    if type(checkpoint['bc_reference_present']) is not bool or (bc is None)==checkpoint['bc_reference_present']:
        raise ValueError('Invalid BC reference state.')
    if bc is not None:
        engine._validate_state_dict(engine.actor,bc,name='BC reference')
        engine._validate_state_dict_fixed_buffers(engine.actor,bc,name='BC reference')
    for name in ('actor','critic'):
        optimizer=getattr(engine,name+'_optimizer')
        saved=checkpoint[name+'_optimizer_state_dict']
        groups=saved['param_groups']
        if len(groups)!=len(optimizer.param_groups) or any(len(a['params'])!=len(b['params']) for a,b in zip(groups,optimizer.param_groups)):
            raise ValueError('ACC optimizer parameter layout mismatch.')
        parameters={index:p for saved_group,live_group in zip(groups,optimizer.param_groups)
            for index,p in zip(saved_group['params'],live_group['params'])}
        if len(parameters)!=sum(len(g['params']) for g in groups) or not saved['state'].keys()<=parameters.keys():
            raise ValueError('ACC optimizer parameter identities mismatch.')
        for index,state in saved['state'].items():
            if not {'step','exp_avg','exp_avg_sq'}<=state.keys():raise ValueError('Incomplete ACC Adam state.')
            for key,value in state.items():
                value=torch.as_tensor(value)
                if not torch.isfinite(value).all():raise ValueError('Nonfinite ACC Adam state.')
                if key=='step':
                    if value.numel()!=1 or value.item()<0:raise ValueError('Invalid ACC Adam step.')
                elif value.shape!=parameters[index].shape:
                    raise ValueError('ACC Adam tensor shape mismatch.')
        # Validate optimizer loading on detached storage before touching any live object.
        deepcopy(optimizer).load_state_dict(deepcopy(saved))
    counts=checkpoint['update_counts']
    for name in ('update_count','critic_update_count','critic_target_update_count','actor_update_count','last_total_steps'):
        if type(counts[name]) is not int or counts[name]<0:raise ValueError('Invalid ACC update counts.')
    hook=getattr(engine,'formal_acc',None)
    if hook is not None and (not isinstance(hook,FormalACC) or hook.controller.eta!=config.acc_eta
            or hook.controller.gamma!=config.gamma or hook.controller.capacity!=5000):
        raise ValueError('Target ACC controller configuration mismatch.')
    if hook is None:
        hook=FormalACC(engine,config)
    algorithm=checkpoint['algorithm_config']
    previous_noise=(engine.policy_noise,engine.noise_clip)
    engine.set_target_noise(policy_noise=algorithm['policy_noise'],noise_clip=algorithm['noise_clip'])
    try:
        engine.load_checkpoint_state_dict(checkpoint)
    except BaseException:
        engine.set_target_noise(policy_noise=previous_noise[0],noise_clip=previous_noise[1])
        raise
    state=deepcopy(payload['acc_state']['controller'])
    for key in ('beta','ema','last_adjust','first_boundary','controls'):
        setattr(hook.controller,key,state[key])
    hook.controller.episodes=deque(state['episodes'])
    return hook
