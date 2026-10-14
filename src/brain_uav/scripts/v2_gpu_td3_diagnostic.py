"""Serial baseline/GPU workers using existing fixed-input and timing fixtures.

No formal checkpoint output. Each worker owns one CUDA context; a failed
capture process is discarded. Correctness, not a configuration flag, gates
timing. The suite is invoked from profile_v2_td3.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import gc
import platform
import subprocess
import sys
import traceback
from time import perf_counter
from statistics import mean, stdev
import numpy as np
import torch


REFERENCE_VERSION = 2


def complete_actor_evidence(tensors, model):
    """Baseline optional term -> explicit diagnostic contract, same forward."""
    if model == 'snn' and 'actor.spike_loss' in tensors and 'actor.weighted_spike' not in tensors:
        tensors['actor.weighted_spike'] = (
            tensors['actor.spike_loss'] * tensors['actor.spike_beta']).detach().clone()


def success_reference_indices(replay, ranks, batch_size):
    count = min(int(batch_size * replay.success_batch_fraction), replay.success_size)
    swaps = {}; success = []
    for i in range(count):
        rank = int(ranks[i]); last = replay.success_size-i-1
        if not 0 <= rank <= last: raise ValueError('Invalid controlled success rank.')
        success.append(swaps.get(rank, rank)); swaps[rank] = swaps.get(last, last)
    return np.asarray(success,dtype=np.int64)


class ControlledInputs:
    def __init__(self, draws, success, success_size):
        self.index=0; self.draws=draws; self.success=success; self.success_size=success_size
    def random(self):
        value=float(self.draws[self.index]); self.index+=1; return value
    def choice(self,size,number,replace=False):
        if size!=self.success_size or number!=len(self.success) or replace:
            raise AssertionError('Unexpected success sampling contract.')
        return self.success.copy()


def controlled_cpu_indices(replay,draws,ranks,batch_size):
    success=success_reference_indices(replay,ranks,batch_size)
    original = replay.rng
    try:
        replay.rng=ControlledInputs(draws,success,replay.success_size)
        primary=replay._sample_primary_indices(batch_size-len(success)) if batch_size>len(success) else np.empty(0,np.int64)
    finally: replay.rng = original
    return primary,success


def controlled_cpu_sample(replay, draws, ranks, batch_size):
    """Original dynamic batch builder; exact primary PPSWOR + partial FY.

    Only the diagnostic success-index generator differs from Generator.choice.
    No changes to replay RNG state, weights, or the model's dynamic cropping.
    """
    success=success_reference_indices(replay,ranks,batch_size)
    original=replay.rng; sampler=replay._sample_primary_indices
    override=replay.__dict__.get('_sample_primary_indices')
    selected=[]
    def indices(number):
        primary=sampler(number); selected.append(primary); return primary
    try:
        replay.rng=ControlledInputs(draws,success,replay.success_size)
        replay._sample_primary_indices=indices
        batch=replay.sample(batch_size)
    finally:
        replay.rng=original
        if override is None: del replay._sample_primary_indices
        else: replay._sample_primary_indices=override
    return selected[0] if selected else np.empty(0,np.int64),success,batch


def expected_rows(adapter, primary, success):
    """CPU oracle from authoritative arrays, not the GPU mirror/gather."""
    result = []
    for indices, prefix in ((primary, ''), (success, 'success_')):
        rows = np.zeros((len(indices), adapter.width), np.float32)
        for name in adapter.fields:
            slot = adapter.layout[name][0]
            rows[:, slot] = getattr(adapter.cpu, prefix+name)[indices].reshape(len(indices), slot.stop-slot.start)
        rows[:, adapter.layout['success'][0]] = 1 if prefix else adapter.cpu.success[indices, None]
        rows[:, adapter.layout['failure'][0]] = 0 if prefix else adapter.cpu.failure[indices, None]
        result.append(rows)
    return torch.from_numpy(np.concatenate(result))


def flatten(value, name='', result=None):
    result={} if result is None else result
    if isinstance(value,dict):
        for key,item in value.items(): flatten(item,name+'.'+str(key),result)
    elif isinstance(value,(tuple,list)):
        for i,item in enumerate(value): flatten(item,name+'.'+str(i),result)
    elif value is None:
        result[name+'.none']=torch.tensor(True)
    else:
        result[name]=torch.as_tensor(value).detach().cpu().clone()
    return result


def snapshot(engine, metric=None):
    from brain_uav.scripts.profile_v2_td3 import _td3_gradient_snapshot
    result={}
    for name in ('actor','critic1','critic2','actor_target','critic1_target','critic2_target'):
        flatten(getattr(engine,name).state_dict(),name,result)
    flatten(engine.actor_optimizer.state_dict(),'optimizer.actor',result)
    flatten(engine.critic_optimizer.state_dict(),'optimizer.critic',result)
    flatten({n:getattr(engine,n) for n in ('update_count','critic_update_count',
        'critic_target_update_count','actor_update_count','last_total_steps')},'counts',result)
    if metric is not None:
        gradients,evidence=_td3_gradient_snapshot(engine,actor_backward=metric.actor_updated)
        result.update(gradients); flatten(asdict(metric),'metrics',result)
    else: evidence={}
    return result,evidence


def pack_fixed(adapter,batch):
    rows=torch.zeros(batch.batch_size,adapter.width,device=adapter.device)
    for prefix,obs in (('',batch.obs),('next_',batch.next_obs)):
        for suffix,tensor in (('ego_features',obs.ego_features),('goal_features',obs.goal_features)):
            rows[:,adapter.layout[prefix+suffix][0]]=tensor
        zones=torch.zeros(batch.batch_size,adapter.cpu.zone_storage_capacity,obs.zone_features.shape[-1],device=adapter.device)
        zones[:,:obs.zone_features.shape[1]]=obs.zone_features
        rows[:,adapter.layout[prefix+'zone_features'][0]]=zones.flatten(1)
        rows[:,adapter.layout[prefix+'zone_count'][0]]=obs.presence_mask.sum(1,keepdim=True)
    for name in ('action','reward','done','success','near_goal','line_to_goal_safe'):
        rows[:,adapter.layout[name][0]]=getattr(batch,name)
    rows[:,adapter.layout['failure'][0]]=batch.sample_failure_fraction
    return rows


def setup(args):
    from brain_uav.scripts import profile_v2_td3 as p
    from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation
    from brain_uav.models.v2_policy_config import V2SpikeRegularizationConfig
    from brain_uav.config import RewardConfig
    from brain_uav.trainers.v2_gpu_update import require_cuda_abi
    require_cuda_abi()
    config=p.V2FormalTrainingConfig(stage='easy',max_steps=500000,seed=args.seed,
        batch_size=args.batch_size,replay_capacity=args.replay_capacity,
        actor_lr=1.55e-4 if args.model=='ann' else 9.5e-4,critic_lr=2.5e-4,
        failure_sample_bias=2.,spike_regularization=V2SpikeRegularizationConfig(beta_max=1. if args.model=='snn' else 0.))
    rewards=RewardConfig(breakthrough_clearance_gate_enabled=False,
                         descent_three_band_enabled=True,boundary_directional_warning_enabled=True)
    prepared=p.prepare_v2_stage_initialization(config,init_checkpoint=args.bc_checkpoint,
        rewards=rewards,device='cuda',model_type=args.model,snn_time_window=4)
    pool=p.load_v2_validation_pool(args.validation_pool,expected_level='easy',
        expected_scenario=prepared.scenario_config,expected_uav_collision_radius=prepared.uav_collision_radius,
        expected_master_seed=args.validation_seed)
    compilation=default_v2_cuda_graph_compilation(model=args.model,resolved_device='cuda')
    return config,prepared,pool,compilation


def correctness(args,config,prepared,pool,compilation,report):
    from brain_uav.scripts import profile_v2_td3 as p
    from brain_uav.scripts.train_v2_td3 import _configure_stage_compilation
    from brain_uav.trainers.v2_gpu_update import GPUUpdate
    kwargs=dict(compilation); fused=kwargs.pop('fused_adam'); reduced=kwargs.pop('reduce_update_stat_syncs')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    e=p.build_v2_stage_engine(None,config,init_checkpoint=args.bc_checkpoint,rewards=prepared.reward_config,
        prepared_initialization=prepared,device='cuda',model_type=args.model,snn_time_window=4,
        fused_adam=fused,reduce_update_stat_syncs=reduced,gpu_td3_update=args.gpu_td3_update).engine
    initial=deepcopy(e.checkpoint_state_dict())
    reference=None
    if args.gpu_td3_update:
        if args.gpu_td3_reference is None: raise ValueError('GPU correctness needs a fresh independent reference.')
        reference=torch.load(args.gpu_td3_reference,map_location='cpu',weights_only=False)
        if reference['contract'].get('version') != REFERENCE_VERSION:
            raise ValueError('Old GPU correctness reference rejected; rebuild the baseline reference (contract version 2).')
        if reference['contract']!=report['contract']: raise ValueError('Reference input/contract differs.')
        initial=reference['initial']; e.load_checkpoint_state_dict(initial)
    report['compilation']=_configure_stage_compilation(e,pool,prepared,
        compile_critic_encoder=False,compile_target_encoders=False,**kwargs)
    batches=p._compile_warmup_batches(pool=pool,prepared=prepared,batch_size=e.batch_size,
        device=torch.device('cuda'),cache_ellipsoid_segment_clearance=True)
    mixed,nonempty=p._numeric_diagnostic_observation_batches(batches,device=torch.device('cuda'))
    from dataclasses import replace
    empty=replace(mixed,zone_features=mixed.zone_features[:,:0],presence_mask=mixed.presence_mask[:,:0])
    fixed=[p._fixed_numeric_replay_batch(obs,action_dim=e.action_dim,device=torch.device('cuda')) for obs in (mixed,empty,nonempty)]
    # Controlled near-goal empty batch exercises the actual terminal regularizer
    # on an actor update, not just its inactive far-goal branch.
    from brain_uav.observations import GOAL_FEATURE_INDEX
    goal=fixed[1].obs.goal_features.detach().clone()
    for key,value in (('goal_forward_norm',100./e.actor.scales.horizontal_span),
                      ('goal_right_norm',0.),('goal_up_norm',0.),
                      ('goal_distance_norm',100./e.actor.scales.world_diagonal)):
        goal[:,GOAL_FEATURE_INDEX[key]]=value
    near=replace(fixed[1].obs,goal_features=goal)
    fixed[1]=replace(fixed[1],obs=near,next_obs=replace(near,goal_features=goal.detach().clone()),
                     line_to_goal_safe=torch.ones_like(fixed[1].line_to_goal_safe))
    fixed=[replace(b,next_obs=type(b.next_obs)(*(t.detach().clone() for t in
        (b.next_obs.ego_features,b.next_obs.goal_features,b.next_obs.zone_features,b.next_obs.presence_mask)))) for b in fixed]
    # Initialize a real replay before the sampler capture, using existing fixed
    # scenarios (no data generation). Also exercise episode events and stale refs.
    env=p.V2StaticNoFlyTrajectoryEnv(prepared.scenario_config,prepared.reward_config,
        seed=pool.stage_seed,uav_collision_radius=prepared.uav_collision_radius,
        fixed_scenarios=[r['payload'] for r in pool.scenarios[:1]],cache_ellipsoid_segment_clearance=True)
    obs=env.reset(options={'scenario':pool.scenarios[0]['payload']})[0]
    refs=[]
    for i in range(e.batch_size): refs.append(e.replay.add(obs,np.zeros(e.action_dim,np.float32),float(i),obs,False))
    steps=p._td3_block_correctness_steps(config,max(8,args.correctness_updates))
    executor=None
    if args.gpu_td3_update:
        report['location']=dict(stage='full_executor_initialization',model=args.model)
        executor=GPUUpdate(e,diagnostic=True); e._gpu_update_executor=executor
        executor.fixed_mode.fill_(True)
    samples=[]; report['updates']=[]
    for index,step in enumerate(steps):
        report['location']=dict(stage='update',index=index,stage_steps=step,model=args.model)
        batch=fixed[index%len(fixed)]
        if executor:
            executor.fixed_rows.copy_(pack_fixed(executor.replay,batch))
        else:
            e.replay.sample=lambda size,b=batch:b
        torch.manual_seed(args.seed+index); torch.cuda.manual_seed_all(args.seed+index)
        _,noise,clip=config.noise_schedule.values(step-1,max_steps=config.max_steps)
        e.set_target_noise(policy_noise=noise,noise_clip=clip)
        before,_=snapshot(e)
        intermediates={}
        def record(name,tensor): intermediates[name]=tensor.detach().cpu().clone()
        original_clip=e._validate_and_clip_gradients
        original_spikes=e.actor.snn_head._forward_impl if args.model=='snn' else None
        def spikes(context):
            raw,first,second=original_spikes(context)
            record('activity.l1',first.detach().bool())
            record('activity.l2',second.detach().bool())
            return raw,first,second
        names={id(p):module+'.'+name for module in ('actor','critic1','critic2') for name,p in getattr(e,module).named_parameters()}
        def clip(parameters,**kw):
            for parameter in parameters:
                if parameter.grad is not None: record('gradient.pre_clip.'+names[id(parameter)],parameter.grad)
            return original_clip(parameters,**kw)
        if not executor:
            e._validate_and_clip_gradients=clip
            if original_spikes is not None: e.actor.snn_head._forward_impl=spikes
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                metric=e.update_once(total_steps=step,bc_lambda=p.v2_bc_lambda(step-1,schedule='smooth-final'),
                    **({'diagnostic_tensor_recorder':record} if not executor else {}))
                torch.cuda.synchronize()
        finally:
            e._validate_and_clip_gradients=original_clip
            if not executor and original_spikes is not None: e.actor.snn_head._forward_impl=original_spikes
        launches=sum(event.count for event in prof.key_averages() if 'cudagraphlaunch' in event.key.lower())
        if not launches: raise AssertionError('No CUDA Graph launch observed in compared update.')
        actual,gradient_evidence=snapshot(e,metric)
        for module in (e.actor_target,e.critic1_target,e.critic2_target,e.bc_reference_actor):
            if module is not None and any(p.grad is not None for p in module.parameters()):
                raise AssertionError('Target/BC reference received parameter gradients.')
        if executor:
            mode='actor' if metric.actor_updated else 'frozen_targets' if metric.critic_targets_updated else 'critic_only'
            intermediates={name:t.detach().cpu().clone() for name,t in executor.tensor_evidence[mode].items()}
            for key in ('replay.rows','replay.mask','replay.next_mask'):
                intermediates.pop(key)  # Fixed-formula checks are separate from actual sampling.
        complete_actor_evidence(intermediates,args.model)
        actual.update({'tensor.'+name:t for name,t in intermediates.items()})
        for name,value in list(actual.items()):
            if name.startswith(('actor.','critic1.','critic2.','actor_target.','critic1_target.','critic2_target.')) and value.is_floating_point():
                actual['delta.'+name]=value-before[name]
        if args.model=='snn':
            for actor in (e.actor,e.actor_target,e.bc_reference_actor):
                if actor is None: continue
                for lif in (actor.snn_head.lif1,actor.snn_head.lif2):
                    membrane=lif.v
                    if isinstance(membrane,torch.Tensor):
                        if bool(torch.count_nonzero(membrane)): raise AssertionError('SNN membrane reset failed.')
                    elif membrane!=0: raise AssertionError('SNN membrane reset failed.')
        context=dict(snapshot_index=index,stage_steps=step,model=args.model,
            obs_shape=list(batch.obs.zone_features.shape), mask=batch.obs.presence_mask.cpu().tolist(),
            padded_zone_capacity=e.replay.zone_storage_capacity,
            actor_backward=metric.actor_updated,gradients=gradient_evidence)
        difference=None
        if reference:
            difference=p._compare_td3_block_snapshot(reference['snapshots'][index],actual,
                context=context,report=report,output=args.output_dir)
        report['updates'].append(dict(stage_steps=step,actor_updated=metric.actor_updated,
            cuda_graph_launches=launches,differences=difference,metrics=asdict(metric),
            gradient_stage='post_clip; current backward strict, unused actor residuals recorded',
            state_reset_checked=args.model=='snn',deltas='post-minus-pre independent CPU copies; continuous steps may carry previous differences'))
        samples.append(actual)
    # Fixed-draw sampling comparison, exact Fenwick restoration, controlled
    # failure/success episode close: never classify a budget fragment as failed.
    if executor:
        e.replay.mark_failure_slots(refs[-2:])
        with e.gpu_episode_submission():
            for _ in range(2): e.replay.add_success_transition(obs,np.zeros(2,np.float32),1.,obs,True)
            e.replay.mark_success_slots(refs[:2])
        d=executor.replay
        torch.testing.assert_close(d.tree.cpu(),torch.from_numpy(e.replay._sampling_weight_tree._tree),rtol=0,atol=0)
        rng=deepcopy(e.replay.rng.bit_generator.state)
        expected=e.replay._sample_primary_indices(e.batch_size)
        e.replay.rng.bit_generator.state=rng
        draws=e.replay.rng.random(e.batch_size)
        previous=d.tree.detach().clone()
        rows=d.sample_tensors(torch.as_tensor(draws,device='cuda'),torch.zeros(e.batch_size,dtype=torch.long,device='cuda'),
            torch.tensor(e.batch_size,device='cuda'),torch.tensor(e.replay.success_size,device='cuda'))
        slot,_=d.layout['reward']
        torch.testing.assert_close(rows[:,slot].cpu(),torch.from_numpy(e.replay.reward[expected]),rtol=0,atol=0)
        torch.testing.assert_close(d.tree,previous,rtol=0,atol=0)
        fixed_rows=d.gather_indices(torch.arange(e.batch_size-2,device='cuda'),torch.arange(2,device='cuda'))
        torch.testing.assert_close(fixed_rows[:e.batch_size-2,slot].cpu(),
            torch.from_numpy(e.replay.reward[:e.batch_size-2]),rtol=0,atol=0)
        torch.testing.assert_close(fixed_rows[-2:,slot].cpu(),
            torch.from_numpy(e.replay.success_reward[:2]),rtol=0,atol=0)
        report['fixed_index_gather_check']=True
        report['episode_and_sampling_check']=True
    errors=[]
    for i,kind in enumerate(('critic_nan_loss','critic_inf_gradient','actor_nan_loss','actor_inf_gradient')):
        report['location']=dict(stage='error_barrier',injection=i,kind=kind,model=args.model)
        before,_=snapshot(e)
        if executor:
            executor.fixed_rows.copy_(pack_fixed(executor.replay,fixed[0])); executor.injections.zero_(); executor.injections[i]=1
            injection=p.nullcontext()
        else:
            base=fixed[0]
            e.replay.sample=lambda size: replace(base,reward=torch.full_like(base.reward,float('nan'))) if i==0 else base
            evidence={}
            injection=p._inject_diagnostic_inf_gradient(e,component='critic' if i==1 else 'actor',evidence=evidence) if i in (1,3) else p.nullcontext()
        original=e._require_finite_loss
        def check(loss,*,component,total_steps):
            if i==2 and component=='actor': loss=loss*float('nan')
            return original(loss,component=component,total_steps=total_steps)
        if not executor: e._require_finite_loss=check
        try:
            torch.manual_seed(args.seed+100); torch.cuda.manual_seed_all(args.seed+100)
            with injection:
                try: e.update_once(total_steps=75002,bc_lambda=p.v2_bc_lambda(75001,schedule='smooth-final'))
                except FloatingPointError as exc:
                    expected='Non-finite '+('critic' if i<2 else 'actor')+' '+('loss' if i in (0,2) else 'gradient')
                    if not str(exc).startswith(expected): raise
                else: raise AssertionError('Error injection did not reach barrier: '+kind)
        finally:
            e._require_finite_loss=original
            if executor: executor.injections.zero_()
        after,_=snapshot(e)
        protected=[k for k in before if i<2 or k.startswith(('actor.','actor_target.','optimizer.actor.','counts.actor_update_count'))]
        p._compare_compiled_numeric_tensors({k:before[k] for k in protected},{k:after[k] for k in protected},rtol=0,atol=0)
        if reference: p._compare_compiled_numeric_tensors(reference['errors'][i],after,rtol=1e-4,atol=1e-5)
        report.setdefault('error_barriers',[]).append(kind); errors.append(after)
    if executor: report['execution']=executor.metadata()
    # Actual public load, independent optimizer state restoration and a real
    # subsequent update, not merely successful reconstruction of graph objects.
    report['location']=dict(stage='checkpoint_rebuild',model=args.model)
    saved=deepcopy(e.checkpoint_state_dict()); e.load_checkpoint_state_dict(saved)
    if e._gpu_update_executor is not None: raise AssertionError('Load retained GPU executor.')
    if executor:
        rebuilt=GPUUpdate(e); e._gpu_update_executor=rebuilt
        rebuilt.fixed_mode.fill_(True); rebuilt.fixed_rows.copy_(pack_fixed(rebuilt.replay,fixed[1]))
    else:
        _configure_stage_compilation(e,pool,prepared,compile_critic_encoder=False,compile_target_encoders=False,**kwargs)
        e.replay.sample=lambda size:fixed[1]
    torch.manual_seed(args.seed+200); torch.cuda.manual_seed_all(args.seed+200)
    m=e.update_once(total_steps=75004,bc_lambda=p.v2_bc_lambda(75003,schedule='smooth-final'))
    rebuild_state,rebuild_gradients=snapshot(e,m)
    if reference:
        p._compare_td3_block_snapshot(reference['rebuild'],rebuild_state,
            context=dict(snapshot_index='rebuild',stage_steps=75004,actor_backward=True,gradients=rebuild_gradients),
            report=report,output=args.output_dir)
    report['checkpoint_rebuild']=True
    end_to_end = sampling_updates(args,e,config,pool,prepared,report,reference)
    if not executor:
        torch.save(dict(initial=initial,snapshots=samples,errors=errors,rebuild=rebuild_state,
            end_to_end=end_to_end,contract=report['contract']),args.output_dir/'reference.pt')
    else: e.invalidate_gpu_update()


def observed_update(e, executor, step, bc_lambda, model):
    """Diagnostic snapshots only, same forward/backward as the real update."""
    tensors = {}
    def record(name, tensor): tensors[name] = tensor.detach().cpu().clone()
    original_clip = e._validate_and_clip_gradients
    original_spikes = e.actor.snn_head._forward_impl if model == 'snn' else None
    names = {id(parameter): module+'.'+name for module in ('actor','critic1','critic2')
             for name,parameter in getattr(e,module).named_parameters()}
    def clip(parameters, **kw):
        for parameter in parameters:
            if parameter.grad is not None: record('gradient.pre_clip.'+names[id(parameter)],parameter.grad)
        return original_clip(parameters, **kw)
    def spikes(context):
        raw, first, second = original_spikes(context)
        record('activity.l1',first.bool()); record('activity.l2',second.bool())
        return raw, first, second
    if not executor:
        e._validate_and_clip_gradients = clip
        if original_spikes: e.actor.snn_head._forward_impl = spikes
    try:
        metric = e.update_once(total_steps=step,bc_lambda=bc_lambda,
            **({'diagnostic_tensor_recorder':record} if not executor else {}))
    finally:
        e._validate_and_clip_gradients = original_clip
        if not executor and original_spikes: e.actor.snn_head._forward_impl = original_spikes
    if executor:
        mode = 'actor' if metric.actor_updated else 'frozen_targets' if metric.critic_targets_updated else 'critic_only'
        tensors = {name:t.detach().cpu().clone() for name,t in executor.tensor_evidence[mode].items()}
    complete_actor_evidence(tensors,model)
    return metric,tensors


def sampling_updates(args,e,config,pool,prepared,report,reference):
    """Four real sample->gather->update replays; independent small ring capacity.

    Baseline still uses original dynamic cropping. Padding appears only in
    evidence, never in baseline network input. No fixed-index update substitute.
    """
    from brain_uav.scripts import profile_v2_td3 as p
    from brain_uav.trainers.v2_gpu_update import GPUUpdate
    from brain_uav.trainers.v2_gpu_replay import DeviceReplay
    from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer
    old = e.replay
    e.invalidate_gpu_update()
    e.replay = V2ReplayBuffer(e.batch_size+4,e.action_dim,old.zone_storage_capacity,
        success_sample_bias=old.success_sample_bias,near_goal_sample_bias=2.,
        success_replay_fraction=old.success_replay_fraction,success_batch_fraction=old.success_batch_fraction,
        failure_sample_bias=old.failure_sample_bias,seed=args.seed)
    r = e.replay
    env = p.V2StaticNoFlyTrajectoryEnv(prepared.scenario_config,prepared.reward_config,
        seed=pool.stage_seed,uav_collision_radius=prepared.uav_collision_radius,
        fixed_scenarios=[v['payload'] for v in pool.scenarios[:3]],cache_ellipsoid_segment_clearance=True)
    observations = [env.reset(options={'scenario':v['payload']})[0] for v in pool.scenarios[:3]]
    # Include the real empty observation contract, without changing the model.
    from dataclasses import replace
    observations.append(replace(observations[0],zone_features=observations[0].zone_features[:0],
                                presence_mask=observations[0].presence_mask[:0]))
    refs = []; serial = 0
    def add(done=False):
        nonlocal serial
        observation = observations[serial % len(observations)]
        following = observations[(serial+1) % len(observations)]
        transition = (observation,np.zeros(e.action_dim,np.float32),1.+serial/100.,following,done)
        ref = r.add(*transition,near_goal=serial%2==0,line_to_goal_safe=True)
        refs.append(ref); serial += 1
        return transition,ref
    for _ in range(r.capacity): add()
    r.mark_failure_slots(refs[-3:])
    executor = GPUUpdate(e,diagnostic=True) if args.gpu_td3_update else None
    if executor:
        e._gpu_update_executor = executor
        executor.fixed_mode.fill_(False)
        adapter = executor.replay
    else:
        adapter = DeviceReplay(r,'cpu'); adapter.detach()  # Layout oracle only, never baseline network inputs.
    results = []; report['sample_to_update'] = []
    report['sample_to_update_configuration'] = dict(capacity=r.capacity,batch_size=e.batch_size,
        success_capacity=r.success_capacity,failure_sample_bias=r.failure_sample_bias,
        near_goal_sample_bias=r.near_goal_sample_bias,controlled_random_inputs=True,
        baseline_dynamic_cropping=True,gpu_fixed_padding=True,reference_version=REFERENCE_VERSION)
    for i,phase in enumerate(('no_success','success_insufficient','complete_success_episode','ring_overwrite')):
        step = 75005+i
        report['location'] = dict(stage='sample_to_update',phase=phase,index=i,stage_steps=step,model=args.model)
        if i == 1:
            transition,ref = add(True)
            with e.gpu_episode_submission():
                r.add_success_transition(*transition,near_goal=True,line_to_goal_safe=True)
                r.mark_success_slots([ref])
        if i == 2:
            episode = [add(j==15) for j in range(16)]
            with e.gpu_episode_submission():
                for transition,ref in episode:
                    r.add_success_transition(*transition,near_goal=bool(r.near_goal[ref[0]]),line_to_goal_safe=True)
                r.mark_success_slots([ref for _,ref in episode])
        if i == 3:
            stale = refs.copy()
            for _ in range(r.capacity+1): add()
            if r.mark_failure_slots(stale) != 0: raise AssertionError('Expired slot references changed weights.')
            r.mark_failure_slots(refs[-4:])
        count = min(int(e.batch_size*r.success_batch_fraction),r.success_size)
        rng = np.random.default_rng(args.seed+400+i)
        draws = np.zeros(e.batch_size,np.float64); draws[:e.batch_size-count] = rng.random(e.batch_size-count)
        ranks = np.zeros(e.batch_size,np.int64)
        for j in range(count): ranks[j] = rng.integers(r.success_size-j)
        primary,success,batch = controlled_cpu_sample(r,draws,ranks,e.batch_size)
        rows = expected_rows(adapter,primary,success)
        tree = torch.from_numpy(r._sampling_weight_tree._tree.copy())
        if executor:
            executor.replay.controls = lambda size: (draws.copy(),ranks.copy(),count)
        else: r.sample = lambda size,b=batch: b
        _,noise,noise_clip = config.noise_schedule.values(step-1,max_steps=config.max_steps)
        e.set_target_noise(policy_noise=noise,noise_clip=noise_clip)
        torch.manual_seed(args.seed+500+i); torch.cuda.manual_seed_all(args.seed+500+i)
        before,_ = snapshot(e)
        metric,tensors = observed_update(e,executor,step,p.v2_bc_lambda(step-1,schedule='smooth-final'),args.model)
        if not executor: del r.sample  # Restore actual class sampler before the next controlled oracle call.
        sampled = tensors.pop('replay.rows') if executor else rows
        try:
            torch.testing.assert_close(sampled,rows,rtol=0,atol=0)
        except AssertionError:
            path=args.output_dir/'first_sample_failure.pt'
            torch.save(dict(context=report['location'],primary_indices=primary,success_indices=success,
                expected=rows,actual=sampled),path)
            report['sample_failure_evidence']=str(path)
            raise
        if executor:
            torch.testing.assert_close(adapter.tree.cpu(),tree,rtol=0,atol=0)
            torch.testing.assert_close(adapter.weights.cpu(),torch.from_numpy(r.sample_weight),rtol=0,atol=0)
        for prefix,obs in (('',batch.obs),('next_',batch.next_obs)):
            counts = sampled[:,adapter.layout[prefix+'zone_count'][0]].flatten()
            actual_mask = (tensors.pop('replay.'+('next_mask' if prefix else 'mask')) if executor
                else torch.arange(r.zone_storage_capacity)[None] < counts[:,None])
            expected_mask = torch.zeros_like(actual_mask)
            expected_mask[:,:obs.presence_mask.shape[1]] = obs.presence_mask
            torch.testing.assert_close(actual_mask,expected_mask,rtol=0,atol=0)
        actual,gradient_evidence = snapshot(e,metric)
        actual.update({'tensor.'+name:t for name,t in tensors.items()})
        for name,value in list(actual.items()):
            if name.startswith(('actor.','critic1.','critic2.','actor_target.','critic1_target.','critic2_target.')) and value.is_floating_point():
                actual['delta.'+name] = value-before[name]
        context = dict(snapshot_index='sample_to_update.'+str(i),model=args.model,stage_steps=step,
            phase=phase,actor_backward=metric.actor_updated,gradients=gradient_evidence,
            obs_shape=list(batch.obs.zone_features.shape),mask=batch.obs.presence_mask.tolist(),
            padded_zone_capacity=r.zone_storage_capacity)
        if reference:
            p._compare_td3_block_snapshot(reference['end_to_end'][i],actual,context=context,report=report,output=args.output_dir)
        report['sample_to_update'].append(dict(phase=phase,stage_steps=step,primary_indices=primary.tolist(),
            success_indices=success.tolist(),actor_updated=metric.actor_updated,tree_restored=True,
            sampled_rows_checked=True,fixed_mode=False if executor else None,
            baseline_dynamic_shape=list(batch.obs.zone_features.shape)))
        results.append(actual)
    report['sample_to_update_passed'] = True
    if executor:
        report['sample_to_update_execution'] = executor.metadata()
        e.invalidate_gpu_update()
    return results


def sampling_cost(args):
    """Network-free, standalone CUDA sampler+gather measurement, not a suite."""
    from brain_uav.scripts import profile_v2_td3 as p
    from brain_uav.trainers.v2_gpu_update import require_cuda_abi
    from brain_uav.trainers.v2_gpu_replay import DeviceReplay
    from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer
    from brain_uav.observations import V2Observation, EGO_FEATURE_DIM, GOAL_FEATURE_DIM, ZONE_FEATURE_DIM
    if args.device != 'cuda': raise ValueError('Sampling cost requires real CUDA.')
    if args.batch_size != 64 or args.replay_capacity != 500000:
        raise ValueError('Sampling cost requires formal capacity 500000 and batch 64.')
    if args.sampling_replays <= 0 or args.sampling_warmup <= 0 or args.repeats <= 0:
        raise ValueError('Sampling replay/warmup/repeat counts must be positive.')
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Refusing nonempty output directory.')
    output.mkdir(parents=True,exist_ok=True)
    report = dict(diagnostic_only=True,passed=False,completed=False,checks=[],location={'stage':'capability'},
        requested=dict(capacity=args.replay_capacity,batch_size=args.batch_size,seed=args.seed,
            measured_replays=args.sampling_replays,warmup_replays=args.sampling_warmup,repeats=args.repeats),
        scope='No network/environment update. GPU graph is actual sample_tensors including gather and validation. Fixed controls replayed; CPU references use same primary draws and partial Fisher-Yates success ranks.',
        timing_note='CPU wall with CUDA synchronization only at each measurement-window boundary. Includes replay submission and completion; not pure kernel time. CPU index-only and CPU index+dynamic-batch references are separate unequal scopes, no speedup claim.')
    adapter = None
    try:
        require_cuda_abi()
        report['provenance'] = dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            dirty=subprocess.check_output(['git','status','--short'],text=True),torch=torch.__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),python=platform.python_version())
        report['location'] = {'stage':'allocation'}
        start = perf_counter()
        replay = V2ReplayBuffer(500000,2,2,failure_sample_bias=2.,near_goal_sample_bias=2.,seed=args.seed)
        observations = [V2Observation(np.zeros(EGO_FEATURE_DIM,np.float32),np.zeros(GOAL_FEATURE_DIM,np.float32),
            np.zeros((n,ZONE_FEATURE_DIM),np.float32),np.ones(n,np.bool_)) for n in (0,1,2)]
        transitions = []; refs = []
        for i in range(80):
            transition = (observations[i%3],np.zeros(2,np.float32),float(i),observations[(i+1)%3],i==79)
            refs.append(replay.add(*transition,near_goal=i%2==0)); transitions.append(transition)
        replay.mark_failure_slots(refs[-3:])
        adapter = DeviceReplay(replay,'cuda')
        torch.cuda.synchronize()
        report['allocation_seconds'] = perf_counter()-start
        report['configuration'] = dict(capacity=500000,batch_size=64,valid_primary=80,
            success_capacity=replay.success_capacity,zone_storage_capacity=2,failure_sample_bias=2.,near_goal_sample_bias=2.,
            measured_replays=args.sampling_replays,warmup_replays=args.sampling_warmup,repeats=args.repeats)
        for mixed in (False,True):
            item = dict(success_mixed=mixed,passed=False)
            report['checks'].append(item)
            report['location'] = dict(stage='inputs',success_mixed=mixed)
            if mixed:
                with adapter.success_upload_batch():
                    for i,transition in enumerate(transitions[:16]):
                        replay.add_success_transition(*transition,near_goal=bool(replay.near_goal[refs[i][0]]))
                    replay.mark_success_slots(refs[:16])
            draws_np,ranks_np,count = adapter.controls(64)
            primary,success,_ = controlled_cpu_sample(replay,draws_np,ranks_np,64)
            item.update(primary_indices=primary.tolist(),success_indices=success.tolist())
            expected = expected_rows(adapter,primary,success)
            if len(set(primary.tolist())) != len(primary) or len(set(success.tolist())) != len(success):
                raise AssertionError('CPU reference contains within-pool replacement.')
            draws = torch.from_numpy(draws_np).cuda(); ranks = torch.from_numpy(ranks_np).cuda()
            primary_count = torch.tensor(64-count,device='cuda'); success_size = torch.tensor(replay.success_size,device='cuda')
            size = torch.tensor(replay.size,device='cuda'); total = torch.tensor(replay.total_sample_weight,dtype=torch.float64,device='cuda')
            original_tree = adapter.tree.detach().clone()
            def sample():
                return adapter.sample_tensors(draws,ranks,primary_count,success_size,
                    size=size,expected_total=total,return_status=True)
            def validate(rows,failed):
                if bool(failed): raise AssertionError('Device sampler reported invalid controls/tree/indices.')
                torch.testing.assert_close(rows.cpu(),expected,rtol=0,atol=0)
                torch.testing.assert_close(adapter.tree,original_tree,rtol=0,atol=0)
            report['location'] = dict(stage='uncaptured_validation',success_mixed=mixed)
            validate(*sample())
            report['location'] = dict(stage='warmup_capture',success_mixed=mixed)
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            start = perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(args.sampling_warmup): sample()
            torch.cuda.synchronize()
            item['warmup_seconds'] = perf_counter()-start
            start_capture = perf_counter()
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): rows,failed = sample()
            torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
            item['capture_seconds'] = perf_counter()-start_capture
            item['warmup_capture_seconds'] = perf_counter()-start
            # Capture records operations; outputs become valid only after replay.
            graph.replay(); torch.cuda.synchronize(); validate(rows,failed)
            item.update(primary_indices=primary.tolist(),success_indices=success.tolist(),
                results_checked=True,tree_restored=True,no_replacement_checked=True,runs=[])
            # No profiler, item/cpu read, or per-replay synchronization below.
            for repeat in range(args.repeats):
                report['location'] = dict(stage='measurement',success_mixed=mixed,repeat=repeat)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start = perf_counter()
                for _ in range(args.sampling_replays): graph.replay()
                torch.cuda.synchronize(); elapsed = perf_counter()-start
                validate(rows,failed)
                # Index-only CPU reference uses original tree sampler + sparse FY,
                # excluding dynamic observation assembly and all tensor transfers.
                start = perf_counter()
                for _ in range(args.sampling_replays): controlled_cpu_indices(replay,draws_np,ranks_np,64)
                indices_seconds = perf_counter()-start
                start = perf_counter()
                for _ in range(args.sampling_replays): controlled_cpu_sample(replay,draws_np,ranks_np,64)
                batch_seconds = perf_counter()-start
                item['runs'].append(dict(calls=args.sampling_replays,total_wall_seconds=elapsed,
                    seconds_per_call=elapsed/args.sampling_replays,
                    cpu_indices_seconds=indices_seconds,cpu_indices_seconds_per_call=indices_seconds/args.sampling_replays,
                    cpu_indices_and_dynamic_batch_seconds=batch_seconds,
                    cpu_indices_and_dynamic_batch_seconds_per_call=batch_seconds/args.sampling_replays,
                    allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            item['passed'] = True
            del graph,rows,failed
        report.update(passed=True,completed=True)
    except BaseException as exc:
        report['failure'] = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        if adapter: adapter.detach()
        p._strict_json_write(output/'summary.json',report)
    return report


def worker(args):
    from brain_uav.scripts import profile_v2_td3 as p
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Refusing nonempty output directory.')
    output.mkdir(parents=True,exist_ok=True)
    report=dict(diagnostic_only=True,passed=False,completed=False,model=args.model,
        requested_gpu_td3_update=args.gpu_td3_update,location={'stage':'setup'})
    report['method']='Fresh matching BC and seeded critics/empty native Adam. Controlled stage counters exercise schedules, not historical training. Diagnostic updates only; no formal checkpoint output.'
    try:
        config,prepared,pool,compilation=setup(args)
        report['contract']=dict(version=REFERENCE_VERSION,checkpoint_sha256=sha256(args.bc_checkpoint.read_bytes()).hexdigest(),
            pool_digest=pool.content_digest,config=config.to_dict(),rewards=asdict(prepared.reward_config),
            model=args.model,seed=args.seed,correctness_updates=max(8,args.correctness_updates))
        report['provenance']=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            dirty=subprocess.check_output(['git','status','--short'],text=True),torch=torch.__version__,cuda=torch.version.cuda,
            python=platform.python_version(),
            gpu=torch.cuda.get_device_name(),compilation=compilation,
            load_before=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
        repository=Path(__file__).resolve().parents[3]
        sources=('src/brain_uav/trainers/v2_gpu_update.py','src/brain_uav/trainers/v2_gpu_replay.py',
                 'src/brain_uav/trainers/v2_td3.py','src/brain_uav/scripts/profile_v2_td3.py',
                 'src/brain_uav/scripts/v2_gpu_td3_diagnostic.py')
        report['provenance']['source_sha256']={name:sha256((repository/name).read_bytes()).hexdigest() for name in sources}
        if args.gpu_td3_worker=='correctness': correctness(args,config,prepared,pool,compilation,report)
        else:
            selected=p.V2ValidationPool(curriculum_level='easy',master_seed=pool.master_seed,stage_seed=pool.stage_seed,
                scenario_config=pool.scenario_config,uav_collision_radius=pool.uav_collision_radius,
                scenarios=pool.scenarios[:args.scenario_count])
            compilation=dict(compilation); compilation.pop('reduce_update_stat_syncs')
            report['runs']=[]
            for repeat in range(args.repeats):
                report['location']=dict(stage='timing',repeat=repeat)
                np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
                result=p._run_diagnostic_level(level='easy',pool=selected,prepared=prepared,formal_config=config,
                    bc_checkpoint=args.bc_checkpoint,model=args.model,snn_time_window=4,device=torch.device('cuda'),
                    warmup_steps=args.warmup_steps,measured_steps=args.steps_per_level,
                    controlled_stage_step_offset=args.controlled_stage_step_offset,bc_schedule='smooth-final',
                    reduce_update_stat_syncs=True,cache_ellipsoid_segment_clearance=True,
                    gpu_td3_update=args.gpu_td3_update,**compilation)
                report['runs'].append(result)
                gc.collect(); torch.cuda.empty_cache()
            stable=all(r['critic_encoder_compile'].get('valid_for_speed_comparison') is not False
                       and r['local_optimizations']['gpu_td3_update'].get('measurement_new_captures',0)==0
                       for r in report['runs'])
            report['stable_timing']=stable
            if not stable: raise RuntimeError('Measurement recompiled/recaptured; speed comparison invalid.')
            report['timing_note']='Existing total environment wall includes replay event uploads; graph capture occurs in warmup. GPU update inner legacy subregions are unmeasured (zero), full sample-to-result is separately reported. No profiler enabled.'
            report['episode_amortization']='Short fragments may have no episode close; controlled correctness event check is not full-course amortization.'
        report.update(passed=True,completed=True)
    except BaseException as exc:
        report['failure']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        if hasattr(exc,'gpu_update_context'): report['failure']['gpu_update_context']=exc.gpu_update_context
        raise
    finally:
        if 'provenance' in report:
            report['provenance']['load_after']=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
            report['cuda_memory']=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        p._strict_json_write(output/'summary.json',report)
    return report


def run(args):
    from brain_uav.scripts import profile_v2_td3 as p
    if args.device!='cuda': raise ValueError('GPU TD3 diagnosis requires CUDA; CPU is not a substitute.')
    if args.comparison_eager_cpu or args.detailed_profiler_updates or args.compiled_performance_diagnostic_updates:
        raise ValueError('GPU TD3 workers use fixed formal CUDA config and no timing profiler.')
    if any(getattr(args,n) for n in ('compile_critic_update_front','compile_actor_objective','cuda_graph_critic_tail',
        'window_update_statistics','reuse_update_buffers','pinned_batch_transfer','aggregate_relation_values_first',
        'single_observation_fast_path','td3_block_suite','snn_path_consistency','td3_block_attribution')):
        raise ValueError('GPU suite conflicts with other experimental paths.')
    if getattr(args,'gpu_td3_sampling_cost',False):
        if args.gpu_td3_suite or args.gpu_td3_worker or args.gpu_td3_update or args.gpu_td3_reference:
            raise ValueError('Sampling cost is standalone, not a correctness/timing suite.')
        return sampling_cost(args)
    if args.gpu_td3_worker: return worker(args)
    if args.gpu_td3_update:
        raise ValueError('Suite owns baseline/GPU selection; do not pass --gpu-td3-update to the driver.')
    if args.model!='ann' or args.snn_bc_checkpoint is None or args.validation_pool is None:
        raise ValueError('Suite requires ANN BC, SNN BC and existing easy fixed pool.')
    root=Path(args.output_dir)
    if root.exists() and any(root.iterdir()): raise FileExistsError('Refusing nonempty suite directory.')
    root.mkdir(parents=True,exist_ok=True)
    summary=dict(diagnostic_only=True,passed=False,configurations=[])
    for model,bc in (('ann',args.bc_checkpoint),('snn',args.snn_bc_checkpoint)):
        baseline_ok=False
        for enabled in (False,True):
            group='gpu' if enabled else 'baseline'; item=dict(model=model,group=group,passed=False)
            summary['configurations'].append(item)
            if enabled and not baseline_ok:
                item['skipped']='Baseline correctness unavailable'; continue
            for phase in ('correctness','timing'):
                out=root/model/(group+'_'+phase)
                command=[sys.executable,'-u','-m','brain_uav.scripts.profile_v2_td3','--model',model,
                    '--bc-checkpoint',str(bc),'--validation-pool',str(args.validation_pool),'--device','cuda',
                    '--output-dir',str(out),'--gpu-td3-worker',phase,'--replay-capacity',str(args.replay_capacity),
                    '--batch-size',str(args.batch_size),'--seed',str(args.seed),'--warmup-steps',str(args.warmup_steps),
                    '--validation-seed',str(args.validation_seed),
                    '--controlled-stage-step-offset',str(args.controlled_stage_step_offset),
                    '--steps-per-level',str(args.steps_per_level),'--repeats',str(args.repeats),
                    '--scenario-count',str(args.scenario_count),'--correctness-updates',str(max(8,args.correctness_updates))]
                if enabled:
                    command+=['--gpu-td3-update']
                    if phase=='correctness': command+=['--gpu-td3-reference',str(root/model/'baseline_correctness'/'reference.pt')]
                out.parent.mkdir(parents=True,exist_ok=True)
                with (out.parent/(group+'_'+phase+'.log')).open('w',encoding='utf-8') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
                item[phase]=dict(returncode=result.returncode,summary=str(out/'summary.json'))
                if result.returncode:
                    if (out/'summary.json').exists():
                        item[phase]['failure']=json.loads((out/'summary.json').read_text(encoding='utf-8')).get('failure')
                    else:
                        item[phase]['failure']={'reason':'worker stopped before summary','log':str(out.parent/(group+'_'+phase+'.log'))}
                    break
                if phase=='correctness' and not enabled: baseline_ok=True
                if phase=='timing': item['passed']=True
            p._strict_json_write(root/f'progress_{len(summary["configurations"])}.json',summary)
        records=summary['configurations'][-2:]
        if all(r['passed'] for r in records):
            values={}
            for r in records:
                report=json.loads(Path(r['timing']['summary']).read_text(encoding='utf-8'))
                raw=[v['timing']['total_wall_seconds']/v['measured_steps'] for v in report['runs']]
                values[r['group']]=dict(raw_seconds_per_step=raw,mean=mean(raw),stdev=stdev(raw) if len(raw)>1 else 0.)
            values['time_saved_percent']=100*(1-values['gpu']['mean']/values['baseline']['mean'])
            values['speedup']=values['baseline']['mean']/values['gpu']['mean']
            summary.setdefault('comparison',{})[model]=values
    summary['passed']=all(r['passed'] for r in summary['configurations'])
    p._strict_json_write(root/'summary.json',summary)
    if not summary['passed']: raise RuntimeError('GPU TD3 suite completed with failures; inspect per-worker summaries/logs.')
    return summary
