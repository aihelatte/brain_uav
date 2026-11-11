"""Bounded eager/real GPUCompute comparison called only by explicit gate workers."""
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import patch
import torch

GATE_STEPS=(1,3,4,25001,25002,26002,26003,26004)
# 26002 first captures actor banks (1,1); 26003 reuses critic-only bank 0.
BETA_REFRESH_STEPS=(25001,26003)


def bank_observation(batch):
    from brain_uav.observations import V2Observation
    mask=batch.presence_mask[0].numpy()
    return V2Observation(batch.ego_features[0].numpy(),batch.goal_features[0].numpy(),
        batch.zone_features[0].numpy()[mask],mask[mask])


def batch_evidence(batch):
    return {'replay.mask':batch.obs.presence_mask.clone(),'replay.next_mask':batch.next_obs.presence_mask.clone()}


def controller_evidence(state):
    from brain_uav.scripts.v2_gpu_td3_diagnostic import flatten
    result={}
    flatten(state['controls'][-1],'controller',result)
    result['beta']=torch.tensor(state['beta'],dtype=torch.float64)
    return result


def executor_evidence(executor):
    return dict(executor=executor,captures=executor.captures,replays=executor.replays)


def require_cached_replay(executor,before):
    if (executor is not before['executor'] or executor.captures!=before['captures']
            or executor.replays<=before['replays']):
        raise AssertionError('Expected cached Graph replay was not observed.')


def beta_target_evidence(batch,gamma,q1,q2,target,beta,previous_beta):
    from brain_uav.trainers.v2_td3 import research_target_pair
    expected=batch.reward+(1-batch.done)*gamma*research_target_pair(q1,q2,'acc',beta)
    stale=batch.reward+(1-batch.done)*gamma*research_target_pair(q1,q2,'acc',previous_beta)
    return dict(runtime_target_matches=bool(torch.allclose(target,expected,rtol=1e-4,atol=1e-5)),
        beta_refresh_distinguishable_on_same_target_q=not bool(torch.allclose(expected,stale,rtol=1e-4,atol=1e-5)))


def checked_bank_score(engine,scorer,rows):
    from brain_uav.trainers.v2_online_research import training_identity
    from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash
    for live,copy in zip((engine.critic1,engine.critic2),scorer.copies):
        if {p.data_ptr() for p in live.parameters()} & {p.data_ptr() for p in copy.parameters()}:
            raise AssertionError('Bank scorer shares training parameter storage.')
    before=state_hash(training_identity(engine))
    values=scorer(engine,rows)
    if before!=state_hash(training_identity(engine)):raise AssertionError('Bank scoring changed training state or RNG.')
    return values


def check(engine,config,pool,prepared,bc,model,*,output=None):
    import time
    from brain_uav.trainers.v2_online_research import atomic_json
    report=dict(execution_completed=False,strict_passed=False,engineering_accepted=False)
    started=time.perf_counter()
    try:
        report.update(_check(engine,config,pool,prepared,bc,model,report))
        return report
    except BaseException as exc:
        report.update(execution_completed=False,strict_passed=False,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['process_wall_seconds']=time.perf_counter()-started
        if output is not None:atomic_json(output,report)


def _check(engine,config,pool,prepared,bc,model,report):
    from brain_uav.scripts import diagnose_v2_mc_return_cuda as helper,profile_v2_td3 as profile
    from brain_uav.scripts.diagnose_v2_online_research import _append_update_evidence
    from brain_uav.scripts.v2_gpu_td3_diagnostic import flatten,snapshot
    from brain_uav.trainers.v2_formal_training import (build_v2_stage_engine,v2_bc_lambda,
        build_v2_periodic_snapshot,load_v2_periodic_snapshot)
    from brain_uav.trainers.v2_formal_acc import load_state
    from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch
    from brain_uav.trainers.v2_gpu_compute import GPUCompute
    # Public builder gives independent objects/optimizer state, no formal trainer loop.
    ref=build_v2_stage_engine(None,config,init_checkpoint=bc,device='cuda',model_type=model,
        prepared_initialization=prepared,fused_adam=True,reduce_update_stat_syncs=True,
        gpu_td3_compute=False,non_blocking_batch_transfer=True).engine
    ref.load_checkpoint_state_dict(deepcopy(engine.checkpoint_state_dict()))
    for e in (ref,engine):profile._set_numeric_engine_modes(e)
    warm=profile._compile_warmup_batches(pool=pool,prepared=prepared,batch_size=64,device=torch.device('cpu'),cache_ellipsoid_segment_clearance=True)
    obs,_=profile._numeric_diagnostic_observation_batches(warm,device=torch.device('cpu'))
    batch=V2ReplayBatch(obs,torch.zeros(64,2),torch.linspace(-2,2,64).reshape(-1,1),deepcopy(obs),
        torch.arange(64).remainder(2).float().reshape(-1,1),torch.zeros(64,1),torch.zeros(64,1),torch.ones(64,1))
    noise=torch.randn(64,2,generator=torch.Generator().manual_seed(config.seed)).cuda()
    checks=[];public=0;physical=0;old_beta=None;load_invalidated=False
    report['checks']=checks
    original=GPUCompute.compute
    def counted(self,*args,**kwargs):
        nonlocal physical
        physical+=1
        report['compute_physical_calls_including_capture']=physical
        if physical+public>128:raise RuntimeError('Bounded gate physical update budget exceeded.')
        return original(self,*args,**kwargs)
    with patch.object(GPUCompute,'compute',counted):
        for index,step in enumerate(GATE_STEPS):
            if config.acc_eta is not None and step in BETA_REFRESH_STEPS:
                from types import SimpleNamespace
                # Explicit complete synthetic diagnostic episode; never added to formal replay.
                rows=[SimpleNamespace(observation=None,action=torch.zeros(2).numpy(),
                    reward=1.,done=True,collected_stage_step=step)]
                # Use a genuine single observation from existing numeric warmup data.
                rows[0].observation=bank_observation(obs)
                states=[]
                for e in (ref,engine):
                    before=e.research_beta
                    if e is engine:old_beta=before
                    hook=e.formal_acc;scorer=hook.scorer
                    hook.scorer=lambda engine,bank:checked_bank_score(engine,scorer,bank)
                    try:hook.episode(SimpleNamespace(engine=e,result=SimpleNamespace(stage_steps=step)),rows)
                    finally:hook.scorer=scorer
                    if e.research_beta==before:raise AssertionError('ACC beta did not change; insufficient gate evidence.')
                    states.append(e.formal_acc.controller.state())
                controls=[controller_evidence(state) for state in states]
                control_match=helper.compare_snapshots(*controls)
                checks.append(dict(phase='bank_controller',**control_match,
                    isolated_scoring_verified=True,training_state_and_rng_unchanged=True))
                if not control_match['passed']:raise AssertionError('Bank/controller comparison failed.')
            snaps=[]
            reuse_before=executor_evidence(engine._gpu_update_executor) if step in BETA_REFRESH_STEPS else None
            for e in (ref,engine):
                tensors=batch_evidence(batch)
                def record(name,value):tensors[name]=value.detach().cpu().clone()
                _,pn,nc=config.noise_schedule.values(step-1,max_steps=500000);e.set_target_noise(policy_noise=pn,noise_clip=nc)
                with patch.object(e.replay,'sample',return_value=batch),patch.object(torch,'randn_like',new=lambda x:noise.to(x.device)), \
                        (helper.eager_evidence(e,record) if e is ref else nullcontext()):
                    public+=1
                    report['public_updates']=public
                    e.update_once(total_steps=step,bc_lambda=v2_bc_lambda(step-1,schedule='smooth-mid-strong'),diagnostic_tensor_recorder=record)
                snap,_=snapshot(e)
                for name in ('actor','critic1','critic2'):
                    for parameter_name,p in getattr(e,name).named_parameters():
                        flatten(p.grad,f'gradient.{name}.{parameter_name}',snap)
                _append_update_evidence(snap,tensors,model);snaps.append(snap)
            if reuse_before is not None:
                require_cached_replay(engine._gpu_update_executor,reuse_before)
                checks.append(dict(phase='cached_beta_replay',step=step,passed=True,
                    captures=engine._gpu_update_executor.captures,replays=engine._gpu_update_executor.replays))
            comparison=helper.compare_snapshots(*snaps);checks.append(dict(phase='update',step=step,**comparison))
            if config.acc_eta is not None and old_beta is not None:
                q1,q2,target=[snaps[1]['tensor.target.'+k] for k in ('q1','q2','td')]
                evidence=beta_target_evidence(batch,config.gamma,q1,q2,target,engine.research_beta,old_beta)
                matches=evidence['runtime_target_matches']
                distinguishable=evidence['beta_refresh_distinguishable_on_same_target_q']
                checks.append(dict(phase='actual_beta_target',step=step,passed=matches and distinguishable,
                    runtime_target_matches=matches,beta_refresh_distinguishable_on_same_target_q=distinguishable))
                if not matches or not distinguishable:raise AssertionError('New beta target mismatch or insufficient distinguishability.')
            if not comparison['passed']:raise AssertionError(f'Strict gate failed at {step}; no engineering exemptions.')
            if index==6:
                old=engine._gpu_update_executor
                for e in (ref,engine):
                    if config.acc_eta is None:
                        e.load_checkpoint_state_dict(deepcopy(e.checkpoint_state_dict()))
                    else:
                        payload=build_v2_periodic_snapshot(e,config,stage_steps=step,
                            scenario=prepared.scenario_config,rewards=prepared.reward_config,
                            uav_collision_radius=prepared.uav_collision_radius,seed_manifest={'base_seed':config.seed},
                            initialization_source={'kind':'bounded_gate'},bc_schedule='smooth-mid-strong')
                        restored=load_v2_periodic_snapshot(payload,expected_model_type=model)
                        load_state(e,config,restored)
                        from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash
                        if (e.research_beta!=restored['acc_state']['controller']['beta'] or
                                state_hash(e.formal_acc.controller.state())!=state_hash(restored['acc_state']['controller'])):
                            raise AssertionError('Loaded ACC beta/controller differs before rebuild update.')
                if engine._gpu_update_executor is not None or old.graphs:raise AssertionError('Load did not invalidate graphs.')
                load_invalidated=True
    if not all(c['passed'] for c in checks):raise AssertionError('Controller gate failed.')
    if not load_invalidated or engine._gpu_update_executor is None or engine._gpu_update_executor.replays<=0:
        raise AssertionError('Missing actual rebuilt Graph replay evidence.')
    graph=engine.local_optimization_metadata()
    if not graph['non_blocking_batch_transfer']['effective'] or graph['gpu_compute_non_blocking_coefficients']['effective']:
        raise AssertionError('Observed transfer path differs from the gate contract.')
    return dict(execution_completed=True,strict_passed=True,engineering_accepted=False,checks=checks,load_invalidated=True,
        public_updates=public,compute_physical_calls_including_capture=physical,budget=128,
        fixture='complete single-transition synthetic bank; not historical training recovery',
        compilation_preparation='separate formal prewarm, no training loop',graph=graph)
