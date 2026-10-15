"""Bounded serial real CPU replay / full-compute CUDA correctness and A/B."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
from contextlib import nullcontext
import json
import random
import subprocess
import sys
import traceback
import numpy as np
import torch

CONTRACT_VERSION = 2

def _prepare_loaded_bc_reference(engine, batches, compilation):
    """Rebind only the newly deep-copied BC actor; retain live engine wrappers."""
    from brain_uav.models.v2_ann import V2ANNPolicyActor
    actor = engine.bc_reference_actor
    if actor is None:
        return
    # The loader deep-copies the online actor, including its runtime wrappers.
    # Those wrappers are not a fresh BC-role compilation of the new module.
    encoder = actor.zone_set_encoder
    encoder._compiled_tensor_forward = None
    encoder._compiled_tensor_forward_config = None
    encoder._compiled_online_actor_contexts = {}
    encoder._compiled_shared_relations = None
    encoder._compiled_shared_relations_config = None
    if isinstance(actor, V2ANNPolicyActor):
        actor._compiled_full_forward = None
        actor._compiled_full_forward_config = None
    # Manual compute uses raw tensor functions, not a compiled BC callable.
    if engine.gpu_td3_compute or not compilation['compile_actors']:
        return
    options = dict(backend='inductor', mode='default', fullgraph=True, dynamic=True)
    if compilation['cuda_graph_actor_update']:
        options.update(mode=None, options={'triton.cudagraphs': True})
    if isinstance(actor, V2ANNPolicyActor):
        actor.enable_compiled_full_forward(**options)
    else:
        encoder.enable_compiled_tensor_forward(role='bc_reference', **options)
    with torch.no_grad():
        for batch in batches:
            engine._mark_cuda_graph_step_begin()
            actor(batch, shared_relations=engine._build_shared_relations(batch))

def correctness(args, *, prepared, config, pool, device, output, full_pool_digest):
    """Fixed noise, original CPU replay, exact error-state protection; no training."""
    from brain_uav.scripts import profile_v2_td3 as p
    from brain_uav.scripts.train_v2_td3 import _configure_stage_compilation, default_v2_cuda_graph_compilation
    from brain_uav.scripts.v2_batch_transfer_diagnostic import cpu_batches
    from brain_uav.scripts.v2_gpu_td3_diagnostic import snapshot, flatten, complete_actor_evidence
    from brain_uav.trainers.v2_gpu_compute import tensors
    from brain_uav.observations import V2Observation
    report = dict(passed=False, completed=False, checks=[], location={},rtol=1e-4,atol=1e-5,
        diagnostic_only=True,requested=args.gpu_td3_compute,
        scope='original CPU replay, dynamic cropping; fixed-noise then continuous original RNG')
    compilation = default_v2_cuda_graph_compilation(model=args.model,resolved_device='cuda')
    fused = compilation.pop('fused_adam'); reduced = compilation.pop('reduce_update_stat_syncs')
    contract = dict(version=CONTRACT_VERSION,model=args.model,config=config.to_dict(),
        rewards=asdict(prepared.reward_config),seed=args.seed,pool_digest=pool.content_digest,
        checkpoint=str(Path(args.bc_checkpoint).resolve()),schedule=args.comparison_bc_schedule)
    if not 8 <= args.correctness_updates <= 16:
        raise ValueError('Compute correctness supports 8..16 controlled steps only.')
    contract['fixed_steps'] = p._td3_block_correctness_steps(config,args.correctness_updates)
    contract['checkpoint_sha256'] = sha256(Path(args.bc_checkpoint).read_bytes()).hexdigest()
    contract['full_pool_digest'] = full_pool_digest
    contract['pool_file_sha256'] = sha256(Path(args.validation_pool).read_bytes()).hexdigest()
    report['inputs'] = contract
    report['code'] = {name:subprocess.run(command,capture_output=True,text=True,check=False).stdout.strip()
        for name,command in (('head',['git','rev-parse','HEAD']),('dirty',['git','status','--short']))}
    reference = None
    if args.gpu_td3_compute:
        reference = torch.load(args.td3_block_reference,map_location='cpu',weights_only=False)
        if reference.get('contract') != contract: raise ValueError('Compute reference contract mismatch; rebuild baseline.')
    engine = None
    try:
        report['location']['phase'] = 'initialization'
        engine = p.build_v2_stage_engine(None,config,init_checkpoint=args.bc_checkpoint,
            rewards=prepared.reward_config,prepared_initialization=prepared,device=device,
            model_type=args.model,snn_time_window=args.snn_time_window,fused_adam=fused,
            reduce_update_stat_syncs=reduced,gpu_td3_compute=args.gpu_td3_compute).engine
        initial = reference['initial'] if reference else deepcopy(engine.checkpoint_state_dict())
        engine.load_checkpoint_state_dict(initial)
        configured = _configure_stage_compilation(engine,pool,prepared,
            compile_critic_encoder=False,compile_target_encoders=False,**compilation)
        batches = p._compile_warmup_batches(pool=pool,prepared=prepared,batch_size=engine.batch_size,
            device=device,cache_ellipsoid_segment_clearance=True)
        mixed,nonempty = p._numeric_diagnostic_observation_batches(batches,device=device)
        fixed = cpu_batches(mixed,nonempty,engine.action_dim,args.seed)
        fixed.append(replace(fixed[0],next_obs=fixed[1].next_obs))
        sample_original = engine.replay.sample
        collected = {}
        def collect(name,value): collected[name] = torch.as_tensor(value).detach().cpu().clone()
        if not args.gpu_td3_compute:
            check_original = engine._validate_and_clip_gradients
            names = {id(param): module+'.'+name for module in ('actor','critic1','critic2')
                for name,param in getattr(engine,module).named_parameters()}
            def check(parameters,**kwargs):
                for param in parameters:
                    if param.grad is not None: collect('gradient.pre_clip.'+names[id(param)],param.grad)
                return check_original(parameters,**kwargs)
            engine._validate_and_clip_gradients = check
            if args.model == 'snn':
                forward = engine.actor.forward_with_activity
                def activity(*a,**kw):
                    actions,rate = forward(*a,**kw); collect('actor.action',actions)
                    return actions,rate
                engine.actor.forward_with_activity = activity
            else: engine.actor.register_forward_hook(lambda m,a,out: collect('actor.action',out))
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        snapshots,contexts = [],[]

        def run_step(total,batch=None,noise=None,failure=None):
            index = len(snapshots); collected.clear()
            report['location'] = dict(phase='update',index=index,steps=total,failure=failure,
                current_zones=None if batch is None else batch.obs.max_zone_count,
                next_zones=None if batch is None else batch.next_obs.max_zone_count)
            def sample(size):
                actual = sample_original(size) if batch is None else batch
                for i,t in enumerate(tensors(actual)): collect('batch.'+str(i),t)
                collect('batch.failure_fraction',actual.sample_failure_fraction)
                collect('replay.mask',actual.obs.presence_mask); collect('replay.next_mask',actual.next_obs.presence_mask)
                return actual
            engine.replay.sample = sample
            _,policy_noise,noise_clip = config.noise_schedule.values(total,max_steps=config.max_steps)
            engine.set_target_noise(policy_noise=policy_noise,noise_clip=noise_clip)
            from unittest.mock import patch
            def fixed_noise(t,**kwargs): return noise.to(t.device)
            noise_context = patch('torch.randn_like',new=fixed_noise) if noise is not None else nullcontext()
            old_loss = engine._require_finite_loss
            injection = p._inject_diagnostic_inf_gradient(engine,
                component='actor' if failure and failure.startswith('actor') else 'critic',evidence={}) \
                if failure and failure.endswith('gradient') and not args.gpu_td3_compute else nullcontext()
            if failure == 'actor_loss' and not args.gpu_td3_compute:
                def loss_check(loss,*,component,total_steps):
                    return old_loss(loss*float('nan') if component == 'actor' else loss,component=component,total_steps=total_steps)
                engine._require_finite_loss = loss_check
            if args.gpu_td3_compute and failure:
                engine._gpu_update_executor.injections[('critic_loss','critic_gradient','actor_loss','actor_gradient').index(failure)] = 1
            metric = None
            try:
                with noise_context,injection:
                    metric = engine.update_once(total_steps=total,
                        bc_lambda=p.v2_bc_lambda(total,schedule=args.comparison_bc_schedule),diagnostic_tensor_recorder=collect)
                if failure: raise AssertionError('Injected error not rejected: '+failure)
            except FloatingPointError as exc:
                if not failure or not str(exc).startswith('Non-finite '+failure.split('_')[0]): raise
            finally:
                engine._require_finite_loss = old_loss
                if args.gpu_td3_compute and engine._gpu_update_executor is not None: engine._gpu_update_executor.injections.zero_()
            actor_backward = (metric.actor_updated if metric is not None else failure == 'actor_gradient')
            snap,evidence = snapshot(engine,metric)
            if engine.bc_reference_actor is not None:
                flatten(engine.bc_reference_actor.state_dict(),'bc_reference',snap)
            for model in (engine.actor_target,engine.critic1_target,engine.critic2_target,engine.bc_reference_actor):
                if model is not None and any(param.grad is not None for param in model.parameters()):
                    raise AssertionError('Target/BC parameter gradient leaked.')
            if metric is None:
                gradients,evidence = p._td3_gradient_snapshot(engine,actor_backward=actor_backward)
                snap.update(gradients)
            else:
                collected.pop('activity.l1',None); collected.pop('activity.l2',None)
                complete_actor_evidence(collected,args.model)
                snap.update({'tensor.'+n:t for n,t in collected.items()})
            # Reuse strictly equal immutable CPU snapshots, reducing reference
            # storage without dropping fields or fabricating gradient values.
            if snapshots:
                for name,t in snap.items():
                    previous = snapshots[-1].get(name)
                    if previous is not None and t.shape==previous.shape and t.dtype==previous.dtype and torch.equal(t,previous):
                        snap[name] = previous
            snap['rng.replay'] = torch.tensor(list(json.dumps(engine.replay.rng.bit_generator.state,sort_keys=True).encode()),dtype=torch.uint8)
            snap['rng.cpu'] = torch.get_rng_state().clone(); snap['rng.cuda'] = torch.cuda.get_rng_state(device).cpu().clone()
            if args.model == 'snn':
                for actor in (engine.actor,engine.actor_target,engine.bc_reference_actor):
                    if actor is not None:
                        for node in (actor.snn_head.lif1,actor.snn_head.lif2):
                            if isinstance(node.v,torch.Tensor): assert torch.count_nonzero(node.v) == 0
                            else: assert node.v == 0
            context = dict(snapshot_index=index,stage_steps=total,actor_backward=actor_backward,gradients=evidence,failure=failure)
            if reference is not None:
                expected = reference['snapshots'][index]
                for name,t in snap.items():
                    if name.startswith(('rng.','counts.','tensor.batch.')) or name.endswith('.step'):
                        torch.testing.assert_close(t,expected[name],rtol=0,atol=0)
                    if failure and name.startswith('grad.') and t.is_floating_point():
                        for pred in (torch.isnan,torch.isposinf,torch.isneginf):
                            if not torch.equal(pred(t),pred(expected[name])): raise AssertionError('Nonfinite gradient differs: '+name)
                        # Error gradients are part of this explicit error-state
                        # contract, including retained values on loss failures.
                        finite = torch.isfinite(expected[name])
                        torch.testing.assert_close(t[finite],expected[name][finite],rtol=1e-4,atol=1e-5)
                p._compare_td3_block_snapshot(expected,snap,context=context,report=report,output=output)
            snapshots.append(snap); contexts.append(context)
            report['checks'].append(dict(index=index,total_steps=total,failure=failure,passed=True))
            return snap

        generator = torch.Generator().manual_seed(args.seed)
        profiled_modes = set()
        for i,total in enumerate(contract['fixed_steps']):
            controlled = torch.randn(config.batch_size,engine.action_dim,generator=generator)
            mode = 'actor' if total%config.policy_delay == 0 and total>config.actor_freeze_steps else 'frozen_targets' if total%config.policy_delay == 0 else 'critic_only'
            if args.gpu_td3_compute and mode not in profiled_modes:
                from brain_uav.trainers.v2_gpu_compute import GPUCompute
                if engine._gpu_update_executor is None:
                    engine._gpu_update_executor = GPUCompute(engine,diagnostic=True)
                executor = engine._gpu_update_executor
                upload = fixed[i%len(fixed)].to(device)
                _,pn,nc = config.noise_schedule.values(total,max_steps=config.max_steps)
                engine.set_target_noise(policy_noise=pn,noise_clip=nc)
                executor.coefficients.copy_(torch.tensor([p.v2_bc_lambda(total,schedule=args.comparison_bc_schedule),
                    engine.spike_regularization.beta(total),pn,nc],device=device))
                key = executor.key(upload,mode)
                if key not in executor.graphs: executor.capture_shape(key,upload)
                evidence = engine._profile_cuda_graph_replay(lambda: run_step(total,fixed[i%len(fixed)],controlled))
                report.setdefault('manual_replay_evidence',{})[mode] = evidence
                if evidence['cuda_graph_launch_count'] < 1: raise AssertionError('No actual CUDA Graph launch: '+mode)
                profiled_modes.add(mode)
            else:
                run_step(total,fixed[i%len(fixed)],controlled)
        transitions,refs = [],[]
        for i in range(3*config.batch_size):
            source = fixed[i%len(fixed)]; row = i%config.batch_size
            def observation(obs):
                active = obs.presence_mask[row].numpy(); zones = obs.zone_features[row].numpy()[active]
                return V2Observation(obs.ego_features[row].numpy(),obs.goal_features[row].numpy(),zones,np.ones(len(zones),np.bool_))
            transition = (observation(source.obs),source.action[row].numpy(),float(i%11)/10.,
                observation(source.next_obs),i%config.batch_size==config.batch_size-1)
            refs.append(engine.replay.add(*transition,near_goal=i%2==0)); transitions.append(transition)
        engine.replay.mark_failure_slots(refs[:config.batch_size])
        engine.replay.mark_success_slots(refs[config.batch_size:2*config.batch_size])
        for transition in transitions[config.batch_size:2*config.batch_size]: engine.replay.add_success_transition(*transition,near_goal=True)
        last = contract['fixed_steps'][-1]
        for total in range(last+1,last+7): run_step(total)
        last += 6
        for failure in ('critic_loss','critic_gradient','actor_loss','actor_gradient'):
            success = last+config.policy_delay-(last%config.policy_delay)
            run_step(success,fixed[0])
            protected,_ = snapshot(engine)
            error_batch = replace(fixed[0],reward=torch.full_like(fixed[0].reward,float('nan'))) if failure == 'critic_loss' else fixed[0]
            last = success+config.policy_delay
            actual = run_step(last,error_batch,failure=failure)
            prefixes = ('actor.','actor_target.','optimizer.actor.','counts.actor_update_count') if failure.startswith('actor') else ('actor.','critic1.','critic2.','actor_target.','critic1_target.','critic2_target.','optimizer.','counts.')
            keep = {n:t for n,t in protected.items() if n.startswith(prefixes)}
            p._compare_compiled_numeric_tensors(keep,{n:actual[n] for n in keep},rtol=0,atol=0)
            report['checks'][-1]['protected_state_exact'] = True
        # Supported load entry point must drop every old graph, then rebuild
        # from the same initial engine state; replay remains the CPU authority.
        report['location'] = {'phase':'checkpoint_load'}
        old_executor = engine._gpu_update_executor
        engine.set_target_noise(policy_noise=initial['algorithm_config']['policy_noise'],
            noise_clip=initial['algorithm_config']['noise_clip'])
        engine.load_checkpoint_state_dict(initial)
        if args.gpu_td3_compute and (engine._gpu_update_executor is not None or old_executor.graphs):
            raise AssertionError('Checkpoint load did not invalidate old compute graphs.')
        # All six live networks retain their wrappers; only BC was rebuilt.
        _prepare_loaded_bc_reference(engine,batches,compilation)
        run_step(last+config.policy_delay,fixed[3])
        report['load_rebuild_verified'] = True
        if args.gpu_td3_compute and profiled_modes != {'critic_only','frozen_targets','actor'}:
            raise AssertionError('Three manual update modes were not verified.')
        report.update(passed=True,completed=True,compilation=configured,
            effective=engine.local_optimization_metadata(),snapshot_count=len(snapshots),
            graph_evidence_scope='one actual prewarmed update per mode profiled; not every update individually profiled')
        if reference is None: torch.save(dict(contract=contract,initial=initial,snapshots=snapshots,contexts=contexts),output/'reference.pt')
    except BaseException as exc:
        report['failure'] = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),gpu_context=getattr(exc,'gpu_compute_context',None))
        raise
    finally:
        if engine is not None: engine.invalidate_gpu_update()
        p._strict_json_write(output/'correctness.json',report)
    return report


def run(args):
    from brain_uav.scripts import profile_v2_td3 as p
    conflicts=('gpu_td3_update','gpu_td3_suite','gpu_td3_worker','gpu_td3_sampling_cost',
        'pinned_batch_transfer','compile_critic_update_front','compile_actor_objective','cuda_graph_critic_tail',
        'window_update_statistics','reuse_update_buffers','aggregate_relation_values_first','single_observation_fast_path',
        'non_blocking_batch_transfer','batch_transfer_comparison','batch_transfer_worker','td3_block_suite','td3_block_correctness','td3_block_comparison','td3_block_attribution',
        'local_optimization_correctness','local_optimization_comparison','critic_tail_correctness','critic_tail_comparison',
        'snn_path_consistency','detailed_profiler_updates','compiled_performance_diagnostic_updates')
    if any(getattr(args,n) for n in conflicts) or args.device!='cuda' or args.comparison_eager_cpu:
        raise ValueError('GPU compute comparison requires CUDA and original formal path, without other experiments/profiler.')
    if args.bc_checkpoint is None or args.validation_pool is None:
        raise ValueError('Matching BC and easy fixed pool are required.')
    if args.gpu_compute_worker:
        args.td3_block_choice='compute' if args.gpu_td3_compute else 'baseline'
        args.td3_block_correctness=args.gpu_compute_worker=='correctness'
        args.td3_block_comparison=args.gpu_compute_worker=='timing'
        args.correctness_updates=max(8,args.correctness_updates)
        return p.run_local_optimization_comparison(args)
    if args.gpu_td3_compute:
        raise ValueError('The serial driver owns baseline/compute selection.')
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Refusing nonempty comparison directory.')
    output.mkdir(parents=True,exist_ok=True)
    report={'diagnostic_only':True,'passed':False,'model':args.model,'groups':{}}
    report['timing_scope']='Original CPU sample/build + original upload + added GPU input copies/selection + full compute + terminal state read. Region walls are CPU submission/wait, not GPU kernel time. TD3 and environment totals include the existing final-window wait; no profiler in throughput.'
    forwarded=('model','bc_checkpoint','validation_pool','validation_seed','seed','batch_size','replay_capacity',
        'scenario_count','warmup_steps','steps_per_level','repeats','controlled_stage_step_offset',
        'comparison_bc_schedule','snn_time_window','correctness_updates')
    if args.expected_pool_digest: forwarded += ('expected_pool_digest',)
    baseline_ok=False
    try:
        # Both correctness processes complete before either throughput worker starts.
        for enabled in (False,True):
            group='compute' if enabled else 'baseline'
            item=report['groups'][group]={'requested':enabled,'correctness_passed':False}
            if enabled and not baseline_ok: item['skipped']='baseline correctness failed'; continue
            command=[sys.executable,'-u','-m','brain_uav.scripts.profile_v2_td3','--device','cuda']
            for name in forwarded: command+=['--'+name.replace('_','-'),str(getattr(args,name))]
            if enabled: command+=['--gpu-td3-compute','--td3-block-reference',str(output/'baseline_correctness'/'reference.pt')]
            with (output/(group+'_correctness.log')).open('w',encoding='utf-8') as log:
                rc=subprocess.run(command+['--gpu-compute-worker','correctness','--output-dir',str(output/(group+'_correctness'))],
                    stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            item['correctness_exit_code']=rc; item['correctness_passed']=rc==0
            if not enabled: baseline_ok=rc==0
        if not all(v['correctness_passed'] for v in report['groups'].values()):
            raise RuntimeError('CPU sampling/full compute correctness failed; this model timing skipped.')
        for enabled in (False,True):
            group='compute' if enabled else 'baseline'
            command=[sys.executable,'-u','-m','brain_uav.scripts.profile_v2_td3','--device','cuda']
            for name in forwarded: command+=['--'+name.replace('_','-'),str(getattr(args,name))]
            if enabled: command+=['--gpu-td3-compute']
            with (output/(group+'_timing.log')).open('w',encoding='utf-8') as log:
                rc=subprocess.run(command+['--gpu-compute-worker','timing','--output-dir',str(output/(group+'_timing'))],
                    stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            report['groups'][group]['timing_exit_code']=rc
            if rc: raise RuntimeError('Timing worker failed: '+group)
            data=json.loads((output/(group+'_timing')/'local_optimization_summary.json').read_text())
            if not data['valid_for_speed_comparison']: raise RuntimeError('Measurement compiled new graphs: '+group)
            report['groups'][group]['effective']=data['runs'][group][0]['local_optimizations']['gpu_td3_compute']
            runs=data['runs'][group]
            if enabled and any(r['local_optimizations']['gpu_td3_compute']['measurement_new_captures'] for r in runs):
                raise RuntimeError('New captures during measurement; not steady state.')
            report['groups'][group]['raw_runs']=[dict(environment_seconds=r['timing']['total_wall_seconds'],
                td3_seconds=r['timing']['td3_update_wall_seconds'],
                completion=r['batch_transfer_window_completion'],
                batch_submission=r['timing']['td3_update_breakdown']['overall_weighted']['sections']['batch_preparation'],
                measured_steps=r['measured_steps'],compilation=r['critic_encoder_compile'],
                compute=r['local_optimizations'].get('gpu_td3_compute')) for r in runs]
            report['groups'][group]['mean_step_seconds']=mean(r['timing']['total_wall_seconds']/r['measured_steps'] for r in runs)
            report['groups'][group]['stdev_step_seconds']=stdev(r['timing']['total_wall_seconds']/r['measured_steps'] for r in runs) if len(runs)>1 else 0.
            report['groups'][group]['mean_td3_seconds']=mean(r['batch_transfer_window_completion']['td3_wall_including_final_window_wait_seconds'] for r in runs)
        a,b=report['groups']['baseline'],report['groups']['compute']
        report['time_saved_percent']=100*(1-b['mean_step_seconds']/a['mean_step_seconds'])
        report['overall_speedup_observed']=b['mean_step_seconds']<a['mean_step_seconds'] and b['mean_td3_seconds']<a['mean_td3_seconds']
        report['repeat_fluctuation_seconds']=a['stdev_step_seconds']+b['stdev_step_seconds']
        report['verdict']=('收益未确定：改善未超过两侧重复波动' if a['mean_step_seconds']-b['mean_step_seconds']<=report['repeat_fluctuation_seconds'] else '两项整步指标改善；短程结果仍需审查') if report['overall_speedup_observed'] else '未取得整体加速'
        report['passed']=True
    finally: p._strict_json_write(output/'summary.json',report)
    return report
