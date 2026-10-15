"""Bounded serial real CPU replay / full-compute CUDA correctness and A/B."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
from contextlib import nullcontext, contextmanager, ExitStack
import json
import random
import subprocess
import sys
import traceback
import numpy as np
import torch

CONTRACT_VERSION = 5
RETAINED_PARAMETER = 'actor.zone_set_encoder.empty_scene_token'


@contextmanager
def trace_retained_gradient(engine, events, *, enabled, context):
    """One named gradient only; optional synchronous copies can perturb timing."""
    if not enabled:
        yield
        return
    from unittest.mock import patch
    parameter = engine.actor.zone_set_encoder.empty_scene_token

    def sample(boundary, **details):
        grad = parameter.grad
        events.append(dict(context, boundary=boundary, parameter=RETAINED_PARAMETER,
            **details, ordinal=len(events),address=None if grad is None else grad.data_ptr(),
            value=None if grad is None else grad.detach().cpu().clone()))

    with ExitStack() as stack:
        def wrap(obj, name):
            original = getattr(obj,name,None)
            if original is None:
                return
            def observed(*a,**kw):
                details = dict(component=kw['component']) if 'component' in kw else {}
                sample(name+'.before',**details)
                try:
                    return original(*a,**kw)
                finally:
                    sample(name+'.after',**details)
            stack.enter_context(patch.object(obj,name,new=observed))
        for name in ('_mark_cuda_graph_step_begin','_build_shared_relations',
                     '_compiled_target_block','_compiled_target_critic_td','_compiled_critic_loss',
                     '_require_finite_loss','_validate_and_clip_gradients'):
            wrap(engine,name)
        # SNN target forward is outside its compiled target-critic block.
        if engine._compiled_target_block is None:
            wrap(engine.actor_target,'forward')
        wrap(engine.replay,'sample')
        wrap(engine.actor_optimizer,'step')
        sample('update.before')
        try:
            yield
        finally:
            sample('update.after')


def retained_trace_summary(events):
    failing = [event for event in events if event.get('failure') == 'critic_loss']
    anchor = next((event for event in reversed(events)
        if event.get('failure') is None and event['boundary']=='update.after'),None)
    origin = anchor or (failing[0] if failing else None)
    first = None
    for event in (failing if anchor else failing[1:]):
        a,b = origin['value'],event['value']
        if (a is None) != (b is None) or (a is not None and not torch.equal(a,b)):
            first = {k:v for k,v in event.items() if k != 'value'}
            break
    return dict(parameter=RETAINED_PARAMETER, sampled_boundaries=len(events),first_observed_change=first,
        origin=None if origin is None else {k:v for k,v in origin.items() if k != 'value'},
        warning='CPU snapshots synchronize and may perturb the phenomenon; non-reproduction is not a fix.')


def error_state(engine):
    """Diagnostic-only owned CPU values; never rebind or mutate live gradients."""
    from brain_uav.scripts.v2_gpu_td3_diagnostic import snapshot, flatten
    values, _ = snapshot(engine)
    if engine.bc_reference_actor is not None:
        flatten(engine.bc_reference_actor.state_dict(),'bc_reference',values)
    gradients, storage = {}, {}
    executor = engine._gpu_update_executor
    for module in ('actor', 'critic1', 'critic2', 'actor_target', 'critic1_target',
                   'critic2_target', 'bc_reference_actor'):
        model = getattr(engine, module)
        if model is None:
            continue
        for name, parameter in model.named_parameters():
            key = 'gradient.' + module + '.' + name
            grad = parameter.grad
            gradients[key] = dict(present=grad is not None, shape=list(parameter.shape),
                dtype=str(parameter.dtype), gradient_dtype=None if grad is None else str(grad.dtype),
                gradient_shape=None if grad is None else list(grad.shape))
            values[key if grad is not None else key + '.none'] = (
                torch.tensor(True) if grad is None else grad.detach().cpu().clone())
            storage[key] = None if grad is None else dict(address=grad.data_ptr(),
                banks=[] if executor is None else [(group, bank, index)
                    for group, banks in enumerate(executor.banks)
                    for bank, tensors in enumerate(banks) for index, tensor in enumerate(tensors)
                    if tensor is grad])
    return dict(values=values, gradients=gradients, storage=storage,
        published=None if executor is None else list(executor.published),
        graph_keys=[] if executor is None else [list(key) for key in executor.graphs])


def state_difference(expected, actual, *, exact=False, evidence=None, label=''):
    """Check every field before reporting failure, including nonfinite patterns."""
    differences = []
    for name in sorted(expected.keys() | actual.keys()):
        a, b = expected.get(name), actual.get(name)
        item = dict(field=name)
        if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype:
            item.update(reason='field/shape/dtype mismatch',
                expected=None if a is None else dict(shape=list(a.shape), dtype=str(a.dtype)),
                actual=None if b is None else dict(shape=list(b.shape), dtype=str(b.dtype)))
        else:
            strict = (exact or not a.is_floating_point() or name.endswith('.step')
                or name.startswith(('counts.','rng.','tensor.batch.')))
            finite = torch.isfinite(a) & torch.isfinite(b)
            same = (a == b) if strict else torch.isclose(a, b, rtol=1e-4, atol=1e-5)
            same |= torch.isnan(a) & torch.isnan(b)
            if bool(same.all()):
                continue
            delta = (a[finite].double() - b[finite].double()).abs()
            def distribution(t):
                return {key:int(fn(t).sum()) for key,fn in (
                    ('finite',torch.isfinite),('nan',torch.isnan),
                    ('positive_inf',torch.isposinf),('negative_inf',torch.isneginf))}
            item.update(reason='value mismatch', shape=list(a.shape), dtype=str(a.dtype),
                elements=a.numel(), mismatches=int((~same).sum()),
                maximum_finite_absolute_error=float(delta.max()) if delta.numel() else None,
                expected_nonfinite=distribution(a), actual_nonfinite=distribution(b))
        differences.append(item)
        # At most four small representative pairs across all comparisons.
        if evidence is not None and len(evidence) < 4:
            evidence[label + ':' + name] = dict(
                expected=None if a is None else a.flatten()[:4096].clone(),
                actual=None if b is None else b.flatten()[:4096].clone(),
                truncated=bool((a is not None and a.numel()>4096) or (b is not None and b.numel()>4096)))
    return dict(passed=not differences, checked_fields=len(expected.keys() | actual.keys()),
        first_difference=next(iter(differences),None), differences=differences)


def record_error_state(output, index, before, after, expected=None):
    """Persist both endpoints once and compact all own/cross-side differences."""
    evidence = {}
    report = dict(own_before_after=state_difference(before['values'],after['values'],exact=True,
        evidence=evidence,label='own'), gradient_inventory=dict(before=before['gradients'],after=after['gradients']),
        storage=dict(before=before['storage'],after=after['storage']),
        banks=dict(before=before['published'],after=after['published']),
        graph_keys=dict(before=before['graph_keys'],after=after['graph_keys']))
    if expected is not None:
        for phase,actual in (('before',before),('after',after)):
            report['cross_'+phase] = state_difference(expected[phase]['values'],actual['values'],
                evidence=evidence,label='cross_'+phase)
    path = output / f'error_state_{index}.pt'
    torch.save(dict(before=before,after=after),path)
    report['snapshot_path'] = str(path)
    if evidence:
        path = output / f'error_differences_{index}.pt'
        torch.save(evidence,path); report['difference_evidence_path'] = str(path)
    from brain_uav.scripts.profile_v2_td3 import _strict_json_write
    _strict_json_write(output / f'error_state_{index}.json',report)
    return report

def _prepare_loaded_bc_reference(engine, batches, compilation):
    """Rebind only the newly deep-copied BC actor; retain live engine wrappers."""
    from brain_uav.scripts.profile_v2_td3 import _set_numeric_engine_modes
    from brain_uav.models.v2_ann import V2ANNPolicyActor
    _set_numeric_engine_modes(engine)
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
    contract['first_critic_loss_only'] = getattr(args,'gpu_compute_first_critic_loss_only',False)
    contract['error_recovery'] = getattr(args,'gpu_compute_error_recovery',False)
    contract['trace_retained_gradient'] = getattr(args,'gpu_compute_trace_retained_gradient',False)
    if (contract['error_recovery'] or contract['trace_retained_gradient']) and not contract['first_critic_loss_only']:
        raise ValueError('Recovery/retained-gradient trace requires --gpu-compute-first-critic-loss-only.')
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
    trace_events = []
    try:
        report['location']['phase'] = 'initialization'
        engine = p.build_v2_stage_engine(None,config,init_checkpoint=args.bc_checkpoint,
            rewards=prepared.reward_config,prepared_initialization=prepared,device=device,
            model_type=args.model,snn_time_window=args.snn_time_window,fused_adam=fused,
            reduce_update_stat_syncs=reduced,gpu_td3_compute=args.gpu_td3_compute).engine
        initial = reference['initial'] if reference else deepcopy(engine.checkpoint_state_dict())
        engine.load_checkpoint_state_dict(initial)
        p._set_numeric_engine_modes(engine)
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
        snapshots,contexts,error_states = [],[],{}
        trace_events, deferred_failures = [],[]

        def run_step(total,batch=None,noise=None,failure=None,*,recovery=False,trace=False):
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
            before_error = error_state(engine) if failure else None
            try:
                with noise_context,injection,trace_retained_gradient(engine,trace_events,
                        enabled=trace,context=dict(index=index,steps=total,failure=failure)):
                    metric = engine.update_once(total_steps=total,
                        bc_lambda=p.v2_bc_lambda(total,schedule=args.comparison_bc_schedule),diagnostic_tensor_recorder=collect)
                if failure: raise AssertionError('Injected error not rejected: '+failure)
            except FloatingPointError as exc:
                if not failure or not str(exc).startswith('Non-finite '+failure.split('_')[0]): raise
            finally:
                engine._require_finite_loss = old_loss
                if args.gpu_td3_compute and engine._gpu_update_executor is not None: engine._gpu_update_executor.injections.zero_()
            if failure:
                after_error = error_state(engine)
                expected_error = None
                if reference is not None:
                    expected_error = torch.load(Path(args.td3_block_reference).parent /
                        reference['error_states'][index],map_location='cpu',weights_only=False)
                state_report = record_error_state(output,index,before_error,after_error,expected_error)
                error_states[index] = Path(state_report['snapshot_path']).name
                report.setdefault('error_states',{})[str(index)] = state_report
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
            if reference is not None and not failure and not recovery:
                expected = reference['snapshots'][index]
                for name,t in snap.items():
                    if name.startswith(('rng.','counts.','tensor.batch.')) or name.endswith('.step'):
                        torch.testing.assert_close(t,expected[name],rtol=0,atol=0)
                p._compare_td3_block_snapshot(expected,snap,context=context,report=report,output=output)
            if reference is not None and failure:
                # Preserve the original snapshot/RNG/counter contract too, but
                # never stop before every error-state field is checked/saved.
                report['error_states'][str(index)]['cross_snapshot'] = state_difference(reference['snapshots'][index],snap)
            if recovery:
                item = dict(index=index,stage_steps=total,actor_backward=actor_backward,
                    diagnostic_only=True, production_resumption_supported=False,
                    comparison='baseline_reference_only' if reference is None else 'baseline_vs_compute',
                    scope='Continuation of interrupted diagnostic engine only; not a production recovery contract.')
                if reference is not None:
                    expected = reference['snapshots'][index]
                    differences = {}
                    item['all_fields'] = state_difference(expected,snap,evidence=differences,label='recovery')
                    # Keep residuals in all_fields; separately show only consumed
                    # backward gradients plus all persistent state/loss evidence.
                    active = {n:t for n,t in expected.items() if actor_backward or not n.startswith('grad.actor.')}
                    item['active_update'] = state_difference(active,{n:t for n,t in snap.items()
                        if actor_backward or not n.startswith('grad.actor.')})
                    for result in (item['all_fields'],item['active_update']):
                        for difference in result['differences']:
                            difference.update(evidence.get(difference['field'].removesuffix('.none'),{}))
                    if differences:
                        path = output/f'recovery_differences_{index}.pt'
                        torch.save(differences,path); item['evidence_path'] = str(path)
                    if not item['all_fields']['passed']:
                        deferred_failures.append(dict(phase='recovery',index=index,
                            first_difference=item['all_fields']['first_difference']))
                report.setdefault('recovery_updates',[]).append(item)
            snapshots.append(snap); contexts.append(context)
            report['checks'].append(dict(index=index,total_steps=total,failure=failure,
                passed=not recovery or reference is None or item['all_fields']['passed']))
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
            tracing = contract['trace_retained_gradient'] and failure == 'critic_loss'
            run_step(success,fixed[0],trace=tracing)
            protected,_ = snapshot(engine)
            error_batch = replace(fixed[0],reward=torch.full_like(fixed[0].reward,float('nan'))) if failure == 'critic_loss' else fixed[0]
            last = success+config.policy_delay
            actual = run_step(last,error_batch,failure=failure,trace=tracing)
            prefixes = ('actor.','actor_target.','optimizer.actor.','counts.actor_update_count') if failure.startswith('actor') else ('actor.','critic1.','critic2.','actor_target.','critic1_target.','critic2_target.','optimizer.','counts.')
            keep = {n:t for n,t in protected.items() if n.startswith(prefixes)}
            protected_report = state_difference(keep,{n:actual[n] for n in keep},exact=True)
            state_report = report['error_states'][str(len(snapshots)-1)]
            state_report['protected_state_exact'] = protected_report
            report['checks'][-1]['protected_state_exact'] = protected_report['passed']
            checks_path = output/f'error_state_{len(snapshots)-1}_checks.json'
            p._strict_json_write(checks_path,dict(location=report['location'],
                protected_state_exact=protected_report,cross_snapshot=state_report.get('cross_snapshot')))
            state_report['checks_path'] = str(checks_path)
            state_report['state_checks_completed'] = True
            failed_checks = [name for name in ('cross_before','cross_after','cross_snapshot','protected_state_exact')
                if name in state_report and not state_report[name]['passed']]
            report['checks'][-1]['passed'] = not failed_checks
            if failed_checks:
                report['first_error_state_failure'] = dict(report['location'],comparisons=failed_checks,
                    first_difference=state_report[failed_checks[0]]['first_difference'])
                if not contract['error_recovery']:
                    raise AssertionError('Error-state mismatch after complete state collection: '+str(report['first_error_state_failure']))
                deferred_failures.append(report['first_error_state_failure'])
            if contract['error_recovery']:
                # Same CPU batch and explicit target-noise inputs on both sides;
                # increasing stage steps exercise critic-only then actor update.
                for total in (last+1,last+config.policy_delay):
                    controlled = torch.randn(config.batch_size,engine.action_dim,generator=generator)
                    run_step(total,fixed[0],controlled,recovery=True)
                report['recovery_completed'] = True
            if contract['first_critic_loss_only']:
                break
        if deferred_failures:
            report['deferred_failures'] = deferred_failures
            report['completed'] = bool(report.get('recovery_completed'))
            raise AssertionError('Error/recovery comparisons completed but inconsistent; see diagnostic JSON.')
        # Supported load entry point must drop every old graph, then rebuild
        # from the same initial engine state; replay remains the CPU authority.
        if not contract['first_critic_loss_only']:
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
        else:
            report['load_rebuild_verified'] = None
            report['omitted_checks'] = ['remaining error injections','checkpoint load/rebuild','timing']
        if args.gpu_td3_compute and profiled_modes != {'critic_only','frozen_targets','actor'}:
            raise AssertionError('Three manual update modes were not verified.')
        report.update(passed=True,completed=True,compilation=configured,
            effective=engine.local_optimization_metadata(),snapshot_count=len(snapshots),
            graph_evidence_scope='one actual prewarmed update per mode profiled; not every update individually profiled')
        if reference is None: torch.save(dict(contract=contract,initial=initial,snapshots=snapshots,contexts=contexts,error_states=error_states),output/'reference.pt')
    except BaseException as exc:
        report['error'] = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),gpu_context=getattr(exc,'gpu_compute_context',None))
        report['failure'] = report['error']
        raise
    finally:
        if trace_events:
            torch.save(trace_events,output/'retained_gradient_trace.pt')
            report['retained_gradient_trace'] = retained_trace_summary(trace_events)
        if contract['error_recovery'] or contract['trace_retained_gradient']:
            compact = dict(passed=report['passed'],completed=report['completed'],parameter=RETAINED_PARAMETER,
                recovery_completed=report.get('recovery_completed',False),
                trace=report.get('retained_gradient_trace'),
                protected_state=[dict(index=index,passed=item.get('protected_state_exact',{}).get('passed'))
                    for index,item in report.get('error_states',{}).items()],
                recovery=[dict(index=item['index'],steps=item['stage_steps'],actor_backward=item['actor_backward'],
                    comparison=item['comparison'],
                    all_fields_passed=item.get('all_fields',{}).get('passed'),
                    active_update_passed=item.get('active_update',{}).get('passed'))
                    for item in report.get('recovery_updates',[])],
                first_failure=report.get('first_error_state_failure'),evidence='correctness.json')
            p._strict_json_write(output/'error_recovery_summary.json',compact)
        if engine is not None: engine.invalidate_gpu_update()
        p._strict_json_write(output/'correctness.json',report)
    return report


def run(args):
    from brain_uav.scripts import profile_v2_td3 as p
    recovery = getattr(args,'gpu_compute_error_recovery',False)
    tracing = getattr(args,'gpu_compute_trace_retained_gradient',False)
    if (recovery or tracing) and (not getattr(args,'gpu_compute_first_critic_loss_only',False)
            or args.gpu_compute_worker == 'timing'):
        raise ValueError('Error recovery/trace is correctness-only and requires first-critic-loss-only.')
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
        try:
            result = p.run_local_optimization_comparison(args)
        except Exception as exc:
            if not (recovery or tracing):
                raise
            print(json.dumps(dict(passed=False,error_type=type(exc).__name__,
                summary=str(Path(args.output_dir)/'error_recovery_summary.json'),
                detail=str(Path(args.output_dir)/'correctness.json'))))
            raise SystemExit(1) from None
        if recovery or tracing:
            print((Path(args.output_dir)/'error_recovery_summary.json').read_text(encoding='utf-8'))
        return result
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
            if getattr(args,'gpu_compute_first_critic_loss_only',False): command+=['--gpu-compute-first-critic-loss-only']
            if recovery: command+=['--gpu-compute-error-recovery']
            if tracing: command+=['--gpu-compute-trace-retained-gradient']
            if enabled: command+=['--gpu-td3-compute','--td3-block-reference',str(output/'baseline_correctness'/'reference.pt')]
            with (output/(group+'_correctness.log')).open('w',encoding='utf-8') as log:
                rc=subprocess.run(command+['--gpu-compute-worker','correctness','--output-dir',str(output/(group+'_correctness'))],
                    stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            item['correctness_exit_code']=rc; item['correctness_passed']=rc==0
            summary_path = output/(group+'_correctness')/'error_recovery_summary.json'
            if summary_path.exists():
                item['error_recovery'] = json.loads(summary_path.read_text(encoding='utf-8'))
            if not enabled: baseline_ok=rc==0
        if not all(v['correctness_passed'] for v in report['groups'].values()):
            if recovery or tracing:
                print(json.dumps(report,ensure_ascii=False))
                raise SystemExit(1)
            raise RuntimeError('CPU sampling/full compute correctness failed; this model timing skipped.')
        if getattr(args,'gpu_compute_first_critic_loss_only',False):
            report.update(passed=True,scope='first critic_loss only; no timing or later checks')
            return report
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
