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
import time
import faulthandler
import numpy as np
import torch

CONTRACT_VERSION = 8
RETAINED_PARAMETER = 'actor.zone_set_encoder.empty_scene_token'
# Original update_once order: loss barrier -> zero_grad(None) -> backward ->
# gradient barrier -> Adam -> targets. Both actor-loss and actor-gradient
# failures occur AFTER the critic/critic-target commit.
EXCEPTION_GRADIENTS = {
    'critic_loss': dict(current=[], old=['actor','critic1','critic2']),
    'critic_gradient': dict(current=['critic1','critic2'], old=['actor']),
    'actor_loss': dict(current=['critic1','critic2'], old=['actor']),
    'actor_gradient': dict(current=['actor','critic1','critic2'], old=[]),
}


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


def limited_retained_comparison(expected, actual, *, phase=None, actor_backward=False, gradients=None):
    """Classify unconsumed old grads, never current backward or storage structure.

    update_once checks critic loss before either backward; both optimizers use
    zero_grad(set_to_none=True) before their next backward/gradient check/step.
    The caller must pair exceptions retaining old gradients with two recovery
    updates. Old values (even NaN/Inf) are not a current backward's evidence.
    """
    modules = (() if actor_backward else ('actor',) if phase == 'critic_only_recovery'
        else EXCEPTION_GRADIENTS.get(phase,{}).get('old',()))
    raw = state_difference(expected,actual)
    waived = []
    for item in raw['differences']:
        key = item['field']
        parameter = None
        for module in modules:
            if key.startswith('gradient.'+module+'.') and not key.endswith('.none'):
                parameter = key.removeprefix('gradient.')
            elif key.startswith('grad.'+module+'.') and not key.endswith('.none'):
                name = (gradients or {}).get(key,{}).get('parameter_name')
                if name is not None and name.startswith(module+'.'):
                    parameter = name
        if parameter is None:
            continue
        a,b = expected.get(key),actual.get(key)
        if (a is None or b is None or a.shape != b.shape or a.dtype != b.dtype
                or not a.is_floating_point()):
            continue
        waived.append(dict(item,parameter_name=parameter,phase=phase,
            reason='Old gradient not consumed before zero_grad; values and nonfinite locations are evidence only; conditional on recovery.'))
    remaining = [item for item in raw['differences'] if item['field'] not in {v['field'] for v in waived}]
    return dict(raw=raw,waived_fields=waived,passed=not remaining,
        first_difference=next(iter(remaining),None),differences=remaining)


def exception_self_protection(before, after, failure):
    """Exact own-side protection, NEVER the cross-path residual exemption."""
    def protected(name):
        if failure == 'critic_loss':
            return True
        if failure == 'critic_gradient':
            return not name.startswith(('gradient.critic1.','gradient.critic2.'))
        # Critic updates are legitimate on actor errors. Its target counter
        # increments before the actor barrier; all other CPU counters stay put.
        return (name.startswith(('actor.','actor_target.','optimizer.actor.'))
            or (failure == 'actor_loss' and name.startswith('gradient.actor.'))
            or (name.startswith('counts.') and name != 'counts.critic_target_update_count'))
    return state_difference({n:t for n,t in before.items() if protected(n)},
        {n:t for n,t in after.items() if protected(n)},exact=True)


def current_gradient_comparison(expected, actual, modules):
    prefixes = tuple(prefix+module+'.' for module in modules for prefix in ('grad.','gradient.'))
    return state_difference({n:t for n,t in expected.items() if n.startswith(prefixes)},
        {n:t for n,t in actual.items() if n.startswith(prefixes)})


def recovery_acceptance(updates):
    return (len(updates)==2 and [item['actor_backward'] for item in updates]==[False,True]
        and all(item.get('active_update',{}).get('passed',item.get('comparison')=='baseline_reference_only') for item in updates)
        and updates[-1].get('all_fields',{}).get('passed',updates[-1].get('comparison')=='baseline_reference_only'))


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

def _load_reference_error_state(args, reference, index):
    return torch.load(Path(args.gpu_compute_reference).parent / reference['error_states'][index],
                      map_location='cpu', weights_only=False)


MPS_REFERENCE = dict(comparison='ordinary_vs_mps', source='ordinary', gpu_td3_compute=True,
                     non_blocking_batch_transfer=True, gpu_compute_non_blocking_coefficients=False)


def _mps_graph_replay_evidence(callback, device):
    """One real update; raw CPU runtime events only, with no aggregation/retry."""
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=(torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA),
        record_shapes=False,profile_memory=False,with_stack=False,
    ) as replay_profile:
        callback()
    torch.cuda.synchronize(device)
    results = replay_profile.profiler.kineto_results
    if results is None:
        raise RuntimeError('ordinary_vs_mps raw Kineto results unavailable.')
    names = []
    for event in results.events():
        if event.device_type() != torch.autograd.DeviceType.CPU:
            continue
        name = event.name()
        if 'cudagraphlaunch' in ''.join(character.lower() for character in name if character.isalnum()):
            names.append(name)
    if not names:
        raise AssertionError('ordinary_vs_mps has no raw CPU cudaGraphLaunch evidence.')
    return dict(cuda_graph_launch_count=len(names),event_names=sorted(set(names)))


class _MPSProgress:
    """Only ordinary_vs_mps: flushed stage evidence and bounded-time stack dump."""
    def __init__(self, args):
        self.root = Path(args.gpu_compute_correctness_control_dir) if getattr(args,'gpu_compute_mps_comparison',None) else None
        self.armed = False

    def stage(self, phase, index=None, **details):
        if self.root is None: return
        from brain_uav.scripts import v2_concurrent_diagnostic as c
        record = dict(phase=phase,index=index,unix_time=time.time(),monotonic=time.monotonic(),**details)
        print('MPS_STAGE '+json.dumps(record),file=sys.stderr,flush=True)
        # The parent reads this bounded single record on timeout, before cleanup.
        c.write_json(self.root/'last_stage.json',record)

    def arm(self):
        if self.root is not None:
            faulthandler.dump_traceback_later(60,repeat=True,file=sys.stderr)
            self.armed = True

    def finish(self):
        if self.armed:
            faulthandler.cancel_dump_traceback_later()
            self.armed = False


def _validate_mps_comparison(args):
    role = getattr(args, 'gpu_compute_mps_comparison', None)
    control = getattr(args, 'gpu_compute_correctness_control_dir', None)
    if not role:
        if control is not None: raise ValueError('Correctness control requires ordinary_vs_mps contract.')
        return None
    write = getattr(args, 'gpu_compute_write_reference', False)
    reference = args.gpu_compute_reference
    if (args.gpu_compute_worker != 'correctness' or not args.gpu_td3_compute
            or not args.non_blocking_batch_transfer or control is None
            or getattr(args, 'gpu_compute_coefficient_comparison', False)
            or getattr(args, 'gpu_compute_non_blocking_coefficients', False)
            or getattr(args, 'gpu_compute_lightweight_timing', False)
            or getattr(args, 'gpu_compute_first_critic_loss_only', False)
            or getattr(args, 'concurrent_window_dir', None) is not None
            or (role == 'ordinary' and (not write or reference is not None))
            or (role == 'mps' and (write or reference is None))):
        raise ValueError('ordinary_vs_mps requires full correctness, nonblocking GPUCompute, original coefficients, PID gate, ordinary write / MPS read.')
    return role


def _mps_correctness_ready(args, device):
    """After initialization/capture, before any controlled update; no timing window."""
    import os
    import time
    from brain_uav.scripts import v2_concurrent_diagnostic as c
    role = _validate_mps_comparison(args)
    root = Path(args.gpu_compute_correctness_control_dir)
    torch.cuda.synchronize(device)
    c.write_json(root/'ready.json', dict(pid=os.getpid(),role=role,synchronized=True,
                                       comparison='ordinary_vs_mps'))
    deadline = time.monotonic() + args.concurrent_timeout
    while True:
        if (root/'abort.json').exists(): raise RuntimeError('MPS correctness parent aborted.')
        if time.monotonic() >= deadline: raise TimeoutError('MPS correctness PID release timeout.')
        if (root/'release.json').exists():
            evidence = c.read_json(root/'release.json')
            if (evidence.get('worker_pid') != os.getpid() or evidence.get('effective') != role
                    or evidence.get('verified_before_update') is not True
                    or evidence.get('comparison') != 'ordinary_vs_mps'):
                raise ValueError('MPS correctness PID release evidence mismatch.')
            return evidence
        time.sleep(.01)


def _load_compute_reference(args, contract):
    reference = torch.load(args.gpu_compute_reference, map_location='cpu', weights_only=False)
    if reference.get('contract') != contract:
        raise ValueError('Compute reference contract mismatch; rebuild baseline.')
    if getattr(args, 'gpu_compute_mps_comparison', None):
        access = reference.get('cuda_access_evidence', {})
        if (reference.get('mps_reference') != MPS_REFERENCE or 'coefficient_reference' in reference
                or reference.get('transfer_reference') != dict(gpu_td3_compute=True,non_blocking_batch_transfer=True)
                or access.get('effective') != 'ordinary' or access.get('verified_before_update') is not True
                or access.get('comparison') != 'ordinary_vs_mps' or not isinstance(access.get('worker_pid'),int)
                or access.get('compute_pids') != [access.get('worker_pid')]):
            raise ValueError('ordinary_vs_mps requires PID-verified ordinary GPUCompute nonblocking reference.')
    elif 'mps_reference' in reference:
        raise ValueError('ordinary_vs_mps reference cannot be used by transfer/coefficient comparisons.')
    elif getattr(args, 'gpu_compute_coefficient_comparison', False):
        if reference.get('coefficient_reference') != {
                'gpu_td3_compute': True, 'non_blocking_batch_transfer': True,
                'gpu_compute_non_blocking_coefficients': False} or reference.get('transfer_reference') != {
                'gpu_td3_compute': True, 'non_blocking_batch_transfer': True}:
            raise ValueError('Coefficient correctness requires GPUCompute nonblocking batch + original coefficient reference.')
    elif args.non_blocking_batch_transfer and reference.get('transfer_reference') != {
            'gpu_td3_compute': True, 'non_blocking_batch_transfer': False}:
        raise ValueError('Nonblocking correctness requires an original GPUCompute blocking reference.')
    return reference


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
    contract['error_recovery'] = (not contract['first_critic_loss_only']
        or getattr(args,'gpu_compute_error_recovery',False))
    contract['trace_retained_gradient'] = getattr(args,'gpu_compute_trace_retained_gradient',False)
    contract['exception_gradient_lifecycle'] = EXCEPTION_GRADIENTS
    contract['checkpoint_sha256'] = sha256(Path(args.bc_checkpoint).read_bytes()).hexdigest()
    contract['full_pool_digest'] = full_pool_digest
    contract['pool_file_sha256'] = sha256(Path(args.validation_pool).read_bytes()).hexdigest()
    mps_role = _validate_mps_comparison(args)
    if mps_role:
        contract['comparison'] = 'ordinary_vs_mps'
        report.update(comparison='ordinary_vs_mps',role=mps_role)
    report['inputs'] = contract
    report['code'] = {name:subprocess.run(command,capture_output=True,text=True,check=False).stdout.strip()
        for name,command in (('head',['git','rev-parse','HEAD']),('dirty',['git','status','--short']))}
    reference = None
    # Optional, isolated ANN investigation; never selected by formal training.
    focus = getattr(args, '_ann_common_state', None)
    if args.gpu_td3_compute and focus is None and not getattr(args, 'gpu_compute_write_reference', False):
        reference = _load_compute_reference(args, contract)
    engine = None
    trace_events = []
    progress = _MPSProgress(args)
    try:
        progress.stage('initialization')
        report['location']['phase'] = 'initialization'
        engine = p.build_v2_stage_engine(None,config,init_checkpoint=args.bc_checkpoint,
            rewards=prepared.reward_config,prepared_initialization=prepared,device=device,
            model_type=args.model,snn_time_window=args.snn_time_window,fused_adam=fused,
            reduce_update_stat_syncs=reduced,gpu_td3_compute=args.gpu_td3_compute,
            non_blocking_batch_transfer=args.non_blocking_batch_transfer,
            gpu_compute_non_blocking_coefficients=getattr(args, 'gpu_compute_non_blocking_coefficients', False)).engine
        initial = reference['initial'] if reference else deepcopy(engine.checkpoint_state_dict())
        if focus is not None:
            initial = focus.initial(engine, initial, contract)
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
            progress.stage('exception_recovery' if recovery else 'update_begin',index,steps=total,failure=failure)
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
            if focus is not None:
                focus.before_step(engine, index, total, batch, noise, failure)
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
                with noise_context,injection,(focus.observe_noise(index) if focus is not None else nullcontext()),trace_retained_gradient(engine,trace_events,
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
                progress.stage('exception_snapshot',index)
                after_error = error_state(engine)
                expected_error = None
                if reference is not None:
                    expected_error = _load_reference_error_state(args, reference, index)
                state_report = record_error_state(output,index,before_error,after_error,expected_error)
                state_report['own_protected_state'] = exception_self_protection(
                    before_error['values'],after_error['values'],failure)
                state_report['backward_lifecycle'] = EXCEPTION_GRADIENTS[failure]
                error_states[index] = Path(state_report['snapshot_path']).name
                report.setdefault('error_states',{})[str(index)] = state_report
            actor_backward = (metric.actor_updated if metric is not None else failure == 'actor_gradient')
            progress.stage('snapshot_begin',index)
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
            progress.stage('numerical_comparison',index,reference=reference is not None)
            if reference is not None:
                prior = reference['contexts'][index]
                if any(context[name] != prior[name] for name in
                        ('snapshot_index','stage_steps','actor_backward','failure')):
                    raise AssertionError('Reference step/context mismatch at index '+str(index))
            if reference is not None and not failure and not recovery:
                expected = reference['snapshots'][index]
                if not error_states:
                    for name,t in snap.items():
                        if name.startswith(('rng.','counts.','tensor.batch.')) or name.endswith('.step'):
                            torch.testing.assert_close(t,expected[name],rtol=0,atol=0)
                    p._compare_td3_block_snapshot(expected,snap,context=context,report=report,output=output)
                else:
                    # No residual exemption on these actual updates. Collect
                    # every field and keep going to the checkpoint check.
                    differences = {}
                    result = state_difference(expected,snap,evidence=differences,label='post_error_update')
                    report.setdefault('post_error_updates',[]).append(dict(context,result=result))
                    if not result['passed']:
                        torch.save(differences,output/f'post_error_update_{index}_differences.pt')
                        failure_info = dict(phase='post_error_update',index=index,
                            first_difference=result['first_difference'])
                        report.setdefault('first_error_state_failure',failure_info)
                        deferred_failures.append(failure_info)
            if reference is not None and failure:
                # Preserve the original snapshot/RNG/counter contract too, but
                # never stop before every error-state field is checked/saved.
                report['error_states'][str(index)]['cross_snapshot'] = state_difference(reference['snapshots'][index],snap)
                state = report['error_states'][str(index)]
                state['current_gradients'] = current_gradient_comparison(
                    expected_error['after']['values'],after_error['values'],EXCEPTION_GRADIENTS[failure]['current'])
                if contract['error_recovery']:
                    state['effective_cross_after'] = limited_retained_comparison(
                        expected_error['after']['values'],after_error['values'],
                        phase=failure,actor_backward=actor_backward)
                    state['effective_cross_snapshot'] = limited_retained_comparison(
                        reference['snapshots'][index],snap,phase=failure,
                        actor_backward=actor_backward,gradients=evidence)
            if recovery:
                item = dict(index=index,stage_steps=total,actor_backward=actor_backward,
                    after_exception=recovery,
                    diagnostic_only=True, production_resumption_supported=False,
                    comparison='baseline_reference_only' if reference is None else 'baseline_vs_compute',
                    scope='Continuation of interrupted diagnostic engine only; not a production recovery contract.')
                if reference is not None:
                    expected = reference['snapshots'][index]
                    differences = {}
                    item['all_fields'] = state_difference(expected,snap,evidence=differences,label='recovery')
                    # Keep residuals in all_fields; separately show only consumed
                    # backward gradients plus all persistent state/loss evidence.
                    item['active_update'] = limited_retained_comparison(expected,snap,
                        phase='critic_only_recovery',actor_backward=actor_backward,gradients=evidence)
                    for result in (item['all_fields'],item['active_update']):
                        for difference in result['differences']:
                            difference.update(evidence.get(difference['field'].removesuffix('.none'),{}))
                    if differences:
                        path = output/f'recovery_differences_{index}.pt'
                        torch.save(differences,path); item['evidence_path'] = str(path)
                    if not item['active_update']['passed']:
                        failure_info = dict(phase='recovery',index=index,after_exception=recovery,
                            first_difference=item['active_update']['first_difference'])
                        report.setdefault('first_error_state_failure',failure_info)
                        deferred_failures.append(failure_info)
                report.setdefault('recovery_updates',[]).append(item)
            snapshots.append(snap); contexts.append(context)
            report['checks'].append(dict(index=index,total_steps=total,failure=failure,
                passed=not recovery or reference is None or item['active_update']['passed']))
            if focus is not None:
                focus.after_step(engine, index, snap)
            progress.stage('update_complete',index,steps=total)
            return snap

        if focus is not None and focus.replaying:
            focus.replay(engine, run_step)
            report.update(passed=True, completed=True, scope='ANN common-state single update only')
            return report

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
                progress.stage('graph_capture_begin',len(snapshots),mode=mode)
                if key not in executor.graphs: executor.capture_shape(key,upload)
                progress.stage('graph_capture_complete',len(snapshots),mode=mode)
                if mps_role and i == 0:
                    progress.stage('synchronize_and_pid_gate',len(snapshots))
                    report['cuda_access_evidence'] = _mps_correctness_ready(args,device)
                    progress.arm()
                    progress.stage('pid_released',len(snapshots))
                progress.stage('profiler_enter',len(snapshots),mode=mode)
                callback = lambda: run_step(total,fixed[i%len(fixed)],controlled)
                evidence = (_mps_graph_replay_evidence(callback,device) if mps_role
                    else engine._profile_cuda_graph_replay(callback))
                progress.stage('profiler_exit',len(snapshots)-1,mode=mode)
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
            progress.stage('exception_recovery_sequence',len(snapshots),failure=failure)
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
            own_required = args.gpu_td3_compute
            state_report['compute_self_protection_required'] = own_required
            cross_names = ('cross_before','effective_cross_after','effective_cross_snapshot') if (
                reference is not None and contract['error_recovery']) else (
                'cross_before','cross_after','cross_snapshot')
            required_checks = ['protected_state_exact']
            if reference is not None:
                required_checks.extend((*cross_names,'current_gradients'))
            if own_required:
                required_checks.append('own_protected_state')
            failed_checks = [name for name in required_checks
                if name not in state_report or not state_report[name]['passed']]
            report['checks'][-1]['passed'] = not failed_checks
            if failed_checks:
                failure_info = dict(report['location'],comparisons=failed_checks,
                    first_difference=state_report.get(failed_checks[0],dict(first_difference=
                        dict(field=failed_checks[0],reason='missing required check')))['first_difference'])
                report.setdefault('first_error_state_failure',failure_info)
                if not contract['error_recovery']:
                    raise AssertionError('Error-state mismatch after complete state collection: '+str(report['first_error_state_failure']))
                deferred_failures.append(failure_info)
            if contract['error_recovery'] and EXCEPTION_GRADIENTS[failure]['old']:
                # Same CPU batch and explicit target-noise inputs on both sides;
                # increasing stage steps exercise critic-only then actor update.
                for total in (last+1,last+config.policy_delay):
                    controlled = torch.randn(config.batch_size,engine.action_dim,generator=generator)
                    run_step(total,fixed[0],controlled,recovery=failure)
                recovered = report['recovery_updates'][-2:]
                if [item['actor_backward'] for item in recovered] != [False,True]:
                    raise AssertionError('Recovery must exercise critic-only then actor backward.')
                state_report['recovery_acceptance'] = dict(required=True,
                    passed=recovery_acceptance(recovered),
                    snapshot_indices=[item['index'] for item in recovered])
                if not state_report['recovery_acceptance']['passed']:
                    failure_info = dict(phase='recovery_gate',snapshot_index=len(snapshots)-1,after_exception=failure)
                    report.setdefault('first_error_state_failure',failure_info)
                    deferred_failures.append(failure_info)
                report['recovery_groups_completed'] = report.get('recovery_groups_completed',0)+1
                report['recovery_completed'] = report['recovery_groups_completed'] == (
                    1 if contract['first_critic_loss_only'] else sum(bool(v['old']) for v in EXCEPTION_GRADIENTS.values()))
                last += config.policy_delay
            if contract['first_critic_loss_only']:
                break
        # Supported load entry point must drop every old graph, then rebuild
        # from the same initial engine state; replay remains the CPU authority.
        if not contract['first_critic_loss_only']:
            progress.stage('checkpoint_rebuild_begin',len(snapshots))
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
            progress.stage('checkpoint_rebuild_complete',len(snapshots)-1)
        else:
            report['load_rebuild_verified'] = None
            report['omitted_checks'] = ['remaining error injections','checkpoint load/rebuild','timing']
        if args.gpu_td3_compute and profiled_modes != {'critic_only','frozen_targets','actor'}:
            raise AssertionError('Three manual update modes were not verified.')
        report['completed'] = True
        if deferred_failures:
            report['deferred_failures'] = deferred_failures
            raise AssertionError('Error/recovery comparisons completed but inconsistent; see diagnostic JSON.')
        report.update(passed=True,completed=True,compilation=configured,
            effective=engine.local_optimization_metadata(),snapshot_count=len(snapshots),
            graph_evidence_scope='one actual prewarmed update per mode profiled; not every update individually profiled')
        progress.stage('reference_save',len(snapshots),required=reference is None)
        if reference is None: torch.save(dict(contract=contract,initial=initial,snapshots=snapshots,contexts=contexts,error_states=error_states,
            **({'mps_reference': dict(MPS_REFERENCE), 'cuda_access_evidence': report['cuda_access_evidence']} if mps_role else {}),
            **({'coefficient_reference': dict(gpu_td3_compute=True, non_blocking_batch_transfer=True,
                gpu_compute_non_blocking_coefficients=False)} if getattr(args, 'gpu_compute_coefficient_comparison', False) else {}),
            transfer_reference=dict(gpu_td3_compute=args.gpu_td3_compute,
                non_blocking_batch_transfer=args.non_blocking_batch_transfer)),output/'reference.pt')
    except BaseException as exc:
        if focus is not None and focus.history_complete(exc):
            report.update(passed=True, completed=True, scope='ANN history through index 26 only; no acceptance or timing')
            return report
        report['error'] = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),gpu_context=getattr(exc,'gpu_compute_context',None))
        report['failure'] = report['error']
        raise
    finally:
        try:
            report['acceptance'] = dict(semantic_passed=report['passed'],completed=report['completed'],
                comparison='ordinary_vs_mps' if mps_role else 'baseline_reference_only' if reference is None else 'baseline_vs_compute',
                raw_all_fields_equal=None if reference is None or not report['completed'] else
                    all(state.get(name,{'passed':True})['passed'] for state in report.get('error_states',{}).values()
                        for name in ('cross_before','cross_after','cross_snapshot'))
                    and all(item.get('all_fields',{'passed':True})['passed'] for item in report.get('recovery_updates',[]))
                    and all(item['result']['passed'] for item in report.get('post_error_updates',[])),
                scope='Exception lifecycle table; old gradients require presence/shape/dtype across paths; all numerical/nonfinite differences recorded; current gradients unchanged.',
                recovery_required=contract['error_recovery'],recovery_completed=report.get('recovery_completed',False),
                checkpoint_reload=report.get('load_rebuild_verified'))
            if trace_events:
                torch.save(trace_events,output/'retained_gradient_trace.pt')
                report['retained_gradient_trace'] = retained_trace_summary(trace_events)
            if contract['error_recovery'] or contract['trace_retained_gradient']:
                compact = dict(passed=report['passed'],completed=report['completed'],trace_parameter=RETAINED_PARAMETER,
                    recovery_completed=report.get('recovery_completed',False),
                    trace=report.get('retained_gradient_trace'),
                    protected_state=[dict(index=index,passed=item.get('protected_state_exact',{}).get('passed'))
                        for index,item in report.get('error_states',{}).items()],
                    recovery=[dict(index=item['index'],steps=item['stage_steps'],actor_backward=item['actor_backward'],
                        after_exception=item['after_exception'],
                        comparison=item['comparison'],
                        all_fields_passed=item.get('all_fields',{}).get('passed'),
                        active_update_passed=item.get('active_update',{}).get('passed'))
                        for item in report.get('recovery_updates',[])],
                    first_failure=report.get('first_error_state_failure'),evidence='correctness.json')
                compact['acceptance'] = report.get('acceptance')
                compact['exception_checks'] = [dict(index=index,lifecycle=state['backward_lifecycle'],
                    current_gradients_passed=state.get('current_gradients',{}).get('passed'),
                    old_gradient_differences=len(state.get('effective_cross_after',{}).get('waived_fields',[])),
                    own_protection=state['own_protected_state']['passed'],
                    persistent_protection=state.get('protected_state_exact',{}).get('passed'),
                    recovery=state.get('recovery_acceptance')) for index,state in report.get('error_states',{}).items()]
                compact['retained_residuals'] = [dict(index=index,
                    waived_field_count=len(state.get('effective_cross_after',{}).get('waived_fields',[])),
                    waived_fields_evidence='correctness.json:error_states.'+index+'.effective_cross_after.waived_fields',
                    compute_self_protection=state.get('own_protected_state',{}).get('passed') if args.gpu_td3_compute else None,
                    baseline_self_unchanged=state.get('own_before_after',{}).get('passed') if not args.gpu_td3_compute else None)
                    for index,state in report.get('error_states',{}).items()]
                progress.stage('recovery_report_write_begin',len(locals().get('snapshots',[])))
                p._strict_json_write(output/'error_recovery_summary.json',compact)
                progress.stage('recovery_report_write_complete',len(locals().get('snapshots',[])))
            if engine is not None:
                progress.stage('executor_invalidate_begin',len(locals().get('snapshots',[])))
                engine.invalidate_gpu_update()
                progress.stage('executor_invalidate_complete',len(locals().get('snapshots',[])))
            progress.stage('final_report_write_begin',len(locals().get('snapshots',[])))
            if focus is None:
                p._strict_json_write(output/'correctness.json',report)
            else:
                # Do not label a truncated investigation as full v8 acceptance.
                p._strict_json_write(focus.root/'review'/(focus.worker+'_preparation.json'),
                    dict(completed=report['completed'], execution_passed=report['passed'],
                        full_acceptance=False, scope=report.get('scope'), code=report.get('code'),
                        compilation=locals().get('configured'), error=report.get('error'),
                        selected_location=report['location']))
            progress.stage('final_report_write_complete',len(locals().get('snapshots',[])))
        finally:
            progress.finish()
    return report


def run(args):
    from brain_uav.scripts import profile_v2_td3 as p
    mps_role = _validate_mps_comparison(args)
    coefficient_comparison = getattr(args, 'gpu_compute_coefficient_comparison', False)
    direct_coefficients = getattr(args, 'gpu_compute_non_blocking_coefficients', False)
    light_timing = getattr(args, 'gpu_compute_lightweight_timing', False)
    write_reference = getattr(args, 'gpu_compute_write_reference', False)
    if (direct_coefficients or light_timing or coefficient_comparison) and not (
            args.gpu_td3_compute and args.non_blocking_batch_transfer and args.gpu_compute_worker):
        raise ValueError('Coefficient experiment requires explicit nonblocking GPUCompute worker.')
    if coefficient_comparison and not (args.gpu_compute_worker == 'correctness'
            and (write_reference or args.gpu_compute_reference is not None)):
        raise ValueError('Coefficient comparison requires correctness reference write/read.')
    if direct_coefficients and args.gpu_compute_worker == 'correctness' and not coefficient_comparison:
        raise ValueError('Direct coefficient correctness requires its explicit coefficient comparison contract.')
    if light_timing and args.gpu_compute_worker != 'timing':
        raise ValueError('Lightweight timing requires timing worker.')
    if write_reference and coefficient_comparison and (direct_coefficients or args.gpu_compute_reference is not None):
        raise ValueError('Coefficient reference must use original coefficients and no input reference.')
    if getattr(args, 'gpu_compute_write_reference', False) and not (
            args.gpu_compute_worker == 'correctness' and args.gpu_td3_compute
            and (not args.non_blocking_batch_transfer or coefficient_comparison or mps_role == 'ordinary')
            and args.gpu_compute_reference is None):
        raise ValueError('Writing transfer reference requires GPUCompute blocking correctness only.')
    if args.non_blocking_batch_transfer and not (args.gpu_compute_worker and args.gpu_td3_compute):
        raise ValueError('Nonblocking GPUCompute requires explicit compute worker; no serial baseline substitution.')
    if args.non_blocking_batch_transfer and args.gpu_compute_worker == 'correctness' and args.gpu_compute_reference is None and not ((coefficient_comparison or mps_role == 'ordinary') and write_reference):
        raise ValueError('Nonblocking correctness requires a GPUCompute blocking reference file.')
    if (getattr(args,'cpu_deduplication',False) or getattr(args,'cpu_deduplication_evidence',False)) and not (args.gpu_compute_worker == 'timing'):
        raise ValueError('CPU deduplication requires --gpu-compute-worker timing --gpu-td3-compute.')
    if getattr(args,'gpu_compute_environment_diagnostic',False) and not (args.gpu_compute_worker == 'timing'):
        raise ValueError('Environment diagnosis requires --gpu-compute-worker timing and --gpu-td3-compute.')
    recovery = getattr(args,'gpu_compute_error_recovery',False)
    tracing = getattr(args,'gpu_compute_trace_retained_gradient',False)
    if (recovery or tracing) and args.gpu_compute_worker == 'timing':
        raise ValueError('Error recovery/trace is correctness-only.')
    conflicts=('gpu_td3_update','gpu_td3_suite','gpu_td3_worker','gpu_td3_sampling_cost',

        'window_update_statistics','reuse_update_buffers',
        'batch_transfer_comparison','batch_transfer_worker',
        'local_optimization_correctness','local_optimization_comparison',
        'snn_path_consistency','detailed_profiler_updates','compiled_performance_diagnostic_updates')
    if any(getattr(args,n) for n in conflicts) or args.device!='cuda' or args.comparison_eager_cpu:
        raise ValueError('GPU compute comparison requires CUDA and original formal path, without other experiments/profiler.')
    if args.bc_checkpoint is None or args.validation_pool is None:
        raise ValueError('Matching BC and easy fixed pool are required.')
    if args.gpu_compute_worker:
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
    forwarded += ('comparison_critic_lr', 'comparison_alpha')
    if getattr(args, 'comparison_actor_lr', None) is not None:
        forwarded += ('comparison_actor_lr',)
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
            if enabled: command+=['--gpu-td3-compute','--gpu-compute-reference',str(output/'baseline_correctness'/'reference.pt')]
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
