"""Bounded MC-return CUDA correctness check; no training run or performance claim."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import traceback
import time
from unittest.mock import patch

import numpy as np
import torch


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bc-checkpoint', type=Path, required=True)
    parser.add_argument('--validation-pool', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-bc-sha256', required=True)
    parser.add_argument('--expected-pool-digest', required=True)
    parser.add_argument('--model', choices=('snn', 'ann'), default='snn')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--launch', action='store_true')
    return parser


def build_plan(args):
    if args.seed < 0 or args.batch_size < 2:
        raise ValueError('seed must be nonnegative and batch-size must be at least two.')
    for name in ('expected_bc_sha256', 'expected_pool_digest'):
        value = getattr(args, name)
        if len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value):
            raise ValueError(f'{name} must be a SHA256 hex digest.')
    return dict(diagnostic='v2_mc_return_cuda_v1', launch=args.launch,
        bc_checkpoint=str(args.bc_checkpoint), validation_pool=str(args.validation_pool),
        output_dir=str(args.output_dir), expected_bc_sha256=args.expected_bc_sha256.lower(),
        expected_pool_digest=args.expected_pool_digest.lower(), model=args.model,
        seed=args.seed, batch_size=args.batch_size, device='cuda',
        source_manifest=None if args.source_manifest is None else str(args.source_manifest),
        target_q_aggregation='mean', mc_return_weight=.1, mc_return_max_age_steps=100000,
        critic_lr=1e-3,
        snn_time_window=4, rtol=1e-4, atol=1e-5,
        scope='controlled fixed diagnostic batches; actual CUDA eager versus manual GPUCompute; no training or throughput evidence')


def source_identity():
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / 'src').rglob('*.py'))
    paths += [root / 'scripts' / 'validate_v2_mc_return_cuda.sh']
    hashes = {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    code = {name: subprocess.run(command, cwd=root, capture_output=True, text=True, check=True).stdout.strip()
            for name, command in (('head', ['git', 'rev-parse', 'HEAD']),
                                  ('status', ['git', 'status', '--short']))}
    return dict(sha256=hashes, **code)


def compare_snapshots(expected, actual, *, rtol=1e-4, atol=1e-5):
    """Owned flat CPU snapshots; nonfinite values always fail, even if matching."""
    differences = []
    for name in sorted(expected.keys() | actual.keys()):
        a, b = expected.get(name), actual.get(name)
        reason = None
        if a is None or b is None:
            reason = 'missing field'
        elif a.shape != b.shape or a.dtype != b.dtype:
            reason = 'shape or dtype mismatch'
        elif not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            reason = 'nonfinite value'
        else:
            exact = (not a.is_floating_point() or name.endswith('.step')
                     or name.startswith(('counts.', 'rng.')))
            if not (torch.equal(a, b) if exact else torch.allclose(a, b, rtol=rtol, atol=atol)):
                reason = 'value mismatch'
        if reason:
            item = dict(field=name, reason=reason)
            if a is not None and b is not None and a.shape == b.shape and a.numel():
                finite = torch.isfinite(a) & torch.isfinite(b)
                if bool(finite.any()):
                    item['max_finite_absolute_error'] = float((a[finite].double() - b[finite].double()).abs().max())
            differences.append(item)
    return dict(passed=not differences, differences=differences)


def mc_batch_cases(replay, batch_size, collected_step=1):
    """Age masks come from the real CPU sampler, using identical sampled rows."""
    state = deepcopy(replay.rng.bit_generator.state)
    cases = {}
    try:
        for name, step in (('active', collected_step), ('age_100000', collected_step + 100000),
                           ('age_100001', collected_step + 100001)):
            replay.rng.bit_generator.state = deepcopy(state)
            cases[name] = replay.sample(batch_size, current_stage_step=step, mc_return_max_age_steps=100000)
    finally:
        replay.rng.bit_generator.state = state
    active = cases['active']
    mask = active.mc_return_mask.clone()
    mask[0] = 0.
    cases['changed'] = replace(active, mc_return=-active.mc_return - 1., mc_return_mask=mask)
    cases['empty'] = replace(active, mc_return_mask=torch.zeros_like(mask))
    zero = active.mc_return.clone()
    zero[0] = 0.
    cases['zero'] = replace(active, mc_return=zero)
    return cases


@contextmanager
def controlled_sample(engine, batch, step):
    """Keep update_once -> real MC sampling API, replacing only its fixed rows."""
    calls = []
    def sample(size, **kwargs):
        if size != batch.batch_size or kwargs != dict(current_stage_step=step,
                mc_return_max_age_steps=engine.mc_return_max_age_steps):
            raise AssertionError('Controlled MC sample contract mismatch.')
        calls.append(kwargs)
        return batch
    with patch.object(engine.replay, 'sample', new=sample):
        yield
    if len(calls) != 1:
        raise AssertionError('Each update must sample its TD batch exactly once.')


@contextmanager
def eager_evidence(engine, record):
    """Observe the same real tensors as GPUCompute, without another forward."""
    original_clip = engine._validate_and_clip_gradients
    names = {id(p): module + '.' + name for module in ('actor', 'critic1', 'critic2')
             for name, p in getattr(engine, module).named_parameters()}
    def clip(parameters, **kwargs):
        parameters = tuple(parameters)
        for p in parameters:
            if p.grad is not None:
                record('gradient.pre_clip.' + names[id(p)], p.grad)
        return original_clip(parameters, **kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(engine, '_validate_and_clip_gradients', clip))
        if engine.model_type == 'snn':
            original_activity = engine.actor.forward_with_activity
            original_head = engine.actor.snn_head._forward_impl
            def activity(*args, **kwargs):
                actions, rate = original_activity(*args, **kwargs)
                record('actor.action', actions)
                return actions, rate
            def head(*args, **kwargs):
                raw, first, second = original_head(*args, **kwargs)
                record('activity.l1', first.detach().bool())
                record('activity.l2', second.detach().bool())
                return raw, first, second
            stack.enter_context(patch.object(engine.actor, 'forward_with_activity', activity))
            stack.enter_context(patch.object(engine.actor.snn_head, '_forward_impl', head))
        else:
            handle = engine.actor.register_forward_hook(lambda module, args, output: record('actor.action', output))
            stack.callback(handle.remove)
        yield


def _snapshot(engine, metric=None, tensors=None):
    from brain_uav.scripts.v2_gpu_td3_diagnostic import snapshot, flatten, complete_actor_evidence
    values, _ = snapshot(engine)
    if engine.bc_reference_actor is not None:
        flatten(engine.bc_reference_actor.state_dict(), 'bc_reference', values)
    for module in ('actor', 'critic1', 'critic2', 'actor_target', 'critic1_target',
                   'critic2_target', 'bc_reference_actor'):
        model = getattr(engine, module)
        if model is not None:
            for name, parameter in model.named_parameters():
                flatten(parameter.grad, 'gradient.' + module + '.' + name, values)
    values['counts.mc_return_updates'] = torch.tensor(engine._mc_return_update_count)
    if metric is not None:
        names = ('critic_loss', 'critic_td_loss', 'critic_mc_loss', 'critic_mc_weighted_loss',
                 'mc_valid_count', 'mc_valid_fraction', 'critic_q1_mean', 'critic_q1_std',
                 'critic_q1_max_abs', 'critic_target_q_mean', 'critic_td_error_mean',
                 'critic_failure_td_error_mean', 'sample_success_fraction', 'sample_failure_fraction',
                 'sample_near_goal_fraction', 'critic_updated', 'critic_targets_updated', 'actor_updated')
        selected = {name: getattr(metric, name) for name in names}
        # Frozen actor losses use None/zero conventions in the existing two paths.
        # Compare all actor quantities only when that backward actually occurred.
        if metric.actor_updated:
            selected.update({name: value for name, value in asdict(metric).items()
                             if name not in selected})
        flatten(selected, 'metrics', values)
    if tensors is not None:
        complete_actor_evidence(tensors, engine.model_type)
        for name, value in tensors.items():
            values['tensor.' + name] = value
    values['rng.cpu'] = torch.get_rng_state().clone()
    values['rng.cuda'] = torch.cuda.get_rng_state(engine.device).cpu().clone()
    return values


def run_update(engine, batch, step, normal, config, *, eager, evidence=None):
    """One actual update, preserving the public sampler's keyword contract."""
    from brain_uav.trainers.v2_formal_training import v2_bc_lambda
    collected = {'replay.mask': batch.obs.presence_mask.clone(),
                 'replay.next_mask': batch.next_obs.presence_mask.clone()}
    def record(name, value): collected[name] = value.detach().cpu().clone()
    _, policy_noise, noise_clip = config.noise_schedule.values(step - 1, max_steps=config.max_steps)
    engine.set_target_noise(policy_noise=policy_noise, noise_clip=noise_clip)
    def update():
        return engine.update_once(total_steps=step,
            bc_lambda=v2_bc_lambda(step - 1, schedule='smooth-mid'), diagnostic_tensor_recorder=record)
    with controlled_sample(engine, batch, step), \
            patch.object(torch, 'randn_like', new=lambda value: normal.to(value.device)), \
            (eager_evidence(engine, record) if eager else ExitStack()):
        if evidence is None:
            metric = update()
        else:
            result = []
            proof = evidence(lambda: result.append(update()), engine.device)
            if len(result) != 1: raise AssertionError('Graph evidence must execute exactly one update.')
            metric = result[0]
    return metric, collected, None if evidence is None else proof


def _jobs():
    # Consecutive odd/even steps warm both critic banks. A critic-only step
    # between actor pairs also warms all four actor/critic bank combinations.
    return [
        (1, 'active', False), (3, 'active', False),
        (4, 'active', False), (6, 'active', False),
        (8, 'changed', True), (10, 'empty', True), (12, 'active', True),
        (13, 'changed', True), (15, 'empty', True), (17, 'active', True),
        (25002, 'active', False), (25004, 'active', False), (25005, 'active', True),
        (25006, 'active', False), (25008, 'active', False),
        (25010, 'changed', True), (25012, 'empty', True), (25014, 'active', True),
        (100001, 'age_100000', True), (100002, 'age_100001', True),
        (100004, 'zero', True),
    ]


def run_diagnostic(args, plan):
    from brain_uav.trainers.v2_gpu_update import require_cuda_abi
    require_cuda_abi()  # Fail before loading any models if CUDA/ABI is unavailable.
    from brain_uav.scripts import profile_v2_td3 as profile
    from brain_uav.scripts.train_v2_td3 import _configure_stage_compilation, default_v2_cuda_graph_compilation
    from brain_uav.scripts.v2_gpu_compute_diagnostic import _mps_graph_replay_evidence
    from brain_uav.trainers.v2_formal_training import (
        V2FormalTrainingConfig, build_v2_stage_engine, prepare_v2_stage_initialization, v2_bc_lambda,
    )
    from brain_uav.trainers.v2_validation import load_v2_validation_pool
    from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer
    from brain_uav.trainers.v2_mc_return import discounted_episode_returns
    from brain_uav.trainers.v2_td3 import V2SpikeRegularizationConfig
    from brain_uav.observations import V2Observation

    report = dict(plan=plan, passed=False, completed=False, execution_completed=False,
                  phase='provenance', checks=[], graph_launch_evidence={})
    try:
        if args.source_manifest is None:
            raise ValueError('--source-manifest is required for CUDA execution; HEAD alone is insufficient.')
        expected_source = json.loads(args.source_manifest.read_text(encoding='utf-8'))
        actual_source = source_identity()
        if expected_source['sha256'] != actual_source['sha256']:
            raise ValueError('Actual source SHA256 differs from reviewed source manifest.')
        report['source'] = actual_source
        report['source_manifest_sha256'] = sha256(args.source_manifest.read_bytes()).hexdigest()
        bc_hash = sha256(args.bc_checkpoint.read_bytes()).hexdigest()
        pool_hash = sha256(args.validation_pool.read_bytes()).hexdigest()
        if bc_hash != plan['expected_bc_sha256']:
            raise ValueError('BC checkpoint SHA256 does not match the requested input.')
        config = V2FormalTrainingConfig(stage='easy', max_steps=500000, seed=args.seed, batch_size=args.batch_size,
            replay_capacity=max(64, args.batch_size * 2), target_q_aggregation='mean',
            critic_lr=plan['critic_lr'],
            mc_return_weight=.1, mc_return_max_age_steps=100000,
            spike_regularization=V2SpikeRegularizationConfig(beta_max=1. if args.model == 'snn' else 0.))
        device = torch.device('cuda')
        prepared = prepare_v2_stage_initialization(config, init_checkpoint=args.bc_checkpoint,
            device=device, model_type=args.model, snn_time_window=4)
        pool = load_v2_validation_pool(args.validation_pool, expected_level='easy',
            expected_scenario=prepared.scenario_config,
            expected_uav_collision_radius=prepared.uav_collision_radius, expected_master_seed=20260904)
        if pool.content_digest != plan['expected_pool_digest']:
            raise ValueError('Validation pool content digest does not match the requested input.')
        report['inputs'] = dict(bc_sha256=bc_hash, pool_file_sha256=pool_hash,
            pool_digest=pool.content_digest, pool_count=pool.scenario_count,
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            device=torch.cuda.get_device_name(device), formal_config=config.to_dict())
        report['phase'] = 'initialization'
        engines = [build_v2_stage_engine(None, config, init_checkpoint=args.bc_checkpoint,
            prepared_initialization=prepared, device=device, model_type=args.model,
            snn_time_window=4, fused_adam=True, reduce_update_stat_syncs=True,
            gpu_td3_compute=enabled, non_blocking_batch_transfer=False,
            gpu_compute_non_blocking_coefficients=False).engine for enabled in (False, True)]
        eager, compute = engines
        compute.load_checkpoint_state_dict(deepcopy(eager.checkpoint_state_dict()))
        for engine in engines:
            profile._set_numeric_engine_modes(engine)
        compilation = default_v2_cuda_graph_compilation(model=args.model, resolved_device='cuda')
        compilation.pop('fused_adam'); compilation.pop('reduce_update_stat_syncs')
        report['compilation'] = _configure_stage_compilation(compute, pool, prepared,
            compile_critic_encoder=False, compile_target_encoders=False, **compilation)
        initial = compare_snapshots(_snapshot(eager), _snapshot(compute))
        report['checks'].append(dict(name='same_initial_state', **initial))
        if not initial['passed']:
            raise AssertionError('Eager and GPUCompute initialization differ.')

        report['phase'] = 'fixed_cpu_batches'
        warmup = profile._compile_warmup_batches(pool=pool, prepared=prepared,
            batch_size=args.batch_size, device=torch.device('cpu'), cache_ellipsoid_segment_clearance=True)
        mixed, _ = profile._numeric_diagnostic_observation_batches(warmup, device=torch.device('cpu'))
        replay = V2ReplayBuffer(args.batch_size, eager.action_dim, mixed.max_zone_count,
            seed=args.seed, success_replay_fraction=0., success_batch_fraction=0.)
        replay.enable_mc_returns()
        refs, rewards = [], []
        for i in range(args.batch_size):
            present = mixed.presence_mask[i].numpy()
            obs = V2Observation(mixed.ego_features[i].numpy(), mixed.goal_features[i].numpy(),
                mixed.zone_features[i].numpy()[present], np.ones(int(present.sum()), dtype=np.bool_))
            reward = float(i % 5 - 2)
            rewards.append(reward)
            refs.append(replay.add(obs, np.zeros(eager.action_dim, np.float32), reward, obs,
                i == args.batch_size - 1, collected_stage_step=1))
        replay.fill_episode_returns(refs, discounted_episode_returns(rewards, config.gamma))
        cases = mc_batch_cases(replay, args.batch_size)
        assert int(cases['age_100000'].mc_return_mask.sum()) == args.batch_size
        assert int(cases['age_100001'].mc_return_mask.sum()) == 0
        report['batch_cases'] = {name: dict(valid_count=int(batch.mc_return_mask.sum()),
            shape=list(batch.mc_return.shape), current_zones=batch.obs.max_zone_count,
            next_zones=batch.next_obs.max_zone_count) for name, batch in cases.items()}
        noise_rng = torch.Generator(device='cpu').manual_seed(args.seed)
        launch_steps = {8: 'frozen_targets', 13: 'critic_only', 25010: 'actor'}
        report['phase'] = 'continuous_updates'
        previous_inputs = {}
        for step, case, require_cached in _jobs():
            report['current_update'] = dict(step=step, case=case)
            batch = cases[case]
            normal = torch.randn(batch.action.shape, generator=noise_rng).to(device)
            metrics, states = [], []
            before = 0 if compute._gpu_update_executor is None else compute._gpu_update_executor.captures
            for engine in engines:
                metric, tensors, proof = run_update(engine, batch, step, normal, config,
                    eager=engine is eager,
                    evidence=_mps_graph_replay_evidence if engine is compute and step in launch_steps else None)
                if proof is not None: report['graph_launch_evidence'][launch_steps[step]] = proof
                metrics.append(metric)
                states.append(_snapshot(engine, metric, tensors))
            compared = compare_snapshots(*states)
            executor = compute._gpu_update_executor
            cached = not require_cached or executor.captures == before
            expected_count = float(batch.mc_return_mask.sum())
            counts = all(metric.mc_valid_count == expected_count for metric in metrics)
            empty = expected_count != 0 or all(metric.critic_mc_loss == 0. for metric in metrics)
            # Expected runtime input contribution from the actual Q tensors, not
            # from another forward. Detect stale G/mask even when both modes run.
            runtime = []
            for state, metric in zip(states, metrics):
                q1, q2 = state['tensor.critic.q1'], state['tensor.critic.q2']
                mask, g = batch.mc_return_mask, batch.mc_return
                expected_mc = (((q1-g).square() + (q2-g).square()) * mask).sum() / mask.sum().clamp_min(1.)
                runtime.append(bool(torch.isclose(torch.tensor(metric.critic_mc_loss), expected_mc, rtol=1e-4, atol=1e-5)))
            mode = 'critic_only' if step % config.policy_delay else 'frozen_targets' if step <= config.actor_freeze_steps else 'actor'
            refresh = None
            if mode in previous_inputs:
                old_g, old_mask = previous_inputs[mode]
                if not torch.equal(old_g, g) or not torch.equal(old_mask, mask):
                    stale_mc = (((q1-old_g).square()+(q2-old_g).square())*old_mask).sum()/old_mask.sum().clamp_min(1.)
                    refresh = dict(previous_input_mc_on_same_current_q=float(stale_mc),
                        current_input_mc_on_same_current_q=float(expected_mc),
                        distinguishable=not bool(torch.isclose(stale_mc,expected_mc,rtol=1e-4,atol=1e-5)))
            previous_inputs[mode] = (g.clone(), mask.clone())
            check = dict(name='update', step=step, case=case, passed=compared['passed'] and cached and counts and empty,
                differences=compared['differences'], cached_graph_required=require_cached,
                captures_before=before, captures_after=executor.captures,
                runtime_valid_count=expected_count, valid_count_matches=counts, empty_mc_exact_zero=empty,
                metrics=[asdict(metric) for metric in metrics])
            check.update(runtime_mc_matches=runtime, passed=check['passed'] and all(runtime))
            check['input_refresh'] = refresh
            if require_cached and refresh is not None:
                check['passed'] &= refresh['distinguishable']
            report['checks'].append(check)
            if not check['passed']:
                raise AssertionError(f'MC CUDA mismatch or unexpected capture at step={step}, case={case}.')
        report['execution'] = compute.local_optimization_metadata()
        evidence = report['graph_launch_evidence']
        actual = report['execution']['gpu_td3_compute']
        if (set(evidence) != {'critic_only', 'frozen_targets', 'actor'}
                or not all(item['cuda_graph_launch_count'] > 0 for item in evidence.values())
                or not actual['effective'] or actual['replays'] != len(_jobs())
                or not all(actual['replay_submissions_by_mode'][mode] > 0 for mode in evidence)):
            raise AssertionError('Missing real CUDA Graph execution evidence.')
        report['phase'] = 'checkpoint_rebuild'
        old_executor = compute._gpu_update_executor
        for engine in engines:
            engine.load_checkpoint_state_dict(deepcopy(engine.checkpoint_state_dict()))
        if compute._gpu_update_executor is not None or old_executor.graphs:
            raise AssertionError('Checkpoint load did not invalidate Graphs.')
        reloaded = []
        for engine in engines:
            metric, collected, _ = run_update(engine, cases['changed'], 100006, normal, config, eager=engine is eager)
            reloaded.append(_snapshot(engine, metric, collected))
        rebuilt = compare_snapshots(*reloaded)
        rebuilt['new_executor'] = compute._gpu_update_executor is not old_executor
        rebuilt['captures'] = compute._gpu_update_executor.captures
        rebuilt['passed'] &= rebuilt['new_executor'] and rebuilt['captures'] > 0
        report['checks'].append(dict(name='checkpoint_rebuild', **rebuilt))
        if not rebuilt['passed']: raise AssertionError('Checkpoint Graph rebuild comparison failed.')
        report['execution_after_reload'] = compute.local_optimization_metadata()
        if (sha256(args.bc_checkpoint.read_bytes()).hexdigest() != bc_hash
                or sha256(args.validation_pool.read_bytes()).hexdigest() != pool_hash):
            raise RuntimeError('Diagnostic inputs changed during execution.')
        if (source_identity()['sha256'] != report['source']['sha256'] or
                sha256(args.source_manifest.read_bytes()).hexdigest() != report['source_manifest_sha256']):
            raise RuntimeError('Diagnostic source changed during execution.')
        report.update(passed=True, completed=True, execution_completed=True, phase='completed')
        return report
    except BaseException as exc:
        exc.mc_return_cuda_report = report
        raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    if not args.launch:
        print(json.dumps(plan, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / 'summary.json'
    report = dict(plan=plan, passed=False, completed=False, execution_completed=False, phase='starting')
    started = time.perf_counter()
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    try:
        report = run_diagnostic(args, plan)
    except BaseException as exc:
        report = getattr(exc, 'mc_return_cuda_report', report)
        report.update(passed=False, completed=False, execution_completed=False,
            error=dict(type=type(exc).__name__, detail=str(exc), traceback=traceback.format_exc()))
    report['process_wall_seconds'] = time.perf_counter() - started
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], summary=str(path)), indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
