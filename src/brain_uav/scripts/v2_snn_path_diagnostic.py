"""Bounded checkpoint-based eager/formal-path comparison, invoked by profile_v2_td3.

All tensor copies/hooks are diagnostic-only. No rollout, training checkpoint or
performance result is produced. CUDA evidence must come from the engine verifier.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import random
import subprocess
import json
from math import isfinite

import numpy as np
import torch

from brain_uav.models import V2ANNCritic
from brain_uav.models.v2_policy_config import V2SpikeRegularizationConfig
from brain_uav.observations import V2ObservationBatch, collate_v2_observations
from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_inputs, file_sha256, _software_versions
from brain_uav.trainers.v2_formal_training import _architecture_from_engine_payload
from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine


def flatten(value, name, result):
    """Independent host snapshots, including None; never write them into live grad."""
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(item, f'{name}.{key}', result)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            flatten(item, f'{name}.{index}', result)
    elif value is None:
        result[name + '.none'] = torch.tensor(True)
    elif isinstance(value, (torch.Tensor, int, float, bool)):
        result[name] = torch.as_tensor(value).detach().cpu().clone()
    elif not isinstance(value, str):
        raise TypeError(f'Unsupported snapshot {name}: {type(value)}')


def engine_snapshot(engine):
    result = {}
    for name in ('actor', 'critic1', 'critic2', 'actor_target', 'critic1_target',
                 'critic2_target', 'bc_reference_actor'):
        module = getattr(engine, name)
        flatten(None if module is None else module.state_dict(), name, result)
    for name in ('actor', 'critic'):
        flatten(getattr(engine, name + '_optimizer').state_dict(), 'optimizer.' + name, result)
    for name in ('update_count', 'critic_update_count', 'critic_target_update_count',
                 'actor_update_count', 'last_total_steps'):
        flatten(getattr(engine, name), 'counts.' + name, result)
    return result


class DiagnosticMismatchError(AssertionError):
    """All requested comparisons completed, but at least one failed."""


def parameter_snapshot(engine):
    return {f'{module_name}.{name}': parameter.detach().cpu().clone()
        for module_name in ('actor', 'critic1', 'critic2', 'actor_target', 'critic1_target', 'critic2_target')
        for name, parameter in getattr(engine, module_name).named_parameters()}


def parameter_delta(before, after):
    if before.keys() != after.keys():
        raise AssertionError('Parameter fields changed during update')
    for name in before:
        if before[name].shape != after[name].shape or before[name].dtype != after[name].dtype:
            raise AssertionError(f'Parameter shape/dtype changed during update: {name}')
    # Subtract in double after independent snapshots, preserving small update evidence.
    return {'update_delta.' + name: after[name].double() - value.double()
            for name, value in before.items()}


def _difference_evidence(output, stage, name, left, right):
    """At most first + six representative files; bounded samples, never full engines."""
    if output is None:
        return []
    output = Path(output)
    categories = (('gradient.pre_clip.', 'pre_clip'), ('gradient.post_clip.', 'post_clip'),
        ('update_delta.', 'delta'), ('optimizer.', 'adam'), ('.spikes.', 'spikes'), ('actor.', 'actor_terms'))
    paths = [output / 'first_difference.pt']
    category = next((label for prefix, label in categories if prefix in name), None)
    if category:
        paths.append(output / f'difference_{category}.pt')
    saved = []
    for path in paths:
        if path.exists():
            continue
        # 16384 elements per side maximum; keep originals' shape and truncation explicit.
        torch.save({'stage': stage, 'name': name, 'eager_shape': list(left.shape),
            'formal_shape': list(right.shape), 'truncated': left.numel() > 16384 or right.numel() > 16384,
            'eager': left.reshape(-1)[:16384].clone(), 'formal': right.reshape(-1)[:16384].clone()}, path)
        saved.append(str(path))
    return saved


def compare_tensors(expected, actual, *, stage, results, output=None,
                    continue_numeric_mismatch=False):
    """Opt-in collection continues finite floating mismatches; default remains fail-fast."""
    from brain_uav.scripts.profile_v2_td3 import COMPILED_NUMERIC_RTOL, COMPILED_NUMERIC_ATOL
    if expected.keys() != actual.keys():
        results.append({'stage': stage, 'passed': False, 'tensor': 'field_set',
                        'failure_kind': 'structural',
                        'missing': sorted(expected.keys() - actual.keys()),
                        'extra': sorted(actual.keys() - expected.keys())})
        raise AssertionError(f'{stage}: tensor fields differ')
    for name, left in expected.items():
        right = actual[name]
        discrete = not left.is_floating_point() or name.startswith('counts.') or name.endswith('.step')
        exact = discrete or '.spikes.' in name
        rtol, atol = (0., 0.) if exact else (COMPILED_NUMERIC_RTOL, COMPILED_NUMERIC_ATOL)
        compatible = left.shape == right.shape and left.dtype == right.dtype
        difference = (right.double() - left.double()).abs() if compatible else None
        finite = compatible and bool(torch.isfinite(left).all() & torch.isfinite(right).all())
        mismatch = None if not finite else difference > (atol + rtol * left.double().abs())
        passed = finite and not bool(mismatch.any())
        def scalar(value):
            value = float(value)
            return value if isfinite(value) else None
        left_l2 = scalar(torch.linalg.vector_norm(left.double().reshape(-1))) if finite else None
        right_l2 = scalar(torch.linalg.vector_norm(right.double().reshape(-1))) if finite else None
        diff_l2 = scalar(torch.linalg.vector_norm(difference.reshape(-1))) if finite else None
        ratio = None
        denominator_status = 'unavailable'
        if left_l2 is not None and diff_l2 is not None:
            if left_l2 == 0:
                denominator_status = 'zero'
                ratio = 0. if diff_l2 == 0 else None
            elif left_l2 <= 1e-12:
                denominator_status = 'near_zero'
            else:
                denominator_status = 'nonzero'
                ratio = scalar(diff_l2 / left_l2)
        numeric = finite and not discrete
        row = {'stage': stage, 'tensor': name, 'passed': passed, 'rtol': rtol, 'atol': atol,
               'element_count': left.numel(), 'shape': list(left.shape),
               'max_abs_error': scalar(difference.max()) if finite and difference.numel() else None,
               'max_relative_error': scalar((difference / left.double().abs().clamp_min(1e-5)).max())
                   if finite and difference.numel() else None,
               'exceeded_count': int(mismatch.sum()) if finite else None,
               'exceeded_fraction': float(mismatch.sum()) / left.numel() if finite and left.numel() else (0. if finite else None),
               'eager_l2': left_l2, 'formal_l2': right_l2, 'difference_l2': diff_l2,
               'relative_l2': ratio, 'relative_l2_denominator': denominator_status,
               'relative_l2_near_zero_threshold': 1e-12,
               'shape_dtype_match': compatible, 'finite': finite,
               'failure_kind': None if passed else ('numeric' if numeric else 'structural_or_discrete')}
        results.append(row)
        if not passed:
            row['evidence_paths'] = _difference_evidence(output, stage, name, left, right)
            if not (continue_numeric_mismatch and numeric):
                raise AssertionError(f'{stage}: {name} differs')



def restore_engine(inputs, *, fused):
    """Restore full engine, including saved optimizers/reference; replay is synthetic."""
    payload = inputs['checkpoint_payload']['engine_checkpoint']
    scales, encoder, radius, _, width1, width2, limit, _ = _architecture_from_engine_payload(payload, model_type='snn')
    actor = deepcopy(inputs['actor']).requires_grad_(True)
    algorithm = deepcopy(payload['algorithm_config'])
    algorithm['spike_regularization'] = V2SpikeRegularizationConfig.from_snapshot(algorithm['spike_regularization'])
    replay = V2ReplayBuffer(max(algorithm['batch_size'], 2), 2, 2, seed=7)
    engine = V2TD3UpdateEngine(actor,
        V2ANNCritic(scales, 2, width1, uav_radius=radius, encoder_config=encoder),
        V2ANNCritic(scales, 2, width2, uav_radius=radius, encoder_config=encoder), replay,
        actor_lr=payload['actor_optimizer_state_dict']['param_groups'][0]['lr'],
        critic_lr=payload['critic_optimizer_state_dict']['param_groups'][0]['lr'],
        action_low=-limit, action_high=limit, device=inputs['device'],
        bc_reference_actor=actor if payload['bc_reference_present'] else None,
        fused_adam=fused, reduce_update_stat_syncs=True, **algorithm)
    engine.load_checkpoint_state_dict(payload)
    from brain_uav.scripts.profile_v2_td3 import _set_numeric_engine_modes
    _set_numeric_engine_modes(engine)
    return engine


def diagnostic_cases(engine):
    """Each short sequence starts from the checkpoint, with synthetic stage counts."""
    delay = engine.policy_delay
    if delay <= 1:
        raise ValueError('This short diagnostic requires policy_delay > 1.')
    def at_or_after(value):
        return ((value + delay - 1) // delay) * delay
    freeze = engine.actor_freeze_steps
    spike = engine.spike_regularization
    return [('freeze_boundary', tuple(dict.fromkeys((max(0, freeze - 1), freeze,
                           *range(freeze + 1, at_or_after(freeze + 1) + 2))))),
            ('ramp_middle', (at_or_after(max(freeze + 1, (spike.ramp_start_steps + spike.ramp_end_steps) // 2)),
                             at_or_after(max(freeze + 1, (spike.ramp_start_steps + spike.ramp_end_steps) // 2)) + 1,
                             at_or_after(max(freeze + 1, (spike.ramp_start_steps + spike.ramp_end_steps) // 2)) + delay)),
            ('ramp_full', (at_or_after(max(freeze + 1, spike.ramp_end_steps)),
                           at_or_after(max(freeze + 1, spike.ramp_end_steps)) + 1,
                           at_or_after(max(freeze + 1, spike.ramp_end_steps)) + delay))]


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def assert_rng_equal(before, after):
    numpy_equal = (before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
                   and before[1][2:] == after[1][2:])
    cuda_equal = ((before[3] is None and after[3] is None) or
                  (before[3] is not None and after[3] is not None and len(before[3]) == len(after[3])
                   and all(torch.equal(a, b) for a, b in zip(before[3], after[3]))))
    if not (before[0] == after[0] and numpy_equal and torch.equal(before[2], after[2]) and cuda_equal):
        raise AssertionError('Compile warmup changed Python/NumPy/torch/CUDA RNG')


@contextmanager
def capture_forward(actor, result, prefix):
    """The LIFs are eager even in the formal compiled-encoder path; no extra forward."""
    handles = []
    for layer in (1, 2):
        def hook(module, args, output, layer=layer):
            result[f'{prefix}.spikes.l{layer}'] = output.detach().cpu().clone()
            result[f'{prefix}.rate.l{layer}'] = output.detach().mean().cpu().clone()
        handles.append(getattr(actor.snn_head, 'lif' + str(layer)).register_forward_hook(hook))
    def action_hook(module, args, output):
        result[prefix + '.action'] = (torch.tanh(output) * actor.action_limit).detach().cpu().clone()
    handles.append(actor.snn_head.action_layer.register_forward_hook(action_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for layer in (1, 2):
            voltage = getattr(actor.snn_head, 'lif' + str(layer)).v
            result[f'{prefix}.reset.l{layer}'] = torch.tensor(
                bool((voltage == 0).all()) if isinstance(voltage, torch.Tensor) else voltage == 0)
            if not bool(result[f'{prefix}.reset.l{layer}']):
                raise AssertionError(f'{prefix}: LIF{layer} not reset')


@contextmanager
def capture_gradients(engine, result):
    """Observe gradients on both sides of the ORIGINAL check/clip, before Adam."""
    original = engine._validate_and_clip_gradients
    previous_override = engine.__dict__.get('_validate_and_clip_gradients')
    had_override = '_validate_and_clip_gradients' in engine.__dict__
    def checked(parameters, **kwargs):
        component = kwargs['component']
        modules = (('actor', engine.actor),) if component == 'actor' else (
            ('critic1', engine.critic1), ('critic2', engine.critic2))
        def snapshot(phase):
            for model_name, module in modules:
                for name, parameter in module.named_parameters():
                    flatten(parameter.grad, f'gradient.{phase}.{model_name}.{name}', result)
        snapshot('pre_clip')
        value = original(parameters, **kwargs)
        snapshot('post_clip')
        return value
    engine._validate_and_clip_gradients = checked
    try:
        yield
    finally:
        if had_override:
            engine._validate_and_clip_gradients = previous_override
        else:
            del engine.__dict__['_validate_and_clip_gradients']


def observation_cases(inputs, trajectory=None):
    """Small fixed pool subset; valid boundary states built by the actual observation packer."""
    from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
    pool = inputs['pool']
    selected = {}
    for record in pool.scenarios:
        count = len(record['payload']['zones'])
        if count in (0, 1, 2):
            selected.setdefault(count, record)
    if set(selected) != {0, 1, 2}:
        raise ValueError('Original pool must contain 0, 1 and 2-zone scenes for this diagnostic.')
    env = V2StaticNoFlyTrajectoryEnv(inputs['scenario'], inputs['rewards'],
        uav_collision_radius=pool.uav_collision_radius, cache_ellipsoid_segment_clearance=True)
    observations, sources = [], []
    for count in (0, 1, 2):
        record = selected[count]
        obs, _ = env.reset(options={'scenario': record['payload']})
        observations.append(obs)
        sources.append({'kind': 'original_pool_initial', 'scene_id': record.get('scene_id', record.get('scenario_id')),
                        'pool_index': list(pool.scenarios).index(record), 'zone_count': count, 'step': 0})
    # Keep these scenarios geometry-free so heights are valid observations, not collision fixtures.
    env.reset(options={'scenario': selected[0]['payload']})
    for label, height in (('near_ground', env.scenario.world_z_min + .1),
                          ('near_upper', env.scenario.world_z_max - .1)):
        env.state[2] = height
        observations.append(env._get_obs())
        sources.append({'kind': 'synthetic_valid_state', 'label': label, 'state': env.state.tolist(),
                        'source_pool_index': list(pool.scenarios).index(selected[0])})
    env.reset(options={'scenario': selected[1]['payload']})
    start = env.state[:3].copy()
    bounds = env.zones[0].shape.bounding_box()
    center = (bounds.min_corner + bounds.max_corner) / 2
    def clearance(position):
        return min(zone.point_clearance(position, env.uav_collision_radius) for zone in env.zones)
    if clearance(start) > 1 and clearance(center) <= 1:
        low, high = start, center
        for _ in range(30):
            middle = (low + high) / 2
            if clearance(middle) > 1:
                low = middle
            else:
                high = middle
        env.state[:3] = low
        observations.append(env._get_obs())
        sources.append({'kind': 'synthetic_valid_state', 'label': 'near_obstacle',
            'state': env.state.tolist(), 'clearance': clearance(low),
            'source_pool_index': list(pool.scenarios).index(selected[1])})
    else:
        raise ValueError('Selected original scene cannot provide a valid near-obstacle observation.')
    if trajectory is None:
        candidates = sorted(inputs['checkpoint_path'].parent.glob(
            '*_metrics_reports/trajectories/training/*_boundary.json'))
        trajectory = candidates[-1] if candidates else None
    if trajectory is not None:
        path = Path(trajectory).resolve()
        saved = json.loads(path.read_text(encoding='utf-8'))
        if (saved['scenario_config'] != inputs['checkpoint_payload']['scenario_config']
                or saved['uav_collision_radius'] != pool.uav_collision_radius
                or saved['model_type'] != 'snn' or saved['stage'] != inputs['checkpoint_payload']['stage']):
            raise ValueError('Saved trajectory configuration/model/stage differs from checkpoint.')
        env.reset(options={'scenario': saved['scenario_payload']})
        state = np.asarray(saved['terminal_state'], dtype=np.float32)
        if state.shape != (5,) or not np.isfinite(state).all():
            raise ValueError('Saved terminal observation state is invalid.')
        env.state = state.copy()
        observations.append(env._get_obs())
        sources.append({'kind': 'saved_training_terminal_state', 'path': str(path), 'sha256': file_sha256(path),
            'global_steps': saved['global_steps'], 'episode_step': saved['episode_length'],
            'outcome': saved['outcome'], 'state': state.tolist(),
            'scenario_seed': saved['scenario_payload'].get('metadata', {}).get('scenario_seed'),
            'note': 'Historical training scene, not fixed-pool sample or terminal-checkpoint rollout; terminal may be outside world. Full intermediate attitude was not saved.'})
    return observations, sources


def run_snn_path_diagnostic(*, checkpoint, validation_pool, output_dir, device='cuda',
                            expected_digest=None, cpu_smoke=False, trajectory=None):
    """CUDA required for formal proof; explicit CPU smoke never reports CUDA passed."""
    from brain_uav.scripts import profile_v2_td3 as profile
    from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation, _configure_stage_compilation
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Output must be new or empty.')
    if device != 'cuda' and not cpu_smoke:
        raise ValueError('Formal path check requires CUDA; CPU only via explicit --snn-path-cpu-smoke.')
    if cpu_smoke and device != 'cpu':
        raise ValueError('CPU smoke must explicitly use --device cpu.')
    inputs = load_diagnostic_inputs(model='snn', checkpoint=checkpoint, validation_pool=validation_pool, device=device)
    if expected_digest is not None and inputs['pool'].content_digest != expected_digest:
        raise ValueError('Requested pool digest differs.')
    saved_rng = rng_state()
    output.mkdir(parents=True, exist_ok=True)
    payload = inputs['checkpoint_payload']
    observations, sources = observation_cases(inputs, trajectory)
    # Small view for existing warmup: original pool records, no generation or changed provenance.
    selected = {}
    for record in inputs['pool'].scenarios:
        selected.setdefault(len(record['payload']['zones']), record)
    warmup_pool = SimpleNamespace(scenarios=tuple(selected[c] for c in (0, 1, 2)), stage_seed=inputs['pool'].stage_seed)
    prepared = SimpleNamespace(scenario_config=inputs['scenario'], reward_config=inputs['rewards'],
                               uav_collision_radius=inputs['pool'].uav_collision_radius)
    repo = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()
    settings = default_v2_cuda_graph_compilation(model='snn', resolved_device=device)
    manifest = {'format': 'v2_snn_path_consistency', 'format_version': 2,
        'diagnostic_only': True, 'checkpoint_path': str(inputs['checkpoint_path']),
        'checkpoint_sha256': inputs['checkpoint_hash'], 'checkpoint_status': payload['status'],
        'passed_validation': payload['passed_validation'], 'training_steps': payload['training_result']['stage_steps'],
        'pool_path': str(inputs['pool_path']), 'pool_sha256': inputs['pool_hash'],
        'pool_content_digest': inputs['pool'].content_digest, 'git_head': git('rev-parse', 'HEAD'),
        'git_dirty': git('status', '--short'), 'software': _software_versions(),
        'device': str(inputs['device']), 'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
        'architecture': payload['engine_checkpoint']['architecture'],
        'algorithm': payload['engine_checkpoint']['algorithm_config'], 'formal_config': payload['formal_config'],
        'scenario_config': payload['scenario_config'], 'reward_config': payload['reward_config'],
        'uav_collision_radius': payload['uav_collision_radius'],
        'initialization_source': payload['initialization_source'],
        'saved_update_counts': payload['engine_checkpoint']['update_counts'],
        'bc_reference_restored': payload['engine_checkpoint']['bc_reference_present'],
        'bc_schedule': payload['bc_schedule'], 'requested_compilation': settings,
        'experimental_optimizations': {'C': False, 'A': False, 'critic_tail': False,
            'window_update_statistics': False, 'reuse_update_buffers': False, 'pinned_batch_transfer': False,
            'aggregate_relation_values_first': False, 'single_observation_fast_path': False},
        'restoration': 'All six networks, BC reference, Adam state and saved counts restored strictly. Only last_total_steps is rebased for independent diagnostic boundary cases; all update counts start at saved values. Replay/RNG history not saved: fixed synthetic batch and matched explicit RNG. Both sides use identical fused Adam and saved target policy_noise/noise_clip (not a reconstruction of historical replay/noise).',
        'observation_sources': sources,
        'saved_failure_observations': 'Saved training terminal state included when found or explicitly supplied; intermediate attitudes are not reconstructed. No claim of reproducing a terminal-policy failed rollout.',
        'candidate_path': 'evaluate_v2_fixed_validation uses actor(batch), eval/inference_mode; compiled online encoder is retained. Compared below on same inputs, without rollouts.',
        'cuda_check': 'not_run_cpu_smoke' if cpu_smoke else 'requested',
        'precision': {'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'default_dtype': str(torch.get_default_dtype()),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
        'gpu_capability': list(torch.cuda.get_device_capability()) if device == 'cuda' else None,
        'torch_num_threads': torch.get_num_threads()}
    profile._strict_json_write(output / 'manifest.json', manifest)
    summary = {'diagnostic_only': True, 'passed': False, 'cuda_status': 'not_run' if cpu_smoke else 'pending',
               'cpu_smoke_passed': None, 'cuda_passed': None,
               'execution_completed': False, 'execution_status': 'running', 'exit_code': 1,
               'delta_semantics': 'Independent before/after snapshots per side; double subtraction. Later steps within a case include accumulated earlier differences; each case starts from checkpoint, not each step.',
               'graph_evidence_scope': 'Warmup verifier proves replay for probed callable/shapes; per-update execution entries prove requested callable use, not independent cudaGraphLaunch verification for every compared call.',
               'comparisons': [], 'cases': [], 'effective_compilation': [], 'completed_comparison_stages': [],
               'gradient_semantics': 'pre_clip and post_clip recorded before Adam, current backward only; unused retained actor gradients not consumed or rewritten.'}
    def check(*args, **kwargs):
        kwargs.setdefault('output', output)
        compare_tensors(*args, **kwargs)
        summary['completed_comparison_stages'].append(kwargs['stage'])

    cpu_configuration = {'requested': False, 'enabled_objects': [], 'cuda_graph': False,
        'cuda_graph_actor_update': False, 'cuda_graph_action_inference': False,
        'reason': 'Explicit CPU smoke: both engines eager, identical fused Adam; no compiled/CUDA correctness claim.'}
    current_stage = 'restore'
    try:
        eager = restore_engine(inputs, fused=True)
        formal = restore_engine(inputs, fused=True)
        check(engine_snapshot(eager), engine_snapshot(formal), stage='restored_initial', results=summary['comparisons'])
        before = engine_snapshot(formal)
        before_rng = rng_state()
        current_stage = 'compile_warmup_and_replay_evidence'
        compile_settings = {k: v for k, v in settings.items() if k not in ('fused_adam', 'reduce_update_stat_syncs')}
        configured = dict(cpu_configuration) if cpu_smoke else _configure_stage_compilation(formal, warmup_pool, prepared,
            compile_critic_encoder=False, compile_target_encoders=False, **compile_settings)
        summary['effective_compilation'].append(configured)
        check(before, engine_snapshot(formal), stage='warmup_preserves_state', results=summary['comparisons'])
        assert_rng_equal(before_rng, rng_state())
        batches = [collate_v2_observations([obs], device=inputs['device']) for obs in observations]
        mixed = collate_v2_observations([observations[i % 3] for i in range(eager.batch_size)], device=inputs['device'])
        # Actual select_action, repeated A->A and A->B->A. Hooks observe this SAME call.
        repeat_reference = {}
        for index in (0, 0, 1, 0, *range(2, len(observations))):
            current_stage = f'select_action.input{index}'
            records = []
            for side, engine in enumerate((eager, formal)):
                row = {}
                with capture_forward(engine.actor, row, 'inference'):
                    action = engine.select_action(observations[index], exploration_noise=0)
                row['select_action'] = torch.from_numpy(action.copy())
                key = (side, index)
                if key in repeat_reference:
                    check(repeat_reference[key], row, stage=f'reset_repeat.side{side}.input{index}',
                                    results=summary['comparisons'], output=output, continue_numeric_mismatch=True)
                else:
                    repeat_reference[key] = row
                records.append(row)
            check(*records, stage=f'select_action.{index}.{len(summary["comparisons"])}', results=summary['comparisons'], output=output, continue_numeric_mismatch=True)
        # Direct actor call used by candidate, and mixed masks not supported by select_action's singleton API.
        for index, batch in enumerate([*batches, mixed]):
            current_stage = f'candidate.input{index}'
            rows = []
            for engine in (eager, formal):
                training = engine.actor.training
                engine.actor.eval()
                try:
                    with torch.inference_mode(), capture_forward(engine.actor, row := {}, 'candidate'):
                        row['candidate.action'] = engine.actor(batch).detach().cpu().clone()
                finally:
                    engine.actor.train(training)
                rows.append(row)
            check(*rows, stage=f'candidate.{index}', results=summary['comparisons'], output=output, continue_numeric_mismatch=True)
            if index < len(observations):
                for side in (0, 1):
                    previous = repeat_reference[(side, index)]
                    # Select_action and candidate are different public calls: report that
                    # comparison explicitly, never substitute one path's activity for another.
                    left = {k.replace('inference.', ''): v for k, v in previous.items() if k.startswith('inference.')}
                    right = {k.replace('candidate.', ''): v for k, v in rows[side].items()}
                    check(left, right, stage=f'candidate_vs_select.side{side}.input{index}',
                                    results=summary['comparisons'], output=output, continue_numeric_mismatch=True)
        for label, steps in diagnostic_cases(eager):
            current_stage = label + '.restore_and_warmup'
            # Fresh complete state for every independent boundary/ramp case. Never load/replace modules after capture.
            eager = restore_engine(inputs, fused=True)
            formal = restore_engine(inputs, fused=True)
            for engine in (eager, formal):
                engine.last_total_steps = max(0, steps[0] - 1)
            before = engine_snapshot(formal)
            before_rng = rng_state()
            configured = dict(cpu_configuration) if cpu_smoke else _configure_stage_compilation(formal, warmup_pool, prepared,
                compile_critic_encoder=False, compile_target_encoders=False, **compile_settings)
            summary['effective_compilation'].append(configured)
            assert_rng_equal(before_rng, rng_state())
            check(before, engine_snapshot(formal), stage=label + '.warmup', results=summary['comparisons'])
            check(engine_snapshot(eager), engine_snapshot(formal), stage=label + '.initial', results=summary['comparisons'])
            for index, step in enumerate(steps):
                current_stage = f'{label}.step{step}'
                obs = mixed if index % 2 == 0 else profile._terminal_numeric_observation_batch(mixed, eager)
                fixed = profile._fixed_numeric_replay_batch(obs, action_dim=2, device=inputs['device'])
                fixed = replace(fixed, next_obs=V2ObservationBatch(*(t.clone() for t in (
                    obs.ego_features, obs.goal_features, obs.zone_features, obs.presence_mask))))
                outputs = []
                torch.manual_seed(7 + index)
                if device == 'cuda':
                    torch.cuda.manual_seed_all(7 + index)
                noise_rng = rng_state()
                bc = payload['bc_schedule']
                # Strict loader already validated this exact metadata; use its boundaries,
                # including historical default schedules, rather than guessing CLI settings.
                bc_lambda = next(value for boundary, value in reversed(list(zip(bc['boundaries'], bc['values'])))
                                 if step >= boundary)
                for engine in (eager, formal):
                    current_stage = f'{label}.step{step}.' + ('eager' if engine is eager else 'formal')
                    restore_rng(noise_rng)
                    engine.replay.sample = lambda size, fixed=fixed: fixed
                    row = {}
                    parameters_before = parameter_snapshot(engine)
                    executions = []
                    def record(name, value):
                        row[name] = value.detach().cpu().clone()
                    with (capture_gradients(engine, row), capture_forward(engine.actor, row, 'actor_forward'),
                          capture_forward(engine.actor_target, row, 'target_forward')):
                        metric = engine.update_once(total_steps=step, bc_lambda=bc_lambda,
                            diagnostic_tensor_recorder=record, compiled_execution_recorder=executions.append)
                    if metric.actor_updated != (step > engine.actor_freeze_steps and step % engine.policy_delay == 0):
                        raise AssertionError('Actor freeze/policy_delay execution differs from requested diagnostic step')
                    if metric.actor_updated:
                        torch.testing.assert_close(row['actor_forward.rate.l1'], row['actor.spike_rate'], rtol=0, atol=0)
                    for target in (engine.actor_target, engine.critic1_target, engine.critic2_target):
                        if any(p.grad is not None for p in target.parameters()):
                            raise AssertionError('Target parameter received a gradient')
                    if engine is formal and not cpu_smoke:
                        required = ['critic_block', 'target_critic_td_block', 'snn_target_encoder']
                        if metric.actor_updated:
                            required += ['actor_loss', 'frozen_critic_context']
                        if any(name not in executions for name in required):
                            raise AssertionError(f'Requested formal path not executed: {required}; got {executions}')
                    flatten(asdict(metric), 'metrics', row)
                    row.update(parameter_delta(parameters_before, parameter_snapshot(engine)))
                    row.update(engine_snapshot(engine))
                    outputs.append(row)
                current_stage = f'{label}.step{step}.comparison'
                check(*outputs, stage=f'{label}.step{step}', results=summary['comparisons'], output=output, continue_numeric_mismatch=True)
                summary['cases'].append({'case': label, 'diagnostic_stage_steps': step,
                    'actor_backward': metric.actor_updated, 'beta': metric.spike_beta_eff,
                    'prior_updates_in_case': index, 'may_include_accumulated_differences': index > 0,
                    'bc_lambda': bc_lambda, 'fixed_input': 'mixed masks; independent next-observation storage',
                    'formal_execution_entries': executions,
                    'noise': 'same explicit torch CPU/CUDA RNG before actual update_once noise consumption'})
        if file_sha256(inputs['checkpoint_path']) != inputs['checkpoint_hash'] or file_sha256(inputs['pool_path']) != inputs['pool_hash']:
            raise AssertionError('Input artifact changed')
        summary['execution_completed'] = True
        summary['passed'] = all(row['passed'] for row in summary['comparisons'])
        summary['exit_code'] = 0 if summary['passed'] else 1
        summary['execution_status'] = 'completed_consistent' if summary['passed'] else 'completed_inconsistent'
        summary['cpu_smoke_passed'] = summary['passed'] if cpu_smoke else None
        summary['cuda_passed'] = None if cpu_smoke else summary['passed']
        summary['cuda_status'] = 'not_run' if cpu_smoke else ('passed' if summary['passed'] else 'failed')
        if not summary['passed']:
            raise DiagnosticMismatchError('Diagnostic completed with tensor mismatches; see comparisons.json')
        return summary
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        summary['cpu_smoke_passed'] = False if cpu_smoke else None
        summary['cuda_passed'] = None if cpu_smoke else False
        if not summary['execution_completed']:
            summary['execution_status'] = 'aborted'
            summary['abort_stage'] = current_stage
            summary['abort_reason'] = summary['error']
        summary['cuda_status'] = 'not_run' if cpu_smoke else 'failed'
        if hasattr(exc, 'cuda_graph_evidence'):
            summary['cuda_graph_failure_evidence'] = exc.cuda_graph_evidence
        raise
    finally:
        restore_rng(saved_rng)
        comparisons = summary.pop('comparisons')
        profile._strict_json_write(output / 'comparisons.json', {'tensors': comparisons})
        by_stage = {}
        for row in comparisons:
            aggregate = by_stage.setdefault(row['stage'], {'passed': True, 'tensor_count': 0,
                'max_abs_error': 0., 'max_relative_error': 0., 'exceeded_count': 0})
            aggregate['passed'] &= row['passed']
            aggregate['tensor_count'] += 1
            for key in ('max_abs_error', 'max_relative_error'):
                aggregate[key] = max(aggregate[key], row.get(key) or 0.)
            aggregate['exceeded_count'] += row.get('exceeded_count') or 0
        summary['stages'] = by_stage
        summary['failed_stages'] = [stage for stage, detail in by_stage.items() if not detail['passed']]
        summary['first_difference'] = next((r for r in comparisons if not r['passed']), None)
        summary['first_failure_stage'] = (summary['first_difference']['stage'] if summary['first_difference'] else summary.get('abort_stage'))
        summary['completed_test_points'] = [r['case'] + '.step' + str(r['diagnostic_stage_steps']) for r in summary['cases']]
        summary['failed_tensor_count'] = sum(not r['passed'] for r in comparisons)
        summary['evidence_paths'] = sorted({path for row in comparisons for path in row.get('evidence_paths', [])})
        profile._strict_json_write(output / 'summary.json', summary)
        print(json.dumps({'snn_path_consistency': {'passed': summary['passed'],
            'cuda_status': summary['cuda_status'], 'execution_status': summary['execution_status'],
            'exit_code': summary['exit_code'], 'summary': str(output / 'summary.json'),
            'error': summary.get('error')}}, allow_nan=False), flush=True)
