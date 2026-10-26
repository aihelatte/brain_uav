"""Diagnostic-only terminal critic ranking against full own-actor continuations.

No optimizer, replay, training engine, compilation or formal validation status.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.models import V2ANNCritic
from brain_uav.observations import collate_v2_observations
from brain_uav.scripts.diagnose_v2_policy import (
    OUTCOMES, _software_versions, _write_json, file_sha256, load_diagnostic_inputs,
    load_diagnostic_checkpoint,
)
from brain_uav.trainers.v2_formal_training import (
    _architecture_from_engine_payload,
)
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from brain_uav.trainers.v2_validation import load_v2_validation_pool


POOL_DIGEST = '2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b'
SELECTION_SEED = 20261026
TIE_ATOL = 1e-3
TIE_RTOL = 1e-4


def serial(value):
    """Strict JSON; geometry's legitimate empty-scene infinities stay explicit."""
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError('NaN in diagnostic state.')
        return '+inf' if value > 0 else '-inf'
    return value


def digest(value):
    return sha256(json.dumps(serial(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f'Non-finite {name}.')
    return value


def discounted_return(rewards, gamma):
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('Invalid gamma.')
    total = 0.
    for reward in reversed(rewards):
        total = float(finite(reward, 'reward')) + gamma * total
    return float(finite(total, 'discounted return'))


def load_models(checkpoint, pool_path, device):
    # Independent diagnostics accept strictly validated non-terminal snapshots too.
    payload = load_diagnostic_checkpoint(checkpoint, expected_stage='easy')
    model = payload.get('model_type', 'ann')
    data = load_diagnostic_inputs(model=model, checkpoint=checkpoint, validation_pool=pool_path, device=device)
    engine = data['checkpoint_payload']['engine_checkpoint']
    scales, encoder, radius, _, hidden1, hidden2, _, _ = _architecture_from_engine_payload(engine, model_type=model)
    critics = []
    for index, hidden in enumerate((hidden1, hidden2), 1):
        critic = V2ANNCritic(scales, 2, hidden, uav_radius=radius, encoder_config=encoder)
        name = f'critic{index}_state_dict'
        state = engine[name]
        V2TD3UpdateEngine._validate_state_dict(critic, state, name=name)
        for key, expected in critic.named_buffers():
            if not torch.equal(state[key].detach().cpu(), expected.detach().cpu()):
                raise ValueError(f'{name} fixed buffer {key} disagrees with architecture.')
        for key, tensor in state.items():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Non-finite {name}.{key}.')
        critic.load_state_dict(state, strict=True)
        critics.append(critic.to(data['device']).eval().requires_grad_(False))
    data['critics'] = critics
    data['model'] = model
    return data


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [row for key in sorted(set(a) | set(b))
                for row in differences(a.get(key), b.get(key), f'{path}.{key}'.strip('.'))]
    return [] if a == b else [{'field': path, 'A': a, 'B': b}]


def compare_inputs(a, b):
    pa, pb = a['checkpoint_payload'], b['checkpoint_payload']
    # Different reward/dynamics/action semantics would invalidate a shared probe.
    for key in ('stage', 'scenario_config', 'reward_config', 'uav_collision_radius'):
        if pa[key] != pb[key]:
            raise ValueError(f'Inputs disagree on shared probe semantics: {key}.')
    for key in ('architecture', 'action_low', 'action_high', 'observation_contract'):
        if pa['engine_checkpoint'][key] != pb['engine_checkpoint'][key]:
            raise ValueError(f'Inputs disagree on {key}.')
    if a['gamma'] != b['gamma'] or a['pool'].content_digest != b['pool'].content_digest:
        raise ValueError('Inputs disagree on gamma/pool.')
    keys = ('formal_config', 'bc_schedule', 'seed_manifest', 'validation_pool', 'initialization_source')
    result = [row for key in keys for row in differences(pa.get(key), pb.get(key), key)]
    result += differences(pa['engine_checkpoint']['algorithm_config'], pb['engine_checkpoint']['algorithm_config'], 'algorithm_config')
    return result


def select_scenarios(pool):
    empty = [r for r in pool.scenarios if not r['payload']['zones']]
    others = [r for r in pool.scenarios if r['payload']['zones']]
    if len(empty) != 7 or len(others) < 13:
        raise ValueError('Selection requires exactly seven empty and at least thirteen nonempty scenes.')
    chosen = np.random.default_rng(SELECTION_SEED).choice(len(others), 13, replace=False)
    selected = sorted(empty + [others[int(i)] for i in chosen], key=lambda r: r['sequence_index'])
    if len({r['scenario_id'] for r in selected}) != 20:
        raise ValueError('Duplicate selected scene IDs.')
    return selected


def make_env(data):
    return V2StaticNoFlyTrajectoryEnv(data['scenario'], data['rewards'],
        seed=data['checkpoint_payload']['formal_config']['seed'],
        uav_collision_radius=data['pool'].uav_collision_radius,
        cache_ellipsoid_segment_clearance=True, cpu_deduplication=True)


def observation_value(observation):
    return {key: getattr(observation, key) for key in
            ('ego_features', 'goal_features', 'zone_features', 'presence_mask')}


def probe_state(env, observation, info):
    """Audit only, never restored by assignment. Replay uses the original reset/step."""
    fields = ('state', 'initial_state', 'goal', 'steps', 'prev_action', 'recent_progress',
              'last_delta_z', 'best_goal_distance_so_far', 'last_segment_goal_distance',
              'last_goal_reached_by_segment')
    state = {key: getattr(env, key) for key in fields}
    state.update(observation=observation_value(observation), info=info,
        trajectory_digest=digest(env.trajectory), trajectory_length=len(env.trajectory),
        environment_rng=env.rng.bit_generator.state)
    previous = getattr(env, '_previous_reward_clearances', None)
    # Identity tuple is process-local; geometry is verified by reset of original
    # payload, and replay audit compares its numerical position/radius/clearances.
    state['previous_reward_clearances'] = None if previous is None else (previous[0], previous[2], previous[3])
    return serial(state)


def reset_replay(env, record, actions, expected_digest=None):
    observation, info = env.reset(options={'scenario': record['payload']})
    for action in actions:
        observation, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            raise ValueError('Anchor prefix terminated before probe.')
    state = probe_state(env, observation, info)
    if expected_digest is not None and digest(state) != expected_digest:
        raise ValueError('Probe replay observation/state/steps/reward-history/RNG mismatch.')
    return observation, state


def actor_action(data, observation):
    batch = collate_v2_observations([observation], device=data['device'])
    with torch.inference_mode():
        action = data['actor'](batch).detach().cpu().numpy()[0].copy()
    finite(action, 'actor action')
    limit = data['actor'].action_limit.detach().cpu().numpy()
    if action.dtype != np.float32 or action.shape != (2,) or np.any(np.abs(action) > limit):
        raise ValueError('Actor action violates float32 two-control contract.')
    return action


def fly(data, env, observation, *, first_action=None, record_probe_hashes=False, initial_info=None):
    finite(env.state, 'movement state')
    actions, rewards, states, hashes = [], [], [env.state.copy().tolist()], []
    info = initial_info
    while True:
        if record_probe_hashes:
            # info is supplied from the last actual reset/step, not reconstructed.
            hashes.append(digest(probe_state(env, observation, info)))
        action = np.asarray(first_action, dtype=np.float32).copy() if not actions and first_action is not None else actor_action(data, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        finite(env.state, 'movement state')
        actions.append(action.tolist())
        rewards.append(float(finite(reward, 'reward')))
        states.append(env.state.copy().tolist())
        if terminated or truncated:
            if info['outcome'] not in OUTCOMES:
                raise ValueError('Unknown terminal outcome.')
            return {'actions': actions, 'rewards': rewards, 'states': states,
                'pre_action_hashes': hashes, 'episode_length': len(actions),
                'episode_return': math.fsum(rewards), 'discounted_return': discounted_return(rewards, data['gamma']),
                'outcome': info['outcome'], 'terminated': bool(terminated), 'truncated': bool(truncated),
                'final_steps': env.steps, 'final_info': serial(info)}
        if env.steps >= data['scenario'].max_steps:
            raise RuntimeError('Environment did not terminate at its original time limit.')


def candidate_actions(action_a, action_b, anchor, limit):
    limit = np.asarray(limit, dtype=np.float32)
    if limit.shape != (2,) or not np.isfinite(limit).all() or np.any(limit <= 0):
        raise ValueError('Invalid action limits.')
    originals = [('actor_A', np.asarray(action_a, dtype=np.float32)),
                 ('actor_B', np.asarray(action_b, dtype=np.float32))]
    center = originals[0 if anchor == 'A' else 1][1] / limit
    for dim, direction in itertools.product(range(2), (-1, 1)):
        normalized = center.copy()
        normalized[dim] += direction * .25
        originals.append((f'anchor_dim{dim}_{direction:+d}', normalized * limit))
    candidates, proposals = [], []
    for source, raw in originals:
        finite(raw, 'candidate action')
        if raw.shape != (2,):
            raise ValueError('Candidate must have two controls.')
        action = np.clip(raw, -limit, limit).astype(np.float32)
        existing = next((i for i, item in enumerate(candidates)
                         if np.array_equal(action, np.asarray(item['action'], dtype=np.float32))), None)
        index = len(candidates) if existing is None else existing
        if existing is None:
            candidates.append({'candidate_index': index, 'action': action.tolist(),
                               'normalized_action': (action / limit).tolist()})
        proposals.append({'source': source, 'raw_action': raw.tolist(), 'candidate_index': index,
                          'clipped': not np.array_equal(raw, action), 'deduplicated': existing is not None})
    return candidates, proposals


def score(data, observation, action):
    batch = collate_v2_observations([observation], device=data['device'])
    tensor = torch.tensor([action], dtype=torch.float32, device=data['device'])
    with torch.inference_mode():
        q1, q2 = [float(critic(batch, tensor).item()) for critic in data['critics']]
    finite([q1, q2], 'Q scores')
    return {'Q1': q1, 'Q2': q2, 'min_Q': min(q1, q2), 'critic_disagreement': abs(q1 - q2)}


def near_tie(a, b):
    return abs(a - b) <= TIE_ATOL + TIE_RTOL * max(abs(a), abs(b))


def ranking(rows, own_index):
    own = rows[own_index]
    advantages = {r['candidate_index']: r['Q1'] - own['Q1'] for r in rows}
    counts = Counter()
    for a, b in itertools.combinations(rows, 2):
        qtie = near_tie(advantages[a['candidate_index']], advantages[b['candidate_index']])
        gtie = near_tie(a['G'], b['G'])
        kind = ('both_tied' if qtie and gtie else 'Q_tied' if qtie else 'G_tied' if gtie
                else 'concordant' if (a['Q1'] - b['Q1']) * (a['G'] - b['G']) > 0 else 'reversed')
        counts[kind] += 1
    best_q = max(rows, key=lambda r: r['Q1'])
    best_g = max(r['G'] for r in rows)
    q_ties = [r for r in rows if near_tie(advantages[r['candidate_index']], advantages[best_q['candidate_index']])]
    effective_pairs = counts['concordant'] + counts['reversed']
    return {'pair_counts': {k: counts[k] for k in ('concordant', 'reversed', 'Q_tied', 'G_tied', 'both_tied')},
        'effective_non_tied_pairs': effective_pairs,
        'ranking_information': 'insufficient' if effective_pairs == 0 else 'descriptive_pairs_available',
        'Q1_best_candidate': best_q['candidate_index'], 'Q1_best_tie_candidates': [r['candidate_index'] for r in q_ties],
        'Q1_best_outcome': best_q['outcome'],
        'G_best_outcomes': sorted({r['outcome'] for r in rows if near_tie(r['G'], best_g)}),
        'goal_available': any(r['outcome']=='goal' for r in rows),
        'G_regret_at_Q1_best': best_g - best_q['G'],
        'G_regret_range_over_Q1_ties': [min(best_g-r['G'] for r in q_ties), max(best_g-r['G'] for r in q_ties)],
        'own_actor_candidate': own_index, 'own_Q1_minus_G': own['Q1'] - own['G'],
        'relative_to_own': [{'candidate_index': r['candidate_index'],
            'Q1_delta': r['Q1'] - own['Q1'], 'G_delta': r['G'] - own['G'],
            'delta_prediction_error': (r['Q1']-own['Q1']) - (r['G']-own['G'])} for r in rows],
        'outcome_counts': dict(Counter(r['outcome'] for r in rows)),
        'mean_critic_disagreement': math.fsum(r['critic_disagreement'] for r in rows) / len(rows)}


def compact_summary(baselines, probes, branches):
    result = {}
    for model in ('A', 'B'):
        result[model] = {}
        for group in ('all', 'empty', 'nonempty'):
            belongs = lambda r: group == 'all' or r['empty_scene'] == (group == 'empty')
            base = [r for r in baselines if r['model'] == model and belongs(r)]
            selected = [r for r in probes if belongs(r)]
            statistics = [r['rankings'][model] for r in selected]
            pairs = Counter()
            for row in statistics:
                pairs.update(row['pair_counts'])
            effective_pairs = pairs['concordant'] + pairs['reversed']
            result[model][group] = {'scenes': len(base), 'baseline_outcomes': dict(Counter(r['outcome'] for r in base)),
                'baseline_subset_success_rate': sum(r['outcome']=='goal' for r in base)/len(base) if base else None,
                'pair_counts_descriptive_not_independent_samples': {k: pairs[k] for k in
                    ('concordant', 'reversed', 'Q_tied', 'G_tied', 'both_tied')},
                'effective_non_tied_pairs': effective_pairs,
                'probes_without_effective_pairs': sum(r['effective_non_tied_pairs'] == 0 for r in statistics),
                'ranking_information': 'insufficient' if effective_pairs == 0 else 'descriptive_pairs_available',
                'probes_with_reversed_pairs': sum(r['pair_counts']['reversed'] > 0 for r in statistics),
                'mean_G_regret_at_Q1_best': math.fsum(r['G_regret_at_Q1_best'] for r in statistics)/len(statistics) if statistics else None,
                'mean_own_Q1_minus_G': math.fsum(r['own_Q1_minus_G'] for r in statistics)/len(statistics) if statistics else None,
                'mean_critic_disagreement': math.fsum(r['mean_critic_disagreement'] for r in statistics)/len(statistics) if statistics else None,
                'branch_outcomes': dict(Counter(r['outcome'] for r in branches if r['model']==model and belongs(r)))}
    return result


def run(*, checkpoint_a, checkpoint_b, validation_pool, output_dir, device='cpu'):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing existing output directory: {output}')
    pool = load_v2_validation_pool(validation_pool, expected_level='easy', expected_count=100)
    if pool.content_digest != POOL_DIGEST:
        raise ValueError('Fixed easy pool content_digest mismatch.')
    selected = select_scenarios(pool)
    inputs = {key: load_models(path, validation_pool, device)
              for key, path in (('A', checkpoint_a), ('B', checkpoint_b))}
    input_differences = compare_inputs(inputs['A'], inputs['B'])
    def metadata(data):
        p = data['checkpoint_payload']
        return {'path': str(data['checkpoint_path']), 'sha256': data['checkpoint_hash'],
            'model': data['model'], **{k: p.get(k) for k in ('status', 'passed_validation', 'stage', 'formal_config',
                'bc_schedule', 'scenario_config', 'reward_config', 'uav_collision_radius', 'seed_manifest', 'validation_pool', 'initialization_source')},
            'checkpoint_format': p['format'],
            'stage_steps': p.get('training_result', p)['stage_steps'],
            'training_outcomes': p.get('training_result', {}).get('outcome_counts'),
            'saved_optimizer_lr_groups': {name: [g['lr'] for g in p['engine_checkpoint'][f'{name}_optimizer_state_dict']['param_groups']]
                                         for name in ('actor', 'critic')},
            'engine': {k: p['engine_checkpoint'][k] for k in ('architecture', 'algorithm_config', 'action_low', 'action_high', 'update_counts')}}
    manifest = {'diagnostic_only': True, 'inputs': {k: metadata(v) for k,v in inputs.items()},
        'input_configuration_differences': input_differences,
        'pool': {'path': str(Path(validation_pool).resolve()), 'sha256': inputs['A']['pool_hash'],
                 'content_digest': pool.content_digest, 'count': pool.scenario_count},
        'selection': {'seed': SELECTION_SEED, 'rule': 'all 7 empty + 13 nonempty without replacement; original sequence_index order',
            'scenarios': [{'scenario_id': r['scenario_id'], 'sequence_index': r['sequence_index'],
                'anchor': 'A' if i%2==0 else 'B',
                'shape_types': [z['shape']['shape_type'] for z in r['payload']['zones']]}
                for i,r in enumerate(selected)]},
        'probe_rule': 'pre-action floor(0.75*(T-1)) of alternating anchor natural trajectory; reset + actual float32 prefix replay',
        'candidate_rule': 'actor A/B plus anchor +/-0.25 normalized per dimension; clip then exact float32 dedup, <=6',
        'return_rule': 'G=sum(gamma**t * reward[t]); includes candidate reward, then own actor to true terminal; original remaining time',
        'primary_score': 'online Q1, matching actor RL objective',
        'ties': {'atol': TIE_ATOL, 'rtol': TIE_RTOL,
            'rule': 'abs(a-b)<=atol+rtol*max(abs(a),abs(b))',
            'Q_inputs': 'delta_Q_i=Q1_i-Q1_own_actor; same advantages for pairs and Q1-best tie set; invariant to common Q shifts',
            'G_inputs': 'raw discounted returns, unchanged',
            'effective_pairs': 'neither Q-advantage nor G tied; concordant + reversed',
            'information': 'zero effective pairs => insufficient, never evidence of reliability; available pairs are descriptive only',
            'raw_values': 'raw Q/G/Q-G, original Q1 argmax and its G regret preserved'},
        'execution': {'device': str(inputs['A']['device']), 'path': 'eager eval inference_mode; SNN forward resets state',
            'cache_ellipsoid_segment_clearance': True, 'cpu_deduplication': True,
            'software': _software_versions(), 'torch_threads': torch.get_num_threads(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
        'limits': ['Terminal-model scoring only, no historical causal ordering.',
            'Enriched 20-scene late-probe subset, not formal 100-scene validation or population success rate.',
            'G is own noise-free actor continuation; target smoothing/lag and unobserved history/time limit affect Q-G.',
            'BC, terminal_geo and spike regularization also enter actor objective; actor need not maximize these candidates.',
            'Discounted reward ranking and task success can disagree; no independent-action significance claims.']}
    repo = Path(__file__).resolve().parents[3]
    manifest['source'] = {key: subprocess.run(['git', *cmd], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
                          for key,cmd in (('HEAD', ['rev-parse', 'HEAD']), ('status', ['status', '--short']))}
    manifest['source']['diagnostic_script_sha256'] = file_sha256(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    # Persist fixed selection and rules before any network forward or trajectory.
    _write_json(output/'manifest.json', manifest)
    (output/'README.txt').write_text(
        'Diagnostic only. execution_completed is NOT a critic-reliability or formal-training verdict.\n'
        'Read manifest selection/config first, baseline_episodes for task results, probes rankings for\n'
        'Q1-vs-G pair counts and regret, and branches for full own-actor discounted continuations.\n'
        'Q/G near ties are separate; candidate actions in one scene are not independent samples.\n'
        'Q ties use Q1-Q1_own_actor for both pairs and best tie sets; G ties use raw G.\n'
        'Effective pairs are concordant+reversed; zero effective pairs means information insufficient, not reliable.\n'
        'A goal may have lower discounted reward than failure. Inspect outcomes alongside G.\n'
        'Only terminal weights and deliberately late probes are inspected: no historical causality.\n'
        'Target smoothing/lag and hidden history/time limit affect Q-G; differences are not code-bug proof.\n'
        'Actor objective also includes BC, terminal geometry and spike regularization.\n'
        'The enriched 20-scene subset is not formal 100-scene validation.\n', encoding='utf-8')
    baselines, probes, branches = [], [], []
    summary = {'diagnostic_only': True, 'execution_completed': False, 'formal_validation': False}
    try:
        with (output/'baseline_episodes.jsonl').open('x', encoding='utf-8') as base_file, \
                (output/'probes.jsonl').open('x', encoding='utf-8') as probe_file, \
                (output/'branches.jsonl').open('x', encoding='utf-8') as branch_file:
            def emit(stream, row):
                stream.write(json.dumps(serial(row), allow_nan=False) + '\n')
                stream.flush()
            for index, record in enumerate(selected):
                identity = {'scenario_id': record['scenario_id'], 'sequence_index': record['sequence_index'],
                            'empty_scene': not record['payload']['zones']}
                natural = {}
                for label, data in inputs.items():
                    env = make_env(data)
                    obs, info = env.reset(options={'scenario': record['payload']})
                    natural[label] = fly(data, env, obs, record_probe_hashes=True, initial_info=info)
                    row = dict(identity, model=label, **natural[label])
                    baselines.append(row)
                    emit(base_file, row)
                anchor = 'A' if index%2==0 else 'B'
                trajectory = natural[anchor]
                t = math.floor(.75 * (trajectory['episode_length'] - 1))
                prefix = trajectory['actions'][:t]
                expected = trajectory['pre_action_hashes'][t]
                env = make_env(inputs[anchor])
                obs, state = reset_replay(env, record, prefix, expected)
                actor_actions = {k: actor_action(v, obs) for k,v in inputs.items()}
                candidates, proposals = candidate_actions(actor_actions['A'], actor_actions['B'], anchor,
                    inputs[anchor]['actor'].action_limit.detach().cpu().numpy())
                probe = dict(identity, anchor=anchor, anchor_length=trajectory['episode_length'],
                    pre_action_index=t, float32_action_prefix=prefix, state=state, replay_digest=expected,
                    candidates=candidates, proposals=proposals, rankings={})
                for label, data in inputs.items():
                    rows = []
                    own_index = next(p['candidate_index'] for p in proposals if p['source'] == f'actor_{label}')
                    for candidate in candidates:
                        env = make_env(data)
                        observation, _ = reset_replay(env, record, prefix, expected)
                        scores = score(data, observation, candidate['action'])
                        reuse = label == anchor and np.array_equal(np.asarray(candidate['action'], np.float32), np.asarray(trajectory['actions'][t], np.float32))
                        if reuse:
                            rewards = trajectory['rewards'][t:]
                            continuation = {k: trajectory[k] for k in ('outcome','terminated','truncated','final_steps','final_info')}
                            continuation.update(actions=trajectory['actions'][t:], rewards=rewards,
                                states=trajectory['states'][t:], episode_length=len(rewards),
                                episode_return=math.fsum(rewards), discounted_return=discounted_return(rewards, data['gamma']))
                        else:
                            continuation = fly(data, env, observation, first_action=candidate['action'])
                        row = dict(identity, model=label, **candidate, **scores,
                            G=continuation['discounted_return'], Q1_minus_G=scores['Q1']-continuation['discounted_return'],
                            finite=True, baseline_suffix_reused=reuse, outcome=continuation['outcome'],
                            continuation=continuation)
                        rows.append(row)
                        branches.append(row)
                        emit(branch_file, row)
                    probe['rankings'][label] = ranking(rows, own_index)
                probes.append(probe)
                emit(probe_file, probe)
        for data in inputs.values():
            if file_sha256(data['checkpoint_path']) != data['checkpoint_hash'] or file_sha256(data['pool_path']) != data['pool_hash']:
                raise RuntimeError('Input artifacts changed during diagnostic.')
        expected_branches = sum(2*len(p['candidates']) for p in probes)
        if len(baselines)!=2*len(selected) or len(probes)!=len(selected) or len(branches)!=expected_branches or expected_branches>240:
            raise RuntimeError('Incomplete diagnostic output.')
        summary.update(execution_completed=True, baseline_episodes=len(baselines), probes=len(probes),
            branches=len(branches), expected_branches=expected_branches,
            statistics=compact_summary(baselines, probes, branches),
            reliability_verdict='Descriptive rankings only; completion is not critic reliability or formal training acceptance.')
    except Exception as exc:
        summary['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        _write_json(output/'summary.json', summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-a', type=Path, required=True)
    parser.add_argument('--checkpoint-b', type=Path, required=True)
    parser.add_argument('--validation-pool', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run(**vars(args))
    print(json.dumps({'diagnostic_only': True, 'execution_completed': result['execution_completed'],
                      'summary': str(args.output_dir/'summary.json')}))


if __name__ == '__main__':
    main()
