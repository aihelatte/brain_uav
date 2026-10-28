"""Fixed-data B-only critic attribution/calibration; never a formal training run."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from brain_uav.observations import collate_v2_observations
from brain_uav.scripts import diagnose_v2_critic as base
from brain_uav.scripts.diagnose_v2_policy import _write_json, file_sha256, _software_versions
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from brain_uav.trainers.v2_validation import load_v2_validation_pool

CHECKPOINT_SHA256 = 'c5cb20f60c8181fa03bc1fcd31901cfeae218b1eecd414b186abd15f5710f36d'
SPLIT_SEED = 20261028
EVALUATIONS = (0, 100, 500, 1000)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Non-finite tensor in diagnostic.')
    elif isinstance(value, dict):
        for child in value.values(): finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value: finite_tree(child)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError('Non-finite scalar in diagnostic.')


def state_hash(value):
    """Exact tensor bytes, including Adam moments; independent of device/address."""
    from hashlib import sha256
    h = sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous()
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str): h.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            for child in x: visit(child)
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def returns(rewards, done, gamma):
    if len(rewards) != len(done) or not len(done) or not done[-1] or any(done[:-1]):
        raise ValueError('Expected one complete natural episode with one final done.')
    if not np.isfinite(gamma) or not 0 <= gamma <= 1: raise ValueError('Invalid gamma.')
    values, remaining = [], 0.
    for reward, terminal in zip(reversed(rewards), reversed(done)):
        remaining = float(reward) + gamma * (0. if terminal else remaining)
        values.append(remaining)
    finite_tree(values)
    return list(reversed(values))


def split_scenes(records):
    ids = [r['scenario_id'] for r in records]
    if len(ids) != 20 or len(set(ids)) != 20: raise ValueError('Require 20 unique B scenes.')
    rng = np.random.default_rng(SPLIT_SEED)
    holdout = []
    for empty in (True, False):
        candidates = sorted(r['scenario_id'] for r in records if r['empty_scene'] == empty)
        if len(candidates) < 2: raise ValueError('Insufficient empty/nonempty scenes.')
        holdout.extend(str(x) for x in rng.choice(candidates, 2, replace=False))
    return dict(seed=SPLIT_SEED, train=sorted(set(ids)-set(holdout)), holdout=sorted(holdout),
                rule='2 empty + 2 nonempty; IDs only, independent of outcome/Q/G')


def replay_episode(data, scene, record):
    """One reset and one linear replay; no position assignment or policy flight."""
    env = base.make_env(data)
    observation, info = env.reset(options={'scenario': scene['payload']})
    actions = record['actions']; n = record['episode_length']
    if not n or not (len(actions) == len(record['rewards']) == len(record['pre_action_hashes']) == n
                     and len(record['states']) == n+1):
        raise ValueError('Incomplete baseline episode.')
    if base.serial(env.state) != record['states'][0]: raise ValueError('Initial state mismatch.')
    rows = []
    for i, action in enumerate(actions):
        if base.digest(base.probe_state(env, observation, info)) != record['pre_action_hashes'][i]:
            raise ValueError(f'Replay hash mismatch at {record["scenario_id"]}:{i}.')
        action = np.asarray(action, dtype=np.float32)
        base.finite(action, 'saved action')
        if action.shape != (2,): raise ValueError('Invalid action shape.')
        previous = deepcopy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        base.finite(reward, 'replayed reward')
        if float(reward) != record['rewards'][i] or base.serial(env.state) != record['states'][i+1]:
            raise ValueError('Replay reward/state mismatch.')
        done = bool(terminated or truncated)
        if done != (i == n-1): raise ValueError('Replay termination step mismatch.')
        rows.append(dict(scenario_id=record['scenario_id'], empty_scene=record['empty_scene'],
            step=i, observation=previous, next_observation=deepcopy(observation),
            action=action.copy(), reward=float(reward), done=done))
    actual = dict(outcome=info['outcome'], terminated=bool(terminated), truncated=bool(truncated),
                  final_steps=env.steps, final_info=base.serial(info))
    if any(actual[k] != record[k] for k in actual): raise ValueError('Replay final state/info mismatch.')
    gs = returns([r['reward'] for r in rows], [r['done'] for r in rows], data['gamma'])
    if gs[0] != record['discounted_return']:
        raise ValueError('Replay return mismatch.')
    import math
    if math.fsum(r['reward'] for r in rows) != record['episode_return']: raise ValueError('Episode return mismatch.')
    for row, g in zip(rows, gs): row['G'] = g
    return rows


def load_targets(data):
    engine = data['checkpoint_payload']['engine_checkpoint']
    result = []
    for template, name in zip([data['actor'], *data['critics']],
            ('actor_target_state_dict', 'critic1_target_state_dict', 'critic2_target_state_dict')):
        target = deepcopy(template)
        state = engine[name]  # Required: never substitute online weights.
        V2TD3UpdateEngine._validate_state_dict(target, state, name=name)
        finite_tree(state)
        for key, expected in target.named_buffers():
            if not torch.equal(state[key].cpu(), expected.detach().cpu()):
                raise ValueError(f'{name} fixed buffer mismatch: {key}.')
        target.load_state_dict(state, strict=True)
        result.append(target.eval().requires_grad_(False))
    return result


def tensor_batch(rows, indices, device, *, next_state=False):
    key = 'next_observation' if next_state else 'observation'
    obs = collate_v2_observations([rows[int(i)][key] for i in indices], device=device)
    obs.validate_finite()
    actions = torch.as_tensor(np.stack([rows[int(i)]['action'] for i in indices]), device=device)
    return obs, actions


def td_target(reward, done, gamma, q1, q2):
    result = reward + (1-done)*gamma*torch.minimum(q1, q2)
    finite_tree(result)
    return result


def compute_labels(data, targets, rows):
    actor_t, c1_t, c2_t = targets
    cfg = data['checkpoint_payload']['engine_checkpoint']['algorithm_config']
    if cfg['policy_noise'] != .006 or cfg['noise_clip'] != .012:
        raise ValueError('Expected saved terminal policy_noise=.006, noise_clip=.012.')
    # Independent CPU RNG; never consume training/global RNG for target noise.
    noise = torch.randn((len(rows), 2), generator=torch.Generator().manual_seed(SPLIT_SEED))
    noise = (noise * cfg['policy_noise']).clamp(-cfg['noise_clip'], cfg['noise_clip'])
    noise_hash = state_hash(noise)
    e = data['checkpoint_payload']['engine_checkpoint']
    low = torch.tensor(e['action_low'], device=data['device'])
    high = torch.tensor(e['action_high'], device=data['device'])
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            indices = list(range(start, min(start+64, len(rows))))
            obs, actions = tensor_batch(rows, indices, data['device'])
            nxt, _ = tensor_batch(rows, indices, data['device'], next_state=True)
            q = [c(obs, actions).reshape(-1) for c in data['critics']]
            online_action = data['actor'](nxt)
            target_action = actor_t(nxt)
            noisy_action = torch.maximum(low, torch.minimum(high, target_action + noise[indices].to(data['device'])))
            online = [c(nxt, online_action).reshape(-1) for c in data['critics']]
            lag = [c(nxt, online_action).reshape(-1) for c in (c1_t, c2_t)]
            actor_lag = [c(nxt, target_action).reshape(-1) for c in (c1_t, c2_t)]
            smoothed = [c(nxt, noisy_action).reshape(-1) for c in (c1_t, c2_t)]
            finite_tree([q, online_action, target_action, noisy_action, online, lag, actor_lag, smoothed])
            reward = torch.tensor([rows[i]['reward'] for i in indices], device=data['device'])
            done = torch.tensor([rows[i]['done'] for i in indices], dtype=torch.float32, device=data['device'])
            ys = [td_target(reward, done, data['gamma'], *pair) for pair in (online, lag, actor_lag, smoothed)]
            weight = (1-done)*data['gamma']
            values = dict(Q1=q[0], Q2=q[1], target_Q1=smoothed[0], target_Q2=smoothed[1], y=ys[3],
                online_noiseless_y=ys[0], target_critic_shift=ys[1]-ys[0],
                target_actor_shift=ys[2]-ys[1], smoothing_shift=ys[3]-ys[2],
                min_vs_mean_shift=weight*(torch.minimum(*smoothed)-(smoothed[0]+smoothed[1])/2))
            arrays = {k: v.cpu().tolist() for k,v in values.items()}
            for j,i in enumerate(indices):
                row = rows[i]
                row['smoothing_noise'] = noise[i].tolist()
                row.update({k: float(v[j]) for k,v in arrays.items()})
                for k in ('Q1','Q2'):
                    row[k+'-G'] = row[k]-row['G']; row[k+'-y'] = row[k]-row['y']
                row['y-G'] = row['y']-row['G']
    return noise_hash


def make_fit(critics, saved_optimizer):
    copies = [deepcopy(c).eval().requires_grad_(True) for c in critics]
    parameters = [p for c in copies for p in c.parameters()]
    state = deepcopy(saved_optimizer)
    finite_tree(state)
    if len(state['param_groups']) != 1 or len(state['param_groups'][0]['params']) != len(parameters):
        raise ValueError('Expected one joint Adam over critic1 then critic2 parameters.')
    # Eager diagnostic backend; preserve Adam hyperparameters, moments and steps.
    for group in state['param_groups']: group.update(fused=False, capturable=False)
    optimizer = torch.optim.Adam(parameters)
    optimizer.load_state_dict(state)
    if state_hash(optimizer.state_dict()['state']) != state_hash(state['state']):
        raise RuntimeError('Restored Adam moments/steps differ from checkpoint.')
    return copies, optimizer, parameters


def statistics(rows, predictions, split):
    output = []
    for partition in ('train','holdout'):
        for geometry in ('all','empty','nonempty'):
            for terminal in ('all','terminal','nonterminal'):
                ids = [i for i,r in enumerate(rows) if r['scenario_id'] in split[partition]
                    and (geometry == 'all' or r['empty_scene'] == (geometry == 'empty'))
                    and (terminal == 'all' or r['done'] == (terminal == 'terminal'))]
                entry = dict(partition=partition, geometry=geometry, terminal=terminal,
                             transitions=len(ids), scenes=len({rows[i]['scenario_id'] for i in ids}))
                if not ids: entry['information'] = 'insufficient'; output.append(entry); continue
                for c in range(2):
                    def metrics(subset):
                        error = np.array([predictions[i][c]-rows[i]['G'] for i in subset], dtype=np.float64)
                        td = np.array([predictions[i][c]-rows[i]['y'] for i in subset], dtype=np.float64)
                        return dict(Q_minus_G=float(error.mean()), MAE=float(abs(error).mean()),
                            MSE=float((error**2).mean()), Q_minus_y=float(td.mean()))
                    weighted = metrics(ids)
                    per_scene = [metrics([i for i in ids if rows[i]['scenario_id'] == sid])
                                 for sid in sorted({rows[i]['scenario_id'] for i in ids})]
                    equal = {k: float(np.mean([s[k] for s in per_scene])) for k in weighted}
                    for summary in (weighted, equal): summary['RMSE'] = float(np.sqrt(summary.pop('MSE')))
                    entry[f'critic{c+1}'] = dict(transition_weighted=weighted, scene_equal=equal)
                for key in ('y-G','min_vs_mean_shift','target_critic_shift','target_actor_shift','smoothing_shift'):
                    if key not in rows[ids[0]]: continue
                    scene_means = [np.mean([rows[i][key] for i in ids if rows[i]['scenario_id'] == sid])
                                   for sid in sorted({rows[i]['scenario_id'] for i in ids})]
                    entry[key] = dict(transition_weighted=float(np.mean([rows[i][key] for i in ids])),
                                      scene_equal=float(np.mean(scene_means)))
                output.append(entry)
    finite_tree(output)
    return output


def evaluate(critics, rows, device, split):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            obs, action = tensor_batch(rows, list(range(start,min(start+64,len(rows)))), device)
            q = torch.stack([c(obs,action).reshape(-1) for c in critics], dim=1)
            finite_tree(q); predictions.extend(q.cpu().tolist())
    return statistics(rows, predictions, split)


def fit(data, targets, rows, split, *, budget=1000, evaluations=EVALUATIONS):
    """Budget overrides are for tiny unit fixtures only; CLI is fixed at 1000."""
    frozen = [data['actor'], *targets]
    before = state_hash([m.state_dict() for m in frozen])
    fits = {name: make_fit(data['critics'], data['checkpoint_payload']['engine_checkpoint']['critic_optimizer_state_dict'])
            for name in ('fixed_td','mc_return')}
    initial = {name: dict(weights=state_hash([c.state_dict() for c in cs]), adam=state_hash(opt.state_dict()))
               for name,(cs,opt,_) in fits.items()}
    if initial['fixed_td'] != initial['mc_return']: raise RuntimeError('Initial fit state mismatch.')
    if initial['fixed_td']['weights'] != state_hash([c.state_dict() for c in data['critics']]):
        raise RuntimeError('Fit copies differ from checkpoint online critics.')
    train = [i for i,r in enumerate(rows) if r['scenario_id'] in split['train']]
    if not train: raise ValueError('Empty training set.')
    batches = np.random.default_rng(SPLIT_SEED).choice(train, size=(budget,64), replace=True)
    batch_hash = base.digest(batches)
    curve, gradients = [], {name: [] for name in fits}
    cfg = data['checkpoint_payload']['engine_checkpoint']['algorithm_config']
    clip = cfg['critic_grad_clip_norm']
    for step in range(budget+1):
        if step in evaluations:
            for name,(cs,_,_) in fits.items():
                norms = gradients[name]
                curve.append(dict(updates=step, fit=name, metrics=evaluate(cs,rows,data['device'],split),
                    gradient=dict(scope='joint critic1+critic2 global norm; training batches only, not attributable to holdout subgroups',
                        count=len(norms), pre_clip_norm_mean=float(np.mean(norms)) if norms else None,
                        clipping_fraction=sum(x>clip for x in norms)/len(norms) if norms and clip is not None else None)))
        if step == budget: break
        indices = batches[step]
        obs, action = tensor_batch(rows, indices, data['device'])
        for name,(cs,opt,parameters) in fits.items():
            label = 'y' if name == 'fixed_td' else 'G'
            target = torch.tensor([rows[int(i)][label] for i in indices], dtype=torch.float32, device=data['device'])
            opt.zero_grad(set_to_none=True)
            prediction = [c(obs,action).reshape(-1) for c in cs]
            loss = sum(F.mse_loss(q,target) for q in prediction)
            finite_tree(loss); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip if clip is not None else float('inf'), error_if_nonfinite=True)
            finite_tree(norm); gradients[name].append(float(norm.detach().cpu()))
            opt.step(); finite_tree([c.state_dict() for c in cs]); finite_tree(opt.state_dict())
    unchanged = before == state_hash([m.state_dict() for m in frozen])
    if not unchanged: raise RuntimeError('Frozen actor/target state changed.')
    return dict(initial=initial, batch_indices_sha256=batch_hash, batch_sampling='uniform with replacement; same precomputed indices for both fits',
                frozen_unchanged=unchanged, curve=curve, gradient_norms=gradients,
                final_state_hashes={name:dict(weights=state_hash([c.state_dict() for c in cs]),
                    adam=state_hash(opt.state_dict())) for name,(cs,opt,_) in fits.items()})


def run(args):
    output = Path(args.output_dir)
    if output.exists(): raise FileExistsError('Refusing existing output directory.')
    output.mkdir(parents=True)
    summary = dict(diagnostic_only=True, execution_completed=False, formal_passed=False)
    try:
        checkpoint = Path(args.checkpoint); history = Path(args.history_dir)
        manifest_path = history/'manifest.json'; episodes_path = history/'baseline_episodes.jsonl'
        hashes = {k:file_sha256(p) for k,p in dict(checkpoint=checkpoint, manifest=manifest_path, episodes=episodes_path).items()}
        if hashes['checkpoint'] != CHECKPOINT_SHA256: raise ValueError('B checkpoint SHA256 mismatch.')
        prior = json.loads(manifest_path.read_text(encoding='utf-8'))
        if prior['inputs']['B']['sha256'] != CHECKPOINT_SHA256: raise ValueError('History checkpoint mismatch.')
        records = [json.loads(line) for line in episodes_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        records = [r for r in records if r['model'] == 'B']
        split = split_scenes(records)
        pool = load_v2_validation_pool(args.validation_pool, expected_level='easy', expected_count=100)
        if pool.content_digest != base.POOL_DIGEST or prior['pool']['content_digest'] != pool.content_digest:
            raise ValueError('Fixed pool digest mismatch.')
        selected = {r['scenario_id']:r for r in pool.scenarios}
        if {r['scenario_id'] for r in records} != {r['scenario_id'] for r in prior['selection']['scenarios']}:
            raise ValueError('History selection mismatch.')
        for r in records:
            scene = selected[r['scenario_id']]
            if r['empty_scene'] != (not scene['payload']['zones']) or r['sequence_index'] != scene['sequence_index']:
                raise ValueError('History scene identity mismatch.')
        data = base.load_models(checkpoint, args.validation_pool, args.device)
        p = data['checkpoint_payload']; engine = p['engine_checkpoint']
        if (data['model'] != 'snn' or p['stage'] != 'easy' or p['formal_config']['seed'] != 23
                or p['training_result']['stage_steps'] != 500000 or p['formal_config']['critic_lr'] != .001
                or any(g['lr'] != .001 for g in engine['critic_optimizer_state_dict']['param_groups'])):
            raise ValueError('Not the specified B terminal experiment.')
        cfg = engine['algorithm_config']
        if cfg['batch_size'] != 64 or cfg['policy_noise'] != .006 or cfg['noise_clip'] != .012:
            raise ValueError('B saved batch/noise configuration mismatch.')
        targets = load_targets(data)
        repo = Path(__file__).resolve().parents[3]
        source = {key:subprocess.run(['git',*cmd],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
                  for key,cmd in (('HEAD',['rev-parse','HEAD']),('status',['status','--short']))}
        manifest = dict(diagnostic_only=True, model='B', input_hashes=hashes, checkpoint=str(checkpoint),
            history=str(history), pool_digest=pool.content_digest, pool_sha256=file_sha256(Path(args.validation_pool)),
            split=split, source=source, script_sha256=file_sha256(Path(__file__)), software=_software_versions(),
            algorithm=engine['algorithm_config'], optimizer_groups=engine['critic_optimizer_state_dict']['param_groups'],
            critic_parameter_order=[f'critic{i}.{name}' for i,c in enumerate(data['critics'],1) for name,_ in c.named_parameters()],
            execution=dict(device=str(data['device']),path='independent eager; no trainer/GPUCompute/soft updates',
                adam_backend='fused=False,capturable=False; saved hyperparameters/moments/steps retained',
                batch_size=64, updates=1000, evaluations=list(EVALUATIONS), split_noise_batch_seed=SPLIT_SEED,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
            definitions=dict(G='remaining natural trajectory discounted return, includes current reward; stops on terminated OR truncated',
                noise='one independent frozen Gaussian sample per transition, not noise expectation',
                attribution='adjacent TD labels: online critics+online actor -> target critics -> target actor -> smoothing; min-vs-mean is a separate overlapping contrast',
                RMSE_scene_equal='sqrt(mean of per-scene MSE); all other scene-equal metrics mean per-scene statistics',
                limits='G follows saved noiseless online actor; y uses target actor, noise and min. y-G is not proof of an implementation error. Fixed offline fitting is not formal TD3; success does not resolve training, failure at 1000 does not prove capacity insufficient.'))
        _write_json(output/'manifest.json',manifest)  # Split persisted before replay/labels/fitting.
        rows = [row for r in records for row in replay_episode(data,selected[r['scenario_id']],r)]
        manifest['noise_sha256'] = compute_labels(data,targets,rows)
        # This run owns the manifest created above; shared writers remain create-only.
        (output/'manifest.json').write_text(
            json.dumps(manifest,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        with (output/'bias_decomposition.jsonl').open('w',encoding='utf-8') as f:
            for row in rows:
                detail = {k:v for k,v in row.items() if k not in ('observation','next_observation','action')}
                finite_tree(detail); f.write(json.dumps(detail,allow_nan=False)+'\n')
        result = fit(data,targets,rows,split)
        _write_json(output/'fit_curves.json',result)
        summary.update(execution_completed=True, transitions=len(rows), split=split,
            frozen_unchanged=result['frozen_unchanged'], initial=result['initial'],
            final=[r for r in result['curve'] if r['updates']==1000])
        for name,path in dict(checkpoint=checkpoint,manifest=manifest_path,episodes=episodes_path).items():
            if file_sha256(path) != hashes[name]: raise RuntimeError('Input changed during diagnostic.')
    except BaseException as exc:
        summary.update(execution_completed=False,error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally: _write_json(output/'summary.json',summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--history-dir',type=Path,required=True)
    parser.add_argument('--validation-pool',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    return parser


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == '__main__': main()
