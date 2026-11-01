"""Fixed-actor, scene-isolated critic intervention. Plan-only unless --launch."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from brain_uav.scripts import diagnose_v2_critic_calibration as cal
from brain_uav.scripts.diagnose_v2_policy import _write_json, file_sha256
from brain_uav.envs.v2_scenario_generator import V2ScenarioGenerator, V2ScenarioGeneratorConfig

SEED = 20261101
SOURCE = {
    'ann': ('c48c94a8fa1d7a2ca2a833cfcf9b103e61655cf089c140237d1ded886449e8af', 2e-4),
    'snn': ('c5cb20f60c8181fa03bc1fcd31901cfeae218b1eecd414b186abd15f5710f36d', 1e-3),
}
ARMS = ('min', 'mean', 'wd3', 'acc')


def protocol(updates=10000):
    if type(updates) is not int or not 1 <= updates <= 10000:
        raise ValueError('Require 1..10000 predeclared critic updates.')
    return dict(diagnostic_only=True, formal_passed=False, version=2, updates=updates,
        batch_size=64, split_seed=SEED, train_scenes=30, calibration_scenes=10, score_scenes=20,
        calibration_composition=dict(empty=1, nonempty=9),
        wd3_min_weight=.5, acc_initial_beta=.5, acc_step_size=.05, acc_interval=100,
        acc_bounds=[0., 1.], acc_normalizer='max(1, p_empty*mean_empty(abs(G))+(1-p_empty)*mean_nonempty(abs(G)))',
        acc_signal='p_empty*mean_empty(G-Qbar)+(1-p_empty)*mean_nonempty(G-Qbar); scene equal within each stratum',
        target_policy='frozen online actor; zero smoothing noise; saved target actor frozen/unused',
        loss='MSE(Q1,y1)+MSE(Q2,y2); G never enters critic loss',
        target_update='saved tau and policy_delay, phase offset by saved last_total_steps',
        evaluations=sorted({x for x in (0, 100, 500, 1000, 2000, 5000, 10000, updates) if x <= updates}),
        selection='predeclared final budget, no best-step or score-based selection',
        paper_acc='https://arxiv.org/abs/2111.12673', paper_wd3='https://arxiv.org/abs/2006.12622')


def code_identity():
    root=Path(__file__).resolve().parents[3]
    return dict(sha256={p.relative_to(root).as_posix():file_sha256(p) for p in sorted((root/'src').rglob('*.py'))},
        head=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip(),
        status=subprocess.run(['git','status','--short'],cwd=root,capture_output=True,text=True,check=True).stdout.strip())


def scene_key(payload):
    """Physical scene identity ignores IDs/provenance metadata, not geometry."""
    def physical(value):
        if isinstance(value,dict):
            return {k:physical(v) for k,v in value.items() if k not in ('schema_version','metadata')}
        if isinstance(value,list): return [physical(v) for v in value]
        if type(value) in (int,float): return float(value)
        return value
    zones = [dict(shape=z['shape'], safety_margin=z.get('safety_margin', 0.)) for z in payload['zones']]
    zones = physical(zones)
    zones.sort(key=lambda z: json.dumps(z, sort_keys=True))
    return cal.base.digest(dict(state=physical(payload['state']), goal=physical(payload['goal']), zones=zones))


def split_scenes(records, fixed_pool_ids, fixed_pool_keys):
    if len(records) != 60 or len({r['scenario_id'] for r in records}) != 60:
        raise ValueError('Require exactly 60 distinct scenes: 30 train, 10 calibration, 20 score.')
    keys = [scene_key(r['payload']) for r in records]
    if len(set(keys)) != 60: raise ValueError('Physical scene duplicated across data partitions.')
    if any(r.get('partition') not in ('train','calibration','score') for r in records):
        raise ValueError('Require predeclared scene partitions, not a transition split.')
    split = {k:sorted(r['scenario_id'] for r in records if r['partition']==k)
             for k in ('train','calibration','score')}
    if [len(split[k]) for k in split] != [30,10,20]: raise ValueError('Wrong 30/10/20 scene split sizes.')
    if sum(not r['payload']['zones'] for r in records if r['partition']=='calibration') != 1:
        raise ValueError('Calibration requires one empty and nine nonempty scenes.')
    if any((r['scenario_id'] in fixed_pool_ids or scene_key(r['payload']) in fixed_pool_keys)
           and r['partition']!='score' for r in records):
        raise ValueError('Fixed validation pool leaked into training/calibration.')
    return split


def target_pair(q1, q2, arm, beta):
    """ACC uses critic-specific labels (paper TD3 variant), not WD3's shared label."""
    low = torch.minimum(q1, q2)
    if arm == 'min': return low, low
    if arm == 'mean':
        mean = (q1 + q2) / 2
        return mean, mean
    if arm == 'wd3':
        value = .5 * low + .5 * ((q1 + q2) / 2)
        return value, value
    if arm == 'acc': return beta*q1 + (1-beta)*low, beta*q2 + (1-beta)*low
    raise ValueError('Unknown intervention arm.')


def labels(data, targets, rows, indices, arm, beta):
    # Intentionally reads only reward/done and next state; never G.
    obs, _ = cal.tensor_batch(rows, indices, data['device'], next_state=True)
    with torch.no_grad():
        action = data['actor'](obs)
        q1, q2 = [c(obs, action).reshape(-1) for c in targets]
        values = target_pair(q1, q2, arm, beta)
        reward = torch.tensor([rows[int(i)]['reward'] for i in indices], device=data['device'], dtype=torch.float32)
        done = torch.tensor([rows[int(i)]['done'] for i in indices], device=data['device'], dtype=torch.float32)
        result = [reward + data['gamma']*(1-done)*q for q in values]
        cal.finite_tree([action, *values, *result])
        return result


def predict(data, critics, rows, indices):
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            obs, actions = cal.tensor_batch(rows, indices[start:start+64], data['device'])
            q = torch.stack([c(obs, actions).reshape(-1) for c in critics], dim=1)
            cal.finite_tree(q); values.extend(q.cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def acc_control(beta, predictions, rows, ids, p_empty):
    """Scene equal within strata; strata weighted by source generator probability."""
    if not np.isfinite(p_empty) or not 0 < p_empty < 1: raise ValueError('Invalid source p_empty.')
    scenes = sorted({rows[i]['scenario_id'] for i in ids})
    if not scenes: raise ValueError('ACC calibration set is empty.')
    residuals, scales, empty_flags = [], [], []
    for sid in scenes:
        selected = [j for j, i in enumerate(ids) if rows[i]['scenario_id'] == sid]
        g = np.asarray([rows[ids[j]]['G'] for j in selected], dtype=np.float64)
        residuals.append(float((g - predictions[selected].mean(axis=1)).mean()))
        scales.append(float(np.abs(g).mean()))
        flags={rows[ids[j]]['empty_scene'] for j in selected}
        if len(flags)!=1: raise ValueError('Inconsistent scene stratum.')
        empty_flags.append(flags.pop())
    def weighted(values):
        means=[]
        for empty in (True,False):
            group=[v for v,flag in zip(values,empty_flags) if flag==empty]
            if not group: raise ValueError('Missing ACC calibration stratum.')
            means.append(float(np.mean(group)))
        return p_empty*means[0]+(1-p_empty)*means[1]
    signal, scale = weighted(residuals), max(1.,weighted(scales))
    if not np.isfinite([beta, signal, scale]).all(): raise ValueError('Non-finite ACC control input.')
    normalized = float(np.clip(signal / scale, -1., 1.))
    following = float(np.clip(beta + .05 * normalized, 0., 1.))
    return following, dict(beta_before=beta, beta_after=following, residual=signal,
                          normalizer=scale, normalized_signal=normalized, p_empty=p_empty)


def metrics(rows, prediction, td_labels, ids):
    if not ids: return dict(information='insufficient', transitions=0, scenes=0)
    scenes = sorted({rows[i]['scenario_id'] for i in ids})
    result = dict(transitions=len(ids), scenes=len(scenes))
    for c in range(2):
        def one(subset):
            error = prediction[subset, c] - np.asarray([rows[i]['G'] for i in subset])
            td = prediction[subset, c] - td_labels[subset, c]
            return dict(bias=float(error.mean()), MAE=float(abs(error).mean()), MSE=float((error**2).mean()),
                TD_bias=float(td.mean()), TD_MAE=float(abs(td).mean()), TD_MSE=float((td**2).mean()),
                overestimate_fraction=float((error > 0).mean()), positive_error_mean=float(np.maximum(error,0).mean()))
        weighted = one(ids)
        by_scene = [one([i for i in ids if rows[i]['scenario_id']==sid]) for sid in scenes]
        equal = {k:float(np.mean([s[k] for s in by_scene])) for k in weighted}
        for value in (weighted, equal):
            value['RMSE'] = float(np.sqrt(value.pop('MSE')))
            value['TD_RMSE'] = float(np.sqrt(value.pop('TD_MSE')))
        result[f'critic{c+1}'] = dict(transition_weighted=weighted, scene_equal=equal)
    cal.finite_tree(result)
    return result


def evaluate(data, critics, targets, rows, split, arm, beta):
    ids = list(range(len(rows)))
    prediction = predict(data, critics, rows, ids)
    td = []
    for start in range(0, len(rows), 64):
        pair = labels(data, targets, rows, ids[start:start+64], arm, beta)
        td.extend(torch.stack(pair, dim=1).cpu().tolist())
    td = np.asarray(td, dtype=np.float64)
    grouped = []
    for partition, scene_ids in split.items():
        for geometry in ('all', 'empty', 'nonempty'):
            for kind in ('all', 'terminal', 'nonterminal', 'failure_path', 'failure_terminal'):
                subset = [i for i,r in enumerate(rows) if r['scenario_id'] in scene_ids
                    and (geometry=='all' or r['empty_scene']==(geometry=='empty'))
                    and (kind not in ('terminal','failure_terminal') or r['done'])
                    and (kind!='nonterminal' or not r['done'])
                    and (not kind.startswith('failure') or r['outcome']!='goal')]
                grouped.append(dict(partition=partition, geometry=geometry, slice=kind,
                                    **metrics(rows, prediction, td, subset)))
    per_scene = {sid:metrics(rows,prediction,td,[i for i,r in enumerate(rows) if r['scenario_id']==sid])
                 for sid in sorted({r['scenario_id'] for r in rows})}
    return dict(grouped=grouped, per_scene=per_scene)


def initial_state(data, critics, targets, optimizer):
    return dict(actor=cal.state_hash(data['actor'].state_dict()),
        online=cal.state_hash([c.state_dict() for c in critics]),
        targets=cal.state_hash([c.state_dict() for c in targets]), adam=cal.state_hash(optimizer.state_dict()))


def fit_arm(data, artifact, arm):
    started = time.perf_counter()
    if arm not in ARMS: raise ValueError('Unknown intervention arm.')
    spec = artifact['protocol']; protocol(spec['updates'])
    if spec != protocol(spec['updates']): raise ValueError('Protocol changed after preparation.')
    if artifact['generation']!=generation_config(data): raise ValueError('Source generation configuration changed.')
    rows, split, batches = artifact['rows'], artifact['split'], artifact['batches']
    e = data['checkpoint_payload']['engine_checkpoint']; cfg=e['algorithm_config']
    critics, optimizer, parameters = cal.make_fit(data['critics'], e['critic_optimizer_state_dict'])
    actor_target, *targets = cal.load_targets(data)
    data['actor'].eval().requires_grad_(False)
    frozen = [data['actor'], actor_target, *data['critics']]
    frozen_before = cal.state_hash([m.state_dict() for m in frozen])
    initial = initial_state(data, critics, targets, optimizer)
    if initial != artifact['initial']: raise ValueError('Arm initial network/target/Adam identity mismatch.')
    validate_batches(rows, split, batches, spec['updates'])
    cal_ids = [i for i,r in enumerate(rows) if r['scenario_id'] in split['calibration']]
    beta = .5
    curve, controls, norms = [], [dict(updates=0, beta_before=.5, beta_after=.5)], []
    target_updates, evaluation_seconds, calibration_seconds = 0, 0., 0.
    initialization_seconds = time.perf_counter()-started
    for step in range(spec['updates']+1):
        if step in spec['evaluations']:
            begin=time.perf_counter()
            curve.append(dict(updates=step, beta=beta,
                metrics=evaluate(data,critics,targets,rows,split,arm,beta)))
            evaluation_seconds += time.perf_counter()-begin
        if step == spec['updates']: break
        indices=batches[step]
        y=labels(data,targets,rows,indices,arm,beta)
        obs, actions=cal.tensor_batch(rows,indices,data['device'])
        optimizer.zero_grad(set_to_none=True)
        loss=sum(F.mse_loss(c(obs,actions).reshape(-1), target) for c,target in zip(critics,y))
        cal.finite_tree(loss); loss.backward()
        clip=cfg['critic_grad_clip_norm']
        norm=torch.nn.utils.clip_grad_norm_(parameters,clip if clip is not None else float('inf'),error_if_nonfinite=True)
        cal.finite_tree(norm); norms.append(float(norm.detach().cpu()))
        optimizer.step()
        if (e['update_counts']['last_total_steps']+step+1)%cfg['policy_delay']==0:
            with torch.no_grad():
                for online,target in zip(critics,targets):
                    ps,ts=list(online.parameters()),list(target.parameters())
                    torch._foreach_mul_(ts,1-cfg['tau']); torch._foreach_add_(ts,ps,alpha=cfg['tau'])
            target_updates+=1
        cal.finite_tree([c.state_dict() for c in [*critics,*targets]])
        cal.finite_tree(optimizer.state_dict())
        if arm=='acc' and (step+1)%spec['acc_interval']==0:
            begin=time.perf_counter()
            beta, control=acc_control(beta,predict(data,critics,rows,cal_ids),rows,cal_ids,artifact['generation']['p_empty'])
            controls.append(dict(updates=step+1,**control))
            calibration_seconds+=time.perf_counter()-begin
    if torch.device(data['device']).type=='cuda': torch.cuda.synchronize(data['device'])
    unchanged=frozen_before==cal.state_hash([m.state_dict() for m in frozen])
    if not unchanged or any(p.grad is not None for m in frozen[:2] for p in m.parameters()):
        raise RuntimeError('Frozen actor/source state changed or received gradients.')
    final_targets=cal.state_hash([m.state_dict() for m in targets])
    final_online=cal.state_hash([m.state_dict() for m in critics])
    elapsed=time.perf_counter()-started
    return dict(diagnostic_only=True, formal_passed=False, arm=arm, initial=initial,
        actual_updates=spec['updates'], batch_sha256=cal.state_hash(batches.tolist()),
        frozen_unchanged=unchanged, frozen_state_sha256=frozen_before,
        target_updates=target_updates, final_targets=final_targets, final_online=final_online,
        gradient=dict(scope='one joint global critic1+critic2 norm per TD batch', pre_clip_norms=norms,
            clipping_fraction=float(np.mean(np.asarray(norms)>clip)) if clip is not None else 0.),
        timing=dict(total_wall_seconds=elapsed,initialization_wall_seconds=initialization_seconds,
            evaluation_wall_seconds=evaluation_seconds,
            calibration_wall_seconds=calibration_seconds,
            note='CPU wall includes device queue/wait; total includes final CUDA synchronize, finite checks and evaluation; not pure GPU time'),
        first_boundary_update=next((c['updates'] for c in controls if c['beta_after'] in (0.,1.)),None),
        controls=controls, curve=curve)


def validate_batches(rows, split, batches, updates):
    if set(split)!= {'train','calibration','score'}: raise ValueError('Missing scene partition.')
    if [len(split[k]) for k in ('train','calibration','score')]!=[30,10,20]:
        raise ValueError('Require 30/10/20 scene partitions.')
    flat=[sid for ids in split.values() for sid in ids]
    if len(flat)!=len(set(flat)): raise ValueError('Scene partition leakage.')
    if set(flat)!={r['scenario_id'] for r in rows}: raise ValueError('Scene coverage mismatch.')
    empty={sid:{r['empty_scene'] for r in rows if r['scenario_id']==sid} for sid in flat}
    if any(len(v)!=1 for v in empty.values()) or sum(empty[sid]=={True} for sid in split['calibration'])!=1:
        raise ValueError('Calibration requires one empty and nine nonempty scenes with consistent metadata.')
    train=[i for i,r in enumerate(rows) if r['scenario_id'] in split['train']]
    expected=np.random.default_rng(SEED).choice(train,size=(updates,64),replace=True)
    if batches.shape != expected.shape or not np.issubdtype(batches.dtype,np.integer) or not np.array_equal(batches,expected):
        raise ValueError('Batch plan mismatch or calibration/score leakage.')


def source_data(args):
    expected,lr=SOURCE[args.model]
    if file_sha256(args.checkpoint)!=expected: raise ValueError('Specified seed23 checkpoint SHA256 mismatch.')
    data=cal.base.load_models(args.checkpoint,args.validation_pool,args.device)
    p=data['checkpoint_payload']; e=p['engine_checkpoint']; cfg=e['algorithm_config']
    if (data['checkpoint_hash']!=expected or data['model']!=args.model or p['stage']!='easy'
            or p['formal_config']['seed']!=23 or p['formal_config']['critic_lr']!=lr
            or p['training_result']['stage_steps']!=500000 or cfg.get('mc_return_weight',0.)!=0.
            or cfg.get('target_q_aggregation','min')!='min'
            or any(g['lr']!=lr for g in e['critic_optimizer_state_dict']['param_groups'])):
        raise ValueError('Source is not the specified historical min/seed23/500k model.')
    cal.finite_tree(e)
    return data


def generation_config(data):
    # Formal training constructs V2ScenarioGenerator without a config override.
    # These legacy checkpoints do NOT serialize that separate dataclass.
    config=V2ScenarioGeneratorConfig()
    p=float(config.zone_count_probabilities['easy'].get(0,0.))
    if not 0 < p < 1: raise ValueError('Source easy generator must support both strata.')
    return dict(config=cal.base.serial(asdict(config)), p_empty=p,
        provenance='legacy source formal default V2ScenarioGeneratorConfig; not a serialized checkpoint field',
        scenario_config=cal.base.serial(data['checkpoint_payload']['scenario_config']))


def execution_backend(data):
    return dict(path='eager eval inference_mode; single-observation actor_action; SNN forward resets state',
        device=str(data['device']),torch=torch.__version__,cuda=torch.version.cuda,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        threads=torch.get_num_threads())


def history_score(args,data):
    """Reuse ALL twenty matching historical episodes; never fly replacements."""
    root=Path(args.score_history)
    prior=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    keys=[k for k,v in prior['inputs'].items() if v['sha256']==data['checkpoint_hash']]
    if len(keys)!=1 or prior['pool']['content_digest']!=data['pool'].content_digest:
        raise ValueError('Existing score history identity mismatch.')
    records=[json.loads(x) for x in (root/'baseline_episodes.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    records=[r for r in records if r['model']==keys[0]]
    scenes=cal.base.select_scenarios(data['pool'])
    if len(records)!=20 or {r['scenario_id'] for r in records}!={s['scenario_id'] for s in scenes}:
        raise ValueError('Require all twenty original identity-matched score episodes.')
    if len({r['scenario_id'] for r in records})!=20: raise ValueError('Duplicate historical score episode.')
    by_id={s['scenario_id']:s for s in scenes}
    for r in records:
        s=by_id[r['scenario_id']]
        if r['sequence_index']!=s['sequence_index'] or r['empty_scene']!=(not s['payload']['zones']):
            raise ValueError('Historical score scene metadata mismatch.')
    return records,[dict(s,partition='score') for s in scenes],prior['execution']


def verify_record(data,scene,record):
    rows=cal.replay_episode(data,scene,record)
    for row in rows:
        if not np.array_equal(cal.base.actor_action(data,row['observation']),row['action']):
            raise ValueError('Saved actions are not this frozen actor without exploration; backend mismatch is not relaxed.')
        row['outcome']=record['outcome']
    return rows


def collection_plan(args,data):
    """Select by independent RNG and geometry ONLY, before any policy flight."""
    spec=generation_config(data)
    generator=V2ScenarioGenerator(data['scenario'],'easy',seed=SEED,config=V2ScenarioGeneratorConfig())
    score=[dict(s,partition='score') for s in cal.base.select_scenarios(data['pool'])]
    used={scene_key(s['payload']) for s in data['pool'].scenarios}
    scenes=[]; candidates=0
    # First thirty distinct nonvalidation draws form the unstratified TD set.
    # Then retain first empty and first nine nonempty draws for calibration.
    counts={True:0,False:0}
    while len(scenes)<40:
        candidates+=1
        if candidates>10000: raise ValueError('Pre-flight scene selection exhausted; no automatic retry.')
        payload=generator.generate(); key=scene_key(payload)
        if key in used: continue
        if len(scenes)<30: partition='train'
        else:
            empty=not payload['zones']
            if counts[empty]>=(1 if empty else 9): continue
            counts[empty]+=1; partition='calibration'
        used.add(key)
        scenes.append(dict(scenario_id=f'wd3acc_{partition}_{len(scenes):05d}',sequence_index=candidates,
            payload=payload,partition=partition))
    scenes+=score
    split_scenes(scenes,{s['scenario_id'] for s in data['pool'].scenarios},
                 {scene_key(s['payload']) for s in data['pool'].scenarios})
    return scenes,dict(spec,seed=SEED,candidate_draws=candidates,
        selection='first 30 distinct nonvalidation easy draws; then first 1 empty and 9 nonempty; no flight outcomes used')


def collect(args,output):
    paths=input_paths(args); hashes={k:file_sha256(p) for k,p in paths.items()}
    data=source_data(args)
    if hashes['checkpoint']!=data['checkpoint_hash'] or hashes['validation_pool']!=data['pool_hash']:
        raise RuntimeError('Input changed while loading.')
    prior_records=[]; prior_backend=None
    if args.model=='snn' and not args.score_history:
        raise ValueError('SNN collect requires existing B --score-history; never recollect score.')
    if args.score_history:
        prior_records,_,prior_backend=history_score(args,data)
    scenes,generation=collection_plan(args,data)
    before=cal.state_hash([data['actor'].state_dict(),*[c.state_dict() for c in data['critics']]])
    manifest=dict(format='v2_frozen_actor_episodes_v2',protocol=protocol(args.updates),
        checkpoint_sha256=data['checkpoint_hash'],actor_sha256=cal.state_hash(data['actor'].state_dict()),
        exploration_noise=0.,scenarios=scenes,generation=generation,code=code_identity(),
        backend=execution_backend(data),reused_score_backend=prior_backend,input_hashes=hashes,
        config=dict(scenario=data['checkpoint_payload']['scenario_config'],
            rewards=data['checkpoint_payload']['reward_config'],gamma=data['gamma']),
        scene_keys={s['scenario_id']:scene_key(s['payload']) for s in scenes})
    _write_json(output/'manifest.json',manifest)  # ALL selection sealed before replay or flight
    saved={r['scenario_id']:r for r in prior_records}
    # Validate available score BEFORE collecting anything new. No replacement flight on failure.
    for scene in scenes:
        if scene['scenario_id'] in saved: verify_record(data,scene,saved[scene['scenario_id']])
    with (output/'baseline_episodes.jsonl').open('x',encoding='utf-8') as stream:
        for scene in scenes:
            record=saved.get(scene['scenario_id'])
            if record is None:
                env=cal.base.make_env(data)
                obs,info=env.reset(options={'scenario':scene['payload']})
                record=dict(cal.base.fly(data,env,obs,record_probe_hashes=True,initial_info=info),
                    scenario_id=scene['scenario_id'],sequence_index=scene['sequence_index'],
                    empty_scene=not scene['payload']['zones'])
                verify_record(data,scene,record)
            stream.write(json.dumps(cal.base.serial(record),allow_nan=False)+'\n'); stream.flush()
    if before!=cal.state_hash([data['actor'].state_dict(),*[c.state_dict() for c in data['critics']]]):
        raise RuntimeError('Frozen networks changed during collection.')
    check_inputs(paths,hashes,manifest['code'])
    evidence=dict(manifest_sha256=file_sha256(output/'manifest.json'),
        episodes_sha256=file_sha256(output/'baseline_episodes.jsonl'))
    _write_json(output/'evidence.json',evidence)
    return dict(episodes=len(scenes),collected=len(scenes)-len(saved),reused=len(saved),frozen_unchanged=True,**evidence)


def check_inputs(paths,hashes,code):
    for k,p in paths.items():
        if file_sha256(p)!=hashes[k]: raise RuntimeError('Input changed during execution: '+k)
    if code_identity()['sha256']!=code['sha256']: raise RuntimeError('Source changed during execution.')


def read_histories(args, data):
    """New independent episodes plus optional identity-matched existing score history."""
    root=Path(args.independent_history)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    records=[json.loads(x) for x in (root/'baseline_episodes.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if (manifest.get('format')!='v2_frozen_actor_episodes_v2' or manifest.get('exploration_noise')!=0.
            or manifest.get('checkpoint_sha256')!=data['checkpoint_hash']
            or manifest.get('actor_sha256')!=cal.state_hash(data['actor'].state_dict())):
        raise ValueError('Independent history actor/checkpoint/no-exploration identity mismatch.')
    if manifest['generation']['config']!=generation_config(data)['config']:
        raise ValueError('History generation configuration mismatch.')
    if manifest['generation']['p_empty']!=generation_config(data)['p_empty']:
        raise ValueError('History source empty probability mismatch.')
    evidence=json.loads((root/'evidence.json').read_text(encoding='utf-8'))
    if (file_sha256(root/'manifest.json')!=evidence['manifest_sha256'] or
            file_sha256(root/'baseline_episodes.jsonl')!=evidence['episodes_sha256']):
        raise ValueError('Collected history SHA256 mismatch.')
    if code_identity()['sha256']!=manifest['code']['sha256']:
        raise ValueError('Source differs from sealed collection implementation.')
    scenes={s['scenario_id']:s for s in manifest['scenarios']}
    if len(scenes)!=len(manifest['scenarios']): raise ValueError('Duplicate scene IDs.')
    if args.score_history:
        score,score_scenes,_=history_score(args,data)
        records+=score
        for s in score_scenes:
            if s['scenario_id'] in scenes: raise ValueError('Independent/score scene IDs overlap; omit --score-history for a sealed 60-scene collection.')
            scenes[s['scenario_id']]=s
    if len(records)!=len(scenes) or {r['scenario_id'] for r in records}!=set(scenes):
        raise ValueError('History scene coverage mismatch.')
    for r in records:
        s=scenes[r['scenario_id']]
        if (s['payload']['curriculum_level']!='easy' or r['sequence_index']!=s['sequence_index']
                or r['empty_scene']!=(not s['payload']['zones'])):
            raise ValueError('History scene metadata mismatch.')
    ordered=[scenes[r['scenario_id']] for r in records]
    pool_ids={s['scenario_id'] for s in data['pool'].scenarios}
    pool_keys={scene_key(s['payload']) for s in data['pool'].scenarios}
    split=split_scenes(ordered,pool_ids,pool_keys)
    if set(split['score'])!={s['scenario_id'] for s in cal.base.select_scenarios(data['pool'])}:
        raise ValueError('Score must use the same original fixed twenty scenes for both models.')
    return records,scenes,split


def input_paths(args):
    paths=dict(checkpoint=Path(args.checkpoint),validation_pool=Path(args.validation_pool))
    for key in ('independent_history','score_history'):
        if getattr(args,key,None):
            root=Path(getattr(args,key))
            paths[key+'_manifest']=root/'manifest.json'
            paths[key+'_episodes']=root/'baseline_episodes.jsonl'
            if key=='independent_history': paths[key+'_evidence']=root/'evidence.json'
    return paths


def prepare(args, output):
    paths=input_paths(args); hashes={k:file_sha256(p) for k,p in paths.items()}
    data=source_data(args)
    if hashes['checkpoint']!=data['checkpoint_hash'] or hashes['validation_pool']!=data['pool_hash']:
        raise RuntimeError('Input changed while loading.')
    records,scenes,split=read_histories(args,data)
    spec=protocol(args.updates)
    generation=generation_config(data)
    history_manifest=json.loads((Path(args.independent_history)/'manifest.json').read_text(encoding='utf-8'))
    manifest=dict(protocol=spec,model=args.model,input_hashes=hashes,split=split,generation=generation,
        code=code_identity(),software=dict(torch=torch.__version__,cuda=torch.version.cuda,
            device=str(data['device']),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32),
        actor_sha256=cal.state_hash(data['actor'].state_dict()),
        scene_keys={sid:scene_key(s['payload']) for sid,s in scenes.items()},
        scene_empty={sid:not s['payload']['zones'] for sid,s in scenes.items()},
        history_backend=history_manifest['backend'],
        reused_score_backend=history_score(args,data)[2] if args.score_history else history_manifest.get('reused_score_backend'),
        replay_backend=execution_backend(data))
    _write_json(output/'manifest.json',manifest)  # split exists BEFORE any replay/fitting
    rows=[]
    for record in records:
        replayed=verify_record(data,scenes[record['scenario_id']],record)
        rows.extend(replayed)
    train=[i for i,r in enumerate(rows) if r['scenario_id'] in split['train']]
    batches=np.random.default_rng(SEED).choice(train,size=(spec['updates'],64),replace=True)
    critics,opt,_=cal.make_fit(data['critics'],data['checkpoint_payload']['engine_checkpoint']['critic_optimizer_state_dict'])
    targets=cal.load_targets(data)[1:]
    artifact=dict(protocol=spec,rows=rows,batches=batches,split=split,input_hashes=hashes,generation=generation,
        initial=initial_state(data,critics,targets,opt))
    if artifact['initial']['actor']!=manifest['actor_sha256']:
        raise RuntimeError('Actor changed during replay/verification.')
    torch.save(artifact,output/'data.pt')
    check_inputs(paths,hashes,manifest['code'])
    evidence=dict(data_sha256=file_sha256(output/'data.pt'),manifest_sha256=file_sha256(output/'manifest.json'),
        batch_sha256=cal.state_hash(batches.tolist()))
    _write_json(output/'evidence.json',evidence)
    return dict(transitions=len(rows),split=split,**evidence)


def load_prepared(args, data):
    root=Path(args.prepared_dir)
    evidence=json.loads((root/'evidence.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name in ('data','manifest'):
        if file_sha256(root/('data.pt' if name=='data' else 'manifest.json'))!=evidence[name+'_sha256']:
            raise ValueError('Prepared artifact SHA256 mismatch.')
    artifact=torch.load(root/'data.pt',map_location='cpu',weights_only=False)
    if (artifact['input_hashes']!=manifest['input_hashes'] or artifact['split']!=manifest['split']
            or artifact['protocol']!=manifest['protocol'] or manifest['model']!=args.model
            or artifact['generation']!=manifest['generation'] or artifact['generation']!=generation_config(data)
            or artifact['input_hashes']['checkpoint']!=data['checkpoint_hash']
            or artifact['input_hashes']['validation_pool']!=data['pool_hash']):
        raise ValueError('Prepared provenance mismatch.')
    if cal.state_hash(artifact['batches'].tolist())!=evidence['batch_sha256']:
        raise ValueError('Prepared batch SHA256 mismatch.')
    spec=artifact['protocol']
    if spec!=protocol(args.updates): raise ValueError('Requested/prepared budget or protocol differs.')
    if code_identity()['sha256']!=manifest['code']['sha256']: raise ValueError('Source differs from prepared protocol implementation.')
    split=artifact['split']
    if [len(split[k]) for k in ('train','calibration','score')]!=[30,10,20]: raise ValueError('Wrong scene split sizes.')
    if sum(manifest['scene_empty'][sid] for sid in split['calibration'])!=1:
        raise ValueError('Calibration must have 1 empty and 9 nonempty scenes.')
    if set(split['score'])!={s['scenario_id'] for s in cal.base.select_scenarios(data['pool'])}:
        raise ValueError('Score does not match original twenty fixed scenes.')
    pool_ids={s['scenario_id'] for s in data['pool'].scenarios}
    pool_keys={scene_key(s['payload']) for s in data['pool'].scenarios}
    for sid in split['train']+split['calibration']:
        if sid in pool_ids or manifest['scene_keys'][sid] in pool_keys:
            raise ValueError('Fixed validation leakage.')
    if len(set(manifest['scene_keys'].values()))!=60: raise ValueError('Physical scene duplication.')
    if any(r['empty_scene']!=manifest['scene_empty'][r['scenario_id']] for r in artifact['rows']):
        raise ValueError('Prepared row stratum mismatch.')
    validate_batches(artifact['rows'],split,artifact['batches'],spec['updates'])
    cal.finite_tree([[r[k] for k in ('G','reward')] for r in artifact['rows']])
    return artifact,evidence


def run(args):
    output=Path(args.output_dir)
    if args.mode=='collect' and output.exists():
        # Explicitly validate sealed existing data; never append, overwrite or refly.
        previous=json.loads((output/'summary.json').read_text(encoding='utf-8'))
        if not previous['execution_completed']: raise ValueError('Existing collection incomplete; automatic recollection forbidden.')
        data=source_data(args)
        existing=deepcopy(args); existing.independent_history=output; existing.score_history=None
        paths=input_paths(existing); hashes={k:file_sha256(p) for k,p in paths.items()}
        manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        records,scenes,_=read_histories(existing,data)
        if hashes['checkpoint']!=data['checkpoint_hash'] or hashes['validation_pool']!=data['pool_hash']:
            raise RuntimeError('Input changed while validating existing collection.')
        for r in records: verify_record(data,scenes[r['scenario_id']],r)
        check_inputs(paths,hashes,manifest['code'])
        return dict(previous,existing_validated=True)
    output.mkdir(parents=True,exist_ok=False)
    summary=dict(diagnostic_only=True,formal_passed=False,execution_completed=False,mode=args.mode)
    started=time.perf_counter()
    try:
        if args.mode=='collect': result=collect(args,output)
        elif args.mode=='prepare': result=prepare(args,output)
        else:
            paths=input_paths(args)
            paths.update({n:Path(args.prepared_dir)/f for n,f in
                (('prepared_data','data.pt'),('prepared_manifest','manifest.json'),('prepared_evidence','evidence.json'))})
            hashes={k:file_sha256(p) for k,p in paths.items()}
            data=source_data(args); artifact,evidence=load_prepared(args,data)
            if hashes['checkpoint']!=data['checkpoint_hash'] or hashes['validation_pool']!=data['pool_hash']:
                raise RuntimeError('Input changed while loading.')
            if hashes['prepared_data']!=evidence['data_sha256'] or hashes['prepared_manifest']!=evidence['manifest_sha256']:
                raise RuntimeError('Prepared input changed while loading.')
            _write_json(output/'manifest.json',dict(protocol=artifact['protocol'],arm=args.arm,model=args.model,
                code=code_identity(),software=dict(torch=torch.__version__,cuda=torch.version.cuda,device=str(data['device'])),
                split=artifact['split'],generation=artifact['generation'],initial=artifact['initial'],input_hashes=hashes,data_evidence=evidence,
                prepared_source_input_hashes=artifact['input_hashes'],
                source_aggregation='min',intervention_aggregation=args.arm))
            fitted=fit_arm(data,artifact,args.arm)
            for k,p in paths.items():
                if file_sha256(p)!=hashes[k]: raise RuntimeError('Input changed during fitting: '+k)
            prepared_manifest=json.loads(paths['prepared_manifest'].read_text(encoding='utf-8'))
            if code_identity()['sha256']!=prepared_manifest['code']['sha256']: raise RuntimeError('Source changed during fitting.')
            _write_json(output/'fit_curves.json',fitted)
            result={k:v for k,v in fitted.items() if k not in ('curve','gradient','controls')}
        summary.update(execution_completed=True,result=result)
    except BaseException as exc:
        summary['error']=dict(type=type(exc).__name__,message=str(exc)); raise
    finally:
        summary['process_wall_seconds']=time.perf_counter()-started
        _write_json(output/'summary.json',summary)
    return summary


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('collect','prepare','fit'),required=True)
    p.add_argument('--model',choices=tuple(SOURCE),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--validation-pool',type=Path,required=True)
    p.add_argument('--independent-history',type=Path)
    p.add_argument('--score-history',type=Path)
    p.add_argument('--prepared-dir',type=Path)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--updates',type=int,default=10000)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--launch',action='store_true')
    return p


def main(argv=None):
    p=build_parser(); args=p.parse_args(argv)
    spec=protocol(args.updates)
    if args.mode=='collect' and args.model=='snn' and args.score_history is None:
        p.error('SNN collect requires --score-history for original B twenty episodes')
    if args.mode=='collect' and args.independent_history is not None:
        p.error('collect writes --output-dir; --independent-history is for prepare')
    if args.mode=='prepare' and args.independent_history is None: p.error('prepare requires --independent-history')
    if args.mode=='fit' and (args.prepared_dir is None or args.arm is None): p.error('fit requires --prepared-dir and --arm')
    if not args.launch:
        print(json.dumps(dict(protocol=spec,model=args.model,arm=args.arm,mode=args.mode,
            expected_checkpoint_sha256=SOURCE[args.model][0],critic_lr=SOURCE[args.model][1],
            inputs={k:str(v) for k,v in input_paths(args).items()},output_dir=str(args.output_dir),
            launch=False),indent=2))
        return 0
    run(args)
    return 0


if __name__=='__main__': raise SystemExit(main())
