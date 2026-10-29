"""Fixed-policy dynamic min/mean critic intervention; never a formal checkpoint."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from brain_uav.scripts import diagnose_v2_critic_calibration as cal
from brain_uav.scripts.diagnose_v2_policy import _write_json, file_sha256, _software_versions
from brain_uav.trainers.v2_td3 import aggregate_v2_target_q

EVALUATIONS = (0, 100, 500, 1000, 2000)


def source_data(args):
    if file_sha256(args.checkpoint) != cal.CHECKPOINT_SHA256:
        raise ValueError('B source checkpoint SHA256 mismatch.')
    data = cal.base.load_models(args.checkpoint, args.validation_pool, args.device)
    p = data['checkpoint_payload']; e = p['engine_checkpoint']; cfg = e['algorithm_config']
    if (data['model'] != 'snn' or p['stage'] != 'easy' or p['formal_config']['seed'] != 23
            or p['training_result']['stage_steps'] != 500000 or p['formal_config']['critic_lr'] != .001
            or cfg.get('target_q_aggregation', 'min') != 'min' or cfg['batch_size'] != 64
            or cfg['policy_noise'] != .006 or cfg['noise_clip'] != .012
            or data['pool'].content_digest != cal.base.POOL_DIGEST
            or any(g['lr'] != .001 for g in e['critic_optimizer_state_dict']['param_groups'])):
        raise ValueError('Source is not the specified historical B/min experiment.')
    cal.finite_tree(e)
    return data


def history_selection(args, pool):
    history = Path(args.history_dir)
    prior = json.loads((history/'manifest.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (history/'baseline_episodes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    records = [r for r in records if r['model'] == 'B']
    split = cal.split_scenes(records)
    if (prior['inputs']['B']['sha256'] != cal.CHECKPOINT_SHA256
            or pool.content_digest != cal.base.POOL_DIGEST
            or prior['pool']['content_digest'] != pool.content_digest):
        raise ValueError('History provenance mismatch.')
    selected = {r['scenario_id']: r for r in pool.scenarios}
    if set(r['scenario_id'] for r in records) != set(r['scenario_id'] for r in prior['selection']['scenarios']):
        raise ValueError('History selection mismatch.')
    for r in records:
        s = selected[r['scenario_id']]
        if r['empty_scene'] != (not s['payload']['zones']) or r['sequence_index'] != s['sequence_index']:
            raise ValueError('History scene identity mismatch.')
    return records, split, selected


def prepare(args, output):
    data = source_data(args)
    records, split, scenes = history_selection(args, data['pool'])
    history = Path(args.history_dir)
    hashes = {name: file_sha256(path) for name, path in {
        'checkpoint': args.checkpoint, 'pool': args.validation_pool,
        'history_manifest': history/'manifest.json', 'episodes': history/'baseline_episodes.jsonl'}.items()}
    manifest = dict(diagnostic_only=True, source='min', input_hashes=hashes,
        pool_digest=data['pool'].content_digest, split=split, software=_software_versions(),
        definition='one strict linear action replay per historical scene; G includes current reward; no new policy flight',
        updates=2000, batch_size=64, target_policy='frozen online actor; zero smoothing noise',
        target_update='saved tau and policy_delay; phase starts at saved last_total_steps',
        scenarios=[dict(scenario_id=r['scenario_id'], empty_scene=r['empty_scene'],
                        sequence_index=r['sequence_index']) for r in records])
    _write_json(output/'manifest.json', manifest)  # Persist split BEFORE any replay.
    rows = [row for r in records for row in cal.replay_episode(data, scenes[r['scenario_id']], r)]
    train = [i for i, r in enumerate(rows) if r['scenario_id'] in split['train']]
    batches = np.random.default_rng(cal.SPLIT_SEED).choice(train, size=(2000, 64), replace=True)
    artifact = dict(rows=rows, split=split, batches=batches,
                    batch_sha256=cal.state_hash(batches.tolist()), input_hashes=hashes)
    torch.save(artifact, output/'data.pt')
    for name, path in {'checkpoint': args.checkpoint, 'pool': args.validation_pool,
                       'history_manifest': history/'manifest.json', 'episodes': history/'baseline_episodes.jsonl'}.items():
        if file_sha256(path) != hashes[name]: raise RuntimeError('Input changed during preparation.')
    _write_json(output/'data_evidence.json', dict(data_sha256=file_sha256(output/'data.pt'),
        manifest_sha256=file_sha256(output/'manifest.json'),
        batch_sha256=artifact['batch_sha256'], transitions=len(rows)))
    return dict(transitions=len(rows), split=split)


def load_prepared(args):
    root = Path(args.prepared_dir)
    evidence = json.loads((root/'data_evidence.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if file_sha256(root/'manifest.json') != evidence['manifest_sha256']:
        raise ValueError('Prepared manifest SHA256 mismatch.')
    if file_sha256(root/'data.pt') != evidence['data_sha256']:
        raise ValueError('Prepared dataset SHA256 mismatch.')
    # Trusted local artifact; observations are project dataclasses, not tensors alone.
    artifact = torch.load(root/'data.pt', map_location='cpu', weights_only=False)
    if artifact['input_hashes'] != manifest['input_hashes'] or artifact['split'] != manifest['split']:
        raise ValueError('Prepared manifest mismatch.')
    if (file_sha256(args.checkpoint) != artifact['input_hashes']['checkpoint']
            or file_sha256(args.validation_pool) != artifact['input_hashes']['pool']):
        raise ValueError('Prepared input mismatch.')
    if manifest['source'] != 'min' or manifest['pool_digest'] != cal.base.POOL_DIGEST:
        raise ValueError('Prepared source mismatch.')
    rows, batches, split = artifact['rows'], artifact['batches'], artifact['split']
    if set(split['train']) & set(split['holdout']) or len(split['train']) != 16 or len(split['holdout']) != 4:
        raise ValueError('Invalid scene split.')
    if split != cal.split_scenes(manifest['scenarios']): raise ValueError('Scene split changed.')
    if set(r['scenario_id'] for r in rows) != set(split['train'] + split['holdout']):
        raise ValueError('Prepared scene coverage mismatch.')
    if batches.shape != (2000, 64) or not np.issubdtype(batches.dtype, np.integer):
        raise ValueError('Invalid batch plan.')
    if batches.min() < 0 or batches.max() >= len(rows): raise ValueError('Batch index out of range.')
    if any(rows[int(i)]['scenario_id'] not in split['train'] for i in np.unique(batches)):
        raise ValueError('Holdout leakage in batch plan.')
    expected = np.random.default_rng(cal.SPLIT_SEED).choice(
        [i for i,r in enumerate(rows) if r['scenario_id'] in split['train']], size=(2000,64), replace=True)
    if not np.array_equal(batches, expected) or cal.state_hash(batches.tolist()) != evidence['batch_sha256']:
        raise ValueError('Batch plan mismatch.')
    cal.finite_tree([r['G'] for r in rows])
    return artifact, evidence


def dynamic_labels(data, targets, rows, indices, aggregation):
    """Recomputed from this arm's CURRENT targets; G is never read here."""
    nxt, _ = cal.tensor_batch(rows, indices, data['device'], next_state=True)
    with torch.no_grad():
        action = data['actor'](nxt)  # SNN forward retains its own reset context.
        q1, q2 = (c(nxt, action).reshape(-1) for c in targets)
        q = aggregate_v2_target_q(q1, q2, aggregation)
        reward = torch.tensor([rows[int(i)]['reward'] for i in indices], dtype=torch.float32, device=data['device'])
        done = torch.tensor([rows[int(i)]['done'] for i in indices], dtype=torch.float32, device=data['device'])
        y = reward + data['gamma']*(1-done)*q
        cal.finite_tree([action, q1, q2, y])
        return y


def evaluate_fixed(data, critics, targets, rows, split, aggregation):
    predictions, labels = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            indices = list(range(start, min(start+64, len(rows))))
            obs, actions = cal.tensor_batch(rows, indices, data['device'])
            q = torch.stack([c(obs, actions).reshape(-1) for c in critics], dim=1)
            cal.finite_tree(q); predictions.extend(q.cpu().tolist())
            labels.extend(dynamic_labels(data, targets, rows, indices, aggregation).cpu().tolist())
    metrics_rows = [dict(r, y=y) for r,y in zip(rows, labels)]
    grouped = cal.statistics(metrics_rows, predictions, split)
    per_scene = {sid: cal.statistics(metrics_rows, predictions, dict(train=[sid], holdout=[]))[0]
                 for sid in sorted({r['scenario_id'] for r in rows})}
    for sid, entry in per_scene.items():
        entry['partition'] = 'train' if sid in split['train'] else 'holdout'
    # Signed residual alone can cancel: also retain TD MAE/RMSE per group/scene.
    entries = [(e, None) for e in grouped] + [(e, sid) for sid,e in per_scene.items()]
    for entry, scene_id in entries:
        if entry.get('information') == 'insufficient': continue
        ids = [i for i,r in enumerate(rows) if r['scenario_id'] in split[entry['partition']]
            and (entry['geometry']=='all' or r['empty_scene']==(entry['geometry']=='empty'))
            and (entry['terminal']=='all' or r['done']==(entry['terminal']=='terminal'))]
        if scene_id is not None:
            ids = [i for i,r in enumerate(rows) if r['scenario_id']==scene_id]
        for c in range(2):
            error = np.array([predictions[i][c]-labels[i] for i in ids])
            scene_errors = [np.array([predictions[i][c]-labels[i] for i in ids if rows[i]['scenario_id']==sid])
                            for sid in sorted({rows[i]['scenario_id'] for i in ids})]
            entry[f'critic{c+1}']['transition_weighted'].update(TD_MAE=float(abs(error).mean()), TD_RMSE=float(np.sqrt((error**2).mean())))
            entry[f'critic{c+1}']['scene_equal'].update(TD_MAE=float(np.mean([abs(e).mean() for e in scene_errors])),
                TD_RMSE=float(np.sqrt(np.mean([(e**2).mean() for e in scene_errors]))))
    return dict(grouped=grouped, per_scene=per_scene)


def fit_arm(data, artifact, aggregation, *, budget=2000, evaluations=EVALUATIONS):
    rows, batches, split = (artifact[k] for k in ('rows','batches','split'))
    e = data['checkpoint_payload']['engine_checkpoint']; cfg = e['algorithm_config']
    critics, optimizer, parameters = cal.make_fit(data['critics'], e['critic_optimizer_state_dict'])
    targets = cal.load_targets(data)[1:]
    data['actor'].eval().requires_grad_(False)
    actor_before = cal.state_hash(data['actor'].state_dict())
    initial = dict(online=cal.state_hash([c.state_dict() for c in critics]),
        targets=cal.state_hash([c.state_dict() for c in targets]), adam=cal.state_hash(optimizer.state_dict()), actor=actor_before)
    curve, norms, target_updates = [], [], 0
    for step in range(budget+1):
        if step in evaluations:
            curve.append(dict(updates=step, metrics=evaluate_fixed(data, critics, targets, rows, split, aggregation)))
        if step == budget: break
        indices = batches[step]
        label = dynamic_labels(data, targets, rows, indices, aggregation)
        obs, actions = cal.tensor_batch(rows, indices, data['device'])
        optimizer.zero_grad(set_to_none=True)
        loss = sum(F.mse_loss(c(obs, actions).reshape(-1), label) for c in critics)
        cal.finite_tree(loss); loss.backward()
        clip = cfg['critic_grad_clip_norm']
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip if clip is not None else float('inf'), error_if_nonfinite=True)
        cal.finite_tree(norm); norms.append(float(norm.detach().cpu()))
        optimizer.step()
        if (e['update_counts']['last_total_steps'] + step + 1) % cfg['policy_delay'] == 0:
            with torch.no_grad():
                for online, target in zip(critics, targets):
                    ps, ts = list(online.parameters()), list(target.parameters())
                    torch._foreach_mul_(ts, 1-cfg['tau'])
                    torch._foreach_add_(ts, ps, alpha=cfg['tau'])
            target_updates += 1
        cal.finite_tree([c.state_dict() for c in [*critics,*targets]])
        cal.finite_tree(optimizer.state_dict())
    if actor_before != cal.state_hash(data['actor'].state_dict()): raise RuntimeError('Actor changed.')
    return dict(source='min', treatment=aggregation, diagnostic_only=True, actual_updates=budget,
        initial=initial, batch_sha256=cal.state_hash(batches.tolist()), actor_unchanged=True,
        target_updates=target_updates, final_targets=cal.state_hash([c.state_dict() for c in targets]),
        final_online=cal.state_hash([c.state_dict() for c in critics]), gradient_norms=norms, curve=curve)


def evaluate_policy(args, output):
    data = cal.base.load_models(args.checkpoint, args.validation_pool, args.device)
    history = Path(args.history_dir)
    paths = dict(checkpoint=Path(args.checkpoint), pool=Path(args.validation_pool),
                 history_manifest=history/'manifest.json', episodes=history/'baseline_episodes.jsonl')
    # Model/pool identity comes from the strict loader, not a later disk version.
    hashes = dict(checkpoint=data['checkpoint_hash'], pool=data['pool_hash'],
                  history_manifest=file_sha256(paths['history_manifest']), episodes=file_sha256(paths['episodes']))
    inputs = {name: dict(path=str(path.resolve()), sha256=hashes[name]) for name,path in paths.items()}
    _write_json(output/'manifest.json', dict(diagnostic_only=True, mode='evaluate', inputs=inputs))
    before = cal.state_hash([m.state_dict() for m in [data['actor'],*data['critics']]])
    _, _, selected = history_selection(args, data['pool'])
    # Identical selected scene IDs for every model, irrespective of its outcome/Q.
    ids = json.loads((Path(args.history_dir)/'manifest.json').read_text(encoding='utf-8'))['selection']['scenarios']
    rows, outcomes = [], {}
    for record in ids:
        scene = selected[record['scenario_id']]
        env = cal.base.make_env(data); obs, info = env.reset(options={'scenario': scene['payload']})
        episode = cal.base.fly(data, env, obs, record_probe_hashes=True, initial_info=info)
        episode.update(scenario_id=scene['scenario_id'], empty_scene=not scene['payload']['zones'])
        rows.extend(cal.replay_episode(data, scene, episode))
        outcomes[scene['scenario_id']] = episode['outcome']
    for row in rows: row['y'] = row['G']  # Only to reuse grouping; omit redundant TD field below.
    stats = [r for r in cal.evaluate(data['critics'], rows, data['device'], dict(train=list(outcomes), holdout=[]))
             if r['partition'] == 'train']
    per_scene = {sid: cal.evaluate(data['critics'],rows,data['device'],dict(train=[sid],holdout=[]))[0] for sid in outcomes}
    for entry in [*stats,*per_scene.values()]:
        entry['partition'] = 'evaluation'
        for c in ('critic1','critic2'):
            for value in entry.get(c,{}).values(): value.pop('Q_minus_y',None)
    if before != cal.state_hash([m.state_dict() for m in [data['actor'],*data['critics']]]):
        raise RuntimeError('Evaluation changed model weights.')
    with (output/'transitions.jsonl').open('w',encoding='utf-8') as f:
        with torch.no_grad():
            for start in range(0,len(rows),64):
                indices=list(range(start,min(start+64,len(rows))))
                obs,actions=cal.tensor_batch(rows,indices,data['device'])
                q=torch.stack([c(obs,actions).reshape(-1) for c in data['critics']],dim=1)
                cal.finite_tree(q)
                for i,values in zip(indices,q.cpu().tolist()):
                    r=rows[i]; f.write(json.dumps(dict(scenario_id=r['scenario_id'],step=r['step'],
                        Q1=values[0],Q2=values[1],G=r['G'],done=r['done']),allow_nan=False)+'\n')
    # Detailed evidence may exist on failure; publish no successful evaluation
    # until every input actually read by this run still has its original identity.
    for name,path in paths.items():
        if file_sha256(path) != hashes[name]:
            raise RuntimeError(f'Evaluation input changed: {name}.')
    _write_json(output/'evaluation.json', dict(metrics=stats, per_scene=per_scene, outcomes=outcomes,
        goal_rate=sum(x=='goal' for x in outcomes.values())/len(outcomes), inputs=inputs,
        checkpoint_sha256=data['checkpoint_hash'], pool_digest=data['pool'].content_digest,
        aggregation=data['checkpoint_payload']['engine_checkpoint']['algorithm_config'].get('target_q_aggregation','min')))
    return dict(scenarios=len(outcomes), frozen_unchanged=True)


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(diagnostic_only=True, formal_passed=False, execution_completed=False, mode=args.mode)
    try:
        if args.mode == 'prepare': result = prepare(args, output)
        elif args.mode == 'fit':
            data = source_data(args); artifact, evidence = load_prepared(args)
            _write_json(output/'manifest.json',dict(diagnostic_only=True,source='min',treatment=args.aggregation,
                data_evidence=evidence, split=artifact['split'], algorithm=data['checkpoint_payload']['engine_checkpoint']['algorithm_config'],
                intervention='frozen online actor, zero noise, dynamic TD; G evaluation only', evaluations=EVALUATIONS))
            result = fit_arm(data, artifact, args.aggregation)
            _write_json(output/'fit_curves.json',result)
            if file_sha256(Path(args.prepared_dir)/'data.pt') != evidence['data_sha256']:
                raise RuntimeError('Prepared data changed during fitting.')
            result = {k:v for k,v in result.items() if k not in ('curve','gradient_norms')}
        else: result = evaluate_policy(args, output)
        summary.update(execution_completed=True, result=result)
    except BaseException as exc:
        summary['error'] = dict(type=type(exc).__name__,message=str(exc)); raise
    finally: _write_json(output/'summary.json',summary)
    return summary


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','fit','evaluate'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--validation-pool',type=Path,required=True)
    p.add_argument('--history-dir',type=Path)
    p.add_argument('--prepared-dir',type=Path)
    p.add_argument('--aggregation',choices=('min','mean'))
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output-dir',type=Path,required=True)
    return p


def main(argv=None):
    p=build_parser(); args=p.parse_args(argv)
    if args.mode=='fit' and (args.prepared_dir is None or args.aggregation is None):
        p.error('fit requires --prepared-dir and --aggregation')
    if args.mode!='fit' and args.history_dir is None: p.error('prepare/evaluate require --history-dir')
    run(args)


if __name__=='__main__': main()
