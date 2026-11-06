"""Locked 400k candidates: independent scenes and development action probes only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace
import math

import numpy as np
import torch

from brain_uav.scripts import diagnose_v2_critic_calibration as cal
from brain_uav.scripts import diagnose_v2_online_research as metrics
from brain_uav.scripts import diagnose_v2_wd3_acc as offline
from brain_uav.scripts import run_v2_online_wd3_acc as research
from brain_uav.scripts.diagnose_v2_policy import _write_json, file_sha256
from brain_uav.trainers.v2_online_research import validate_snapshot
from brain_uav.trainers.v2_validation import (V2ValidationPool, load_v2_validation_pool,
    save_v2_validation_pool, derive_validation_stage_seed)

IDS = tuple(f'{m}_{a}' for m in ('ann', 'snn') for a in ('min', 'wd3_045', 'acc_001'))
SHAPES = {'sphere', 'ellipsoid', 'box', 'triangular_pyramid', 'quadrangular_pyramid'}
SEED = 20261102


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {x}')))


class Inputs:
    def __init__(self):
        self.files = {}
        self.source = research.identity()

    def add(self, path):
        path = Path(path).resolve()
        value = file_sha256(path)
        if str(path) in self.files and self.files[str(path)] != value:
            raise ValueError('Input replaced while reading.')
        self.files[str(path)] = value
        return path

    def check(self):
        if any(file_sha256(Path(p)) != h for p, h in self.files.items()):
            raise ValueError('Input identity changed.')
        if research.identity()['sha256'] != self.source['sha256']:
            raise ValueError('Diagnostic source changed.')


def candidate_records(lock, path_field, inputs):
    rows = read(inputs.add(lock))['candidates']
    if len(rows) != 6 or {r['id'] for r in rows} != set(IDS):
        raise ValueError('Require exactly the six locked candidates.')
    result = []
    for r in rows:
        if r['id'] != f"{r['model']}_{r['arm']}" or r['requested_step'] != 400000 or r['actual_step'] != 400000:
            raise ValueError('Candidate lock metadata mismatch.')
        path = inputs.add(r[path_field])
        if file_sha256(path) != r['sha256']:
            raise ValueError('Candidate SHA256 mismatch.')
        result.append(dict(r, path=str(path)))
    return sorted(result, key=lambda r: IDS.index(r['id']))


def candidates(lock, path_field, inputs):
    records = candidate_records(lock, path_field, inputs)
    for r in records:
        raw = torch.load(r['path'], map_location='cpu', weights_only=False)
        validate_snapshot(raw, expected_stage='easy', expected_model_type=r['model'])
        if any(raw[k] != r[k] for k in ('arm', 'requested_step', 'actual_step')):
            raise ValueError('Candidate snapshot disagrees with lock.')
    return records


def exclusions(args, pool, inputs):
    """Read EVERY reset log, including failed ACC jobs and all continuation stages."""
    keys = {offline.scene_key(s['payload']) for s in pool.scenarios}
    evidence = [dict(kind='historical_pool', path=str(args.validation_pool), count=len(pool.scenarios))]
    expected = [set(j['id'] for j in research.matrix()),
                {f'{m}_acc_{e}' for m in ('ann', 'snn') for e in ('001', '010')}]
    for root, wanted in zip((args.original_batch, args.acc_batch), expected):
        manifest = read(inputs.add(root / 'manifest.json'))
        jobs = [j['id'] for j in manifest['plan']['jobs']]
        if set(jobs) != wanted or len(jobs) != len(wanted):
            raise ValueError('Incomplete historical batch selection.')
        for job in jobs:
            path = inputs.add(root / job / 'other/training_scenes.jsonl')
            count, previous, unique, starts = 0, -1, set(), []
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    step = row['start_step']
                    if step <= previous or (count == 0 and step != 0):
                        raise ValueError('Incomplete/unordered training reset history.')
                    key = offline.scene_key(row['payload'])
                    if key != row['physical_sha256']:
                        raise ValueError('Training scene fingerprint mismatch.')
                    previous = step; count += 1; keys.add(key); unique.add(key)
                    starts.append(step)
            if not count:
                raise ValueError('Missing training scenes.')
            episodes = inputs.add(root / job / 'other/metrics_reports/episodes.jsonl')
            with episodes.open(encoding='utf-8') as stream:
                ends = [json.loads(line)['stage_steps'] for line in stream]
            if (not ends or any(b <= a for a,b in zip([0]+ends,ends))
                    or starts not in ([0]+ends,[0]+ends[:-1])):
                raise ValueError('Training reset log does not cover full episode history.')
            evidence.append(dict(kind='online_all_stages', job=job, path=str(path),
                count=count, unique=len(unique), first_start_step=0, last_start_step=previous,
                episode_history=str(episodes),completed_episodes=len(ends),
                coverage='entire append-only reset log, including failed and unfinished episodes',
                historical_source=manifest['preflight']['source']))
    # Explicit collect manifests bind full physical payloads to the prepared split.
    for model in ('ann', 'snn'):
        collect = inputs.add(args.offline_root / f'{model}_collect/other/manifest.json')
        prepared = inputs.add(args.offline_root / f'{model}_prepare/other/manifest.json')
        cm, pm = read(collect), read(prepared)
        selected = [s for s in cm['scenarios'] if s['partition'] in ('train', 'calibration')]
        if Counter(s['partition'] for s in selected) != {'train': 30, 'calibration': 10}:
            raise ValueError('Incomplete offline train/calibration scenes.')
        for s in selected:
            sid = s['scenario_id']; key = offline.scene_key(s['payload'])
            if sid not in pm['split'][s['partition']] or pm['scene_keys'][sid] != key or cm['scene_keys'][sid] != key:
                raise ValueError('Offline collection/preparation identity mismatch.')
            keys.add(key)
        evidence.append(dict(kind='offline_train_calibration', model=model,
            collection=str(collect), preparation=str(prepared), count=len(selected)))
    return keys, evidence


def check_scenes(scenes, excluded):
    keys = [offline.scene_key(s['payload']) for s in scenes]
    empty = sum(not s['payload']['zones'] for s in scenes)
    shapes = {z['shape']['shape_type'] for s in scenes for z in s['payload']['zones']}
    if (len(scenes) != 40 or empty != 8 or len(set(keys)) != 40
            or set(keys) & excluded or not SHAPES <= shapes):
        raise ValueError('Independent scene count/strata/uniqueness/coverage/leakage failure.')


def select_scenes(data, excluded):
    seed = derive_validation_stage_seed(SEED, 'easy')
    generator = offline.V2ScenarioGenerator(data['scenario'], 'easy', seed=seed)
    scenes, used, counts = [], set(excluded), Counter()
    for draw in range(100000):
        payload = generator.generate(); key = offline.scene_key(payload)
        empty = not payload['zones']
        if key in used or counts[empty] >= (8 if empty else 32):
            continue
        used.add(key); counts[empty] += 1
        scenes.append(dict(scenario_id=f'confirmation_{len(scenes):02d}', sequence_index=len(scenes),
            scenario_seed=payload['metadata']['scenario_seed'], payload=payload))
        if len(scenes) == 40:
            break
    # Never replace a scene for coverage, flight, Q or G after selection.
    check_scenes(scenes, excluded)
    return V2ValidationPool('easy', SEED, seed, data['pool'].scenario_config,
        data['pool'].uav_collision_radius, scenes)


def frozen_identity(data):
    models = [data['actor'], *data['critics']]
    if any(m.training or any(p.requires_grad for p in m.parameters()) for m in models):
        raise ValueError('Evaluation networks must be frozen/eval.')
    return cal.state_hash([m.state_dict() for m in models])


def complete_episode(record):
    n = record['episode_length']
    if (n <= 0 or len(record['actions']) != n or len(record['rewards']) != n
            or len(record['pre_action_hashes']) != n or len(record['states']) != n + 1
            or not (record['terminated'] or record['truncated'])):
        raise ValueError('Incomplete natural episode.')
    cal.finite_tree(record)


def episode(data, scene, path):
    """Reuse only a complete, strict-replayable record; never repair or retry."""
    if path.exists():
        record = read(path)
    else:
        env = cal.base.make_env(data)
        obs, info = env.reset(options={'scenario': scene['payload']})
        record = dict(cal.base.fly(data, env, obs, initial_info=info, record_probe_hashes=True),
            scenario_id=scene['scenario_id'], empty_scene=not scene['payload']['zones'])
        _write_json(path, record)
    complete_episode(record)
    if record['scenario_id'] != scene['scenario_id'] or record['empty_scene'] != (not scene['payload']['zones']):
        raise ValueError('Episode scene identity mismatch.')
    return record, cal.replay_episode(data, scene, record)


def validate_loaded(data, candidate):
    if (data['checkpoint_hash'] != candidate['sha256'] or data['model'] != candidate['model']
            or data['pool'].content_digest != research.DIGEST):
        raise ValueError('Strict historical checkpoint/pool identity mismatch.')


def flight_metrics(data, scenes, directory):
    rows, qs, records, scene_info = [], [], [], {}
    for s in scenes:
        record, transitions = episode(data, s, directory / f"{s['scenario_id']}.json")
        records.append(record)
        scene_info[s['scenario_id']] = dict(outcome=record['outcome'],
            shape_types=sorted({z['shape']['shape_type'] for z in s['payload']['zones']}))
        for row in transitions:
            row.update(outcome=record['outcome'], remaining_steps=len(transitions)-row['step'])
            q = cal.base.score(data, row['observation'], row['action'])
            qs.append([q['Q1'], q['Q2']]); rows.append(row)
    result = metrics.grouped_statistics(rows, np.asarray(qs), scene_info)
    result['note'] = 'independent stratified 8/32; overlapping subgroups; no method pass threshold'
    result['flights'] = [dict(scenario_id=r['scenario_id'], outcome=r['outcome'],
        length=r['episode_length'], episode_return=r['episode_return'], G0=r['discounted_return']) for r in records]
    result['outcomes'] = dict(Counter(r['outcome'] for r in records))
    result['goal_rate'] = sum(r['outcome'] == 'goal' for r in records) / len(records)
    p = offline.generation_config(data)['p_empty']
    result['source_p_empty'] = p
    result['descriptive_reweighted'] = {}
    for c in ('Q1', 'Q2'):
        a, b = [result['groups'][k]['scene_equal'][c] for k in ('empty', 'nonempty')]
        result['descriptive_reweighted'][c] = {k:p*a[k]+(1-p)*b[k] for k in ('bias','MAE','positive_error_mean')}
        result['descriptive_reweighted'][c]['RMSE'] = float(np.sqrt(p*a['RMSE']**2+(1-p)*b['RMSE']**2))
    result['descriptive_reweighted_goal_rate'] = sum(weight * np.mean([
        r['outcome']=='goal' for r in records if r['empty_scene']==empty])
        for empty,weight in ((True,p),(False,1-p)))
    result['transition_scores'] = [dict(scenario_id=r['scenario_id'], step=r['step'],
        reward=r['reward'], done=r['done'], G=r['G'], Q1=q[0], Q2=q[1]) for r,q in zip(rows,qs)]
    cal.finite_tree(result)
    return result


def actions(bc, own, limit):
    proposals = [('BC', np.asarray(bc, dtype=np.float32))]
    for dim in range(2):
        for sign in (-1, 1):
            a = proposals[0][1].copy(); a[dim] += sign*.2*limit[dim]
            proposals.append((f'BC_{dim}_{sign:+d}', a))
    proposals.append(('own', own))
    unique, sources = [], {}
    for name, value in proposals:
        a = np.clip(value, -limit, limit).astype(np.float32)
        cal.base.finite(a, 'candidate action')
        if a.shape != (2,): raise ValueError('Expected two controls.')
        index = next((i for i,b in enumerate(unique) if np.array_equal(a,b)), len(unique))
        if index == len(unique): unique.append(a)
        sources[name] = index
    return unique, sources


def rankings(rows, sources):
    common = sorted({v for k,v in sources.items() if k != 'own'})
    result = {}
    for c in ('Q1','Q2'):
        view = [dict(r, Q1=r[c]) for r in rows]
        # ranking takes a POSITION for its reference, candidate IDs stay global.
        result[c] = dict(common=cal.base.ranking([view[i] for i in common], common.index(sources['BC'])),
            including_own=cal.base.ranking(view, sources['own']),
            common_reference='BC', own_reference='own',
            delta_Q_common=[r[c]-rows[sources['BC']][c] for r in rows],
            delta_Q_own=[r[c]-rows[sources['own']][c] for r in rows])
    return result


def bc_data(data, model, path, *, seed=23):
    from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig, prepare_v2_stage_initialization
    p = prepare_v2_stage_initialization(V2FormalTrainingConfig(stage='easy', seed=seed),
        init_checkpoint=path, device=data['device'], model_type=model)
    return dict(data, actor=p.bc_initialization.actor.to(data['device']).eval().requires_grad_(False))


def anchors(data, directory, supplied, inputs):
    selected = cal.base.select_scenarios(data['pool'])[:12]
    identity = dict(actor=cal.state_hash(data['actor'].state_dict()), historical_pool=data['pool'].content_digest,
        backend=offline.execution_backend(data), scenario_ids=[s['scenario_id'] for s in selected])
    if supplied:
        saved = read(inputs.add(supplied))
        if saved['identity'] != identity or len(saved['episodes']) != 12:
            raise ValueError('BC anchor identity/backend/count mismatch.')
        records = saved['episodes']
    else:
        records = [episode(data,s,directory/f"{s['scenario_id']}.json")[0] for s in selected]
    plans = []
    for s,r in zip(selected,records):
        complete_episode(r)
        if r['scenario_id'] != s['scenario_id']: raise ValueError('BC anchor order mismatch.')
        # Strict linear replay plus actor ownership; never relax backend agreement.
        offline.verify_record(data,s,r)
        index = (r['episode_length']-1)//2
        plans.append(dict(scene=s,prefix=r['actions'][:index],hash=r['pre_action_hashes'][index]))
    _write_json(directory/'anchors.json',dict(identity=identity,episodes=records))
    return plans


def seal_probe_plans(data, bc, plans, directory):
    sealed = []
    for index,p in enumerate(plans):
        env = cal.base.make_env(data)
        obs,_ = cal.base.reset_replay(env,p['scene'],p['prefix'],p['hash'])
        candidates_, sources = actions(cal.base.actor_action(bc,obs),cal.base.actor_action(data,obs),
            data['actor'].action_limit.detach().cpu().numpy())
        # Seal all actions for this immutable shared prefix before any branch.
        _write_json(directory/f'plan_{index:02d}.json',dict(prefix=p,sources=sources,
            actions=[a.tolist() for a in candidates_]))
        sealed.append((p,candidates_,sources))
    return sealed


def run_probe_branches(data, sealed, directory):
    result = []
    for index,(p,candidates_,sources) in enumerate(sealed):
        env = cal.base.make_env(data)
        rows = []
        for i,a in enumerate(candidates_):
            obs,state = cal.base.reset_replay(env,p['scene'],p['prefix'],p['hash'])
            q = cal.base.score(data,obs,a)
            branch = cal.base.fly(data,env,obs,first_action=a,record_probe_hashes=True,initial_info=state['info'])
            complete_episode(branch)
            _write_json(directory/f'branch_{index:02d}_{i}.json',branch)
            rows.append(dict(q,candidate_index=i,action=a.tolist(),G=branch['discounted_return'],
                outcome=branch['outcome'],remaining_initial_steps=data['scenario'].max_steps-len(p['prefix']),
                consumed_initial_steps=len(p['prefix']),
                Q1_minus_G=q['Q1']-branch['discounted_return'],Q2_minus_G=q['Q2']-branch['discounted_return']))
        result.append(dict(scenario_id=p['scene']['scenario_id'],state_hash=p['hash'],
            sources=sources,rows=rows,rankings=rankings(rows,sources)))
    return result


def probe(data, bc, plans, directory):
    return run_probe_branches(data,seal_probe_plans(data,bc,plans,directory),directory)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('candidate-lock','completion-audit','original-batch','acc-batch','offline-root',
                'validation-pool','ann-bc','snn-bc','output-dir','historical-repo'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--checkpoint-path-field',choices=('local_checkpoint','server_checkpoint'),default='local_checkpoint')
    p.add_argument('--ann-anchors',type=Path); p.add_argument('--snn-anchors',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--source-manifest',type=Path,default=research.REPO/'scripts/v2_online_wd3_acc_source.json')
    p.add_argument('--launch',action='store_true')
    return p


def plan(args):
    for k,v in vars(args).items():
        if isinstance(v,Path) and not v.is_absolute(): raise ValueError(f'{k} must be absolute.')
    return dict(diagnostic_only=True,formal_passed=False,launch=args.launch,candidates=IDS,
        independent_flights=240,optional_BC_anchor_flights=24,branch_limit=432,
        selection_seed=SEED,scenes=40,empty=8,nonempty=32,device=args.device,
        checkpoint_path_field=args.checkpoint_path_field,
        inputs={k:str(v) for k,v in vars(args).items() if isinstance(v,Path)},
        note='execution completion is not algorithm acceptance; no new seeds or training')


def execute(args):
    spec = plan(args)
    if not args.launch: return spec
    out = args.output_dir.resolve(); old = args.historical_repo.resolve()
    if research.REPO.resolve().is_relative_to(old) or out.is_relative_to(old) or any(out.is_relative_to(p.resolve())
            for p in (args.original_batch,args.acc_batch,args.offline_root)):
        raise ValueError('Independent code/output directories required.')
    if out.exists():
        return reuse(out,spec)
    out.mkdir(parents=True,exist_ok=False)
    review,other = out/'review',out/'other'; review.mkdir(); other.mkdir()
    result = dict(diagnostic_only=True,formal_passed=False,execution_completed=False)
    started = time.perf_counter()
    try:
        inputs = Inputs()
        if read(inputs.add(args.source_manifest))['sha256'] != inputs.source['sha256']:
            raise ValueError('Reviewed source manifest mismatch.')
        locked = candidates(args.candidate_lock,args.checkpoint_path_field,inputs)
        audit = read(inputs.add(args.completion_audit))
        for c in locked:
            matches = [j for j in audit['jobs'] if j['job'] == c['id']]
            if len(matches) != 1 or not matches[0]['execution_completed'] or matches[0]['stage_steps'] != 400000:
                raise ValueError('Candidate completion audit mismatch.')
        inputs.add(args.validation_pool)
        for m in ('ann','snn'):
            if file_sha256(inputs.add(getattr(args,m+'_bc'))) != research.BC[m][1]:
                raise ValueError('BC best SHA256 mismatch.')
            anchor = getattr(args,m+'_anchors')
            if anchor: inputs.add(anchor)
        data = cal.base.load_models(locked[0]['path'],args.validation_pool,args.device)
        validate_loaded(data,locked[0])
        excluded,evidence = exclusions(args,data['pool'],inputs)
        # Input identity and ALL exclusions are sealed BEFORE generating any scene.
        _write_json(other/'excluded_scene_keys.json',sorted(excluded))
        _write_json(review/'manifest.json',dict(plan=spec,candidates=locked,inputs=inputs.files.copy(),
            diagnostic_source=inputs.source,exclusions=evidence,excluded_unique=len(excluded),
            excluded_keys=dict(path='other/excluded_scene_keys.json',sha256=file_sha256(other/'excluded_scene_keys.json'))))
        pool = select_scenes(data,excluded)
        save_v2_validation_pool(other/'independent_pool.json',pool)
        pool_path = inputs.add(other/'independent_pool.json')
        pool = load_v2_validation_pool(pool_path,expected_level='easy',expected_count=40,
            expected_scenario=data['scenario'],expected_uav_collision_radius=data['pool'].uav_collision_radius)
        _write_json(review/'scene_plan.json',dict(path=str(pool_path),sha256=file_sha256(pool_path),
            content_digest=pool.content_digest,scene_keys=[offline.scene_key(s['payload']) for s in pool.scenarios],
            historical_pool=dict(path=str(args.validation_pool),sha256=file_sha256(args.validation_pool),digest=research.DIGEST)))
        shared = {}; bc_models = {}
        for m in ('ann','snn'):
            c = next(c for c in locked if c['model']==m)
            d = cal.base.load_models(c['path'],args.validation_pool,args.device); validate_loaded(d,c)
            bc = bc_data(d,m,getattr(args,m+'_bc')); before = frozen_identity(bc)
            directory = other/f'BC_{m}'; directory.mkdir()
            shared[m] = anchors(bc,directory,getattr(args,m+'_anchors'),inputs)
            if frozen_identity(bc) != before: raise ValueError('Frozen BC changed.')
            bc_models[m] = bc
        _write_json(review/'probe_plan.json',shared)
        _write_json(review/'sealed_inputs.json',inputs.files)
        summaries = {}
        for c in locked:
            candidate_started = time.perf_counter()
            d = cal.base.load_models(c['path'],args.validation_pool,args.device); validate_loaded(d,c)
            before = frozen_identity(d)
            directory = other/c['id']; directory.mkdir()
            flights = directory/'independent'; flights.mkdir()
            stats = flight_metrics(d,pool.scenarios,flights)
            _write_json(directory/'independent_scores.json',stats)
            probes = directory/'development_probes'; probes.mkdir()
            pr = probe(d,bc_models[c['model']],shared[c['model']],probes)
            _write_json(directory/'probe_results.json',pr)
            if frozen_identity(d) != before: raise ValueError('Frozen candidate changed.')
            inputs.check()
            summaries[c['id']] = dict(goal_rate=stats['goal_rate'],outcomes=stats['outcomes'],
                per_scene=stats['flights'],frozen_unchanged=True,checkpoint_sha256=d['checkpoint_hash'],
                probe_count=len(pr),branches=sum(len(p['rows']) for p in pr),
                process_wall_seconds=time.perf_counter()-candidate_started)
        inputs.check()
        result.update(results=summaries,inputs=inputs.files,
            diagnostic_source=inputs.source,independent_pool_digest=pool.content_digest)
        # Every evidence file is immutable on reuse, including all flight/branch records.
        result['artifacts'] = {p.relative_to(out).as_posix():file_sha256(p) for p in out.rglob('*') if p.is_file()}
        validate_bundle(out,result)
        inputs.check()
        result['execution_completed'] = True
    except BaseException as exc:
        result['execution_completed'] = False
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['process_wall_seconds'] = time.perf_counter()-started
        _write_json(review/'summary.json',result)
    return result


def reuse(out,expected_plan=None):
    if expected_plan is not None and read(out/'review/manifest.json')['plan'] != json.loads(json.dumps(expected_plan)):
        raise ValueError('Reuse request differs from sealed plan.')
    result = read(out/'review/summary.json')
    if not result['execution_completed'] or set(result['results']) != set(IDS):
        raise ValueError('Incomplete output; no automatic repair/retry.')
    if 'error' in result:
        raise ValueError('Completed output cannot contain an error.')
    validate_bundle(out,result)
    return result


def checked_episode(record, gamma, max_steps, consumed=0):
    """Read-only structural/return checks; no environment reset or model forward."""
    complete_episode(record)
    n = record['episode_length']
    if (np.asarray(record['actions']).shape != (n,2) or np.asarray(record['states']).shape != (n+1,5)
            or any(not isinstance(h,str) or len(h)!=64 for h in record['pre_action_hashes'])
            or record['final_steps'] != consumed+n or record['final_steps'] > max_steps
            or record['outcome'] not in cal.base.OUTCOMES
            or record['final_info']['outcome'] != record['outcome']
            or record['episode_return'] != math.fsum(record['rewards'])
            or record['discounted_return'] != cal.base.discounted_return(record['rewards'],gamma)):
        raise ValueError('Incomplete/mismatched episode evidence.')


def validate_bundle(out, result):
    """Required evidence is derived from inputs/plans, never from summary counts.

    Used before first completion and on reuse. Reads JSON/pools/hashes only;
    deliberately does not load checkpoints/models or replay/fly trajectories.
    """
    artifacts = result['artifacts']
    if not artifacts or not result['inputs'] or set(result['results']) != set(IDS):
        raise ValueError('Empty/incomplete evidence inventory.')
    required = set()
    def get(name):
        required.add(name)
        if name not in artifacts or file_sha256(out/name) != artifacts[name]:
            raise ValueError(f'Required artifact missing/unsealed/changed: {name}')
        value = read(out/name); cal.finite_tree(value)
        return value

    manifest = get('review/manifest.json'); spec = manifest['plan']
    args = SimpleNamespace(**{k:Path(v) for k,v in spec['inputs'].items()})
    args.checkpoint_path_field = spec['checkpoint_path_field']
    tracked = Inputs()
    source = read(tracked.add(args.source_manifest))
    if (source['sha256'] != tracked.source['sha256'] or not source['sha256']
            or manifest['diagnostic_source'] != result['diagnostic_source']
            or result['diagnostic_source']['sha256'] != source['sha256']):
        raise ValueError('Source identity mismatch.')
    locked = candidate_records(args.candidate_lock,args.checkpoint_path_field,tracked)
    if locked != manifest['candidates']:
        raise ValueError('Sealed candidates differ from lock.')
    audit = read(tracked.add(args.completion_audit))
    for c in locked:
        matches = [j for j in audit['jobs'] if j['job']==c['id']]
        if len(matches)!=1 or matches[0]['execution_completed'] is not True or matches[0]['stage_steps']!=400000:
            raise ValueError('Candidate completion audit mismatch.')
    historical = load_v2_validation_pool(tracked.add(args.validation_pool),expected_level='easy',expected_count=100)
    if historical.content_digest != research.DIGEST:
        raise ValueError('Historical pool identity mismatch.')
    for m in ('ann','snn'):
        if file_sha256(tracked.add(getattr(args,m+'_bc'))) != research.BC[m][1]:
            raise ValueError('BC best identity mismatch.')
        anchor = getattr(args,m+'_anchors',None)
        if anchor: tracked.add(anchor)
    excluded,evidence = exclusions(args,historical,tracked)
    if manifest['inputs'] != tracked.files or manifest['exclusions'] != evidence:
        raise ValueError('Required historical input inventory differs from seal.')
    validate_evaluation_artifacts(out,result,get,required,manifest,args,tracked,locked,historical,excluded)


def validate_evaluation_artifacts(out,result,get,required,manifest,args,tracked,locked,historical,excluded,
        *,models=('ann','snn')):
    """Shared JSON-only flight/probe evidence checks; input contracts stay in each entry."""
    artifacts=result['artifacts']
    keys = get('other/excluded_scene_keys.json')
    if (keys != sorted(excluded) or manifest['excluded_unique']!=len(excluded)
            or manifest['excluded_keys'] != dict(path='other/excluded_scene_keys.json',
                sha256=artifacts['other/excluded_scene_keys.json'])):
        raise ValueError('Exclusion seal mismatch.')
    get('other/independent_pool.json')
    pool_path = tracked.add(out/'other/independent_pool.json')
    independent = load_v2_validation_pool(pool_path,expected_level='easy',expected_count=40)
    if (independent.scenario_config != historical.scenario_config
            or independent.uav_collision_radius != historical.uav_collision_radius
            or independent.master_seed != SEED):
        raise ValueError('Independent pool configuration mismatch.')
    check_scenes(independent.scenarios,excluded)
    scene_plan = get('review/scene_plan.json')
    if scene_plan != dict(path=str(pool_path),sha256=file_sha256(pool_path),content_digest=independent.content_digest,
            scene_keys=[offline.scene_key(s['payload']) for s in independent.scenarios],
            historical_pool=dict(path=str(args.validation_pool),sha256=file_sha256(args.validation_pool),digest=research.DIGEST)):
        raise ValueError('Two-pool scene plan mismatch.')
    if result['independent_pool_digest'] != independent.content_digest:
        raise ValueError('Independent pool summary mismatch.')
    if get('review/sealed_inputs.json') != tracked.files or result['inputs'] != tracked.files:
        raise ValueError('Required input inventory missing/changed.')
    selected = cal.base.select_scenarios(historical)[:12]
    shared = get('review/probe_plan.json')
    if set(shared) != set(models):
        raise ValueError('Missing model probe plan.')
    cfg = historical.scenario_config
    max_steps = cfg['max_steps']; limit = np.array([cfg['delta_gamma_max'],cfg['delta_psi_max']],dtype=np.float32)
    gamma = research.matrix()[0]['gamma']
    for m in models:
        anchor = get(f'other/BC_{m}/anchors.json'); identity = anchor['identity']
        if (identity['historical_pool'] != historical.content_digest or not identity['backend']
                or len(identity['actor'])!=64 or identity['scenario_ids'] != [s['scenario_id'] for s in selected]
                or len(anchor['episodes'])!=12 or len(shared[m])!=12):
            raise ValueError('Incomplete BC anchor identity.')
        supplied = getattr(args,m+'_anchors',None)
        if supplied and read(supplied) != anchor:
            raise ValueError('Reused BC anchor differs from sealed input.')
        for s,r,p in zip(selected,anchor['episodes'],shared[m]):
            checked_episode(r,gamma,max_steps)
            mid = (r['episode_length']-1)//2
            if (r['scenario_id']!=s['scenario_id'] or r['empty_scene']!=(not s['payload']['zones'])
                    or p != dict(scene=s,prefix=r['actions'][:mid],hash=r['pre_action_hashes'][mid])):
                raise ValueError('BC anchor/shared state mismatch.')
    scene_ids = [s['scenario_id'] for s in independent.scenarios]
    if len(set(scene_ids)) != 40: raise ValueError('Duplicate independent scene IDs.')
    common_actions = {}
    for c in locked:
        job,m = c['id'],c['model']; base = f'other/{job}'
        stats = get(base+'/independent_scores.json'); summary = result['results'][job]
        if (set(stats['per_scene'])!=set(scene_ids) or len(stats['flights'])!=40
                or [r['scenario_id'] for r in stats['flights']] != scene_ids
                or summary['per_scene']!=stats['flights'] or summary['checkpoint_sha256']!=c['sha256']
                or summary['frozen_unchanged'] is not True):
            raise ValueError('Incomplete/mismatched independent scores.')
        scores = stats['transition_scores']; offset = 0; outcomes = Counter()
        for s,f in zip(independent.scenarios,stats['flights']):
            sid = s['scenario_id']; r = get(base+f'/independent/{sid}.json')
            checked_episode(r,gamma,max_steps); n = r['episode_length']
            if (r['scenario_id']!=sid or r['empty_scene']!=(not s['payload']['zones'])
                    or f != dict(scenario_id=sid,outcome=r['outcome'],length=n,
                        episode_return=r['episode_return'],G0=r['discounted_return'])
                    or stats['per_scene'][sid]['count']!=n
                    or not {'Q1','Q2'} <= set(stats['per_scene'][sid])):
                raise ValueError('Flight/score scene mismatch.')
            gs = cal.returns(r['rewards'],[False]*(n-1)+[True],gamma)
            for i,g in enumerate(gs):
                row = scores[offset+i]
                if (row['scenario_id']!=sid or row['step']!=i or row['G']!=g
                        or row['reward']!=r['rewards'][i] or row['done']!=(i==n-1)
                        or not {'Q1','Q2'} <= set(row)):
                    raise ValueError('Transition score missing/duplicated/mismatched.')
            expected_scores = metrics.statistics([dict(G=g) for g in gs],np.asarray([
                [r['Q1'],r['Q2']] for r in scores[offset:offset+n]]))
            if stats['per_scene'][sid] != expected_scores:
                raise ValueError('Per-scene scoring statistics incomplete/mismatched.')
            offset += n; outcomes[r['outcome']] += 1
        rate = outcomes['goal']/40
        if (offset!=len(scores) or stats['outcomes']!=dict(outcomes) or summary['outcomes']!=dict(outcomes)
                or stats['goal_rate']!=rate or summary['goal_rate']!=rate):
            raise ValueError('Flight summary mismatch.')
        required_groups = {'all','empty','nonempty','terminal','nonterminal','failure_terminal','failure_last5'} | SHAPES | set(cal.base.OUTCOMES)
        if (not required_groups <= stats['groups'].keys()
                or stats['groups']['all']['transition_weighted']['count']!=len(scores)
                or stats['groups']['all']['scenes']!=40):
            raise ValueError('Missing grouped scoring results.')
        probes = get(base+'/probe_results.json')
        if len(probes)!=12: raise ValueError('Incomplete probes.')
        branches = 0
        for index,(p,pr) in enumerate(zip(shared[m],probes)):
            directory = base+'/development_probes'
            ap = get(directory+f'/plan_{index:02d}.json'); aa,ss = ap['actions'],ap['sources']
            if ap['prefix']!=p or not 1<=len(aa)<=6:
                raise ValueError('Action plan/prefix mismatch.')
            rebuilt,mapping = actions(aa[ss['BC']],aa[ss['own']],limit)
            if mapping!=ss or not np.array_equal(np.asarray(aa,dtype=np.float32),rebuilt):
                raise ValueError('Action plan is not clipped/deduplicated.')
            common = [aa[i] for i in sorted({v for k,v in ss.items() if k!='own'})]
            key = (m,index)
            if key in common_actions and common_actions[key]!=common:
                raise ValueError('Same-model public actions differ.')
            common_actions[key] = common
            rows = pr['rows']
            if (pr['scenario_id']!=p['scene']['scenario_id'] or pr['state_hash']!=p['hash'] or pr['sources']!=ss
                    or len(rows)!=len(aa) or [r['candidate_index'] for r in rows]!=list(range(len(aa)))
                    or pr['rankings']!=rankings(rows,ss)):
                raise ValueError('Probe ID/action/ranking mismatch.')
            consumed = len(p['prefix'])
            for i,row in enumerate(rows):
                branch = get(directory+f'/branch_{index:02d}_{i}.json')
                checked_episode(branch,gamma,max_steps,consumed)
                if (row['action']!=aa[i] or branch['actions'][0]!=aa[i]
                        or branch['pre_action_hashes'][0]!=p['hash']
                        or row['G']!=branch['discounted_return'] or row['outcome']!=branch['outcome']
                        or row['remaining_initial_steps']!=max_steps-consumed or row['consumed_initial_steps']!=consumed
                        or row['Q1_minus_G']!=row['Q1']-row['G'] or row['Q2_minus_G']!=row['Q2']-row['G']):
                    raise ValueError('Branch/action/remaining-time evidence mismatch.')
            branches += len(aa)
        if summary['branches']!=branches or summary['probe_count']!=12:
            raise ValueError('Probe summary mismatch.')
    # No required evidence may be omitted from the claimed hash inventory.
    if not required <= artifacts.keys(): raise ValueError('Missing required artifact hashes.')
    actual = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p!=out/'review/summary.json'}
    if actual != set(artifacts): raise ValueError('Artifact inventory differs from actual files.')
    for name,h in artifacts.items():
        if file_sha256(out/name)!=h: raise ValueError('Artifact changed after seal.')
    tracked.check()


def main(argv=None):
    result = execute(parser().parse_args(argv))
    print(json.dumps(cal.base.serial(result),allow_nan=False,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
