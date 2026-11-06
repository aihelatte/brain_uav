"""Seed7 SNN final400k only; reuse sealed scenes, never generate or train."""
import argparse
import json
from pathlib import Path
import time

import torch

from brain_uav.scripts import diagnose_v2_online_confirmation as base

IDS=('snn_min','snn_fixed_own','snn_acc_001')
POOL_HASH='f64082daec0e15cd02e3604b66bf23ecf633a8a6fc12142b1ad67e1dcac8c93e'
POOL_DIGEST='27a6e7066272e451cf9483c73ef8bfa6358ecefa583020291f5dea47fac86316'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-lock','batch','sealed-confirmation','validation-pool','snn-bc','output-dir','historical-repo'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--snn-anchors',type=Path)
    p.add_argument('--checkpoint-path-field',choices=('local_checkpoint','server_checkpoint'),default='local_checkpoint')
    p.add_argument('--source-manifest',type=Path,default=base.research.REPO/'scripts/v2_online_wd3_acc_source.json')
    p.add_argument('--launch',action='store_true')
    return p


def plan(a):
    for k,v in vars(a).items():
        if isinstance(v,Path) and not v.is_absolute():raise ValueError(f'{k} must be absolute.')
    return dict(contract='seed7_snn_final400k_v1',diagnostic_only=True,formal_passed=False,
        launch=a.launch,candidates=list(IDS),seed=7,device='cuda',independent_flights=120,
        optional_BC_anchor_flights=12,branch_limit=216,checkpoint_path_field=a.checkpoint_path_field,
        inputs={k:str(v) for k,v in vars(a).items() if isinstance(v,Path)})


def candidate_inputs(a,inputs,*,load):
    rows=base.read(inputs.add(a.candidate_lock))['candidates']
    if len(rows)!=3 or {r['id'] for r in rows}!=set(IDS):raise ValueError('Require locked seed7 three candidates.')
    batch=base.read(inputs.add(a.batch/'manifest.json'))
    jobs=batch['plan']['jobs']
    if len(jobs)!=3 or {j['id'] for j in jobs}!=set(IDS) or any(j['seed']!=7 for j in jobs):
        raise ValueError('Wrong seed7 batch identity.')
    result=[]
    for r in rows:
        if (r['model']!='snn' or r['seed']!=7 or r['id']!='snn_'+r['arm']
                or r['requested_step']!=400000 or r['actual_step']!=400000):raise ValueError('Wrong lock metadata.')
        path=inputs.add(r[a.checkpoint_path_field])
        if base.file_sha256(path)!=r['sha256']:raise ValueError('Checkpoint hash mismatch.')
        if path.parent.parent.resolve()!=(a.batch/r['id']).resolve():raise ValueError('Checkpoint outside declared job.')
        directory=a.batch/r['id']/'review'
        summary,progress,process,sidecar=[base.read(inputs.add(directory/(n+'.json'))) for n in
            ('summary','progress','process','snapshot_400000_400000')]
        job=next(j for j in jobs if j['id']==r['id'])
        if (summary['job']!=job or summary['execution_completed'] is not True or summary['actual_step']!=400000
                or progress['state']!='completed_budget' or progress['stage_steps']!=400000
                or progress['arm']!=r['arm'] or process['exit_code']!=0
                or sidecar['sha256']!=r['sha256'] or sidecar['requested_step']!=400000 or sidecar['actual_step']!=400000
                or sidecar['pool_digest']!=base.research.DIGEST or sidecar['BC_sha256']!=base.research.BC['snn'][1]):
            raise ValueError('Final summary/progress/process/sidecar mismatch.')
        if load:
            raw=torch.load(path,map_location='cpu',weights_only=False)
            base.validate_snapshot(raw,expected_stage='easy',expected_model_type='snn')
            if (raw['arm']!=r['arm'] or raw['requested_step']!=400000 or raw['actual_step']!=400000
                    or raw['snapshot']['formal_config']['seed']!=7):raise ValueError('Snapshot seed/arm/step mismatch.')
        result.append(dict(r,path=str(path)))
    return sorted(result,key=lambda r:IDS.index(r['id']))


def provenance(a,inputs):
    locked=candidate_inputs(a,inputs,load=False)
    if base.read(inputs.add(a.source_manifest))['sha256']!=inputs.source['sha256']:raise ValueError('Source mismatch.')
    historical=base.load_v2_validation_pool(inputs.add(a.validation_pool),expected_level='easy',expected_count=100)
    if historical.content_digest!=base.research.DIGEST:raise ValueError('Historical pool mismatch.')
    if base.file_sha256(inputs.add(a.snn_bc))!=base.research.BC['snn'][1]:raise ValueError('BC identity mismatch.')
    if a.snn_anchors:inputs.add(a.snn_anchors)
    prior=a.sealed_confirmation
    summary=base.read(inputs.add(prior/'review/summary.json'))
    manifest=base.read(inputs.add(prior/'review/manifest.json'))
    # Verify every previously sealed artifact, including all exclusion evidence files.
    if summary['execution_completed'] is not True or summary.get('error') or not summary['artifacts']:
        raise ValueError('Prior confirmation incomplete.')
    for name,h in summary['artifacts'].items():
        p=inputs.add(prior/name)
        if base.file_sha256(p)!=h:raise ValueError('Prior sealed artifact changed.')
    for name in ('review/manifest.json','review/scene_plan.json','other/excluded_scene_keys.json','other/independent_pool.json'):
        if name not in summary['artifacts']:raise ValueError('Prior exclusion/pool seal missing.')
    excluded=set(base.read(inputs.add(prior/'other/excluded_scene_keys.json')))
    sources=manifest['exclusions']
    online=[s for s in sources if s['kind']=='online_all_stages']
    if (len(online)!=20 or {s['job'] for s in online}!={j['id'] for j in base.research.matrix()}
            or len([s for s in sources if s['kind']=='offline_train_calibration' and s['count']==40])!=2
            or any(s['count']<=0 or s['first_start_step']!=0 for s in online)):
        raise ValueError('Prior exclusion coverage incomplete.')
    if len(excluded)!=manifest['excluded_unique'] or not excluded:raise ValueError('Prior exclusion count mismatch.')
    if manifest['excluded_keys']['sha256']!=base.file_sha256(prior/'other/excluded_scene_keys.json'):
        raise ValueError('Prior exclusion fingerprint seal mismatch.')
    pool_path=inputs.add(prior/'other/independent_pool.json')
    if base.file_sha256(pool_path)!=POOL_HASH:raise ValueError('Sealed 40 pool raw hash mismatch.')
    pool=base.load_v2_validation_pool(pool_path,expected_level='easy',expected_count=40)
    scene_plan=base.read(prior/'review/scene_plan.json')
    if pool.content_digest!=POOL_DIGEST or scene_plan['content_digest']!=POOL_DIGEST:raise ValueError('Sealed pool digest mismatch.')
    if pool.scenario_config!=historical.scenario_config or pool.uav_collision_radius!=historical.uav_collision_radius:
        raise ValueError('Two-pool configuration mismatch.')
    evidence=training_exclusions(a.batch,inputs,excluded)
    base.check_scenes(pool.scenarios,excluded)
    return locked,historical,pool,excluded,evidence


def training_exclusions(batch,inputs,excluded):
    """All three reset histories, including their unfinished final episodes."""
    evidence=[]
    for job in IDS:
        path=inputs.add(batch/job/'other/training_scenes.jsonl')
        ep=inputs.add(batch/job/'other/metrics_reports/episodes.jsonl')
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        ends=[json.loads(line)['stage_steps'] for line in ep.read_text(encoding='utf-8').splitlines()]
        summary_path=inputs.add(batch/job/'review/summary.json')
        summary=base.read(summary_path)
        reported=[r['stage_steps'] for r in summary['episodes']]
        starts=[r['start_step'] for r in rows]
        expected_starts=([0]+ends[:-1]) if ends and ends[-1]==400000 else ([0]+ends)
        if (summary['execution_completed'] is not True or summary['actual_step']!=400000
                or not ends or ends!=reported or any(type(b) is not int or b>400000 or b<=c for c,b in zip([0]+ends,ends))
                or starts!=expected_starts):
            raise ValueError('Seed7 reset log incomplete.')
        for r in rows:
            key=base.offline.scene_key(r['payload'])
            if key!=r['physical_sha256']:raise ValueError('Seed7 scene fingerprint mismatch.')
            excluded.add(key)
        evidence.append(dict(job=job,path=str(path),episode_history=str(ep),summary=str(summary_path),
            count=len(rows),last_start_step=starts[-1]))
    return evidence


def validate_bundle(out,result):
    if not result['inputs'] or not result['artifacts'] or set(result['results'])!=set(IDS):raise ValueError('Incomplete bundle.')
    required=set();artifacts=result['artifacts']
    def get(name):
        required.add(name)
        if name not in artifacts or base.file_sha256(out/name)!=artifacts[name]:raise ValueError('Missing/unsealed artifact: '+name)
        value=base.read(out/name);base.cal.finite_tree(value);return value
    manifest=get('review/manifest.json');spec=manifest['plan']
    a=argparse.Namespace(**{k:Path(v) for k,v in spec['inputs'].items()})
    a.checkpoint_path_field=spec['checkpoint_path_field'];a.snn_anchors=getattr(a,'snn_anchors',None)
    inputs=base.Inputs();locked,historical,pool,excluded,evidence=provenance(a,inputs)
    if (manifest['inputs']!=inputs.files or manifest['candidates']!=locked or manifest['exclusions']!=evidence
            or result['diagnostic_source']!=manifest['diagnostic_source']
            or result['diagnostic_source']['sha256']!=inputs.source['sha256']):raise ValueError('Input/source seal mismatch.')
    # Local copy is exact sealed data, not generated or substituted scenes.
    get('other/independent_pool.json')
    if base.file_sha256(out/'other/independent_pool.json')!=POOL_HASH:raise ValueError('Local sealed pool copy changed.')
    base.validate_evaluation_artifacts(out,result,get,required,manifest,a,inputs,locked,historical,excluded,models=('snn',))


def execute(a):
    spec=plan(a)
    if not a.launch:return spec
    out=a.output_dir.resolve()
    if (base.research.REPO.resolve().is_relative_to(a.historical_repo.resolve()) or any(out.is_relative_to(p.resolve())
            for p in (a.historical_repo,a.batch,a.sealed_confirmation))):raise ValueError('Independent code/output required.')
    if out.exists():
        result=base.read(out/'review/summary.json')
        if not result['execution_completed'] or result.get('error') or base.read(out/'review/manifest.json')['plan']!=spec:
            raise ValueError('Incomplete or different request; no retry/repair.')
        validate_bundle(out,result);return result
    out.mkdir(parents=True);(out/'review').mkdir();(out/'other').mkdir()
    started=time.perf_counter();result=dict(diagnostic_only=True,formal_passed=False,execution_completed=False)
    try:
        inputs=base.Inputs();locked,historical,pool,excluded,evidence=provenance(a,inputs)
        if candidate_inputs(a,inputs,load=True)!=locked:raise ValueError('Candidates changed during load.')
        base._write_json(out/'other/excluded_scene_keys.json',sorted(excluded))
        base._write_json(out/'review/manifest.json',dict(plan=spec,candidates=locked,inputs=inputs.files.copy(),
            diagnostic_source=inputs.source,exclusions=evidence,excluded_unique=len(excluded),
            excluded_keys=dict(path='other/excluded_scene_keys.json',sha256=base.file_sha256(out/'other/excluded_scene_keys.json'))))
        pool_path=out/'other/independent_pool.json'
        pool_path.write_bytes((a.sealed_confirmation/'other/independent_pool.json').read_bytes());inputs.add(pool_path)
        base._write_json(out/'review/scene_plan.json',dict(path=str(pool_path),sha256=POOL_HASH,content_digest=POOL_DIGEST,
            scene_keys=[base.offline.scene_key(s['payload']) for s in pool.scenarios],
            historical_pool=dict(path=str(a.validation_pool),sha256=base.file_sha256(a.validation_pool),digest=base.research.DIGEST)))
        def load(c):
            data=base.cal.base.load_models(c['path'],a.validation_pool,'cuda');base.validate_loaded(data,c)
            if data['checkpoint_payload']['formal_config']['seed']!=7:raise ValueError('Loaded seed mismatch.')
            return data
        first=load(locked[0]);bc=base.bc_data(first,'snn',a.snn_bc,seed=7);before_bc=base.frozen_identity(bc)
        directory=out/'other/BC_snn';directory.mkdir()
        shared=base.anchors(bc,directory,a.snn_anchors,inputs)
        base._write_json(out/'review/probe_plan.json',dict(snn=shared));base._write_json(out/'review/sealed_inputs.json',inputs.files)
        sealed={}
        for c in locked:
            data=load(c);before=base.frozen_identity(data)
            directory=out/'other'/c['id'];directory.mkdir()
            probes=directory/'development_probes';probes.mkdir()
            sealed[c['id']]=base.seal_probe_plans(data,bc,shared,probes)
            if base.frozen_identity(data)!=before or base.frozen_identity(bc)!=before_bc:raise ValueError('Frozen planning networks changed.')
        results={}
        for c in locked:
            data=load(c);before=base.frozen_identity(data);directory=out/'other'/c['id']
            flights=directory/'independent';flights.mkdir();stats=base.flight_metrics(data,pool.scenarios,flights)
            base._write_json(directory/'independent_scores.json',stats)
            probes=directory/'development_probes';pr=base.run_probe_branches(data,sealed[c['id']],probes)
            base._write_json(directory/'probe_results.json',pr)
            if base.frozen_identity(data)!=before or base.frozen_identity(bc)!=before_bc:raise ValueError('Frozen networks changed.')
            inputs.check();results[c['id']]=dict(goal_rate=stats['goal_rate'],outcomes=stats['outcomes'],per_scene=stats['flights'],
                frozen_unchanged=True,checkpoint_sha256=data['checkpoint_hash'],probe_count=len(pr),branches=sum(len(p['rows']) for p in pr))
        inputs.check();result.update(results=results,inputs=inputs.files,diagnostic_source=inputs.source,independent_pool_digest=POOL_DIGEST)
        result['artifacts']={p.relative_to(out).as_posix():base.file_sha256(p) for p in out.rglob('*') if p.is_file()}
        validate_bundle(out,result);inputs.check();result['execution_completed']=True
    except BaseException as exc:
        result['execution_completed']=False;result['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        result['process_wall_seconds']=time.perf_counter()-started;base._write_json(out/'review/summary.json',result)
    return result


def main(argv=None):
    print(json.dumps(execute(parser().parse_args(argv)),allow_nan=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
