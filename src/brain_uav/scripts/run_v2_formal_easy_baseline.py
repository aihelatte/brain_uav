"""Four fresh-BC formal easy baselines; private MPS and fail-closed release."""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4

from brain_uav.scripts import run_v2_online_wd3_acc as provenance
from brain_uav.trainers.v2_online_research import atomic_json

REPO=provenance.REPO
POOL_SHA='ebf605a59e1e88590d1fe8d7ab39eabb190e097085a20b8a4c38e5e890a4edcc'
CONFIG=REPO/'scripts/v2_formal_easy_baseline_config.json'
SOURCE_MANIFEST=REPO/'scripts/v2_formal_easy_baseline_source.json'
LR_EXPERIMENT='lr-final-20261113'
LR_CONFIG=REPO/'scripts/v2_formal_easy_lr_final_config.json'
LR_SOURCE_MANIFEST=REPO/'scripts/v2_formal_easy_lr_final_source.json'


class GateComplete(Exception):pass


def file_hash(p):return sha256(Path(p).read_bytes()).hexdigest()


def identity(experiment='baseline'):
    source=provenance.identity()
    source['sha256'][CONFIG.relative_to(REPO).as_posix()]=sha256(CONFIG.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    if experiment==LR_EXPERIMENT:
        source['sha256'][LR_CONFIG.relative_to(REPO).as_posix()]=sha256(LR_CONFIG.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    elif experiment!='baseline':raise ValueError('Unknown formal easy experiment.')
    return source


def jobs():
    return [dict(id=f'{m}_seed{s}',model=m,seed=s,method='min' if m=='ann' else 'acc',acc_eta=None if m=='ann' else .01)
        for m in ('ann','snn') for s in (7,23)]


def resolved(job,experiment='baseline'):
    from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig
    from brain_uav.models.v2_policy_config import V2SpikeRegularizationConfig
    source=json.loads(CONFIG.read_text(encoding='utf-8'))[job['model']]
    old=source['historical']['job']
    config=V2FormalTrainingConfig(stage='easy',seed=job['seed'],max_steps=500000,curriculum_mix={'easy':1.},
        replay_capacity=500000,zone_storage_capacity=6,batch_size=64,success_sample_bias=1.,failure_sample_bias=2.,
        near_goal_sample_bias=2.,near_goal_radius=250.,success_replay_fraction=.25,success_batch_fraction=.25,
        warmup_steps=1280,warmup_strategy='policy_with_noise',actor_freeze_steps=25000,
        actor_lr=old['actor_lr'],critic_lr=old['critic_lr'],gamma=.99,tau=.005,policy_delay=2,
        actor_grad_clip_norm=1.,critic_grad_clip_norm=1.,actor_rl_scale_alpha=2.5,acc_eta=job['acc_eta'],
        spike_regularization=V2SpikeRegularizationConfig(beta_max=0. if job['model']=='ann' else 1.,
            target_rate=.17,ramp_start_steps=25000,ramp_end_steps=75000),mc_return_weight=0.,
        terminal_geo_regularization_enabled=True,terminal_geo_lambda=3000.,terminal_geo_radius=250.,terminal_geo_safe_clearance=40.,
        early_stop_min_steps=125000,window_episode_count=15,consecutive_qualified_windows=4,max_failures_per_window=1,
        validation_max_failures=5)
    from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation
    result=dict(job=job,formal_config=config.to_dict(),scenario_config=source['historical']['scenario_config'],
        reward_config=source['historical']['reward_config'],bc_schedule='smooth-mid-strong',target_noise_scale=1.,
        actor_projection_dim=32,actor_hidden_dim=192,snn_time_window=4,uav_collision_radius=0.,
        gpu_td3_compute=True,cpu_deduplication=True,non_blocking_batch_transfer=True,
        gpu_compute_non_blocking_coefficients=False,compilation=default_v2_cuda_graph_compilation(model=job['model'],resolved_device='cuda'),
        changes=dict(seed=[23,job['seed']],budget=[400000,500000],pause=[200000,None],
            non_blocking_batch_transfer=[False,True],early_stop=['research_disabled','formal_enabled']))
    # Compare every inherited job field rather than silently take CLI defaults.
    mapping=dict(actor_lr='actor_lr',critic_lr='critic_lr',batch_size='batch_size',gamma='gamma',tau='tau',policy_delay='policy_delay',
        actor_freeze_steps='actor_freeze_steps',warmup_steps='warmup_steps',warmup_strategy='warmup_strategy',alpha='actor_rl_scale_alpha')
    if any(old[k]!=config.to_dict()[v] for k,v in mapping.items()):raise ValueError('Historical job configuration drift.')
    if (old['model']!=job['model'] or old['acc_eta']!=job['acc_eta'] or old['bc_schedule']!=result['bc_schedule']
            or old['spike_beta_max']!=config.spike_regularization.beta_max):raise ValueError('Historical method mismatch.')
    if source['resolved'][job['id']]!=result:
        raise ValueError('Resolved baseline configuration differs from the reviewed lock.')
    if experiment==LR_EXPERIMENT:
        candidate=json.loads(LR_CONFIG.read_text(encoding='utf-8'))
        if (candidate['name']!=experiment or candidate['baseline_commit']!='8d6d9f6ec01f2af4354371a1735384aead6ddf42'
                or candidate['baseline_config_sha256']!=sha256(CONFIG.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                or set(candidate['overrides'])!={'ann','snn'}):
            raise ValueError('Learning-rate candidate baseline identity mismatch.')
        changes=candidate['overrides'][job['model']]
        if set(changes)!=({'actor_lr'} if job['model']=='ann' else {'critic_lr'}):
            raise ValueError('Only the reviewed per-model learning rate may change.')
        for name,value in changes.items():
            if (set(value)!={'from','to'} or getattr(config,name)!=value['from']
                    or type(value['to']) not in (int,float) or not math.isfinite(value['to']) or value['to']<=0):
                raise ValueError('Invalid explicit learning-rate difference.')
        config=replace(config,**{name:value['to'] for name,value in changes.items()})
        result.update(formal_config=config.to_dict(),experiment=experiment,
            baseline_commit=candidate['baseline_commit'],learning_rate_changes=changes)
    elif experiment!='baseline':raise ValueError('Unknown formal easy experiment.')
    return result,config


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',choices=('plan','launch','worker','status','collect'),default='plan')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--ann-bc',type=Path,required=True);p.add_argument('--snn-bc',type=Path,required=True)
    p.add_argument('--validation-pool',type=Path,required=True)
    p.add_argument('--source-manifest',type=Path,default=SOURCE_MANIFEST)
    p.add_argument('--experiment',choices=('baseline',LR_EXPERIMENT),default='baseline')
    p.add_argument('--job',choices=[j['id'] for j in jobs()])
    p.add_argument('--timeout',type=float,default=1800.)
    p.add_argument('--launch',action='store_true')
    p.add_argument('--gate-only',action='store_true',help='Bounded four-process MPS CUDA correctness, never train.')
    p.add_argument('--gate-report',type=Path)
    p.add_argument('--gate-timeout',type=float,default=300.)
    return p


def plan(a):
    return dict(mode=a.mode,launch=a.launch,gate_only=a.gate_only,root=str(a.root),experiment=a.experiment,
        jobs=[resolved(j,a.experiment)[0] for j in jobs()],
        initialization_timeout=a.timeout,gate_timeout=a.gate_timeout if a.gate_only else None,
        gate_physical_update_budget_per_worker=128,
        formal_early_stop=True,research_pause=False,inputs=dict(ann_bc=str(a.ann_bc),snn_bc=str(a.snn_bc),pool=str(a.validation_pool)),
        new_combination='formal ACC + isolated scoring + batch nonblocking + MPS; CUDA gate required')


def inputs(a):
    import inspect
    from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
    if inspect.signature(V2StaticNoFlyTrajectoryEnv).parameters['cache_ellipsoid_segment_clearance'].default is not True:
        raise ValueError('Exact ellipsoid cache baseline default is not enabled.')
    source=identity(a.experiment)
    if json.loads(a.source_manifest.read_text())['sha256']!=source['sha256']:raise ValueError('Actual source files differ from reviewed hashes.')
    paths={'ann_bc':a.ann_bc,'snn_bc':a.snn_bc,'pool':a.validation_pool,'config':CONFIG,'source_manifest':a.source_manifest}
    if a.experiment==LR_EXPERIMENT:paths['experiment_config']=LR_CONFIG
    locked=json.loads(CONFIG.read_text())
    for model in ('ann','snn'):
        path=Path(locked[model]['stage_start_server'])
        if file_hash(path)!=locked[model]['stage_start_sha256'] or json.loads(path.read_text())!=locked[model]['historical']:
            raise ValueError('Historical stage_start identity/configuration mismatch.')
        paths[model+'_stage_start']=path
    hashes={k:file_hash(p) for k,p in paths.items()}
    if any(hashes[m+'_bc']!=provenance.BC[m][1] for m in ('ann','snn')) or hashes['pool']!=POOL_SHA:
        raise ValueError('BC best/fixed pool raw SHA256 mismatch.')
    from brain_uav.trainers.v2_validation import load_v2_validation_pool
    pool=load_v2_validation_pool(a.validation_pool,expected_level='easy',expected_count=100)
    if pool.content_digest!=provenance.DIGEST:raise ValueError('Fixed pool digest mismatch.')
    return dict(source=source,experiment=a.experiment,paths={k:str(p.resolve()) for k,p in paths.items()},hashes=hashes,pool_digest=pool.content_digest)


def check_inputs(evidence):
    if any(file_hash(p)!=evidence['hashes'][k] for k,p in evidence['paths'].items()):raise ValueError('Input changed during batch.')
    if identity(evidence.get('experiment','baseline'))['sha256']!=evidence['source']['sha256']:raise ValueError('Source changed during batch.')


def wait_ready(processes,root,timeout):
    deadline=time.monotonic()+timeout
    while True:
        if any(p.poll() is not None for p in processes.values()):raise RuntimeError('Worker exited before release.')
        if all((root/j/'ready.json').exists() for j in processes):return
        if time.monotonic()>deadline:raise TimeoutError('Initialization deadline exceeded.')
        time.sleep(.1)


def release(processes,root,session,experiment='baseline'):
    records={j:json.loads((root/j/'ready.json').read_text()) for j in processes}
    if set(records)!={j['id'] for j in jobs()} or len({p.pid for p in processes.values()})!=4:raise ValueError('Four unique worker PIDs required.')
    for job in jobs():
        r=records[job['id']]
        if (r['pid']!=processes[job['id']].pid or r['job']!=job or r['resolved']!=resolved(job,experiment)[0]
                or processes[job['id']].poll() is not None or r['synchronized'] is not True):
            raise ValueError('Worker identity/configuration/readiness mismatch.')
        from brain_uav.scripts.run_v2_online_wd3_acc import validate_seed_manifest
        validate_seed_manifest(resolved(job,experiment)[1],r['seed_manifest'])
    access=session.verify(processes)
    if any(p.poll() is not None for p in processes.values()):
        raise RuntimeError('Worker exited during MPS membership verification.')
    atomic_json(root/'mps_access.json',access)
    atomic_json(root/'release.json',dict(pids=[p.pid for p in processes.values()],time=time.time()))
    return records


def stop_owned(processes,session):
    errors=[]
    for name,p in processes.items():
        if p.poll() is None:
            try:
                session.before_terminate(p)  # must confirm terminate_client == 0 first
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
            except BaseException as exc:errors.append(f'{name}: {exc}')
    if errors:raise RuntimeError('Cleanup blocked; service retained: '+'; '.join(errors))


def worker(a):
    import torch
    from brain_uav.config import reward_config_from_snapshot
    from brain_uav.trainers.v2_validation import scenario_config_from_snapshot,evaluate_v2_fixed_validation
    from brain_uav.scripts.train_v2_td3 import run_v2_td3_stage
    manifest=json.loads((a.root/'manifest.json').read_text());check_inputs(manifest['inputs'])
    if manifest['plan']['gate_only']!=a.gate_only or manifest['plan'].get('experiment','baseline')!=a.experiment:
        raise ValueError('Worker mode differs from batch manifest.')
    for name,path in (('ann_bc',a.ann_bc),('snn_bc',a.snn_bc),('pool',a.validation_pool)):
        if str(path.resolve())!=manifest['inputs']['paths'][name]:
            raise ValueError('Worker input path differs from batch manifest.')
    job=next(j for j in jobs() if j['id']==a.job);spec,cfg=resolved(job,a.experiment)
    directory=a.root/job['id'];directory.mkdir(exist_ok=False);started=time.perf_counter()
    status=dict(execution_completed=False,job=job,formal_passed=None,pid=os.getpid(),batch_id=manifest['batch_id'])
    def gate(engine,config,seeds,initial,compilation):
        from brain_uav.models.v2_policy_config import projection_from_architecture
        arch=engine.checkpoint_state_dict()['architecture']
        if (arch['actor_hidden_dim']!=192 or arch['critic1_hidden_dim']!=128 or arch['critic2_hidden_dim']!=128
                or projection_from_architecture(arch)!=32 or engine.actor.zone_set_encoder.output_dim!=128
                or (job['model']=='snn' and (arch['time_window']!=4 or arch['tau']!=2.))):
            raise ValueError('BC/critic architecture differs from locked baseline.')
        if config.to_dict()!=spec['formal_config'] or not engine.gpu_td3_compute or not engine.non_blocking_batch_transfer:
            raise ValueError('Actual formal/update configuration differs from locked baseline.')
        torch.cuda.synchronize(engine.device)
        atomic_json(directory/'ready.json',dict(pid=os.getpid(),job=job,resolved=spec,seed_manifest=dict(seeds),
            initial_state_sha256=initial,compilation=compilation,synchronized=True))
        deadline=time.monotonic()+a.timeout
        while not (a.root/'release.json').exists():
            if (a.root/'abort.json').exists():raise RuntimeError('Parent initialization aborted.')
            if time.monotonic()>deadline:raise TimeoutError('Release deadline exceeded.')
            time.sleep(.1)
        if os.getpid() not in json.loads((a.root/'release.json').read_text())['pids']:
            raise ValueError('Worker PID absent from release.')
        check_inputs(manifest['inputs'])
        if a.gate_only:
            from brain_uav.trainers.v2_formal_training import prepare_v2_stage_initialization
            from brain_uav.trainers.v2_validation import load_v2_validation_pool
            from brain_uav.scripts.v2_formal_easy_cuda_gate import check
            bc=getattr(a,job['model']+'_bc')
            prepared=prepare_v2_stage_initialization(config,init_checkpoint=bc,device='cuda',model_type=job['model'],
                scenario=scenario_config_from_snapshot(spec['scenario_config']),rewards=reward_config_from_snapshot(spec['reward_config']))
            pool=load_v2_validation_pool(a.validation_pool,expected_level='easy',expected_count=100)
            gate_result=check(engine,config,pool,prepared,bc,job['model'],output=directory/'cuda_gate.json')
            gate_result['identity']=dict(job=job,pid=os.getpid(),batch_id=manifest['batch_id'],resolved=spec,
                source_sha256=manifest['inputs']['source']['sha256'],input_hashes=manifest['inputs']['hashes'])
            atomic_json(directory/'cuda_gate.json',gate_result)
            check_inputs(manifest['inputs'])
            raise GateComplete()
    def evaluate(engine,pool,rewards,result,path):
        # Same worker, no bank feedback, no training exploration. Validator owns SNN resets.
        value=evaluate_v2_fixed_validation(engine.actor,pool,rewards,max_failures=5,device='cuda',cpu_deduplication=True).to_dict()
        atomic_json(directory/'terminal_evaluation.json',dict(checkpoint=str(path),checkpoint_sha256=file_hash(path),
            pool_sha256=POOL_SHA,pool_digest=pool.content_digest,reward_config=asdict(rewards),evaluation=value,
            formal_passed=result.passed_validation,actual_steps=result.stage_steps))
    try:
        summary=run_v2_td3_stage(stage='easy',init_checkpoint=getattr(a,job['model']+'_bc'),output=directory/'model.pt',
            metrics_out=directory/'metrics.json',validation_pool=a.validation_pool,seed=job['seed'],device='cuda',model=job['model'],
            max_stage_steps=500000,actor_lr=cfg.actor_lr,critic_lr=cfg.critic_lr,failure_sample_bias=2.,acc_eta=cfg.acc_eta,
            gamma=.99,target_q_aggregation='min',mc_return_weight=0.,bc_schedule='smooth-mid-strong',target_noise_scale=1.,
            actor_projection_dim=32,actor_hidden_dim=192,snn_time_window=4,actor_rl_scale_alpha=2.5,
            spike_regularization=cfg.spike_regularization,formal_config=cfg,
            scenario=scenario_config_from_snapshot(spec['scenario_config']),rewards=reward_config_from_snapshot(spec['reward_config']),
            uav_collision_radius=0.,gpu_td3_compute=True,cpu_deduplication=True,non_blocking_batch_transfer=True,
            initialization_gate=gate,terminal_evaluation_sink=evaluate,**spec['compilation'])
        check_inputs(manifest['inputs']);status.update(execution_completed=True,formal_passed=summary['passed'],result=summary)
        execution=summary['execution_configuration']
        if not execution['gpu_td3_compute']['effective'] or not (execution['non_blocking_batch_transfer']['actual_execution'] or {}).get('effective'):
            raise ValueError('Missing observed GPUCompute/nonblocking transfer execution.')
    except GateComplete:status.update(execution_completed=True,diagnostic_only=True,strict_passed=True)
    except BaseException as exc:
        status.update(execution_completed=False,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['process_wall_seconds']=time.perf_counter()-started
        atomic_json(directory/'worker_summary.json',status)


def choose(records):
    if len(records)!=2 or {r['seed'] for r in records}!={7,23} or any(not r['execution_completed'] or len(r['evaluation']['scenarios'])!=100 for r in records):
        raise ValueError('Incomplete two-seed terminal evaluation.')
    for r in records:
        scenes=r['evaluation']['scenarios']
        if len({s['scenario_id'] for s in scenes})!=100 or not all(math.isfinite(s['episode_return']) for s in scenes):
            raise ValueError('Duplicate/nonfinite terminal evaluation.')
    return min(records,key=lambda r:(not r['formal_passed'],-r['evaluation']['goal_count'],
        -sum(s['episode_return'] for s in r['evaluation']['scenarios'])/100,r['actual_steps'],r['seed']))


def gate_bindings(root,manifest):
    bindings={}
    for job in jobs():
        directory=root/job['id']
        reports={name:json.loads((directory/name).read_text()) for name in
            ('cuda_gate.json','worker_summary.json','process.json','ready.json')}
        gate,worker,process,ready=[reports[k] for k in ('cuda_gate.json','worker_summary.json','process.json','ready.json')]
        expected=dict(job=job,pid=ready['pid'],batch_id=manifest['batch_id'],resolved=resolved(job,manifest['plan'].get('experiment','baseline'))[0],
            source_sha256=manifest['inputs']['source']['sha256'],input_hashes=manifest['inputs']['hashes'])
        if (gate.get('identity')!=expected or gate['strict_passed'] is not True or gate['execution_completed'] is not True
                or worker['execution_completed'] is not True or worker['strict_passed'] is not True
                or worker['job']!=job or worker['pid']!=ready['pid'] or worker['batch_id']!=manifest['batch_id']
                or process['exit_code']!=0 or process['pid']!=ready['pid'] or ready['job']!=job):
            raise ValueError('CUDA gate worker/model/seed/PID/exit identity mismatch.')
        bindings[job['id']]={name:file_hash(directory/name) for name in reports}
    return bindings


def validate_gate(a,evidence):
    if a.gate_report is None:raise ValueError('Formal launch requires --gate-report.')
    root=a.gate_report
    status=json.loads((root/'batch_summary.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    if (status['execution_completed'] is not True or manifest['plan']['gate_only'] is not True
            or manifest['plan'].get('experiment','baseline')!=a.experiment
            or manifest['inputs']['hashes']!=evidence['hashes']
            or manifest['inputs'].get('paths')!=evidence.get('paths')
            or manifest['inputs']['source']['sha256']!=evidence['source']['sha256']):
        raise ValueError('CUDA gate input/source/batch mismatch.')
    bindings=gate_bindings(root,manifest)
    if bindings!=status['gate_bindings']:raise ValueError('CUDA gate report hashes changed.')
    return dict(root=str(root.resolve()),batch_id=manifest['batch_id'],bindings=bindings,
        manifest_sha256=file_hash(root/'manifest.json'),summary_sha256=file_hash(root/'batch_summary.json'))


def collect(a):
    from brain_uav.trainers.v2_formal_training import load_v2_formal_checkpoint,v2_bc_schedule_metadata
    from brain_uav.trainers.v2_validation import load_v2_validation_pool
    import tempfile
    summary_path=a.root/'selection_summary.json'
    previous=json.loads(summary_path.read_text()) if summary_path.exists() else None
    result=dict(execution_completed=False,models={})
    staging=None
    try:
        manifest=json.loads((a.root/'manifest.json').read_text())
        if manifest['plan'].get('experiment','baseline')!=a.experiment:
            raise ValueError('Collect experiment differs from batch manifest.')
        evidence=manifest['inputs']
        check_inputs(evidence)
        pool=load_v2_validation_pool(evidence['paths']['pool'],expected_level='easy',expected_count=100)
        scene_ids=[s['scenario_id'] for s in pool.scenarios]
        if pool.content_digest!=provenance.DIGEST:
            raise ValueError('Selection pool identity mismatch.')
        # Validate all four terminal candidates before creating any selected output.
        for model in ('ann','snn'):
            records=[]
            for seed in (7,23):
                job=next(j for j in jobs() if j['model']==model and j['seed']==seed)
                spec,expected=resolved(job,a.experiment)
                directory=a.root/job['id']
                status=json.loads((directory/'worker_summary.json').read_text())
                process=json.loads((directory/'process.json').read_text())
                ready=json.loads((directory/'ready.json').read_text())
                evaluation=json.loads((directory/'terminal_evaluation.json').read_text())
                passed=evaluation['formal_passed']
                path=directory/('model.pt' if passed else 'model_failed.pt')
                if Path(evaluation['checkpoint']).resolve()!=path.resolve() or path.resolve().parent!=directory.resolve():
                    raise ValueError('Checkpoint is not this batch worker terminal model.')
                if (status['job']!=job or ready['job']!=job or status['batch_id']!=manifest['batch_id']
                        or type(ready['pid']) is not int or ready['pid']<=0
                        or type(status['pid']) is not int or type(process['pid']) is not int
                        or status['pid']!=ready['pid'] or process['pid']!=ready['pid']
                        or not status['execution_completed'] or process['exit_code']!=0
                        or status['formal_passed']!=passed or file_hash(path)!=evaluation['checkpoint_sha256']
                        or evaluation['pool_digest']!=provenance.DIGEST or evaluation['pool_sha256']!=POOL_SHA):
                    raise ValueError('Terminal worker/evaluation identity incomplete.')
                raw=load_v2_formal_checkpoint(path,expected_stage='easy',expected_model_type=model,require_passed=False)
                training=raw['training_result']
                worker_result=status['result']
                source=raw['initialization_source']
                stop='fixed_validation_passed' if passed else 'max_steps_without_validation'
                if (raw['formal_config']!=expected.to_dict() or raw['passed_validation']!=passed
                        or training['stage_steps']!=evaluation['actual_steps'] or training['stop_reason']!=stop
                        or training['status']!=('passed' if passed else 'failed')
                        or worker_result['steps']!=training['stage_steps'] or worker_result['stop_reason']!=stop
                        or worker_result['passed']!=passed or Path(worker_result['checkpoint']).resolve()!=path.resolve()
                        or (not passed and training['stage_steps']!=500000)
                        or raw['scenario_config']!=spec['scenario_config'] or raw['reward_config']!=spec['reward_config']
                        or evaluation['reward_config']!=spec['reward_config'] or raw['uav_collision_radius']!=0.
                        or raw['validation_pool']['content_digest']!=pool.content_digest
                        or source['kind']!='v2_bc_best' or source['checkpoint_kind']!='best'
                        or Path(source['path']).resolve()!=Path(evidence['paths'][model+'_bc']).resolve()
                        or raw['bc_schedule']!=v2_bc_schedule_metadata('easy',schedule='smooth-mid-strong')):
                    raise ValueError('Terminal configuration/BC/steps/stop identity mismatch.')
                scenes=evaluation['evaluation']['scenarios']
                if ([s['scenario_id'] for s in scenes]!=scene_ids or any(s['episode_length']<=0 for s in scenes)
                        or evaluation['evaluation']['goal_count']!=sum(s['outcome']=='goal' for s in scenes)):
                    raise ValueError('Terminal fixed-pool evaluation incomplete.')
                records.append(dict(evaluation,seed=seed,execution_completed=True))
            winner=choose(records)
            result['models'][model]=dict(candidates=records,selected=winner,
                kind='formal_passed' if winner['formal_passed'] else 'best_available',
                rule='formal pass, goals descending, mean return descending, steps ascending, seed ascending',
                selection_bias='fixed validation pool best-of-two; not independent test performance')
        check_inputs(evidence)
        selected=a.root/'selected'
        if selected.exists():
            for model,selection in result['models'].items():
                destination=selected/model
                if (json.loads((destination/'selection.json').read_text())!=selection
                        or file_hash(destination/'baseline.pt')!=selection['selected']['checkpoint_sha256']):
                    raise ValueError('Existing selected output is incomplete or differs; retained for review.')
        else:
            staging=Path(tempfile.mkdtemp(prefix='.selected-',dir=a.root))
            for model,selection in result['models'].items():
                destination=staging/model
                destination.mkdir()
                shutil.copyfile(selection['selected']['checkpoint'],destination/'baseline.pt')
                if file_hash(destination/'baseline.pt')!=selection['selected']['checkpoint_sha256']:
                    raise ValueError('Selected copy hash mismatch.')
                atomic_json(destination/'selection.json',selection)
            os.replace(staging,selected)
            staging=None
        result['execution_completed']=True
        if previous!=result:
            atomic_json(summary_path,result)
        return result
    except BaseException as exc:
        if not previous or previous.get('execution_completed') is not True:
            result.update(execution_completed=False,kind='incomplete',error=str(exc))
            try:
                atomic_json(summary_path,result)
            except Exception as report_error:
                print(f'Selection error report failed: {report_error}',file=sys.stderr,flush=True)
        raise
    finally:
        if staging is not None:
            try:
                shutil.rmtree(staging)
            except Exception as exc:
                print(f'Staging cleanup failed: {exc}',file=sys.stderr,flush=True)


def launch(a):
    previous=signal.getsignal(signal.SIGTERM)
    terminating=False
    def terminate(signum,frame):
        nonlocal terminating
        if not terminating:
            terminating=True
            raise InterruptedError('SIGTERM received by formal baseline launcher.')
    signal.signal(signal.SIGTERM,terminate)
    try:
        return _launch(a)
    finally:
        signal.signal(signal.SIGTERM,previous)


def _launch(a):
    from brain_uav.scripts.v2_mps_diagnostic import Session,capability
    if Path(sys.executable).resolve()!=Path(provenance.PYTHON).resolve():
        raise ValueError('Use the declared server Python.')
    if not a.gate_only and a.gate_report is None:
        raise ValueError('Formal launch requires --gate-report.')
    evidence=inputs(a)
    gate_proof=None if a.gate_only else validate_gate(a,evidence)
    a.root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    processes={}
    logs=[]
    worker_started={}
    status=dict(execution_completed=False,formal_passed=None)
    session=None
    failure=None
    def report(path,value):
        try:
            atomic_json(path,value)
        except BaseException as exc:
            status.setdefault('report_errors',[]).append(f'{path}: {exc}')
            try:
                print(f'Report write failed: {path}: {exc}',file=sys.stderr,flush=True)
            except Exception:
                pass
            return exc
        return None
    try:
        manifest=dict(plan=plan(a),inputs=evidence,gate_evidence=gate_proof,batch_id=uuid4().hex)
        atomic_json(a.root/'manifest.json',manifest)
        session=Session(a.root/'mps')
        session.start()
        capability(a.root/'capability',session)
        capability(a.root/'termination_capability',session,terminate_client=True)
        for job in jobs():
            log=(a.root/(job['id']+'.log')).open('x',encoding='utf-8')
            logs.append(log)
            command=[sys.executable,'-u','-B','-m','brain_uav.scripts.run_v2_formal_easy_baseline','worker',
                '--root',str(a.root),'--ann-bc',str(a.ann_bc),'--snn-bc',str(a.snn_bc),'--validation-pool',str(a.validation_pool),
                '--source-manifest',str(a.source_manifest),'--experiment',a.experiment,
                '--job',job['id'],'--timeout',str(a.timeout),'--launch']
            if a.gate_only:
                command.append('--gate-only')
            worker_started[job['id']]=time.perf_counter()
            process=subprocess.Popen(command,cwd=REPO,env=dict(session.env,PYTHONPATH='src'),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[job['id']]=process
            session.register(processes)
        wait_ready(processes,a.root,a.timeout)
        release(processes,a.root,session,a.experiment)
        deadline=time.monotonic()+a.gate_timeout if a.gate_only else None
        pending=set(processes)
        while pending:
            if deadline is not None and time.monotonic()>deadline:
                raise TimeoutError('CUDA gate deadline exceeded after release.')
            for name in list(pending):
                rc=processes[name].poll()
                if rc is not None:
                    atomic_json(a.root/name/'process.json',dict(pid=processes[name].pid,exit_code=rc,
                        observed_wall_seconds=time.perf_counter()-worker_started[name]))
                    pending.remove(name)
                    if rc:
                        raise RuntimeError(f'{name} failed with {rc}')
            if pending:
                time.sleep(.5)
        check_inputs(evidence)
        if a.gate_only:
            status['gate_bindings']=gate_bindings(a.root,manifest)
        else:
            selection=collect(a)
            status['formal_passed']={model:entry['selected']['formal_passed'] for model,entry in selection['models'].items()}
        status['execution_completed']=True
    except BaseException as exc:
        failure=exc
        status.update(execution_completed=False,error=f'{type(exc).__name__}: {exc}')
        report(a.root/'abort.json',status)
        if not a.gate_only and not (a.root/'selection_summary.json').exists():
            report(a.root/'selection_summary.json',dict(execution_completed=False,kind='incomplete',models={},error=status['error']))
    finally:
        if failure is not None and session is not None:
            try:
                stop_owned(processes,session)
            except BaseException as exc:
                status['cleanup_error']=str(exc)
        for log in logs:
            try:
                log.close()
            except Exception as exc:
                status.setdefault('report_errors',[]).append(str(exc))
        for name,process in processes.items():
            path=a.root/name/'process.json'
            if not path.exists():
                report(path,dict(pid=process.pid,exit_code=process.poll(),
                    observed_wall_seconds=time.perf_counter()-worker_started[name]))
        if session is not None:
            try:
                cleanup=session.finish()
            except BaseException as exc:
                cleanup=dict(stopped=False,cleanup_error=str(exc))
            status['mps_cleanup']=cleanup
            if not cleanup.get('stopped'):
                status['execution_completed']=False
        if status.get('report_errors'):
            status['execution_completed']=False
        status['process_wall_seconds']=time.perf_counter()-started
        write_error=report(a.root/'batch_summary.json',status)
        if failure is None and write_error is not None:
            failure=write_error
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    if not status['execution_completed']:
        raise RuntimeError('Batch cleanup/reporting incomplete.')


def main(argv=None):
    a=parser().parse_args(argv)
    if not math.isfinite(a.timeout) or a.timeout<=0:raise ValueError('Timeout must be finite and positive.')
    if not math.isfinite(a.gate_timeout) or a.gate_timeout<=0:raise ValueError('Gate timeout must be finite and positive.')
    if a.mode=='status':
        print(json.dumps({j['id']:json.loads(p.read_text()) if (p:=a.root/j['id']/'worker_summary.json').exists() else 'not_finished' for j in jobs()},indent=2));return 0
    if not a.launch:print(json.dumps(plan(a),indent=2));return 0
    if a.mode=='launch':launch(a)
    elif a.mode=='worker':
        if a.job is None:raise ValueError('Worker requires explicit --job.')
        worker(a)
    elif a.mode=='collect':collect(a)
    else:raise ValueError('Select launch/worker/collect explicitly.')
    return 0


if __name__=='__main__':raise SystemExit(main())
