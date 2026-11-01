"""Pre-registered online WD3/ACC research. Plan by default; no automatic resume."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

import torch

from brain_uav.trainers.v2_online_research import (
    ARMS, FORMAT, OnlineResearch, atomic_json, target_config, validate_snapshot,
    training_identity,
)

REPO=Path(__file__).resolve().parents[3]
PYTHON='/home/zj/envs/uav_torch/bin/python'
POOL=Path('outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json')
DIGEST='2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b'
BC={
 'ann':(Path('outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt'),
        '0e103aaad88bb7c60236d5c8719b126c597288af39f6e9c07d4b087809d93b7b'),
 'snn':(Path('outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt'),
        '7b2b266fdf32cdb8b4eb220ef26a45008717a0af1538f8d478019f37ddc9adbc')}


def matrix():
    return [dict(id=f'{model}_{arm}',model=model,arm=arm,seed=23,stage='easy',device='cuda',
        schedule_horizon_steps=500000,pause_requested_step=200000,hard_limit=400000,
        actor_lr=.000155 if model=='ann' else .000825,critic_lr=.0002 if model=='ann' else .001,
        alpha=2.5,gamma=.99,tau=.005,policy_delay=2,batch_size=64,warmup_steps=1280,
        warmup_strategy='policy_with_noise',actor_freeze_steps=25000,
        bc_schedule='smooth-mid-strong',mc_return_weight=0.,target_config=target_config(arm),
        target_q_aggregation='mean' if arm=='mean' else 'min',
        acc_eta=.01 if arm=='acc_001' else .1 if arm=='acc_010' else None,
        actor_projection_dim=32,actor_hidden_dim=192,P1=False,P2=True,P3=True,
        gpu_td3_compute=True,cpu_deduplication=True,cache_ellipsoid_segment_clearance=True,
        non_blocking_batch_transfer=False,gpu_compute_non_blocking_coefficients=False,
        snn_time_window=4,spike_beta_max=1. if model=='snn' else 0.,target_rate=.17,
        ramp_start_steps=25000,ramp_end_steps=75000,
        requested_snapshots=[75000,150000,200000,300000,400000])
        for model in ('ann','snn') for arm in ARMS]


def identity():
    paths=sorted((REPO/'src').rglob('*.py'))+[REPO/'scripts/online_v2_wd3_acc.sh']
    # Git may convert CRLF to LF between Windows review and Linux execution.
    return dict(sha256={p.relative_to(REPO).as_posix():sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths},
        source_hash_normalization='CRLF-to-LF',
        **{name:subprocess.run(['git',*cmd],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
           for name,cmd in [('HEAD',['rev-parse','HEAD']),('status',['status','--short'])]})


def plan(args):
    from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation
    from brain_uav.envs.v2_scenario_generator import V2ScenarioGeneratorConfig
    return dict(diagnostic_only=True,formal_passed=False,mode=args.mode,launch=args.launch,root=str(args.root),jobs=matrix(),
        checkpoint=None if args.checkpoint is None else str(args.checkpoint),
        cuda_gate=dict(model=args.model,arm=args.job,updates_per_arm_per_path=25,
            max_actual_updates_per_model=400,rtol=1e-4,atol=1e-5,strict=True,new_exemptions=False,
            masks=['mixed','empty'],modes=['critic_only','frozen_targets','actor'],
            checks=['targets','loss','gradients','Adam','parameter_updates','beta_runtime_refresh','Graph_cache','load_rebuild']),
        compilation={m:default_v2_cuda_graph_compilation(model=m,resolved_device='cuda') for m in ('ann','snn')},
        controller=dict(beta_initial=.5,bounds=[0.,1.],eta=[.01,.1],start_steps=25000,
            min_interval_env_steps=1000,episode_boundary_only=True,capacity_transitions=5000,
            EMA_rho=.05,C='mean_training_bank(G-(Q1+Q2)/2)',A='mean(abs(G-Qbar))',
            formula='EMA=A first, then .95*EMA+.05*A; S=max(1,EMA); beta=clip(beta+eta*clip(C/S,-1,1),0,1)',
            semantics='recent exploratory behavior returns, not unbiased current-policy calibration'),
        inputs=dict(BC={m:dict(path=str(p),sha256=h) for m,(p,h) in BC.items()},
            pool=str(POOL),pool_content_digest=DIGEST),
        confirmation=dict(plan_only=True,scenes=40,empty=8,nonempty=32,all_five_geometries=True,
            selection_seed=20261102,exclude='all training/calibration payload fingerprints plus old development pool',
            reporting=dict(macro='scene equal',subgroups='empty/nonempty, all five geometries, outcomes',
                easy_probability_empty=V2ScenarioGeneratorConfig().zone_count_probabilities['easy'][0],
                weighted='p_empty*mean_empty+(1-p_empty)*mean_nonempty; actual source formal generator probabilities'),
            generation='not authorized; candidate lock and new authorization required'),
        probes=dict(plan_only=True,count=12,selection='first 12 selected development scenes by sequence_index; mid-prefix floor((T-1)/2) of BC natural flight',
            candidates='BC +/- .2 per dimension of action limit + BC + tested actor, clip and exact float32 dedup'),
        limitations=['200k alive pause; process loss cannot be exactly resumed from snapshot',
            'candidate/formal pass cannot terminate this research; no medium or 500k run',
            'all sixteen initialize/hash-match before any training; no automatic restart'])


def preflight(args, *, cuda=True):
    """Real Path arguments through strict BC and pool provenance checks."""
    from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig, prepare_v2_stage_initialization
    from brain_uav.trainers.v2_validation import load_v2_validation_pool
    from brain_uav.models.v2_policy_config import V2SpikeRegularizationConfig
    from brain_uav.envs.v2_static_no_fly_env import V2StaticNoFlyTrajectoryEnv
    import inspect
    source=identity()
    if args.source_manifest is None: raise ValueError('Explicit reviewed --source-manifest required for execution.')
    expected=json.loads(Path(args.source_manifest).read_text(encoding='utf-8'))
    if expected['sha256']!=source['sha256']: raise ValueError('Source file hashes differ from reviewed manifest.')
    if inspect.signature(V2StaticNoFlyTrajectoryEnv).parameters['cache_ellipsoid_segment_clearance'].default is not True:
        raise ValueError('Adopted exact ellipsoid cache must default on.')
    if cuda and (not torch.cuda.is_available() or shutil.which('tmux') is None):
        raise RuntimeError('CUDA and tmux are required; no fallback or reduced concurrency.')
    result={}
    for model,(relative,expected_hash) in BC.items():
        path=REPO/relative
        if sha256(path.read_bytes()).hexdigest()!=expected_hash: raise ValueError(f'{model} BC best SHA mismatch.')
        job=next(j for j in matrix() if j['model']==model)
        config=make_config(job)
        prepared=prepare_v2_stage_initialization(config,init_checkpoint=path,device='cpu',model_type=model,rewards=research_rewards())
        pool=load_v2_validation_pool(REPO/POOL,expected_level='easy',expected_count=100,
            expected_scenario=prepared.scenario_config,expected_uav_collision_radius=prepared.uav_collision_radius)
        if pool.content_digest!=DIGEST: raise ValueError('Fixed development pool digest mismatch.')
        from brain_uav.trainers.v2_validation import scenario_config_snapshot
        result[model]=dict(path=str(path),sha256=expected_hash,pool_file_sha256=sha256((REPO/POOL).read_bytes()).hexdigest(),
            effective_formal_config=config.to_dict(),effective_reward_config=asdict(prepared.reward_config),
            effective_scenario_config=scenario_config_snapshot(prepared.scenario_config))
    return dict(inputs=result,source=source,source_manifest_sha256=sha256(Path(args.source_manifest).read_bytes()).hexdigest(),
        gpu=None if not cuda else subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=True).stdout,
        torch_threads=torch.get_num_threads())


def make_config(job):
    from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig
    from brain_uav.models.v2_policy_config import V2SpikeRegularizationConfig
    return V2FormalTrainingConfig(stage='easy',seed=23,max_steps=500000,batch_size=64,
        actor_lr=job['actor_lr'],critic_lr=job['critic_lr'],actor_rl_scale_alpha=2.5,
        failure_sample_bias=2.,target_q_aggregation=job['target_q_aggregation'],
        spike_regularization=V2SpikeRegularizationConfig(beta_max=job['spike_beta_max'],target_rate=.17,
            ramp_start_steps=25000,ramp_end_steps=75000))


def research_rewards():
    from brain_uav.config import RewardConfig
    return RewardConfig(breakthrough_clearance_gate_enabled=False,
        descent_three_band_enabled=True,boundary_directional_warning_enabled=True)


def wait_for(path, root, timeout):
    started=time.monotonic()
    while not path.exists():
        if (root/'abort.json').exists(): raise RuntimeError('Batch aborted before common release.')
        if time.monotonic()-started>timeout: raise TimeoutError(f'Timed out waiting for {path.name}.')
        time.sleep(.2)
    if (root/'abort.json').exists(): raise RuntimeError('Batch aborted.')


def worker(args):
    from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation, _configure_stage_compilation
    from brain_uav.trainers.v2_formal_training import (build_v2_stage_engine,prepare_v2_stage_initialization,
        build_v2_periodic_snapshot,V2FormalStageTrainer,v2_initial_state_summary)
    from brain_uav.trainers.v2_reporting import V2ExperimentReporter
    from brain_uav.trainers.v2_validation import load_v2_validation_pool
    root=args.root.resolve(); job=next(j for j in matrix() if j['id']==args.job)
    directory=root/args.job;directory.mkdir(parents=True,exist_ok=True)
    (directory/'review').mkdir(exist_ok=True);(directory/'other').mkdir(exist_ok=True)
    hook=OnlineResearch(job['arm'],directory)
    started=time.perf_counter();evaluation_seconds=0.
    summary=dict(execution_completed=False,diagnostic_only=True,formal_passed=False,job=job)
    try:
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['preflight']['source']['sha256']!=identity()['sha256']: raise ValueError('Worker source mismatch.')
        config=make_config(job);bc=REPO/BC[job['model']][0]
        if sha256(bc.read_bytes()).hexdigest()!=BC[job['model']][1]: raise ValueError('Worker BC identity mismatch.')
        prepared=prepare_v2_stage_initialization(config,init_checkpoint=bc,device='cuda',model_type=job['model'],rewards=research_rewards())
        pool=load_v2_validation_pool(REPO/POOL,expected_level='easy',expected_count=100,
            expected_scenario=prepared.scenario_config,expected_uav_collision_radius=prepared.uav_collision_radius)
        if pool.content_digest!=DIGEST: raise ValueError('Worker pool digest mismatch.')
        pool_file_hash=sha256((REPO/POOL).read_bytes()).hexdigest()
        if pool_file_hash!=manifest['preflight']['inputs'][job['model']]['pool_file_sha256']:
            raise ValueError('Worker pool file identity differs from preflight.')
        from brain_uav.scripts.diagnose_v2_wd3_acc import scene_key
        hook.forbidden_scene_keys={scene_key(s['payload']) for s in pool.scenarios}
        defaults=default_v2_cuda_graph_compilation(model=job['model'],resolved_device='cuda')
        components=build_v2_stage_engine(None,config,init_checkpoint=bc,device='cuda',model_type=job['model'],
            prepared_initialization=prepared,fused_adam=defaults['fused_adam'],
            reduce_update_stat_syncs=defaults['reduce_update_stat_syncs'],gpu_td3_compute=True,
            non_blocking_batch_transfer=job['non_blocking_batch_transfer'])
        engine=components.engine
        engine.online_research_statistics=True
        initial=v2_initial_state_summary(engine)
        from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash
        initial['bc_reference_actor']=state_hash(engine.bc_reference_actor.state_dict())
        if job['target_config'] is not None: engine.configure_research_targets(job['target_config'])
        compilation=dict(defaults);compilation.pop('fused_adam');compilation.pop('reduce_update_stat_syncs')
        before_compile=time.perf_counter()
        compiled=_configure_stage_compilation(engine,pool,prepared,compile_critic_encoder=False,
            compile_target_encoders=False,**compilation)
        compile_seconds=time.perf_counter()-before_compile
        after_compile=v2_initial_state_summary(engine)
        after_compile['bc_reference_actor']=state_hash(engine.bc_reference_actor.state_dict())
        if initial!=after_compile:raise RuntimeError('Compilation/prewarm changed initial networks/Adam.')
        atomic_json(directory/'review/initialized.json',dict(job=job,initial_state_sha256=initial,
            compilation=compiled,pid=os.getpid(),time=time.time()))
        hook.status('initialized')
        gate_started=time.perf_counter()
        wait_for(root/'release.json',root,args.timeout)
        gate_seconds=time.perf_counter()-gate_started
        hook.status('running',stage_steps=0)
        reporter=V2ExperimentReporter(directory/'other/metrics_reports',stage='easy',model_type=job['model'],
            scenario=prepared.scenario_config,rewards=prepared.reward_config,uav_collision_radius=prepared.uav_collision_radius,
            max_steps=500000,required_qualified_windows=config.consecutive_qualified_windows)
        reporter.start_stage(dict(research=True,formal_passed=False,job=job,initial_state_sha256=initial,
            compilation=compiled,schedule_horizon_steps=500000))
        saved=set();previous_snapshot_parameters=None
        def snapshot(requested,actual):
            nonlocal evaluation_seconds,previous_snapshot_parameters
            if (requested,actual) in saved: return
            snapshot_started=time.perf_counter()
            raw=dict(format=FORMAT,format_version=1,diagnostic_only=True,formal_passed=False,arm=job['arm'],
                target_config=job['target_config'],beta=getattr(engine,'research_beta',.5),
                controller=None if hook.acc is None else hook.acc.state(),requested_step=requested,actual_step=actual,
                snapshot=build_v2_periodic_snapshot(engine,config,stage_steps=actual,scenario=prepared.scenario_config,
                    rewards=prepared.reward_config,uav_collision_radius=prepared.uav_collision_radius,
                    seed_manifest=components.seed_manifest,initialization_source=components.initialization_source,
                    bc_schedule='smooth-mid-strong'))
            raw['snapshot']['research_only']=True
            validate_snapshot(raw)
            path=directory/'other'/f'research_requested_{requested}_actual_{actual}.pt'
            if path.exists(): raise FileExistsError(path)
            torch.save(raw,path); saved.add((requested,actual))
            vector=torch.cat([p.detach().cpu().reshape(-1) for m in (engine.actor,engine.critic1,engine.critic2) for p in m.parameters()])
            atomic_json(directory/'review'/f'snapshot_{requested}_{actual}.json',dict(requested_step=requested,
                actual_step=actual,sha256=sha256(path.read_bytes()).hexdigest(),BC_sha256=BC[job['model']][1],
                pool_sha256=pool_file_hash,pool_digest=DIGEST,source_sha256=sha256(json.dumps(manifest['preflight']['source']['sha256'],sort_keys=True).encode()).hexdigest(),
                online_parameter_l2=float(torch.linalg.vector_norm(vector)),
                delta_from_previous_snapshot_l2=None if previous_snapshot_parameters is None else float(torch.linalg.vector_norm(vector-previous_snapshot_parameters)),
                update_counts=raw['snapshot']['engine_checkpoint']['update_counts']))
            previous_snapshot_parameters=vector
            hook.status('evaluating',requested_step=requested,actual_step=actual)
            before_evaluation=training_identity(engine,components.exploration_rng,components.selector)
            # Independent process owns all evaluation state/RNG/SNN resets.
            command=[sys.executable,'-u','-B','-m','brain_uav.scripts.run_v2_online_wd3_acc','evaluate',
                '--checkpoint',str(path),'--root',str(directory/'other'/f'eval_{requested}_{actual}'),'--launch']
            begin=time.perf_counter()
            subprocess.run(command,cwd=REPO,check=True)
            evaluation_elapsed=time.perf_counter()-begin
            if training_identity(engine,components.exploration_rng,components.selector)!=before_evaluation:
                raise RuntimeError('Independent evaluation polluted live training state/RNG.')
            evaluation_seconds+=evaluation_elapsed
            hook.times['evaluation_seconds']=evaluation_seconds
            hook.times['snapshot_seconds']=hook.times.get('snapshot_seconds',0.)+time.perf_counter()-snapshot_started-evaluation_elapsed
            hook.status('running',stage_steps=actual)
        trainer=V2FormalStageTrainer(prepared.scenario_config,prepared.reward_config,config,engine,
            scenario_sources=components.scenario_generators,selector=components.selector,
            exploration_rng=components.exploration_rng,uav_collision_radius=prepared.uav_collision_radius,
            validation_runner=lambda actor: (_ for _ in ()).throw(AssertionError('No formal validation in research')),
            reporter=reporter,bc_schedule='smooth-mid-strong',cpu_deduplication=True,
            weight_snapshot_steps=(75000,150000,200000,300000),weight_snapshot_sink=snapshot,online_research=hook)
        train_start=time.perf_counter(); result=trainer.run(); train_wall=time.perf_counter()-train_start
        evaluation_during_training=evaluation_seconds;snapshot_during_training=hook.times.get('snapshot_seconds',0.)
        snapshot(400000,result.stage_steps)
        if (identity()['sha256']!=manifest['preflight']['source']['sha256'] or sha256(bc.read_bytes()).hexdigest()!=BC[job['model']][1]
                or sha256((REPO/POOL).read_bytes()).hexdigest()!=pool_file_hash):
            raise RuntimeError('Research input/source changed during execution.')
        execution=engine.local_optimization_metadata()
        compute=execution['gpu_td3_compute']
        if not compute['effective'] or compute.get('replays',0)<=0:
            raise RuntimeError('Research run has no actual GPUCompute replay evidence; no successful execution report.')
        reporter.finish_stage(dict(result.to_dict(),status='research_completed_budget',passed_validation=False,
            stop_reason='research_hard_limit',diagnostic_only=True))
        summary.update(execution_completed=True,actual_step=result.stage_steps,episodes=result.episodes,
            execution=execution,compile_initialization_seconds=compile_seconds,
            initial_gate_wait_seconds=gate_seconds,training_loop_wall_inclusive_seconds=train_wall,
            training_wall_seconds=train_wall-hook.times['pause_seconds']-evaluation_during_training-snapshot_during_training-hook.times['acc_seconds']
                -compute.get('capture_seconds',0.)-compute.get('initialization_seconds',0.),
            actual_concurrency={j['id']:json.loads(p.read_text(encoding='utf-8')) if (p:=root/j['id']/'review/progress.json').exists() else None for j in matrix()},
            **hook.times,initial_state_sha256=initial,
            acc=None if hook.acc is None else {k:v for k,v in hook.acc.state().items() if k!='episodes'})
        hook.status('completed_budget',stage_steps=result.stage_steps)
    except BaseException as exc:
        summary['error']=dict(type=type(exc).__name__,message=str(exc));hook.status('failed',error=summary['error'])
        if not (root/'release.json').exists(): atomic_json(root/'abort.json',summary['error'])
        raise
    finally:
        summary['process_wall_seconds']=time.perf_counter()-started
        atomic_json(directory/'review/summary.json',summary)


def tmux(args,*command,check=True):
    return subprocess.run(['tmux',*(['-L',args.socket] if args.socket else []),*command],
        capture_output=True,text=True,check=check)


def verify_initializations(root,jobs):
    summaries=[]
    for job in jobs:
        value=json.loads((root/job['id']/'review/initialized.json').read_text(encoding='utf-8'))
        if value['job']!=job: raise ValueError('Worker configuration differs from plan.')
        summaries.append(value)
    for model in ('ann','snn'):
        selected=[v['initial_state_sha256'] for v in summaries if v['job']['model']==model]
        if len(selected)!=8 or any(v!=selected[0] for v in selected): raise ValueError('Same-model initial state hashes differ.')
    return summaries


def launch(args):
    if args.python!=PYTHON: raise ValueError('Use the declared server Python; no silent environment substitution.')
    args.root.mkdir(parents=True,exist_ok=False)
    started=[]
    try:
        evidence=preflight(args)
        if tmux(args,'has-session','-t',args.session,check=False).returncode==0: raise FileExistsError('tmux session already exists.')
        manifest=dict(plan=plan(args),preflight=evidence,session=args.session,socket=args.socket,time=time.time())
        atomic_json(args.root/'manifest.json',manifest)
        for index,job in enumerate(matrix()):
            (args.root/job['id']/'review').mkdir(parents=True)
            command=shlex.join(['bash',str(REPO/'scripts/online_v2_wd3_acc.sh'),'--worker-run',str(args.root),job['id'],args.python,str(args.timeout)])
            if index==0: tmux(args,'new-session','-d','-s',args.session,'-n',job['id'],command)
            else: tmux(args,'new-window','-d','-t',args.session,'-n',job['id'],command)
            started.append(job['id'])
        atomic_json(args.root/'review_dispatch.json',dict(dispatched=started,not_dispatched=[],training_released=False))
        for job in matrix(): wait_for(args.root/job['id']/'review/initialized.json',args.root,args.timeout)
        initial=verify_initializations(args.root,matrix())
        if len({v['pid'] for v in initial})!=16:raise ValueError('Workers must have sixteen independent live PIDs.')
        for v in initial:os.kill(v['pid'],0)  # read-only existence probe before release
        if (args.root/'abort.json').exists():raise RuntimeError('Initialization aborted before release.')
        atomic_json(args.root/'release.json',dict(time=time.time(),initialization_checked=True,pids=[v['pid'] for v in initial]))
        print('All 16 initialized and released; training is NOT completed.',flush=True)
    except BaseException as exc:
        atomic_json(args.root/'abort.json',dict(error=str(exc),started=started,
            not_started=[j['id'] for j in matrix() if j['id'] not in started],training_released=False))
        print(f'Batch blocked: {exc}; initialized jobs will exit before training. Logs: {args.root}',file=sys.stderr,flush=True)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',default='plan',choices=('plan','source','launch','worker','status','continue','evaluate','probe','confirmation','cuda-check'))
    p.add_argument('--root',type=Path,default=Path('outputs')/('v2_online_wd3_acc_'+time.strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--source-manifest',type=Path,default=REPO/'scripts/v2_online_wd3_acc_source.json')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--model',choices=('ann','snn'),default='snn')
    p.add_argument('--job',choices=tuple(j['id'] for j in matrix()))
    p.add_argument('--jobs',nargs='+',choices=tuple(j['id'] for j in matrix()))
    p.add_argument('--session',default='uav_online_wd3_acc')
    p.add_argument('--socket')
    p.add_argument('--python',default=PYTHON)
    p.add_argument('--timeout',type=float,default=1800.)
    p.add_argument('--launch',action='store_true')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode=='worker' and args.job is None:raise ValueError('worker requires an explicit --job.')
    if args.mode=='cuda-check' and args.job is not None and not args.job.startswith(args.model+'_'):
        raise ValueError('CUDA model/job mismatch.')
    if args.mode=='status':
        values={j['id']:json.loads(p.read_text(encoding='utf-8')) if (p:=args.root/j['id']/'review/progress.json').exists() else 'not_initialized' for j in matrix()}
        for value in values.values():
            if isinstance(value,dict) and value['state'] in ('initialized','running','evaluating','paused_at_200k'):
                try:os.kill(value['pid'],0)
                except (ProcessLookupError,PermissionError):value.update(state='failed',reason='process_lost_or_unverifiable; no exact resume from weights')
        print(json.dumps(values,indent=2));return 0
    if not args.launch:
        print(json.dumps(plan(args),indent=2));return 0
    if args.timeout<=0: raise ValueError('Timeout must be positive.')
    if args.mode in ('evaluate','probe') and args.checkpoint is None: raise ValueError('--checkpoint required.')
    if args.mode=='source':
        if args.root.exists(): raise FileExistsError(args.root)
        args.root.parent.mkdir(parents=True,exist_ok=True);args.root.write_text(json.dumps(identity(),indent=2),encoding='utf-8')
    elif args.mode=='launch': launch(args)
    elif args.mode=='worker': worker(args)
    elif args.mode=='continue':
        if not args.jobs: raise ValueError('Explicit --jobs required; never continue all implicitly.')
        for job in args.jobs:
            path=args.root/job;v=json.loads((path/'review/progress.json').read_text(encoding='utf-8'))
            if v['state']!='paused_at_200k': raise ValueError(f'{job} is not paused.')
            os.kill(v['pid'],0)  # existence only, never terminate
            cmdline=Path(f"/proc/{v['pid']}/cmdline")
            if not cmdline.exists() or job not in cmdline.read_bytes().decode(errors='replace') or 'run_v2_online_wd3_acc' not in cmdline.read_bytes().decode(errors='replace'):
                raise RuntimeError('Paused worker process identity lost; snapshots cannot provide exact continuation.')
            atomic_json(path/'continue_to_400k.json',dict(pid=v['pid'],arm=v['arm'],actual_step=v['actual_step'],continue_to=400000))
    elif args.mode in ('evaluate','probe'):
        from brain_uav.scripts.diagnose_v2_online_research import run
        run(args)
    elif args.mode=='cuda-check':
        from brain_uav.scripts.diagnose_v2_online_research import cuda_check
        cuda_check(args)
    elif args.mode=='confirmation': raise ValueError('Confirmation is plan-only; candidate lock and new authorization required.')
    else: raise ValueError('Use an executable mode.')
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
