"""Small isolated A/B driver around existing formal correctness/timing workers."""
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
import json
import subprocess
import sys
import numpy as np
import torch
from brain_uav.observations import V2Observation
from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer


def cpu_batches(mixed, nonempty, action_dim, seed):
    """Actual CPU replay sampling, including empty and dynamically cropped batches."""
    result=[]
    for case,observation in enumerate((mixed,nonempty,mixed)):
        observation=observation.to('cpu')
        b=observation.batch_size
        replay=V2ReplayBuffer(b,action_dim,observation.max_zone_count,failure_sample_bias=2.,seed=seed+case)
        for i in range(b):
            mask=observation.presence_mask[i].numpy()
            zones=observation.zone_features[i].numpy()[mask] if case else observation.zone_features[i].numpy()[:0]
            obs=V2Observation(observation.ego_features[i].numpy(),observation.goal_features[i].numpy(),
                zones,np.ones(len(zones),np.bool_))
            replay.add(obs,np.zeros(action_dim,np.float32),float(i+case*b),obs,False,near_goal=i%2==0)
        replay.mark_failure_slots([(i,int(replay.write_id[i])) for i in range(0,b,3)])
        result.append(replay.sample(b))
    return result


def fields(batch):
    result={}
    for prefix in ('obs','next_obs'):
        obs=getattr(batch,prefix)
        for name in ('ego_features','goal_features','zone_features','presence_mask'):
            result[prefix+'.'+name]=getattr(obs,name)
    for name in ('action','reward','done','success','near_goal','line_to_goal_safe'):
        result[name]=getattr(batch,name)
    return result


def check_transfers(batches,device,repeats):
    """Completion wall and submission wall are distinct; no network timing here."""
    if torch.device(device).type!='cuda': raise ValueError('Transfer validation requires real CUDA.')
    if repeats<1: raise ValueError('Transfer repeat count must be positive.')
    report={'scope':'pageable CPU batch -> CUDA, completion includes one end-window synchronization',
        'groups':{},'exact_fields':True,'consecutive_batches_checked':len(batches)}
    for enabled in (False,True):
        runs=[]
        # Untimed preparation avoids attributing initial device allocation to steady copies.
        warm=[b.to(device,non_blocking=enabled) for b in batches]
        torch.cuda.synchronize(device); del warm
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start=perf_counter()
            outputs=[b.to(device,non_blocking=enabled) for b in batches]
            submitted=perf_counter()-start
            torch.cuda.synchronize(device); completed=perf_counter()-start
            for source,actual in zip(batches,outputs):
                if source.sample_failure_fraction!=actual.sample_failure_fraction:
                    raise AssertionError('sample_failure_fraction changed during transfer.')
                for name,value in fields(source).items():
                    got=fields(actual)[name]
                    target=torch.device(device)
                    target=torch.device('cuda',torch.cuda.current_device()) if target.index is None else target
                    if value.shape!=got.shape or value.dtype!=got.dtype or got.device!=target:
                        raise AssertionError('Transfer metadata changed: '+name)
                    torch.testing.assert_close(value,got.cpu(),rtol=0,atol=0)
            runs.append({'batch_count':len(batches),'submission_seconds':submitted,'completion_seconds':completed})
        report['groups']['nonblocking' if enabled else 'baseline']={'requested':enabled,'effective':enabled,'runs':runs}
    return report


def run(args):
    from brain_uav.scripts import profile_v2_td3 as p
    conflicts=('gpu_td3_update','gpu_td3_suite','gpu_td3_worker','gpu_td3_sampling_cost',
        'pinned_batch_transfer','compile_critic_update_front','compile_actor_objective','cuda_graph_critic_tail',
        'window_update_statistics','reuse_update_buffers','aggregate_relation_values_first','single_observation_fast_path',
        'td3_block_suite','td3_block_correctness','td3_block_comparison','td3_block_attribution',
        'local_optimization_correctness','local_optimization_comparison','critic_tail_correctness','critic_tail_comparison',
        'snn_path_consistency','detailed_profiler_updates','compiled_performance_diagnostic_updates')
    if any(getattr(args,n) for n in conflicts) or args.device!='cuda' or args.comparison_eager_cpu:
        raise ValueError('Batch transfer comparison requires CUDA and original formal path, without other experiments/profiler.')
    if args.bc_checkpoint is None or args.validation_pool is None:
        raise ValueError('Matching BC and easy fixed pool are required.')
    if args.batch_transfer_worker:
        args.td3_block_choice='nonblocking' if args.non_blocking_batch_transfer else 'baseline'
        args.td3_block_correctness=args.batch_transfer_worker=='correctness'
        args.td3_block_comparison=args.batch_transfer_worker=='timing'
        args.correctness_updates=max(8,args.correctness_updates)
        return p.run_local_optimization_comparison(args)
    if args.non_blocking_batch_transfer:
        raise ValueError('The serial driver owns baseline/nonblocking selection.')
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Refusing nonempty comparison directory.')
    output.mkdir(parents=True,exist_ok=True)
    report={'diagnostic_only':True,'passed':False,'model':args.model,'groups':{}}
    report['timing_scope']='batch_preparation is CPU submission/allocation/validation wall, not transfer completion. Independent transfer check synchronizes matching window boundaries. TD3 totals below include the original timing window completion wait; full environment totals already include it.'
    forwarded=('model','bc_checkpoint','validation_pool','validation_seed','seed','batch_size','replay_capacity',
        'scenario_count','warmup_steps','steps_per_level','repeats','controlled_stage_step_offset',
        'comparison_bc_schedule','snn_time_window','correctness_updates')
    baseline_ok=False
    try:
        # Both correctness processes complete before either throughput worker starts.
        for enabled in (False,True):
            group='nonblocking' if enabled else 'baseline'
            item=report['groups'][group]={'requested':enabled,'correctness_passed':False}
            if enabled and not baseline_ok: item['skipped']='baseline correctness failed'; continue
            command=[sys.executable,'-u','-m','brain_uav.scripts.profile_v2_td3','--device','cuda']
            for name in forwarded: command+=['--'+name.replace('_','-'),str(getattr(args,name))]
            if enabled: command+=['--non-blocking-batch-transfer','--td3-block-reference',str(output/'baseline_correctness'/'reference.pt')]
            with (output/(group+'_correctness.log')).open('w',encoding='utf-8') as log:
                rc=subprocess.run(command+['--batch-transfer-worker','correctness','--output-dir',str(output/(group+'_correctness'))],
                    stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            item['correctness_exit_code']=rc; item['correctness_passed']=rc==0
            if not enabled: baseline_ok=rc==0
        if not all(v['correctness_passed'] for v in report['groups'].values()):
            raise RuntimeError('Transfer/update correctness failed; this model timing skipped.')
        for enabled in (False,True):
            group='nonblocking' if enabled else 'baseline'
            command=[sys.executable,'-u','-m','brain_uav.scripts.profile_v2_td3','--device','cuda']
            for name in forwarded: command+=['--'+name.replace('_','-'),str(getattr(args,name))]
            if enabled: command+=['--non-blocking-batch-transfer']
            with (output/(group+'_timing.log')).open('w',encoding='utf-8') as log:
                rc=subprocess.run(command+['--batch-transfer-worker','timing','--output-dir',str(output/(group+'_timing'))],
                    stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            report['groups'][group]['timing_exit_code']=rc
            if rc: raise RuntimeError('Timing worker failed: '+group)
            data=json.loads((output/(group+'_timing')/'local_optimization_summary.json').read_text())
            if not data['valid_for_speed_comparison']: raise RuntimeError('Measurement compiled new graphs: '+group)
            report['groups'][group]['effective']=data['runs'][group][0]['local_optimizations']['non_blocking_batch_transfer']
            runs=data['runs'][group]
            report['groups'][group]['raw_runs']=[dict(environment_seconds=r['timing']['total_wall_seconds'],
                td3_seconds=r['timing']['td3_update_wall_seconds'],
                completion=r['batch_transfer_window_completion'],
                batch_submission=r['timing']['td3_update_breakdown']['overall_weighted']['sections']['batch_preparation'],
                measured_steps=r['measured_steps'],compilation=r['critic_encoder_compile']) for r in runs]
            report['groups'][group]['mean_step_seconds']=mean(r['timing']['total_wall_seconds']/r['measured_steps'] for r in runs)
            report['groups'][group]['stdev_step_seconds']=stdev(r['timing']['total_wall_seconds']/r['measured_steps'] for r in runs) if len(runs)>1 else 0.
            report['groups'][group]['mean_td3_seconds']=mean(r['batch_transfer_window_completion']['td3_wall_including_final_window_wait_seconds'] for r in runs)
        a,b=report['groups']['baseline'],report['groups']['nonblocking']
        report['time_saved_percent']=100*(1-b['mean_step_seconds']/a['mean_step_seconds'])
        report['overall_speedup_observed']=b['mean_step_seconds']<a['mean_step_seconds'] and b['mean_td3_seconds']<a['mean_td3_seconds']
        report['verdict']='short comparison improved both totals; stability/adoption not established' if report['overall_speedup_observed'] else '未取得整体加速'
        report['passed']=True
    finally: p._strict_json_write(output/'summary.json',report)
    return report
