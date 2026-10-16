"""Serial default-off CPU reuse acceptance via existing GPU compute workers.

Correctness hashes add synchronization and are never throughput measurements.
No checkpoint copying, tensor dumps, profiler, server connection or training.
"""
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
import json
import pickle
import random
import subprocess
import sys
import traceback

import numpy as np
import torch


def digest(value):
    """Exact numeric hash, immediately consumed; no live storage retained."""
    def numeric(v):
        if isinstance(v,torch.Tensor):
            v=v.detach().cpu().numpy()
        if isinstance(v,np.ndarray):
            return ('array',v.dtype.str,v.shape,v.tobytes(order='C'))
        if isinstance(v,dict):
            return [(k,numeric(v[k])) for k in sorted(v)]
        if isinstance(v,(list,tuple)):
            return [numeric(x) for x in v]
        return v
    return sha256(pickle.dumps(numeric(value),protocol=4)).hexdigest()


def step_evidence(e, exploration_rng, step, measured, observation, action, following,
                  reward, terminated, truncated, info, state, line_safe):
    def fields(obs):
        return {n:getattr(obs,n) for n in ('ego_features','goal_features','zone_features','presence_mask')}
    return dict(step=step,measured=measured,action=np.asarray(action).tolist(),
        action_bits=digest(np.asarray(action)),input=digest(fields(observation)),
        output=digest(fields(following)),state=digest(state),
        outcome=digest((reward,terminated,truncated,info,line_safe)),
        counts=[getattr(e,n) for n in ('update_count','critic_update_count',
            'critic_target_update_count','actor_update_count','last_total_steps')],
        rng=digest(dict(torch_cpu=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state(e.device) if e.device.type=='cuda' else None,
            python=random.getstate(),numpy=np.random.get_state(),
            replay=e.replay.rng.bit_generator.state,exploration=exploration_rng.bit_generator.state)),
        precision=[torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,torch.are_deterministic_algorithms_enabled()])


def compare_evidence(a,b):
    difference=None
    if len(a)!=len(b) or not a:
        difference=dict(field='step_count',baseline=len(a),candidate=len(b))
    else:
        for index,(x,y) in enumerate(zip(a,b)):
            for name in sorted(x.keys()|y.keys()):
                if x.get(name)!=y.get(name):
                    difference=dict(index=index,field=name,baseline=x.get(name),candidate=y.get(name))
                    break
            if difference: break
    return dict(passed=difference is None,steps=len(a),first_difference=difference,
        scope='Exact actions, observations/mask, environment, counters and all RNG streams; warmup and measurement. Not the full TD3 correctness matrix.')


def read_summary(root):
    path=root/'other/local_optimization_summary.json'
    return json.loads((path if path.exists() else root/'local_optimization_summary.json').read_text(encoding='utf-8'))


def run(args):
    from brain_uav.scripts import profile_v2_td3 as p
    forwarded=('model','bc_checkpoint','validation_pool','expected_pool_digest','validation_seed',
        'seed','batch_size','replay_capacity','scenario_count','warmup_steps','steps_per_level',
        'comparison_bc_schedule','controlled_stage_step_offset','snn_time_window')
    defaults=p.build_parser().parse_args(['--model',args.model,'--output-dir',str(args.output_dir)])
    allowed=set(forwarded)|{'device','gpu_td3_compute','gpu_compute_environment_diagnostic','output_dir','repeats'}
    root=Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Refusing nonempty deduplication output directory.')
    (root/'review').mkdir(parents=True)
    (root/'other').mkdir()
    report=dict(passed=False,diagnostic_only=True,formal_training=False,workers={},
        limitations=['Only reward-local clearance reuse and action-input copy removal are enabled.',
            'No raw geometry/static-feature caching; formal training entry is not wired.',
            'Fixed-pool fragments do not establish full-course stability or all geometry/lifecycle coverage.'])
    try:
        for name,value in vars(args).items():
            if name not in allowed and value!=getattr(defaults,name):
                raise ValueError('Unsupported deduplication-suite option: '+name)
        if args.model!='snn' or args.device!='cuda' or not args.gpu_td3_compute:
            raise ValueError('This isolated suite requires SNN, CUDA and --gpu-td3-compute.')
        if args.bc_checkpoint is None or args.validation_pool is None or args.repeats<1:
            raise ValueError('BC, fixed pool and positive repeats are required.')
        base=[sys.executable,'-u','-m','brain_uav.scripts.profile_v2_td3','--device','cuda',
            '--gpu-td3-compute','--gpu-compute-worker','timing']
        for name in forwarded:
            if getattr(args,name) is not None:
                base+=['--'+name.replace('_','-'),str(getattr(args,name))]

        def worker(name,enabled,*,evidence=False,instrumented=False):
            output=root/'other'/name
            command=base+['--output-dir',str(output),'--repeats',str(1 if evidence or instrumented else args.repeats)]
            if enabled: command+=['--cpu-deduplication']
            if evidence: command+=['--cpu-deduplication-evidence']
            if instrumented: command+=['--gpu-compute-environment-diagnostic']
            with (root/'review'/f'{name}.log').open('w',encoding='utf-8') as log:
                rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            report['workers'][name]=dict(exit_code=rc,command=command)
            if rc: raise RuntimeError('Deduplication worker failed: '+name)
            data=read_summary(output)
            if data['cpu_deduplication']!=enabled:
                raise RuntimeError('Effective deduplication flag differs: '+name)
            for group in data['runs'].values():
                for item in group:
                    compute=item['local_optimizations']['gpu_td3_compute']
                    if (item['critic_encoder_compile']['valid_for_speed_comparison'] is not True
                            or not compute['effective'] or compute['measurement_new_captures']!=0):
                        raise RuntimeError('Unstable compile/capture in '+name)
            return output,data

        a,ac=worker('correctness_A',False,evidence=True)
        b,bc=worker('correctness_B',True,evidence=True)
        for key in ('checkpoint_sha256','pool_sha256','original_pool_digest','selected_pool_digest',
                    'config','reward_config','software','compilation_requested','scenario_ids','scenario_shapes','bc_schedule'):
            if ac[key]!=bc[key]: raise RuntimeError('Correctness contract differs: '+key)
        evidence=[json.loads((path/'other/rollout_0.json').read_text(encoding='utf-8'))['steps'] for path in (a,b)]
        report['correctness']=compare_evidence(*evidence)
        if not report['correctness']['passed']:
            raise RuntimeError('Exact rollout/action/RNG comparison failed; timing skipped.')
        if any(sum(bool(x['measured']) for x in rows)!=args.steps_per_level for rows in evidence):
            raise RuntimeError('Incomplete correctness measurement coverage; timing skipped.')
        report['provenance']={k:ac[k] for k in ('checkpoint','checkpoint_sha256','original_pool_digest',
            'selected_pool_digest','scenario_ids','scenario_shapes','config','reward_config','software','compilation_requested')}
        covered={shape for item in ac['scenario_shapes'] for shape in item['shape_types']}
        report['geometry_types_not_covered']=sorted({'sphere','ellipsoid','box','triangular_pyramid','quadrangular_pyramid'}-covered)
        report['timing']={}
        for name,enabled in (('A',False),('B',True)):
            _,data=worker('timing_'+name,enabled)
            if not data['valid_for_speed_comparison'] or data['cpu_deduplication_evidence']:
                raise RuntimeError('Instrumented or invalid timing: '+name)
            rows=[]
            for group in data['runs'].values():
                for item in group:
                    t=item['timing']; n=item['measured_steps']
                    rows.append(dict(repeat=item['repeat'],measured_steps=n,scenario_coverage=item['scenario_coverage'],
                        milliseconds_per_step={key:1000*t[key]/n for key in (
                            'action_inference_wall_seconds','pre_action_geometry_wall_seconds',
                            'environment_step_wall_seconds','td3_update_wall_seconds','total_wall_seconds')},
                        gpu_compute=item['local_optimizations']['gpu_td3_compute'],
                        compilation=item['critic_encoder_compile'],server_load_before=item['server_load_before'],
                        server_load_after=item['server_load_after']))
                    rows[-1]['milliseconds_per_step']['td3_with_completion_wall_seconds']=(
                        1000*item['batch_transfer_window_completion']['td3_wall_including_final_window_wait_seconds']/n)
            report['timing'][name]=dict(cpu_deduplication=enabled,runs=rows,
                mean_ms={k:mean(r['milliseconds_per_step'][k] for r in rows) for k in rows[0]['milliseconds_per_step']},
                stdev_ms={k:stdev(r['milliseconds_per_step'][k] for r in rows) if len(rows)>1 else 0.
                    for k in rows[0]['milliseconds_per_step']})
        report['milliseconds_saved_per_step']={k:report['timing']['A']['mean_ms'][k]-report['timing']['B']['mean_ms'][k]
            for k in report['timing']['A']['mean_ms']}
        report['measurement_note']='Existing short diagnostic wall sections include CPU submission/wait. Full-step time includes final completion wait. Evidence and environment instrumentation are separate; no automatic long-run speed acceptance.'
        if args.gpu_compute_environment_diagnostic:
            path,_=worker('environment_B',True,instrumented=True)
            p._strict_json_write(root/'review/environment_summary.json',
                json.loads((path/'review/environment_summary.json').read_text(encoding='utf-8')))
            report['environment_summary']='review/environment_summary.json'
        report['passed']=True
    except BaseException as exc:
        report['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        p._strict_json_write(root/'review/summary.json',report)
    return report


def main(argv=None):
    from brain_uav.scripts.profile_v2_td3 import build_parser
    run(build_parser().parse_args(argv))


if __name__=='__main__':
    main()
