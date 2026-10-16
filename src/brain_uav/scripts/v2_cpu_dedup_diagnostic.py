"""Serial default-off CPU reuse acceptance via existing GPU compute workers.

Correctness hashes add synchronization and are never throughput measurements.
No checkpoint copying, tensor dumps, profiler, server connection or training.
"""
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
import json
import os
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


def compare_environment_evidence(a, b):
    """Shared fixed actions only; trained-policy outputs/counters are not a gate."""
    keys = ('step', 'measured', 'action_bits', 'input', 'output', 'state', 'outcome')
    return compare_evidence([{k: row[k] for k in keys} for row in a],
                            [{k: row[k] for k in keys} for row in b]) | {
        'scope': 'Exact environment/observation/info under one recorded fixed action sequence; independent of TD3 trajectory drift.'}


def checked_action(engine, observation, exploration_rng, **kwargs):
    """Same actor and input, with identical RNG; publish the second call's RNG."""
    from copy import deepcopy
    def state():
        return dict(cpu=torch.get_rng_state().clone(),
            cuda=torch.cuda.get_rng_state(engine.device).clone() if engine.device.type=='cuda' else None,
            python=random.getstate(), numpy=deepcopy(np.random.get_state()),
            exploration=deepcopy(exploration_rng.bit_generator.state))
    before = state()
    a = engine.select_action(observation, exploration_rng=exploration_rng,
                            cpu_deduplication=False, **kwargs)
    action_bits = digest(a)
    after = state()
    torch.set_rng_state(before['cpu'])
    if before['cuda'] is not None: torch.cuda.set_rng_state(before['cuda'], engine.device)
    random.setstate(before['python']); np.random.set_state(before['numpy'])
    exploration_rng.bit_generator.state = before['exploration']
    b = engine.select_action(observation, exploration_rng=exploration_rng,
                            cpu_deduplication=True, **kwargs)
    if action_bits != digest(b) or digest(after) != digest(state()):
        raise AssertionError('CPU action-input reuse differs at identical actor/input/RNG; timing skipped.')
    return b


def require_strict(report, *, basis='strict'):
    """No machine-verifiable whitelist exists for the historical ANN exception."""
    strict = report.get('passed') is True
    verdict = dict(strict_passed=strict, engineering_accepted=False, acceptance_basis=basis)
    if basis != 'strict':
        verdict['gap'] = ('Historical ANN evidence has no complete machine-verifiable acceptance contract; '
                          'cannot identify all allowed fields/states or exclude new failures. Timing blocked.')
    return verdict


def capture_controlled_worker(argv, output):
    """Launch either saved or current source; retain owned existing CPU snapshots.

    The wrapper itself is current code, but profile/engine imports follow the worker's
    PYTHONPATH. It changes only evidence collection, never update execution.
    """
    from brain_uav.scripts import profile_v2_td3 as profile, v2_gpu_td3_diagnostic as diagnostic
    original = diagnostic.snapshot
    snapshots = []
    def collect(*args, **kwargs):
        result = original(*args, **kwargs)
        snapshots.append(result[0])
        return result
    diagnostic.snapshot = collect
    try:
        profile.main(argv)
    finally:
        diagnostic.snapshot = original
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshots, output)


def compare_adopted_snapshots(before, after):
    """Direct old-compute/new-compute regression, original tolerances, no new waiver."""
    from brain_uav.scripts.v2_gpu_compute_diagnostic import state_difference
    if not before or len(before) != len(after):
        return dict(passed=False, reason='Missing or unequal adopted snapshot counts')
    for index, (a, b) in enumerate(zip(before, after)):
        result = state_difference(a, b)
        if not result['passed']:
            return dict(passed=False, index=index, result=result)
    return dict(passed=True, snapshot_count=len(before), rtol=1e-4, atol=1e-5)


def read_summary(root):
    path=root/'other/local_optimization_summary.json'
    return json.loads((path if path.exists() else root/'local_optimization_summary.json').read_text(encoding='utf-8'))


def run(args):
    from brain_uav.scripts import profile_v2_td3 as p
    forwarded=('model','bc_checkpoint','validation_pool','expected_pool_digest','validation_seed',
        'seed','batch_size','replay_capacity','scenario_count','warmup_steps','steps_per_level',
        'comparison_bc_schedule','controlled_stage_step_offset','snn_time_window')
    defaults=p.build_parser().parse_args(['--model',args.model,'--output-dir',str(args.output_dir)])
    allowed=set(forwarded)|{'device','gpu_td3_compute','gpu_compute_environment_diagnostic','output_dir','repeats','acceptance_comparison','baseline_source','acceptance_basis'}
    root=Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Refusing nonempty deduplication output directory.')
    (root/'review').mkdir(parents=True)
    (root/'other').mkdir()
    report=dict(passed=False,diagnostic_only=True,formal_training=False,workers={},
        ab_changes=dict(
            common=['Pyramid distance/normal share one closest-boundary solve in both A and B; not measured by the switch delta.'],
            switch=['Reward-local clearance reuse', 'Action-input copy removal',
                    'Call-local ellipsoid raw intersection reuse from observation to info']),
        limitations=['No cross-step raw-intersection or static-feature cache.',
            'Fixed-pool fragments do not establish full-course stability or all geometry/lifecycle coverage.'])
    comparison = getattr(args, 'acceptance_comparison', None)
    before_source = getattr(args, 'baseline_source', None)
    report['acceptance_comparison'] = comparison or 'legacy_cpu_dedup_only'
    report['strict_td3_passed'] = False
    report['strict_passed'] = False
    report['engineering_accepted'] = False
    report['acceptance_basis'] = getattr(args, 'acceptance_basis', 'strict')
    report['common_configuration'] = {'ellipsoid_segment_clearance_cache': True,
        'pyramid_combined_geometry': True, 'schedule_steps': 500000}
    report['ab_changes']['common'].append('Exact ellipsoid clearance cache ON in both sides.')
    if comparison:
        report['ab_changes']['switch'] = (['Pre-cleanup adopted path versus post-cleanup default-equivalent path']
            if comparison == 'A' else ['GPU compute and CPU deduplication OFF versus ON; ellipsoid cache remains ON'])
    try:
        if report['acceptance_basis'] != 'strict':
            report['acceptance'] = require_strict({}, basis=report['acceptance_basis'])
            raise RuntimeError(report['acceptance']['gap'])
        if comparison == 'A' and (before_source is None or not (Path(before_source)/'src/brain_uav').is_dir()):
            raise ValueError('Comparison A requires --baseline-source pointing to the saved source directory.')
        if comparison == 'A':
            source = Path(before_source).resolve()
            manifest_path = source.parent/'manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            for relative, expected in manifest['files'].items():
                path = (source/relative).resolve()
                if not path.is_relative_to(source) or sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('Saved baseline source hash mismatch: '+relative)
            report['baseline_source'] = {'path': str(source), 'head': manifest['head'],
                'manifest_sha256': sha256(manifest_path.read_bytes()).hexdigest()}
        for name,value in vars(args).items():
            if name not in allowed and value!=getattr(defaults,name,None):
                raise ValueError('Unsupported deduplication-suite option: '+name)
        if args.model not in ('ann','snn') or args.device!='cuda' or not args.gpu_td3_compute:
            raise ValueError('This isolated suite requires ANN/SNN, CUDA and --gpu-td3-compute.')
        if args.bc_checkpoint is None or args.validation_pool is None or args.repeats<1:
            raise ValueError('BC, fixed pool and positive repeats are required.')
        base=[sys.executable,'-u','-m','brain_uav.scripts.profile_v2_td3','--device','cuda',
            '--gpu-td3-compute','--gpu-compute-worker','timing']
        for name in forwarded:
            if getattr(args,name) is not None:
                value = getattr(args,name)
                if name in ('bc_checkpoint','validation_pool'):
                    value = Path(value).resolve()
                base+=['--'+name.replace('_','-'),str(value)]

        def invocation(name, *, old=False):
            command = list(base)
            options = {}
            if old:
                source = Path(before_source).resolve()
                options = {'cwd': str(source), 'env': {**os.environ, 'PYTHONPATH': str(source/'src')}}
            return command, options

        # Existing full lifecycle correctness gate, unchanged tolerances.
        # A validates each revision; B validates the current original/update pair.
        for revision,old in ([('before',True),('after',False)] if comparison=='A' else [('after',False)]):
            reference = root/'other'/('strict_'+revision+'_baseline')
            for enabled in (False,True):
                name='strict_'+revision+('_compute' if enabled else '_baseline')
                command,options=invocation(name,old=old)
                command[command.index('timing')] = 'correctness'
                command.remove('--gpu-td3-compute')
                command += ['--output-dir',str(root/'other'/name),'--correctness-updates','8']
                if enabled:
                    command += ['--gpu-td3-compute',
                        '--td3-block-reference' if old else '--gpu-compute-reference', str(reference/'reference.pt')]
                if comparison == 'A' and enabled:
                    wrapper = str(Path(__file__).resolve())
                    payload = ('import runpy,sys; runpy.run_path(sys.argv[1])["capture_controlled_worker"]'
                               '(sys.argv[3:],sys.argv[2])')
                    command = [sys.executable, '-u', '-c', payload, wrapper,
                        str(root/'other'/name/'adopted_snapshots.pt'), *command[4:]]
                with (root/'review'/f'{name}.log').open('w',encoding='utf-8') as log:
                    rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False,**options).returncode
                report['workers'][name]={'exit_code':rc,'command':command,'source':str(before_source) if old else 'current'}
                strict_path = root/'other'/name/'correctness.json'
                evidence = json.loads(strict_path.read_text(encoding='utf-8')) if strict_path.is_file() else {}
                report['workers'][name].update(require_strict(evidence))
                report['workers'][name]['evidence'] = str(strict_path.relative_to(root))
                if rc:
                    raise RuntimeError('Strict TD3 correctness failed; timing skipped: '+name)
                if not evidence.get('passed') is True:
                    raise RuntimeError('Missing or failing strict TD3 evidence; timing skipped: '+name)
        if comparison == 'A':
            before = torch.load(root/'other/strict_before_compute/adopted_snapshots.pt', map_location='cpu', weights_only=False)
            after = torch.load(root/'other/strict_after_compute/adopted_snapshots.pt', map_location='cpu', weights_only=False)
            report['cross_revision_td3'] = compare_adopted_snapshots(before, after)
            if not report['cross_revision_td3']['passed']:
                raise RuntimeError('Cross-revision adopted-path TD3 regression; timing skipped.')
        report['strict_td3_passed']=True

        def worker(name,enabled,*,evidence=False,instrumented=False,fixed_actions=None):
            output=root/'other'/name
            side_a = name.endswith('_A')
            old = comparison == 'A' and side_a
            compute = not (comparison == 'B' and side_a)
            cpu = (True if comparison == 'A' else enabled)
            command,options=invocation(name,old=old)
            command+=['--output-dir',str(output),'--repeats',str(1 if evidence or instrumented else args.repeats)]
            # Current workers always receive one explicit positive/negative value.
            if not old or not compute:
                command.remove('--gpu-td3-compute')
            if not old:
                command += ['--gpu-td3-compute' if compute else '--no-gpu-td3-compute',
                            '--cpu-deduplication' if cpu else '--no-cpu-deduplication']
            elif cpu:
                command+=['--cpu-deduplication']
            if evidence: command+=['--cpu-deduplication-evidence']
            if fixed_actions is not None: command+=['--cpu-deduplication-fixed-actions', str(fixed_actions)]
            if instrumented: command+=['--gpu-compute-environment-diagnostic']
            with (root/'review'/f'{name}.log').open('w',encoding='utf-8') as log:
                rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False,**options).returncode
            report['workers'][name]=dict(exit_code=rc,command=command,
                source=str(before_source) if old else 'current',
                gpu_td3_compute=compute,cpu_deduplication=cpu,
                cache_ellipsoid_segment_clearance=True)
            if rc: raise RuntimeError('Deduplication worker failed: '+name)
            data=read_summary(output)
            if data['cpu_deduplication']!=cpu:
                raise RuntimeError('Effective deduplication flag differs: '+name)
            for group in data['runs'].values():
                for item in group:
                    compute_expected = not (comparison == 'B' and side_a)
                    compute=item['local_optimizations']['gpu_td3_compute']
                    if (item['critic_encoder_compile']['valid_for_speed_comparison'] is not True
                            or (compute_expected and (not compute['effective'] or compute['measurement_new_captures']!=0))
                            or (not compute_expected and compute['effective'])):
                        raise RuntimeError('Unstable compile/capture in '+name)
            return output,data

        a,ac=worker('correctness_A',False,evidence=True)
        b,bc=worker('correctness_B',True,evidence=True, fixed_actions=a/'other/rollout_0.json')
        for key in ('checkpoint_sha256','pool_sha256','original_pool_digest','selected_pool_digest',
                    'config','reward_config','software','compilation_requested','scenario_ids','scenario_shapes','bc_schedule'):
            if ac[key]!=bc[key]: raise RuntimeError('Correctness contract differs: '+key)
        evidence=[json.loads((path/'other/rollout_0.json').read_text(encoding='utf-8'))['steps'] for path in (a,b)]
        report['trajectory_diagnostic']=compare_evidence(*evidence)
        report['correctness']=compare_environment_evidence(*evidence)
        if not evidence[1] or not all(row.get('cpu_same_input_passed') is True for row in evidence[1]):
            raise RuntimeError('Missing/failing same-input CPU environment/action/RNG gate; timing skipped.')
        if not report['correctness']['passed']:
            raise RuntimeError('Fixed-action environment regression; timing skipped.')
        if any(sum(bool(x['measured']) for x in rows)!=args.steps_per_level for rows in evidence):
            raise RuntimeError('Incomplete correctness measurement coverage; timing skipped.')
        report['strict_passed'] = report['strict_td3_passed'] and report['correctness']['passed']
        report['trajectory_diagnostic']['note'] = 'B replays A actions; policy predictions/counters/RNG differences are diagnostic only. CPU same-input RNG is checked locally.'
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
            report['timing'][name]=dict(cpu_deduplication=True if comparison=='A' else enabled,runs=rows,
                mean_ms={k:mean(r['milliseconds_per_step'][k] for r in rows) for k in rows[0]['milliseconds_per_step']},
                stdev_ms={k:stdev(r['milliseconds_per_step'][k] for r in rows) if len(rows)>1 else 0.
                    for k in rows[0]['milliseconds_per_step']})
        report['milliseconds_saved_per_step']={k:report['timing']['A']['mean_ms'][k]-report['timing']['B']['mean_ms'][k]
            for k in report['timing']['A']['mean_ms']}
        report['timing_acceptance_basis'] = report['acceptance_basis']
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
    parser=build_parser()
    parser.add_argument('--acceptance-comparison', choices=('A','B'))
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--acceptance-basis', choices=('strict','ann-existing-engineering'), default='strict',
        help='Historical ANN engineering selection is blocked until its exact accepted scope can be verified.')
    run(parser.parse_args(argv))


if __name__=='__main__':
    main()
