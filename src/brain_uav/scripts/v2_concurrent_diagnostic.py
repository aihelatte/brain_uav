"""Same-host ANN/SNN load comparison; plan-only unless --launch is explicit.

Fresh BC plus a counter offset is not a resumed training state. This bounded
loop omits formal reporting, three-view plots and candidate validation load.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class WorkerWindow:
    """One batch directory, one ready file per owned worker; no cross-host clock."""
    def __init__(self, directory, worker_id, *, timeout):
        if not worker_id or Path(worker_id).name != worker_id:
            raise ValueError('Invalid worker id.')
        self.root, self.worker_id, self.timeout = Path(directory), worker_id, timeout
        self.window = None
        self.next_abort_check = 0.

    def check_abort(self):
        if (self.root / 'abort.json').exists():
            raise RuntimeError(f'Batch aborted: {read_json(self.root / "abort.json")}')

    def ready(self):
        write_json(self.root / f'{self.worker_id}.ready.json', {'ready_at': time.perf_counter()})
        limit = time.perf_counter() + self.timeout
        while True:
            self.check_abort()
            if time.perf_counter() >= limit:
                raise TimeoutError('Timed out waiting for common measurement window.')
            if (self.root / 'window.json').exists():
                self.window = read_json(self.root / 'window.json')
                if time.perf_counter() >= self.window['end']:
                    raise TimeoutError('Worker missed the common measurement window.')
                if time.perf_counter() >= self.window['start']:
                    return self.window
            time.sleep(.01)

    def finished(self):
        now = time.perf_counter()
        # Bounded failure propagation without filesystem I/O on every step.
        if now >= self.next_abort_check:
            self.check_abort()
            self.next_abort_check = now + .25
        return now >= self.window['end']


def inductor_capture_snapshot(device):
    """Read only, outside measured work. Unsupported internals invalidate timing."""
    try:
        import torch
        from torch._inductor.cudagraph_trees import get_manager
        index = device.index if device.index is not None else torch.cuda.current_device()
        manager = get_manager(index, create_if_none_exists=False)
        if manager is None:
            raise RuntimeError('No action-inference graph tree manager.')
        # itertools.count.__reduce__ observes the next id without consuming it.
        return {'available': True, 'manager_id': id(manager),
                'recordings': manager.graph_counter.__reduce__()[1][0]}
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        return {'available': False, 'reason': str(exc)}


def worker_command(args, model, worker_id, root):
    root = Path(root).resolve()
    command = [sys.executable, '-u', '-m', 'brain_uav.scripts.profile_v2_td3',
        '--gpu-compute-worker', 'timing', '--gpu-td3-compute', '--cpu-deduplication',
        '--device', 'cuda', '--model', model, '--repeats', '1',
        '--bc-checkpoint', str(getattr(args, model + '_bc_checkpoint').resolve()),
        '--comparison-actor-lr', str(getattr(args, model + '_actor_lr')),
        '--comparison-critic-lr', str(args.critic_lr), '--comparison-alpha', str(args.alpha),
        '--comparison-bc-schedule', args.bc_schedule,
        '--concurrent-window-dir', str(root / 'other' / 'control'), '--concurrent-worker-id', worker_id,
        '--concurrent-timeout', str(args.ready_timeout),
        '--output-dir', str(root / 'other' / worker_id)]
    if args.non_blocking_batch_transfer:
        command += ['--non-blocking-batch-transfer']
    for key in ('validation_pool', 'expected_pool_digest', 'validation_seed', 'seed',
                'replay_capacity', 'batch_size', 'scenario_count', 'warmup_steps',
                'steps_per_level', 'controlled_stage_step_offset', 'snn_time_window'):
        value = getattr(args, key)
        command += ['--' + key.replace('_', '-'), str(value.resolve() if isinstance(value, Path) else value)]
    if worker_id in args.batch_preparation_workers:
        for key in ('batch_preparation_updates', 'batch_preparation_trace_updates', 'gpu_compute_submission_trace_updates'):
            if getattr(args, key):
                command += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    return command


def _hash(path):
    result = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def build_plan(args):
    from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation
    from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig, v2_bc_schedule_metadata
    from brain_uav.trainers.v2_td3 import V2SpikeRegularizationConfig
    from brain_uav.trainers.v2_validation import load_v2_validation_pool
    for key in ('measurement_seconds', 'ready_timeout', 'completion_timeout', 'start_delay'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if (args.repeats < 1 or args.warmup_steps < 1 or args.scenario_count < 1
            or args.steps_per_level < args.scenario_count or args.controlled_stage_step_offset < 75000):
        raise ValueError('Invalid repeat/warmup/scenario budget or stage offset.')
    if len(set(args.concurrency)) != len(args.concurrency):
        raise ValueError('Repeated concurrency conditions are not allowed.')
    submission_updates = args.gpu_compute_submission_trace_updates
    diagnostic_updates = args.batch_preparation_updates or args.batch_preparation_trace_updates or submission_updates
    if (args.batch_preparation_updates or args.batch_preparation_trace_updates) and args.non_blocking_batch_transfer:
        raise ValueError('Nonblocking A/B must run separately from batch instrumentation/trace.')
    if submission_updates and (not args.non_blocking_batch_transfer
            or args.batch_preparation_updates or args.batch_preparation_trace_updates):
        raise ValueError('Submission trace requires real nonblocking, separate from blocking batch diagnosis.')
    if diagnostic_updates:
        if not 2 <= diagnostic_updates <= (8 if args.batch_preparation_trace_updates or submission_updates else 64):
            raise ValueError('Select tensor-wall 2..64 OR trace 2..8 updates.')
        eligible = {f'{model}_{i}' for model in ('ann', 'snn') for i in range(min(args.concurrency) // 2)}
        if (not args.batch_preparation_workers or len(set(args.batch_preparation_workers)) != len(args.batch_preparation_workers)
                or not set(args.batch_preparation_workers) <= eligible):
            raise ValueError('Batch diagnostic worker ids must exist in every condition, without duplicates.')
    pool = load_v2_validation_pool(args.validation_pool, expected_level='easy',
                                   expected_master_seed=args.validation_seed)
    if pool.content_digest != args.expected_pool_digest:
        raise ValueError('Fixed pool content_digest mismatch.')
    if args.scenario_count > pool.scenario_count:
        raise ValueError('Requested scenes exceed fixed pool.')
    configurations = {}
    for model in ('ann', 'snn'):
        path = getattr(args, model + '_bc_checkpoint')
        if not path.is_file():
            raise FileNotFoundError(path)
        config = V2FormalTrainingConfig(stage='easy', max_steps=500000, seed=args.seed,
            batch_size=args.batch_size, replay_capacity=args.replay_capacity,
            actor_lr=getattr(args, model + '_actor_lr'), critic_lr=args.critic_lr,
            actor_rl_scale_alpha=args.alpha, failure_sample_bias=2.,
            spike_regularization=V2SpikeRegularizationConfig(beta_max=1. if model == 'snn' else 0.))
        if args.controlled_stage_step_offset + args.warmup_steps + args.batch_size + 4 >= config.max_steps:
            raise ValueError('No remaining stage horizon for measurement.')
        configurations[model] = {'config': config.to_dict(), 'checkpoint': str(path.resolve()),
            'checkpoint_sha256': _hash(path), 'snn_time_window': args.snn_time_window,
            'compilation': default_v2_cuda_graph_compilation(model=model, resolved_device='cuda')}
    repo = Path(__file__).resolve().parents[3]
    def git(*command):
        return subprocess.run(['git', *command], cwd=repo, capture_output=True,
                              text=True, encoding='utf-8', check=True).stdout.strip()
    return {'diagnostic_only': True, 'formal_training': False,
        'batch_preparation_diagnostic': {'instrumented': bool(diagnostic_updates),
            'mode': 'gpu_compute_submission_trace' if submission_updates else
                    ('trace' if args.batch_preparation_trace_updates else ('tensor_wall' if diagnostic_updates else 'off')),
            'updates': diagnostic_updates, 'workers': args.batch_preparation_workers},
        'conditions': args.concurrency, 'repeats': args.repeats,
        'output_dir': str(args.output_dir.resolve()),
        'measurement_seconds': args.measurement_seconds, 'warmup_steps': args.warmup_steps,
        'scenario_cycle_steps': args.steps_per_level,
        'controlled_stage_step_offset': args.controlled_stage_step_offset,
        'bc_schedule': v2_bc_schedule_metadata('easy', schedule=args.bc_schedule),
        'models': configurations, 'source': {'head': git('rev-parse', 'HEAD'),
            'status': git('status', '--short'), 'diff_sha256': sha256(git('diff').encode()).hexdigest()},
        'pool': {'path': str(args.validation_pool.resolve()), 'content_digest': pool.content_digest,
            'total_scenarios': pool.scenario_count,
            'selected_scenarios': [{'scenario_id': r['scenario_id'],
                'shape_types': [z['shape']['shape_type'] for z in r['payload']['zones']]}
                for r in pool.scenarios[:args.scenario_count]]},
        'execution_requested': {'gpu_td3_compute': True, 'cpu_deduplication': True,
            'non_blocking_batch_transfer': args.non_blocking_batch_transfer,
            'cache_ellipsoid_segment_clearance': True, 'other_experiments': False},
        'limitations': ['Fresh BC + stage offset is not historical training resume.',
            'Short loop omits formal reporting, three-view and candidate load; no 500k speed guarantee.',
            'Same per-model input/config/seed and scene-order prefix; duration permits different step counts.',
            'CPU wall sections include submission and waits, not pure GPU time; parents/children are not additive.']}


def wait_workers(processes, predicate, *, timeout):
    limit = time.perf_counter() + timeout
    while True:
        for name, process in processes.items():
            code = process.poll()
            if code is not None and (code != 0 or not predicate(name)):
                raise RuntimeError(f'Worker {name} exited before successful completion: {code}')
        if all(predicate(name) for name in processes):
            return
        if time.perf_counter() >= limit:
            raise TimeoutError('Concurrent worker wait timed out.')
        time.sleep(.05)


def summarize_worker(summary, window, *, start_tolerance=.1):
    run = summary['runs']['compute'][0]
    compute = run['local_optimizations']['gpu_td3_compute']
    measured = run['concurrent_window']
    reasons = []
    if run['critic_encoder_compile']['measurement_new_graph_count'] != 0:
        reasons.append('New/unknown Dynamo compilation during measurement.')
    if compute.get('measurement_new_captures') != 0:
        reasons.append('New/unknown manual CUDA Graph captures during measurement.')
    captures = run['concurrent_inductor_captures']
    before, after = captures['before'], captures['after']
    if (not before.get('available') or not after.get('available')
            or before.get('manager_id') != after.get('manager_id')
            or before.get('recordings') != after.get('recordings')):
        reasons.append('New/unknown action Inductor CUDA Graph captures during measurement.')
    if not compute.get('effective') or run['measurement_graph_replays'] <= 0:
        reasons.append('Actual manual CUDA Graph replay evidence missing.')
    if not all(run['environment_execution'].values()) or not run['cpu_deduplication']:
        reasons.append('Required environment CPU reuse/cache did not take effect.')
    if not run['critic_encoder_compile'].get('cuda_graph_action_inference_evidence', {}).get('verified'):
        reasons.append('Action-inference warmup replay evidence missing.')
    if measured['start'] != window['start'] or measured['end'] != window['end']:
        reasons.append('Worker used a different measurement window.')
    if (abs(measured['actual_start'] - window['start']) > start_tolerance
            or measured['actual_end'] < window['end']):
        reasons.append('Worker did not sustain the full common window.')
    if measured['completed_steps_in_window'] <= 0:
        reasons.append('No completed steps in common window.')
    if any(s['completed_steps_in_window'] <= 0 for s in run['scenario_coverage']):
        reasons.append('Selected scene coverage incomplete within the common measurement window.')
    timing = run['timing']
    parent_keys = ('scenario_reset_wall_seconds', 'pre_action_geometry_wall_seconds',
        'action_inference_wall_seconds', 'environment_step_wall_seconds',
        'replay_write_wall_seconds', 'td3_update_wall_seconds')
    parents = {key: timing[key] for key in parent_keys}
    # Final drain/statistics event readback is inside total, outside these sections.
    other = timing['total_wall_seconds'] - sum(parents.values())
    batch_diagnostic = run.get('batch_preparation_diagnostic')
    instrumented = bool(batch_diagnostic)
    if batch_diagnostic and not batch_diagnostic['diagnostic_valid']:
        reasons.extend(batch_diagnostic['invalid_reasons'])
    diagnostic_valid = not reasons
    if instrumented:
        reasons = reasons + ['Batch preparation instrumentation/profiler overhead; not pure throughput.']
    return {
        'instrumented': instrumented, 'diagnostic_valid': diagnostic_valid,
        'batch_preparation_diagnostic': batch_diagnostic,
        'valid_for_speed_comparison': not reasons, 'invalid_reasons': reasons,
        'window': measured, 'steps': run['measured_steps'],
        'seconds_per_step_including_final_drain': timing['total_wall_seconds'] / run['measured_steps'],
        'window_steps_per_second': measured['completed_steps_in_window'] / (window['end'] - window['start']),
        'parent_wall_seconds': parents, 'other_wall_seconds': other,
        'children_within_td3': {'replay_sample_wall_seconds': timing['replay_sample_wall_seconds'],
                              'update_breakdown': timing['td3_update_breakdown']},
        'final_completion': run['batch_transfer_window_completion'],
        'warmup': {'loop_seconds': run['loop_warmup_wall_seconds'],
            'registration_seconds': run['critic_encoder_compile']['registration_wall_seconds'],
            'compile_warmup_seconds': run['critic_encoder_compile']['warmup_wall_seconds'],
            'manual_capture_seconds': compute['capture_seconds'],
            'manual_initialization_seconds': compute['initialization_seconds']},
        'execution': {'gpu_td3_compute': {'requested': True, 'effective': compute['effective'],
            'reason': 'Actual manual graph replay observed' if run['measurement_graph_replays'] else 'No replay evidence',
            'measurement_replays': run['measurement_graph_replays']},
            'cpu_deduplication': {'requested': True, 'effective': run['cpu_deduplication'],
                'reason': 'Explicit environment + select_action argument'},
            'ellipsoid_cache': {'requested': True,
                'effective': run['environment_execution']['cache_ellipsoid_segment_clearance'],
                'reason': 'Explicit environment constructor configuration'},
            'compilation': run['critic_encoder_compile']},
        'replay_occupancy': run['replay_occupancy'], 'scenarios': run['scenario_coverage'],
        'server_load_before': run['server_load_before'], 'server_load_after': run['server_load_after']}


def run_batch(args, count, repeat, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'review').mkdir()
    (root / 'other').mkdir()
    control = root / 'other' / 'control'
    control.mkdir()
    processes, logs = {}, []
    report = {'concurrency': count, 'repeat': repeat, 'completed': False,
              'valid_for_speed_comparison': False, 'diagnostic_valid': False, 'workers': {}}
    try:
        for model in ('ann', 'snn'):
            for index in range(count // 2):
                name = f'{model}_{index}'
                log = (root / 'review' / f'{name}.log').open('w', encoding='utf-8')
                logs.append(log)
                processes[name] = subprocess.Popen(worker_command(args, model, name, root),
                    cwd=Path(__file__).resolve().parents[3], stdout=log, stderr=subprocess.STDOUT)
        ready = lambda name: (control / f'{name}.ready.json').is_file()
        wait_workers(processes, ready, timeout=args.ready_timeout)
        # An exited ready worker is still a failed participant.
        if any(p.poll() is not None for p in processes.values()):
            raise RuntimeError('A ready worker exited before window release.')
        start = time.perf_counter() + args.start_delay
        window = {'start': start, 'end': start + args.measurement_seconds,
                  'clock': 'same-host time.perf_counter', 'published_unix_time': time.time()}
        write_json(control / 'window.json', window)
        report['common_window'] = window
        wait_workers(processes, lambda name: processes[name].poll() == 0,
            timeout=args.start_delay + args.measurement_seconds + args.completion_timeout)
        for name in processes:
            detail = root / 'other' / name / 'other' / 'local_optimization_summary.json'
            worker_detail = read_json(detail)
            result = summarize_worker(worker_detail, window)
            transfer = worker_detail['runs']['compute'][0]['local_optimizations'].get('non_blocking_batch_transfer')
            result['execution']['non_blocking_batch_transfer'] = transfer
            if (transfer and transfer['requested'] != args.non_blocking_batch_transfer
                    or args.non_blocking_batch_transfer and (not transfer or not transfer['effective']
                    or not transfer['requested'] or transfer.get('transfer_calls', 0) <= 0)):
                result['invalid_reasons'].append('Requested nonblocking GPUCompute upload evidence missing.')
                result['diagnostic_valid'] = result['valid_for_speed_comparison'] = False
            requested = (name in args.batch_preparation_workers
                and bool(args.batch_preparation_updates or args.batch_preparation_trace_updates or args.gpu_compute_submission_trace_updates))
            result['batch_preparation_execution'] = {'requested': requested,
                'effective': result['instrumented'],
                'reason': 'Worker returned bounded instrumentation evidence' if result['instrumented']
                    else 'No batch preparation instrumentation evidence'}
            if requested != result['instrumented']:
                result['invalid_reasons'].append('Requested/actual batch instrumentation mismatch.')
                result['diagnostic_valid'] = result['valid_for_speed_comparison'] = False
            if requested and args.gpu_compute_submission_trace_updates:
                evidence = result['batch_preparation_diagnostic'] or {}
                if (evidence.get('mode') != 'gpu_compute_submission_trace'
                        or not evidence.get('execution', {}).get('effective')):
                    result['invalid_reasons'].append('Real nonblocking submission trace evidence missing.')
                    result['diagnostic_valid'] = result['valid_for_speed_comparison'] = False
            result['model'] = name.split('_')[0]
            result['detail'] = str(detail.relative_to(root))
            report['workers'][name] = result
        report['completed'] = True
        report['instrumented'] = any(r['instrumented'] for r in report['workers'].values())
        report['diagnostic_valid'] = all(r['diagnostic_valid'] for r in report['workers'].values())
        report['valid_for_speed_comparison'] = all(r['valid_for_speed_comparison'] for r in report['workers'].values())
        report['batch_steps_per_second'] = sum(r['window_steps_per_second'] for r in report['workers'].values())
        report['actual_concurrent_overlap'] = {
            'start': max(r['window']['actual_start'] for r in report['workers'].values()),
            'end': min(r['window']['actual_end'] for r in report['workers'].values())}
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        write_json(control / 'abort.json', report['error'])
    finally:
        # Only Popen objects created by this batch; never enumerate/kill other jobs.
        for name, process in processes.items():
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired) as exc:
                report.setdefault('cleanup_errors', {})[name] = str(exc)
                report['valid_for_speed_comparison'] = False
                report['diagnostic_valid'] = False
            report.setdefault('exit_codes', {})[name] = process.returncode
        for log in logs:
            log.close()
        write_json(root / 'review' / 'summary.json', report)
    return report


def summarize_batches(reports, expected_batches):
    from statistics import mean
    conditions = {}
    for count in sorted({r['concurrency'] for r in reports}):
        batches = [r for r in reports if r['concurrency'] == count]
        valid = all(r['valid_for_speed_comparison'] for r in batches)
        item = {'valid_for_speed_comparison': valid, 'completed_repeats': len(batches),
            'instrumented': any(r.get('instrumented', False) for r in batches),
            'diagnostic_valid': all(r.get('diagnostic_valid', r['valid_for_speed_comparison']) for r in batches)}
        if valid:
            item['mean_batch_steps_per_second'] = mean(r['batch_steps_per_second'] for r in batches)
            item['models'] = {model: {
                'mean_seconds_per_step': mean(w['seconds_per_step_including_final_drain']
                    for r in batches for w in r['workers'].values() if w['model'] == model),
                'mean_task_window_steps_per_second': mean(w['window_steps_per_second']
                    for r in batches for w in r['workers'].values() if w['model'] == model)}
                for model in ('ann', 'snn')}
        conditions[str(count)] = item
    return {'batches': reports, 'conditions': conditions,
        'diagnostic_valid': len(reports) == expected_batches
            and all(r.get('diagnostic_valid', r['valid_for_speed_comparison']) for r in reports),
        'instrumented': any(r.get('instrumented', False) for r in reports),
        'valid_for_speed_comparison': len(reports) == expected_batches
            and all(r['valid_for_speed_comparison'] for r in reports),
        'timing_note': 'Whole-step totals contain TD3 and environment; TD3 children are not additive. '
            'Batch throughput uses completed steps divided by the shared window, excluding tail steps.'}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch', action='store_true', help='Explicitly start short CUDA timing workers.')
    p.add_argument('--non-blocking-batch-transfer', action='store_true',
        help='Opt-in GPUCompute pageable nonblocking upload A/B; default blocking.')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--ann-bc-checkpoint', type=Path, required=True)
    p.add_argument('--snn-bc-checkpoint', type=Path, required=True)
    p.add_argument('--validation-pool', type=Path, required=True)
    p.add_argument('--expected-pool-digest', required=True)
    p.add_argument('--ann-actor-lr', type=float, required=True)
    p.add_argument('--snn-actor-lr', type=float, required=True)
    p.add_argument('--critic-lr', type=float, default=2.5e-4)
    p.add_argument('--alpha', type=float, default=2.5)
    p.add_argument('--bc-schedule', choices=('default', 'smooth-final', 'smooth-mid', 'smooth-mid-strong'), required=True)
    p.add_argument('--concurrency', nargs='+', type=int, choices=(2, 6, 8), default=[2, 8])
    p.add_argument('--measurement-seconds', type=float, default=30.)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--warmup-steps', type=int, default=128)
    p.add_argument('--steps-per-level', type=int, default=256, help='Scene cycle length, NOT a measurement step cap.')
    p.add_argument('--ready-timeout', type=float, default=1800.)
    p.add_argument('--completion-timeout', type=float, default=120.)
    p.add_argument('--start-delay', type=float, default=1.)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--validation-seed', type=int, default=20260904)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--replay-capacity', type=int, default=500000)
    p.add_argument('--scenario-count', type=int, default=3)
    p.add_argument('--controlled-stage-step-offset', type=int, default=75000)
    p.add_argument('--snn-time-window', type=int, default=4)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--batch-preparation-updates', type=int, default=0,
        help='Separate instrumented run: 2..64 steady updates, default off.')
    group.add_argument('--batch-preparation-trace-updates', type=int, default=0,
        help='Separate trace run: 2..8 steady updates, default off; perturbs concurrency.')
    group.add_argument('--gpu-compute-submission-trace-updates', type=int, default=0,
        help='Real nonblocking GPUCompute submission observer: 2..8 updates, default off.')
    p.add_argument('--batch-preparation-workers', nargs='+', default=['ann_0', 'snn_0'],
        help='Only these workers collect batch detail/trace; others sustain normal load.')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    plan = build_plan(args)
    if not args.launch:
        print(json.dumps({'mode': 'plan_only', 'plan': plan}, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'review').mkdir()
    write_json(args.output_dir / 'review' / 'plan.json', plan)
    reports = []
    try:
        # Conditions and repeats are serial; workers within one condition concurrent.
        for count in args.concurrency:
            for repeat in range(args.repeats):
                report = run_batch(args, count, repeat, args.output_dir / f'{count}_{repeat}')
                reports.append(report)
                if not report.get('diagnostic_valid', report['valid_for_speed_comparison']):
                    return 1
        return 0
    finally:
        write_json(args.output_dir / 'review' / 'summary.json',
            summarize_batches(reports, len(args.concurrency) * args.repeats))


if __name__ == '__main__':
    raise SystemExit(main())
