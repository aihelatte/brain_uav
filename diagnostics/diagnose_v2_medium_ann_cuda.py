"""Bounded ANN medium CUDA reproduction using the sealed e6bc9be course."""

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4
from unittest.mock import patch

from brain_uav.observations.v2_batch import V2ObservationBatch
from brain_uav.scripts import run_v2_formal_easy_baseline as formal
from brain_uav.scripts import train_v2_td3 as stage
from brain_uav.scripts.v2_formal_easy_guidance import read_json
from brain_uav.scripts.v2_mps_diagnostic import Session, capability
from brain_uav.trainers.v2_formal_training import V2FormalStageTrainer
from brain_uav.trainers.v2_gpu_compute import GPUCompute


BASE = 'e6bc9be6b1cd95ac5c23b370f373b7eb3050c6df'
COURSE_ROOT = Path('/home/zj/diagnostics/outputs/v2_course_e6bc9be')
BATCH = 'medium-ann'
ALL_SEEDS = (7, 11, 23, 41)
STOP_STEP = 1000
WINDOW = (675, 700)
ATTEMPT = Path('batches/medium-ann/attempt-000001')


class DiagnosticComplete(Exception):
    """Planned exit after the last step, before formal checkpoint publication."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course-root', type=Path, default=COURSE_ROOT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41])
    parser.add_argument('--single-result', type=Path, help='Completed async seed-41 output required for the four-seed run.')
    parser.add_argument('--sync-window', action='store_true', help='Synchronize between marked CUDA sections only in steps 675–700.')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', type=int, choices=ALL_SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.course_root, args.out = args.course_root.resolve(), args.out.resolve()
    args.seeds = tuple(args.seeds)
    if args.seeds not in ((41,), ALL_SEEDS):
        parser.error('Choose seed 41 alone or ordered seeds 7 11 23 41.')
    if (args.seeds == ALL_SEEDS) != (args.single_result is not None):
        parser.error('The four-seed run requires --single-result from the async seed-41 run.')
    if args.single_result is not None:
        args.single_result = args.single_result.resolve()
    if args.out.is_relative_to(args.course_root) or args.course_root.is_relative_to(args.out):
        parser.error('--out must be separate from the original course.')
    if args.worker is not None and (not args.run or args.worker not in args.seeds):
        parser.error('Worker requires a selected seed and --run.')
    return args


def describe(args):
    return dict(diagnostic_only=True, base=BASE, batch=BATCH, seeds=list(args.seeds), formal_max_steps=750000,
                stop_step=STOP_STEP, trace_window=list(WINDOW), sync_window=args.sync_window,
                course_root=str(args.course_root), output=str(args.out))


def source_args(root, source, *, job=None):
    paths, hashes = source['inputs']['paths'], source['inputs']['hashes']
    argv = [('--root', root), ('--ann-bc', paths['ann_bc']), ('--snn-bc', paths['snn_bc']),
            ('--validation-pool', paths['pool']), ('--parent', paths['parent']),
            ('--source-manifest', paths['source_manifest']), ('--timeout', source['plan']['initialization_timeout']),
            ('--pool-sha256', hashes['pool']), ('--parent-sha256', hashes['parent']),
            ('--experiment', formal.COURSE_EXPERIMENT), ('--batch', BATCH)]
    if job is not None:
        argv.append(('--job', job))
    return formal.parser().parse_args(['worker' if job else 'launch', '--launch',
                                        *(str(item) for pair in argv for item in pair)])


def load_original(args):
    from brain_uav.trainers.v2_validation import load_v2_validation_pool

    train = args.course_root / ATTEMPT / 'train'
    manifest_path = train / 'manifest.json'
    source = read_json(manifest_path)
    identity = source['inputs']
    wanted = formal.jobs(formal.COURSE_EXPERIMENT, BATCH)
    resolved = [formal.resolved(job, formal.COURSE_EXPERIMENT)[0] for job in wanted]
    selection = read_json(args.course_root / 'batches/easy-ann/selection.json')['winner']
    pool = read_json(args.course_root / 'pools.json')['validation']['medium']
    expected_paths = dict(parent=args.course_root / 'batches/easy-ann/winner.pt',
                          pool=args.course_root / 'pools/validation/v2_validation_medium.json')
    if (identity['source']['HEAD'] != BASE or source['plan']['root'] != str(train)
            or source['plan']['batch'] != BATCH or source['plan']['gate_only'] is not False
            or source['plan']['experiment'] != formal.COURSE_EXPERIMENT
            or source['plan']['jobs'] != resolved
            or any(Path(identity['paths'][key]) != path for key, path in expected_paths.items())
            or selection['checkpoint'] != identity['paths']['parent']
            or selection['checkpoint_sha256'] != identity['hashes']['parent']
            or pool['path'] != identity['paths']['pool'] or pool['sha256'] != identity['hashes']['pool']
            or pool['content_digest'] != identity['pool_digest']):
        raise ValueError('Original committed medium ANN batch, parent, or pool differs.')
    formal.check_inputs(identity)
    parsed = source_args(train, source)
    parsed.gate_report = train.parent / 'gate'
    if formal.validate_gate(parsed, identity) != source['gate_evidence']:
        raise ValueError('Original medium ANN gate evidence differs.')
    # Recheck the existing input contract without opening a trainer or writing to the course.
    actual = formal.inputs(parsed)
    if any(actual[key] != identity[key] for key in ('paths', 'hashes', 'pool_digest', 'global_steps_start')):
        raise ValueError('Original formal initialization evidence differs.')
    load_v2_validation_pool(parsed.validation_pool, expected_level='medium', expected_count=100,
                            expected_master_seed=pool['master_seed'], expected_stage_seed=pool['stage_seed'])
    binding = dict(manifest=str(manifest_path), sha256=formal.file_hash(manifest_path),
                   batch_id=source['batch_id'], parent_sha256=identity['hashes']['parent'],
                   pool_sha256=identity['hashes']['pool'])
    return source, wanted, binding


def require_single_result(path, binding):
    summary = read_json(path / 'summary.json')
    worker = read_json(path / 'ann_seed41/worker_summary.json')
    process = summary.get('processes', {}).get('ann_seed41', {})
    if (summary.get('original') != binding or summary.get('diagnostic_only') is not True
            or summary.get('execution_completed') is not True or summary.get('seeds') != [41]
            or summary.get('mps_cleanup', {}).get('stopped') is not True
            or process.get('exit_code') != 0 or process.get('pid') != worker.get('pid')
            or summary.get('sync_window') is not False or worker.get('diagnostic_completed') is not True
            or worker.get('stage_steps') != STOP_STEP):
        raise ValueError('Four-seed run requires the completed asynchronous seed-41 diagnostic on these inputs.')


def shapes(observation):
    return {name: list(getattr(observation, name).shape) for name in
            ('ego_features', 'goal_features', 'zone_features', 'presence_mask')}


class Trace:
    def __init__(self, stream, *, sync_window):
        self.stream = stream
        self.sync_window = sync_window
        self.step = 0
        self.device = None
        self.phase = None
        self.scene = {}

    @property
    def active(self):
        return WINDOW[0] <= self.step <= WINDOW[1]

    def emit(self, event, section, **details):
        if self.active:
            print(json.dumps(dict(pid=os.getpid(), step=self.step, event=event, section=section,
                                  monotonic_ns=time.monotonic_ns(), **self.scene, **details)),
                  file=self.stream, flush=True)

    def synchronize(self):
        import torch
        torch.cuda.synchronize(self.device)

    @contextmanager
    def scope(self, name):
        self.emit('begin', name)
        try:
            yield
            self.emit('host_return', name)
            if self.active and self.sync_window:
                self.emit('sync_begin', name)
                self.synchronize()
                self.emit('sync_done', name)
        except BaseException as exc:
            self.emit('error', name, error_type=type(exc).__name__, error=str(exc))
            raise

    def observe_batch(self, batch):
        self.emit('shape', 'batch_upload', obs=shapes(batch.obs), next_obs=shapes(batch.next_obs))


def run_bounded(run, trainer, trace):
    engine = trainer.engine
    if trainer.config.max_steps != 750000:
        raise ValueError('The diagnostic requires the original 750000-step schedule.')
    trace.device = engine.device
    new_episode, select_action = trainer._new_episode, engine.select_action
    action_tensor, update = engine._action_inference_tensor, engine.update_once
    action_to, graph_key, capture = V2ObservationBatch.to, GPUCompute.key, GPUCompute.capture_shape

    def stop_if_done():
        if trainer.result.stage_steps == STOP_STEP:
            raise DiagnosticComplete(f'Completed {STOP_STEP} diagnostic steps.')
        if trainer.result.stage_steps > STOP_STEP:
            raise RuntimeError('Diagnostic exceeded the stop step.')

    def bounded_reset(*args, **kwargs):
        stop_if_done()
        return new_episode(*args, **kwargs)

    def traced_select(*args, **kwargs):
        stop_if_done()
        trace.step = trainer.result.stage_steps + 1
        if not trace.active:
            return select_action(*args, **kwargs)
        trace.scene = dict(level=trainer.env.last_curriculum_level, obstacles=len(trainer.env.zones))
        if trace.sync_window and trace.step == WINDOW[0]:
            trace.emit('sync_begin', 'window_entry')
            trace.synchronize()
            trace.emit('sync_done', 'window_entry')
        trace.phase = 'action'
        try:
            with trace.scope('action_inference_and_readback'):
                result = select_action(*args, **kwargs)
                trace.emit('host_return', 'action_result_readback')
                return result
        finally:
            trace.phase = None

    def traced_action_to(observation, *args, **kwargs):
        if trace.phase != 'action':
            return action_to(observation, *args, **kwargs)
        trace.emit('shape', 'action_upload', obs=shapes(observation))
        with trace.scope('action_upload'):
            return action_to(observation, *args, **kwargs)

    def traced_tensor(*args, **kwargs):
        if trace.phase != 'action':
            return action_tensor(*args, **kwargs)
        with trace.scope('action_forward'):
            result = action_tensor(*args, **kwargs)
        trace.emit('begin', 'action_result_readback')
        return result

    def traced_update(**kwargs):
        if not trace.active:
            return update(**kwargs)
        if kwargs['total_steps'] != trace.step:
            raise RuntimeError('Trace step differs from the actual update step.')
        with trace.scope('update'):
            return update(**kwargs, gpu_compute_submission_diagnostic=trace)

    def traced_key(executor, batch, mode):
        key = graph_key(executor, batch, mode)
        if trace.active:
            trace.emit('lookup', 'update_graph', key=list(key), cache='reuse' if key in executor.graphs else 'new')
        return key

    def traced_capture(executor, key, batch):
        if not trace.active:
            return capture(executor, key, batch)
        with trace.scope('update_graph_capture'):
            return capture(executor, key, batch)

    with (patch.object(trainer, '_new_episode', bounded_reset),
          patch.object(engine, 'select_action', traced_select),
          patch.object(engine, '_action_inference_tensor', traced_tensor),
          patch.object(engine, 'update_once', traced_update),
          patch.object(V2ObservationBatch, 'to', traced_action_to),
          patch.object(GPUCompute, 'key', traced_key),
          patch.object(GPUCompute, 'capture_shape', traced_capture)):
        run(trainer)
    raise RuntimeError('Formal trainer returned before the diagnostic stop hook.')


def worker(args):
    scratch = read_json(args.out / 'manifest.json')
    description = read_json(args.out / 'diagnostic.json')
    if any(description[key] != value for key, value in describe(args).items()):
        raise ValueError('Diagnostic worker arguments differ from the launcher.')
    spec = next(spec for spec in scratch['plan']['jobs'] if spec['job']['seed'] == args.worker)
    job = spec['job']['id']
    output = args.out / job
    original_run, original_write = V2FormalStageTrainer.run, stage._write_strict_json
    write_json = formal.atomic_json
    completed = False

    def run(trainer):
        nonlocal completed
        with (output / 'cuda_trace.jsonl').open('x', encoding='utf-8', buffering=1) as events:
            try:
                return run_bounded(original_run, trainer, Trace(events, sync_window=args.sync_window))
            except DiagnosticComplete:
                completed = True
                raise

    def stage_report(path, value):
        if (completed and path == output / 'metrics_execution_error.json'
                and value['error_type'] == DiagnosticComplete.__name__):
            return
        return original_write(path, value)

    def worker_report(path, value):
        if completed and path == output / 'worker_summary.json':
            if value.get('error', '').startswith(DiagnosticComplete.__name__ + ':'):
                value = {key: item for key, item in value.items() if key != 'error'}
                value.update(diagnostic_only=True, diagnostic_completed=True,
                             execution_completed=True, stage_steps=STOP_STEP, formal_passed=None)
        return write_json(path, value)

    def reject_checkpoint(*args, **kwargs):
        raise RuntimeError('The short diagnostic cannot publish a checkpoint.')

    with (patch.object(formal, 'batch_jobs', return_value=[spec['job'] for spec in scratch['plan']['jobs']]),
          patch.object(V2FormalStageTrainer, 'run', run),
          patch.object(stage, '_write_strict_json', stage_report),
          patch.object(stage, 'save_v2_formal_checkpoint', reject_checkpoint),
          patch.object(stage, 'save_v2_periodic_snapshot', reject_checkpoint),
          patch.object(formal, 'atomic_json', worker_report)):
        try:
            formal.worker(source_args(args.out, scratch, job=job))
        except DiagnosticComplete:
            report = read_json(output / 'worker_summary.json')
            if report.get('diagnostic_completed') is True and report['stage_steps'] == STOP_STEP:
                return 0
            raise
    raise RuntimeError('Worker exited without the bounded diagnostic stop.')


def launch(args, original, wanted, binding):
    chosen = [job for job in wanted if job['seed'] in args.seeds]
    evidence = original['inputs']
    args.out.mkdir(parents=True, exist_ok=False)
    description = dict(describe(args), original=binding)
    scratch = dict(original, batch_id=uuid4().hex,
                   plan=dict(original['plan'], root=str(args.out),
                             jobs=[formal.resolved(job, formal.COURSE_EXPERIMENT)[0] for job in chosen]))
    formal.atomic_json(args.out / 'diagnostic.json', description)
    formal.atomic_json(args.out / 'manifest.json', scratch)
    argv = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--run',
            '--course-root', str(args.course_root), '--out', str(args.out),
            '--seeds', *map(str, args.seeds)]
    if args.single_result is not None:
        argv.extend(('--single-result', str(args.single_result)))
    if args.sync_window:
        argv.append('--sync-window')
    status = dict(description, execution_completed=False)
    processes, session = {}, None
    failure = None
    started = time.monotonic()
    first_signal = True

    def terminate(signum, frame):
        nonlocal first_signal
        if first_signal:
            first_signal = False
            raise InterruptedError('SIGTERM received by diagnostic launcher.')

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        with ExitStack() as logs:
            session = Session(args.out / 'mps')
            formal.atomic_json(args.out / 'mps/namespace.json', dict(batch_id=scratch['batch_id'],
                pipe_directory=str(session.pipe), log_directory=str(session.root / 'logs')))
            session.start()
            capability(args.out / 'capability', session)
            capability(args.out / 'termination_capability', session, terminate_client=True)
            for job in chosen:
                name = job['id']
                log = logs.enter_context((args.out / f'{name}.log').open('x', encoding='utf-8'))
                process = subprocess.Popen([*argv, '--worker', str(job['seed'])], cwd=formal.REPO,
                    env=dict(session.env, PYTHONPATH=str(formal.REPO / 'src')),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes[name] = process
                session.register({name: process})
                formal.atomic_json(args.out / 'pids.json', {key: p.pid for key, p in processes.items()})
            formal.wait_ready(processes, args.out, original['plan']['initialization_timeout'])
            old = args.course_root / ATTEMPT / 'train'
            for name in processes:
                ready, reference = [read_json(root / name / 'ready.json') for root in (args.out, old)]
                for key in ('seed_manifest', 'initial_state_sha256'):
                    if ready[key] != reference[key]:
                        raise ValueError(f'Diagnostic initialization differs from original {name}: {key}')
            original_jobs = formal.jobs

            def release_jobs(experiment='baseline', batch=None):
                if (experiment, batch) == (formal.COURSE_EXPERIMENT, BATCH):
                    return chosen
                return original_jobs(experiment, batch)

            with patch.object(formal, 'jobs', new=release_jobs):
                formal.release(processes, args.out, session, formal.COURSE_EXPERIMENT, batch=BATCH)
            while True:
                returncodes = {name: process.poll() for name, process in processes.items()}
                if any(code not in (None, 0) for code in returncodes.values()):
                    raise RuntimeError(f'Diagnostic workers returned {returncodes}. See worker and MPS logs.')
                if all(code == 0 for code in returncodes.values()):
                    break
                time.sleep(.2)
            for name, process in processes.items():
                report = read_json(args.out / name / 'worker_summary.json')
                if (report.get('diagnostic_completed') is not True or report.get('stage_steps') != STOP_STEP
                        or report.get('pid') != process.pid or report.get('batch_id') != scratch['batch_id']):
                    raise ValueError(f'Incomplete diagnostic worker {name}.')
            formal.check_inputs(evidence)
            status['execution_completed'] = True
    except BaseException as exc:
        failure = exc
        status['error'] = f'{type(exc).__name__}: {exc}'
        try:
            formal.atomic_json(args.out / 'abort.json', status)
        finally:
            if session is not None:
                try:
                    formal.stop_owned(processes, session)
                except BaseException as cleanup:
                    status['worker_cleanup_error'] = str(cleanup)
    finally:
        try:
            if session is not None:
                try:
                    status['mps_cleanup'] = session.finish()
                except BaseException as cleanup:
                    status['mps_cleanup'] = dict(stopped=False, error=str(cleanup))
            status['execution_completed'] &= status.get('mps_cleanup', {}).get('stopped') is True
            status['processes'] = {name: dict(pid=p.pid, exit_code=p.poll()) for name, p in processes.items()}
            status['process_wall_seconds'] = time.monotonic() - started
            formal.atomic_json(args.out / 'summary.json', status)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    if failure is not None:
        raise failure
    return 0 if status['execution_completed'] else 1


def main(argv=None):
    args = parse_args(argv)
    if not args.run:
        print(json.dumps(describe(args), indent=2))
        return 0
    if os.name != 'posix' or Path(sys.executable).resolve() != Path(formal.provenance.PYTHON).resolve():
        raise ValueError('Run the diagnostic in the declared Linux server Python.')
    if os.environ.get('CUDA_LAUNCH_BLOCKING', '0') != '0':
        raise ValueError('Keep CUDA_LAUNCH_BLOCKING unset or 0; explicit synchronization is window-scoped.')
    if args.worker is not None:
        return worker(args)
    print(f'Diagnostic output: {args.out}', flush=True)
    original, jobs, binding = load_original(args)
    if args.single_result is not None:
        require_single_result(args.single_result, binding)
    return launch(args, original, jobs, binding)


if __name__ == '__main__':
    raise SystemExit(main())
