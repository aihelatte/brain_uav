"""Run configurable V2 formal batches with light trajectory archives and reports."""

import argparse
from copy import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from brain_uav.scripts import run_v2_formal_course as course
from brain_uav.scripts.build_v2_report import build_report
from brain_uav.trainers.v2_online_research import atomic_json
from brain_uav.trainers.v2_reporting import V2_TRAJECTORY_ARCHIVE_MODES
from brain_uav.v2_curriculum import V2_TD3_STAGES


batch_runner = course.batch_runner


def _selection(value, allowed):
    fields = [item.strip() for item in value.split(',')]
    if len(set(fields)) != len(fields) or not set(fields).issubset(allowed):
        raise argparse.ArgumentTypeError(f'Choose distinct values from {", ".join(allowed)}.')
    return [item for item in allowed if item in fields]


def _positive(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected a positive integer.') from exc
    if number <= 0:
        raise argparse.ArgumentTypeError('Expected a positive integer.')
    return number


def _seeds(value):
    seeds = [_positive(item) for item in value.split(',')]
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError('Training seeds must be distinct.')
    return seeds


def _settings(args):
    width = 4 if args.device == 'cuda' else 1
    for offset in range(0, len(args.seeds), width):
        yield dict(seeds=args.seeds[offset:offset + width], max_steps=args.max_steps,
                   trajectory_archive=args.trajectory_archive, device=args.device)


def _plan(args):
    batches = {}
    for stage in args.stages:
        for model in args.models:
            batch = f'{stage}-{model}'
            batches[batch] = [
                [course.resolve_course_job(job, course_profile='500k', pipeline=settings)[0]
                 for job in batch_runner.jobs(course.EXPERIMENT, batch, course_profile='500k', pipeline=settings)]
                for settings in _settings(args)
            ]
    return dict(format='v2_training_pipeline', models=args.models, stages=args.stages, seeds=args.seeds,
                max_steps=args.max_steps, device=args.device, trajectory_archive=args.trajectory_archive,
                eval_independent=args.eval_independent, batches=batches,
                parents={model: str(path) for model, path in args.parents.items()})


def _run_cpu(args, evidence):
    args.root.mkdir(parents=True, exist_ok=False)
    atomic_json(args.root / 'manifest.json', dict(plan=batch_runner.plan(args), inputs=evidence,
                                                 gate_evidence=None, batch_id=uuid4().hex))
    job, = batch_runner.batch_jobs(args)
    worker = copy(args)
    worker.mode, worker.job = 'worker', job['id']
    batch_runner.worker(worker)
    directory = args.root / job['id']
    atomic_json(directory / 'process.json', dict(pid=os.getpid(), exit_code=0))
    atomic_json(args.root / 'batch_summary.json', dict(execution_completed=True,
        formal_passed={job['id']: course.read_json(directory / 'worker_summary.json')['formal_passed']}))


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _publish_seed(source, target):
    target.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.name == 'metrics_reports':
            for entry in item.iterdir():
                dest = target / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, dest, copy_function=_link_or_copy)
                else:
                    _link_or_copy(entry, dest)
        elif item.is_dir():
            shutil.copytree(item, target / item.name, copy_function=_link_or_copy)
        else:
            _link_or_copy(item, target / item.name)


def _run_batch(args, batch, pool_info, pool, parent):
    candidates = []
    for index, settings in enumerate(_settings(args), 1):
        chunk = args.root / '.batches' / batch / f'part-{index:06d}'
        training = course.batch_args(args, batch, chunk / 'train', pool_info, parent,
                                     gate_report=chunk / 'gate' if args.device == 'cuda' else None)
        training.pipeline = settings
        evidence = batch_runner.inputs(training)
        if args.device == 'cuda':
            gate = course.batch_args(args, batch, chunk / 'gate', pool_info, parent, gate_only=True)
            gate.pipeline = settings
            batch_runner.launch(gate)
            batch_runner.launch(training)
        else:
            _run_cpu(training, evidence)
        for record in course.collect_batch(training, pool, evidence):
            source = Path(record['checkpoint']).parent
            target = args.root / 'runs' / batch / f"seed{record['seed']}"
            _publish_seed(source, target)
            candidates.append(dict(record, checkpoint=str(target / Path(record['checkpoint']).name),
                                   terminal_evaluation=str(target / 'terminal_evaluation.json')))
    winner = course.select_winner(candidates, seeds=args.seeds)
    selection = args.root / 'winners' / f'{batch}.selection.json'
    if winner is not None:
        checkpoint = args.root / 'winners' / f'{batch}.pt'
        course.seal_copy(Path(winner['checkpoint']), checkpoint, winner['checkpoint_sha256'])
        winner = dict(winner, candidate_checkpoint=winner['checkpoint'], checkpoint=str(checkpoint))
    course.seal_json(selection, dict(candidates=candidates, winner=winner,
                                    status='promoted' if winner else 'no_formal_pass'))
    return winner


def _artifact_hashes(root):
    hashes = {}
    for name in ('pools.json', 'independent.csv'):
        path = root / name
        if path.is_file():
            hashes[name] = batch_runner.file_hash(path)
    for name in ('runs', 'winners', 'pools', 'independent'):
        for path in sorted((root / name).rglob('*')):
            if path.is_file():
                with path.open('rb') as stream:
                    digest = sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                    hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
    return hashes


def run_pipeline(args, plan):
    if args.device == 'cuda' and Path(sys.executable).resolve() != Path(course.provenance.PYTHON).resolve():
        raise ValueError('Use the declared server Python for CUDA batches.')
    identity = batch_runner.identity(course.EXPERIMENT)
    if course.read_json(args.source_manifest)['sha256'] != identity['sha256']:
        raise ValueError('Actual source files differ from the reviewed pipeline manifest.')
    external = {model: dict(checkpoint=str(path), checkpoint_sha256=batch_runner.file_hash(path))
                for model, path in args.parents.items()}
    for stage in args.stages:
        if stage != 'easy' and course.PREDECESSOR[stage] not in args.stages:
            for model, parent in external.items():
                course.parent_steps(stage, model, parent['checkpoint'], parent['checkpoint_sha256'],
                                    {'parent': parent['checkpoint_sha256']}, pipeline=True)
    args.root.mkdir(parents=True, exist_ok=False)
    (args.root / 'runs').mkdir()
    (args.root / 'winners').mkdir()
    manifest = dict(plan=plan, source=identity, source_manifest_sha256=batch_runner.file_hash(args.source_manifest),
                    external_parents=external, status='running', batches={})
    manifest_path = args.root / 'manifest.json'
    atomic_json(manifest_path, manifest)
    try:
        evidence, pools = course.prepare_pools(args.root, args.validation_pool_dir)
        manifest['pools'] = evidence
        winners = {}
        for stage in args.stages:
            previous = course.PREDECESSOR.get(stage)
            for model in args.models:
                batch = f'{stage}-{model}'
                parent = None
                if previous in args.stages:
                    parent = winners.get(f'{previous}-{model}')
                    if parent is None:
                        manifest['batches'][batch] = dict(status='blocked_by_no_formal_pass')
                        continue
                elif previous is not None:
                    parent = external[model]
                winner = _run_batch(args, batch, evidence['validation'][stage], pools['validation'][stage], parent)
                if winner is not None:
                    winners[batch] = winner
                selection = args.root / 'winners' / f'{batch}.selection.json'
                manifest['batches'][batch] = dict(status='promoted' if winner else 'no_formal_pass',
                    selection=str(selection), selection_sha256=batch_runner.file_hash(selection))
            atomic_json(manifest_path, manifest)
            build_report(args.root)
        manifest['independent_evaluations'] = 0
        if args.eval_independent and len(winners) == len(plan['batches']):
            final = args.stages[-1]
            rows = course.evaluate_winners(
                args.root, {model: winners[f'{final}-{model}'] for model in args.models}, evidence, pools,
                promotion_stage=final, device=args.device,
            )
            course.write_table(args.root / 'independent.csv',
                ('model', 'stage', 'winner_seed', *course.OUTCOMES, 'scenarios'), rows)
            manifest['independent_evaluations'] = sum(sum(row[key] for key in course.OUTCOMES) for row in rows)
        manifest.update(status='completed', artifact_sha256=_artifact_hashes(args.root))
    except BaseException as exc:
        manifest.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        atomic_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=lambda value: _selection(value, course.MODELS), required=True)
    parser.add_argument('--stages', type=lambda value: _selection(value, V2_TD3_STAGES), required=True)
    parser.add_argument('--seeds', type=_seeds, required=True)
    parser.add_argument('--max-steps', type=_positive, default=course.COURSE_MAX_STEPS)
    parser.add_argument('--parent-ann', type=Path)
    parser.add_argument('--parent-snn', type=Path)
    parser.add_argument('--trajectory-archive', choices=V2_TRAJECTORY_ARCHIVE_MODES, default='full')
    parser.add_argument('--eval-independent', action='store_true')
    parser.add_argument('--output-root', dest='root', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--input-root', type=Path, default=Path('/home/zj/projects/my_project'))
    parser.add_argument('--validation-pool-dir', type=Path)
    parser.add_argument('--source-manifest', type=Path, default=course.HARD_ANN_ACC_SOURCE_MANIFEST)
    parser.add_argument('--plan', action='store_true', help='Print all jobs without writing files or running training.')
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.input_root = args.input_root.resolve()
    args.source_manifest = args.source_manifest.resolve()
    args.validation_pool_dir = (args.validation_pool_dir or args.input_root / course.provenance.POOL.parent).resolve()
    args.timeout, args.gate_timeout = 1800., 300.
    args.parents = {model: path.resolve() for model in course.MODELS
                    if (path := getattr(args, f'parent_{model}')) is not None}
    needs_parent = any(stage != 'easy' and course.PREDECESSOR[stage] not in args.stages for stage in args.stages)
    if set(args.parents) != (set(args.models) if needs_parent else set()):
        parser.error('Supply one --parent-<model> per selected model exactly when its predecessor stage is skipped.')
    plan = _plan(args)
    if args.plan:
        print(json.dumps(plan, indent=2))
    else:
        print(f'Total result directory: {args.root}', flush=True)
        run_pipeline(args, plan)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
