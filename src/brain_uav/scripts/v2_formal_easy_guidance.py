"""Collection and fixed diagnostics for the formal easy safe-direct experiment."""
from collections import Counter
import json
from math import pi
from pathlib import Path

from brain_uav.scripts import run_v2_formal_easy_baseline as run

CATEGORIES = ('empty', 'clear', 'detour')
OUTCOMES = ('goal', 'upper_boundary', 'other_boundary', 'collision', 'ground', 'timeout')


def diagnostic_records():
    from brain_uav.envs import V2_ENV_SCENARIO_FORMAT, V2_ENV_SCENARIO_VERSION
    from brain_uav.geometry import NoFlyZone, Sphere

    records = []
    for category in CATEGORIES:
        for index, (start_x, goal_x, height) in enumerate(
            ((-500., 500., 180.), (500., -500., 260.), (-650., 650., 360.))
        ):
            name = f'{category}_{index + 1}'
            zones = []
            if category != 'empty':
                center_y = 280. if category == 'clear' else 0.
                zones.append(NoFlyZone(name, Sphere((0., center_y, height), 120.)).to_dict())
            records.append(dict(scenario_id=name, payload=dict(
                format=V2_ENV_SCENARIO_FORMAT, format_version=V2_ENV_SCENARIO_VERSION,
                state=[start_x, 0., height, 0., 0. if start_x < goal_x else pi],
                goal=[goal_x, 0., height], zones=zones, curriculum_level='easy',
            )))
    return records


def diagnostic_evaluation(actor, pool, rewards, *, device):
    from brain_uav.trainers.v2_validation import (
        evaluate_v2_diagnostic_scenarios, scenario_config_from_snapshot,
    )

    scenario = scenario_config_from_snapshot(pool.scenario_config)
    records = diagnostic_records()
    result = evaluate_v2_diagnostic_scenarios(
        actor, records, scenario, pool.uav_collision_radius, rewards, device=device,
    )
    categories = {}
    for name in CATEGORIES:
        rows = [row for row in result.scenarios if row['scenario_id'].startswith(name + '_')]
        distances = [row['flight_distance_m'] for row in rows if row['outcome'] == 'goal']
        categories[name] = dict(
            scenario_count=len(rows), goal_count=len(distances), arrival_rate=len(distances) / len(rows),
            mean_success_flight_distance_m=sum(distances) / len(distances) if distances else None,
            scenarios=rows,
        )
    return categories


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def outcome(row):
    if row['outcome'] == 'boundary':
        return 'upper_boundary' if any(
            v['axis'] == 'z' and v['direction'] == 'positive' for v in row['boundary_violations']
        ) else 'other_boundary'
    if row['outcome'] not in ('goal', 'collision', 'ground', 'timeout'):
        raise ValueError('Unexpected terminal outcome.')
    return row['outcome']


def last_complete_episodes(path, actual_steps):
    cutoff = max(0, actual_steps - 50000)
    previous = success = count = 0
    with path.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream, 1):
            row = json.loads(line)
            end, length = row['stage_steps'], row['episode_length']
            if (row['episode'] != index or type(length) is not int or length <= 0
                    or type(end) is not int or end != previous + length or end > actual_steps):
                raise ValueError('Episode history is incomplete or inconsistent.')
            label = outcome(row)
            if previous >= cutoff:
                count += 1
                success += label == 'goal'
            previous = end
    if previous == 0:
        raise ValueError('Missing complete training episode history.')
    return dict(from_step=cutoff, through_step=actual_steps, count=count, successes=success,
                success_rate=success / count if count else None)


def paired_outcomes(baseline, candidate):
    before = {row['scenario_id']: outcome(row) for row in baseline}
    after = {row['scenario_id']: outcome(row) for row in candidate}
    if len(before) != 100 or len(after) != 100 or before.keys() != after.keys():
        raise ValueError('Expected 100 paired scenario identities.')
    pairs = [dict(scenario_id=key, before=before[key], after=after[key]) for key in before]
    transitions = {}
    for source in ('upper_boundary', 'timeout'):
        counts = Counter(row['after'] for row in pairs if row['before'] == source)
        transitions[source] = dict(
            total=sum(counts.values()), to_goal=counts['goal'], unchanged=counts[source],
            to_other_failure=sum(counts[label] for label in OUTCOMES if label not in ('goal', source)),
            by_outcome={label: counts[label] for label in OUTCOMES},
        )
    return dict(transitions=transitions, scenarios=pairs)


def terminal_metrics(record, history):
    counts = Counter(outcome(row) for row in record['evaluation']['scenarios'])
    return dict(
        fixed_easy100={label: counts[label] for label in OUTCOMES},
        last_50k_complete_episodes=history, formal_passed=record['formal_passed'],
        stage_steps=record['actual_steps'],
    )


def validated_batch(args, root, experiment, batch, pool, evidence, *, historical=False):
    manifest = read_json(root / 'manifest.json')
    summary = read_json(root / 'batch_summary.json')
    spec = manifest['plan']
    expected_jobs = run.jobs(experiment, batch)
    if (spec.get('experiment', 'baseline') != experiment or spec.get('batch') != batch
            or spec['mode'] != 'launch' or spec['gate_only'] is not False or spec['launch'] is not True
            or spec['jobs'] != [run.resolved(job, experiment)[0] for job in expected_jobs]
            or summary['execution_completed'] is not True or summary['mps_cleanup']['stopped'] is not True):
        raise ValueError(f'Batch identity or completion mismatch: {root}')
    recorded = manifest['inputs']
    recorded_root = None
    if not historical:
        if (recorded['source']['sha256'] != evidence['source']['sha256']
                or any(recorded[k] != evidence[k] for k in ('paths', 'hashes', 'pool_digest'))
                or recorded['experiment'] != experiment
                or Path(spec['root']).resolve() != root.resolve()):
            raise ValueError('Batch source or input identity differs.')
        run.check_inputs(recorded)
        gate = manifest['gate_evidence']
        gate_args = replace_args(args, experiment, batch, Path(gate['root']))
        if run.validate_gate(gate_args, recorded) != gate:
            raise ValueError('Training batch no longer matches its CUDA gate.')
    else:
        old_manifest = read_json(run.SOURCE_MANIFEST)
        if (recorded['source']['sha256'] != old_manifest['sha256']
                or recorded['pool_digest'] != evidence['pool_digest']
                or any(recorded['hashes'][key] != evidence['hashes'][key]
                       for key in ('ann_bc', 'snn_bc', 'pool', 'ann_stage_start', 'snn_stage_start',
                                   'config'))):
            raise ValueError('Historical baseline does not match the reviewed inputs.')
        recorded_root = Path(spec['root'])
    output = {}
    for job in expected_jobs:
        directory = root / job['id']
        record = run.validate_terminal(root, manifest, job, pool, recorded_root=recorded_root)
        history_path = directory / 'metrics_reports' / 'episodes.jsonl'
        if not historical:
            worker = read_json(directory / 'worker_summary.json')
            if (worker['episodes_sha256'] != run.file_hash(history_path)
                    or worker['terminal_evaluation_sha256'] != run.file_hash(directory / 'terminal_evaluation.json')
                    or summary['formal_passed'][job['id']] is not record['formal_passed']):
                raise ValueError('Terminal evidence changed after the worker completed.')
        history = last_complete_episodes(history_path, record['actual_steps'])
        diagnostics = None
        if experiment == run.GUIDANCE_EXPERIMENT:
            path = directory / 'diagnostic_evaluation.json'
            diagnostics = read_json(path)
            if (worker['diagnostic_sha256'] != run.file_hash(path)
                    or diagnostics['checkpoint_sha256'] != record['checkpoint_sha256']
                    or set(diagnostics['categories']) != set(CATEGORIES)):
                raise ValueError('Diagnostic report identity mismatch.')
        output[(job['model'], job['seed'])] = (record, terminal_metrics(record, history), diagnostics)
    return manifest['batch_id'], output


def replace_args(args, experiment, batch, report):
    from argparse import Namespace
    return Namespace(**{**vars(args), 'experiment': experiment, 'batch': batch, 'gate_report': report})


def collect_guidance(args):
    from brain_uav.trainers.v2_validation import load_v2_validation_pool

    if args.experiment != run.GUIDANCE_EXPERIMENT or args.batch != 'main':
        raise ValueError('Collect all six safe-direct jobs from the main batch.')
    roots = (args.root, args.second_batch, args.baseline_root, args.control_root)
    if any(root is None for root in roots) or len({root.resolve() for root in roots}) != 4:
        raise ValueError('Four distinct batch directories are required.')
    evidence = run.inputs(args)
    pool = load_v2_validation_pool(args.validation_pool, expected_level='easy', expected_count=100)
    lock = read_json(run.GUIDANCE_CONFIG)['historical_artifact_sha256']
    historical_jobs = run.jobs()
    filenames = ('metrics_reports/episodes.jsonl', 'terminal_evaluation.json',
                 'ready.json', 'process.json', 'worker_summary.json')
    required = {'manifest.json', 'batch_summary.json'} | {
        f"{job['id']}/{name}" for job in historical_jobs for name in filenames
    }
    if set(lock) != required or any(run.file_hash(args.baseline_root / name) != lock[name] for name in required):
        raise ValueError('Historical baseline artifacts differ from the reviewed copy.')
    identities = []
    results = []
    for root, mode, batch, historical in (
        (args.root, args.experiment, 'main', False),
        (args.second_batch, args.experiment, 'seed11', False),
        (args.baseline_root, 'baseline', None, True),
        (args.control_root, run.GUIDANCE_CONTROL, 'seed11', False),
    ):
        identity, records = validated_batch(args, root, mode, batch, pool, evidence, historical=historical)
        identities.append(identity)
        results.append(records)
    if len(set(identities)) != 4:
        raise ValueError('Duplicate batch identity.')
    candidate = results[0] | results[1]
    baseline = results[2] | results[3]
    expected = {(model, seed) for model in ('ann', 'snn') for seed in (7, 11, 23)}
    if set(candidate) != expected or set(baseline) != expected:
        raise ValueError('Incomplete or unexpected candidate/control groups.')
    models = {}
    for name in ('ann', 'snn'):
        records = []
        for seed in (7, 11, 23):
            record, metrics, diagnostics = candidate[(name, seed)]
            reference, reference_metrics, _ = baseline[(name, seed)]
            records.append(dict(
                record, metrics=metrics, baseline_metrics=reference_metrics,
                comparison=paired_outcomes(reference['evaluation']['scenarios'], record['evaluation']['scenarios']),
                diagnostics=diagnostics['categories'],
            ))
        winner = run.choose(records, (7, 11, 23))
        models[name] = dict(
            candidates=records, selected=winner,
            kind='formal_passed' if winner['formal_passed'] else 'best_available',
            rule='formal pass, goals descending, mean return descending, steps ascending, seed ascending',
        )
    result = dict(execution_completed=True, experiment=args.experiment, batch_ids=identities, models=models)
    path = args.root / 'selection_summary.json'
    if path.exists() and read_json(path) != result:
        raise ValueError('Existing combined summary differs; retained for review.')
    run.publish_selection(args.root, models)
    if not path.exists():
        run.atomic_json(path, result)
    return result
