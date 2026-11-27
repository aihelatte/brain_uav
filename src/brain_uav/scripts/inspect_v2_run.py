"""Inspect archived training trajectories or replay a checkpoint on fixed scenes."""

import argparse
import gzip
import json
from pathlib import Path
import re

import numpy as np

from brain_uav.geometry import no_fly_zone_from_dict
from brain_uav.scripts.v2_formal_easy_guidance import outcome, read_json
from brain_uav.trainers.v2_online_research import atomic_json
from brain_uav.trainers.v2_reporting import V2_REPORT_OUTCOMES, export_v2_trajectory_views


def _ranges(value):
    intervals = []
    for token in value.split(','):
        match = re.fullmatch(r'(\d+)(?:-(\d+))?', token.strip())
        if match is None:
            raise argparse.ArgumentTypeError('Use nonnegative integers or inclusive ranges.')
        first, last = match.groups()
        start, end = int(first), int(last or first)
        if start > end:
            raise argparse.ArgumentTypeError('Range start must not exceed its end.')
        intervals.append((start, end))
    return intervals


def _contains(intervals, value):
    return intervals is None or any(start <= value <= end for start, end in intervals)


def _csv(value):
    values = value.split(',')
    if any(not item.strip() for item in values):
        raise argparse.ArgumentTypeError('Use a comma-separated list of nonempty values.')
    return {item.strip() for item in values}


def _limit(value):
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError('Limit must be nonnegative.')
    return count


def _check_output(out, *sources):
    for source in sources:
        directory = source.resolve()
        if not directory.is_dir():
            directory = directory.parent
        roots = [p.parent for p in (directory, *directory.parents)
                 if p.name in ('runs', 'batches', '.batches', 'pools', 'winners')]
        if any(out.is_relative_to(root) for root in (directory, *roots)):
            raise ValueError(f'Inspection output must be outside source data: {source}')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Choose a new or empty inspection directory: {out}')


def _row(payload):
    points = np.asarray(payload['trajectory'], dtype=np.float64)
    zones = [no_fly_zone_from_dict(item) for item in payload['scenario_payload']['zones']]
    radius = payload['uav_collision_radius']
    clearance = min((zone.segment_clearance(start, end, uav_radius=radius)
                     for start, end in zip(points[:-1], points[1:]) for zone in zones), default=None)
    vertical = np.diff(points[:, 2])[-10:]
    return {
        **{name: payload[name] for name in ('episode', 'window_index', 'stage_steps', 'scenario_id',
                                           'outcome', 'episode_length', 'episode_return') if name in payload},
        'start_altitude': float(points[0, 2]),
        'min_altitude': float(points[:, 2].min()),
        'max_altitude': float(points[:, 2].max()),
        'terminal_position': points[-1].tolist(),
        'min_zone_clearance': clearance,
        'mean_vz_last_10': float(vertical.mean() / payload['scenario_config']['dt']) if vertical.size else None,
    }


def _summaries(out, rows, *, source):
    atomic_json(out / 'summary.json', dict(source=source, rows=rows))
    columns = ('episode', 'scenario_id', 'window_index', 'stage_steps', 'outcome', 'start_altitude',
               'max_altitude', 'min_altitude', 'terminal_position', 'min_zone_clearance', 'mean_vz_last_10', 'view')
    lines = ['# V2 trajectory inspection', '', f'Matching episodes: {len(rows)}', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        cells = [str(row.get(key, '—')).replace('|', '\\|') for key in columns[:-1]]
        cells.append(f"[XY / XZ / YZ]({row['view']})" if 'view' in row else '—')
        lines.append('| ' + ' | '.join(cells) + ' |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _add_row(out, rows, row, payload, stem, limit, *, render):
    if render and (limit is None or len(rows) < limit):
        export_v2_trajectory_views(out / 'views', stem, payload)
        row['view'] = f'views/{stem}.png'
    rows.append(row)


def _training(args, out):
    seed = args.seed_dir.resolve()
    _check_output(out, seed)
    if not (seed / 'stage_start.json').exists():
        seed = seed / 'metrics_reports'
    stage = read_json(seed / 'stage_start.json')
    metadata = {key: stage[key] for key in ('scenario_config', 'reward_config', 'uav_collision_radius', 'model_type')}
    rows = []
    with gzip.open(seed / 'trajectories.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if (not _contains(args.windows, record['window_index'])
                    or not _contains(args.episodes, record['episode'])
                    or not _contains(args.step_range, record['stage_steps'])
                    or (args.outcomes and record['outcome'] not in args.outcomes)):
                continue
            payload = dict(record, **metadata, source='archived_training', selection_reasons=['inspection'])
            _add_row(out, rows, _row(payload), payload, f"ep_{record['episode']:06d}", args.limit,
                     render=not args.no_render)
    _summaries(out, rows, source=str(seed / 'trajectories.jsonl.gz'))


def _eval(args, out):
    from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_inputs
    from brain_uav.trainers.v2_validation import _evaluate_v2_scenarios

    _check_output(out, args.checkpoint, args.pool)
    inputs = load_diagnostic_inputs(model=None, checkpoint=args.checkpoint, validation_pool=args.pool,
                                    device=args.device)
    pool, checkpoint = inputs['pool'], inputs['checkpoint_payload']
    records = pool.scenarios
    if args.scenarios is not None:
        unknown = args.scenarios - {record['scenario_id'] for record in records}
        if unknown:
            raise ValueError(f'Unknown scenario IDs: {sorted(unknown)}')
        records = [record for record in records if record['scenario_id'] in args.scenarios]
    metadata = {key: checkpoint[key] for key in ('scenario_config', 'reward_config', 'uav_collision_radius', 'stage')}
    steps = checkpoint.get('training_result', checkpoint)
    rows = []

    def capture(detail, **trajectory):
        if args.outcomes and not ({detail['outcome'], outcome(detail)} & args.outcomes):
            return
        payload = dict(detail, **trajectory, **metadata, model_type=checkpoint.get('model_type', 'ann'),
                       stage_steps=steps['stage_steps'], global_steps=steps.get('global_steps_end', steps['stage_steps']),
                       source='checkpoint_replay', selection_reasons=['inspection'])
        _add_row(out, rows, _row(payload), payload, f'scene_{len(rows) + 1:04d}', args.limit, render=True)

    _evaluate_v2_scenarios(inputs['actor'], records, inputs['scenario'], pool.uav_collision_radius,
                           inputs['rewards'], curriculum_level=pool.curriculum_level, max_failures=0,
                           device=inputs['device'], reporter=None, cpu_deduplication=True, scenario_sink=capture)
    _summaries(out, rows, source={'checkpoint_sha256': inputs['checkpoint_hash'], 'pool_sha256': inputs['pool_hash']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    training = modes.add_parser('training')
    training.add_argument('--seed-dir', type=Path, required=True)
    training.add_argument('--windows', type=_ranges)
    training.add_argument('--episodes', type=_ranges)
    training.add_argument('--step-range', type=_ranges)
    training.add_argument('--no-render', action='store_true')
    evaluate = modes.add_parser('eval')
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--pool', type=Path, required=True)
    evaluate.add_argument('--scenarios', type=_csv)
    evaluate.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    for mode in (training, evaluate):
        mode.add_argument('--out', type=Path, required=True)
        mode.add_argument('--limit', type=_limit, help='Maximum number of rendered views.')
        mode.add_argument('--outcomes', type=_csv)
    args = parser.parse_args(argv)
    outcomes = set(V2_REPORT_OUTCOMES)
    if args.mode == 'eval':
        outcomes.update(('upper_boundary', 'other_boundary'))
    if args.outcomes and args.outcomes - outcomes:
        parser.error(f'Unknown outcomes: {sorted(args.outcomes - outcomes)}')
    out = args.out.resolve()
    (_training if args.mode == 'training' else _eval)(args, out)
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
