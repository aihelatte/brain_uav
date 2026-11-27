"""Build a readable V2 report from a pipeline or historical course directory."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from brain_uav.scripts.v2_formal_easy_guidance import OUTCOMES, outcome, read_json
from brain_uav.trainers.v2_reporting import V2_OUTCOME_COLORS, _plot_training_windows
from brain_uav.v2_curriculum import V2_TD3_STAGES


def _batches(root):
    if (root / 'manifest.json').is_file() and (root / 'runs').is_dir():
        for directory in sorted((root / 'runs').iterdir()):
            if directory.is_dir() and directory.name in _BATCH_NAMES:
                yield directory.name, root / 'winners' / f'{directory.name}.selection.json', sorted(directory.glob('seed*'))
        return
    if not (root / 'batches').is_dir():
        raise ValueError(f'No V2 runs or historical batches in {root}')
    for directory in sorted((root / 'batches').iterdir()):
        if not directory.is_dir() or directory.name not in _BATCH_NAMES:
            continue
        selection = directory / 'selection.json'
        attempts = sorted(directory.glob('attempt-*/train'))
        finished = [path for path in attempts if (path / 'batch_summary.json').is_file()
                    and read_json(path / 'batch_summary.json')['execution_completed'] is True]
        if len(finished) > 1:
            raise ValueError(f'Multiple completed attempts in {directory}')
        if finished or len(attempts) == 1:
            seeds = sorted((finished or attempts)[0].glob('*_seed*'))
        elif selection.is_file():
            # A continued course may keep its imported candidates in the source course.
            seeds = [Path(row['terminal_evaluation']).parent
                     for row in read_json(selection).get('candidates', ())]
        else:
            seeds = []
        yield directory.name, selection, seeds


_COLORS = {**V2_OUTCOME_COLORS, 'upper_boundary': 'tab:red', 'other_boundary': 'tab:orange'}
_ORDER = {stage: i for i, stage in enumerate(V2_TD3_STAGES)}
_BATCH_NAMES = {f'{stage}-{model}' for stage in V2_TD3_STAGES for model in ('ann', 'snn')}


def _bars(path, labels, counts, *, title, rates):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7, min(18, len(labels) * 1.35)), 4.6))
    try:
        positions = list(range(len(labels)))
        bottom = [0] * len(labels)
        totals = [sum(row.values()) for row in counts]
        for key in OUTCOMES:
            values = [row[key] * 100 / total if rates and total else row[key]
                      for row, total in zip(counts, totals)]
            ax.bar(positions, values, bottom=bottom, label=key, color=_COLORS[key])
            bottom = [last + value for last, value in zip(bottom, values)]
        ax.set_xticks(positions, labels)
        ax.set_ylabel('Fixed evaluation (%)' if rates else 'Fixed evaluation (scenarios)')
        if rates:
            ax.set_ylim(0, 100)
        if not labels:
            ax.text(0.5, 0.5, 'No completed evaluations', transform=ax.transAxes, ha='center')
        ax.set_title(title)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)


def build_report(run_dir, output_dir=None):
    root = Path(run_dir).resolve()
    target = Path(output_dir).resolve() if output_dir is not None else root / 'report'
    if target == root or any(target.is_relative_to(root / name)
                             for name in ('runs', 'batches', '.batches', 'winners', 'pools')):
        raise ValueError('Choose a report directory outside the raw V2 data directories.')
    manifest = next((read_json(root / name) for name in ('manifest.json', 'course.json')
                     if (root / name).is_file()), {})
    config = manifest.get('plan', manifest)
    rows, groups = [], []
    target.mkdir(parents=True, exist_ok=True)
    for batch, selection, seeds in sorted(_batches(root), key=lambda entry: (
            _ORDER.get(entry[0].split('-')[0], len(_ORDER)), entry[0])):
        stage, model = batch.split('-')
        selection_data = read_json(selection) if selection.exists() else {}
        winner = (selection_data.get('winner') or {}).get('seed')
        directory = target / batch
        batch_rows = []
        for candidate in seeds:
            path = candidate / 'terminal_evaluation.json'
            if not path.is_file():
                continue
            seed = int(candidate.name.split('seed')[-1])
            evaluation = read_json(path)
            counts = Counter(outcome(scene) for scene in evaluation['evaluation']['scenarios'])
            log = candidate / 'metrics_reports' if (candidate / 'metrics_reports').is_dir() else candidate
            start = read_json(log / 'stage_start.json') if (log / 'stage_start.json').exists() else {}
            end = read_json(log / 'stage_end.json') if (log / 'stage_end.json').exists() else {}
            worker = read_json(candidate / 'worker_summary.json') if (candidate / 'worker_summary.json').exists() else {}
            curve = log / 'training_curves.png'
            curve_name = f'seed{seed}_training_curves.png'
            curve_link = ''
            if curve.exists() or (log / 'windows.jsonl').exists():
                directory.mkdir(parents=True, exist_ok=True)
                if curve.exists():
                    shutil.copyfile(curve, directory / curve_name)
                else:
                    with (log / 'windows.jsonl').open(encoding='utf-8') as stream:
                        _plot_training_windows(directory / curve_name, [json.loads(line) for line in stream],
                            required_qualified_windows=start.get('formal_config', {}).get('consecutive_qualified_windows', 4))
                curve_link = f'[curve]({batch}/{curve_name})'
            batch_rows.append(dict(stage=stage, model=model, seed=seed,
                passed=evaluation['formal_passed'], won=seed == winner, steps=evaluation['actual_steps'],
                elapsed=end.get('stage_elapsed_seconds', worker.get('process_wall_seconds')),
                max_steps=start.get('max_steps'), curve=curve_link,
                counts={key: counts[key] for key in OUTCOMES}))
        reported_seeds = {row['seed'] for row in batch_rows}
        for item in selection_data.get('candidates', ()):
            if item['seed'] in reported_seeds or not all(key in item for key in OUTCOMES):
                continue
            # Imported courses can be copied without the original workers' logs.
            batch_rows.append(dict(stage=stage, model=model, seed=item['seed'],
                passed=item['formal_passed'], won=item['seed'] == winner, steps=item['steps'],
                elapsed=None, max_steps=None, curve='', counts={key: item[key] for key in OUTCOMES}))
        if batch_rows:
            batch_rows.sort(key=lambda row: row['seed'])
            directory.mkdir(parents=True, exist_ok=True)
            _bars(directory / 'batch_seeds_summary.png',
                  [str(row['seed']) + (' ★' if row['won'] else '') for row in batch_rows],
                  [row['counts'] for row in batch_rows], title=f'{batch} · fixed evaluation', rates=False)
            groups.append((batch, {key: sum(row['counts'][key] for row in batch_rows) for key in OUTCOMES}))
            rows.extend(batch_rows)
    _bars(target / 'overview_all_stages.png', [batch for batch, _ in groups],
          [counts for _, counts in groups], title='Fixed evaluation across completed seeds', rates=True)
    if config.get('format') == 'v2_training_pipeline':
        seen = {(row['stage'], row['model'], row['seed']) for row in rows}
        for stage in config['stages']:
            for model in config['models']:
                status = manifest.get('batches', {}).get(f'{stage}-{model}', {}).get('status')
                for seed in config['seeds']:
                    if (stage, model, seed) not in seen:
                        rows.append(dict(stage=stage, model=model, seed=seed, passed=None, won=False,
                            steps='—', elapsed=None, max_steps=config['max_steps'], curve='',
                            counts=dict.fromkeys(OUTCOMES, '—'),
                            status=('blocked' if status == 'blocked_by_no_formal_pass' else
                                    'pending' if manifest.get('status') == 'running' else 'incomplete')))
    rows.sort(key=lambda row: (_ORDER[row['stage']], row['model'], row['seed']))
    models = config.get('models') or sorted({row['model'] for row in rows})
    stages = config.get('stages') or [stage for stage in V2_TD3_STAGES if any(row['stage'] == stage for row in rows)]
    seeds = config.get('seeds') or sorted({row['seed'] for row in rows})
    plan_budgets = {spec['formal_config']['max_steps'] for batch in config.get('batches', {}).values()
                    for spec in batch if isinstance(spec, dict) and 'formal_config' in spec}
    budgets = config.get('max_steps') or sorted(plan_budgets | {
        row['max_steps'] for row in rows if row['max_steps'] is not None})
    lines = ['# V2 training report', '',
             f'- Models: {", ".join(models)}', f'- Stages: {", ".join(stages)}',
             f'- Seeds: {", ".join(map(str, seeds))}', f'- Stage step cap: {budgets or "unavailable"}',
             '', '![Overview](overview_all_stages.png)', '',
             '| Stage | Model | Seed | Passed | Winner | ' + ' | '.join(OUTCOMES) + ' | Steps | Elapsed (s) | Plot |',
             '| ' + ' | '.join(['---'] * (8 + len(OUTCOMES))) + ' |']
    for row in rows:
        elapsed = f"{row['elapsed']:.1f}" if row['elapsed'] is not None else '—'
        lines.append('| ' + ' | '.join(map(str, (
            row['stage'], row['model'].upper(), row['seed'], row.get('status', 'yes' if row['passed'] else 'no'),
            'yes' if row['won'] else 'no', *row['counts'].values(), row['steps'], elapsed, row['curve'],
        ))) + ' |')
    for batch, _ in groups:
        lines.extend(['', f'![{batch} seeds]({batch}/batch_seeds_summary.png)'])
    (target / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    print(build_report(args.run_dir, args.output_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
