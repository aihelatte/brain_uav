"""Run the full pipeline from dataset generation to curriculum training."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..curriculum import CURRICULUM_LEVELS
from ..scripts.common import DEVICE_CHOICES, SNN_BACKEND_CHOICES, build_log_prefix
from ..scripts.generate_dataset import DATASET_VERSION
from ..utils.io import ensure_dir, save_json


class FullRunStageError(RuntimeError):
    """Raised when one full-run stage fails validation."""


@dataclass(slots=True)
class FullRunLayout:
    root: Path
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    reports_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run dataset generation, BC, and full curriculum TD3 training.')
    parser.add_argument('--model', choices=['ann', 'snn'], required=True)
    parser.add_argument('--tag', type=str, default='run')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-stage', choices=list(CURRICULUM_LEVELS), default='hard')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/full_run'))
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--snn-backend', choices=SNN_BACKEND_CHOICES, default='torch')
    return parser


def sanitize_tag(tag: str) -> str:
    cleaned = ''.join(ch.lower() if ch.isalnum() else '_' for ch in tag.strip())
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    return cleaned.strip('_') or 'run'


def make_run_name(model: str, tag: str, now: datetime | None = None) -> str:
    clock = now or datetime.now()
    return f"{clock.strftime('%m%d_%H%M%S')}_{model}_{sanitize_tag(tag)}"


def create_full_run_layout(output_root: Path, model: str, tag: str, now: datetime | None = None) -> FullRunLayout:
    run_root = ensure_dir(output_root / make_run_name(model, tag, now=now))
    data_dir = ensure_dir(run_root / 'data')
    models_dir = ensure_dir(run_root / 'models')
    logs_dir = ensure_dir(run_root / 'logs')
    reports_dir = ensure_dir(run_root / 'reports')
    return FullRunLayout(
        root=run_root,
        data_dir=data_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
    )


def stage_sequence(max_stage: str) -> list[str]:
    max_idx = CURRICULUM_LEVELS.index(max_stage)
    return list(CURRICULUM_LEVELS[: max_idx + 1])


def build_subprocess_env(project_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    src_root = str(project_root / 'src')
    current = env.get('PYTHONPATH')
    env['PYTHONPATH'] = src_root if not current else f'{src_root}{os.pathsep}{current}'
    return env


def run_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def find_latest_metrics_file(log_root: Path, metrics_name: str) -> Path:
    matches = sorted(log_root.rglob(metrics_name), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f'Could not find metrics file {metrics_name} under {log_root}')
    return matches[0]


def ensure_stage_stopped_early(metrics_path: Path, stage: str) -> dict:
    import json

    payload = json.loads(metrics_path.read_text(encoding='utf-8'))
    if payload.get('stopped_early') is True:
        return payload
    raise FullRunStageError(f'Stage {stage} did not stop early successfully: {metrics_path}')


def run_full_pipeline(args: argparse.Namespace) -> dict:
    project_root = Path(__file__).resolve().parents[3]
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    layout = create_full_run_layout(output_root, args.model, args.tag)
    env = build_subprocess_env(project_root)
    stages = stage_sequence(args.max_stage)
    report: dict[str, object] = {
        'model': args.model,
        'tag': sanitize_tag(args.tag),
        'seed': args.seed,
        'max_stage': args.max_stage,
        'stages': stages,
        'run_root': str(layout.root),
        'timesteps_source': 'train_td3 defaults',
        'dataset_version': DATASET_VERSION,
        'device': args.device,
        'snn_backend': args.snn_backend if args.model == 'snn' else None,
    }

    dataset_path = layout.data_dir / f'bc_dataset_easy_{DATASET_VERSION}.npz'
    data_prefix = build_log_prefix(args.model, 'data')
    print(f'{data_prefix} starting dataset generation')
    run_command(
        [
            sys.executable,
            '-m',
            'brain_uav.scripts.generate_dataset',
            '--output',
            str(dataset_path),
            '--seed',
            str(args.seed),
            '--curriculum-level',
            'easy',
        ],
        cwd=project_root,
        env=env,
    )
    print(f'{data_prefix} dataset saved to {dataset_path}')
    report['dataset_path'] = str(dataset_path)

    bc_output = layout.models_dir / f'bc_{args.model}_final.pt'
    bc_best_output = layout.models_dir / f'bc_{args.model}_best.pt'
    bc_log_root = layout.logs_dir / 'bc'
    bc_prefix = build_log_prefix(args.model, 'bc')
    print(f'{bc_prefix} starting BC training')
    run_command(
        [
            sys.executable,
            '-m',
            'brain_uav.scripts.train_bc',
            '--dataset',
            str(dataset_path),
            '--model',
            args.model,
            '--output',
            str(bc_output),
            '--best-output',
            str(bc_best_output),
            '--metrics-out',
            f'bc_{args.model}_metrics.json',
            '--log-root',
            str(bc_log_root),
            '--device',
            args.device,
            '--snn-backend',
            args.snn_backend,
        ],
        cwd=project_root,
        env=env,
    )
    bc_metrics = find_latest_metrics_file(bc_log_root, f'bc_{args.model}_metrics.json')
    print(f'{bc_prefix} finished BC training best_checkpoint={bc_best_output}')
    report['bc'] = {
        'final_checkpoint': str(bc_output),
        'best_checkpoint': str(bc_best_output),
        'metrics_path': str(bc_metrics),
    }

    init_checkpoint = bc_best_output
    stage_reports: list[dict[str, object]] = []
    for stage in stages:
        stage_output = layout.models_dir / f'td3_{args.model}_{stage}.pt'
        stage_log_root = layout.logs_dir / 'td3' / stage
        metrics_name = f'td3_{args.model}_{stage}_metrics.json'
        stage_prefix = build_log_prefix(args.model, stage)
        print(f'{stage_prefix} starting TD3 stage')
        run_command(
            [
                sys.executable,
                '-m',
                'brain_uav.scripts.train_td3',
                '--model',
                args.model,
                '--curriculum-level',
                stage,
                '--init-checkpoint',
                str(init_checkpoint),
                '--output',
                str(stage_output),
                '--metrics-out',
                metrics_name,
                '--log-root',
                str(stage_log_root),
                '--seed',
                str(args.seed),
                '--early-stop-enabled',
                '--summary-every-episodes',
                '15',
                '--early-stop-windows',
                '4',
                '--early-stop-max-failures-per-window',
                '1',
                '--early-stop-goal-rate',
                '0.95',
                '--early-stop-min-steps',
                '125000',
                '--device',
                args.device,
                '--snn-backend',
                args.snn_backend,
            ],
            cwd=project_root,
            env=env,
        )
        metrics_path = find_latest_metrics_file(stage_log_root, metrics_name)
        metrics_payload = ensure_stage_stopped_early(metrics_path, stage)
        print(f'{stage_prefix} finished TD3 stage stop_reason={metrics_payload.get("stop_reason")}')
        stage_reports.append(
            {
                'stage': stage,
                'checkpoint': str(stage_output),
                'metrics_path': str(metrics_path),
                'stop_reason': metrics_payload.get('stop_reason'),
            }
        )
        init_checkpoint = stage_output

    report['td3_stages'] = stage_reports
    report['final_checkpoint'] = str(init_checkpoint)
    save_json(layout.reports_dir / 'full_run_summary.json', report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_full_pipeline(args)
    except (subprocess.CalledProcessError, FullRunStageError, FileNotFoundError) as exc:
        print(f'Full run failed: {exc}')
        raise SystemExit(1) from exc
    print(report)


if __name__ == '__main__':
    main()
