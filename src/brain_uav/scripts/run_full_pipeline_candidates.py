"""Run the full pipeline with parallel TD3 candidates for each curriculum stage."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..curriculum import CURRICULUM_LEVELS
from ..scripts.common import DEVICE_CHOICES, SNN_BACKEND_CHOICES, build_log_prefix
from ..scripts.generate_dataset import DATASET_VERSION
from ..utils.io import ensure_dir, save_json
from .run_full_pipeline import (
    FullRunLayout,
    FullRunStageError,
    build_subprocess_env,
    create_full_run_layout,
    find_latest_metrics_file,
    run_command,
    sanitize_tag,
    stage_sequence,
)


@dataclass(slots=True)
class CandidateRun:
    candidate_id: int
    seed: int
    stage: str
    run_dir: Path
    checkpoint_path: Path
    metrics_name: str
    log_root: Path
    command: list[str]
    process: subprocess.Popen | None = None
    return_code: int | None = None
    metrics_path: Path | None = None
    stopped_early: bool = False
    stop_reason: str | None = None
    total_steps: int | None = None
    episodes: int | None = None
    outcomes: dict[str, int] | None = None
    status: str = 'pending'

    def summary(self) -> dict[str, Any]:
        return {
            'candidate_id': self.candidate_id,
            'seed': self.seed,
            'stage': self.stage,
            'output': str(self.checkpoint_path),
            'checkpoint_path': str(self.checkpoint_path),
            'metrics_path': str(self.metrics_path) if self.metrics_path else None,
            'stopped_early': self.stopped_early,
            'stop_reason': self.stop_reason,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'outcomes': self.outcomes or {},
            'return_code': self.return_code,
            'status': self.status,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run full pipeline with parallel TD3 candidates per stage.')
    parser.add_argument('--model', choices=['ann', 'snn'], required=True)
    parser.add_argument('--tag', type=str, default='candidates')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-stage', choices=list(CURRICULUM_LEVELS), default='hard')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/full_run'))
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--snn-backend', choices=SNN_BACKEND_CHOICES, default='torch')
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--candidate-workers', type=int, default=4)
    parser.add_argument('--keep-candidate-runs', action='store_true')
    parser.add_argument('--poll-interval', type=float, default=5.0)
    return parser


def candidate_seed(base_seed: int, candidate_id: int) -> int:
    return int(base_seed) + int(candidate_id) * 100


def build_candidate_command(
    *,
    model: str,
    stage: str,
    init_checkpoint: Path,
    candidate: CandidateRun,
    summary_every_episodes: int,
    early_stop_min_steps: int,
    device: str,
    snn_backend: str,
) -> list[str]:
    return [
        sys.executable,
        '-m',
        'brain_uav.scripts.train_td3',
        '--model',
        model,
        '--curriculum-level',
        stage,
        '--init-checkpoint',
        str(init_checkpoint),
        '--output',
        str(candidate.checkpoint_path),
        '--metrics-out',
        candidate.metrics_name,
        '--log-root',
        str(candidate.log_root),
        '--seed',
        str(candidate.seed),
        '--early-stop-enabled',
        '--summary-every-episodes',
        str(summary_every_episodes),
        '--early-stop-windows',
        '4',
        '--early-stop-max-failures-per-window',
        '1',
        '--early-stop-goal-rate',
        '0.95',
        '--early-stop-min-steps',
        str(early_stop_min_steps),
        '--device',
        device,
        '--snn-backend',
        snn_backend,
    ]


def make_candidate_run(
    *,
    model: str,
    stage: str,
    candidate_id: int,
    base_seed: int,
    init_checkpoint: Path,
    stage_candidate_root: Path,
    summary_every_episodes: int,
    early_stop_min_steps: int,
    device: str,
    snn_backend: str,
) -> CandidateRun:
    seed = candidate_seed(base_seed, candidate_id)
    run_dir = ensure_dir(stage_candidate_root / f'cand_{candidate_id:02d}_seed{seed}')
    checkpoint_path = run_dir / f'td3_{model}_{stage}.pt'
    metrics_name = f'td3_{model}_{stage}_metrics.json'
    candidate = CandidateRun(
        candidate_id=candidate_id,
        seed=seed,
        stage=stage,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        metrics_name=metrics_name,
        log_root=ensure_dir(run_dir / 'logs'),
        command=[],
    )
    candidate.command = build_candidate_command(
        model=model,
        stage=stage,
        init_checkpoint=init_checkpoint,
        candidate=candidate,
        summary_every_episodes=summary_every_episodes,
        early_stop_min_steps=early_stop_min_steps,
        device=device,
        snn_backend=snn_backend,
    )
    return candidate


def load_candidate_metrics(candidate: CandidateRun) -> dict[str, Any]:
    metrics_path = find_latest_metrics_file(candidate.log_root, candidate.metrics_name)
    import json

    payload = json.loads(metrics_path.read_text(encoding='utf-8'))
    candidate.metrics_path = metrics_path
    candidate.stopped_early = payload.get('stopped_early') is True
    candidate.stop_reason = payload.get('stop_reason')
    candidate.total_steps = payload.get('steps')
    candidate.episodes = payload.get('episodes')
    candidate.outcomes = payload.get('outcomes') or {}
    return payload


def finalize_candidate(candidate: CandidateRun) -> None:
    try:
        load_candidate_metrics(candidate)
    except FileNotFoundError:
        candidate.metrics_path = None
        candidate.stopped_early = False
        candidate.stop_reason = 'metrics_not_found'
    if candidate.return_code != 0:
        candidate.status = 'failed'
    elif candidate.stopped_early:
        candidate.status = 'winner'
    else:
        candidate.status = 'completed_no_early_stop'


def terminate_candidate(candidate: CandidateRun, *, timeout: float = 20.0) -> None:
    if candidate.process is None or candidate.process.poll() is not None:
        return
    candidate.status = 'terminated'
    candidate.process.terminate()
    try:
        candidate.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        candidate.process.kill()
        candidate.process.wait(timeout=timeout)
    candidate.return_code = candidate.process.returncode


def copy_winner_outputs(winner: CandidateRun, *, final_checkpoint: Path, winner_log_dir: Path) -> None:
    ensure_dir(final_checkpoint.parent)
    shutil.copy2(winner.checkpoint_path, final_checkpoint)
    if winner_log_dir.exists():
        shutil.rmtree(winner_log_dir)
    shutil.copytree(winner.run_dir, winner_log_dir)


def cleanup_candidate_runs(candidates: list[CandidateRun], *, winner: CandidateRun | None, keep_candidate_runs: bool) -> None:
    if keep_candidate_runs:
        return
    for candidate in candidates:
        if winner is not None and candidate.candidate_id == winner.candidate_id:
            continue
        if candidate.run_dir.exists():
            shutil.rmtree(candidate.run_dir)


def run_candidate_stage(
    *,
    project_root: Path,
    env: dict[str, str],
    model: str,
    stage: str,
    init_checkpoint: Path,
    stage_checkpoint: Path,
    stage_candidate_root: Path,
    winner_log_dir: Path,
    base_seed: int,
    candidates: int,
    candidate_workers: int,
    keep_candidate_runs: bool,
    device: str,
    snn_backend: str,
    poll_interval: float = 5.0,
    summary_every_episodes: int = 15,
    early_stop_min_steps: int = 125000,
) -> tuple[CandidateRun, list[dict[str, Any]]]:
    if candidates < 1:
        raise ValueError('candidates must be at least 1')
    if candidate_workers < 1:
        raise ValueError('candidate_workers must be at least 1')

    pending = [
        make_candidate_run(
            model=model,
            stage=stage,
            candidate_id=candidate_id,
            base_seed=base_seed,
            init_checkpoint=init_checkpoint,
            stage_candidate_root=stage_candidate_root,
            summary_every_episodes=summary_every_episodes,
            early_stop_min_steps=early_stop_min_steps,
            device=device,
            snn_backend=snn_backend,
        )
        for candidate_id in range(candidates)
    ]
    all_candidates: list[CandidateRun] = list(pending)
    active: list[CandidateRun] = []
    winner: CandidateRun | None = None
    max_workers = min(candidate_workers, candidates)
    stage_prefix = build_log_prefix(model, stage)

    def launch_more() -> None:
        while pending and len(active) < max_workers:
            candidate = pending.pop(0)
            candidate.status = 'running'
            print(f'{stage_prefix} launching candidate {candidate.candidate_id} seed={candidate.seed}')
            candidate.process = subprocess.Popen(candidate.command, cwd=str(project_root), env=env)
            active.append(candidate)

    launch_more()
    while active:
        for candidate in list(active):
            assert candidate.process is not None
            return_code = candidate.process.poll()
            if return_code is None:
                continue
            candidate.return_code = return_code
            active.remove(candidate)
            finalize_candidate(candidate)
            print(
                f'{stage_prefix} candidate {candidate.candidate_id} finished '
                f'status={candidate.status} stopped_early={candidate.stopped_early}'
            )
            if candidate.stopped_early and return_code == 0:
                winner = candidate
                pending.clear()
                for other in list(active):
                    terminate_candidate(other)
                active.clear()
                break
        if winner is not None:
            break
        launch_more()
        if active:
            time.sleep(poll_interval)

    if winner is None:
        cleanup_candidate_runs(all_candidates, winner=None, keep_candidate_runs=keep_candidate_runs)
        summaries = [candidate.summary() for candidate in all_candidates]
        raise FullRunStageError(f'Stage {stage} had no candidate with stopped_early=True: {summaries}')

    copy_winner_outputs(winner, final_checkpoint=stage_checkpoint, winner_log_dir=winner_log_dir)
    cleanup_candidate_runs(all_candidates, winner=winner, keep_candidate_runs=keep_candidate_runs)
    return winner, [candidate.summary() for candidate in all_candidates]


def run_full_pipeline_candidates(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    layout: FullRunLayout = create_full_run_layout(output_root, args.model, args.tag)
    env = build_subprocess_env(project_root)
    stages = stage_sequence(args.max_stage)
    candidate_root = ensure_dir(layout.root / 'candidate_runs')
    report: dict[str, Any] = {
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
        'candidates': args.candidates,
        'candidate_workers': args.candidate_workers,
        'keep_candidate_runs': args.keep_candidate_runs,
        'candidate_stages': [],
    }

    dataset_path = layout.data_dir / f'bc_dataset_easy_{DATASET_VERSION}.npz'
    print(f'{build_log_prefix(args.model, "data")} starting dataset generation')
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
    report['dataset_path'] = str(dataset_path)

    bc_output = layout.models_dir / f'bc_{args.model}_final.pt'
    bc_best_output = layout.models_dir / f'bc_{args.model}_best.pt'
    bc_log_root = layout.logs_dir / 'bc'
    print(f'{build_log_prefix(args.model, "bc")} starting BC training')
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
    report['bc'] = {
        'final_checkpoint': str(bc_output),
        'best_checkpoint': str(bc_best_output),
        'metrics_path': str(bc_metrics),
    }

    init_checkpoint = bc_best_output
    stage_reports: list[dict[str, Any]] = []
    for stage in stages:
        stage_checkpoint = layout.models_dir / f'td3_{args.model}_{stage}.pt'
        winner_log_dir = layout.logs_dir / 'td3' / stage / 'winner_run'
        winner, summaries = run_candidate_stage(
            project_root=project_root,
            env=env,
            model=args.model,
            stage=stage,
            init_checkpoint=init_checkpoint,
            stage_checkpoint=stage_checkpoint,
            stage_candidate_root=ensure_dir(candidate_root / stage),
            winner_log_dir=winner_log_dir,
            base_seed=args.seed,
            candidates=args.candidates,
            candidate_workers=args.candidate_workers,
            keep_candidate_runs=args.keep_candidate_runs,
            device=args.device,
            snn_backend=args.snn_backend,
            poll_interval=args.poll_interval,
            summary_every_episodes=15,
            early_stop_min_steps=125000,
        )
        stage_report = {
            'stage': stage,
            'winner_candidate_id': winner.candidate_id,
            'winner_seed': winner.seed,
            'checkpoint': str(stage_checkpoint),
            'winner_log_dir': str(winner_log_dir),
            'metrics_path': str(winner.metrics_path) if winner.metrics_path else None,
            'stop_reason': winner.stop_reason,
            'candidates': summaries,
        }
        stage_reports.append(stage_report)
        report['candidate_stages'] = stage_reports
        save_json(layout.reports_dir / 'selection_summary.json', report)
        init_checkpoint = stage_checkpoint

    report['td3_stages'] = [
        {
            'stage': row['stage'],
            'checkpoint': row['checkpoint'],
            'metrics_path': row['metrics_path'],
            'stop_reason': row['stop_reason'],
            'winner_candidate_id': row['winner_candidate_id'],
            'winner_seed': row['winner_seed'],
        }
        for row in stage_reports
    ]
    report['final_checkpoint'] = str(init_checkpoint)
    save_json(layout.reports_dir / 'full_run_summary.json', report)
    save_json(layout.reports_dir / 'selection_summary.json', report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_full_pipeline_candidates(args)
    except (subprocess.CalledProcessError, FullRunStageError, FileNotFoundError) as exc:
        print(f'Candidate full run failed: {exc}')
        raise SystemExit(1) from exc
    print(report)


if __name__ == '__main__':
    main()
