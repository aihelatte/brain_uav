"""Run pure rollout tests for expert baseline planners."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from ..baselines import ArtificialPotentialFieldPlanner, HeuristicPlanner
from ..config import ExperimentConfig
from ..curriculum import parse_curriculum_mix
from ..scenarios import DEFAULT_BENCHMARK_SUITE_PATH, build_benchmark_scenarios, load_benchmark_suite
from ..scripts.common import make_env
from ..scripts.train_td3 import export_episode_result
from ..utils.io import ensure_dir, now_timestamp, save_json
from ..utils.seeding import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run baseline planner rollouts without training a model.')
    parser.add_argument('--planner', choices=['heuristic', 'apf'], required=True)
    parser.add_argument('--evaluation-mode', choices=['curriculum', 'benchmark'], default='curriculum')
    parser.add_argument('--curriculum-level', choices=['easy', 'easy_two_zone', 'medium', 'hard'], default='hard')
    parser.add_argument('--curriculum-mix', type=str, default=None)
    parser.add_argument('--benchmark-suite', type=Path, default=DEFAULT_BENCHMARK_SUITE_PATH)
    parser.add_argument('--max-total-steps', type=int, default=50000)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-root', type=Path, default=Path('outputs/baseline_tests'))
    parser.add_argument('--save-artifacts', action='store_true')
    return parser


def build_planner(planner_name: str, env):
    if planner_name == 'heuristic':
        return HeuristicPlanner(env)
    if planner_name == 'apf':
        return ArtificialPotentialFieldPlanner(env)
    raise ValueError(f'Unsupported planner: {planner_name}')


def episode_min_zone_clearance(env) -> float:
    if not getattr(env, 'zones', None):
        return float('inf')
    min_clearance = float('inf')
    for point in getattr(env, 'trajectory', []):
        pos = np.asarray(point, dtype=np.float32)
        for zone in env.zones:
            distance = float(
                np.linalg.norm(
                    np.array([pos[0] - zone.center_xy[0], pos[1] - zone.center_xy[1], pos[2]], dtype=np.float32)
                )
            )
            min_clearance = min(min_clearance, distance - float(zone.radius))
    return min_clearance


def _jsonify_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload['final_state'] = [float(value) for value in payload['final_state']]
    payload['trajectory'] = [[float(coord) for coord in point] for point in payload['trajectory']]
    payload['scenario'] = payload['scenario']
    payload['info'] = payload['info']
    return payload


def _make_output_dir(args: argparse.Namespace) -> Path:
    timestamp = now_timestamp()
    suffix = (
        f'{args.evaluation_mode}_{args.curriculum_level}'
        if args.evaluation_mode == 'curriculum'
        else args.evaluation_mode
    )
    return ensure_dir(args.output_root / f'{timestamp}_{args.planner}_{suffix}')


def _record_info(info: dict[str, Any], clearance: float) -> dict[str, Any]:
    return {
        'goal_distance': float(info.get('goal_distance', 0.0)),
        'segment_goal_distance': float(info.get('segment_goal_distance', 0.0)),
        'goal_reached_by_segment': bool(info.get('goal_reached_by_segment', False)),
        'progress': float(info.get('progress', 0.0)),
        'steps': int(info.get('steps', 0)),
        'curriculum_level': info.get('curriculum_level'),
        'episode_min_zone_clearance': float(clearance),
        'active_goal_radius': float(info.get('active_goal_radius', 0.0)),
    }


def _build_summary(
    *,
    args: argparse.Namespace,
    episodes_requested: int | None,
    output_dir: Path,
    artifacts_dir: Path | None,
    records: list[dict[str, Any]],
    total_steps: int,
) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    for record in records:
        outcome = str(record['outcome'])
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    episodes = len(records)
    success_count = outcomes.get('goal', 0)
    final_goal_distances = [float(record['info']['goal_distance']) for record in records]
    clearances = [float(record['info']['episode_min_zone_clearance']) for record in records]
    finite_clearances = [value for value in clearances if np.isfinite(value)]
    simple_records = [
        {
            'episode': record['episode'],
            'outcome': record['outcome'],
            'length': record['length'],
            'return': record['return'],
            'total_steps_at_end': record['total_steps_at_end'],
            'episode_min_zone_clearance': record['info']['episode_min_zone_clearance'],
        }
        for record in records
    ]
    return {
        'planner': args.planner,
        'evaluation_mode': args.evaluation_mode,
        'curriculum_level': args.curriculum_level,
        'curriculum_mix': args.curriculum_mix,
        'seed': args.seed,
        'max_total_steps': args.max_total_steps,
        'episodes_requested': episodes_requested,
        'episodes': episodes,
        'total_steps': total_steps,
        'success_count': success_count,
        'success_rate': 0.0 if episodes == 0 else success_count / episodes,
        'outcomes': outcomes,
        'collision_rate': 0.0 if episodes == 0 else outcomes.get('collision', 0) / episodes,
        'boundary_rate': 0.0 if episodes == 0 else outcomes.get('boundary', 0) / episodes,
        'ground_rate': 0.0 if episodes == 0 else outcomes.get('ground', 0) / episodes,
        'timeout_rate': 0.0 if episodes == 0 else outcomes.get('timeout', 0) / episodes,
        'avg_steps': 0.0 if episodes == 0 else statistics.mean(record['length'] for record in records),
        'avg_return': 0.0 if episodes == 0 else statistics.mean(record['return'] for record in records),
        'median_steps': 0.0 if episodes == 0 else float(statistics.median(record['length'] for record in records)),
        'avg_goal_distance_final': 0.0 if episodes == 0 else statistics.mean(final_goal_distances),
        'avg_episode_min_zone_clearance': (
            float('inf') if not finite_clearances else statistics.mean(finite_clearances)
        ),
        'output_dir': str(output_dir),
        'artifacts_dir': None if artifacts_dir is None else str(artifacts_dir),
        'records': simple_records,
    }


def rollout_planner(
    env,
    planner,
    *,
    seed: int,
    episodes: int | None,
    max_total_steps: int,
    evaluation_mode: str,
    config_payload: dict[str, Any],
    artifacts_dir: Path | None = None,
    named_scenarios: list[Any] | None = None,
    save_artifacts: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    total_steps = 0
    episode_idx = 0
    target_episodes = episodes
    while True:
        if target_episodes is not None and episode_idx >= target_episodes:
            break
        if target_episodes is None and episode_idx > 0 and total_steps >= max_total_steps:
            break

        if evaluation_mode == 'benchmark':
            assert named_scenarios is not None
            scenario = named_scenarios[episode_idx]
            obs, _ = env.reset(options={'scenario': scenario.scenario})
        else:
            obs, _ = env.reset(seed=seed + episode_idx)

        done = False
        episode_return = 0.0
        info: dict[str, Any] = {}
        while not done:
            action = planner.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = bool(terminated or truncated)

        total_steps += int(env.steps)
        clearance = episode_min_zone_clearance(env)
        record = {
            'episode': episode_idx + 1,
            'seed': seed + episode_idx,
            'total_steps_at_end': total_steps,
            'return': float(episode_return),
            'length': int(env.steps),
            'outcome': str(info.get('outcome', 'other')),
            'final_state': [float(value) for value in env.state.tolist()],
            'info': _record_info(info, clearance),
            'scenario': env.export_scenario(),
            'trajectory': [point.astype(float).tolist() for point in env.trajectory],
        }
        if save_artifacts and artifacts_dir is not None:
            artifact_paths = export_episode_result(
                artifacts_dir,
                f"ep{record['episode']:05d}_{record['outcome']}",
                {
                    'episode': record['episode'],
                    'total_steps': record['total_steps_at_end'],
                    'return': record['return'],
                    'length': record['length'],
                    'outcome': record['outcome'],
                    'actor_loss': 0.0,
                    'critic_loss': 0.0,
                    'scenario': record['scenario'],
                    'trajectory': record['trajectory'],
                    'final_state': record['final_state'],
                    'info': record['info'],
                },
                config_payload,
            )
            record['artifacts'] = artifact_paths
        records.append(_jsonify_record(record))
        episode_idx += 1
    return records, total_steps


def run_baseline_test(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ExperimentConfig()
    set_global_seed(args.seed)
    output_dir = _make_output_dir(args)
    artifacts_dir = ensure_dir(output_dir / 'artifacts') if args.save_artifacts else None
    episodes_requested = args.episodes

    curriculum_mix = None
    named_scenarios = None
    if args.evaluation_mode == 'curriculum':
        curriculum_mix = parse_curriculum_mix(args.curriculum_mix, fallback_level=args.curriculum_level)
        env = make_env(
            cfg,
            seed=args.seed,
            curriculum_level=args.curriculum_level,
            curriculum_mix=curriculum_mix,
            goal_radius_curriculum_enabled=True,
        )
    else:
        suite = load_benchmark_suite(args.benchmark_suite)
        named_scenarios = build_benchmark_scenarios(args.benchmark_suite)
        if args.episodes is None:
            args.episodes = int(suite['total_scenarios'])
        if args.episodes > len(named_scenarios):
            raise ValueError(
                f'--episodes={args.episodes} exceeds benchmark suite size {len(named_scenarios)} at {args.benchmark_suite}.'
            )
        env = make_env(
            cfg,
            seed=args.seed,
            goal_radius_curriculum_enabled=True,
        )

    planner = build_planner(args.planner, env)
    config_payload = cfg.to_dict()
    config_payload['evaluation_mode'] = args.evaluation_mode
    config_payload['curriculum_level'] = args.curriculum_level
    config_payload['curriculum_mix'] = curriculum_mix

    records, total_steps = rollout_planner(
        env,
        planner,
        seed=args.seed,
        episodes=args.episodes,
        max_total_steps=args.max_total_steps,
        evaluation_mode=args.evaluation_mode,
        config_payload=config_payload,
        artifacts_dir=artifacts_dir,
        named_scenarios=named_scenarios,
        save_artifacts=args.save_artifacts,
    )

    episodes_path = save_json(output_dir / 'episodes.json', records)
    summary = _build_summary(
        args=args,
        episodes_requested=episodes_requested,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        records=records,
        total_steps=total_steps,
    )
    summary['episodes_path'] = str(episodes_path)
    save_json(output_dir / 'summary.json', summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_baseline_test(args)
    print(summary)


if __name__ == '__main__':
    main()
