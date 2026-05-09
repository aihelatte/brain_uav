"""Evaluate APF / Heuristic baselines on target-switch curriculum episodes."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys
from typing import Any, TextIO

import numpy as np

from ..baselines import ArtificialPotentialFieldPlanner, HeuristicPlanner
from ..baselines.common import heading_to_action
from ..config import ExperimentConfig
from ..envs import TargetSwitchTrajectoryEnv
from ..scripts.run_target_switch import _plot_episode
from ..target_switch import TARGET_SWITCH_LEVELS, target_switch_config_for_level
from ..utils.io import ensure_dir, now_timestamp, save_json
from ..utils.seeding import set_global_seed


class _TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.secondary.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate APF / Heuristic baselines on target-switch episodes.')
    parser.add_argument('--policy', choices=['apf', 'heuristic'], required=True)
    parser.add_argument('--target-switch-level', choices=TARGET_SWITCH_LEVELS, default='target_switch_hard')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('outputs/target_switch_baselines'))
    parser.add_argument('--summary-every-episodes', type=int, default=50)
    parser.add_argument('--allow-fallback', action='store_true')
    parser.add_argument('--save-episode-json', action='store_true')
    parser.add_argument('--save-failures-only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-saved-episodes', type=int, default=100)
    parser.add_argument('--save-snapshots', action='store_true')
    parser.add_argument('--snapshot-every-window', type=int, default=5)
    return parser


def build_baseline_policy(policy: str, env):
    if policy == 'apf':
        return ArtificialPotentialFieldPlanner(env)
    if policy == 'heuristic':
        return HeuristicPlanner(env)
    raise ValueError(f'Unsupported baseline policy: {policy}')


def _build_output_dir(args: argparse.Namespace) -> Path:
    return ensure_dir(args.output / f'{now_timestamp()}_{args.policy}_{args.target_switch_level}')


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _fallback_goal_action(env) -> np.ndarray:
    direction = np.asarray(env.goal - env.state[:3], dtype=np.float32)
    limits = np.array([env.scenario.delta_gamma_max, env.scenario.delta_psi_max], dtype=np.float32)
    return heading_to_action(float(env.state[3]), float(env.state[4]), direction, limits)


def _safe_policy_action(
    policy,
    obs: np.ndarray,
    env,
    *,
    allow_fallback: bool,
    episode: int,
    step: int,
) -> tuple[np.ndarray, bool, dict[str, Any] | None]:
    try:
        action = policy.act(obs)
        return np.asarray(action, dtype=np.float32), False, None
    except Exception as exc:
        if not allow_fallback:
            raise
        error_payload = {
            'episode': int(episode),
            'step': int(step),
            'error_type': type(exc).__name__,
            'error_message': str(exc),
        }
        return _fallback_goal_action(env), True, error_payload


def _episode_switch_success(record: dict[str, Any]) -> bool:
    return bool(record.get('switched')) and record.get('outcome') == 'goal' and not bool(record.get('pre_switch_done', False))


def _target_switch_window_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    raw_goal_count = 0
    switch_success_count = 0
    for record in records:
        outcomes[record['outcome']] = outcomes.get(record['outcome'], 0) + 1
        if bool(record.get('raw_goal', False)):
            raw_goal_count += 1
        if bool(record.get('switch_success', False)):
            switch_success_count += 1
    count = len(records)
    return {
        'episode_start': int(records[0]['episode']) if records else 0,
        'episode_end': int(records[-1]['episode']) if records else 0,
        'episode_count': count,
        'raw_goal_count': raw_goal_count,
        'raw_goal_rate': raw_goal_count / count if count else 0.0,
        'switch_success_count': switch_success_count,
        'switch_success_rate': switch_success_count / count if count else 0.0,
        'goal_count': switch_success_count,
        'goal_rate': switch_success_count / count if count else 0.0,
        'collision_count': outcomes.get('collision', 0),
        'boundary_count': outcomes.get('boundary', 0),
        'ground_count': outcomes.get('ground', 0),
        'timeout_count': outcomes.get('timeout', 0),
        'pre_switch_done_count': sum(1 for record in records if bool(record.get('pre_switch_done', False))),
        'avg_return': float(np.mean([record.get('return', 0.0) for record in records])) if records else 0.0,
        'avg_length': float(np.mean([record.get('length', 0) for record in records])) if records else 0.0,
        'avg_final_to_new_distance': _mean_or_none(records, 'final_to_new_distance'),
        'avg_post_switch_steps': float(np.mean([record.get('post_switch_steps', 0) or 0 for record in records]))
        if records
        else 0.0,
        'avg_switch_alignment_reward': float(
            np.mean([record.get('switch_alignment_reward_mean', 0.0) or 0.0 for record in records])
        )
        if records
        else 0.0,
        'avg_ceiling_penalty': float(np.mean([record.get('ceiling_penalty_mean', 0.0) or 0.0 for record in records]))
        if records
        else 0.0,
    }


def _best_window_score(window: dict[str, Any]) -> tuple[Any, ...]:
    final_distance = window.get('avg_final_to_new_distance')
    if final_distance is None:
        final_distance = float('inf')
    return (
        float(window.get('switch_success_rate', 0.0)),
        int(window.get('switch_success_count', 0)),
        -int(window.get('ground_count', 0)),
        -int(window.get('boundary_count', 0)),
        -int(window.get('collision_count', 0)),
        -int(window.get('timeout_count', 0)),
        -int(window.get('pre_switch_done_count', 0)),
        -float(final_distance),
        float(window.get('avg_return', 0.0)),
    )


def _is_better_window(candidate: dict[str, Any], current_best: dict[str, Any] | None) -> bool:
    return current_best is None or _best_window_score(candidate) > _best_window_score(current_best)


def _mean_or_none(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record.get(key) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _records_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    raw_goal_count = 0
    switch_success_count = 0
    for record in records:
        outcomes[record['outcome']] = outcomes.get(record['outcome'], 0) + 1
        raw_goal_count += int(bool(record.get('raw_goal', False)))
        switch_success_count += int(bool(record.get('switch_success', False)))
    return {
        'episodes': len(records),
        'raw_goal_count': raw_goal_count,
        'raw_goal_rate': raw_goal_count / len(records) if records else 0.0,
        'switch_success_count': switch_success_count,
        'switch_success_rate': switch_success_count / len(records) if records else 0.0,
        'success_count': switch_success_count,
        'success_rate': switch_success_count / len(records) if records else 0.0,
        'outcomes': outcomes,
        'avg_return': float(np.mean([record['return'] for record in records])) if records else 0.0,
        'avg_length': float(np.mean([record['length'] for record in records])) if records else 0.0,
        'avg_final_to_new_distance': _mean_or_none(records, 'final_to_new_distance'),
        'avg_post_switch_steps': float(np.mean([record['post_switch_steps'] for record in records])) if records else 0.0,
        'avg_switch_alignment_reward': float(
            np.mean([record.get('switch_alignment_reward_mean', 0.0) for record in records])
        )
        if records
        else 0.0,
        'avg_ceiling_penalty': float(np.mean([record.get('ceiling_penalty_mean', 0.0) for record in records]))
        if records
        else 0.0,
        'pre_switch_done_count': sum(1 for record in records if bool(record.get('pre_switch_done', False))),
        'fallback_count': sum(int(record.get('fallback_count', 0)) for record in records),
    }


def _build_episode_payload(
    *,
    args: argparse.Namespace,
    env,
    info: dict[str, Any],
    record: dict[str, Any],
    target_switch_cfg,
) -> dict[str, Any]:
    return {
        'episode': record['episode'],
        'seed': args.seed + record['episode'] - 1,
        'policy': args.policy,
        'target_switch_level': args.target_switch_level,
        'base_curriculum_level': target_switch_cfg.base_curriculum_level,
        'initial_scenario': env.export_scenario(),
        'old_goal': record['old_goal'],
        'new_goal': record['new_goal'],
        'switch_step': record['switch_step'],
        'switch_position': record['switch_position'],
        'switched': record['switched'],
        'pre_switch_done': record['pre_switch_done'],
        'total_steps': record['total_steps'],
        'post_switch_steps': record['post_switch_steps'],
        'outcome': record['outcome'],
        'return': record['return'],
        'boundary_reason': record['boundary_reason'],
        'final_state': env.state.tolist(),
        'final_to_new_distance': record['final_to_new_distance'],
        'min_to_new_distance': record['min_to_new_distance'],
        'distance_reduction': record['distance_reduction'],
        'trajectory': [point.tolist() for point in env.trajectory],
        'trajectory_switch_index': env.switch_index,
        'zones': [{'center_xy': z.center_xy.tolist(), 'radius': z.radius} for z in env.zones],
        'active_goal_radius': env._active_goal_radius(),
        'final_goal_distance': float(info.get('goal_distance', 0.0)),
        'switch_alignment_reward_mean': record['switch_alignment_reward_mean'],
        'ceiling_penalty_mean': record['ceiling_penalty_mean'],
        'raw_goal': record['raw_goal'],
        'switch_success': record['switch_success'],
        'fallback_count': record['fallback_count'],
    }


def evaluate_target_switch_baselines(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    console_log: Path | None = None,
) -> dict[str, Any]:
    del console_log
    cfg = ExperimentConfig()
    set_global_seed(args.seed)
    target_switch_cfg = target_switch_config_for_level(args.target_switch_level)
    env = TargetSwitchTrajectoryEnv(cfg.scenario, cfg.rewards, target_switch_cfg, seed=args.seed)
    policy = build_baseline_policy(args.policy, env)

    records: list[dict[str, Any]] = []
    best_window: dict[str, Any] | None = None
    fallback_errors: list[dict[str, Any]] = []
    saved_episode_json_count = 0
    records_path = output_dir / 'records.json'
    best_window_path = output_dir / 'best_window.json'
    fallback_errors_path = output_dir / 'fallback_errors.json'
    episodes_dir = ensure_dir(output_dir / 'episodes') if args.save_episode_json else None

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        fallback_count = 0
        episode_return = 0.0
        done = False
        step_idx = 0
        while not done:
            step_idx += 1
            action, used_fallback, fallback_error = _safe_policy_action(
                policy,
                obs,
                env,
                allow_fallback=args.allow_fallback,
                episode=episode + 1,
                step=step_idx,
            )
            fallback_count += int(used_fallback)
            if fallback_error is not None:
                fallback_errors.append(fallback_error)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = bool(terminated or truncated)

        switch_summary = env.last_episode_summary or {}
        final_to_new_distance = switch_summary.get('final_to_new_distance')
        raw_goal = str(info.get('outcome', 'unknown')) == 'goal'
        record = {
            'episode': episode + 1,
            'total_steps': len(env.trajectory) - 1,
            'return': float(episode_return),
            'length': len(env.trajectory) - 1,
            'outcome': str(info.get('outcome', 'unknown')),
            'switch_step': switch_summary.get('switch_step', env.switch_step),
            'post_switch_steps': int(switch_summary.get('post_switch_steps') or 0),
            'pre_switch_done': bool(switch_summary.get('pre_switch_done', env.pre_switch_done)),
            'boundary_reason': switch_summary.get('boundary_reason'),
            'old_goal': switch_summary.get('old_goal') or (env.old_goal.tolist() if env.old_goal is not None else None),
            'new_goal': switch_summary.get('new_goal') or (env.new_goal.tolist() if env.new_goal is not None else None),
            'switch_position': switch_summary.get('switch_position'),
            'final_to_new_distance': final_to_new_distance,
            'min_to_new_distance': switch_summary.get('min_to_new_distance'),
            'distance_reduction': switch_summary.get('distance_reduction'),
            'switch_alignment_reward_mean': float(switch_summary.get('switch_alignment_reward_mean', 0.0)),
            'ceiling_penalty_mean': float(switch_summary.get('ceiling_penalty_mean', 0.0)),
            'fallback_count': fallback_count,
            'switched': bool(switch_summary.get('switched', env.switched)),
            'raw_goal': raw_goal,
            'active_goal_after_episode': env.goal.tolist(),
            'active_goal_matches_new_goal': (
                False
                if env.new_goal is None
                else bool(np.allclose(np.asarray(env.goal), np.asarray(env.new_goal)))
            ),
            'episode_json_path': None,
        }
        record['switch_success'] = _episode_switch_success(record)

        episode_payload = _build_episode_payload(
            args=args,
            env=env,
            info=info,
            record=record,
            target_switch_cfg=target_switch_cfg,
        )
        should_save_episode_json = False
        if args.save_episode_json and episodes_dir is not None and saved_episode_json_count < args.max_saved_episodes:
            should_save_episode_json = True
            if args.save_failures_only and record['switch_success']:
                should_save_episode_json = False
        if should_save_episode_json:
            episode_json_path = episodes_dir / f"episode_{record['episode']:04d}_{record['outcome']}.json"
            save_json(episode_json_path, episode_payload)
            record['episode_json_path'] = str(episode_json_path)
            saved_episode_json_count += 1

        records.append(record)
        print(
            f"[{args.policy.upper()} {args.target_switch_level}] episode={record['episode']} "
            f"return={record['return']:.2f} length={record['length']} outcome={record['outcome']} "
            f"switch_success={record['switch_success']} pre_switch_done={record['pre_switch_done']} "
            f"final_to_new_distance={record['final_to_new_distance']}"
        )

        if args.summary_every_episodes > 0 and len(records) % args.summary_every_episodes == 0:
            window = _target_switch_window_stats(records[-args.summary_every_episodes :])
            print(
                f"[{args.policy.upper()} {args.target_switch_level}] window="
                f"{window['episode_start']}-{window['episode_end']} raw_goal_rate={window['raw_goal_rate']:.3f} "
                f"switch_success_rate={window['switch_success_rate']:.3f} "
                f"goal={window['raw_goal_count']} switch_success={window['switch_success_count']} "
                f"collision={window['collision_count']} "
                f"boundary={window['boundary_count']} ground={window['ground_count']}"
            )
            if _is_better_window(window, best_window):
                best_window = dict(window)
                save_json(best_window_path, best_window)
            if args.save_snapshots:
                window_idx = len(records) // args.summary_every_episodes
                if window_idx % max(args.snapshot_every_window, 1) == 0:
                    snapshot_path = output_dir / f"ep{record['episode']:04d}_{record['outcome']}.png"
                    _plot_episode(snapshot_path, episode_payload, cfg)

    save_json(records_path, records)
    if fallback_errors:
        save_json(fallback_errors_path, fallback_errors)
    aggregate = _records_summary(records)
    summary = {
        'policy': args.policy,
        'target_switch_level': args.target_switch_level,
        'episodes': args.episodes,
        'raw_goal_count': aggregate['raw_goal_count'],
        'raw_goal_rate': aggregate['raw_goal_rate'],
        'switch_success_count': aggregate['switch_success_count'],
        'switch_success_rate': aggregate['switch_success_rate'],
        'success_count': aggregate['success_count'],
        'success_rate': aggregate['success_rate'],
        'outcomes': aggregate['outcomes'],
        'best_window': best_window,
        'best_goal_rate': None if best_window is None else best_window.get('switch_success_rate'),
        'avg_return': aggregate['avg_return'],
        'avg_length': aggregate['avg_length'],
        'avg_final_to_new_distance': aggregate['avg_final_to_new_distance'],
        'avg_post_switch_steps': aggregate['avg_post_switch_steps'],
        'avg_switch_alignment_reward': aggregate['avg_switch_alignment_reward'],
        'avg_ceiling_penalty': aggregate['avg_ceiling_penalty'],
        'pre_switch_done_count': aggregate['pre_switch_done_count'],
        'fallback_count': aggregate['fallback_count'],
        'fallback_error_count': len(fallback_errors),
        'fallback_errors_path': str(fallback_errors_path) if fallback_errors else None,
        'saved_episode_json_count': saved_episode_json_count,
        'episodes_dir': None if episodes_dir is None else str(episodes_dir),
        'records_path': str(records_path),
        'best_window_path': str(best_window_path) if best_window is not None else None,
        'output_dir': str(output_dir),
    }
    save_json(output_dir / 'summary.json', summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = _build_output_dir(args)
    console_log = output_dir / 'console.log'
    save_json(output_dir / 'args.json', _jsonable_args(args))
    with console_log.open('w', encoding='utf-8') as log_file:
        stdout_tee = _TeeStream(sys.stdout, log_file)
        stderr_tee = _TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f'[target_switch_baselines] output_dir={output_dir}')
            summary = evaluate_target_switch_baselines(args, output_dir=output_dir, console_log=console_log)
            print(summary)


if __name__ == '__main__':
    main()
