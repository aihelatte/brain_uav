"""Closed-loop rollout evaluation for behavior-cloned actors."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
import torch

from ..config import ExperimentConfig
from ..scripts.common import make_actor, make_env
from ..utils.io import load_checkpoint, save_json
from ..utils.seeding import set_global_seed


OUTCOME_KEYS = ['goal', 'timeout', 'boundary', 'ground', 'collision']


def _safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean_vector(values: list[list[float] | None]) -> list[float] | None:
    valid = [item for item in values if item is not None]
    if not valid:
        return None
    arr = np.asarray(valid, dtype=float)
    return arr.mean(axis=0).astype(float).tolist()


def evaluate_bc_rollout(
    model: str,
    checkpoint_path: Path,
    episodes: int,
    seed: int,
    curriculum_level: str,
) -> dict:
    cfg = ExperimentConfig()
    set_global_seed(seed)
    env = make_env(cfg, seed=seed, curriculum_level=curriculum_level)
    obs, _ = env.reset(seed=seed)
    actor = make_actor(cfg, model, obs.shape[0], env.action_space.shape[0])
    checkpoint = load_checkpoint(checkpoint_path)
    actor.load_state_dict(checkpoint['state_dict'])
    device = torch.device(cfg.training.device)
    actor.to(device)
    actor.eval()

    records: list[dict] = []
    outcome_counts: dict[str, int] = {key: 0 for key in OUTCOME_KEYS}
    returns: list[float] = []
    lengths: list[int] = []
    min_goal_distances: list[float] = []
    final_goal_distances: list[float] = []
    steps_at_min_goal_distance: list[float] = []
    near_goal_step_counts: list[float] = []
    mean_abs_actions: list[list[float] | None] = []
    near_goal_mean_abs_actions: list[list[float] | None] = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        scenario = env.export_scenario()
        episode_return = 0.0
        length = 0
        outcome = 'timeout'
        min_goal_distance = float(info.get('goal_distance', env._goal_distance(env.state[:3])))
        final_goal_distance = min_goal_distance
        step_at_min_goal_distance = 0
        position_at_min_goal_distance = env.state[:3].copy()
        actions_abs: list[np.ndarray] = []
        near_goal_actions_abs: list[np.ndarray] = []

        for _ in range(env.scenario.max_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(obs_tensor).squeeze(0).detach().cpu().numpy()
            actions_abs.append(np.abs(action).astype(float))
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            length += 1
            final_goal_distance = float(info.get('goal_distance', env._goal_distance(env.state[:3])))
            if final_goal_distance <= 50.0:
                near_goal_actions_abs.append(np.abs(action).astype(float))
            if final_goal_distance < min_goal_distance:
                min_goal_distance = final_goal_distance
                step_at_min_goal_distance = length
                position_at_min_goal_distance = env.state[:3].copy()
            if terminated or truncated:
                outcome = str(info.get('outcome', 'unknown'))
                break

        if outcome in outcome_counts:
            outcome_counts[outcome] += 1
        else:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        returns.append(episode_return)
        lengths.append(length)
        min_goal_distances.append(min_goal_distance)
        final_goal_distances.append(final_goal_distance)
        mean_abs_action = (
            np.asarray(actions_abs, dtype=float).mean(axis=0).astype(float).tolist()
            if actions_abs
            else None
        )
        near_goal_mean_abs_action = (
            np.asarray(near_goal_actions_abs, dtype=float).mean(axis=0).astype(float).tolist()
            if near_goal_actions_abs
            else None
        )
        steps_at_min_goal_distance.append(float(step_at_min_goal_distance))
        near_goal_step_counts.append(float(len(near_goal_actions_abs)))
        mean_abs_actions.append(mean_abs_action)
        near_goal_mean_abs_actions.append(near_goal_mean_abs_action)
        records.append(
            {
                'episode': episode + 1,
                'outcome': outcome,
                'length': int(length),
                'return': float(episode_return),
                'min_goal_distance': float(min_goal_distance),
                'final_goal_distance': float(final_goal_distance),
                'start_goal_distance_km': scenario.get('start_goal_distance_km'),
                'step_at_min_goal_distance': int(step_at_min_goal_distance),
                'position_at_min_goal_distance': position_at_min_goal_distance.astype(float).tolist(),
                'final_position': env.state[:3].astype(float).tolist(),
                'mean_abs_action': mean_abs_action,
                'near_goal_mean_abs_action': near_goal_mean_abs_action,
                'near_goal_step_count': int(len(near_goal_actions_abs)),
            }
        )

    goal_count = int(outcome_counts.get('goal', 0))
    summary = {
        'model': model,
        'checkpoint': str(checkpoint_path),
        'episodes': int(episodes),
        'seed': int(seed),
        'curriculum_level': curriculum_level,
        'goal_count': goal_count,
        'goal_rate': float(goal_count / episodes) if episodes > 0 else 0.0,
        'timeout_count': int(outcome_counts.get('timeout', 0)),
        'boundary_count': int(outcome_counts.get('boundary', 0)),
        'ground_count': int(outcome_counts.get('ground', 0)),
        'collision_count': int(outcome_counts.get('collision', 0)),
        'outcome_counts': outcome_counts,
        'avg_length': _safe_mean([float(v) for v in lengths]),
        'avg_return': _safe_mean(returns),
        'avg_min_goal_distance': _safe_mean(min_goal_distances),
        'median_min_goal_distance': _safe_median(min_goal_distances),
        'min_min_goal_distance': float(min(min_goal_distances)) if min_goal_distances else 0.0,
        'avg_final_goal_distance': _safe_mean(final_goal_distances),
        'median_final_goal_distance': _safe_median(final_goal_distances),
        'avg_step_at_min_goal_distance': _safe_mean(steps_at_min_goal_distance),
        'avg_near_goal_step_count': _safe_mean(near_goal_step_counts),
        'avg_mean_abs_action': _mean_vector(mean_abs_actions),
        'avg_near_goal_mean_abs_action': _mean_vector(near_goal_mean_abs_actions),
        'records': records,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a BC actor with closed-loop environment rollouts.')
    parser.add_argument('--model', choices=['snn', 'ann'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument(
        '--curriculum-level',
        choices=['easy', 'easy_two_zone', 'medium', 'hard'],
        default='easy',
    )
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    summary = evaluate_bc_rollout(
        model=args.model,
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        curriculum_level=args.curriculum_level,
    )
    print(
        f"[BC Rollout] model={args.model} episodes={args.episodes} "
        f"goal_rate={summary['goal_rate']:.3f} goals={summary['goal_count']} "
        f"timeouts={summary['timeout_count']} avg_len={summary['avg_length']:.1f} "
        f"avg_min_goal_distance={summary['avg_min_goal_distance']:.2f}"
    )
    if args.output is not None:
        save_json(args.output, summary)
        print(f"Saved BC rollout summary to {args.output}")


if __name__ == '__main__':
    main()
