"""Generate behavior cloning dataset from baseline planners.

这里会优先只保留成功到达 goal 的轨迹，避免把撞墙/超时的坏轨迹教给 BC。
默认面向课程学习的第一层 easy 生成 BC 数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..baselines import AStarPlanner, ArtificialPotentialFieldPlanner, HeuristicPlanner
from ..config import ExperimentConfig
from ..curriculum import describe_curriculum_mix, parse_curriculum_mix
from ..scripts.common import make_env
from ..utils.io import ensure_parent
from ..utils.seeding import set_global_seed


DATASET_VERSION = 'v5'


def collect_rollout(planner, env, max_steps: int | None = None):
    """Run one planner episode and return samples plus final outcome."""

    obs, _ = env.reset()
    steps = max_steps or env.scenario.max_steps
    samples = []
    outcome = 'timeout'
    for _ in range(steps):
        action = planner.act(obs)
        samples.append((obs.copy(), action.copy()))
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            outcome = info['outcome']
            break
    return samples, outcome


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate behavior cloning dataset.')
    parser.add_argument('--output', type=Path, default=Path('data/bc_dataset_easy_v5.npz'))
    parser.add_argument('--episodes', type=int, default=180)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--curriculum-level', choices=['easy', 'medium', 'hard'], default='easy')
    parser.add_argument('--curriculum-mix', type=str, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    curriculum_mix = parse_curriculum_mix(args.curriculum_mix, fallback_level=args.curriculum_level)
    set_global_seed(args.seed)
    env = make_env(cfg, seed=args.seed, curriculum_level=args.curriculum_level, curriculum_mix=curriculum_mix)
    planners = [HeuristicPlanner(env), ArtificialPotentialFieldPlanner(env), AStarPlanner(env)]
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    planner_tags: list[str] = []
    success_count = 0
    fallback_samples: list[tuple[np.ndarray, np.ndarray, str]] = []
    for episode in range(args.episodes):
        planner = planners[episode % len(planners)]
        rollout, outcome = collect_rollout(planner, env)
        if outcome == 'goal':
            observations.extend(obs for obs, _ in rollout)
            actions.extend(action for _, action in rollout)
            planner_tags.extend([planner.__class__.__name__] * len(rollout))
            success_count += 1
        elif not fallback_samples:
            fallback_samples = [(obs, action, planner.__class__.__name__) for obs, action in rollout]
        print(
            f"[Dataset:{DATASET_VERSION}] episode {episode + 1}/{args.episodes} planner={planner.__class__.__name__} "
            f"outcome={outcome} level={args.curriculum_level} mix={describe_curriculum_mix(curriculum_mix)} "
            f"kept_samples={len(observations)}"
        )
    if not observations and fallback_samples:
        observations = [item[0] for item in fallback_samples]
        actions = [item[1] for item in fallback_samples]
        planner_tags = [item[2] for item in fallback_samples]
        print('[Dataset] warning: no successful trajectories found, using one fallback rollout to avoid empty dataset')
    if not observations:
        raise RuntimeError('Dataset generation produced zero samples. Please increase episodes or improve baselines.')
    target = ensure_parent(args.output)
    np.savez_compressed(
        target,
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        planner_tags=np.array(planner_tags),
        dataset_version=np.array(DATASET_VERSION),
        curriculum_level=np.array(args.curriculum_level),
        curriculum_mix=np.array(json.dumps(curriculum_mix, ensure_ascii=False)),
        config_json=np.array(json.dumps(cfg.to_dict(), ensure_ascii=False)),
    )
    print(
        f'Saved dataset {DATASET_VERSION} with {len(observations)} samples from '
        f'{success_count} successful episodes to {target}'
    )


if __name__ == '__main__':
    main()
