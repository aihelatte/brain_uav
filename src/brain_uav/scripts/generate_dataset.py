"""Generate behavior cloning dataset from baseline planners.

运行这个脚本后，会在 `data/` 目录下生成 `.npz` 数据集：
- observations: 状态
- actions: 基线动作
- planner_tags: 这条样本来自哪种基线

这里会优先只保留成功到达 goal 的轨迹，避免把撞墙/超时的坏轨迹教给 BC。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..baselines import AStarPlanner, ArtificialPotentialFieldPlanner, HeuristicPlanner
from ..config import ExperimentConfig
from ..envs import StaticNoFlyTrajectoryEnv
from ..utils.io import ensure_parent
from ..utils.seeding import set_global_seed


def collect_rollout(planner, env: StaticNoFlyTrajectoryEnv, max_steps: int | None = None):
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
    parser.add_argument('--output', type=Path, default=Path('data/bc_dataset.npz'))
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    set_global_seed(args.seed)
    env = StaticNoFlyTrajectoryEnv(cfg.scenario, cfg.rewards, seed=args.seed)
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
            # 兜底：如果全都失败，保留第一条失败轨迹，避免数据集为空导致脚本崩溃。
            fallback_samples = [(obs, action, planner.__class__.__name__) for obs, action in rollout]
        print(
            f"[Dataset] episode {episode + 1}/{args.episodes} planner={planner.__class__.__name__} "
            f"outcome={outcome} kept_samples={len(observations)}"
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
    )
    print(f'Saved dataset with {len(observations)} samples from {success_count} successful episodes to {target}')


if __name__ == '__main__':
    main()
