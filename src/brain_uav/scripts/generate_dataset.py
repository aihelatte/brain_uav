from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..baselines import AStarPlanner, ArtificialPotentialFieldPlanner, HeuristicPlanner
from ..config import ExperimentConfig
from ..envs import StaticNoFlyTrajectoryEnv
from ..utils.io import ensure_parent
from ..utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate behavior cloning dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/bc_dataset.npz"))
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    set_global_seed(args.seed)
    env = StaticNoFlyTrajectoryEnv(cfg.scenario, cfg.rewards, seed=args.seed)
    planners = [HeuristicPlanner(env), ArtificialPotentialFieldPlanner(env), AStarPlanner(env)]
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in range(args.episodes):
        rollout = planners[episode % len(planners)].rollout()
        observations.extend(obs for obs, _ in rollout)
        actions.extend(action for _, action in rollout)
    target = ensure_parent(args.output)
    np.savez_compressed(
        target,
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
    )
    print(f"Saved dataset with {len(observations)} samples to {target}")


if __name__ == "__main__":
    main()

