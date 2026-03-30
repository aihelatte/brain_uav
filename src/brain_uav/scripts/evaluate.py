from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..scripts.common import make_actor, make_env
from ..utils.io import load_checkpoint
from ..utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["snn", "ann"], required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    set_global_seed(args.seed)
    env = make_env(cfg, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    actor = make_actor(cfg, args.model, obs.shape[0], env.action_space.shape[0])
    actor.load_state_dict(load_checkpoint(args.checkpoint)["state_dict"])
    actor.eval()

    successes = 0
    collisions = 0
    step_counts = []
    episode_times = []
    per_inference = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            episode_start = time.perf_counter()
            action = actor(torch.tensor(obs[None, :], dtype=torch.float32)).detach().numpy()[0]
            per_inference.append(time.perf_counter() - episode_start)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        episode_times.append(sum(per_inference[-info["steps"] :]))
        step_counts.append(info["steps"])
        if info["outcome"] == "goal":
            successes += 1
        if info["outcome"] == "collision":
            collisions += 1
    print(
        {
            "success_rate": successes / args.episodes,
            "collision_rate": collisions / args.episodes,
            "avg_steps": statistics.mean(step_counts),
            "avg_episode_time_s": statistics.mean(episode_times),
            "avg_inference_time_ms": 1000.0 * statistics.mean(per_inference),
            "max_inference_time_ms": 1000.0 * max(per_inference),
        }
    )


if __name__ == "__main__":
    main()

