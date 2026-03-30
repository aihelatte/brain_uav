from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..scenarios import build_benchmark_scenarios
from ..scripts.common import make_actor, make_env
from ..utils.io import load_checkpoint, save_json
from ..utils.seeding import set_global_seed


def evaluate_policy(checkpoint: Path, model: str, episodes: int, seed: int, scenario_suite: str) -> dict:
    cfg = ExperimentConfig()
    set_global_seed(seed)
    env = make_env(cfg, seed=seed, scenario_suite=scenario_suite if scenario_suite == 'benchmark' else None)
    obs, _ = env.reset(seed=seed)
    actor = make_actor(cfg, model, obs.shape[0], env.action_space.shape[0])
    actor.load_state_dict(load_checkpoint(checkpoint)['state_dict'])
    actor.eval()

    successes = 0
    collisions = 0
    step_counts = []
    episode_times = []
    per_inference = []
    outcomes = {}
    records = []
    named_scenarios = build_benchmark_scenarios() if scenario_suite == 'benchmark' else []
    for ep in range(episodes):
        if scenario_suite == 'benchmark':
            scenario = named_scenarios[ep % len(named_scenarios)]
            obs, _ = env.reset(options={'scenario': scenario.scenario})
            scenario_name = scenario.name
        else:
            obs, _ = env.reset(seed=seed + ep)
            scenario_name = f'random_{ep:03d}'
        done = False
        steps = 0
        inference_times = []
        while not done:
            step_start = time.perf_counter()
            with torch.no_grad():
                action = actor(torch.tensor(obs[None, :], dtype=torch.float32)).cpu().numpy()[0]
            infer_t = time.perf_counter() - step_start
            inference_times.append(infer_t)
            per_inference.append(infer_t)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        episode_time = sum(inference_times)
        episode_times.append(episode_time)
        step_counts.append(steps)
        outcome = info['outcome']
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == 'goal':
            successes += 1
        if outcome == 'collision':
            collisions += 1
        records.append(
            {
                'scenario': scenario_name,
                'outcome': outcome,
                'steps': steps,
                'goal_distance': info['goal_distance'],
                'avg_inference_time_ms': 1000.0 * statistics.mean(inference_times),
                'max_inference_time_ms': 1000.0 * max(inference_times),
            }
        )
    return {
        'model': model,
        'episodes': episodes,
        'success_rate': successes / episodes,
        'collision_rate': collisions / episodes,
        'avg_steps': statistics.mean(step_counts),
        'avg_episode_time_s': statistics.mean(episode_times),
        'avg_inference_time_ms': 1000.0 * statistics.mean(per_inference),
        'max_inference_time_ms': 1000.0 * max(per_inference),
        'outcomes': outcomes,
        'records': records,
        'scenario_suite': scenario_suite,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a trained policy.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', choices=['snn', 'ann'], required=True)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--scenario-suite', choices=['benchmark', 'random'], default='benchmark')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    results = evaluate_policy(args.checkpoint, args.model, args.episodes, args.seed, args.scenario_suite)
    if args.output:
        save_json(args.output, results)
    print(results)


if __name__ == '__main__':
    main()
