"""Evaluate a trained checkpoint on benchmark, curriculum, or random scenarios."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..curriculum import describe_curriculum_mix, parse_curriculum_mix
from ..scenarios import build_benchmark_scenarios
from ..scripts.common import make_actor, make_env
from ..utils.io import load_checkpoint, save_json
from ..utils.seeding import set_global_seed


def evaluate_policy(
    checkpoint: Path,
    model: str,
    episodes: int,
    seed: int,
    evaluation_mode: str,
    curriculum_level: str | None = None,
    curriculum_mix: dict[str, float] | None = None,
) -> dict:
    """Run evaluation and return a rich result dict."""

    cfg = ExperimentConfig()
    set_global_seed(seed)
    env = make_env(
        cfg,
        seed=seed,
        scenario_suite='benchmark' if evaluation_mode == 'benchmark' else None,
        curriculum_level=curriculum_level if evaluation_mode == 'curriculum' else None,
        curriculum_mix=curriculum_mix if evaluation_mode == 'curriculum' else None,
    )
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
    named_scenarios = build_benchmark_scenarios() if evaluation_mode == 'benchmark' else []
    for ep in range(episodes):
        if evaluation_mode == 'benchmark':
            scenario = named_scenarios[ep % len(named_scenarios)]
            obs, _ = env.reset(options={'scenario': scenario.scenario})
            scenario_name = scenario.name
        else:
            obs, _ = env.reset(seed=seed + ep)
            if evaluation_mode == 'curriculum':
                scenario_name = f"{env.last_curriculum_level}_{ep:03d}"
            else:
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
                'curriculum_level': info.get('curriculum_level'),
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
        'evaluation_mode': evaluation_mode,
        'curriculum_level': curriculum_level,
        'curriculum_mix': curriculum_mix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a trained policy.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', choices=['snn', 'ann'], required=True)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--evaluation-mode', choices=['benchmark', 'curriculum', 'random'], default='benchmark')
    parser.add_argument('--curriculum-level', choices=['easy', 'medium', 'hard'], default=None)
    parser.add_argument('--curriculum-mix', type=str, default=None)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    curriculum_mix = None
    if args.evaluation_mode == 'curriculum':
        if args.curriculum_level is None:
            raise ValueError('--curriculum-level is required when --evaluation-mode curriculum')
        curriculum_mix = parse_curriculum_mix(args.curriculum_mix, fallback_level=args.curriculum_level)
    results = evaluate_policy(
        args.checkpoint,
        args.model,
        args.episodes,
        args.seed,
        args.evaluation_mode,
        curriculum_level=args.curriculum_level,
        curriculum_mix=curriculum_mix,
    )
    if args.output:
        save_json(args.output, results)
    if args.evaluation_mode == 'curriculum':
        print(f"Curriculum evaluation: level={args.curriculum_level}, mix={describe_curriculum_mix(curriculum_mix)}")
    print(results)


if __name__ == '__main__':
    main()
