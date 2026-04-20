"""Generate behavior cloning dataset from baseline planners.

这里会优先只保留成功到达 goal 的轨迹，避免把撞墙/超时的坏轨迹教给 BC。
默认面向课程学习的第一层 easy 生成 BC 数据。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..baselines import AStarPlanner, ArtificialPotentialFieldPlanner, HeuristicPlanner
from ..config import ExperimentConfig
from ..curriculum import describe_curriculum_mix, parse_curriculum_mix
from ..scripts.common import make_env
from ..utils.io import ensure_parent
from ..utils.seeding import set_global_seed


DATASET_VERSION = 'v8_restore'
TERMINAL_AUG_THRESHOLDS_KM = [100.0, 50.0, 20.0]
TERMINAL_AUG_EXTRA_REPEATS = {
    100.0: 2,
    50.0: 4,
    20.0: 8,
}
TERMINAL_DISTANCE_BIN_LABELS = ['<=5km', '5-10km', '10-20km', '20-50km', '50-100km', '>100km']


def build_planners(env, planner_names: str):
    """Build selected baseline planners for dataset generation."""

    registry = {
        'heuristic': HeuristicPlanner,
        'apf': ArtificialPotentialFieldPlanner,
        'astar': AStarPlanner,
    }
    planners = []
    unknown = []
    for raw_name in planner_names.split(','):
        key = raw_name.strip().lower()
        if not key:
            continue
        planner_cls = registry.get(key)
        if planner_cls is None:
            unknown.append(key)
            continue
        planners.append(planner_cls(env))
    if unknown:
        allowed = ', '.join(sorted(registry))
        raise ValueError(f"Unknown planner(s): {', '.join(unknown)}. Allowed planners: {allowed}.")
    if not planners:
        raise ValueError('At least one planner must be selected.')
    return planners


def _wrap_angle(value: float) -> float:
    return ((value + math.pi) % (2 * math.pi)) - math.pi


def _terminal_homing_action(env) -> np.ndarray:
    delta = env.goal - env.state[:3]
    horizontal_distance = float(np.linalg.norm(delta[:2]))
    psi_target = math.atan2(float(delta[1]), float(delta[0]))
    gamma_target = math.atan2(float(delta[2]), horizontal_distance)
    delta_gamma = gamma_target - float(env.state[3])
    delta_psi = _wrap_angle(psi_target - float(env.state[4]))
    return np.array(
        [
            np.clip(delta_gamma, -env.scenario.delta_gamma_max, env.scenario.delta_gamma_max),
            np.clip(delta_psi, -env.scenario.delta_psi_max, env.scenario.delta_psi_max),
        ],
        dtype=np.float32,
    )


def _terminal_homing_alpha(goal_distance: float, radius: float, full_radius: float) -> float:
    if goal_distance > radius:
        return 0.0
    if goal_distance <= full_radius:
        return 1.0
    return float(np.clip((radius - goal_distance) / max(radius - full_radius, 1e-6), 0.0, 1.0))


def collect_rollout(
    planner,
    env,
    max_steps: int | None = None,
    terminal_homing_enabled: bool = False,
    terminal_homing_radius: float = 80.0,
    terminal_homing_full_radius: float = 20.0,
):
    """Run one planner episode and return samples plus final outcome."""

    obs, _ = env.reset()
    steps = max_steps or env.scenario.max_steps
    samples = []
    outcome = 'timeout'
    homing_step_count = 0
    homing_alpha_sum = 0.0
    for _ in range(steps):
        goal_distance = float(env._goal_distance(env.state[:3]))
        planner_action = planner.act(obs)
        alpha = 0.0
        if terminal_homing_enabled:
            alpha = _terminal_homing_alpha(goal_distance, terminal_homing_radius, terminal_homing_full_radius)
        if alpha > 0.0:
            homing_action = _terminal_homing_action(env)
            action = (1.0 - alpha) * planner_action + alpha * homing_action
            action = np.asarray(action, dtype=np.float32).clip(env.action_space.low, env.action_space.high)
            homing_step_count += 1
            homing_alpha_sum += alpha
        else:
            action = planner_action
        samples.append((obs.copy(), action.copy(), goal_distance))
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            outcome = info['outcome']
            break
    return samples, outcome, homing_step_count, homing_alpha_sum


def terminal_extra_repeats(goal_distance_km: float) -> int:
    """Return extra terminal-capture repeats for one successful rollout sample."""

    if goal_distance_km <= 20.0:
        return TERMINAL_AUG_EXTRA_REPEATS[20.0]
    if goal_distance_km <= 50.0:
        return TERMINAL_AUG_EXTRA_REPEATS[50.0]
    if goal_distance_km <= 100.0:
        return TERMINAL_AUG_EXTRA_REPEATS[100.0]
    return 0


def compute_terminal_diagnostics(
    observations: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute terminal distance bin action diagnostics from final dataset samples."""

    rel_goal = observations[:, 5:8]
    distances = np.linalg.norm(rel_goal, axis=1)
    abs_actions = np.abs(actions)
    masks = [
        distances <= 5.0,
        (distances > 5.0) & (distances <= 10.0),
        (distances > 10.0) & (distances <= 20.0),
        (distances > 20.0) & (distances <= 50.0),
        (distances > 50.0) & (distances <= 100.0),
        distances > 100.0,
    ]
    counts = np.zeros(len(masks), dtype=np.int64)
    action_abs_mean = np.full((len(masks), actions.shape[1]), np.nan, dtype=np.float32)
    action_abs_std = np.full((len(masks), actions.shape[1]), np.nan, dtype=np.float32)
    for idx, mask in enumerate(masks):
        counts[idx] = int(mask.sum())
        if counts[idx] > 0:
            selected = abs_actions[mask]
            action_abs_mean[idx] = selected.mean(axis=0).astype(np.float32)
            action_abs_std[idx] = selected.std(axis=0).astype(np.float32)
    return counts, action_abs_mean, action_abs_std


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate behavior cloning dataset.')
    parser.add_argument('--output', type=Path, default=Path('data/bc_dataset_easy_v8_restore.npz'))
    parser.add_argument('--episodes', type=int, default=180)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--curriculum-level', choices=['easy', 'easy_two_zone', 'medium', 'hard'], default='easy')
    parser.add_argument('--curriculum-mix', type=str, default=None)
    parser.add_argument(
        '--planners',
        type=str,
        default='heuristic,apf',
        help='Comma-separated planners used for dataset generation: heuristic,apf,astar',
    )
    terminal_group = parser.add_mutually_exclusive_group()
    terminal_group.add_argument(
        '--terminal-augment',
        dest='terminal_augment',
        action='store_true',
        help='Enable terminal capture sample augmentation for successful rollouts.',
    )
    terminal_group.add_argument(
        '--no-terminal-augment',
        dest='terminal_augment',
        action='store_false',
        help='Disable terminal capture sample augmentation.',
    )
    parser.set_defaults(terminal_augment=True)
    homing_group = parser.add_mutually_exclusive_group()
    homing_group.add_argument(
        '--terminal-homing',
        dest='terminal_homing',
        action='store_true',
        help='Enable terminal homing action blending for expert dataset generation.',
    )
    homing_group.add_argument(
        '--no-terminal-homing',
        dest='terminal_homing',
        action='store_false',
        help='Disable terminal homing action blending.',
    )
    parser.add_argument('--terminal-homing-radius', type=float, default=80.0)
    parser.add_argument('--terminal-homing-full-radius', type=float, default=20.0)
    parser.set_defaults(terminal_homing=False)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    curriculum_mix = parse_curriculum_mix(args.curriculum_mix, fallback_level=args.curriculum_level)
    set_global_seed(args.seed)
    env = make_env(cfg, seed=args.seed, curriculum_level=args.curriculum_level, curriculum_mix=curriculum_mix)
    planners = build_planners(env, args.planners)
    print(f"[Dataset:{DATASET_VERSION}] enabled planners: {', '.join(p.__class__.__name__ for p in planners)}")
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    planner_tags: list[str] = []
    success_count = 0
    raw_success_samples = 0
    augmented_samples = 0
    terminal_augmented_sample_count = 0
    terminal_homing_step_count = 0
    terminal_homing_alpha_sum = 0.0
    fallback_samples: list[tuple[np.ndarray, np.ndarray, str]] = []
    for episode in range(args.episodes):
        planner = planners[episode % len(planners)]
        rollout, outcome, homing_steps, homing_alpha_sum = collect_rollout(
            planner,
            env,
            terminal_homing_enabled=args.terminal_homing,
            terminal_homing_radius=args.terminal_homing_radius,
            terminal_homing_full_radius=args.terminal_homing_full_radius,
        )
        terminal_homing_step_count += homing_steps
        terminal_homing_alpha_sum += homing_alpha_sum
        if outcome == 'goal':
            raw_success_samples += len(rollout)
            for obs, action, goal_distance in rollout:
                repeat_count = 1
                if args.terminal_augment:
                    repeat_count += terminal_extra_repeats(goal_distance)
                for _ in range(repeat_count):
                    observations.append(obs)
                    actions.append(action)
                    planner_tags.append(planner.__class__.__name__)
                augmented_samples += repeat_count
                terminal_augmented_sample_count += repeat_count - 1
            success_count += 1
        elif not fallback_samples:
            fallback_samples = [(obs, action, planner.__class__.__name__) for obs, action, _ in rollout]
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
    observations_array = np.stack(observations).astype(np.float32)
    actions_array = np.stack(actions).astype(np.float32)
    bin_counts, bin_action_abs_mean, bin_action_abs_std = compute_terminal_diagnostics(
        observations_array,
        actions_array,
    )
    np.savez_compressed(
        target,
        observations=observations_array,
        actions=actions_array,
        planner_tags=np.array(planner_tags),
        dataset_version=np.array(DATASET_VERSION),
        curriculum_level=np.array(args.curriculum_level),
        curriculum_mix=np.array(json.dumps(curriculum_mix, ensure_ascii=False)),
        config_json=np.array(json.dumps(cfg.to_dict(), ensure_ascii=False)),
        terminal_augmentation_enabled=np.array(bool(args.terminal_augment)),
        terminal_aug_thresholds_km=np.array(TERMINAL_AUG_THRESHOLDS_KM, dtype=np.float32),
        terminal_aug_extra_repeats=np.array(
            [
                TERMINAL_AUG_EXTRA_REPEATS[100.0],
                TERMINAL_AUG_EXTRA_REPEATS[50.0],
                TERMINAL_AUG_EXTRA_REPEATS[20.0],
            ],
            dtype=np.int32,
        ),
        raw_success_samples=np.array(raw_success_samples, dtype=np.int64),
        augmented_samples=np.array(augmented_samples, dtype=np.int64),
        terminal_augmented_sample_count=np.array(terminal_augmented_sample_count, dtype=np.int64),
        terminal_homing_enabled=np.array(bool(args.terminal_homing)),
        terminal_homing_radius=np.array(float(args.terminal_homing_radius), dtype=np.float32),
        terminal_homing_full_radius=np.array(float(args.terminal_homing_full_radius), dtype=np.float32),
        terminal_homing_step_count=np.array(terminal_homing_step_count, dtype=np.int64),
        terminal_homing_mean_alpha=np.array(
            terminal_homing_alpha_sum / max(terminal_homing_step_count, 1),
            dtype=np.float32,
        ),
        terminal_distance_bin_labels=np.array(TERMINAL_DISTANCE_BIN_LABELS),
        terminal_distance_bin_counts=bin_counts,
        terminal_distance_bin_action_abs_mean=bin_action_abs_mean,
        terminal_distance_bin_action_abs_std=bin_action_abs_std,
    )
    print(
        f'Saved dataset {DATASET_VERSION} with {len(observations)} samples from '
        f'{success_count} successful episodes to {target}'
    )
    print(
        f'[Dataset:{DATASET_VERSION}] raw_success_samples={raw_success_samples} '
        f'augmented_samples={augmented_samples} terminal_augmented_sample_count={terminal_augmented_sample_count}'
    )
    print(
        f'[Dataset:{DATASET_VERSION}] terminal_homing_enabled={args.terminal_homing} '
        f'terminal_homing_step_count={terminal_homing_step_count} '
        f'terminal_homing_mean_alpha={terminal_homing_alpha_sum / max(terminal_homing_step_count, 1):.4f}'
    )
    print(f'[Dataset:{DATASET_VERSION}] terminal distance diagnostics:')
    for label, count, mean in zip(TERMINAL_DISTANCE_BIN_LABELS, bin_counts, bin_action_abs_mean):
        mean_text = 'nan' if count == 0 else f'[{mean[0]:.6f}, {mean[1]:.6f}]'
        print(f'  {label}: count={int(count)} action_abs_mean={mean_text}')


if __name__ == '__main__':
    main()
