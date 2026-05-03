"""Evaluate a trained policy under a mid-flight target switch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import ExperimentConfig
from ..envs import TargetSwitchTrajectoryEnv
from ..scripts.common import DEVICE_CHOICES, SNN_BACKEND_CHOICES, configure_training_runtime, make_actor
from ..scripts.evaluate import _apply_checkpoint_config
from ..target_switch import TargetSwitchConfig, target_switch_config_for_level
from ..utils.io import ensure_dir, load_checkpoint, now_timestamp, save_json
from ..utils.seeding import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run target-switch rollout evaluation.')
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/target_switch_models/td3_snn_hard.pt'))
    parser.add_argument('--model', choices=['snn', 'ann'], default='snn')
    parser.add_argument('--curriculum-level', choices=['easy', 'easy_two_zone', 'medium', 'hard'], default='hard')
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--switch-step', type=int, default=400)
    parser.add_argument('--switch-mode', choices=['forward', 'lateral', 'reverse'], default='forward')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--snn-backend', choices=SNN_BACKEND_CHOICES, default='torch')
    return parser


def _plot_episode(path: Path, record: dict[str, Any], cfg: ExperimentConfig) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    traj = np.asarray(record['trajectory'], dtype=float)
    switch_idx = int(record.get('trajectory_switch_index') or 0)
    pre = traj[: switch_idx + 1] if record.get('switched') else traj
    post = traj[switch_idx:] if record.get('switched') else np.empty((0, 3))
    start = traj[0]
    old_goal = np.asarray(record['old_goal'], dtype=float)
    new_goal = np.asarray(record['new_goal'], dtype=float) if record.get('new_goal') is not None else old_goal
    switch_position = np.asarray(record['switch_position'], dtype=float) if record.get('switch_position') is not None else None

    fig, axes = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [6, 1, 1]})
    views = [
        (axes[0], (0, 1), 'Top View (X-Y)', 'x (km)', 'y (km)'),
        (axes[1], (0, 2), 'Side View (X-Z)', 'x (km)', 'z (km)'),
        (axes[2], (1, 2), 'Front View (Y-Z)', 'y (km)', 'z (km)'),
    ]
    for ax, dims, title, xlabel, ylabel in views:
        ax.plot(pre[:, dims[0]], pre[:, dims[1]], color='tab:blue', linewidth=2.0, label='pre-switch')
        if len(post):
            ax.plot(post[:, dims[0]], post[:, dims[1]], color='tab:orange', linewidth=2.0, label='post-switch')
        ax.scatter(start[dims[0]], start[dims[1]], color='tab:blue', marker='o', s=55, label='start')
        ax.scatter(old_goal[dims[0]], old_goal[dims[1]], color='gray', marker='*', s=90, label='old goal')
        ax.scatter(new_goal[dims[0]], new_goal[dims[1]], color='tab:green', marker='*', s=90, label='new goal')
        if switch_position is not None:
            ax.scatter(
                switch_position[dims[0]],
                switch_position[dims[1]],
                color='purple',
                marker='D',
                s=55,
                label='switch point',
            )
            ax.plot(
                [switch_position[dims[0]], new_goal[dims[0]]],
                [switch_position[dims[1]], new_goal[dims[1]]],
                color='tab:green',
                linestyle='--',
                alpha=0.6,
                label='switch target vector',
            )
        for zone in record['zones']:
            center = zone['center_xy']
            radius = float(zone['radius'])
            if dims == (0, 1):
                ax.add_patch(patches.Circle(center, radius, fill=False, color='tab:red', linewidth=1.4))
                ax.add_patch(
                    patches.Circle(
                        center,
                        radius + cfg.scenario.warning_distance,
                        fill=False,
                        color='tab:red',
                        linestyle='--',
                        alpha=0.4,
                    )
                )
            else:
                center_value = center[0] if dims[0] == 0 else center[1]
                xs = np.linspace(center_value - radius, center_value + radius, 100)
                zs = np.sqrt(np.maximum(radius**2 - (xs - center_value) ** 2, 0.0))
                ax.plot(xs, zs, color='tab:red', linewidth=1.2, alpha=0.7)
        if dims == (0, 1):
            ax.add_patch(
                patches.Circle(
                    (new_goal[0], new_goal[1]),
                    record['active_goal_radius'],
                    fill=False,
                    color='tab:green',
                    linestyle='--',
                    alpha=0.5,
                )
            )
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.axhline(cfg.scenario.ground_warning_height, color='tab:orange', linestyle='--', alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=8, ncol=2)
    fig.suptitle(f"Episode {record['episode']} - {record['outcome']}", fontsize=15)
    fig.text(
        0.02,
        0.015,
        (
            f"total_steps={record['total_steps']} | switch_step={record['switch_step']} | "
            f"post_switch_steps={record['post_switch_steps']} | "
            f"final_goal_distance={record['final_goal_distance']:.2f}"
        ),
        fontsize=9,
        family='monospace',
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _select_action(actor, obs: np.ndarray) -> np.ndarray:
    device = next(actor.parameters()).device
    with torch.no_grad():
        return actor(torch.tensor(obs[None, :], dtype=torch.float32, device=device)).cpu().numpy()[0]


def main() -> None:
    args = build_parser().parse_args()
    cfg = ExperimentConfig()
    payload = load_checkpoint(args.checkpoint)
    _apply_checkpoint_config(cfg, payload)
    configure_training_runtime(cfg, model_type=args.model, device=args.device, snn_backend=args.snn_backend)
    set_global_seed(args.seed)
    target_cfg = target_switch_config_for_level('target_switch_hard')
    target_cfg.base_curriculum_level = args.curriculum_level
    output_root = ensure_dir(
        args.output or Path('outputs/target_switch_models') / f'{args.curriculum_level}_{now_timestamp()}'
    )

    env = TargetSwitchTrajectoryEnv(cfg.scenario, cfg.rewards, target_cfg, seed=args.seed, switch_mode=args.switch_mode)
    obs, _ = env.reset(seed=args.seed)
    actor = make_actor(cfg, args.model, obs.shape[0], env.action_space.shape[0])
    actor.load_state_dict(payload['state_dict'])
    actor.to(cfg.training.device)
    actor.eval()

    records: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        env.switch_step = min(max(1, args.switch_step), env.scenario.max_steps - 1)
        episode_return = 0.0
        max_total_steps = env.switch_step + env.scenario.max_steps + 5
        info: dict[str, Any] = {}
        for _ in range(max_total_steps):
            action = _select_action(actor, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        summary = env.last_episode_summary or {}
        outcome = info.get('outcome', 'unknown')
        record = {
            'episode': episode,
            'seed': args.seed + episode,
            'model': args.model,
            'curriculum_level': args.curriculum_level,
            'checkpoint': str(args.checkpoint),
            'initial_scenario': env.export_scenario(),
            'old_goal': summary.get('old_goal') or (env.old_goal.tolist() if env.old_goal is not None else None),
            'new_goal': summary.get('new_goal'),
            'switch_step': env.switch_step,
            'switch_position': summary.get('switch_position'),
            'switched': bool(summary.get('switched', env.switched)),
            'pre_switch_done': bool(summary.get('pre_switch_done', env.pre_switch_done)),
            'total_steps': len(env.trajectory) - 1,
            'post_switch_steps': int(summary.get('post_switch_steps') or 0),
            'outcome': outcome,
            'return': episode_return,
            'final_state': env.state.tolist(),
            'final_goal_distance': float(info.get('goal_distance', env._goal_distance(env.state[:3]))),
            'trajectory': [point.tolist() for point in env.trajectory],
            'trajectory_switch_index': env.switch_index,
            'zones': [{'center_xy': z.center_xy.tolist(), 'radius': z.radius} for z in env.zones],
            'active_goal_radius': env._active_goal_radius(),
            **summary,
        }
        json_path = output_root / f'episode_{episode:04d}.json'
        png_path = output_root / f'episode_{episode:04d}_{outcome}.png'
        save_json(json_path, record)
        _plot_episode(png_path, record, cfg)
        records.append({**record, 'json': str(json_path), 'png': str(png_path)})

    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record['outcome']] = outcomes.get(record['outcome'], 0) + 1
    summary = {
        'episodes': args.episodes,
        'switch_mode': args.switch_mode,
        'curriculum_level': args.curriculum_level,
        'checkpoint': str(args.checkpoint),
        'model': args.model,
        'switched_count': sum(1 for r in records if r.get('switched')),
        'pre_switch_done_count': sum(1 for r in records if r.get('pre_switch_done')),
        'success_count': outcomes.get('goal', 0),
        'success_rate': outcomes.get('goal', 0) / args.episodes if args.episodes else 0.0,
        'outcomes': outcomes,
        'avg_total_steps': float(np.mean([r['total_steps'] for r in records])) if records else 0.0,
        'avg_post_switch_steps': float(np.mean([r['post_switch_steps'] for r in records])) if records else 0.0,
        'records': records,
    }
    save_json(output_root / 'summary.json', summary)
    print(f'[target_switch] saved results to {output_root}')


if __name__ == '__main__':
    main()
