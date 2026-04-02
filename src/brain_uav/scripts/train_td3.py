"""Train the TD3 policy after behavior cloning initialization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from ..config import ExperimentConfig
from ..scripts.common import make_actor, make_critics, make_env
from ..trainers import TD3Trainer
from ..utils.io import (
    build_log_paths,
    ensure_dir,
    load_checkpoint,
    now_timestamp,
    save_checkpoint,
    save_csv_rows,
    save_json,
)
from ..utils.seeding import set_global_seed


OUTCOME_KEYS = ['goal', 'timeout', 'boundary', 'ground', 'collision', 'other']
OUTCOME_COLORS = {
    'goal': 'tab:green',
    'timeout': 'tab:orange',
    'boundary': 'tab:red',
    'ground': 'tab:brown',
    'collision': 'tab:purple',
    'other': 'tab:gray',
}


def export_training_report(base_metrics_path: Path, metrics: dict) -> dict[str, str]:
    """Write AI-friendly training summary files."""

    window_rows = metrics.get('episode_window_stats', [])
    outputs: dict[str, str] = {}
    outputs['json'] = str(save_json(base_metrics_path, metrics))

    csv_path = base_metrics_path.with_name(f'{base_metrics_path.stem}_episode_windows.csv')
    save_csv_rows(csv_path, window_rows)
    outputs['csv'] = str(csv_path)

    try:
        import matplotlib.pyplot as plt

        if window_rows:
            labels = [f"{row['episode_start']}-{row['episode_end']}" for row in window_rows]
            x = list(range(len(labels)))
            tick_step = max(1, len(labels) // 18)
            tick_positions = x[::tick_step]
            tick_labels = labels[::tick_step]

            fig, axes = plt.subplots(3, 1, figsize=(20, 16))

            bottoms = [0] * len(labels)
            for key in OUTCOME_KEYS:
                values = [row.get(f'{key}_count', 0) for row in window_rows]
                axes[0].bar(x, values, bottom=bottoms, label=key, color=OUTCOME_COLORS[key], width=0.85)
                bottoms = [b + v for b, v in zip(bottoms, values)]
            axes[0].set_title('Outcome Counts Per Episode Window')
            axes[0].set_xticks(tick_positions)
            axes[0].set_xticklabels(tick_labels, rotation=35, ha='right')
            axes[0].set_ylabel('count')
            axes[0].legend(ncol=3)
            axes[0].grid(axis='y', alpha=0.25)

            avg_returns = [row['avg_return'] for row in window_rows]
            avg_lengths = [row['avg_length'] for row in window_rows]
            axes[1].plot(x, avg_returns, marker='o', markersize=4, label='avg_return', color='tab:blue')
            axes[1].plot(x, avg_lengths, marker='s', markersize=4, label='avg_length', color='tab:cyan')
            axes[1].set_title('Average Return And Episode Length')
            axes[1].set_xticks(tick_positions)
            axes[1].set_xticklabels(tick_labels, rotation=35, ha='right')
            axes[1].set_ylabel('value')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            avg_actor_losses = [row['avg_actor_loss'] for row in window_rows]
            avg_critic_losses = [row['avg_critic_loss'] for row in window_rows]
            axes[2].plot(x, avg_actor_losses, marker='o', markersize=4, label='avg_actor_loss', color='tab:olive')
            axes[2].plot(x, avg_critic_losses, marker='s', markersize=4, label='avg_critic_loss', color='tab:pink')
            axes[2].set_title('Average Actor And Critic Loss')
            axes[2].set_xticks(tick_positions)
            axes[2].set_xticklabels(tick_labels, rotation=35, ha='right')
            axes[2].set_ylabel('loss')
            axes[2].legend()
            axes[2].grid(alpha=0.3)

            fig.tight_layout(pad=2.0)
            plot_path = base_metrics_path.with_name(f'{base_metrics_path.stem}_episode_windows.png')
            fig.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            outputs['plot'] = str(plot_path)
    except Exception:
        outputs['plot'] = 'skipped'

    return outputs


def _draw_zone_top_view(ax, center_xy: list[float], radius: float, warning_distance: float) -> None:
    import matplotlib.patches as patches

    zone_patch = patches.Circle(center_xy, radius, fill=False, color='tab:red', linewidth=1.6)
    warn_patch = patches.Circle(
        center_xy,
        radius + warning_distance,
        fill=False,
        color='tab:red',
        linestyle='--',
        linewidth=1.0,
        alpha=0.45,
    )
    ax.add_patch(zone_patch)
    ax.add_patch(warn_patch)


def _draw_zone_vertical_projection(ax, center_value: float, radius: float, label: str, color: str = 'tab:red') -> None:
    import numpy as np

    xs = np.linspace(center_value - radius, center_value + radius, 120)
    zs = np.sqrt(np.maximum(radius**2 - (xs - center_value) ** 2, 0.0))
    ax.plot(xs, zs, color=color, linewidth=1.4, alpha=0.8, label=label)


def export_episode_result(
    target_dir: Path,
    stem: str,
    record: dict[str, Any],
    config_payload: dict[str, Any],
) -> dict[str, str]:
    """Save one episode's scenario parameters, trajectory and visualization."""

    import matplotlib.pyplot as plt
    import numpy as np

    target_dir = ensure_dir(target_dir)
    json_path = target_dir / f'{stem}.json'
    png_path = target_dir / f'{stem}.png'

    payload = {
        'episode': record['episode'],
        'total_steps': record['total_steps'],
        'return': record['return'],
        'length': record['length'],
        'outcome': record['outcome'],
        'actor_loss': record['actor_loss'],
        'critic_loss': record['critic_loss'],
        'scenario': record['scenario'],
        'trajectory': record['trajectory'],
        'final_state': record['final_state'],
        'info': record['info'],
        'config': config_payload,
    }
    save_json(json_path, payload)

    traj = np.asarray(record['trajectory'], dtype=float)
    start = np.asarray(record['scenario']['state'][:3], dtype=float)
    goal = np.asarray(record['scenario']['goal'], dtype=float)
    zones = record['scenario']['zones']
    scenario_cfg = config_payload['scenario']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_xy = axes[0, 0]
    ax_xz = axes[0, 1]
    ax_yz = axes[1, 0]
    ax_text = axes[1, 1]

    ax_xy.plot(traj[:, 0], traj[:, 1], color='tab:blue', linewidth=2.0, label='trajectory')
    ax_xy.scatter(start[0], start[1], color='tab:blue', s=55, marker='o', label='start')
    ax_xy.scatter(goal[0], goal[1], color='tab:green', s=70, marker='*', label='goal')
    for idx, zone in enumerate(zones, start=1):
        center_xy = zone['center_xy']
        radius = zone['radius']
        _draw_zone_top_view(ax_xy, center_xy, radius, scenario_cfg['warning_distance'])
        ax_xy.text(center_xy[0], center_xy[1], f'Z{idx}', fontsize=8, color='tab:red')
    ax_xy.set_title('Top View (X-Y)')
    ax_xy.set_xlabel('x')
    ax_xy.set_ylabel('y')
    ax_xy.set_xlim(-scenario_cfg['world_xy'], scenario_cfg['world_xy'])
    ax_xy.set_ylim(-scenario_cfg['world_xy'], scenario_cfg['world_xy'])
    ax_xy.legend(loc='upper left')
    ax_xy.grid(alpha=0.3)
    ax_xy.set_aspect('equal', adjustable='box')

    ax_xz.plot(traj[:, 0], traj[:, 2], color='tab:blue', linewidth=2.0, label='trajectory')
    ax_xz.scatter(start[0], start[2], color='tab:blue', s=55, marker='o', label='start')
    ax_xz.scatter(goal[0], goal[2], color='tab:green', s=70, marker='*', label='goal')
    for idx, zone in enumerate(zones, start=1):
        _draw_zone_vertical_projection(ax_xz, zone['center_xy'][0], zone['radius'], f'zone {idx}')
    ax_xz.axhline(scenario_cfg['ground_warning_height'], color='tab:orange', linestyle='--', alpha=0.7, label='ground warning')
    ax_xz.set_title('Side View (X-Z)')
    ax_xz.set_xlabel('x')
    ax_xz.set_ylabel('z')
    ax_xz.set_xlim(-scenario_cfg['world_xy'], scenario_cfg['world_xy'])
    ax_xz.set_ylim(0.0, scenario_cfg['world_z_max'])
    ax_xz.grid(alpha=0.3)
    ax_xz.legend(loc='upper left', ncol=2)

    ax_yz.plot(traj[:, 1], traj[:, 2], color='tab:blue', linewidth=2.0, label='trajectory')
    ax_yz.scatter(start[1], start[2], color='tab:blue', s=55, marker='o', label='start')
    ax_yz.scatter(goal[1], goal[2], color='tab:green', s=70, marker='*', label='goal')
    for idx, zone in enumerate(zones, start=1):
        _draw_zone_vertical_projection(ax_yz, zone['center_xy'][1], zone['radius'], f'zone {idx}')
    ax_yz.axhline(scenario_cfg['ground_warning_height'], color='tab:orange', linestyle='--', alpha=0.7, label='ground warning')
    ax_yz.set_title('Front View (Y-Z)')
    ax_yz.set_xlabel('y')
    ax_yz.set_ylabel('z')
    ax_yz.set_xlim(-scenario_cfg['world_xy'], scenario_cfg['world_xy'])
    ax_yz.set_ylim(0.0, scenario_cfg['world_z_max'])
    ax_yz.grid(alpha=0.3)
    ax_yz.legend(loc='upper left', ncol=2)

    ax_text.axis('off')
    zone_lines = [
        f"zone {idx}: center=({zone['center_xy'][0]:.1f}, {zone['center_xy'][1]:.1f}), r={zone['radius']:.1f}"
        for idx, zone in enumerate(zones, start=1)
    ]
    summary = [
        f"episode: {record['episode']}",
        f"steps consumed: {record['total_steps']}",
        f"outcome: {record['outcome']}",
        f"return: {record['return']:.2f}",
        f"length: {record['length']}",
        f"goal distance: {record['info']['goal_distance']:.2f}",
        f"start: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})",
        f"goal: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})",
        f"goal radius: {scenario_cfg['goal_radius']}",
        f"warning_distance: {scenario_cfg['warning_distance']}",
        f"boundary_warning_distance: {scenario_cfg['boundary_warning_distance']}",
        f"ground_warning_height: {scenario_cfg['ground_warning_height']}",
        '',
        *zone_lines,
    ]
    ax_text.text(0.0, 1.0, '\n'.join(summary), va='top', ha='left', fontsize=10, family='monospace')
    ax_text.set_title('Scenario Summary')

    fig.suptitle(f"Episode {record['episode']} - {record['outcome']}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=2.0)
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return {'json': str(json_path), 'plot': str(png_path)}


def make_episode_capture_callback(
    result_root: Path,
    summary_every_episodes: int,
    total_timesteps: int,
    config_payload: dict[str, Any],
) -> Callable[[dict[str, Any]], None]:
    """Create a callback that stores step snapshots and sparse goal examples."""

    snapshot_dir = ensure_dir(result_root / 'step_snapshots')
    goal_dir = ensure_dir(result_root / 'goal_examples')

    snapshot_interval = max(1, total_timesteps // 10)
    next_snapshot_step = snapshot_interval
    saved_goal_groups: set[int] = set()

    def callback(record: dict[str, Any]) -> None:
        nonlocal next_snapshot_step

        while record['total_steps'] >= next_snapshot_step and next_snapshot_step <= total_timesteps:
            snapshot_idx = max(1, next_snapshot_step // snapshot_interval)
            stem = (
                f"step_{snapshot_idx:02d}_s{next_snapshot_step:06d}_"
                f"ep{record['episode']:05d}_{record['outcome']}"
            )
            export_episode_result(snapshot_dir, stem, record, config_payload)
            next_snapshot_step += snapshot_interval

        if summary_every_episodes > 0 and record['outcome'] == 'goal':
            window_idx = (record['episode'] - 1) // summary_every_episodes
            goal_group_idx = window_idx // 2
            if goal_group_idx not in saved_goal_groups:
                saved_goal_groups.add(goal_group_idx)
                stem = f'goal_group_{goal_group_idx + 1:02d}_ep{record["episode"]:05d}'
                export_episode_result(goal_dir, stem, record, config_payload)

    return callback


def main() -> None:
    parser = argparse.ArgumentParser(description='Train TD3 on the static no-fly-zone task.')
    parser.add_argument('--model', choices=['snn', 'ann'], default='snn')
    parser.add_argument('--timesteps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--bc-checkpoint', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--metrics-out', type=Path, default=None)
    parser.add_argument('--summary-every-episodes', type=int, default=50)
    parser.add_argument('--actor-freeze-steps', type=int, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.actor_freeze_steps is not None:
        cfg.training.actor_freeze_steps = args.actor_freeze_steps
    set_global_seed(args.seed)

    finished_at = now_timestamp()
    base_output = args.output or Path(f'outputs/td3_{args.model}.pt')
    base_metrics = args.metrics_out or Path(f'outputs/td3_{args.model}_metrics.json')
    log_dir, output, metrics_out = build_log_paths(base_output, base_metrics, finished_at)
    results_dir = ensure_dir(log_dir / 'results')

    env = make_env(cfg, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    actor = make_actor(cfg, args.model, obs.shape[0], env.action_space.shape[0])
    warmup_strategy = 'random'
    if args.bc_checkpoint:
        actor.load_state_dict(load_checkpoint(args.bc_checkpoint)['state_dict'])
        warmup_strategy = 'policy'
    critic1, critic2 = make_critics(cfg, obs.shape[0], env.action_space.shape[0])
    trainer = TD3Trainer(
        env=env,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_lr=cfg.training.actor_lr,
        critic_lr=cfg.training.critic_lr,
        gamma=cfg.training.gamma,
        tau=cfg.training.tau,
        policy_noise=cfg.training.policy_noise,
        noise_clip=cfg.training.noise_clip,
        policy_delay=cfg.training.policy_delay,
        replay_size=cfg.training.replay_size,
        batch_size=cfg.training.batch_size,
        warmup_steps=cfg.training.warmup_steps,
        exploration_noise=cfg.training.exploration_noise,
        success_sample_bias=cfg.training.success_sample_bias,
        actor_freeze_steps=cfg.training.actor_freeze_steps,
        warmup_strategy=warmup_strategy,
        device=cfg.training.device,
    )
    episode_callback = make_episode_capture_callback(
        result_root=results_dir,
        summary_every_episodes=args.summary_every_episodes,
        total_timesteps=args.timesteps,
        config_payload=cfg.to_dict(),
    )
    metrics = trainer.train(
        args.timesteps,
        log_interval=max(100, args.timesteps // 10),
        verbose=True,
        summary_every_episodes=args.summary_every_episodes,
        episode_callback=episode_callback,
    )

    metrics_dict = metrics.to_dict()
    metrics_dict['finished_at'] = finished_at
    metrics_dict['summary_every_episodes'] = args.summary_every_episodes
    metrics_dict['log_dir'] = str(log_dir)
    metrics_dict['results_dir'] = str(results_dir)
    metrics_dict['actor_freeze_steps'] = cfg.training.actor_freeze_steps
    metrics_dict['success_sample_bias'] = cfg.training.success_sample_bias

    save_checkpoint(
        output,
        {
            'model_type': args.model,
            'state_dict': actor.state_dict(),
            'metrics': metrics_dict,
            'config': cfg.to_dict(),
            'finished_at': finished_at,
            'log_dir': str(log_dir),
            'results_dir': str(results_dir),
        },
    )
    report_outputs = export_training_report(metrics_out, metrics_dict)
    print(f'Saved TD3 checkpoint to {output}')
    print(f"Episodes: {metrics.episodes}, Steps: {metrics.steps}, Critic loss: {metrics.critic_loss:.4f}")
    print(f'Result directory: {results_dir}')
    print(f"Training reports: {report_outputs}")


if __name__ == '__main__':
    main()
