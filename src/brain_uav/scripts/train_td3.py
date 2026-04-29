"""Train the TD3 policy after behavior cloning initialization."""

from __future__ import annotations

import argparse
from pathlib import Path
from copy import deepcopy
from typing import Any, Callable

from ..config import ExperimentConfig
from ..curriculum import describe_curriculum_mix, parse_curriculum_mix
from ..scripts.common import (
    DEVICE_CHOICES,
    SNN_BACKEND_CHOICES,
    build_log_prefix,
    configure_training_runtime,
    make_actor,
    make_critics,
    make_env,
)
from ..trainers import TD3Trainer
from ..utils.io import (
    build_log_paths,
    ensure_dir,
    load_checkpoint,
    log_root_path,
    model_output_path,
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


def _draw_goal_radius_projection(ax, center: tuple[float, float], radius: float) -> None:
    import matplotlib.patches as patches

    goal_patch = patches.Circle(
        center,
        radius,
        fill=False,
        color='tab:green',
        linestyle='--',
        linewidth=1.2,
        alpha=0.5,
        label='goal radius',
    )
    ax.add_patch(goal_patch)


def _resolve_active_goal_radius(record: dict[str, Any], scenario_cfg: dict[str, Any]) -> float:
    return float(record['info'].get('active_goal_radius', scenario_cfg['goal_radius']))


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
        'zone_count': len(record['scenario']['zones']),
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
    active_goal_radius = _resolve_active_goal_radius(record, scenario_cfg)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 18),
        gridspec_kw={'height_ratios': [6, 1, 1]},
    )
    ax_xy, ax_xz, ax_yz = axes

    ax_xy.plot(traj[:, 0], traj[:, 1], color='tab:blue', linewidth=2.0, label='trajectory')
    ax_xy.scatter(start[0], start[1], color='tab:blue', s=55, marker='o', label='start')
    ax_xy.scatter(goal[0], goal[1], color='tab:green', s=70, marker='*', label='goal')
    _draw_goal_radius_projection(ax_xy, (goal[0], goal[1]), active_goal_radius)
    for idx, zone in enumerate(zones, start=1):
        center_xy = zone['center_xy']
        radius = zone['radius']
        _draw_zone_top_view(ax_xy, center_xy, radius, scenario_cfg['warning_distance'])
        ax_xy.text(center_xy[0], center_xy[1], f'Z{idx}', fontsize=8, color='tab:red')
    ax_xy.set_title('Top View (X-Y)')
    ax_xy.set_xlabel('x (km)')
    ax_xy.set_ylabel('y (km)')
    ax_xy.set_xlim(-scenario_cfg['world_xy'], scenario_cfg['world_xy'])
    ax_xy.set_ylim(-scenario_cfg['world_xy'], scenario_cfg['world_xy'])
    ax_xy.legend(loc='upper left')
    ax_xy.grid(alpha=0.3)
    ax_xy.set_aspect('equal', adjustable='box')

    ax_xz.plot(traj[:, 0], traj[:, 2], color='tab:blue', linewidth=2.0, label='trajectory')
    ax_xz.scatter(start[0], start[2], color='tab:blue', s=55, marker='o', label='start')
    ax_xz.scatter(goal[0], goal[2], color='tab:green', s=70, marker='*', label='goal')
    _draw_goal_radius_projection(ax_xz, (goal[0], goal[2]), active_goal_radius)
    for idx, zone in enumerate(zones, start=1):
        _draw_zone_vertical_projection(ax_xz, zone['center_xy'][0], zone['radius'], f'zone {idx}')
    ax_xz.axhline(scenario_cfg['ground_warning_height'], color='tab:orange', linestyle='--', alpha=0.7, label='ground warning')
    ax_xz.set_title('Side View (X-Z)')
    ax_xz.set_xlabel('x (km)')
    ax_xz.set_ylabel('z (km)')
    ax_xz.set_xlim(-scenario_cfg['world_xy'], scenario_cfg['world_xy'])
    ax_xz.set_ylim(0.0, scenario_cfg['world_z_max'])
    ax_xz.grid(alpha=0.3)
    ax_xz.legend(loc='upper left', ncol=2)

    ax_yz.plot(traj[:, 1], traj[:, 2], color='tab:blue', linewidth=2.0, label='trajectory')
    ax_yz.scatter(start[1], start[2], color='tab:blue', s=55, marker='o', label='start')
    ax_yz.scatter(goal[1], goal[2], color='tab:green', s=70, marker='*', label='goal')
    _draw_goal_radius_projection(ax_yz, (goal[1], goal[2]), active_goal_radius)
    for idx, zone in enumerate(zones, start=1):
        _draw_zone_vertical_projection(ax_yz, zone['center_xy'][1], zone['radius'], f'zone {idx}')
    ax_yz.axhline(scenario_cfg['ground_warning_height'], color='tab:orange', linestyle='--', alpha=0.7, label='ground warning')
    ax_yz.set_title('Front View (Y-Z)')
    ax_yz.set_xlabel('y (km)')
    ax_yz.set_ylabel('z (km)')
    ax_yz.set_xlim(-scenario_cfg['world_xy'], scenario_cfg['world_xy'])
    ax_yz.set_ylim(0.0, scenario_cfg['world_z_max'])
    ax_yz.grid(alpha=0.3)
    ax_yz.legend(loc='upper left', ncol=2)

    zone_lines = [
        f"zone {idx}: center=({zone['center_xy'][0]:.1f}, {zone['center_xy'][1]:.1f}) km, r={zone['radius']:.1f} km"
        for idx, zone in enumerate(zones, start=1)
    ]
    scenario_label = record['info'].get('scenario_name')
    scenario_id = record['info'].get('scenario_id')
    category = record['info'].get('category')
    corridor_width = record['info'].get('corridor_width')
    min_clearance = record['info'].get('min_clearance_to_boundary')
    summary = [
        f"episode: {record['episode']}",
        f"steps consumed: {record['total_steps']}",
        f"steps: {record['length']}",
        f"outcome: {record['outcome']}",
        f"return: {record['return']:.2f}",
        f"scenario: {scenario_label or 'n/a'}",
        f"scenario_id: {scenario_id or 'n/a'}",
        f"category: {category or 'n/a'}",
        f"curriculum_level: {record['info'].get('curriculum_level', 'unknown')}",
        f"goal distance (km): {record['info']['goal_distance']:.2f}",
        f"start (km): ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})",
        f"goal (km): ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})",
        f"zone_count: {len(zones)}",
        f"corridor_width (km): {corridor_width if corridor_width is not None else 'n/a'}",
        f"min_clearance_to_boundary (km): {min_clearance if min_clearance is not None else 'n/a'}",
        f"active goal radius (km): {active_goal_radius}",
        f"warning_distance (km): {scenario_cfg['warning_distance']}",
        f"boundary_warning_distance (km): {scenario_cfg['boundary_warning_distance']}",
        f"ground_warning_height (km): {scenario_cfg['ground_warning_height']}",
        '',
        *zone_lines,
    ]
    fig.text(0.02, 0.015, ' | '.join(summary[:10]), ha='left', va='bottom', fontsize=8, family='monospace')

    fig.suptitle(f"Episode {record['episode']} - {record['outcome']}", fontsize=15)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97], pad=2.0)
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return {'json': str(json_path), 'png': str(png_path)}

def make_episode_capture_callback(
    result_root: Path,
    summary_every_episodes: int,
    total_timesteps: int,
    config_payload: dict[str, Any],
) -> Callable[[dict[str, Any]], None]:
    """Create a callback that stores sparse step snapshots and goal examples.

    Current policy:
    - save one step snapshot every 1/20 of total training steps
    - save at most one goal example per 5 episode windows
    """
    snapshot_dir = ensure_dir(result_root / 'step_snapshots')
    goal_dir = ensure_dir(result_root / 'goal_examples')
    snapshot_interval = max(1, total_timesteps // 20)
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
            goal_group_idx = window_idx // 5
            if goal_group_idx not in saved_goal_groups:
                saved_goal_groups.add(goal_group_idx)
                stem = f'goal_group_{goal_group_idx + 1:02d}_ep{record["episode"]:05d}'
                export_episode_result(goal_dir, stem, record, config_payload)

    return callback


def is_qualified_early_stop_window(
    window_row: dict[str, Any],
    *,
    goal_rate_threshold: float,
    max_failures_per_window: int,
) -> tuple[bool, dict[str, float]]:
    episode_count = max(int(window_row.get('episode_count', 0)), 1)
    goal_count = int(window_row.get('goal_count', 0))
    failures = episode_count - goal_count
    goal_rate = float(goal_count) / episode_count
    qualified = failures <= max_failures_per_window or goal_rate >= goal_rate_threshold
    return qualified, {
        'episode_count': float(episode_count),
        'goal_count': float(goal_count),
        'failures': float(failures),
        'goal_rate': goal_rate,
    }


def make_early_stop_callback(
    enabled: bool,
    goal_rate_threshold: float,
    consecutive_windows: int,
    min_steps: int,
    max_failures_per_window: int,
) -> Callable[[dict[str, Any]], str | None] | None:
    if not enabled:
        return None

    consecutive_qualified = 0

    def callback(window_row: dict[str, Any]) -> str | None:
        nonlocal consecutive_qualified
        qualified, stats = is_qualified_early_stop_window(
            window_row,
            goal_rate_threshold=goal_rate_threshold,
            max_failures_per_window=max_failures_per_window,
        )
        if qualified:
            consecutive_qualified += 1
        else:
            consecutive_qualified = 0
        if int(window_row.get('total_steps', 0)) < min_steps:
            return None
        if consecutive_qualified < consecutive_windows:
            return None
        return (
            f'qualified_windows={consecutive_qualified}/{consecutive_windows} after min_steps={min_steps} '
            f"(failures={int(stats['failures'])}, goal_rate={stats['goal_rate']:.2f}, "
            f'failures_limit={max_failures_per_window}, goal_rate_threshold={goal_rate_threshold:.2f})'
        )

    return callback


def load_training_state(init_checkpoint: Path | None, actor, critic1, critic2, trainer, log_prefix: str) -> str:
    """Restore checkpoint according to whether it is BC or TD3."""

    if init_checkpoint is None:
        return 'random'

    checkpoint = load_checkpoint(init_checkpoint)
    actor.load_state_dict(checkpoint['state_dict'])
    has_critics = 'critic1_state_dict' in checkpoint and 'critic2_state_dict' in checkpoint
    if not has_critics:
        trainer.actor_target.load_state_dict(checkpoint['state_dict'])
        trainer.set_bc_reference_actor(deepcopy(actor), source='bc')
        print(f'{log_prefix} loaded actor-only BC checkpoint; critics are randomly initialized.')
        return 'policy'

    critic1.load_state_dict(checkpoint['critic1_state_dict'])
    critic2.load_state_dict(checkpoint['critic2_state_dict'])
    if 'actor_target_state_dict' in checkpoint:
        trainer.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
    else:
        trainer.actor_target.load_state_dict(checkpoint['state_dict'])
    if 'critic1_target_state_dict' in checkpoint:
        trainer.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
    else:
        trainer.critic1_target.load_state_dict(checkpoint['critic1_state_dict'])
    if 'critic2_target_state_dict' in checkpoint:
        trainer.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
    else:
        trainer.critic2_target.load_state_dict(checkpoint['critic2_state_dict'])
    if 'actor_optimizer_state_dict' in checkpoint:
        trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    if 'critic_optimizer_state_dict' in checkpoint:
        trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    trainer.set_bc_reference_actor(deepcopy(actor), source='previous_stage')
    print(f'{log_prefix} loaded actor and twin critics from TD3 checkpoint for continuation.')
    return 'policy'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the TD3 curriculum policy stage by stage.')
    parser.add_argument('--model', choices=['snn', 'ann'], default='snn')
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--curriculum-level', choices=['easy', 'easy_two_zone', 'medium', 'hard'], required=True)
    parser.add_argument('--curriculum-mix', type=str, default=None)
    parser.add_argument('--init-checkpoint', type=Path, default=None)
    parser.add_argument('--bc-checkpoint', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--metrics-out', type=Path, default=None)
    parser.add_argument('--log-root', type=Path, default=None)
    parser.add_argument('--summary-every-episodes', type=int, default=15)
    parser.add_argument('--actor-freeze-steps', type=int, default=None)
    parser.add_argument('--critic-grad-clip-norm', type=float, default=None)
    parser.add_argument('--early-stop-enabled', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--early-stop-goal-rate', type=float, default=0.95)
    parser.add_argument('--early-stop-windows', type=int, default=4)
    parser.add_argument('--early-stop-max-failures-per-window', type=int, default=1)
    parser.add_argument('--early-stop-min-steps', type=int, default=125000)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--snn-backend', choices=SNN_BACKEND_CHOICES, default='torch')
    return parser


def apply_model_training_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.model == 'ann':
        # Use more conservative defaults for ANN to avoid late-training collapse.
        cfg.training.actor_lr = 1.5e-4
        cfg.training.critic_lr = 2.5e-4
        if args.actor_freeze_steps is None:
            cfg.training.actor_freeze_steps = 25_000
        else:
            cfg.training.actor_freeze_steps = args.actor_freeze_steps
        if args.critic_grad_clip_norm is None:
            cfg.training.critic_grad_clip_norm = 1.0
        else:
            cfg.training.critic_grad_clip_norm = args.critic_grad_clip_norm
        if args.curriculum_level == 'hard':
            cfg.training.actor_lr *= 0.75
            cfg.training.critic_lr *= 0.85
            cfg.training.success_sample_bias = 4.0
        return

    if args.actor_freeze_steps is not None:
        cfg.training.actor_freeze_steps = args.actor_freeze_steps
    if args.critic_grad_clip_norm is not None:
        cfg.training.critic_grad_clip_norm = args.critic_grad_clip_norm


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.timesteps is None:
        args.timesteps = 1200000 if args.curriculum_level == 'hard' else 900000

    cfg = ExperimentConfig()
    resolved_device = configure_training_runtime(
        cfg,
        model_type=args.model,
        device=args.device,
        snn_backend=args.snn_backend,
    )
    log_prefix = build_log_prefix(args.model, args.curriculum_level)
    apply_model_training_overrides(cfg, args)

    set_global_seed(args.seed)

    curriculum_mix = parse_curriculum_mix(args.curriculum_mix, fallback_level=args.curriculum_level)
    finished_at = now_timestamp()
    base_output = args.output or model_output_path('td3', model=args.model, level=args.curriculum_level)
    base_metrics = args.metrics_out or Path(f'td3_{args.model}_{args.curriculum_level}_metrics.json')
    log_dir, output, metrics_out = build_log_paths(
        base_output,
        base_metrics,
        finished_at,
        log_root=args.log_root or log_root_path('td3', level=args.curriculum_level),
    )
    results_dir = ensure_dir(log_dir / 'results')

    env = make_env(
        cfg,
        seed=args.seed,
        curriculum_level=args.curriculum_level,
        curriculum_mix=curriculum_mix,
        goal_radius_curriculum_enabled=True,
    )
    obs, _ = env.reset(seed=args.seed)
    actor = make_actor(cfg, args.model, obs.shape[0], env.action_space.shape[0])
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
        near_goal_sample_bias=cfg.training.near_goal_sample_bias,
        actor_freeze_steps=cfg.training.actor_freeze_steps,
        actor_grad_clip_norm=cfg.training.actor_grad_clip_norm,
        actor_rl_scale_alpha=cfg.training.actor_rl_scale_alpha,
        terminal_geo_regularization_enabled=cfg.training.terminal_geo_regularization_enabled,
        terminal_geo_radius=cfg.training.terminal_geo_radius,
        terminal_geo_lambda=cfg.training.terminal_geo_lambda,
        terminal_geo_safe_clearance=cfg.training.terminal_geo_safe_clearance,
        near_goal_radius=cfg.training.near_goal_radius,
        success_replay_fraction=cfg.training.success_replay_fraction,
        success_batch_fraction=cfg.training.success_batch_fraction,
        noise_decay_fraction=cfg.training.noise_decay_fraction,
        exploration_noise_final=cfg.training.exploration_noise_final,
        policy_noise_final=cfg.training.policy_noise_final,
        noise_clip_final=cfg.training.noise_clip_final,
        critic_grad_clip_norm=cfg.training.critic_grad_clip_norm,
        warmup_strategy='random',
        device=cfg.training.device,
        curriculum_level=args.curriculum_level,
    )
    init_checkpoint = args.init_checkpoint or args.bc_checkpoint
    trainer.warmup_strategy = load_training_state(init_checkpoint, actor, critic1, critic2, trainer, log_prefix)
    print(
        f'{log_prefix} bc_regularization_enabled={trainer.metrics.bc_regularization_enabled} '
        f'reference_source={trainer.metrics.reference_source} '
        f'actor_freeze_steps={cfg.training.actor_freeze_steps}'
    )

    config_payload = cfg.to_dict()
    config_payload['scenario']['goal_radius_curriculum_enabled'] = True
    config_payload['curriculum_level'] = args.curriculum_level
    config_payload['curriculum_mix'] = curriculum_mix
    episode_callback = make_episode_capture_callback(
        result_root=results_dir,
        summary_every_episodes=args.summary_every_episodes,
        total_timesteps=args.timesteps,
        config_payload=config_payload,
    )
    early_stop_callback = make_early_stop_callback(
        enabled=args.early_stop_enabled and args.summary_every_episodes > 0,
        goal_rate_threshold=args.early_stop_goal_rate,
        consecutive_windows=args.early_stop_windows,
        min_steps=args.early_stop_min_steps,
        max_failures_per_window=args.early_stop_max_failures_per_window,
    )
    metrics = trainer.train(
        args.timesteps,
        log_interval=max(100, args.timesteps // 10),
        verbose=True,
        summary_every_episodes=args.summary_every_episodes,
        episode_callback=episode_callback,
        window_callback=early_stop_callback,
        log_prefix=log_prefix,
    )

    metrics_dict = metrics.to_dict()
    metrics_dict['finished_at'] = finished_at
    metrics_dict['summary_every_episodes'] = args.summary_every_episodes
    metrics_dict['log_dir'] = str(log_dir)
    metrics_dict['results_dir'] = str(results_dir)
    metrics_dict['actor_freeze_steps'] = cfg.training.actor_freeze_steps
    metrics_dict['actor_grad_clip_norm'] = cfg.training.actor_grad_clip_norm
    metrics_dict['critic_grad_clip_norm'] = cfg.training.critic_grad_clip_norm
    metrics_dict['exploration_noise_current'] = trainer.exploration_noise_current
    metrics_dict['policy_noise_current'] = trainer.policy_noise_current
    metrics_dict['noise_clip_current'] = trainer.noise_clip_current
    metrics_dict['success_sample_bias'] = cfg.training.success_sample_bias
    metrics_dict['near_goal_radius'] = cfg.training.near_goal_radius
    metrics_dict['near_goal_sample_bias'] = cfg.training.near_goal_sample_bias
    metrics_dict['success_replay_fraction'] = cfg.training.success_replay_fraction
    metrics_dict['success_batch_fraction'] = cfg.training.success_batch_fraction
    metrics_dict['noise_decay_fraction'] = cfg.training.noise_decay_fraction
    metrics_dict['exploration_noise_final'] = cfg.training.exploration_noise_final
    metrics_dict['policy_noise_final'] = cfg.training.policy_noise_final
    metrics_dict['noise_clip_final'] = cfg.training.noise_clip_final
    metrics_dict['bc_regularization_enabled'] = trainer.metrics.bc_regularization_enabled
    metrics_dict['reference_source'] = trainer.metrics.reference_source
    metrics_dict['bc_lambda'] = trainer.metrics.bc_lambda
    metrics_dict['bc_loss'] = trainer.metrics.bc_loss
    metrics_dict['rl_actor_loss'] = trainer.metrics.rl_actor_loss
    metrics_dict['scaled_rl_actor_loss'] = trainer.metrics.scaled_rl_actor_loss
    metrics_dict['actor_rl_scale'] = trainer.metrics.actor_rl_scale
    metrics_dict['actor_rl_scale_alpha'] = cfg.training.actor_rl_scale_alpha
    metrics_dict['terminal_geo_loss'] = trainer.metrics.terminal_geo_loss
    metrics_dict['terminal_geo_lambda'] = trainer.metrics.terminal_geo_lambda
    metrics_dict['terminal_geo_regularization_enabled'] = cfg.training.terminal_geo_regularization_enabled
    metrics_dict['terminal_geo_radius'] = cfg.training.terminal_geo_radius
    metrics_dict['terminal_geo_safe_clearance'] = cfg.training.terminal_geo_safe_clearance
    metrics_dict['replay_success_fraction'] = trainer.metrics.replay_success_fraction
    metrics_dict['replay_near_goal_fraction'] = trainer.metrics.replay_near_goal_fraction
    metrics_dict['success_replay_size'] = trainer.metrics.success_replay_size
    metrics_dict['sample_success_fraction'] = trainer.metrics.sample_success_fraction
    metrics_dict['sample_near_goal_fraction'] = trainer.metrics.sample_near_goal_fraction
    metrics_dict['goal_radius_curriculum_enabled'] = True
    metrics_dict['goal_radius_curriculum'] = cfg.scenario.goal_radius_curriculum
    metrics_dict['total_actor_loss'] = trainer.metrics.actor_loss
    metrics_dict['early_stop_enabled'] = bool(args.early_stop_enabled and args.summary_every_episodes > 0)
    metrics_dict['early_stop_goal_rate'] = args.early_stop_goal_rate
    metrics_dict['early_stop_windows'] = args.early_stop_windows
    metrics_dict['early_stop_max_failures_per_window'] = args.early_stop_max_failures_per_window
    metrics_dict['early_stop_min_steps'] = args.early_stop_min_steps
    metrics_dict['stopped_early'] = trainer.early_stopped
    metrics_dict['stop_reason'] = trainer.stop_reason
    metrics_dict['success_sample_bias'] = cfg.training.success_sample_bias
    metrics_dict['curriculum_level'] = args.curriculum_level
    metrics_dict['curriculum_mix'] = curriculum_mix
    metrics_dict['init_checkpoint'] = str(init_checkpoint) if init_checkpoint else None
    metrics_dict['device'] = resolved_device
    metrics_dict['snn_backend'] = cfg.training.snn_backend if args.model == 'snn' else None

    save_checkpoint(
        output,
        {
            'model_type': args.model,
            'state_dict': actor.state_dict(),
            'critic1_state_dict': critic1.state_dict(),
            'critic2_state_dict': critic2.state_dict(),
            'actor_target_state_dict': trainer.actor_target.state_dict(),
            'critic1_target_state_dict': trainer.critic1_target.state_dict(),
            'critic2_target_state_dict': trainer.critic2_target.state_dict(),
            'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
            'metrics': metrics_dict,
            'config': config_payload,
            'finished_at': finished_at,
            'log_dir': str(log_dir),
            'results_dir': str(results_dir),
            'curriculum_level': args.curriculum_level,
            'curriculum_mix': curriculum_mix,
            'init_checkpoint': str(init_checkpoint) if init_checkpoint else None,
        },
    )
    report_outputs = export_training_report(metrics_out, metrics_dict)
    print(f'{log_prefix} saved checkpoint to {output}')
    print(f'{log_prefix} curriculum mix={describe_curriculum_mix(curriculum_mix)} device={resolved_device}')
    print(f"{log_prefix} episodes={metrics.episodes} steps={metrics.steps} critic_loss={metrics.critic_loss:.4f}")
    if trainer.stop_reason is not None:
        print(f"{log_prefix} early stop: {trainer.stop_reason}")
    print(f'{log_prefix} result directory: {results_dir}')
    print(f"{log_prefix} training reports: {report_outputs}")


if __name__ == '__main__':
    main()


