"""Train the TD3 policy after behavior cloning initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ExperimentConfig
from ..scripts.common import make_actor, make_critics, make_env
from ..trainers import TD3Trainer
from ..utils.io import load_checkpoint, now_timestamp, save_checkpoint, save_csv_rows, save_json, with_timestamp_suffix
from ..utils.seeding import set_global_seed


def export_training_report(base_metrics_path: Path, metrics: dict) -> dict[str, str]:
    """Write AI-friendly training summary files.

    输出三类结果：
    - 完整 JSON
    - 分段 episode 汇总 CSV
    - 分段 episode 柱状图 PNG（如果 matplotlib 可用）
    """

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
            goal_counts = [row['goal_count'] for row in window_rows]
            timeout_counts = [row['timeout_count'] for row in window_rows]
            boundary_counts = [row['boundary_count'] for row in window_rows]
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            x = range(len(labels))
            axes[0].bar(x, goal_counts, label='goal', color='tab:green')
            axes[0].bar(x, timeout_counts, bottom=goal_counts, label='timeout', color='tab:orange')
            axes[0].bar(
                x,
                boundary_counts,
                bottom=[g + t for g, t in zip(goal_counts, timeout_counts)],
                label='boundary',
                color='tab:red',
            )
            axes[0].set_title('Outcome Counts Per Episode Window')
            axes[0].set_xticks(list(x))
            axes[0].set_xticklabels(labels, rotation=45, ha='right')
            axes[0].legend()

            avg_returns = [row['avg_return'] for row in window_rows]
            avg_lengths = [row['avg_length'] for row in window_rows]
            axes[1].plot(labels, avg_returns, marker='o', label='avg_return', color='tab:blue')
            axes[1].plot(labels, avg_lengths, marker='s', label='avg_length', color='tab:purple')
            axes[1].set_title('Average Return / Length Per Episode Window')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend()
            fig.tight_layout()
            plot_path = base_metrics_path.with_name(f'{base_metrics_path.stem}_episode_windows.png')
            fig.savefig(plot_path, dpi=180)
            plt.close(fig)
            outputs['plot'] = str(plot_path)
    except Exception:
        outputs['plot'] = 'skipped'

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description='Train TD3 on the static no-fly-zone task.')
    parser.add_argument('--model', choices=['snn', 'ann'], default='snn')
    parser.add_argument('--timesteps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--bc-checkpoint', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--metrics-out', type=Path, default=None)
    parser.add_argument('--summary-every-episodes', type=int, default=50)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    set_global_seed(args.seed)
    env = make_env(cfg, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    actor = make_actor(cfg, args.model, obs.shape[0], env.action_space.shape[0])
    warmup_strategy = 'random'
    if args.bc_checkpoint:
        # TD3 从 BC 的结果出发，而不是从纯随机权重开始。
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
        warmup_strategy=warmup_strategy,
        device=cfg.training.device,
    )
    metrics = trainer.train(
        args.timesteps,
        log_interval=max(100, args.timesteps // 10),
        verbose=True,
        summary_every_episodes=args.summary_every_episodes,
    )

    finished_at = now_timestamp()
    base_output = args.output or Path(f'outputs/td3_{args.model}.pt')
    base_metrics = args.metrics_out or Path(f'outputs/td3_{args.model}_metrics.json')
    output = with_timestamp_suffix(base_output, finished_at)
    metrics_out = with_timestamp_suffix(base_metrics, finished_at)

    metrics_dict = metrics.to_dict()
    metrics_dict['finished_at'] = finished_at
    metrics_dict['summary_every_episodes'] = args.summary_every_episodes

    save_checkpoint(
        output,
        {
            'model_type': args.model,
            'state_dict': actor.state_dict(),
            'metrics': metrics_dict,
            'config': cfg.to_dict(),
            'finished_at': finished_at,
        },
    )
    report_outputs = export_training_report(metrics_out, metrics_dict)
    print(f'Saved TD3 checkpoint to {output}')
    print(f"Episodes: {metrics.episodes}, Steps: {metrics.steps}, Critic loss: {metrics.critic_loss:.4f}")
    print(f"Training reports: {report_outputs}")


if __name__ == '__main__':
    main()
