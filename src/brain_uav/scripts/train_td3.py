"""Train the TD3 policy after behavior cloning initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ExperimentConfig
from ..scripts.common import make_actor, make_critics, make_env
from ..trainers import TD3Trainer
from ..utils.io import (
    build_timestamped_run_paths,
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
    """Write AI-friendly training summary files.

    输出三类结果：
    - 完整 JSON
    - 分段 episode 汇总 CSV
    - 更完整的训练过程图 PNG
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
            x = list(range(len(labels)))

            fig, axes = plt.subplots(3, 1, figsize=(14, 12))

            bottoms = [0] * len(labels)
            for key in OUTCOME_KEYS:
                values = [row.get(f'{key}_count', 0) for row in window_rows]
                axes[0].bar(x, values, bottom=bottoms, label=key, color=OUTCOME_COLORS[key])
                bottoms = [b + v for b, v in zip(bottoms, values)]
            axes[0].set_title('Outcome Counts Per Episode Window')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels, rotation=45, ha='right')
            axes[0].set_ylabel('count')
            axes[0].legend(ncol=3)

            avg_returns = [row['avg_return'] for row in window_rows]
            avg_lengths = [row['avg_length'] for row in window_rows]
            axes[1].plot(labels, avg_returns, marker='o', label='avg_return', color='tab:blue')
            axes[1].plot(labels, avg_lengths, marker='s', label='avg_length', color='tab:cyan')
            axes[1].set_title('Average Return And Episode Length')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].set_ylabel('value')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            avg_actor_losses = [row['avg_actor_loss'] for row in window_rows]
            avg_critic_losses = [row['avg_critic_loss'] for row in window_rows]
            axes[2].plot(labels, avg_actor_losses, marker='o', label='avg_actor_loss', color='tab:olive')
            axes[2].plot(labels, avg_critic_losses, marker='s', label='avg_critic_loss', color='tab:pink')
            axes[2].set_title('Average Actor And Critic Loss')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].set_ylabel('loss')
            axes[2].legend()
            axes[2].grid(alpha=0.3)

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
    run_dir, output, metrics_out = build_timestamped_run_paths(base_output, base_metrics, finished_at)

    metrics_dict = metrics.to_dict()
    metrics_dict['finished_at'] = finished_at
    metrics_dict['summary_every_episodes'] = args.summary_every_episodes
    metrics_dict['run_dir'] = str(run_dir)

    save_checkpoint(
        output,
        {
            'model_type': args.model,
            'state_dict': actor.state_dict(),
            'metrics': metrics_dict,
            'config': cfg.to_dict(),
            'finished_at': finished_at,
            'run_dir': str(run_dir),
        },
    )
    report_outputs = export_training_report(metrics_out, metrics_dict)
    print(f'Saved TD3 checkpoint to {output}')
    print(f"Episodes: {metrics.episodes}, Steps: {metrics.steps}, Critic loss: {metrics.critic_loss:.4f}")
    print(f"Training reports: {report_outputs}")


if __name__ == '__main__':
    main()
