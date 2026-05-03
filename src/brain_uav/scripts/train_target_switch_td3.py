"""Fine-tune TD3 policies on target-switch curriculum episodes."""

from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, TextIO

import numpy as np

from ..config import ExperimentConfig
from ..envs import TargetSwitchTrajectoryEnv
from ..scripts.common import (
    DEVICE_CHOICES,
    SNN_BACKEND_CHOICES,
    build_log_prefix,
    configure_training_runtime,
    make_actor,
    make_critics,
)
from ..scripts.evaluate import _apply_checkpoint_config
from ..scripts.train_td3 import load_training_state
from ..target_switch import TARGET_SWITCH_LEVELS, target_switch_config_for_level
from ..trainers import TD3Trainer
from ..utils.io import ensure_dir, load_checkpoint, now_timestamp, save_checkpoint, save_json
from ..utils.seeding import set_global_seed


class _TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.secondary.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _target_switch_checkpoint_paths(run_dir: Path, model: str, level: str) -> dict[str, Path]:
    stem = f'td3_{model}_{level}'
    return {
        'compat': run_dir / f'{stem}.pt',
        'final': run_dir / f'{stem}_final.pt',
        'best': run_dir / f'{stem}_best.pt',
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Fine-tune TD3 on target-switch curriculum episodes.')
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/target_switch_models/td3_snn_hard.pt'))
    parser.add_argument('--model', choices=['snn', 'ann'], default='snn')
    parser.add_argument('--target-switch-level', choices=TARGET_SWITCH_LEVELS, default='target_switch_easy')
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('outputs/target_switch_training'))
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--snn-backend', choices=SNN_BACKEND_CHOICES, default='torch')
    parser.add_argument('--summary-every-episodes', type=int, default=50)
    parser.add_argument('--actor-freeze-steps', type=int, default=None)
    parser.add_argument('--critic-grad-clip-norm', type=float, default=None)
    return parser


def _apply_model_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.model == 'ann':
        cfg.training.actor_lr = 1.5e-4
        cfg.training.critic_lr = 2.5e-4
        cfg.training.actor_freeze_steps = 25_000 if args.actor_freeze_steps is None else args.actor_freeze_steps
        cfg.training.critic_grad_clip_norm = 1.0 if args.critic_grad_clip_norm is None else args.critic_grad_clip_norm
    else:
        if args.actor_freeze_steps is not None:
            cfg.training.actor_freeze_steps = args.actor_freeze_steps
        if args.critic_grad_clip_norm is not None:
            cfg.training.critic_grad_clip_norm = args.critic_grad_clip_norm


def _apply_target_switch_runtime_overrides(cfg: ExperimentConfig) -> None:
    cfg.rewards.terminal_guidance_radius = 150.0
    cfg.rewards.terminal_tangential_radius = 150.0
    cfg.training.terminal_geo_radius = 150.0


def _make_trainer(env: TargetSwitchTrajectoryEnv, actor, critic1, critic2, cfg: ExperimentConfig, level: str) -> TD3Trainer:
    return TD3Trainer(
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
        curriculum_level=level,
    )


def _mean_optional(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record.get(key) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _target_switch_window_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record['outcome']] = outcomes.get(record['outcome'], 0) + 1
    count = len(records)
    return {
        'episode_start': int(records[0]['episode']) if records else 0,
        'episode_end': int(records[-1]['episode']) if records else 0,
        'episode_count': count,
        'goal_count': outcomes.get('goal', 0),
        'goal_rate': outcomes.get('goal', 0) / count if count else 0.0,
        'collision_count': outcomes.get('collision', 0),
        'boundary_count': outcomes.get('boundary', 0),
        'ground_count': outcomes.get('ground', 0),
        'timeout_count': outcomes.get('timeout', 0),
        'pre_switch_done_count': sum(1 for record in records if bool(record.get('pre_switch_done', False))),
        'avg_return': float(np.mean([record.get('return', 0.0) for record in records])) if records else 0.0,
        'avg_length': float(np.mean([record.get('length', 0) for record in records])) if records else 0.0,
        'avg_final_to_new_distance': _mean_optional(records, 'final_to_new_distance'),
        'avg_post_switch_steps': float(np.mean([record.get('post_switch_steps', 0) or 0 for record in records]))
        if records
        else 0.0,
        'avg_switch_alignment_reward': float(
            np.mean([record.get('switch_alignment_reward_mean', 0.0) or 0.0 for record in records])
        )
        if records
        else 0.0,
        'avg_ceiling_penalty': float(np.mean([record.get('ceiling_penalty_mean', 0.0) or 0.0 for record in records]))
        if records
        else 0.0,
    }


def _best_window_score(window: dict[str, Any]) -> tuple:
    final_distance = window.get('avg_final_to_new_distance')
    if final_distance is None:
        final_distance = float('inf')
    return (
        float(window.get('goal_rate', 0.0)),
        int(window.get('goal_count', 0)),
        -int(window.get('ground_count', 0)),
        -int(window.get('boundary_count', 0)),
        -int(window.get('collision_count', 0)),
        -int(window.get('timeout_count', 0)),
        -int(window.get('pre_switch_done_count', 0)),
        -float(final_distance),
        float(window.get('avg_return', 0.0)),
    )


def _is_better_window(candidate: dict[str, Any], current_best: dict[str, Any] | None) -> bool:
    return current_best is None or _best_window_score(candidate) > _best_window_score(current_best)


def _checkpoint_payload(
    *,
    args: argparse.Namespace,
    trainer: TD3Trainer,
    cfg: ExperimentConfig,
    target_switch_cfg,
    metrics_dict: dict[str, Any],
    summary: dict[str, Any] | None,
    timestamp: str,
    console_log: Path,
    best_window: dict[str, Any] | None,
    checkpoint_kind: str,
) -> dict[str, Any]:
    config_payload = cfg.to_dict()
    config_payload['target_switch'] = target_switch_cfg.to_dict()
    return {
        'model_type': args.model,
        'state_dict': trainer.actor.state_dict(),
        'critic1_state_dict': trainer.critic1.state_dict(),
        'critic2_state_dict': trainer.critic2.state_dict(),
        'actor_target_state_dict': trainer.actor_target.state_dict(),
        'critic1_target_state_dict': trainer.critic1_target.state_dict(),
        'critic2_target_state_dict': trainer.critic2_target.state_dict(),
        'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
        'metrics': metrics_dict,
        'summary': summary,
        'config': config_payload,
        'target_switch_config': target_switch_cfg.to_dict(),
        'best_window': best_window,
        'source_checkpoint': str(args.checkpoint),
        'current_step': trainer.total_steps,
        'current_episode': trainer.metrics.episodes,
        'checkpoint_kind': checkpoint_kind,
        'finished_at': timestamp,
        'target_switch_level': args.target_switch_level,
        'init_checkpoint': str(args.checkpoint),
        'console_log': str(console_log),
        'args': _jsonable_args(args),
    }


def _save_target_switch_checkpoint(path: Path, **kwargs) -> Path:
    return save_checkpoint(path, _checkpoint_payload(**kwargs))


def _summarize_episode_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record['outcome']] = outcomes.get(record['outcome'], 0) + 1
    return {
        'episodes': len(records),
        'outcomes': outcomes,
        'success_count': outcomes.get('goal', 0),
        'success_rate': outcomes.get('goal', 0) / len(records) if records else 0.0,
        'avg_post_switch_steps': float(np.mean([r.get('post_switch_steps', 0) or 0 for r in records])) if records else 0.0,
        'avg_final_to_new_distance': _mean_optional(records, 'final_to_new_distance'),
        'avg_switch_alignment_reward': float(np.mean([r.get('switch_alignment_reward_mean', 0.0) or 0.0 for r in records]))
        if records
        else 0.0,
        'avg_ceiling_penalty': float(np.mean([r.get('ceiling_penalty_mean', 0.0) or 0.0 for r in records]))
        if records
        else 0.0,
        'records': records,
    }


def _run_training(args: argparse.Namespace, run_dir: Path, timestamp: str, console_log: Path) -> None:
    cfg = ExperimentConfig()
    checkpoint_payload = load_checkpoint(args.checkpoint)
    _apply_checkpoint_config(cfg, checkpoint_payload)
    resolved_device = configure_training_runtime(
        cfg,
        model_type=args.model,
        device=args.device,
        snn_backend=args.snn_backend,
    )
    _apply_model_overrides(cfg, args)
    _apply_target_switch_runtime_overrides(cfg)
    target_switch_cfg = target_switch_config_for_level(args.target_switch_level)
    set_global_seed(args.seed)

    paths = _target_switch_checkpoint_paths(run_dir, args.model, args.target_switch_level)
    metrics_out = run_dir / 'metrics.json'
    summary_out = run_dir / 'summary.json'
    best_window_out = run_dir / 'best_window.json'
    save_json(run_dir / 'args.json', _jsonable_args(args))
    log_prefix = build_log_prefix(args.model, args.target_switch_level)

    env = TargetSwitchTrajectoryEnv(cfg.scenario, cfg.rewards, target_switch_cfg, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    actor = make_actor(cfg, args.model, obs.shape[0], env.action_space.shape[0])
    critic1, critic2 = make_critics(cfg, obs.shape[0], env.action_space.shape[0])
    trainer = _make_trainer(env, actor, critic1, critic2, cfg, args.target_switch_level)
    trainer.warmup_strategy = load_training_state(args.checkpoint, actor, critic1, critic2, trainer, log_prefix)
    if trainer.metrics.reference_source is None:
        trainer.set_bc_reference_actor(deepcopy(actor), source='target_switch_init')

    records: list[dict[str, Any]] = []
    best_window: dict[str, Any] | None = None

    def episode_callback(record: dict[str, Any]) -> None:
        nonlocal best_window
        switch_summary = env.last_episode_summary or {}
        merged = {
            'episode': record['episode'],
            'total_steps': record['total_steps'],
            'return': record['return'],
            'length': record['length'],
            'outcome': record['outcome'],
            'switch_step': switch_summary.get('switch_step'),
            'switch_position': switch_summary.get('switch_position'),
            'old_goal': switch_summary.get('old_goal'),
            'new_goal': switch_summary.get('new_goal'),
            'post_switch_steps': switch_summary.get('post_switch_steps'),
            'pre_switch_done': switch_summary.get('pre_switch_done'),
            'boundary_reason': switch_summary.get('boundary_reason'),
            'final_to_new_distance': switch_summary.get('final_to_new_distance'),
            'switch_alignment_reward_mean': switch_summary.get('switch_alignment_reward_mean', 0.0),
            'ceiling_penalty_mean': switch_summary.get('ceiling_penalty_mean', 0.0),
            'switch_to_new_distance': switch_summary.get('switch_to_new_distance'),
            'min_to_new_distance': switch_summary.get('min_to_new_distance'),
            'distance_reduction': switch_summary.get('distance_reduction'),
        }
        records.append(merged)
        print(
            f"{log_prefix} ep={merged['episode']} switch_step={merged['switch_step']} "
            f"post_switch_steps={merged['post_switch_steps']} outcome={merged['outcome']} "
            f"final_to_new_distance={merged['final_to_new_distance']} "
            f"boundary_reason={merged['boundary_reason']} "
            f"switch_alignment_reward_mean={merged['switch_alignment_reward_mean']:.4f} "
            f"ceiling_penalty_mean={merged['ceiling_penalty_mean']:.4f}"
        )
        if args.summary_every_episodes > 0 and len(records) % args.summary_every_episodes == 0:
            window = _target_switch_window_stats(records[-args.summary_every_episodes :])
            if _is_better_window(window, best_window):
                best_window = dict(window)
                best_window['checkpoint'] = str(paths['best'])
                best_metrics = trainer.metrics.to_dict()
                best_metrics['best_window'] = best_window
                save_json(best_window_out, best_window)
                _save_target_switch_checkpoint(
                    paths['best'],
                    args=args,
                    trainer=trainer,
                    cfg=cfg,
                    target_switch_cfg=target_switch_cfg,
                    metrics_dict=best_metrics,
                    summary=None,
                    timestamp=timestamp,
                    console_log=console_log,
                    best_window=best_window,
                    checkpoint_kind='best',
                )
                print(
                    f"{log_prefix} saved new best window ep={best_window['episode_start']}-"
                    f"{best_window['episode_end']} goal_rate={best_window['goal_rate']:.3f} checkpoint={paths['best']}"
                )

    print(
        f'{log_prefix} start target-switch training level={args.target_switch_level} '
        f'timesteps={args.timesteps} checkpoint={args.checkpoint} device={resolved_device}'
    )
    print(f'{log_prefix} early_stop=disabled fixed_timesteps={args.timesteps}')
    metrics = trainer.train(
        args.timesteps,
        log_interval=max(100, args.timesteps // 10),
        verbose=True,
        summary_every_episodes=args.summary_every_episodes,
        episode_callback=episode_callback,
        log_prefix=log_prefix,
    )
    metrics_dict = metrics.to_dict()
    metrics_dict.update(
        {
            'target_switch_level': args.target_switch_level,
            'target_switch_config': target_switch_cfg.to_dict(),
            'checkpoint': str(args.checkpoint),
            'best_checkpoint': str(paths['best']) if best_window is not None else None,
            'final_checkpoint': str(paths['final']),
            'best_window': best_window,
            'device': resolved_device,
            'snn_backend': cfg.training.snn_backend if args.model == 'snn' else None,
        }
    )
    summary = _summarize_episode_records(records)
    summary.update(
        {
            'best_checkpoint': str(paths['best']) if best_window is not None else None,
            'final_checkpoint': str(paths['final']),
            'best_window': best_window,
            'best_goal_rate': best_window.get('goal_rate') if best_window is not None else None,
            'best_episode_start': best_window.get('episode_start') if best_window is not None else None,
            'best_episode_end': best_window.get('episode_end') if best_window is not None else None,
        }
    )
    save_json(metrics_out, metrics_dict)
    save_json(summary_out, summary)
    final_payload = _checkpoint_payload(
        args=args,
        trainer=trainer,
        cfg=cfg,
        target_switch_cfg=target_switch_cfg,
        metrics_dict=metrics_dict,
        summary=summary,
        timestamp=timestamp,
        console_log=console_log,
        best_window=best_window,
        checkpoint_kind='final',
    )
    save_checkpoint(paths['compat'], final_payload)
    save_checkpoint(paths['final'], final_payload)
    print(f'{log_prefix} saved final checkpoint to {paths["final"]}')
    print(f'{log_prefix} saved compatibility checkpoint to {paths["compat"]}')
    print(f'{log_prefix} saved metrics to {metrics_out}')
    print(f'{log_prefix} saved summary to {summary_out}')


def main() -> None:
    args = build_parser().parse_args()
    timestamp = now_timestamp()
    run_dir = ensure_dir(args.output / f'{args.target_switch_level}_{timestamp}')
    console_log = run_dir / 'console.log'
    with console_log.open('w', encoding='utf-8') as log_file:
        stdout_tee = _TeeStream(sys.stdout, log_file)
        stderr_tee = _TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f'[target_switch] run_dir={run_dir}')
            _run_training(args, run_dir, timestamp, console_log)


if __name__ == '__main__':
    main()
