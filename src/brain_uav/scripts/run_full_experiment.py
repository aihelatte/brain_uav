"""Run the full experiment pipeline end to end.

如果你什么都不想分步操作，直接跑这个脚本就行。
它会依次完成：
1. 生成 BC 数据集
2. 训练 BC 初始化模型
3. 训练 TD3 主模型
4. 做 benchmark 评估
5. 汇总 profile 和最终指标
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..baselines import AStarPlanner, ArtificialPotentialFieldPlanner, HeuristicPlanner
from ..config import ExperimentConfig
from ..envs import StaticNoFlyTrajectoryEnv
from ..scenarios import build_benchmark_scenarios
from ..scripts.common import make_actor, make_critics, make_env
from ..scripts.evaluate import evaluate_policy
from ..scripts.profile_models import describe_ann, describe_snn
from ..trainers import TD3Trainer, train_behavior_cloning
from ..utils.io import load_checkpoint, save_checkpoint, save_json
from ..utils.seeding import set_global_seed


def collect_rollout(planner, env: StaticNoFlyTrajectoryEnv, max_steps: int | None = None):
    obs, _ = env.reset()
    steps = max_steps or env.scenario.max_steps
    samples = []
    outcome = 'timeout'
    for _ in range(steps):
        action = planner.act(obs)
        samples.append((obs.copy(), action.copy()))
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            outcome = info['outcome']
            break
    return samples, outcome


def generate_dataset(dataset_path: Path, episodes: int, seed: int) -> dict:
    cfg = ExperimentConfig()
    set_global_seed(seed)
    env = StaticNoFlyTrajectoryEnv(cfg.scenario, cfg.rewards, seed=seed)
    planners = [HeuristicPlanner(env), ArtificialPotentialFieldPlanner(env), AStarPlanner(env)]
    observations = []
    actions = []
    planner_tags = []
    success_count = 0
    fallback_samples: list[tuple[np.ndarray, np.ndarray, str]] = []
    print(f"[Stage 1/5] Generating BC dataset -> {dataset_path}")
    for episode in range(episodes):
        planner = planners[episode % len(planners)]
        rollout, outcome = collect_rollout(planner, env)
        if outcome == 'goal':
            observations.extend(obs for obs, _ in rollout)
            actions.extend(action for _, action in rollout)
            planner_tags.extend([planner.__class__.__name__] * len(rollout))
            success_count += 1
        elif not fallback_samples:
            fallback_samples = [(obs, action, planner.__class__.__name__) for obs, action in rollout]
        print(
            f"[Dataset] episode {episode + 1}/{episodes} planner={planner.__class__.__name__} "
            f"outcome={outcome} kept_samples={len(observations)}"
        )
    if not observations and fallback_samples:
        observations = [item[0] for item in fallback_samples]
        actions = [item[1] for item in fallback_samples]
        planner_tags = [item[2] for item in fallback_samples]
        print('[Dataset] warning: no successful trajectories found, using one fallback rollout to avoid empty dataset')
    if not observations:
        raise RuntimeError('Dataset generation produced zero samples. Please increase episodes or improve baselines.')
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        planner_tags=np.array(planner_tags),
    )
    return {
        'samples': len(observations),
        'episodes': episodes,
        'successful_episodes': success_count,
        'dataset': str(dataset_path),
    }


def train_bc_model(model: str, dataset: Path, epochs: int, output_dir: Path) -> tuple[Path, dict]:
    print(f"[Stage 2/5] Training BC model={model}")
    cfg = ExperimentConfig()
    data = np.load(dataset)
    actor = make_actor(cfg, model, data['observations'].shape[1], data['actions'].shape[1])
    history = train_behavior_cloning(
        actor,
        dataset,
        epochs=epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.actor_lr,
        device=cfg.training.device,
        verbose=True,
    )
    ckpt = output_dir / f'bc_{model}.pt'
    save_checkpoint(ckpt, {'model_type': model, 'state_dict': actor.state_dict(), 'loss_history': history, 'config': cfg.to_dict()})
    metrics = {'model': model, 'final_loss': history[-1], 'loss_history': history}
    save_json(output_dir / f'bc_{model}_metrics.json', metrics)
    print(f"[BC] saved checkpoint={ckpt}")
    return ckpt, metrics


def train_td3_model(model: str, bc_checkpoint: Path, timesteps: int, seed: int, output_dir: Path) -> tuple[Path, dict]:
    print(f"[Stage 3/5] Training TD3 model={model}")
    cfg = ExperimentConfig()
    set_global_seed(seed)
    env = make_env(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    actor = make_actor(cfg, model, obs.shape[0], env.action_space.shape[0])
    actor.load_state_dict(load_checkpoint(bc_checkpoint)['state_dict'])
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
        actor_freeze_steps=cfg.training.actor_freeze_steps,
        warmup_strategy='policy',
        device=cfg.training.device,
    )
    metrics = trainer.train(timesteps, log_interval=max(100, timesteps // 10), verbose=True).to_dict()
    ckpt = output_dir / f'td3_{model}.pt'
    save_checkpoint(ckpt, {'model_type': model, 'state_dict': actor.state_dict(), 'metrics': metrics, 'config': cfg.to_dict()})
    save_json(output_dir / f'td3_{model}_metrics.json', metrics)
    print(f"[TD3] saved checkpoint={ckpt}")
    return ckpt, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the full research experiment pipeline.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--dataset-episodes', type=int, default=64)
    parser.add_argument('--bc-epochs', type=int, default=8)
    parser.add_argument('--td3-timesteps', type=int, default=5000)
    parser.add_argument('--eval-episodes', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/full_run'))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path('data') / 'bc_dataset_full.npz'

    print('[Run] Full experiment started')
    dataset_info = generate_dataset(dataset_path, args.dataset_episodes, args.seed)
    bc_snn_ckpt, bc_snn = train_bc_model('snn', dataset_path, args.bc_epochs, output_dir)
    bc_ann_ckpt, bc_ann = train_bc_model('ann', dataset_path, args.bc_epochs, output_dir)
    td3_snn_ckpt, td3_snn = train_td3_model('snn', bc_snn_ckpt, args.td3_timesteps, args.seed, output_dir)
    td3_ann_ckpt, td3_ann = train_td3_model('ann', bc_ann_ckpt, args.td3_timesteps, args.seed, output_dir)

    print('[Stage 4/5] Evaluating trained models on benchmark scenarios')
    eval_snn = evaluate_policy(td3_snn_ckpt, 'snn', args.eval_episodes, args.seed, 'benchmark')
    eval_ann = evaluate_policy(td3_ann_ckpt, 'ann', args.eval_episodes, args.seed, 'benchmark')

    print('[Stage 5/5] Profiling model complexity and writing summary')
    snn_dense_macs, snn_params, snn_effective_macs, spike_rate_l1, spike_rate_l2, backend = describe_snn(td3_snn_ckpt)
    ann_macs, ann_params = describe_ann(td3_ann_ckpt)

    summary = {
        'dataset': dataset_info,
        'bc': {'snn': bc_snn, 'ann': bc_ann},
        'training': {'snn': td3_snn, 'ann': td3_ann},
        'evaluation': {'snn': eval_snn, 'ann': eval_ann},
        'profile': {
            'snn_backend': backend,
            'snn_dense_macs': snn_dense_macs,
            'snn_effective_macs': snn_effective_macs,
            'ann_macs': ann_macs,
            'snn_params': snn_params,
            'ann_params': ann_params,
            'spike_rate_l1': spike_rate_l1,
            'spike_rate_l2': spike_rate_l2,
            'effective_mac_reduction_ratio': 1.0 - (snn_effective_macs / ann_macs if ann_macs else 0.0),
        },
        'benchmark_scenarios': [{'name': item.name, 'description': item.description} for item in build_benchmark_scenarios()],
        'acceptance': {
            'inference_under_1s': eval_snn['max_inference_time_ms'] < 1000.0,
            'mac_reduction_over_50pct': (1.0 - (snn_effective_macs / ann_macs if ann_macs else 0.0)) >= 0.5,
        },
    }
    save_json(output_dir / 'summary.json', summary)
    print(f"[Run] Summary written to {output_dir / 'summary.json'}")
    print(summary)


if __name__ == '__main__':
    main()

