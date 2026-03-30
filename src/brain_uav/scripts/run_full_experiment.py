from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ExperimentConfig
from ..scenarios import build_benchmark_scenarios
from ..scripts.evaluate import evaluate_policy
from ..scripts.generate_dataset import main as _unused
from ..scripts.common import make_actor
from ..scripts.profile_models import main as _unused2
from ..scripts.train_bc import main as _unused3
from ..scripts.train_td3 import main as _unused4
from ..utils.io import load_checkpoint, save_checkpoint, save_json
from ..utils.seeding import set_global_seed
from ..trainers import train_behavior_cloning
from ..scripts.common import make_actor, make_critics, make_env
from ..trainers import TD3Trainer
from .profile_models import describe_ann, describe_snn
import numpy as np
from ..baselines import AStarPlanner, ArtificialPotentialFieldPlanner, HeuristicPlanner
from ..envs import StaticNoFlyTrajectoryEnv


def generate_dataset(dataset_path: Path, episodes: int, seed: int) -> dict:
    cfg = ExperimentConfig()
    set_global_seed(seed)
    env = StaticNoFlyTrajectoryEnv(cfg.scenario, cfg.rewards, seed=seed)
    planners = [HeuristicPlanner(env), ArtificialPotentialFieldPlanner(env), AStarPlanner(env)]
    observations = []
    actions = []
    planner_tags = []
    for episode in range(episodes):
        planner = planners[episode % len(planners)]
        rollout = planner.rollout()
        observations.extend(obs for obs, _ in rollout)
        actions.extend(action for _, action in rollout)
        planner_tags.extend([planner.__class__.__name__] * len(rollout))
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        planner_tags=np.array(planner_tags),
    )
    return {'samples': len(observations), 'episodes': episodes, 'dataset': str(dataset_path)}


def train_bc_model(model: str, dataset: Path, epochs: int, output_dir: Path) -> tuple[Path, dict]:
    cfg = ExperimentConfig()
    data = np.load(dataset)
    actor = make_actor(cfg, model, data['observations'].shape[1], data['actions'].shape[1])
    history = train_behavior_cloning(
        actor, dataset, epochs=epochs, batch_size=cfg.training.batch_size, lr=cfg.training.actor_lr, device=cfg.training.device
    )
    ckpt = output_dir / f'bc_{model}.pt'
    save_checkpoint(ckpt, {'model_type': model, 'state_dict': actor.state_dict(), 'loss_history': history, 'config': cfg.to_dict()})
    metrics = {'model': model, 'final_loss': history[-1], 'loss_history': history}
    save_json(output_dir / f'bc_{model}_metrics.json', metrics)
    return ckpt, metrics


def train_td3_model(model: str, bc_checkpoint: Path, timesteps: int, seed: int, output_dir: Path) -> tuple[Path, dict]:
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
        device=cfg.training.device,
    )
    metrics = trainer.train(timesteps).to_dict()
    ckpt = output_dir / f'td3_{model}.pt'
    save_checkpoint(ckpt, {'model_type': model, 'state_dict': actor.state_dict(), 'metrics': metrics, 'config': cfg.to_dict()})
    save_json(output_dir / f'td3_{model}_metrics.json', metrics)
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

    dataset_info = generate_dataset(dataset_path, args.dataset_episodes, args.seed)
    bc_snn_ckpt, bc_snn = train_bc_model('snn', dataset_path, args.bc_epochs, output_dir)
    bc_ann_ckpt, bc_ann = train_bc_model('ann', dataset_path, args.bc_epochs, output_dir)
    td3_snn_ckpt, td3_snn = train_td3_model('snn', bc_snn_ckpt, args.td3_timesteps, args.seed, output_dir)
    td3_ann_ckpt, td3_ann = train_td3_model('ann', bc_ann_ckpt, args.td3_timesteps, args.seed, output_dir)

    eval_snn = evaluate_policy(td3_snn_ckpt, 'snn', args.eval_episodes, args.seed, 'benchmark')
    eval_ann = evaluate_policy(td3_ann_ckpt, 'ann', args.eval_episodes, args.seed, 'benchmark')
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
    print(summary)


if __name__ == '__main__':
    main()
