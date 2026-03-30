from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ExperimentConfig
from ..scripts.common import make_actor, make_critics, make_env
from ..trainers import TD3Trainer
from ..utils.io import load_checkpoint, save_checkpoint, save_json
from ..utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description='Train TD3 on the static no-fly-zone task.')
    parser.add_argument('--model', choices=['snn', 'ann'], default='snn')
    parser.add_argument('--timesteps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--bc-checkpoint', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--metrics-out', type=Path, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    set_global_seed(args.seed)
    env = make_env(cfg, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    actor = make_actor(cfg, args.model, obs.shape[0], env.action_space.shape[0])
    if args.bc_checkpoint:
        actor.load_state_dict(load_checkpoint(args.bc_checkpoint)['state_dict'])
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
    metrics = trainer.train(args.timesteps)
    output = args.output or Path(f'outputs/td3_{args.model}.pt')
    metrics_out = args.metrics_out or Path(f'outputs/td3_{args.model}_metrics.json')
    save_checkpoint(
        output,
        {
            'model_type': args.model,
            'state_dict': actor.state_dict(),
            'metrics': metrics.to_dict(),
            'config': cfg.to_dict(),
        },
    )
    save_json(metrics_out, metrics.to_dict())
    print(f'Saved TD3 checkpoint to {output}')
    print(f"Episodes: {metrics.episodes}, Steps: {metrics.steps}, Critic loss: {metrics.critic_loss:.4f}")


if __name__ == '__main__':
    main()
