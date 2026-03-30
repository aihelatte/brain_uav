from __future__ import annotations

import torch

from ..config import ExperimentConfig
from ..envs import StaticNoFlyTrajectoryEnv
from ..models import ANNCritic, ANNPolicyActor, SNNPolicyActor


def make_env(cfg: ExperimentConfig, seed: int | None = None) -> StaticNoFlyTrajectoryEnv:
    return StaticNoFlyTrajectoryEnv(cfg.scenario, cfg.rewards, seed=seed)


def make_actor(cfg: ExperimentConfig, model_type: str, state_dim: int, action_dim: int):
    action_limit = torch.tensor(
        [cfg.scenario.delta_gamma_max, cfg.scenario.delta_psi_max], dtype=torch.float32
    )
    if model_type == "snn":
        return SNNPolicyActor(
            state_dim, action_dim, cfg.training.hidden_dim, cfg.training.snn_time_window, action_limit
        )
    if model_type == "ann":
        return ANNPolicyActor(state_dim, action_dim, cfg.training.hidden_dim, action_limit)
    raise ValueError(f"Unsupported model_type: {model_type}")


def make_critics(cfg: ExperimentConfig, state_dim: int, action_dim: int):
    return (
        ANNCritic(state_dim, action_dim, cfg.training.hidden_dim),
        ANNCritic(state_dim, action_dim, cfg.training.hidden_dim),
    )

