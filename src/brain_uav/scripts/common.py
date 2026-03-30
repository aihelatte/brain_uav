"""Shared factory helpers for scripts.

脚本层不直接手写模型和环境，而是统一从这里创建，避免各个脚本写重复代码。
"""

from __future__ import annotations

import torch

from ..config import ExperimentConfig
from ..envs import StaticNoFlyTrajectoryEnv
from ..models import ANNCritic, ANNPolicyActor, SNNPolicyActor
from ..scenarios import build_benchmark_scenarios


def make_env(cfg: ExperimentConfig, seed: int | None = None, scenario_suite: str | None = None) -> StaticNoFlyTrajectoryEnv:
    """Build one environment instance.

    如果 scenario_suite='benchmark'，就会加载固定基准场景；
    否则默认使用随机采样场景。
    """

    fixed_scenarios = None
    if scenario_suite == 'benchmark':
        fixed_scenarios = [item.scenario for item in build_benchmark_scenarios()]
    return StaticNoFlyTrajectoryEnv(cfg.scenario, cfg.rewards, seed=seed, fixed_scenarios=fixed_scenarios)


def make_actor(cfg: ExperimentConfig, model_type: str, state_dim: int, action_dim: int):
    """Create either the SNN actor or the ANN actor."""

    action_limit = torch.tensor(
        [cfg.scenario.delta_gamma_max, cfg.scenario.delta_psi_max], dtype=torch.float32
    )
    if model_type == 'snn':
        return SNNPolicyActor(
            state_dim, action_dim, cfg.training.hidden_dim, cfg.training.snn_time_window, action_limit
        )
    if model_type == 'ann':
        return ANNPolicyActor(state_dim, action_dim, cfg.training.hidden_dim, action_limit)
    raise ValueError(f'Unsupported model_type: {model_type}')


def make_critics(cfg: ExperimentConfig, state_dim: int, action_dim: int):
    """Create the twin critics required by TD3."""

    return (
        ANNCritic(state_dim, action_dim, cfg.training.hidden_dim),
        ANNCritic(state_dim, action_dim, cfg.training.hidden_dim),
    )
