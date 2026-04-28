"""Shared factory helpers for scripts.

脚本层不直接手写模型和环境，而是统一从这里创建，避免各个脚本写重复代码。
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..config import ExperimentConfig
from ..curriculum import normalize_curriculum_mix, parse_curriculum_mix
from ..envs import StaticNoFlyTrajectoryEnv
from ..models import ANNCritic, ANNPolicyActor, SNNPolicyActor
from ..scenarios import build_benchmark_scenarios


DEVICE_CHOICES = ('auto', 'cpu', 'cuda')
SNN_BACKEND_CHOICES = ('torch', 'cupy')


def build_log_prefix(model_type: str, stage: str) -> str:
    return f'[{model_type.upper()} {stage}]'


def resolve_training_device(device: str) -> str:
    if device == 'auto':
        resolved = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False.')
        resolved = 'cuda'
    elif device == 'cpu':
        resolved = 'cpu'
    else:
        raise ValueError(f'Unsupported device: {device}')
    if resolved == 'cuda':
        torch.set_num_threads(2)
    return resolved


def configure_training_runtime(
    cfg: ExperimentConfig,
    *,
    model_type: str,
    device: str,
    snn_backend: str,
) -> str:
    resolved_device = resolve_training_device(device)
    cfg.training.device = resolved_device
    if model_type == 'snn':
        if snn_backend not in SNN_BACKEND_CHOICES:
            raise ValueError(f'Unsupported snn_backend: {snn_backend}')
        if snn_backend == 'cupy' and resolved_device != 'cuda':
            raise RuntimeError('SNN CuPy backend requires device=cuda.')
        cfg.training.snn_backend = snn_backend
    return resolved_device


def make_env(
    cfg: ExperimentConfig,
    seed: int | None = None,
    scenario_suite: str | None = None,
    curriculum_level: str | None = None,
    curriculum_mix: dict[str, float] | str | None = None,
    goal_radius_curriculum_enabled: bool = False,
) -> StaticNoFlyTrajectoryEnv:
    """Build one environment instance.

    - `scenario_suite='benchmark'` 会加载固定测试场景。
    - 否则默认使用课程场景或随机场景。
    """

    fixed_scenarios = None
    mix_payload = None
    scenario_cfg = (
        replace(cfg.scenario, goal_radius_curriculum_enabled=goal_radius_curriculum_enabled)
        if cfg.scenario.goal_radius_curriculum_enabled != goal_radius_curriculum_enabled
        else cfg.scenario
    )
    if scenario_suite == 'benchmark':
        fixed_scenarios = [item.scenario for item in build_benchmark_scenarios()]
    elif curriculum_level is not None:
        if isinstance(curriculum_mix, str) or curriculum_mix is None:
            mix_payload = parse_curriculum_mix(curriculum_mix, fallback_level=curriculum_level)
        else:
            mix_payload = normalize_curriculum_mix(curriculum_mix, fallback_level=curriculum_level)
    return StaticNoFlyTrajectoryEnv(
        scenario_cfg,
        cfg.rewards,
        seed=seed,
        fixed_scenarios=fixed_scenarios,
        curriculum_mix=mix_payload,
    )


def make_actor(cfg: ExperimentConfig, model_type: str, state_dim: int, action_dim: int):
    """Create either the SNN actor or the ANN actor."""

    action_limit = torch.tensor(
        [cfg.scenario.delta_gamma_max, cfg.scenario.delta_psi_max], dtype=torch.float32
    )
    if model_type == 'snn':
        return SNNPolicyActor(
            state_dim,
            action_dim,
            cfg.training.hidden_dim,
            cfg.training.snn_time_window,
            action_limit,
            cfg.scenario,
            backend=cfg.training.snn_backend,
        )
    if model_type == 'ann':
        return ANNPolicyActor(state_dim, action_dim, cfg.training.hidden_dim, action_limit, cfg.scenario)
    raise ValueError(f'Unsupported model_type: {model_type}')


def make_critics(cfg: ExperimentConfig, state_dim: int, action_dim: int):
    """Create the twin critics required by TD3."""

    return (
        ANNCritic(state_dim, action_dim, cfg.training.hidden_dim, cfg.scenario),
        ANNCritic(state_dim, action_dim, cfg.training.hidden_dim, cfg.scenario),
    )
