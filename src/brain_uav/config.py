from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ScenarioConfig:
    dt: float = 1.0
    speed: float = 25.0
    gamma_max: float = 0.6
    delta_gamma_max: float = 0.12
    delta_psi_max: float = 0.2
    goal_radius: float = 25.0
    world_xy: float = 800.0
    world_z_min: float = 1.0
    world_z_max: float = 400.0
    max_steps: int = 200
    min_no_fly_zones: int = 1
    max_no_fly_zones: int = 3
    no_fly_radius_range: tuple[float, float] = (60.0, 140.0)
    warning_distance: float = 20.0
    nearest_zone_count: int = 3


@dataclass(slots=True)
class RewardConfig:
    progress_weight: float = 1.0
    goal_reward: float = 500.0
    zone_penalty_weight: float = 2.5
    collision_penalty: float = 1000.0
    step_penalty: float = 1.0
    smoothness_weight: float = 2.0
    boundary_penalty: float = 800.0


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 7
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.08
    noise_clip: float = 0.15
    policy_delay: int = 2
    exploration_noise: float = 0.1
    replay_size: int = 100_000
    warmup_steps: int = 256
    bc_epochs: int = 10
    snn_time_window: int = 4
    hidden_dim: int = 128


@dataclass(slots=True)
class ExperimentConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = Path("outputs")
    data_dir: Path = Path("data")

