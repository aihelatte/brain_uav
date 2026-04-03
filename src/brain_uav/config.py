"""Central configuration definitions for the whole project.

读项目时建议先看这个文件，因为环境、模型、训练脚本都会从这里拿参数。
如果你后面要调实验，大多数时候改这里就够了。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ScenarioConfig:
    """Environment-side parameters."""

    dt: float = 1.0
    speed: float = 25.0
    gamma_max: float = 0.6
    delta_gamma_max: float = 0.14
    delta_psi_max: float = 0.2
    goal_radius: float = 45.0
    world_xy: float = 800.0
    world_z_min: float = 1.0
    world_z_max: float = 400.0
    max_steps: int = 200
    min_no_fly_zones: int = 1
    max_no_fly_zones: int = 3
    no_fly_radius_range: tuple[float, float] = (60.0, 140.0)
    warning_distance: float = 100.0
    boundary_warning_distance: float = 100.0
    ground_warning_height: float = 40.0
    descent_penalty_height: float = 120.0
    descent_gamma_threshold: float = 0.08
    nearest_zone_count: int = 3
    scenario_max_sampling_attempts: int = 80
    start_zone_clearance: float = 25.0
    zone_overlap_ratio_limit: float = 0.55
    corridor_blocking_margin: float = 35.0
    max_corridor_blockers: int = 2
    max_start_goal_height_gap: float = 110.0


@dataclass(slots=True)
class RewardConfig:
    """Reward weights used by reinforcement learning."""

    progress_weight: float = 2.4
    goal_reward: float = 3500.0
    zone_penalty_weight: float = 180.0
    zone_penalty_cap: float = 300.0
    boundary_soft_penalty_weight: float = 120.0
    boundary_soft_penalty_cap: float = 160.0
    ground_soft_penalty_weight: float = 60.0
    ground_soft_penalty_cap: float = 80.0
    descent_trend_penalty_weight: float = 35.0
    descent_trend_penalty_cap: float = 60.0
    inefficiency_penalty_weight: float = 12.0
    inefficiency_penalty_cap: float = 24.0
    progress_window_size: int = 8
    min_progress_per_window: float = 18.0
    action_delta_gamma_weight: float = 10.0
    action_delta_psi_weight: float = 4.5
    smoothness_weight: float = 0.15
    collision_penalty: float = 6000.0
    step_penalty: float = 3.0
    boundary_penalty: float = 6000.0
    timeout_penalty: float = 1000.0
    terminal_convergence_distance: float = 140.0
    terminal_convergence_progress_weight: float = 0.18
    terminal_convergence_reward_cap: float = 6.0
    climb_trend_high_altitude_ratio: float = 0.75
    climb_trend_gamma_threshold: float = 0.08
    climb_trend_penalty_weight: float = 6.0
    climb_trend_penalty_cap: float = 6.0


@dataclass(slots=True)
class TrainingConfig:
    """Model and optimizer settings."""

    seed: int = 7
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.015
    noise_clip: float = 0.03
    policy_delay: int = 2
    exploration_noise: float = 0.02
    replay_size: int = 100_000
    warmup_steps: int = 256
    actor_freeze_steps: int = 5_000
    success_sample_bias: float = 2.5
    bc_epochs: int = 10
    snn_time_window: int = 4
    hidden_dim: int = 128
    device: str = 'cpu'


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level config container."""

    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = Path('outputs')
    data_dir: Path = Path('data')

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload['output_dir'] = str(self.output_dir)
        payload['data_dir'] = str(self.data_dir)
        return payload
