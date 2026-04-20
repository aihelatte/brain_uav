"""Central configuration definitions for the whole project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ScenarioConfig:
    """Environment-side parameters.

    Unit convention inside the environment:
    - distance: km
    - time: s
    - speed: km/s
    - angles: unchanged from the existing code path

    The XY range is a long-range planning box. The Z upper bound is an
    independent altitude scale and is not meant to grow with ``world_xy``.
    """

    dt: float = 1.0  # s per environment step
    speed: float = 2.72  # km/s, about Mach 8 at the intended mission scale
    max_time_s: float = 1010.0  # s, maximum episode flight time
    gamma_max: float = 0.6
    delta_gamma_max: float = 0.14
    delta_psi_max: float = 0.2
    goal_radius: float = 5.0  # km, fixed training success radius
    easy_goal_radius: float = 10.0  # km, curriculum-only relaxed capture radius
    easy_two_zone_goal_radius: float = 8.0  # km, curriculum-only relaxed capture radius
    medium_goal_radius: float = 6.0  # km, curriculum-only relaxed capture radius
    hard_goal_radius: float = 5.0  # km, formal/default capture radius
    world_xy: float = 1600.0  # km, boundary is [-world_xy, world_xy] on X/Y
    world_z_min: float = 0.0  # km, ground boundary
    world_z_max: float = 400.0  # km, true environment altitude ceiling; render axes may extend higher
    max_steps: int | None = None  # defaults to int(max_time_s / dt); explicit values are preserved
    min_no_fly_zones: int = 1
    max_no_fly_zones: int = 3
    no_fly_radius_range: tuple[float, float] = (60.0, 140.0)  # km
    easy_start_goal_distance_range: tuple[float, float] = (1200.0, 1700.0)  # km
    easy_two_zone_start_goal_distance_range: tuple[float, float] = (1400.0, 1900.0)  # km
    medium_start_goal_distance_range: tuple[float, float] = (1600.0, 2100.0)  # km
    hard_start_goal_distance_range: tuple[float, float] = (1800.0, 2300.0)  # km
    easy_no_fly_radius_range: tuple[float, float] = (80.0, 140.0)  # km
    easy_two_zone_no_fly_radius_range: tuple[float, float] = (100.0, 170.0)  # km
    medium_no_fly_radius_range: tuple[float, float] = (140.0, 220.0)  # km
    hard_no_fly_radius_range: tuple[float, float] = (200.0, 250.0)  # km
    start_z_range: tuple[float, float] = (100.0, 180.0)  # km
    goal_z_range: tuple[float, float] = (95.0, 205.0)  # km
    warning_distance: float = 100.0  # km around no-fly zones
    boundary_warning_distance: float = 100.0  # km to X/Y/Z upper boundary
    ground_warning_height: float = 40.0  # km
    descent_penalty_height: float = 120.0  # km
    descent_gamma_threshold: float = 0.08
    nearest_zone_count: int = 3
    scenario_max_sampling_attempts: int = 80
    start_zone_clearance: float = 25.0
    zone_overlap_ratio_limit: float = 0.55
    easy_min_zone_surface_gap: float = 30.0  # km
    easy_min_corridor_warning_gap: float = 60.0  # km, warning zone must stay this far from easy corridor
    easy_two_zone_min_zone_surface_gap: float = 40.0  # km
    medium_min_zone_surface_gap: float = 50.0  # km
    hard_min_zone_surface_gap: float = 60.0  # km
    corridor_blocking_margin: float = 35.0
    max_corridor_blockers: int = 2
    max_start_goal_height_gap: float = 110.0
    dual_zone_min_margin: float = 40.0  # km
    easy_two_zone_min_gap: float = 60.0  # km
    easy_two_zone_blocker_probability: float = 0.5

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError('dt must be positive.')
        if self.max_time_s <= 0.0:
            raise ValueError('max_time_s must be positive.')
        if self.max_steps is None:
            self.max_steps = int(self.max_time_s / self.dt)
        else:
            self.max_steps = int(self.max_steps)
            if self.max_steps <= 0:
                raise ValueError('max_steps must be positive.')
            self.max_time_s = float(self.max_steps) * self.dt


EnvConfig = ScenarioConfig


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
    inefficiency_penalty_weight: float = 14.0
    inefficiency_penalty_cap: float = 30.0
    progress_window_size: int = 10
    min_progress_per_window: float = 20.0
    action_delta_gamma_weight: float = 8.0
    action_delta_psi_weight: float = 3.0
    smoothness_weight: float = 0.15
    collision_penalty: float = 6000.0
    step_penalty: float = 3.0
    boundary_penalty: float = 6000.0
    timeout_penalty: float = 1500.0
    breakthrough_reward_distance: float = 220.0
    breakthrough_progress_threshold: float = 22.0
    breakthrough_reward_weight: float = 0.35
    breakthrough_reward_cap: float = 10.0
    terminal_guidance_radius: float = 100.0
    terminal_progress_weight: float = 4.0
    terminal_z_weight: float = 2.0
    terminal_stall_radius: float = 50.0
    terminal_stall_penalty: float = 5.0
    terminal_los_radius: float = 200.0
    terminal_los_weight: float = 16.0
    terminal_los_penalty_weight: float = 8.0
    terminal_lateral_penalty_weight: float = 0.0
    terminal_miss_truncation_enabled: bool = False


@dataclass(slots=True)
class TrainingConfig:
    """Model and optimizer settings."""

    seed: int = 7
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.015
    noise_clip: float = 0.03
    policy_delay: int = 2
    exploration_noise: float = 0.02
    min_exploration_noise: float | None = None
    exploration_noise_decay_start_fraction: float = 0.5
    exploration_noise_decay_end_fraction: float = 1.0
    replay_size: int = 100_000
    warmup_steps: int = 256
    actor_freeze_steps: int = 50_000
    success_sample_bias: float = 2.5
    near_goal_sample_bias: float = 4.0
    near_goal_sample_radius: float = 150.0
    bc_epochs: int = 10
    snn_time_window: int = 8
    hidden_dim: int = 128
    critic_grad_clip_norm: float | None = 1.0
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
