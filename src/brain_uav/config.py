"""Central configuration definitions for the whole project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from pathlib import Path


@dataclass(slots=True)
class ScenarioConfig:
    """Environment-side parameters with distance values expressed in km."""

    dt: float = 1.0
    speed: float = 2.5
    gamma_max: float = 0.6
    delta_gamma_max: float = 0.14
    delta_psi_max: float = 0.2
    target_distance: float = 1750.0
    curriculum_distance_ratios: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            'easy': (0.55, 0.70),
            'easy_two_zone': (0.70, 0.85),
            'medium': (0.80, 0.95),
            'hard': (0.90, 1.10),
            'benchmark': (0.90, 1.10),
        }
    )
    world_xy_margin_ratio: float = 0.75
    goal_radius: float = 5.0
    goal_radius_curriculum_enabled: bool = False
    goal_radius_curriculum: dict[str, float] = field(
        default_factory=lambda: {
            'easy': 10.0,
            'easy_two_zone': 8.0,
            'medium': 6.5,
            'hard': 5.0,
            'benchmark': 5.0,
        }
    )
    world_xy: float | None = None
    world_z_min: float = 0.1
    world_z_max: float | None = None
    max_steps: int = 1000
    min_no_fly_zones: int = 1
    max_no_fly_zones: int = 3
    no_fly_radius_range: tuple[float, float] = (200.0, 250.0)
    no_fly_radius_curriculum: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            'easy': (120.0, 160.0),
            'easy_two_zone': (150.0, 190.0),
            'medium': (180.0, 220.0),
            'hard': (200.0, 250.0),
            'benchmark': (200.0, 250.0),
        }
    )
    warning_distance: float = 100.0
    boundary_warning_distance: float = 10.0
    ground_warning_height: float = 4.0
    descent_penalty_height: float = 12.0
    descent_gamma_threshold: float = 0.08
    nearest_zone_count: int = 3
    scenario_max_sampling_attempts: int = 80
    start_zone_clearance: float = 60.0
    zone_overlap_ratio_limit: float = 0.55
    corridor_blocking_margin: float = 40.0
    max_corridor_blockers: int = 2
    max_start_goal_height_gap: float = 11.0
    dual_zone_min_margin: float = 110.0
    easy_two_zone_min_gap: float = 130.0
    easy_two_zone_blocker_probability: float = 0.5

    def __post_init__(self) -> None:
        if self.world_xy is None:
            object.__setattr__(self, 'world_xy', self.target_distance * self.world_xy_margin_ratio)
        if self.world_z_max is None:
            object.__setattr__(self, 'world_z_max', self.world_xy / 3.0)

    def distance_ratio_range_for_level(self, level: str) -> tuple[float, float]:
        try:
            ratio_range = self.curriculum_distance_ratios[level]
        except KeyError as exc:
            raise ValueError(f'Unsupported curriculum level: {level}') from exc
        return float(ratio_range[0]), float(ratio_range[1])

    def distance_range_for_level(self, level: str) -> tuple[float, float]:
        ratio_min, ratio_max = self.distance_ratio_range_for_level(level)
        return self.target_distance * ratio_min, self.target_distance * ratio_max

    def radius_range_for_level(self, level: str) -> tuple[float, float]:
        try:
            radius_range = self.no_fly_radius_curriculum[level]
        except KeyError as exc:
            raise ValueError(f'Unsupported curriculum level: {level}') from exc
        return float(radius_range[0]), float(radius_range[1])


@dataclass(slots=True)
class RewardConfig:
    """Reward weights used by reinforcement learning."""

    progress_weight: float = 2.0
    goal_reward: float = 5000.0
    zone_penalty_weight: float = 450.0
    zone_penalty_cap: float = 1200.0
    zone_secondary_penalty_ratio: float = 0.2
    boundary_soft_penalty_weight: float = 180.0
    boundary_soft_penalty_cap: float = 240.0
    ground_soft_penalty_weight: float = 180.0
    ground_soft_penalty_cap: float = 300.0
    descent_trend_penalty_weight: float = 120.0
    descent_trend_penalty_cap: float = 260.0
    inefficiency_penalty_weight: float = 14.0
    inefficiency_penalty_cap: float = 30.0
    progress_window_size: int = 10
    min_progress_per_window: float = 2.0
    action_delta_gamma_weight: float = 12.0
    action_delta_psi_weight: float = 8.0
    smoothness_weight: float = 1.0
    collision_penalty: float = 24000.0
    step_penalty: float = 2.5
    boundary_penalty: float = 24000.0
    timeout_penalty: float = 5000.0
    breakthrough_reward_distance: float = 120.0
    breakthrough_progress_threshold: float = 2.2
    breakthrough_reward_weight: float = 0.35
    breakthrough_reward_cap: float = 10.0
    terminal_guidance_radius: float = 250.0
    terminal_los_weight: float = 45.0
    terminal_los_penalty_weight: float = 70.0
    terminal_los_reward_cap: float = 80.0
    terminal_tangential_radius: float = 250.0
    terminal_radial_weight: float = 45.0
    terminal_tangential_penalty_weight: float = 60.0
    terminal_tangential_penalty_cap: float = 80.0
    breakthrough_clearance_gate_enabled: bool = False
    descent_three_band_enabled: bool = False
    descent_low_height: float = 20.0
    descent_high_height: float = 60.0
    descent_transition_factor: float = 0.35
    boundary_directional_warning_enabled: bool = False
    boundary_directional_warning_distance: float = 60.0
    boundary_directional_penalty_weight: float = 90.0

    def __post_init__(self) -> None:
        for name in ('breakthrough_clearance_gate_enabled', 'descent_three_band_enabled',
                     'boundary_directional_warning_enabled'):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f'{name} must be a bool.')
        for name in ('descent_low_height', 'descent_high_height', 'descent_transition_factor',
                     'boundary_directional_warning_distance', 'boundary_directional_penalty_weight'):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise ValueError(f'{name} must be a finite number.')
            try:
                value = float(value)
            except OverflowError as exc:
                raise ValueError(f'{name} must be a finite number.') from exc
            if not isfinite(value):
                raise ValueError(f'{name} must be a finite number.')
            setattr(self, name, value)
        if not 0 <= self.descent_low_height < self.descent_high_height:
            raise ValueError('descent heights must satisfy 0 <= low < high.')
        if not 0 <= self.descent_transition_factor <= 1:
            raise ValueError('descent_transition_factor must be in [0, 1].')
        if self.boundary_directional_warning_distance <= 0:
            raise ValueError('boundary_directional_warning_distance must be positive.')
        if self.boundary_directional_penalty_weight < 0:
            raise ValueError('boundary_directional_penalty_weight must be nonnegative.')
        try:
            ratio = float(self.zone_secondary_penalty_ratio)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                'zone_secondary_penalty_ratio must be finite and in [0, 1].'
            ) from exc
        if not isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                'zone_secondary_penalty_ratio must be finite and in [0, 1].'
            )
        self.zone_secondary_penalty_ratio = ratio

    def validate_scenario(self, scenario: ScenarioConfig) -> None:
        if self.boundary_directional_warning_enabled and not (
            self.boundary_directional_warning_distance > scenario.boundary_warning_distance
        ):
            raise ValueError('Directional boundary warning requires a distance above boundary_warning_distance.')
        if self.descent_three_band_enabled and not (
            isfinite(scenario.world_z_min)
            and scenario.world_z_min < self.descent_low_height < self.descent_high_height
        ):
            raise ValueError('Three-band descent requires world_z_min < low < high.')


def reward_config_from_snapshot(snapshot: dict) -> RewardConfig:
    """Only optional P1/P2/P3 reward fields may be absent in old artifacts."""
    added = ('breakthrough_clearance_gate_enabled', 'descent_three_band_enabled',
             'descent_low_height', 'descent_high_height', 'descent_transition_factor',
             'boundary_directional_warning_enabled', 'boundary_directional_warning_distance',
             'boundary_directional_penalty_weight')
    names = {item.name for item in fields(RewardConfig)}
    if not isinstance(snapshot, dict) or set(snapshot) - names or (names - set(snapshot)) - set(added):
        raise ValueError('RewardConfig snapshot has missing or unknown fields.')
    effective = dict(snapshot)
    defaults = RewardConfig()
    for name in added:
        effective.setdefault(name, getattr(defaults, name))
    restored = RewardConfig(**effective)
    if asdict(restored) != effective:
        raise ValueError('RewardConfig snapshot does not round-trip.')
    return restored


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
    replay_size: int = 500_000
    warmup_steps: int = 1280
    actor_freeze_steps: int = 25_000
    success_sample_bias: float = 4.0
    success_replay_min_zone_clearance: float = 30.0
    success_primary_min_zone_clearance: float = 80.0
    success_min_zone_clearance: float = 30.0
    near_goal_sample_bias: float = 2.0
    near_goal_radius: float = 250.0
    success_replay_fraction: float = 0.25
    success_batch_fraction: float = 0.25
    actor_grad_clip_norm: float | None = 1.0
    actor_rl_scale_alpha: float = 2.5
    terminal_geo_regularization_enabled: bool = True
    terminal_geo_radius: float = 250.0
    terminal_geo_lambda: float = 3000.0
    terminal_geo_safe_clearance: float = 40.0
    noise_decay_fraction: float = 0.5
    exploration_noise_final: float = 0.005
    policy_noise_final: float = 0.006
    noise_clip_final: float = 0.012
    bc_epochs: int = 10
    snn_time_window: int = 4
    snn_backend: str = 'torch'
    hidden_dim: int = 128
    critic_grad_clip_norm: float | None = None
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
