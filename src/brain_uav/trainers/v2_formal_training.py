"""Formal single-stage V2 ANN/SNN-actor TD3 training orchestration.

This module composes the already validated V2 environment, replay, dynamic-set
actor, ANN critics, and TD3 update engine. It deliberately contains no legacy
flat-observation path, benchmark logic, or replay checkpointing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
import json
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import V2ScenarioGenerator, V2StaticNoFlyTrajectoryEnv
from brain_uav.models import V2ANNCritic, V2ANNPolicyActor, V2SNNPolicyActor
from brain_uav.models.zone_set_encoder import ZoneSetEncoderConfig
from brain_uav.observations import V2Observation, V2ObservationScales
from brain_uav.trainers.v2_bc import (
    V2_BC_CHECKPOINT_FORMAT,
    V2_BC_CHECKPOINT_VERSION,
    V2_SNN_BC_CHECKPOINT_FORMAT,
    V2_SNN_BC_CHECKPOINT_VERSION,
    load_v2_bc_actor_checkpoint,
    load_v2_snn_bc_actor_checkpoint,
)
from brain_uav.utils.seeding import set_global_seed
from brain_uav.v2_curriculum import (
    V2CurriculumSelector,
    V2NoiseSchedule,
    V2_TD3_STAGES,
    default_v2_curriculum_mix,
    derive_v2_component_seed,
    normalize_v2_curriculum_mix,
    v2_bc_lambda,
    v2_stage_defaults,
)

from .v2_replay_buffer import V2ReplayBuffer
from .v2_reporting import V2ExperimentReporter
from .v2_td3 import (
    V2_TD3_CHECKPOINT_FORMAT,
    V2_TD3_CHECKPOINT_VERSION,
    V2_SNN_TD3_CHECKPOINT_FORMAT,
    V2_SNN_TD3_CHECKPOINT_VERSION,
    V2TD3UpdateEngine,
    V2TD3UpdateMetrics,
)
from .v2_validation import (
    V2ValidationResult,
    scenario_config_from_snapshot,
    scenario_config_snapshot,
)


V2_FORMAL_CHECKPOINT_FORMAT = 'v2_formal_ann_td3_stage'
V2_FORMAL_CHECKPOINT_VERSION = 1
V2_SNN_FORMAL_CHECKPOINT_FORMAT = 'v2_formal_snn_td3_stage'
V2_SNN_FORMAL_CHECKPOINT_VERSION = 1
_FORMAL_CHECKPOINT_FIELDS = {
    'format',
    'format_version',
    'status',
    'stage',
    'passed_validation',
    'engine_checkpoint',
    'formal_config',
    'scenario_config',
    'reward_config',
    'uav_collision_radius',
    'seed_manifest',
    'bc_schedule',
    'training_result',
    'validation_pool',
    'initialization_source',
}
_SNN_FORMAL_CHECKPOINT_FIELDS = _FORMAL_CHECKPOINT_FIELDS | {'model_type'}
_PREVIOUS_STAGE = {'medium': 'easy', 'hard': 'medium'}
_OUTCOMES = ('goal', 'ground', 'boundary', 'collision', 'timeout')
V2_BC_SCHEDULE_METADATA = {
    'kind': 'stage_local_piecewise_constant',
    'boundaries': [0, 75_000, 150_000, 250_000],
    'values': [500.0, 150.0, 30.0, 5.0],
}


def _nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a non-negative integer.')
    return value


def _positive_int(value: Any, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return result


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite.')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite.') from exc
    if not isfinite(result):
        raise ValueError(f'{name} must be finite.')
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f'{name} must be greater than zero.')
    return result


def _nonnegative_float(value: Any, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f'{name} must be non-negative.')
    return result


def _fraction(value: Any, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f'{name} must be in [0, 1].')
    return result


def _optional_positive(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name=name)


def _strict_json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Formal training metadata must be strict JSON.') from exc


@dataclass(frozen=True, slots=True)
class V2FormalTrainingConfig:
    stage: str
    seed: int = 7
    max_steps: int | None = None
    curriculum_mix: Mapping[str, float] | None = None
    replay_capacity: int = 500_000
    zone_storage_capacity: int = 6
    batch_size: int = 64
    success_sample_bias: float = 1.0
    near_goal_sample_bias: float = 2.0
    success_replay_fraction: float = 0.25
    success_batch_fraction: float = 0.25
    warmup_steps: int = 1_280
    warmup_strategy: str = 'policy_with_noise'
    actor_freeze_steps: int = 25_000
    actor_lr: float | None = None
    critic_lr: float | None = None
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    actor_grad_clip_norm: float | None = 1.0
    critic_grad_clip_norm: float | None = 1.0
    actor_rl_scale_alpha: float = 2.5
    terminal_geo_regularization_enabled: bool = True
    terminal_geo_radius: float = 250.0
    terminal_geo_lambda: float = 3000.0
    terminal_geo_safe_clearance: float = 40.0
    near_goal_radius: float = 250.0
    noise_schedule: V2NoiseSchedule = field(default_factory=V2NoiseSchedule)
    window_episode_count: int = 15
    max_failures_per_window: int = 1
    consecutive_qualified_windows: int = 4
    early_stop_min_steps: int = 125_000
    validation_max_failures: int = 6

    def __post_init__(self) -> None:
        if self.stage not in V2_TD3_STAGES:
            raise ValueError('stage must be easy, medium, or hard.')
        seed = _nonnegative_int(self.seed, name='seed')
        default_steps, default_actor_lr, default_critic_lr = v2_stage_defaults(self.stage)
        max_steps = default_steps if self.max_steps is None else _positive_int(self.max_steps, name='max_steps')
        mix = (
            default_v2_curriculum_mix(self.stage)
            if self.curriculum_mix is None
            else normalize_v2_curriculum_mix(self.curriculum_mix, stage=self.stage)
        )
        actor_lr = default_actor_lr if self.actor_lr is None else _positive_float(self.actor_lr, name='actor_lr')
        critic_lr = default_critic_lr if self.critic_lr is None else _positive_float(self.critic_lr, name='critic_lr')
        replay_capacity = _positive_int(self.replay_capacity, name='replay_capacity')
        zone_capacity = _positive_int(self.zone_storage_capacity, name='zone_storage_capacity')
        batch_size = _positive_int(self.batch_size, name='batch_size')
        if batch_size > replay_capacity:
            raise ValueError('batch_size must not exceed replay_capacity.')
        success_bias = _finite(self.success_sample_bias, name='success_sample_bias')
        near_goal_bias = _finite(self.near_goal_sample_bias, name='near_goal_sample_bias')
        if success_bias < 1.0 or near_goal_bias < 1.0:
            raise ValueError('Replay sampling biases must be at least 1.')
        success_replay_fraction = _fraction(self.success_replay_fraction, name='success_replay_fraction')
        success_batch_fraction = _fraction(self.success_batch_fraction, name='success_batch_fraction')
        warmup_steps = _nonnegative_int(self.warmup_steps, name='warmup_steps')
        if self.warmup_strategy != 'policy_with_noise':
            raise ValueError('warmup_strategy must be policy_with_noise.')
        actor_freeze_steps = _nonnegative_int(self.actor_freeze_steps, name='actor_freeze_steps')
        gamma = _fraction(self.gamma, name='gamma')
        tau = _positive_float(self.tau, name='tau')
        if tau > 1.0:
            raise ValueError('tau must be in (0, 1].')
        policy_delay = _positive_int(self.policy_delay, name='policy_delay')
        actor_clip = _optional_positive(self.actor_grad_clip_norm, name='actor_grad_clip_norm')
        critic_clip = _optional_positive(self.critic_grad_clip_norm, name='critic_grad_clip_norm')
        actor_scale = _positive_float(self.actor_rl_scale_alpha, name='actor_rl_scale_alpha')
        if type(self.terminal_geo_regularization_enabled) is not bool:
            raise ValueError('terminal_geo_regularization_enabled must be bool.')
        terminal_radius = _nonnegative_float(self.terminal_geo_radius, name='terminal_geo_radius')
        terminal_lambda = _nonnegative_float(self.terminal_geo_lambda, name='terminal_geo_lambda')
        terminal_clearance = _nonnegative_float(self.terminal_geo_safe_clearance, name='terminal_geo_safe_clearance')
        near_goal_radius = _nonnegative_float(self.near_goal_radius, name='near_goal_radius')
        if not isinstance(self.noise_schedule, V2NoiseSchedule):
            raise TypeError('noise_schedule must be a V2NoiseSchedule.')
        window_count = _positive_int(self.window_episode_count, name='window_episode_count')
        max_window_failures = _nonnegative_int(self.max_failures_per_window, name='max_failures_per_window')
        if max_window_failures >= window_count:
            raise ValueError('max_failures_per_window must be below window_episode_count.')
        consecutive = _positive_int(self.consecutive_qualified_windows, name='consecutive_qualified_windows')
        minimum_steps = _nonnegative_int(self.early_stop_min_steps, name='early_stop_min_steps')
        validation_failures = _nonnegative_int(self.validation_max_failures, name='validation_max_failures')
        object.__setattr__(self, 'seed', seed)
        object.__setattr__(self, 'max_steps', max_steps)
        object.__setattr__(self, 'curriculum_mix', mix)
        object.__setattr__(self, 'replay_capacity', replay_capacity)
        object.__setattr__(self, 'zone_storage_capacity', zone_capacity)
        object.__setattr__(self, 'batch_size', batch_size)
        object.__setattr__(self, 'success_sample_bias', success_bias)
        object.__setattr__(self, 'near_goal_sample_bias', near_goal_bias)
        object.__setattr__(self, 'success_replay_fraction', success_replay_fraction)
        object.__setattr__(self, 'success_batch_fraction', success_batch_fraction)
        object.__setattr__(self, 'warmup_steps', warmup_steps)
        object.__setattr__(self, 'actor_freeze_steps', actor_freeze_steps)
        object.__setattr__(self, 'actor_lr', actor_lr)
        object.__setattr__(self, 'critic_lr', critic_lr)
        object.__setattr__(self, 'gamma', gamma)
        object.__setattr__(self, 'tau', tau)
        object.__setattr__(self, 'policy_delay', policy_delay)
        object.__setattr__(self, 'actor_grad_clip_norm', actor_clip)
        object.__setattr__(self, 'critic_grad_clip_norm', critic_clip)
        object.__setattr__(self, 'actor_rl_scale_alpha', actor_scale)
        object.__setattr__(self, 'terminal_geo_radius', terminal_radius)
        object.__setattr__(self, 'terminal_geo_lambda', terminal_lambda)
        object.__setattr__(self, 'terminal_geo_safe_clearance', terminal_clearance)
        object.__setattr__(self, 'near_goal_radius', near_goal_radius)
        object.__setattr__(self, 'window_episode_count', window_count)
        object.__setattr__(self, 'max_failures_per_window', max_window_failures)
        object.__setattr__(self, 'consecutive_qualified_windows', consecutive)
        object.__setattr__(self, 'early_stop_min_steps', minimum_steps)
        object.__setattr__(self, 'validation_max_failures', validation_failures)

    def to_dict(self) -> dict[str, Any]:
        return _strict_json_copy(asdict(self))


class V2EarlyStopController:
    """Non-overlapping complete-episode windows and validation gating."""

    def __init__(
        self,
        *,
        window_episode_count: int,
        max_failures_per_window: int,
        consecutive_qualified_windows: int,
        early_stop_min_steps: int,
    ) -> None:
        self.window_episode_count = _positive_int(window_episode_count, name='window_episode_count')
        self.max_failures_per_window = _nonnegative_int(max_failures_per_window, name='max_failures_per_window')
        if self.max_failures_per_window >= self.window_episode_count:
            raise ValueError('max_failures_per_window must be below window_episode_count.')
        self.required_consecutive = _positive_int(consecutive_qualified_windows, name='consecutive_qualified_windows')
        self.early_stop_min_steps = _nonnegative_int(early_stop_min_steps, name='early_stop_min_steps')
        self._outcomes: list[str] = []
        self.completed_windows = 0
        self.consecutive_qualified = 0
        self.candidate_count = 0

    def add_episode(self, outcome: str, *, stage_steps: int) -> dict[str, Any] | None:
        if outcome not in _OUTCOMES:
            raise ValueError('outcome must be a terminal V2 outcome.')
        steps = _nonnegative_int(stage_steps, name='stage_steps')
        self._outcomes.append(outcome)
        if len(self._outcomes) < self.window_episode_count:
            return None
        failures = sum(value != 'goal' for value in self._outcomes)
        qualified = failures <= self.max_failures_per_window
        self.consecutive_qualified = self.consecutive_qualified + 1 if qualified else 0
        self.completed_windows += 1
        candidate = (
            qualified
            and steps >= self.early_stop_min_steps
            and self.consecutive_qualified >= self.required_consecutive
        )
        if candidate:
            self.candidate_count += 1
        row = {
            'window_index': self.completed_windows,
            'episode_count': self.window_episode_count,
            'goal_count': self.window_episode_count - failures,
            'failure_count': failures,
            'qualified': qualified,
            'consecutive_qualified_windows': self.consecutive_qualified,
            'stage_steps': steps,
            'candidate': candidate,
        }
        self._outcomes = []
        return row

    def record_validation(self, passed: bool) -> None:
        if type(passed) is not bool:
            raise ValueError('passed must be bool.')
        if not passed:
            self.consecutive_qualified = 0


@dataclass(slots=True)
class V2FormalTrainingResult:
    stage: str
    status: str
    passed_validation: bool
    stop_reason: str
    stage_steps: int
    global_steps_start: int
    global_steps_end: int
    episodes: list[dict[str, Any]]
    windows: list[dict[str, Any]]
    validation_records: list[dict[str, Any]]
    outcome_counts: dict[str, int]
    curriculum_sample_counts: dict[str, int]
    update_count: int
    candidate_validation_count: int
    replay_size: int
    replay_success_fraction: float
    success_replay_size: int

    @classmethod
    def empty(cls, stage: str, *, global_steps_start: int) -> 'V2FormalTrainingResult':
        if stage not in V2_TD3_STAGES:
            raise ValueError('stage must be easy, medium, or hard.')
        start = _nonnegative_int(global_steps_start, name='global_steps_start')
        return cls(
            stage=stage,
            status='running',
            passed_validation=False,
            stop_reason='running',
            stage_steps=0,
            global_steps_start=start,
            global_steps_end=start,
            episodes=[],
            windows=[],
            validation_records=[],
            outcome_counts={name: 0 for name in _OUTCOMES},
            curriculum_sample_counts={},
            update_count=0,
            candidate_validation_count=0,
            replay_size=0,
            replay_success_fraction=0.0,
            success_replay_size=0,
        )

    def to_dict(self) -> dict[str, Any]:
        return _strict_json_copy(asdict(self))


@dataclass(frozen=True, slots=True)
class _EpisodeTransition:
    observation: V2Observation
    action: np.ndarray
    reward: float
    next_observation: V2Observation
    done: bool
    near_goal: bool
    line_to_goal_safe: bool


@dataclass(frozen=True, slots=True)
class V2BCFormalInitialization:
    """Strict V2 BC actor plus the scenario contract that produced its data."""

    actor: V2ANNPolicyActor | V2SNNPolicyActor
    scenario_config: ScenarioConfig
    uav_collision_radius: float
    model_type: str


@dataclass(frozen=True, slots=True)
class V2PreparedStageInitialization:
    """Validated upstream artifact bound to one stage, seed, and source.

    File-backed sources are stored as normalized absolute paths. In-memory
    checkpoint mappings are deliberately bound by object identity: callers
    must reuse the same mapping object that was strictly validated.
    """

    stage: str
    model_seed: int
    verified_initialization_source: Path | Mapping[str, Any]
    snn_time_window: int | None
    scenario_config: ScenarioConfig
    reward_config: RewardConfig
    uav_collision_radius: float
    model_type: str
    bc_initialization: V2BCFormalInitialization | None
    formal_checkpoint: Mapping[str, Any] | None
    torch_rng_state: torch.Tensor


def _normalize_initialization_source(
    source: str | Path | Mapping[str, Any],
) -> Path | Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    try:
        return Path(source).expanduser().resolve(strict=False)
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError('init_checkpoint must be a path or Mapping.') from exc


def _verified_initialization_source_path(
    prepared: V2PreparedStageInitialization,
) -> str:
    source = prepared.verified_initialization_source
    if isinstance(source, Path):
        return str(source)
    if isinstance(source, Mapping):
        return '<in-memory>'
    raise ValueError('prepared initialization source binding is invalid.')


def load_v2_bc_formal_initialization(
    checkpoint: str | Path,
    *,
    expected_scenario: ScenarioConfig | None = None,
    expected_uav_collision_radius: float | None = None,
    device: str | torch.device = 'cpu',
    model_type: str = 'ann',
) -> V2BCFormalInitialization:
    """Load a BC best checkpoint and bind its complete dataset scenario contract."""

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f'V2 BC checkpoint does not exist: {path}')
    if model_type not in ('ann', 'snn'):
        raise ValueError('model_type must be "ann" or "snn".')
    raw = torch.load(path, map_location='cpu', weights_only=False)
    expected_format = (
        V2_BC_CHECKPOINT_FORMAT
        if model_type == 'ann'
        else V2_SNN_BC_CHECKPOINT_FORMAT
    )
    expected_version = (
        V2_BC_CHECKPOINT_VERSION
        if model_type == 'ann'
        else V2_SNN_BC_CHECKPOINT_VERSION
    )
    if (
        not isinstance(raw, dict)
        or raw.get('format') != expected_format
        or raw.get('format_version') != expected_version
        or raw.get('checkpoint_kind') != 'best'
    ):
        raise ValueError(
            f'easy init_checkpoint must be a strict V2 {model_type.upper()} BC best checkpoint.'
        )

    # The strict BC loader validates the complete checkpoint, including the
    # observation/model/action contract and the dataset provenance snapshot.
    actor = (
        load_v2_bc_actor_checkpoint(path, device=device)
        if model_type == 'ann'
        else load_v2_snn_bc_actor_checkpoint(path, device=device)
    )
    provenance = raw.get('dataset_provenance')
    if not isinstance(provenance, dict):
        raise ValueError('V2 BC checkpoint dataset_provenance is missing or invalid.')
    if 'scenario_config' not in provenance:
        raise ValueError('V2 BC checkpoint dataset_provenance scenario_config is missing.')
    if 'uav_collision_radius' not in provenance:
        raise ValueError(
            'V2 BC checkpoint dataset_provenance uav_collision_radius is missing.'
        )
    scenario = scenario_config_from_snapshot(provenance['scenario_config'])
    radius = _nonnegative_float(
        provenance['uav_collision_radius'],
        name='BC dataset provenance uav_collision_radius',
    )
    if expected_scenario is not None:
        if not isinstance(expected_scenario, ScenarioConfig):
            raise TypeError('expected_scenario must be a ScenarioConfig.')
        if scenario_config_snapshot(scenario) != scenario_config_snapshot(expected_scenario):
            raise ValueError(
                'V2 BC dataset provenance ScenarioConfig is incompatible with '
                'the formal TD3 ScenarioConfig.'
            )
    if expected_uav_collision_radius is not None:
        expected_radius = _nonnegative_float(
            expected_uav_collision_radius,
            name='expected_uav_collision_radius',
        )
        if radius != expected_radius:
            raise ValueError(
                'V2 BC dataset provenance uav_collision_radius is incompatible '
                'with formal TD3.'
            )
    return V2BCFormalInitialization(
        actor=actor,
        scenario_config=scenario,
        uav_collision_radius=radius,
        model_type=model_type,
    )


def prepare_v2_stage_initialization(
    config: V2FormalTrainingConfig,
    *,
    init_checkpoint: str | Path | Mapping[str, Any],
    scenario: ScenarioConfig | None = None,
    rewards: RewardConfig | None = None,
    uav_collision_radius: float | None = None,
    device: str | torch.device = 'cpu',
    model_type: str = 'ann',
    snn_time_window: int = 4,
) -> V2PreparedStageInitialization:
    """Load one upstream artifact and derive the stage runtime contract once."""

    if not isinstance(config, V2FormalTrainingConfig):
        raise TypeError('config must be a V2FormalTrainingConfig.')
    if model_type not in ('ann', 'snn'):
        raise ValueError('model_type must be "ann" or "snn".')
    if type(snn_time_window) is not int or snn_time_window <= 0:
        raise ValueError('snn_time_window must be a positive integer.')
    if scenario is not None and not isinstance(scenario, ScenarioConfig):
        raise TypeError('scenario must be a ScenarioConfig when provided.')
    if rewards is not None and not isinstance(rewards, RewardConfig):
        raise TypeError('rewards must be a RewardConfig when provided.')
    requested_radius = (
        None
        if uav_collision_radius is None
        else _nonnegative_float(
            uav_collision_radius,
            name='uav_collision_radius',
        )
    )
    verified_source = _normalize_initialization_source(init_checkpoint)

    model_seed = derive_v2_component_seed(config.seed, config.stage, 'model')
    set_global_seed(model_seed)
    if config.stage == 'easy':
        if isinstance(verified_source, Mapping):
            raise ValueError('easy init_checkpoint must be a strict V2 BC checkpoint path.')
        bc_initialization = load_v2_bc_formal_initialization(
            verified_source,
            expected_scenario=scenario,
            expected_uav_collision_radius=requested_radius,
            device=device,
            model_type=model_type,
        )
        if model_type == 'snn' and (
            not isinstance(bc_initialization.actor, V2SNNPolicyActor)
            or bc_initialization.actor.time_window != snn_time_window
        ):
            raise ValueError('V2 SNN BC time_window is incompatible with training.')
        effective_scenario = bc_initialization.scenario_config
        effective_rewards = RewardConfig() if rewards is None else rewards
        effective_radius = bc_initialization.uav_collision_radius
        formal_checkpoint = None
        prepared_snn_time_window = (
            bc_initialization.actor.time_window if model_type == 'snn' else None
        )
    else:
        formal_checkpoint = load_v2_formal_checkpoint(
            verified_source,
            require_passed=True,
            next_stage=config.stage,
            expected_model_type=model_type,
        )
        effective_scenario = scenario_config_from_snapshot(
            formal_checkpoint['scenario_config']
        )
        effective_rewards = RewardConfig(**formal_checkpoint['reward_config'])
        effective_radius = _nonnegative_float(
            formal_checkpoint['uav_collision_radius'],
            name='previous stage uav_collision_radius',
        )
        if scenario is not None and (
            scenario_config_snapshot(scenario)
            != scenario_config_snapshot(effective_scenario)
        ):
            raise ValueError('Previous stage ScenarioConfig is incompatible.')
        if rewards is not None and (
            _strict_json_copy(asdict(rewards))
            != _strict_json_copy(asdict(effective_rewards))
        ):
            raise ValueError('Previous stage RewardConfig is incompatible.')
        if requested_radius is not None and requested_radius != effective_radius:
            raise ValueError('Previous stage uav_collision_radius is incompatible.')
        if model_type == 'snn':
            *_, snn_metadata = _architecture_from_engine_payload(
                formal_checkpoint['engine_checkpoint'],
                model_type=model_type,
            )
            if snn_metadata['time_window'] != snn_time_window:
                raise ValueError(
                    'Previous stage SNN time_window is incompatible with training.'
                )
            prepared_snn_time_window = snn_metadata['time_window']
        else:
            prepared_snn_time_window = None
        bc_initialization = None

    return V2PreparedStageInitialization(
        stage=config.stage,
        model_seed=model_seed,
        verified_initialization_source=verified_source,
        snn_time_window=prepared_snn_time_window,
        scenario_config=effective_scenario,
        reward_config=effective_rewards,
        uav_collision_radius=effective_radius,
        model_type=model_type,
        bc_initialization=bc_initialization,
        formal_checkpoint=formal_checkpoint,
        torch_rng_state=torch.get_rng_state().clone(),
    )


@dataclass(frozen=True, slots=True)
class V2StageComponents:
    engine: V2TD3UpdateEngine
    selector: V2CurriculumSelector
    scenario_generators: Mapping[str, V2ScenarioGenerator]
    exploration_rng: np.random.Generator
    seed_manifest: Mapping[str, int]
    initialization_source: Mapping[str, Any]


def _architecture_from_engine_payload(
    engine_payload: Mapping[str, Any],
    *,
    model_type: str,
) -> tuple[
    V2ObservationScales,
    ZoneSetEncoderConfig,
    float,
    int,
    int,
    int,
    torch.Tensor,
    dict[str, Any],
]:
    expected_format = (
        V2_TD3_CHECKPOINT_FORMAT
        if model_type == 'ann'
        else V2_SNN_TD3_CHECKPOINT_FORMAT
    )
    expected_version = (
        V2_TD3_CHECKPOINT_VERSION
        if model_type == 'ann'
        else V2_SNN_TD3_CHECKPOINT_VERSION
    )
    if engine_payload.get('format') != expected_format or engine_payload.get('format_version') != expected_version:
        raise ValueError('Previous stage does not contain a strict V2 TD3 engine checkpoint.')
    architecture = engine_payload.get('architecture')
    if not isinstance(architecture, Mapping):
        raise ValueError('Previous stage engine architecture is missing.')
    try:
        scales = V2ObservationScales(**architecture['scales'])
        encoder = ZoneSetEncoderConfig(**architecture['encoder_config'])
        radius = _nonnegative_float(architecture['uav_radius'], name='uav_radius')
        actor_hidden = _positive_int(architecture['actor_hidden_dim'], name='actor_hidden_dim')
        critic1_hidden = _positive_int(architecture['critic1_hidden_dim'], name='critic1_hidden_dim')
        critic2_hidden = _positive_int(architecture['critic2_hidden_dim'], name='critic2_hidden_dim')
        action_dim = _positive_int(engine_payload['action_dim'], name='action_dim')
        action_high = torch.tensor(engine_payload['action_high'], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Previous stage engine architecture is invalid.') from exc
    if action_high.shape != (action_dim,):
        raise ValueError('Previous stage action range is invalid.')
    snn_metadata: dict[str, Any] = {}
    if model_type == 'snn':
        try:
            snn_metadata = {
                'time_window': _positive_int(
                    architecture['time_window'], name='time_window'
                ),
                'tau': _positive_float(architecture['tau'], name='tau'),
                'surrogate': architecture['surrogate'],
                'backend': architecture['backend'],
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Previous stage SNN architecture is invalid.') from exc
        if (
            architecture.get('model_type') != 'snn'
            or snn_metadata['surrogate'] != 'atan'
            or snn_metadata['backend'] != 'torch'
        ):
            raise ValueError('Previous stage SNN implementation is incompatible.')
    return (
        scales,
        encoder,
        radius,
        actor_hidden,
        critic1_hidden,
        critic2_hidden,
        action_high,
        snn_metadata,
    )


def validate_v2_prepared_stage_initialization(
    prepared: V2PreparedStageInitialization,
    config: V2FormalTrainingConfig,
    *,
    init_checkpoint: str | Path | Mapping[str, Any],
    scenario: ScenarioConfig | None = None,
    rewards: RewardConfig | None = None,
    uav_collision_radius: float | None = None,
    model_type: str = 'ann',
    snn_time_window: int = 4,
) -> None:
    """Validate a cached initialization before any model or RNG state is used."""

    if not isinstance(prepared, V2PreparedStageInitialization):
        raise TypeError(
            'prepared_initialization must be a V2PreparedStageInitialization.'
        )
    if not isinstance(config, V2FormalTrainingConfig):
        raise TypeError('config must be a V2FormalTrainingConfig.')
    if model_type not in ('ann', 'snn'):
        raise ValueError('model_type must be "ann" or "snn".')
    if type(snn_time_window) is not int or snn_time_window <= 0:
        raise ValueError('snn_time_window must be a positive integer.')
    if prepared.stage != config.stage:
        raise ValueError('prepared_initialization stage is incompatible.')
    expected_model_seed = derive_v2_component_seed(
        config.seed, config.stage, 'model'
    )
    if prepared.model_seed != expected_model_seed:
        raise ValueError('prepared_initialization model seed is incompatible.')
    if prepared.model_type != model_type:
        raise ValueError('prepared_initialization model_type is incompatible.')
    expected_snn_time_window = snn_time_window if model_type == 'snn' else None
    if prepared.snn_time_window != expected_snn_time_window:
        raise ValueError('prepared_initialization SNN time_window is incompatible.')

    verified_source = prepared.verified_initialization_source
    requested_source = _normalize_initialization_source(init_checkpoint)
    if isinstance(verified_source, Path):
        source_matches = (
            isinstance(requested_source, Path)
            and requested_source == verified_source
        )
    elif isinstance(verified_source, Mapping):
        source_matches = (
            isinstance(requested_source, Mapping)
            and requested_source is verified_source
        )
    else:
        raise ValueError('prepared initialization source binding is invalid.')
    if not source_matches:
        raise ValueError('prepared_initialization initialization source is incompatible.')

    if scenario is not None:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError('scenario must be a ScenarioConfig when provided.')
        if (
            scenario_config_snapshot(prepared.scenario_config)
            != scenario_config_snapshot(scenario)
        ):
            raise ValueError('prepared_initialization ScenarioConfig is incompatible.')
    if rewards is not None:
        if not isinstance(rewards, RewardConfig):
            raise TypeError('rewards must be a RewardConfig when provided.')
        if (
            _strict_json_copy(asdict(prepared.reward_config))
            != _strict_json_copy(asdict(rewards))
        ):
            raise ValueError('prepared_initialization RewardConfig is incompatible.')
    if uav_collision_radius is not None:
        requested_radius = _nonnegative_float(
            uav_collision_radius,
            name='uav_collision_radius',
        )
        if requested_radius != prepared.uav_collision_radius:
            raise ValueError(
                'prepared_initialization uav_collision_radius is incompatible.'
            )

    if config.stage == 'easy':
        initialization = prepared.bc_initialization
        if initialization is None or prepared.formal_checkpoint is not None:
            raise ValueError('prepared easy initialization is missing the BC actor.')
        if (
            initialization.model_type != model_type
            or scenario_config_snapshot(initialization.scenario_config)
            != scenario_config_snapshot(prepared.scenario_config)
            or initialization.uav_collision_radius
            != prepared.uav_collision_radius
        ):
            raise ValueError('prepared easy initialization contract is incompatible.')
        expected_actor_type = (
            V2ANNPolicyActor if model_type == 'ann' else V2SNNPolicyActor
        )
        if type(initialization.actor) is not expected_actor_type:
            raise ValueError('prepared easy actor model type is incompatible.')
        if model_type == 'snn' and (
            initialization.actor.time_window != snn_time_window
        ):
            raise ValueError('prepared easy SNN time_window is incompatible.')
        return

    wrapper = prepared.formal_checkpoint
    if prepared.bc_initialization is not None or not isinstance(wrapper, Mapping):
        raise ValueError('prepared stage initialization is missing its predecessor.')
    expected_previous_stage = _PREVIOUS_STAGE[config.stage]
    if wrapper.get('stage') != expected_previous_stage:
        raise ValueError(
            'prepared formal checkpoint is not the required direct predecessor.'
        )
    if (
        wrapper.get('status') != 'passed'
        or wrapper.get('passed_validation') is not True
    ):
        raise ValueError(
            'prepared formal checkpoint must remain completed and passed validation.'
        )
    expected_format = (
        V2_FORMAL_CHECKPOINT_FORMAT
        if model_type == 'ann'
        else V2_SNN_FORMAL_CHECKPOINT_FORMAT
    )
    expected_version = (
        V2_FORMAL_CHECKPOINT_VERSION
        if model_type == 'ann'
        else V2_SNN_FORMAL_CHECKPOINT_VERSION
    )
    if (
        wrapper.get('format') != expected_format
        or wrapper.get('format_version') != expected_version
    ):
        raise ValueError('prepared formal checkpoint model type is incompatible.')
    if model_type == 'snn' and wrapper.get('model_type') != 'snn':
        raise ValueError('prepared formal SNN checkpoint model type is incompatible.')
    if (
        wrapper.get('scenario_config')
        != scenario_config_snapshot(prepared.scenario_config)
        or wrapper.get('reward_config')
        != _strict_json_copy(asdict(prepared.reward_config))
        or wrapper.get('uav_collision_radius')
        != prepared.uav_collision_radius
    ):
        raise ValueError('prepared formal checkpoint runtime contract is incompatible.')
    if model_type == 'snn':
        engine_payload = wrapper.get('engine_checkpoint')
        if not isinstance(engine_payload, Mapping):
            raise ValueError('prepared formal SNN engine checkpoint is invalid.')
        *_, snn_metadata = _architecture_from_engine_payload(
            engine_payload,
            model_type='snn',
        )
        if snn_metadata['time_window'] != snn_time_window:
            raise ValueError('prepared formal SNN time_window is incompatible.')


def build_v2_stage_engine(
    scenario: ScenarioConfig | None,
    config: V2FormalTrainingConfig,
    *,
    init_checkpoint: str | Path | Mapping[str, Any],
    rewards: RewardConfig | None = None,
    uav_collision_radius: float | None = None,
    device: str | torch.device = 'cpu',
    model_type: str = 'ann',
    snn_time_window: int = 4,
    prepared_initialization: V2PreparedStageInitialization | None = None,
) -> V2StageComponents:
    if scenario is not None and not isinstance(scenario, ScenarioConfig):
        raise TypeError('scenario must be a ScenarioConfig when provided.')
    if not isinstance(config, V2FormalTrainingConfig):
        raise TypeError('config must be a V2FormalTrainingConfig.')
    if model_type not in ('ann', 'snn'):
        raise ValueError('model_type must be "ann" or "snn".')
    if type(snn_time_window) is not int or snn_time_window <= 0:
        raise ValueError('snn_time_window must be a positive integer.')
    prepared = prepared_initialization
    if prepared is None:
        prepared = prepare_v2_stage_initialization(
            config,
            init_checkpoint=init_checkpoint,
            scenario=scenario,
            rewards=rewards,
            uav_collision_radius=uav_collision_radius,
            device=device,
            model_type=model_type,
            snn_time_window=snn_time_window,
        )
    validate_v2_prepared_stage_initialization(
        prepared,
        config,
        init_checkpoint=init_checkpoint,
        scenario=scenario,
        rewards=rewards,
        uav_collision_radius=uav_collision_radius,
        model_type=model_type,
        snn_time_window=snn_time_window,
    )
    effective_scenario = prepared.scenario_config
    effective_rewards = prepared.reward_config
    radius = prepared.uav_collision_radius
    scenario = effective_scenario
    rewards = effective_rewards
    model_seed = prepared.model_seed
    set_global_seed(model_seed)
    torch.set_rng_state(prepared.torch_rng_state.clone())
    expected_scales = V2ObservationScales(
        scenario.world_xy,
        scenario.world_z_min,
        scenario.world_z_max,
        scenario.gamma_max,
    )
    expected_limit = torch.tensor(
        [scenario.delta_gamma_max, scenario.delta_psi_max], dtype=torch.float32
    )
    previous_engine_payload: Mapping[str, Any] | None = None
    if config.stage == 'easy':
        bc_initialization = prepared.bc_initialization
        if bc_initialization is None:
            raise ValueError('prepared easy initialization is missing the BC actor.')
        bc_actor = bc_initialization.actor
        if (
            bc_actor.scales != expected_scales
            or bc_actor.uav_radius != radius
            or not torch.equal(bc_actor.action_limit.detach().cpu(), expected_limit)
        ):
            raise ValueError('V2 BC checkpoint is incompatible with the training ScenarioConfig.')
        actor = deepcopy(bc_actor)
        encoder_config = actor.encoder_config
        actor_hidden = actor.hidden_dim
        if model_type == 'snn' and (
            not isinstance(actor, V2SNNPolicyActor)
            or actor.time_window != snn_time_window
        ):
            raise ValueError('V2 SNN BC time_window is incompatible with training.')
        critic1_hidden = actor_hidden
        critic2_hidden = actor_hidden
        initialization_source = {
            'kind': 'v2_bc_best',
            'path': _verified_initialization_source_path(prepared),
            'format': (
                V2_BC_CHECKPOINT_FORMAT
                if model_type == 'ann'
                else V2_SNN_BC_CHECKPOINT_FORMAT
            ),
            'format_version': (
                V2_BC_CHECKPOINT_VERSION
                if model_type == 'ann'
                else V2_SNN_BC_CHECKPOINT_VERSION
            ),
            'checkpoint_kind': 'best',
        }
        if model_type == 'snn':
            initialization_source['model_type'] = 'snn'
    else:
        wrapper = prepared.formal_checkpoint
        if wrapper is None:
            raise ValueError('prepared stage initialization is missing its predecessor.')
        if wrapper['scenario_config'] != scenario_config_snapshot(scenario):
            raise ValueError('Previous stage ScenarioConfig is incompatible.')
        if wrapper['uav_collision_radius'] != radius:
            raise ValueError('Previous stage uav_collision_radius is incompatible.')
        if rewards is not None:
            if not isinstance(rewards, RewardConfig):
                raise TypeError('rewards must be a RewardConfig when provided.')
            if wrapper['reward_config'] != _strict_json_copy(asdict(rewards)):
                raise ValueError('Previous stage RewardConfig is incompatible.')
        previous_engine_payload = wrapper['engine_checkpoint']
        (
            scales,
            encoder_config,
            checkpoint_radius,
            actor_hidden,
            critic1_hidden,
            critic2_hidden,
            action_limit,
            snn_metadata,
        ) = _architecture_from_engine_payload(
            previous_engine_payload, model_type=model_type
        )
        if scales != expected_scales or checkpoint_radius != radius or not torch.equal(action_limit, expected_limit):
            raise ValueError('Previous stage checkpoint is incompatible with ScenarioConfig.')
        if model_type == 'ann':
            actor = V2ANNPolicyActor(
                scales,
                int(action_limit.shape[0]),
                actor_hidden,
                action_limit,
                uav_radius=radius,
                encoder_config=encoder_config,
            )
        else:
            if snn_metadata['time_window'] != snn_time_window:
                raise ValueError(
                    'Previous stage SNN time_window is incompatible with training.'
                )
            actor = V2SNNPolicyActor(
                scales,
                int(action_limit.shape[0]),
                actor_hidden,
                action_limit,
                time_window=snn_metadata['time_window'],
                tau=snn_metadata['tau'],
                uav_radius=radius,
                encoder_config=encoder_config,
            )
        V2TD3UpdateEngine._validate_state_dict(
            actor,
            previous_engine_payload['actor_state_dict'],
            name='actor_state_dict',
        )
        actor.load_state_dict(previous_engine_payload['actor_state_dict'], strict=True)
        bc_actor = deepcopy(actor)
        initialization_source = {
            'kind': 'validated_v2_td3_stage',
            'previous_stage': wrapper['stage'],
            'path': _verified_initialization_source_path(prepared),
            'format': (
                V2_FORMAL_CHECKPOINT_FORMAT
                if model_type == 'ann'
                else V2_SNN_FORMAL_CHECKPOINT_FORMAT
            ),
            'format_version': (
                V2_FORMAL_CHECKPOINT_VERSION
                if model_type == 'ann'
                else V2_SNN_FORMAL_CHECKPOINT_VERSION
            ),
            'passed_validation': True,
        }
        if model_type == 'snn':
            initialization_source['model_type'] = 'snn'
    critic1 = V2ANNCritic(
        expected_scales,
        2,
        critic1_hidden,
        uav_radius=radius,
        encoder_config=encoder_config,
    )
    critic2 = V2ANNCritic(
        expected_scales,
        2,
        critic2_hidden,
        uav_radius=radius,
        encoder_config=encoder_config,
    )
    replay_seed = derive_v2_component_seed(config.seed, config.stage, 'replay')
    replay = V2ReplayBuffer(
        config.replay_capacity,
        2,
        config.zone_storage_capacity,
        success_sample_bias=config.success_sample_bias,
        near_goal_sample_bias=config.near_goal_sample_bias,
        success_replay_fraction=config.success_replay_fraction,
        success_batch_fraction=config.success_batch_fraction,
        seed=replay_seed,
    )
    initial_policy_noise = config.noise_schedule.policy_initial
    initial_noise_clip = config.noise_schedule.clip_initial
    engine = V2TD3UpdateEngine(
        actor,
        critic1,
        critic2,
        replay,
        config.actor_lr,
        config.critic_lr,
        config.gamma,
        config.tau,
        initial_policy_noise,
        initial_noise_clip,
        config.policy_delay,
        config.batch_size,
        -expected_limit.numpy(),
        expected_limit.numpy(),
        actor_freeze_steps=config.actor_freeze_steps,
        actor_grad_clip_norm=config.actor_grad_clip_norm,
        critic_grad_clip_norm=config.critic_grad_clip_norm,
        actor_rl_scale_alpha=config.actor_rl_scale_alpha,
        terminal_geo_regularization_enabled=config.terminal_geo_regularization_enabled,
        terminal_geo_radius=config.terminal_geo_radius,
        terminal_geo_lambda=config.terminal_geo_lambda,
        bc_reference_actor=bc_actor,
        device=device,
    )
    if previous_engine_payload is not None:
        engine.load_network_state_dicts(dict(previous_engine_payload))
    curriculum_seed = derive_v2_component_seed(config.seed, config.stage, 'curriculum')
    selector = V2CurriculumSelector(config.curriculum_mix, seed=curriculum_seed)
    generator_seeds = {
        level: derive_v2_component_seed(config.seed, config.stage, f'{level}_generator')
        for level in config.curriculum_mix
    }
    generators = {
        level: V2ScenarioGenerator(scenario, level, seed=generator_seeds[level])
        for level in config.curriculum_mix
    }
    exploration_seed = derive_v2_component_seed(config.seed, config.stage, 'exploration')
    seed_manifest = {
        'base_seed': config.seed,
        'model_seed': model_seed,
        'curriculum_seed': curriculum_seed,
        'replay_seed': replay_seed,
        'exploration_seed': exploration_seed,
        **{f'{level}_generator_seed': value for level, value in generator_seeds.items()},
    }
    return V2StageComponents(
        engine=engine,
        selector=selector,
        scenario_generators=generators,
        exploration_rng=np.random.default_rng(exploration_seed),
        seed_manifest=seed_manifest,
        initialization_source=initialization_source,
    )


class V2FormalStageTrainer:
    """One formal V2 stage; validation is injected and isolated from training."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        rewards: RewardConfig,
        config: V2FormalTrainingConfig,
        engine: V2TD3UpdateEngine,
        *,
        scenario_sources: Mapping[str, Any],
        validation_runner: Callable[
            [V2ANNPolicyActor | V2SNNPolicyActor], V2ValidationResult
        ],
        selector: V2CurriculumSelector | None = None,
        exploration_rng: np.random.Generator | None = None,
        global_steps_start: int = 0,
        uav_collision_radius: float = 0.0,
        reporter: V2ExperimentReporter | None = None,
    ) -> None:
        if not isinstance(scenario, ScenarioConfig) or not isinstance(rewards, RewardConfig):
            raise TypeError('scenario and rewards must use project config classes.')
        if not isinstance(config, V2FormalTrainingConfig):
            raise TypeError('config must be V2FormalTrainingConfig.')
        if not isinstance(engine, V2TD3UpdateEngine):
            raise TypeError('engine must be V2TD3UpdateEngine.')
        if not callable(validation_runner):
            raise TypeError('validation_runner must be callable.')
        if reporter is not None and not isinstance(reporter, V2ExperimentReporter):
            raise TypeError('reporter must be a V2ExperimentReporter when provided.')
        if set(scenario_sources) != set(config.curriculum_mix):
            raise ValueError('scenario_sources must exactly match curriculum_mix levels.')
        for level, source in scenario_sources.items():
            if not callable(getattr(source, 'generate', None)):
                raise TypeError(f'scenario source {level!r} must provide generate().')
        expected_low = np.array([-scenario.delta_gamma_max, -scenario.delta_psi_max], dtype=np.float32)
        expected_high = -expected_low
        if engine.batch_size != config.batch_size:
            raise ValueError('engine batch_size must match formal config.')
        if not np.array_equal(engine.action_low.detach().cpu().numpy(), expected_low) or not np.array_equal(engine.action_high.detach().cpu().numpy(), expected_high):
            raise ValueError('engine action range must match ScenarioConfig.')
        radius = _nonnegative_float(uav_collision_radius, name='uav_collision_radius')
        if engine.actor.uav_radius != radius:
            raise ValueError('engine actor uav_radius is incompatible.')
        self.scenario = scenario
        self.rewards = rewards
        self.config = config
        self.engine = engine
        self.scenario_sources = dict(scenario_sources)
        self.validation_runner = validation_runner
        self.reporter = reporter
        self.selector = selector or V2CurriculumSelector(
            config.curriculum_mix,
            seed=derive_v2_component_seed(config.seed, config.stage, 'curriculum'),
        )
        self.exploration_rng = exploration_rng or np.random.default_rng(
            derive_v2_component_seed(config.seed, config.stage, 'exploration')
        )
        self.result = V2FormalTrainingResult.empty(
            config.stage,
            global_steps_start=global_steps_start,
        )
        self.result.curriculum_sample_counts = {
            level: 0 for level in config.curriculum_mix
        }
        self.controller = V2EarlyStopController(
            window_episode_count=config.window_episode_count,
            max_failures_per_window=config.max_failures_per_window,
            consecutive_qualified_windows=config.consecutive_qualified_windows,
            early_stop_min_steps=config.early_stop_min_steps,
        )
        self.env = V2StaticNoFlyTrajectoryEnv(
            scenario,
            rewards,
            uav_collision_radius=radius,
        )

    def _new_episode(self) -> tuple[V2Observation, str]:
        level = self.selector.sample()
        payload = self.scenario_sources[level].generate()
        observation, _ = self.env.reset(options={'scenario': payload})
        self.result.curriculum_sample_counts[level] = self.result.curriculum_sample_counts.get(level, 0) + 1
        return observation, level

    def _near_goal(self, info: Mapping[str, Any]) -> bool:
        values = (
            float(info.get('goal_distance', float('inf'))),
            float(info.get('segment_goal_distance', float('inf'))),
        )
        return bool(info.get('goal_reached_by_segment', False)) or min(values) <= self.config.near_goal_radius

    def run(self) -> V2FormalTrainingResult:
        if self.reporter is not None:
            self.reporter.begin_episode()
        observation, episode_level = self._new_episode()
        episode_return = 0.0
        episode_length = 0
        episode_warmup_steps = 0
        transitions: list[_EpisodeTransition] = []
        episode_actions: list[np.ndarray] = []
        slot_refs: list[tuple[int, int]] = []
        update_metrics: list[V2TD3UpdateMetrics] = []
        self.engine.actor.train()
        while self.result.stage_steps < self.config.max_steps:
            local_step_index = self.result.stage_steps
            in_warmup = local_step_index < self.config.warmup_steps
            exploration_noise, policy_noise, noise_clip = self.config.noise_schedule.values(
                local_step_index,
                max_steps=self.config.max_steps,
            )
            self.engine.set_target_noise(policy_noise=policy_noise, noise_clip=noise_clip)
            line_safe = self.env.line_to_goal_is_safe(
                self.env.state[:3],
                clearance=self.config.terminal_geo_safe_clearance,
            )
            action = self.engine.select_action(
                observation,
                exploration_noise=exploration_noise,
                exploration_rng=self.exploration_rng,
            )
            if in_warmup:
                episode_warmup_steps += 1
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            if self.reporter is not None:
                episode_actions.append(self.env.prev_action.copy())
            done = bool(terminated or truncated)
            near_goal = self._near_goal(info)
            slot_ref = self.engine.replay.add(
                observation,
                action,
                reward,
                next_observation,
                done,
                success=False,
                near_goal=near_goal,
                line_to_goal_safe=line_safe,
            )
            slot_refs.append(slot_ref)
            transitions.append(_EpisodeTransition(
                observation=observation,
                action=np.asarray(action, dtype=np.float32).copy(),
                reward=float(reward),
                next_observation=next_observation,
                done=done,
                near_goal=near_goal,
                line_to_goal_safe=line_safe,
            ))
            episode_return += float(reward)
            episode_length += 1
            self.result.stage_steps += 1
            self.result.global_steps_end = self.result.global_steps_start + self.result.stage_steps
            current_bc_lambda = v2_bc_lambda(local_step_index)
            if len(self.engine.replay) >= self.engine.batch_size:
                try:
                    metrics = self.engine.update_once(
                        total_steps=self.result.stage_steps,
                        bc_lambda=current_bc_lambda,
                    )
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        'V2 TD3 numerical failure in '
                        f'stage={self.config.stage} at '
                        f'stage_steps={self.result.stage_steps}: {exc}'
                    ) from exc
                update_metrics.append(metrics)
                self.result.update_count += 1
            observation = next_observation

            if self.reporter is not None:
                self.reporter.maybe_report_progress(
                    stage_steps=self.result.stage_steps,
                    completed_episodes=len(self.result.episodes),
                    current_episode_steps=episode_length,
                    actor_active=self.result.stage_steps > self.config.actor_freeze_steps,
                )

            if done:
                outcome = str(info.get('outcome', ''))
                if outcome not in _OUTCOMES:
                    raise RuntimeError('Training environment returned an invalid terminal outcome.')
                self.result.outcome_counts[outcome] += 1
                if outcome == 'goal':
                    for transition in transitions:
                        self.engine.replay.add_success_transition(
                            transition.observation,
                            transition.action,
                            transition.reward,
                            transition.next_observation,
                            transition.done,
                            near_goal=transition.near_goal,
                            line_to_goal_safe=transition.line_to_goal_safe,
                        )
                    self.engine.replay.mark_success_slots(slot_refs, success=True)
                actor_updates = [value for value in update_metrics if value.actor_updated]
                episode_record = {
                    'episode': len(self.result.episodes) + 1,
                    'stage': self.config.stage,
                    'curriculum_level': episode_level,
                    'stage_steps': self.result.stage_steps,
                    'global_steps': self.result.global_steps_end,
                    'outcome': outcome,
                    'episode_return': episode_return,
                    'episode_length': episode_length,
                    'policy_warmup_steps': episode_warmup_steps,
                    'critic_loss': mean([value.critic_loss for value in update_metrics]) if update_metrics else 0.0,
                    'actor_loss': mean([value.actor_loss for value in actor_updates]) if actor_updates else 0.0,
                    'bc_loss': mean([value.bc_loss for value in actor_updates]) if actor_updates else 0.0,
                    'weighted_bc_contribution': mean([value.bc_loss * value.bc_lambda for value in actor_updates]) if actor_updates else 0.0,
                    'terminal_geo_loss': mean([value.terminal_geo_loss for value in actor_updates]) if actor_updates else 0.0,
                    'bc_lambda': current_bc_lambda,
                    'exploration_noise': exploration_noise,
                    'policy_noise': policy_noise,
                    'noise_clip': noise_clip,
                    'replay_size': len(self.engine.replay),
                    'replay_success_fraction': self.engine.replay.success_fraction(),
                    'success_replay_size': self.engine.replay.success_size,
                    'batch_success_fraction': update_metrics[-1].sample_success_fraction if update_metrics else 0.0,
                }
                self.result.episodes.append(episode_record)
                if self.reporter is not None:
                    reporting_record = dict(episode_record)
                    reporting_record['actor_updated'] = bool(actor_updates)
                    reporting_record['actor_update_status'] = (
                        'updated'
                        if actor_updates
                        else (
                            'frozen'
                            if self.result.stage_steps <= self.config.actor_freeze_steps
                            else 'not_updated'
                        )
                    )
                    self.reporter.record_episode(
                        reporting_record,
                        scenario_payload=self.env.export_scenario(),
                        trajectory=[point.tolist() for point in self.env.trajectory],
                        actions=[value.tolist() for value in episode_actions],
                        terminal_state=self.env.state.copy().tolist(),
                    )
                window = self.controller.add_episode(outcome, stage_steps=self.result.stage_steps)
                if window is not None:
                    window_episodes = self.result.episodes[-self.config.window_episode_count:]
                    window.update({
                        'episode_start': window_episodes[0]['episode'],
                        'episode_end': window_episodes[-1]['episode'],
                        'average_return': mean(value['episode_return'] for value in window_episodes),
                        'average_length': mean(value['episode_length'] for value in window_episodes),
                        'average_actor_loss': mean(value['actor_loss'] for value in window_episodes),
                        'average_critic_loss': mean(value['critic_loss'] for value in window_episodes),
                        'bc_lambda': current_bc_lambda,
                        'average_bc_loss': mean(value['bc_loss'] for value in window_episodes),
                        'average_weighted_bc_contribution': mean(value['weighted_bc_contribution'] for value in window_episodes),
                        'global_steps': self.result.global_steps_end,
                        'exploration_noise': exploration_noise,
                        'policy_noise': policy_noise,
                        'noise_clip': noise_clip,
                    })
                    self.result.windows.append(window)
                    if self.reporter is not None:
                        self.reporter.record_window(window)
                    if window['candidate']:
                        if self.reporter is not None:
                            self.reporter.prepare_validation_candidate(
                                self.controller.candidate_count,
                                global_steps=self.result.global_steps_end,
                            )
                        before_updates = (
                            self.engine.update_count,
                            len(self.engine.replay),
                            self.result.stage_steps,
                        )
                        validation = self.validation_runner(self.engine.actor)
                        if not isinstance(validation, V2ValidationResult):
                            raise RuntimeError('validation_runner returned an invalid result.')
                        after_updates = (
                            self.engine.update_count,
                            len(self.engine.replay),
                            self.result.stage_steps,
                        )
                        if before_updates != after_updates:
                            raise RuntimeError('Fixed validation polluted training state.')
                        validation_record = validation.to_dict()
                        validation_record['candidate_index'] = self.controller.candidate_count
                        validation_record['stage_steps'] = self.result.stage_steps
                        self.result.validation_records.append(validation_record)
                        self.result.candidate_validation_count = self.controller.candidate_count
                        self.controller.record_validation(validation.passed)
                        if validation.passed:
                            self.result.status = 'passed'
                            self.result.passed_validation = True
                            self.result.stop_reason = 'fixed_validation_passed'
                            break
                if self.result.stage_steps >= self.config.max_steps:
                    break
                if self.reporter is not None:
                    self.reporter.begin_episode()
                observation, episode_level = self._new_episode()
                episode_return = 0.0
                episode_length = 0
                episode_warmup_steps = 0
                transitions = []
                episode_actions = []
                slot_refs = []
                update_metrics = []

        if not self.result.passed_validation:
            self.result.status = 'failed'
            self.result.stop_reason = 'max_steps_without_validation'
        self.result.replay_size = len(self.engine.replay)
        self.result.replay_success_fraction = self.engine.replay.success_fraction()
        self.result.success_replay_size = self.engine.replay.success_size
        self.result.candidate_validation_count = self.controller.candidate_count
        return self.result


def build_v2_formal_checkpoint(
    engine: V2TD3UpdateEngine,
    result: V2FormalTrainingResult,
    config: V2FormalTrainingConfig,
    *,
    scenario: ScenarioConfig,
    rewards: RewardConfig,
    uav_collision_radius: float,
    seed_manifest: Mapping[str, Any],
    validation_pool_metadata: Mapping[str, Any],
    initialization_source: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(engine, V2TD3UpdateEngine):
        raise TypeError('engine must be V2TD3UpdateEngine.')
    if not isinstance(result, V2FormalTrainingResult) or not isinstance(config, V2FormalTrainingConfig):
        raise TypeError('result and config must be formal V2 types.')
    if not isinstance(scenario, ScenarioConfig) or not isinstance(rewards, RewardConfig):
        raise TypeError('scenario and rewards must use project config classes.')
    radius = _nonnegative_float(uav_collision_radius, name='uav_collision_radius')
    if result.stage != config.stage:
        raise ValueError('result stage and config stage must match.')
    if result.status not in ('passed', 'failed'):
        raise ValueError('result must be terminal before checkpointing.')
    if result.passed_validation != (result.status == 'passed'):
        raise ValueError('passed_validation and status are inconsistent.')
    if result.passed_validation and (
        not result.validation_records
        or result.validation_records[-1].get('passed') is not True
    ):
        raise ValueError('A passed stage requires a recorded passing fixed validation.')
    model_type = engine.model_type
    payload = {
        'format': (
            V2_FORMAL_CHECKPOINT_FORMAT
            if model_type == 'ann'
            else V2_SNN_FORMAL_CHECKPOINT_FORMAT
        ),
        'format_version': (
            V2_FORMAL_CHECKPOINT_VERSION
            if model_type == 'ann'
            else V2_SNN_FORMAL_CHECKPOINT_VERSION
        ),
        'status': result.status,
        'stage': result.stage,
        'passed_validation': result.passed_validation,
        'engine_checkpoint': engine.checkpoint_state_dict(),
        'formal_config': config.to_dict(),
        'scenario_config': scenario_config_snapshot(scenario),
        'reward_config': _strict_json_copy(asdict(rewards)),
        'uav_collision_radius': radius,
        'seed_manifest': _strict_json_copy(seed_manifest),
        'bc_schedule': _strict_json_copy(V2_BC_SCHEDULE_METADATA),
        'training_result': result.to_dict(),
        'validation_pool': _strict_json_copy(validation_pool_metadata),
        'initialization_source': _strict_json_copy(initialization_source),
    }
    if model_type == 'snn':
        payload['model_type'] = 'snn'
    return payload


def save_v2_formal_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    validated = load_v2_formal_checkpoint(payload)
    output = Path(path)
    if output.exists():
        raise FileExistsError(f'Formal V2 checkpoint already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(validated, output)


def load_v2_formal_checkpoint(
    source: str | Path | Mapping[str, Any],
    *,
    expected_stage: str | None = None,
    require_passed: bool = False,
    next_stage: str | None = None,
    expected_model_type: str | None = None,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f'Formal V2 checkpoint does not exist: {path}')
        payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError('Formal V2 checkpoint payload must be a mapping.')
    checkpoint_format = payload.get('format')
    if checkpoint_format == V2_FORMAL_CHECKPOINT_FORMAT:
        model_type = 'ann'
        expected_version = V2_FORMAL_CHECKPOINT_VERSION
        expected_fields = _FORMAL_CHECKPOINT_FIELDS
    elif checkpoint_format == V2_SNN_FORMAL_CHECKPOINT_FORMAT:
        model_type = 'snn'
        expected_version = V2_SNN_FORMAL_CHECKPOINT_VERSION
        expected_fields = _SNN_FORMAL_CHECKPOINT_FIELDS
    else:
        raise ValueError('Formal V2 checkpoint format is incompatible.')
    if expected_model_type is not None:
        if expected_model_type not in ('ann', 'snn'):
            raise ValueError('expected_model_type must be "ann" or "snn".')
        if model_type != expected_model_type:
            raise ValueError('Formal V2 checkpoint model type is incompatible.')
    if payload.get('format_version') != expected_version:
        raise ValueError('Formal V2 checkpoint format_version is incompatible.')
    if set(payload) != expected_fields:
        raise ValueError('Formal V2 checkpoint has missing or unknown fields.')
    if model_type == 'snn' and payload.get('model_type') != 'snn':
        raise ValueError('Formal V2 SNN checkpoint model_type is invalid.')
    stage = payload['stage']
    if stage not in V2_TD3_STAGES:
        raise ValueError('Formal V2 checkpoint stage is invalid.')
    if expected_stage is not None and stage != expected_stage:
        raise ValueError('Formal V2 checkpoint stage is incompatible.')
    if next_stage is not None:
        if next_stage not in _PREVIOUS_STAGE or stage != _PREVIOUS_STAGE[next_stage]:
            raise ValueError('Formal V2 checkpoint is not the required predecessor stage.')
    if payload['status'] not in ('passed', 'failed') or type(payload['passed_validation']) is not bool:
        raise ValueError('Formal V2 checkpoint terminal status is invalid.')
    if payload['passed_validation'] != (payload['status'] == 'passed'):
        raise ValueError('Formal V2 checkpoint status and validation flag disagree.')
    if require_passed and not payload['passed_validation']:
        raise ValueError('Formal V2 checkpoint has not passed validation.')
    engine_payload = payload['engine_checkpoint']
    expected_engine_format = (
        V2_TD3_CHECKPOINT_FORMAT
        if model_type == 'ann'
        else V2_SNN_TD3_CHECKPOINT_FORMAT
    )
    expected_engine_version = (
        V2_TD3_CHECKPOINT_VERSION
        if model_type == 'ann'
        else V2_SNN_TD3_CHECKPOINT_VERSION
    )
    if not isinstance(engine_payload, dict) or engine_payload.get('format') != expected_engine_format or engine_payload.get('format_version') != expected_engine_version:
        raise ValueError('Formal V2 checkpoint engine payload is incompatible.')
    if model_type == 'snn' and engine_payload.get('model_type') != 'snn':
        raise ValueError('Formal V2 SNN engine model_type is incompatible.')
    formal_config = payload['formal_config']
    if not isinstance(formal_config, dict) or formal_config.get('stage') != stage:
        raise ValueError('Formal V2 checkpoint config is incompatible.')
    config_values = dict(formal_config)
    try:
        noise_values = config_values.pop('noise_schedule')
        restored_config = V2FormalTrainingConfig(
            noise_schedule=V2NoiseSchedule(**noise_values),
            **config_values,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError('Formal V2 checkpoint config is incompatible.') from exc
    if restored_config.to_dict() != formal_config:
        raise ValueError('Formal V2 checkpoint config does not round-trip.')
    scenario_config_from_snapshot(payload['scenario_config'])
    reward_config = payload['reward_config']
    if not isinstance(reward_config, dict):
        raise ValueError('Formal V2 checkpoint RewardConfig is invalid.')
    try:
        restored_rewards = RewardConfig(**reward_config)
    except (TypeError, ValueError) as exc:
        raise ValueError('Formal V2 checkpoint RewardConfig is invalid.') from exc
    if _strict_json_copy(asdict(restored_rewards)) != reward_config:
        raise ValueError('Formal V2 checkpoint RewardConfig does not round-trip.')
    _nonnegative_float(payload['uav_collision_radius'], name='uav_collision_radius')
    if not isinstance(payload['seed_manifest'], dict) or not payload['seed_manifest']:
        raise ValueError('Formal V2 checkpoint seed_manifest is invalid.')
    for name, value in payload['seed_manifest'].items():
        if not isinstance(name, str):
            raise ValueError('Formal V2 checkpoint seed_manifest keys must be strings.')
        _nonnegative_int(value, name=f'seed_manifest[{name!r}]')
    if payload['bc_schedule'] != V2_BC_SCHEDULE_METADATA:
        raise ValueError('Formal V2 checkpoint BC schedule is incompatible.')
    training_result = payload['training_result']
    if not isinstance(training_result, dict) or training_result.get('stage') != stage or training_result.get('passed_validation') != payload['passed_validation']:
        raise ValueError('Formal V2 checkpoint result is incompatible.')
    if set(training_result) != {item.name for item in fields(V2FormalTrainingResult)}:
        raise ValueError('Formal V2 checkpoint result fields are incompatible.')
    validation_records = training_result.get('validation_records')
    if payload['passed_validation'] and (
        not isinstance(validation_records, list)
        or not validation_records
        or validation_records[-1].get('passed') is not True
    ):
        raise ValueError('Passed formal V2 checkpoint lacks a passing validation record.')
    validation_pool = payload['validation_pool']
    required_pool_fields = {
        'path',
        'format_version',
        'curriculum_level',
        'master_seed',
        'stage_seed',
        'scenario_count',
        'content_digest',
    }
    if not isinstance(validation_pool, dict) or set(validation_pool) != required_pool_fields:
        raise ValueError('Formal V2 checkpoint provenance is invalid.')
    if validation_pool['curriculum_level'] != stage:
        raise ValueError('Formal V2 checkpoint validation pool stage is incompatible.')
    for name in ('format_version', 'master_seed', 'stage_seed', 'scenario_count'):
        value = _nonnegative_int(validation_pool[name], name=f'validation_pool.{name}')
        if name in ('format_version', 'scenario_count') and value == 0:
            raise ValueError(f'validation_pool.{name} must be positive.')
    if not isinstance(validation_pool['path'], str) or not validation_pool['path']:
        raise ValueError('Formal V2 checkpoint validation pool path is invalid.')
    if not isinstance(validation_pool['content_digest'], str) or not validation_pool['content_digest']:
        raise ValueError('Formal V2 checkpoint validation pool digest is invalid.')
    initialization = payload['initialization_source']
    if not isinstance(initialization, dict):
        raise ValueError('Formal V2 checkpoint initialization source is invalid.')
    expected_kind = 'v2_bc_best' if stage == 'easy' else 'validated_v2_td3_stage'
    if initialization.get('kind') != expected_kind:
        raise ValueError('Formal V2 checkpoint initialization source is incompatible.')
    if not isinstance(initialization.get('path'), str) or not initialization['path']:
        raise ValueError('Formal V2 checkpoint initialization path is invalid.')
    if stage != 'easy' and initialization.get('previous_stage') != _PREVIOUS_STAGE[stage]:
        raise ValueError('Formal V2 checkpoint initialization predecessor is invalid.')
    if model_type == 'snn' and initialization.get('model_type') != 'snn':
        raise ValueError('Formal V2 SNN initialization source is incompatible.')
    return payload


__all__ = [
    'V2_BC_SCHEDULE_METADATA',
    'V2_FORMAL_CHECKPOINT_FORMAT',
    'V2_FORMAL_CHECKPOINT_VERSION',
    'V2_SNN_FORMAL_CHECKPOINT_FORMAT',
    'V2_SNN_FORMAL_CHECKPOINT_VERSION',
    'V2EarlyStopController',
    'V2BCFormalInitialization',
    'V2FormalStageTrainer',
    'V2FormalTrainingConfig',
    'V2FormalTrainingResult',
    'V2PreparedStageInitialization',
    'V2StageComponents',
    'build_v2_formal_checkpoint',
    'build_v2_stage_engine',
    'load_v2_formal_checkpoint',
    'load_v2_bc_formal_initialization',
    'prepare_v2_stage_initialization',
    'save_v2_formal_checkpoint',
    'validate_v2_prepared_stage_initialization',
]
