"""Strict deterministic fixed validation for formal V2 ANN training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs.v2_scenario_generator import (
    V2_CURRICULUM_LEVELS,
    V2_SCENARIO_GENERATOR_NAME,
    V2_SCENARIO_GENERATOR_VERSION,
    V2ScenarioGenerator,
)
from brain_uav.envs.v2_static_no_fly_env import (
    V2_ENV_SCENARIO_FORMAT,
    V2_ENV_SCENARIO_VERSION,
    V2StaticNoFlyTrajectoryEnv,
)
from brain_uav.geometry import no_fly_zone_from_dict
from brain_uav.models.v2_ann import V2ANNPolicyActor
from brain_uav.models.v2_snn import V2SNNPolicyActor
from brain_uav.observations import V2ObservationScales, collate_v2_observations
from brain_uav.trainers.v2_reporting import V2ExperimentReporter


V2_VALIDATION_POOL_FORMAT = 'v2_td3_fixed_validation_pool'
V2_VALIDATION_POOL_VERSION = 1

_POOL_FIELDS = {
    'format',
    'format_version',
    'curriculum_level',
    'master_seed',
    'stage_seed',
    'scenario_count',
    'scenario_config',
    'uav_collision_radius',
    'scenarios',
}
_SCENARIO_RECORD_FIELDS = {
    'scenario_id',
    'sequence_index',
    'scenario_seed',
    'payload',
}
_PAYLOAD_FIELDS = {
    'format',
    'format_version',
    'state',
    'goal',
    'zones',
    'curriculum_level',
    'metadata',
}
_METADATA_FIELDS = {
    'generator',
    'generator_version',
    'scenario_seed',
    'requested_curriculum_level',
    'effective_curriculum_level',
    'requested_zone_count',
    'effective_zone_count',
    'shape_counts',
    'requested_shape_types',
    'requested_ground_contact',
    'requested_reference_scales',
    'overlap_allowed',
    'aabb_overlap_pair_count',
    'requested_direct_path_blocker',
    'direct_path_blocker_count',
    'feasibility_check',
    'feasibility_passed',
    'feasibility_examined_nodes',
    'feasibility_edge_checks',
    'generation_attempts',
    'rejection_counts',
}
_SHAPE_TYPES = (
    'sphere',
    'ellipsoid',
    'box',
    'triangular_pyramid',
    'quadrangular_pyramid',
)
_VALIDATION_STAGE_OFFSETS = {'easy': 101, 'medium': 211, 'hard': 307}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f'Non-finite JSON constant {value!r} is not allowed.')


def _strict_json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, allow_nan=False, sort_keys=True),
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Value must be strict JSON data.') from exc


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a non-negative integer.')
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a non-negative integer.') from exc
    if result < 0 or result != value:
        raise ValueError(f'{name} must be a non-negative integer.')
    return result


def _positive_int(value: Any, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return result


def _finite_nonnegative(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and non-negative.') from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and non-negative.')
    return result


def _finite_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite vector of length {length}.') from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be a finite vector of length {length}.')
    return result


def scenario_config_snapshot(scenario: ScenarioConfig) -> dict[str, Any]:
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError('scenario must be a ScenarioConfig.')
    copied = _strict_json_copy(asdict(scenario))
    if not isinstance(copied, dict):
        raise RuntimeError('ScenarioConfig serialization must produce an object.')
    return copied


def scenario_config_from_snapshot(value: Any) -> ScenarioConfig:
    copied = _strict_json_copy(value)
    if not isinstance(copied, dict):
        raise ValueError('scenario_config must be a strict JSON object.')
    expected_fields = {item.name for item in fields(ScenarioConfig)}
    if set(copied) != expected_fields:
        raise ValueError('scenario_config fields do not match ScenarioConfig.')
    copied['no_fly_radius_range'] = tuple(copied['no_fly_radius_range'])
    copied['curriculum_distance_ratios'] = {
        key: tuple(pair) for key, pair in copied['curriculum_distance_ratios'].items()
    }
    copied['no_fly_radius_curriculum'] = {
        key: tuple(pair) for key, pair in copied['no_fly_radius_curriculum'].items()
    }
    try:
        result = ScenarioConfig(**copied)
    except (TypeError, ValueError) as exc:
        raise ValueError('scenario_config is invalid.') from exc
    if scenario_config_snapshot(result) != _strict_json_copy(value):
        raise ValueError('scenario_config does not round-trip exactly.')
    return result


def _validate_scenario_payload(
    payload: Any,
    *,
    stage: str,
    corridor_blocking_margin: float,
) -> dict[str, Any]:
    copied = _strict_json_copy(payload)
    if not isinstance(copied, dict) or set(copied) != _PAYLOAD_FIELDS:
        raise ValueError('Validation scenario payload has missing or unknown fields.')
    if copied['format'] != V2_ENV_SCENARIO_FORMAT:
        raise ValueError('Validation scenario payload format is invalid.')
    if copied['format_version'] != V2_ENV_SCENARIO_VERSION:
        raise ValueError('Validation scenario payload format_version is invalid.')
    if copied['curriculum_level'] != stage:
        raise ValueError('Validation scenario curriculum_level is incompatible.')
    _finite_vector(copied['state'], length=5, name='scenario state')
    _finite_vector(copied['goal'], length=3, name='scenario goal')
    if not isinstance(copied['zones'], list):
        raise ValueError('Validation scenario zones must be a list.')
    zones = []
    seen_zone_ids: set[str] = set()
    actual_shape_types: list[str] = []
    for index, value in enumerate(copied['zones']):
        try:
            zone = no_fly_zone_from_dict(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Validation scenario zone {index} is invalid.') from exc
        if zone.zone_id in seen_zone_ids:
            raise ValueError('Validation scenario contains duplicate zone_id.')
        seen_zone_ids.add(zone.zone_id)
        zones.append(zone)
        actual_shape_types.append(zone.shape.to_dict()['shape_type'])

    metadata = copied['metadata']
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
        raise ValueError('Validation scenario metadata has missing or unknown fields.')
    if metadata['generator'] != V2_SCENARIO_GENERATOR_NAME:
        raise ValueError('Validation scenario generator is incompatible.')
    if metadata['generator_version'] != V2_SCENARIO_GENERATOR_VERSION:
        raise ValueError('Validation scenario generator_version is incompatible.')
    if metadata['requested_curriculum_level'] != stage or metadata['effective_curriculum_level'] != stage:
        raise ValueError('Validation scenario metadata curriculum is incompatible.')
    requested_count = _nonnegative_int(metadata['requested_zone_count'], name='requested_zone_count')
    effective_count = _nonnegative_int(metadata['effective_zone_count'], name='effective_zone_count')
    if requested_count != len(zones) or effective_count != len(zones):
        raise ValueError('Validation scenario zone counts are inconsistent.')
    requested_shapes = metadata['requested_shape_types']
    requested_ground = metadata['requested_ground_contact']
    requested_scales = metadata['requested_reference_scales']
    if requested_shapes != actual_shape_types:
        raise ValueError('Validation scenario requested_shape_types are inconsistent.')
    if not isinstance(requested_ground, list) or len(requested_ground) != len(zones) or any(type(value) is not bool for value in requested_ground):
        raise ValueError('Validation scenario requested_ground_contact is invalid.')
    if not isinstance(requested_scales, list) or len(requested_scales) != len(zones):
        raise ValueError('Validation scenario requested_reference_scales is invalid.')
    for value in requested_scales:
        if _finite_nonnegative(value, name='requested_reference_scale') <= 0.0:
            raise ValueError('requested_reference_scales must be positive.')
    shape_counts = metadata['shape_counts']
    if not isinstance(shape_counts, dict) or set(shape_counts) != set(_SHAPE_TYPES):
        raise ValueError('Validation scenario shape_counts is invalid.')
    expected_counts = {name: actual_shape_types.count(name) for name in _SHAPE_TYPES}
    if any(type(shape_counts[name]) is not int or shape_counts[name] < 0 for name in _SHAPE_TYPES) or shape_counts != expected_counts:
        raise ValueError('Validation scenario shape_counts are inconsistent.')
    if type(metadata['overlap_allowed']) is not bool:
        raise ValueError('Validation scenario overlap_allowed must be bool.')
    if type(metadata['requested_direct_path_blocker']) is not bool:
        raise ValueError(
            'Validation scenario requested_direct_path_blocker must be bool.'
        )
    for name in (
        'aabb_overlap_pair_count',
        'direct_path_blocker_count',
        'feasibility_examined_nodes',
        'feasibility_edge_checks',
        'generation_attempts',
    ):
        value = _nonnegative_int(metadata[name], name=name)
        if name == 'generation_attempts' and value == 0:
            raise ValueError('generation_attempts must be positive.')
    if type(metadata['feasibility_passed']) is not bool or not metadata['feasibility_passed']:
        raise ValueError('Validation scenario feasibility_passed must be True.')
    if not isinstance(metadata['feasibility_check'], str) or not metadata['feasibility_check']:
        raise ValueError('Validation scenario feasibility_check is invalid.')
    rejection_counts = metadata['rejection_counts']
    if not isinstance(rejection_counts, dict) or any(
        not isinstance(key, str) or _nonnegative_int(value, name='rejection count') < 0
        for key, value in rejection_counts.items()
    ):
        raise ValueError('Validation scenario rejection_counts is invalid.')
    scenario_seed = _nonnegative_int(metadata['scenario_seed'], name='scenario_seed')
    if stage == 'easy' and metadata['overlap_allowed']:
        raise ValueError('Easy validation scenarios must not allow zone overlap.')
    if stage == 'hard' and not metadata['overlap_allowed']:
        raise ValueError('Hard validation scenarios must allow natural overlap.')
    requested_blocker = metadata['requested_direct_path_blocker']
    blocker_count = metadata['direct_path_blocker_count']
    if requested_blocker and blocker_count < 1:
        raise ValueError(
            'Validation scenarios requesting a blocker need '
            'direct_path_blocker_count >= 1.'
        )
    if not requested_blocker and blocker_count != 0:
        raise ValueError(
            'Validation scenarios without a requested blocker need '
            'direct_path_blocker_count == 0.'
        )
    if (
        not requested_blocker
        and metadata['feasibility_check'] != 'direct_safe_corridor'
    ):
        raise ValueError(
            'Validation scenarios without a requested blocker must use the '
            'direct_safe_corridor feasibility check.'
        )
    if (
        requested_blocker
        and metadata['feasibility_check'] == 'direct_safe_corridor'
    ):
        raise ValueError(
            'Validation scenarios requesting a blocker must not use the '
            'direct_safe_corridor feasibility check.'
        )
    if stage == 'hard' and len(zones) > 0 and blocker_count < 1:
        raise ValueError('Non-empty hard validation scenarios require a blocker.')
    actual_blocker_count = sum(
        zone.violates_segment(
            np.asarray(copied['state'], dtype=np.float64)[:3],
            np.asarray(copied['goal'], dtype=np.float64),
            uav_radius=corridor_blocking_margin,
        )
        for zone in zones
    )
    if actual_blocker_count != blocker_count:
        raise ValueError(
            'actual direct path blocker count '
            f'{actual_blocker_count} does not match metadata '
            f'direct_path_blocker_count={blocker_count}.'
        )
    copied['metadata']['scenario_seed'] = scenario_seed
    return copied


@dataclass(frozen=True, slots=True)
class V2ValidationPool:
    curriculum_level: str
    master_seed: int
    stage_seed: int
    scenario_config: Mapping[str, Any]
    uav_collision_radius: float
    scenarios: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        if self.curriculum_level not in V2_CURRICULUM_LEVELS:
            raise ValueError('curriculum_level must be easy, medium, or hard.')
        master_seed = _nonnegative_int(self.master_seed, name='master_seed')
        stage_seed = _nonnegative_int(self.stage_seed, name='stage_seed')
        derived_stage_seed = derive_validation_stage_seed(
            master_seed,
            self.curriculum_level,
        )
        if stage_seed != derived_stage_seed:
            raise ValueError(
                'Validation pool stage_seed is not the derived seed for '
                'master_seed and curriculum_level.'
            )
        scenario = scenario_config_from_snapshot(self.scenario_config)
        radius = _finite_nonnegative(self.uav_collision_radius, name='uav_collision_radius')
        if isinstance(self.scenarios, (str, bytes)) or not isinstance(self.scenarios, Sequence):
            raise ValueError('scenarios must be a sequence.')
        if len(self.scenarios) == 0:
            raise ValueError('Validation pool must contain at least one scenario.')
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for expected_index, record in enumerate(self.scenarios):
            copied = _strict_json_copy(record)
            if not isinstance(copied, dict) or set(copied) != _SCENARIO_RECORD_FIELDS:
                raise ValueError('Validation scenario record has missing or unknown fields.')
            scenario_id = copied['scenario_id']
            if not isinstance(scenario_id, str) or not scenario_id:
                raise ValueError('scenario_id must be a non-empty string.')
            if scenario_id in seen_ids:
                raise ValueError('Validation pool contains duplicate scenario_id.')
            seen_ids.add(scenario_id)
            sequence_index = _nonnegative_int(copied['sequence_index'], name='sequence_index')
            if sequence_index != expected_index:
                raise ValueError('Validation scenario sequence_index is not contiguous.')
            scenario_seed = _nonnegative_int(copied['scenario_seed'], name='scenario_seed')
            payload = _validate_scenario_payload(
                copied['payload'],
                stage=self.curriculum_level,
                corridor_blocking_margin=float(scenario.corridor_blocking_margin),
            )
            if payload['metadata']['scenario_seed'] != scenario_seed:
                raise ValueError('Validation scenario_seed disagrees with payload metadata.')
            records.append({
                'scenario_id': scenario_id,
                'sequence_index': sequence_index,
                'scenario_seed': scenario_seed,
                'payload': payload,
            })
        object.__setattr__(self, 'master_seed', master_seed)
        object.__setattr__(self, 'stage_seed', stage_seed)
        object.__setattr__(self, 'scenario_config', scenario_config_snapshot(scenario))
        object.__setattr__(self, 'uav_collision_radius', radius)
        object.__setattr__(self, 'scenarios', tuple(records))

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    def to_dict(self) -> dict[str, Any]:
        return _strict_json_copy({
            'format': V2_VALIDATION_POOL_FORMAT,
            'format_version': V2_VALIDATION_POOL_VERSION,
            'curriculum_level': self.curriculum_level,
            'master_seed': self.master_seed,
            'stage_seed': self.stage_seed,
            'scenario_count': self.scenario_count,
            'scenario_config': self.scenario_config,
            'uav_collision_radius': self.uav_collision_radius,
            'scenarios': self.scenarios,
        })

    @property
    def content_digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        return sha256(encoded).hexdigest()


def derive_validation_stage_seed(master_seed: int, stage: str) -> int:
    if stage not in V2_CURRICULUM_LEVELS:
        raise ValueError('stage must be easy, medium, or hard.')
    base = _nonnegative_int(master_seed, name='master_seed')
    sequence = np.random.SeedSequence([base, _VALIDATION_STAGE_OFFSETS[stage]])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def generate_v2_validation_pool(
    scenario: ScenarioConfig,
    stage: str,
    *,
    scenario_count: int = 100,
    master_seed: int = 20260904,
    uav_collision_radius: float = 0.0,
) -> V2ValidationPool:
    count = _positive_int(scenario_count, name='scenario_count')
    stage_seed = derive_validation_stage_seed(master_seed, stage)
    generator = V2ScenarioGenerator(scenario, stage, seed=stage_seed)
    records = []
    for index in range(count):
        payload = generator.generate()
        records.append({
            'scenario_id': f'{stage}_{index:05d}',
            'sequence_index': index,
            'scenario_seed': payload['metadata']['scenario_seed'],
            'payload': payload,
        })
    return V2ValidationPool(
        curriculum_level=stage,
        master_seed=master_seed,
        stage_seed=stage_seed,
        scenario_config=scenario_config_snapshot(scenario),
        uav_collision_radius=uav_collision_radius,
        scenarios=records,
    )


def save_v2_validation_pool(path: str | Path, pool: V2ValidationPool) -> None:
    if not isinstance(pool, V2ValidationPool):
        raise TypeError('pool must be a V2ValidationPool.')
    output = Path(path)
    if output.exists():
        raise FileExistsError(f'Validation pool already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(pool.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )


def load_v2_validation_pool(
    path: str | Path,
    *,
    expected_level: str,
    expected_scenario: ScenarioConfig | None = None,
    expected_count: int | None = None,
    expected_uav_collision_radius: float | None = None,
    expected_master_seed: int | None = None,
    expected_stage_seed: int | None = None,
) -> V2ValidationPool:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f'Validation pool does not exist: {input_path}')
    try:
        raw = json.loads(
            input_path.read_text(encoding='utf-8'),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError('Validation pool is not strict JSON.') from exc
    if not isinstance(raw, dict):
        raise ValueError('Validation pool must be a JSON object.')
    if raw.get('format') != V2_VALIDATION_POOL_FORMAT:
        raise ValueError('Validation pool format is incompatible.')
    if raw.get('format_version') != V2_VALIDATION_POOL_VERSION:
        raise ValueError('Validation pool format_version is incompatible.')
    if set(raw) != _POOL_FIELDS:
        raise ValueError('Validation pool has missing or unknown fields.')
    if raw['curriculum_level'] != expected_level:
        raise ValueError('Validation pool curriculum_level is incompatible.')
    pool = V2ValidationPool(
        curriculum_level=raw['curriculum_level'],
        master_seed=raw['master_seed'],
        stage_seed=raw['stage_seed'],
        scenario_config=raw['scenario_config'],
        uav_collision_radius=raw['uav_collision_radius'],
        scenarios=raw['scenarios'],
    )
    if _positive_int(raw['scenario_count'], name='scenario_count') != pool.scenario_count:
        raise ValueError('Validation pool scenario_count is inconsistent.')
    if expected_count is not None and pool.scenario_count != _positive_int(expected_count, name='expected_count'):
        raise ValueError('Validation pool scenario_count is incompatible.')
    if expected_scenario is not None and pool.scenario_config != scenario_config_snapshot(expected_scenario):
        raise ValueError('Validation pool ScenarioConfig is incompatible.')
    if expected_uav_collision_radius is not None and pool.uav_collision_radius != _finite_nonnegative(expected_uav_collision_radius, name='expected_uav_collision_radius'):
        raise ValueError('Validation pool uav_collision_radius is incompatible.')
    if expected_master_seed is not None:
        expected_master = _nonnegative_int(
            expected_master_seed,
            name='expected_master_seed',
        )
        if pool.master_seed != expected_master:
            raise ValueError('Validation pool master_seed is incompatible.')
    if expected_stage_seed is not None:
        expected_stage = _nonnegative_int(
            expected_stage_seed,
            name='expected_stage_seed',
        )
        if pool.stage_seed != expected_stage:
            raise ValueError('Validation pool stage_seed is incompatible.')
    return pool


def validation_passes(failure_count: int, *, max_failures: int = 5) -> bool:
    failures = _nonnegative_int(failure_count, name='failure_count')
    maximum = _nonnegative_int(max_failures, name='max_failures')
    return failures <= maximum


@dataclass(frozen=True, slots=True)
class V2ValidationResult:
    curriculum_level: str
    scenario_count: int
    goal_count: int
    failure_count: int
    outcome_counts: Mapping[str, int]
    scenarios: tuple[Mapping[str, Any], ...]
    max_failures: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return _strict_json_copy(asdict(self))


def evaluate_v2_fixed_validation(
    actor: V2ANNPolicyActor | V2SNNPolicyActor,
    pool: V2ValidationPool,
    rewards: RewardConfig,
    *,
    max_failures: int = 5,
    device: str | torch.device = 'cpu',
    reporter: V2ExperimentReporter | None = None,
) -> V2ValidationResult:
    if not isinstance(actor, (V2ANNPolicyActor, V2SNNPolicyActor)):
        raise TypeError('actor must be a V2ANNPolicyActor or V2SNNPolicyActor.')
    if not isinstance(pool, V2ValidationPool):
        raise TypeError('pool must be a V2ValidationPool.')
    if not isinstance(rewards, RewardConfig):
        raise TypeError('rewards must be a RewardConfig.')
    if reporter is not None and not isinstance(reporter, V2ExperimentReporter):
        raise TypeError('reporter must be a V2ExperimentReporter when provided.')
    maximum_failures = _nonnegative_int(max_failures, name='max_failures')
    scenario = scenario_config_from_snapshot(pool.scenario_config)
    expected_scales = V2ObservationScales(
        scenario.world_xy,
        scenario.world_z_min,
        scenario.world_z_max,
        scenario.gamma_max,
    )
    if actor.scales != expected_scales or actor.uav_radius != pool.uav_collision_radius:
        raise ValueError('Actor observation architecture is incompatible with validation pool.')
    expected_limit = torch.tensor(
        [scenario.delta_gamma_max, scenario.delta_psi_max],
        dtype=torch.float32,
    )
    if not torch.equal(actor.action_limit.detach().cpu(), expected_limit):
        raise ValueError('Actor action bounds are incompatible with validation pool.')
    target_device = torch.device(device)
    if target_device.type == 'cuda' and target_device.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA validation was requested but torch.cuda.is_available() is False.'
            )
        target_device = torch.device('cuda', torch.cuda.current_device())
    actor_devices = {parameter.device for parameter in actor.parameters()}
    if actor_devices != {target_device}:
        raise ValueError('Actor must already be on the requested validation device.')
    env = V2StaticNoFlyTrajectoryEnv(
        scenario,
        rewards,
        fixed_scenarios=[record['payload'] for record in pool.scenarios],
        uav_collision_radius=pool.uav_collision_radius,
    )
    was_training = actor.training
    actor.eval()
    counts = {name: 0 for name in ('goal', 'ground', 'boundary', 'collision', 'timeout')}
    details: list[dict[str, Any]] = []
    if reporter is not None:
        reporter.begin_validation(
            curriculum_level=pool.curriculum_level,
            scenario_count=pool.scenario_count,
        )
    try:
        for record in pool.scenarios:
            observation, _ = env.reset()
            episode_return = 0.0
            outcome = 'running'
            episode_length = 0
            actions: list[list[float]] = []
            while outcome == 'running':
                batch = collate_v2_observations([observation], device=target_device)
                with torch.inference_mode():
                    action = actor(batch)[0].detach().cpu().numpy().astype(np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                if reporter is not None:
                    actions.append(env.prev_action.copy().tolist())
                episode_return += float(reward)
                episode_length += 1
                if terminated or truncated:
                    outcome = info.get('outcome')
                    if outcome not in counts:
                        raise RuntimeError('Validation environment returned an invalid outcome.')
            counts[outcome] += 1
            detail = {
                'scenario_id': record['scenario_id'],
                'outcome': outcome,
                'episode_length': episode_length,
                'episode_return': episode_return,
            }
            details.append(detail)
            if reporter is not None:
                reporter.record_validation_scenario(
                    detail,
                    scenario_payload=record['payload'],
                    trajectory=[point.tolist() for point in env.trajectory],
                    actions=actions,
                    terminal_state=env.state.copy().tolist(),
                )
    except Exception:
        if reporter is not None:
            reporter.abort_validation()
        raise
    finally:
        actor.train(was_training)
    goal_count = counts['goal']
    failure_count = pool.scenario_count - goal_count
    result = V2ValidationResult(
        curriculum_level=pool.curriculum_level,
        scenario_count=pool.scenario_count,
        goal_count=goal_count,
        failure_count=failure_count,
        outcome_counts=counts,
        scenarios=tuple(details),
        max_failures=maximum_failures,
        passed=validation_passes(failure_count, max_failures=maximum_failures),
    )
    if reporter is not None:
        reporter.finish_validation(result.to_dict())
    return result


__all__ = [
    'V2_VALIDATION_POOL_FORMAT',
    'V2_VALIDATION_POOL_VERSION',
    'V2ValidationPool',
    'V2ValidationResult',
    'derive_validation_stage_seed',
    'evaluate_v2_fixed_validation',
    'generate_v2_validation_pool',
    'load_v2_validation_pool',
    'save_v2_validation_pool',
    'scenario_config_from_snapshot',
    'scenario_config_snapshot',
    'validation_passes',
]
