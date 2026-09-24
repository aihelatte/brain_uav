"""Generate sharded successful expert trajectories from V2 easy training scenes.

This module is intentionally independent from the legacy dataset generator.  It
does not train BC/TD3 models and it never substitutes failed rollouts for expert
successes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from brain_uav.baselines import (
    V2ArtificialPotentialFieldPlanner,
    V2HeuristicPlanner,
)
from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import (
    V2_ENV_SCENARIO_FORMAT,
    V2_ENV_SCENARIO_VERSION,
    V2ScenarioGenerator,
    V2StaticNoFlyTrajectoryEnv,
)
from brain_uav.envs.v2_scenario_generator import (
    V2_SCENARIO_GENERATOR_NAME,
    V2_SCENARIO_GENERATOR_VERSION,
)
from brain_uav.geometry import (
    GEOMETRY_TOLERANCE,
    NoFlyZone,
    no_fly_zone_from_dict,
)
from brain_uav.observations import V2Observation

from .v2_trajectory_io import (
    SuccessfulV2Trajectory,
    V2TrajectoryShardBuffer,
    build_successful_v2_trajectory,
)


V2_SCENARIO_POOL_FORMAT = 'v2_easy_training_scenario_pool'
V2_SCENARIO_POOL_VERSION = 2
V2_TRAJECTORY_CLUSTER_FORMAT = 'v2_easy_expert_trajectory_clusters'
V2_TRAJECTORY_CLUSTER_VERSION = 2
V2_SHAPE_TYPES = (
    'sphere',
    'ellipsoid',
    'box',
    'triangular_pyramid',
    'quadrangular_pyramid',
)
V2_ROLLOUT_OUTCOMES = ('goal', 'collision', 'timeout', 'ground', 'boundary')


class _V2Planner(Protocol):
    def act(self, observation: V2Observation) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class V2PlannerSpec:
    name: str
    factory: Callable[[V2StaticNoFlyTrajectoryEnv], _V2Planner]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError('Planner name must be a non-empty string.')
        if not callable(self.factory):
            raise TypeError('Planner factory must be callable.')


@dataclass(frozen=True, slots=True)
class V2PlannerRolloutResult:
    scenario_id: str
    planner_name: str
    outcome: str
    step_count: int
    trajectory: SuccessfulV2Trajectory | None

    def summary(self) -> dict[str, Any]:
        return {
            'scenario_id': self.scenario_id,
            'planner_name': self.planner_name,
            'outcome': self.outcome,
            'step_count': self.step_count,
        }


DEFAULT_V2_PLANNER_SPECS = (
    V2PlannerSpec('v2_heuristic', V2HeuristicPlanner),
    V2PlannerSpec('v2_apf', V2ArtificialPotentialFieldPlanner),
)


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _nonnegative_seed(value: Any, *, name: str = 'seed') -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a non-negative integer.')
    return value


def _nonnegative_finite(value: Any, *, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f'{name} must be finite and non-negative.')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and non-negative.') from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and non-negative.')
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f'Non-finite JSON number {value!r} is not allowed.')


def _strict_json_copy(value: Any) -> Any:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
    )
    return json.loads(encoded, parse_constant=_reject_json_constant)


def _scenario_config_snapshot(scenario: ScenarioConfig) -> dict[str, Any]:
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError('scenario must be a ScenarioConfig.')
    return _strict_json_copy(asdict(scenario))


def _scenario_config_from_snapshot(value: Any) -> ScenarioConfig:
    if type(value) is not dict:
        raise ValueError('scenario_config must be a strict JSON object.')
    normalized = _strict_json_copy(value)
    expected_fields = {field.name for field in fields(ScenarioConfig)}
    if set(normalized) != expected_fields:
        raise ValueError('scenario_config must contain the complete ScenarioConfig.')
    try:
        scenario = ScenarioConfig(**deepcopy(normalized))
    except (TypeError, ValueError) as exc:
        raise ValueError('scenario_config cannot construct ScenarioConfig.') from exc
    if _scenario_config_snapshot(scenario) != normalized:
        raise ValueError('scenario_config is not a normalized ScenarioConfig snapshot.')
    return scenario


def _write_strict_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(text + '\n', encoding='utf-8')


def _aabb_overlaps(first: NoFlyZone, second: NoFlyZone) -> bool:
    first_bounds = first.shape.bounding_box()
    second_bounds = second.shape.bounding_box()
    return bool(
        np.all(
            first_bounds.min_corner
            <= second_bounds.max_corner + GEOMETRY_TOLERANCE
        )
        and np.all(
            second_bounds.min_corner
            <= first_bounds.max_corner + GEOMETRY_TOLERANCE
        )
    )


def _aabb_overlap_pair_count(zones: Sequence[NoFlyZone]) -> int:
    return sum(
        _aabb_overlaps(first, second)
        for index, first in enumerate(zones)
        for second in zones[index + 1 :]
    )


def _direct_path_blocker_count(
    state: np.ndarray,
    goal: np.ndarray,
    zones: Sequence[NoFlyZone],
    scenario: ScenarioConfig,
) -> int:
    return sum(
        zone.violates_segment(
            state[:3],
            goal,
            uav_radius=scenario.corridor_blocking_margin,
        )
        for zone in zones
    )


def _validate_easy_scenario_payload(
    payload: Any,
    scenario: ScenarioConfig,
) -> dict[str, Any]:
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError('scenario must be a ScenarioConfig.')
    if type(payload) is not dict:
        raise ValueError('Each scenario payload must be a strict JSON object.')
    required = {'format', 'format_version', 'state', 'goal', 'zones', 'curriculum_level', 'metadata'}
    unknown = set(payload) - required
    missing = required - set(payload)
    if missing or unknown:
        raise ValueError(
            f'Invalid V2 easy scenario fields; missing={sorted(missing)}, '
            f'unknown={sorted(unknown)}.'
        )
    if payload['format'] != V2_ENV_SCENARIO_FORMAT:
        raise ValueError('Scenario pool contains a non-V2 scenario payload.')
    if payload['format_version'] != V2_ENV_SCENARIO_VERSION:
        raise ValueError('Scenario pool contains an unsupported V2 scenario version.')
    if payload['curriculum_level'] != 'easy':
        raise ValueError('V2 expert scenario pools may contain only easy scenarios.')
    state = np.asarray(payload['state'], dtype=np.float64)
    goal = np.asarray(payload['goal'], dtype=np.float64)
    if state.shape != (5,) or goal.shape != (3,):
        raise ValueError('V2 scenario state and goal have invalid shapes.')
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(goal)):
        raise ValueError('V2 scenario state and goal must be finite.')
    zones = payload['zones']
    if type(zones) is not list or len(zones) > 2:
        raise ValueError('V2 easy scenarios must contain zero, one, or two zones.')
    zone_ids: set[str] = set()
    actual_zones: list[NoFlyZone] = []
    for raw_zone in zones:
        zone = no_fly_zone_from_dict(raw_zone)
        if zone.zone_id in zone_ids:
            raise ValueError(f'Duplicate zone_id {zone.zone_id!r} in scenario payload.')
        zone_ids.add(zone.zone_id)
        actual_zones.append(zone)
    metadata = payload['metadata']
    if type(metadata) is not dict:
        raise ValueError('Generated V2 scenario metadata must be a JSON object.')
    scenario_seed = metadata.get('scenario_seed')
    _nonnegative_seed(scenario_seed, name='scenario_seed')
    required_metadata = {
        'generator': V2_SCENARIO_GENERATOR_NAME,
        'generator_version': V2_SCENARIO_GENERATOR_VERSION,
        'requested_curriculum_level': 'easy',
        'effective_curriculum_level': 'easy',
        'overlap_allowed': False,
        'aabb_overlap_pair_count': 0,
        'feasibility_passed': True,
    }
    for key, expected in required_metadata.items():
        actual = metadata.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f'Scenario metadata field {key!r} must equal {expected!r}.'
            )
    requested_blocker = metadata.get('requested_direct_path_blocker')
    if type(requested_blocker) is not bool:
        raise ValueError(
            'Scenario metadata requested_direct_path_blocker must be bool.'
        )
    claimed_blocker_count = metadata.get('direct_path_blocker_count')
    if type(claimed_blocker_count) is not int or claimed_blocker_count < 0:
        raise ValueError(
            'Scenario metadata direct_path_blocker_count must be a '
            'non-negative integer.'
        )
    if requested_blocker and claimed_blocker_count < 1:
        raise ValueError(
            'requested_direct_path_blocker requires direct_path_blocker_count >= 1.'
        )
    if not requested_blocker and claimed_blocker_count != 0:
        raise ValueError(
            'Unblocked V2 easy scenarios require direct_path_blocker_count == 0.'
        )
    feasibility_check = metadata.get('feasibility_check')
    if type(feasibility_check) is not str or not feasibility_check:
        raise ValueError(
            'Scenario metadata feasibility_check must be a non-empty string.'
        )
    if not requested_blocker and feasibility_check != 'direct_safe_corridor':
        raise ValueError(
            'Unblocked V2 easy scenarios must use the direct_safe_corridor '
            'feasibility_check.'
        )
    if requested_blocker and feasibility_check == 'direct_safe_corridor':
        raise ValueError(
            'Blocked V2 easy scenarios must not use the direct_safe_corridor '
            'feasibility_check.'
        )
    requested_count = metadata.get('requested_zone_count')
    effective_count = metadata.get('effective_zone_count')
    if (
        type(requested_count) is not int
        or requested_count != len(zones)
        or type(effective_count) is not int
        or effective_count != len(zones)
    ):
        raise ValueError('Scenario zone counts are inconsistent with the payload.')
    requested_shapes = metadata.get('requested_shape_types')
    requested_ground = metadata.get('requested_ground_contact')
    if (
        type(requested_shapes) is not list
        or len(requested_shapes) != len(zones)
        or any(shape_type not in V2_SHAPE_TYPES for shape_type in requested_shapes)
    ):
        raise ValueError('Scenario requested_shape_types are invalid.')
    if (
        type(requested_ground) is not list
        or len(requested_ground) != len(zones)
        or any(type(value) is not bool for value in requested_ground)
    ):
        raise ValueError('Scenario requested_ground_contact values are invalid.')
    actual_shapes = [raw_zone['shape']['shape_type'] for raw_zone in zones]
    if actual_shapes != requested_shapes:
        raise ValueError('Requested and effective V2 zone shape order must match.')
    actual_overlap_count = _aabb_overlap_pair_count(actual_zones)
    if actual_overlap_count != metadata['aabb_overlap_pair_count']:
        raise ValueError(
            'actual AABB overlap pair count '
            f'{actual_overlap_count} does not match metadata '
            f'aabb_overlap_pair_count={metadata["aabb_overlap_pair_count"]}.'
        )
    if actual_overlap_count != 0:
        raise ValueError('actual AABB overlap pair count must be zero for V2 easy.')
    actual_blocker_count = _direct_path_blocker_count(
        state,
        goal,
        actual_zones,
        scenario,
    )
    if actual_blocker_count != metadata['direct_path_blocker_count']:
        raise ValueError(
            'actual direct path blocker count '
            f'{actual_blocker_count} does not match metadata '
            f'direct_path_blocker_count={metadata["direct_path_blocker_count"]}.'
        )
    return _strict_json_copy(payload)


def generate_easy_scenario_pool(
    scenario: ScenarioConfig,
    scenario_count: int,
    *,
    seed: int,
    uav_collision_radius: float = 0.0,
) -> dict[str, Any]:
    """Generate an ordered deterministic pool using only V2ScenarioGenerator."""

    if not isinstance(scenario, ScenarioConfig):
        raise TypeError('scenario must be a ScenarioConfig.')
    count = _positive_int(scenario_count, name='scenario_count')
    master_seed = _nonnegative_seed(seed)
    collision_radius = _nonnegative_finite(
        uav_collision_radius,
        name='uav_collision_radius',
    )
    generator = V2ScenarioGenerator(scenario, 'easy', seed=master_seed)
    items: list[dict[str, Any]] = []
    for index in range(count):
        payload = _validate_easy_scenario_payload(generator.generate(), scenario)
        items.append(
            {
                'scenario_id': f'scenario_{index + 1:06d}',
                'sequence_index': index,
                'scenario_seed': payload['metadata']['scenario_seed'],
                'payload': payload,
            }
        )
    return {
        'format': V2_SCENARIO_POOL_FORMAT,
        'format_version': V2_SCENARIO_POOL_VERSION,
        'master_seed': master_seed,
        'scenario_count': count,
        'curriculum_level': 'easy',
        'scenario_config': _scenario_config_snapshot(scenario),
        'uav_collision_radius': collision_radius,
        'scenarios': items,
    }


def _validate_easy_scenario_pool(pool: Any) -> dict[str, Any]:
    if type(pool) is not dict:
        raise ValueError('V2 easy scenario pool must be a strict JSON object.')
    required = {
        'format',
        'format_version',
        'master_seed',
        'scenario_count',
        'curriculum_level',
        'scenario_config',
        'uav_collision_radius',
        'scenarios',
    }
    if set(pool) != required:
        raise ValueError('V2 easy scenario pool has missing or unknown fields.')
    if pool['format'] != V2_SCENARIO_POOL_FORMAT:
        raise ValueError('Unsupported V2 easy scenario pool format.')
    if pool['format_version'] != V2_SCENARIO_POOL_VERSION:
        raise ValueError('Unsupported V2 easy scenario pool version.')
    if pool['curriculum_level'] != 'easy':
        raise ValueError('V2 expert scenario pool curriculum must be easy.')
    master_seed = _nonnegative_seed(pool['master_seed'], name='master_seed')
    count = _positive_int(pool['scenario_count'], name='scenario_count')
    scenario_config = _scenario_config_from_snapshot(pool['scenario_config'])
    scenario_snapshot = _scenario_config_snapshot(scenario_config)
    collision_radius = _nonnegative_finite(
        pool['uav_collision_radius'],
        name='uav_collision_radius',
    )
    items = pool['scenarios']
    if type(items) is not list or len(items) != count:
        raise ValueError('Scenario pool count does not match its scenarios list.')
    validated_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if type(item) is not dict or set(item) != {
            'scenario_id',
            'sequence_index',
            'scenario_seed',
            'payload',
        }:
            raise ValueError(f'Scenario pool item {index} has invalid fields.')
        expected_id = f'scenario_{index + 1:06d}'
        if item['scenario_id'] != expected_id or item['scenario_id'] in seen_ids:
            raise ValueError('Scenario pool IDs must be unique and sequential.')
        if item['sequence_index'] != index:
            raise ValueError('Scenario pool sequence_index values must be ordered.')
        payload = _validate_easy_scenario_payload(item['payload'], scenario_config)
        if item['scenario_seed'] != payload['metadata']['scenario_seed']:
            raise ValueError('Scenario pool scenario_seed does not match its payload.')
        seen_ids.add(item['scenario_id'])
        validated_items.append(
            {
                'scenario_id': item['scenario_id'],
                'sequence_index': index,
                'scenario_seed': item['scenario_seed'],
                'payload': payload,
            }
        )
    return {
        'format': V2_SCENARIO_POOL_FORMAT,
        'format_version': V2_SCENARIO_POOL_VERSION,
        'master_seed': master_seed,
        'scenario_count': count,
        'curriculum_level': 'easy',
        'scenario_config': scenario_snapshot,
        'uav_collision_radius': collision_radius,
        'scenarios': validated_items,
    }


def save_easy_scenario_pool(path: str | Path, pool: Mapping[str, Any]) -> None:
    validated = _validate_easy_scenario_pool(_strict_json_copy(pool))
    _write_strict_json(Path(path), validated)


def load_easy_scenario_pool(path: str | Path) -> dict[str, Any]:
    try:
        raw = json.loads(
            Path(path).read_text(encoding='utf-8'),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError('Scenario pool is not valid strict JSON.') from exc
    return _validate_easy_scenario_pool(raw)


def collect_v2_planner_rollout(
    scenario_payload: Mapping[str, Any],
    *,
    scenario_id: str,
    trajectory_id: str,
    planner_spec: V2PlannerSpec,
    scenario: ScenarioConfig,
    rewards: RewardConfig,
    uav_collision_radius: float = 0.0,
) -> V2PlannerRolloutResult:
    """Run one fresh V2 environment and retain data only for a goal outcome."""

    payload = _validate_easy_scenario_payload(
        _strict_json_copy(scenario_payload),
        scenario,
    )
    if not isinstance(planner_spec, V2PlannerSpec):
        raise TypeError('planner_spec must be a V2PlannerSpec.')
    collision_radius = _nonnegative_finite(
        uav_collision_radius,
        name='uav_collision_radius',
    )
    env = V2StaticNoFlyTrajectoryEnv(
        scenario,
        rewards,
        fixed_scenarios=[payload],
        uav_collision_radius=collision_radius,
    )
    observation, _ = env.reset()
    planner = planner_spec.factory(env)
    observations: list[V2Observation] = []
    states_before_action: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    outcome = 'running'

    for _ in range(env.scenario.max_steps):
        raw_action = np.asarray(planner.act(observation), dtype=np.float32)
        if raw_action.shape != (2,) or not np.all(np.isfinite(raw_action)):
            raise ValueError(
                f'Planner {planner_spec.name!r} returned a non-finite action '
                'or an action with shape other than (2,).'
            )
        action = np.clip(raw_action, env.action_space.low, env.action_space.high).astype(
            np.float32,
            copy=False,
        )
        observations.append(observation)
        states_before_action.append(env.state.copy())
        executed_actions.append(action.copy())
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            outcome = str(info['outcome'])
            break
    if outcome not in V2_ROLLOUT_OUTCOMES:
        raise RuntimeError(
            f'Planner rollout ended without a recognized terminal outcome: {outcome!r}.'
        )
    trajectory = None
    if outcome == 'goal':
        trajectory = build_successful_v2_trajectory(
            trajectory_id=trajectory_id,
            scenario_id=scenario_id,
            planner_name=planner_spec.name,
            scenario_seed=payload['metadata']['scenario_seed'],
            observations=observations,
            states_before_action=np.asarray(states_before_action, dtype=np.float32),
            actions=np.asarray(executed_actions, dtype=np.float32),
            terminal_state=env.state.copy(),
            outcome=outcome,
        )
    return V2PlannerRolloutResult(
        scenario_id=scenario_id,
        planner_name=planner_spec.name,
        outcome=outcome,
        step_count=len(executed_actions),
        trajectory=trajectory,
    )


def _new_statistics(
    scenario_count: int,
    planner_specs: Sequence[V2PlannerSpec],
) -> dict[str, Any]:
    return {
        'requested_scenarios': scenario_count,
        'successful_trajectories': 0,
        'planners': {
            spec.name: {
                'attempts': 0,
                'goal': 0,
                'collision': 0,
                'timeout': 0,
                'ground': 0,
                'boundary': 0,
            }
            for spec in planner_specs
        },
        'by_zone_count': {
            str(count): {'requested_scenarios': 0, 'successful_trajectories': 0}
            for count in (0, 1, 2)
        },
        'shapes': {
            shape_type: {
                'pool_requested': 0,
                'pool_effective': 0,
                'successful_trajectory_coverage': 0,
            }
            for shape_type in V2_SHAPE_TYPES
        },
        'ground_contact': {
            'grounded': {
                'pool_zone_count': 0,
                'successful_trajectory_zone_coverage': 0,
            },
            'suspended': {
                'pool_zone_count': 0,
                'successful_trajectory_zone_coverage': 0,
            },
        },
        'rollout_summaries': [],
        'successful_trajectory_summaries': [],
    }


def _zone_descriptors(payload: Mapping[str, Any]) -> list[tuple[str, bool]]:
    requested_ground = payload['metadata']['requested_ground_contact']
    descriptors: list[tuple[str, bool]] = []
    for index, raw_zone in enumerate(payload['zones']):
        shape_type = raw_zone['shape']['shape_type']
        raw_metadata = raw_zone.get('metadata', {})
        grounded = raw_metadata.get('ground_contact')
        if type(grounded) is not bool:
            grounded = requested_ground[index]
        if type(grounded) is not bool:
            raise ValueError('Each generated easy zone must declare ground contact.')
        descriptors.append((shape_type, grounded))
    return descriptors


def _record_pool_scenario(statistics: dict[str, Any], payload: Mapping[str, Any]) -> None:
    zone_count = len(payload['zones'])
    statistics['by_zone_count'][str(zone_count)]['requested_scenarios'] += 1
    for shape_type in payload['metadata']['requested_shape_types']:
        statistics['shapes'][shape_type]['pool_requested'] += 1
    for shape_type, grounded in _zone_descriptors(payload):
        statistics['shapes'][shape_type]['pool_effective'] += 1
        key = 'grounded' if grounded else 'suspended'
        statistics['ground_contact'][key]['pool_zone_count'] += 1


def _record_rollout(
    statistics: dict[str, Any],
    payload: Mapping[str, Any],
    result: V2PlannerRolloutResult,
) -> None:
    planner = statistics['planners'][result.planner_name]
    planner['attempts'] += 1
    planner[result.outcome] += 1
    statistics['rollout_summaries'].append(result.summary())
    if result.trajectory is None:
        return
    statistics['successful_trajectories'] += 1
    zone_count = len(payload['zones'])
    statistics['by_zone_count'][str(zone_count)]['successful_trajectories'] += 1
    for shape_type, grounded in _zone_descriptors(payload):
        statistics['shapes'][shape_type]['successful_trajectory_coverage'] += 1
        key = 'grounded' if grounded else 'suspended'
        statistics['ground_contact'][key][
            'successful_trajectory_zone_coverage'
        ] += 1
    statistics['successful_trajectory_summaries'].append(
        {
            'trajectory_id': result.trajectory.trajectory_id,
            'scenario_id': result.scenario_id,
            'planner_name': result.planner_name,
            'step_count': result.step_count,
            'outcome': 'goal',
        }
    )


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f'Output path already exists and is not a directory: {path}')
        if any(path.iterdir()):
            raise FileExistsError(f'Refusing to overwrite non-empty output directory: {path}')
    else:
        path.mkdir(parents=True)


def generate_v2_trajectory_clusters(
    *,
    output_dir: str | Path,
    scenario_count: int,
    seed: int,
    shard_size: int,
    scenario_pool_path: str | Path | None = None,
    scenario: ScenarioConfig | None = None,
    rewards: RewardConfig | None = None,
    planner_specs: Sequence[V2PlannerSpec] | None = None,
    uav_collision_radius: float | None = None,
) -> dict[str, Any]:
    """Create an easy pool, run both experts, and save only goal rollouts."""

    count = _positive_int(scenario_count, name='scenario_count')
    master_seed = _nonnegative_seed(seed)
    per_shard = _positive_int(shard_size, name='shard_size')
    reward_config = rewards or RewardConfig()
    if not isinstance(reward_config, RewardConfig):
        raise TypeError('rewards must be a RewardConfig.')
    specs = tuple(DEFAULT_V2_PLANNER_SPECS if planner_specs is None else planner_specs)
    if len(specs) != 2 or any(not isinstance(spec, V2PlannerSpec) for spec in specs):
        raise ValueError('Exactly two V2PlannerSpec values are required.')
    if len({spec.name for spec in specs}) != 2:
        raise ValueError('The two planner names must be distinct.')

    if scenario_pool_path is None:
        scenario_config = scenario or ScenarioConfig()
        if not isinstance(scenario_config, ScenarioConfig):
            raise TypeError('scenario must be a ScenarioConfig.')
        collision_radius = _nonnegative_finite(
            0.0 if uav_collision_radius is None else uav_collision_radius,
            name='uav_collision_radius',
        )
        pool = generate_easy_scenario_pool(
            scenario_config,
            count,
            seed=master_seed,
            uav_collision_radius=collision_radius,
        )
    else:
        pool = load_easy_scenario_pool(scenario_pool_path)
        if pool['scenario_count'] != count:
            raise ValueError('scenario_count must match the loaded scenario pool.')
        master_seed = pool['master_seed']
        pool_scenario = _scenario_config_from_snapshot(pool['scenario_config'])
        if scenario is None:
            scenario_config = pool_scenario
        elif not isinstance(scenario, ScenarioConfig):
            raise TypeError('scenario must be a ScenarioConfig.')
        elif _scenario_config_snapshot(scenario) != pool['scenario_config']:
            raise ValueError(
                'Explicit ScenarioConfig does not match the loaded scenario pool.'
            )
        else:
            scenario_config = scenario
        pool_collision_radius = pool['uav_collision_radius']
        if uav_collision_radius is None:
            collision_radius = pool_collision_radius
        else:
            collision_radius = _nonnegative_finite(
                uav_collision_radius,
                name='uav_collision_radius',
            )
            if collision_radius != pool_collision_radius:
                raise ValueError(
                    'Explicit uav_collision_radius does not match the loaded '
                    'scenario pool.'
                )
    destination = Path(output_dir)
    _prepare_output_directory(destination)
    save_easy_scenario_pool(destination / 'scenario_pool.json', pool)

    statistics = _new_statistics(count, specs)
    shard_buffer = V2TrajectoryShardBuffer()
    shards: list[dict[str, Any]] = []
    next_trajectory_number = 1
    shard_number = 1
    chunk_start = 0

    def flush_chunk(end_index: int) -> None:
        nonlocal shard_number
        if shard_buffer.trajectory_count == 0:
            return
        filename = f'shard_{shard_number:04d}.npz'
        summary = shard_buffer.flush(destination / filename)
        shards.append(
            {
                'file': filename,
                'scenario_sequence_start': chunk_start,
                'scenario_sequence_end': end_index,
                **summary,
            }
        )
        shard_number += 1

    for index, item in enumerate(pool['scenarios']):
        payload = item['payload']
        _record_pool_scenario(statistics, payload)
        for planner_spec in specs:
            trajectory_id = f'trajectory_{next_trajectory_number:08d}'
            next_trajectory_number += 1
            rollout_started = perf_counter()
            result = collect_v2_planner_rollout(
                deepcopy(payload),
                scenario_id=item['scenario_id'],
                trajectory_id=trajectory_id,
                planner_spec=planner_spec,
                scenario=scenario_config,
                rewards=reward_config,
                uav_collision_radius=collision_radius,
            )
            _record_rollout(statistics, payload, result)
            print(
                f"[scenario {index + 1}/{count}] {item['scenario_id']} "
                f"{planner_spec.name}: outcome={result.outcome} "
                f"steps={result.step_count} elapsed={perf_counter() - rollout_started:.2f}s",
                flush=True,
            )
            if result.trajectory is not None:
                shard_buffer.add(result.trajectory)
        if (index + 1) % per_shard == 0:
            flush_chunk(index)
            chunk_start = index + 1
    if chunk_start < count:
        flush_chunk(count - 1)

    manifest = {
        'format': V2_TRAJECTORY_CLUSTER_FORMAT,
        'format_version': V2_TRAJECTORY_CLUSTER_VERSION,
        'status': (
            'complete'
            if statistics['successful_trajectories'] > 0
            else 'failed_zero_success'
        ),
        'scenario_pool_file': 'scenario_pool.json',
        'master_seed': master_seed,
        'requested_scenarios': count,
        'shard_size': per_shard,
        'scenario_config': _scenario_config_snapshot(scenario_config),
        'uav_collision_radius': collision_radius,
        'shards': shards,
        'statistics': statistics,
    }
    _write_strict_json(destination / 'manifest.json', manifest)
    if statistics['successful_trajectories'] == 0:
        raise RuntimeError(
            'V2 expert collection produced zero successful trajectories; '
            'failure summaries were written without a fallback dataset.'
        )
    return _strict_json_copy(manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate V2 easy Heuristic/APF expert trajectory clusters.'
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scenario-count', type=int, required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--shard-size', type=int, default=25)
    parser.add_argument('--scenario-pool', type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    manifest = generate_v2_trajectory_clusters(
        output_dir=args.output_dir,
        scenario_count=args.scenario_count,
        seed=args.seed,
        shard_size=args.shard_size,
        scenario_pool_path=args.scenario_pool,
    )
    print(
        json.dumps(
            {
                'output_dir': str(args.output_dir),
                'successful_trajectories': manifest['statistics'][
                    'successful_trajectories'
                ],
                'shard_count': len(manifest['shards']),
            },
            allow_nan=False,
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()


__all__ = [
    'DEFAULT_V2_PLANNER_SPECS',
    'V2_SCENARIO_POOL_FORMAT',
    'V2_SCENARIO_POOL_VERSION',
    'V2_TRAJECTORY_CLUSTER_FORMAT',
    'V2_TRAJECTORY_CLUSTER_VERSION',
    'V2PlannerRolloutResult',
    'V2PlannerSpec',
    'collect_v2_planner_rollout',
    'generate_easy_scenario_pool',
    'generate_v2_trajectory_clusters',
    'load_easy_scenario_pool',
    'save_easy_scenario_pool',
]
