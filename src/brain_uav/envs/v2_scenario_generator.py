"""Reproducible random V2 training-scenario generation.

This module is independent from the legacy hemisphere curriculum and benchmark.
It generates strict V2 payloads containing unified ``NoFlyZone`` geometry.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isclose, isfinite, sqrt
from typing import Any

import numpy as np

from brain_uav.config import ScenarioConfig
from brain_uav.geometry import (
    GEOMETRY_TOLERANCE,
    Box,
    Ellipsoid,
    NoFlyZone,
    QuadrangularPyramid,
    Sphere,
    TriangularPyramid,
)

from .v2_feasibility import (
    V2FeasibilityConfig,
    check_v2_geometric_feasibility,
)


V2_ENV_SCENARIO_FORMAT = 'v2_static_no_fly_scenario'
V2_ENV_SCENARIO_VERSION = 1
V2_SCENARIO_GENERATOR_NAME = 'brain_uav_v2_random_training'
V2_SCENARIO_GENERATOR_VERSION = 2
V2_CURRICULUM_LEVELS = ('easy', 'medium', 'hard')
V2_SHAPE_TYPES = (
    'sphere',
    'ellipsoid',
    'box',
    'triangular_pyramid',
    'quadrangular_pyramid',
)

DEFAULT_V2_ZONE_COUNT_PROBABILITIES: dict[str, dict[int, float]] = {
    'easy': {0: 0.05, 1: 0.475, 2: 0.475},
    'medium': {0: 0.03, 2: 0.485, 3: 0.485},
    'hard': {0: 0.02, 2: 0.28, 3: 0.30, 4: 0.30, 5: 0.06, 6: 0.04},
}
DEFAULT_V2_SHAPE_PROBABILITIES: dict[str, float] = {
    shape_type: 0.20 for shape_type in V2_SHAPE_TYPES
}
# For easy and medium, the overall per-scenario probability that at least one
# zone blocks the straight start-to-goal safety corridor with
# corridor_blocking_margin, including zero-zone scenarios. The conditional
# probability for non-empty scenarios is derived in the config. Hard's 1.0 is
# a fixed sentinel for its original rule: every non-empty scenario must block.
DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES: dict[str, float] = {
    'easy': 0.50,
    'medium': 0.80,
    'hard': 1.00,
}


@dataclass(frozen=True, slots=True)
class _StageSamplingSpec:
    distance_ratio: tuple[float, float]
    state_z_ratio: tuple[float, float]
    goal_z_ratio: tuple[float, float]
    max_height_gap_ratio: float
    mean_y_ratio: float
    lateral_offset_ratio: float
    psi_range: tuple[float, float]
    reference_scale_range: tuple[float, float]


_STAGE_SPECS = {
    'easy': _StageSamplingSpec(
        (0.55, 0.85), (0.16, 0.28), (0.16, 0.33), 0.12, 0.12, 0.08,
        (-0.12, 0.12), (120.0, 190.0),
    ),
    'medium': _StageSamplingSpec(
        (0.80, 0.95), (0.18, 0.30), (0.18, 0.35), 0.14, 0.16, 0.10,
        (-0.15, 0.15), (180.0, 220.0),
    ),
    'hard': _StageSamplingSpec(
        (0.90, 1.10), (0.18, 0.30), (0.18, 0.36), 0.15, 0.22, 0.12,
        (-0.20, 0.20), (200.0, 250.0),
    ),
}


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite.') from exc
    if not isfinite(result):
        raise ValueError(f'{name} must be finite.')
    return result


def _probability(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f'{name} must be in [0, 1].')
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f'{name} must be greater than zero.')
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _validate_count_probabilities(value: Any) -> dict[str, dict[int, float]]:
    if not isinstance(value, Mapping) or set(value) != set(V2_CURRICULUM_LEVELS):
        raise ValueError('zone_count_probabilities must define easy, medium, and hard.')
    result: dict[str, dict[int, float]] = {}
    for level in V2_CURRICULUM_LEVELS:
        raw_distribution = value[level]
        if not isinstance(raw_distribution, Mapping) or not raw_distribution:
            raise ValueError(f'zone_count_probabilities[{level!r}] must be a non-empty mapping.')
        distribution: dict[int, float] = {}
        for count, raw_probability in raw_distribution.items():
            if type(count) is not int or count < 0:
                raise ValueError('V2 zone counts must be non-negative integers.')
            distribution[count] = _probability(
                raw_probability,
                name=f'zone_count_probabilities[{level!r}][{count!r}]',
            )
        if not isclose(sum(distribution.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f'zone_count_probabilities[{level!r}] must sum to 1.')
        result[level] = dict(sorted(distribution.items()))
    return result


def _validate_shape_probabilities(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(V2_SHAPE_TYPES):
        raise ValueError(
            'shape_probabilities must define exactly the five supported V2 shape types.'
        )
    result = {
        shape_type: _probability(
            value[shape_type], name=f'shape_probabilities[{shape_type!r}]'
        )
        for shape_type in V2_SHAPE_TYPES
    }
    if not isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError('shape_probabilities must sum to 1.')
    return result


def _validate_blocker_probabilities(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(V2_CURRICULUM_LEVELS):
        raise ValueError(
            'direct_path_blocker_probability must define easy, medium, and hard.'
        )
    return {
        level: _probability(
            value[level],
            name=f'direct_path_blocker_probability[{level!r}]',
        )
        for level in V2_CURRICULUM_LEVELS
    }


def _nonzero_blocker_probability(
    level: str,
    target: float,
    zero_zone_probability: float,
) -> float:
    """Derive the conditional blocker probability for non-empty scenarios.

    Zero-zone scenarios can never block, so reaching the overall target
    requires ``P(block | zones>=1) = target / (1 - P(0 zones))``. The
    reachable overall interval is ``[0, 1 - P(0 zones)]``.
    """

    non_empty_probability = 1.0 - zero_zone_probability
    if target > non_empty_probability:
        raise ValueError(
            f'direct_path_blocker_probability[{level!r}]={target!r} is '
            f'unreachable with zone_count_probabilities[{level!r}] zero-zone '
            f'probability {zero_zone_probability!r}; the achievable overall '
            f'interval is [0.0, {non_empty_probability!r}].'
        )
    if non_empty_probability <= 0.0:
        return 0.0
    return target / non_empty_probability


@dataclass(slots=True)
class V2ScenarioGeneratorConfig:
    """Auditable sampling and validation settings for the V2 generator."""

    zone_count_probabilities: Mapping[str, Mapping[int, float]] = field(
        default_factory=lambda: {
            level: dict(values)
            for level, values in DEFAULT_V2_ZONE_COUNT_PROBABILITIES.items()
        }
    )
    shape_probabilities: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_V2_SHAPE_PROBABILITIES)
    )
    ground_contact_probability: float = 0.5
    medium_overlap_probability: float = 0.10
    direct_path_blocker_probability: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_V2_DIRECT_PATH_BLOCKER_PROBABILITIES)
    )
    start_surface_clearance: float = 160.0
    goal_surface_clearance: float = 106.0
    start_goal_sampling_attempts: int = 40
    zone_candidate_attempts: int = 50
    ellipsoid_horizontal_ratio_range: tuple[float, float] = (0.70, 1.00)
    ellipsoid_vertical_ratio_range: tuple[float, float] = (0.50, 0.80)
    pyramid_base_ratio_range: tuple[float, float] = (1.50, 2.00)
    pyramid_height_ratio_range: tuple[float, float] = (0.75, 1.25)
    feasibility: V2FeasibilityConfig = field(default_factory=V2FeasibilityConfig)
    # Derived in __post_init__: easy/medium use P(blocker | zones >= 1);
    # hard is always 1.0 for its fixed non-empty-scenario rule.
    direct_path_blocker_probability_nonzero: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.zone_count_probabilities = _validate_count_probabilities(
            self.zone_count_probabilities
        )
        self.direct_path_blocker_probability = _validate_blocker_probabilities(
            self.direct_path_blocker_probability
        )
        if self.direct_path_blocker_probability['hard'] != 1.0:
            raise ValueError(
                "direct_path_blocker_probability['hard'] must be 1.0; hard "
                'requires a blocker whenever at least one zone is present.'
            )
        self.direct_path_blocker_probability_nonzero = {
            level: (
                1.0
                if level == 'hard'
                else _nonzero_blocker_probability(
                    level,
                    self.direct_path_blocker_probability[level],
                    self.zone_count_probabilities[level].get(0, 0.0),
                )
            )
            for level in V2_CURRICULUM_LEVELS
        }
        self.shape_probabilities = _validate_shape_probabilities(
            self.shape_probabilities
        )
        self.ground_contact_probability = _probability(
            self.ground_contact_probability, name='ground_contact_probability'
        )
        self.medium_overlap_probability = _probability(
            self.medium_overlap_probability, name='medium_overlap_probability'
        )
        self.start_surface_clearance = _positive_float(
            self.start_surface_clearance, name='start_surface_clearance'
        )
        self.goal_surface_clearance = _positive_float(
            self.goal_surface_clearance, name='goal_surface_clearance'
        )
        self.start_goal_sampling_attempts = _positive_int(
            self.start_goal_sampling_attempts, name='start_goal_sampling_attempts'
        )
        self.zone_candidate_attempts = _positive_int(
            self.zone_candidate_attempts, name='zone_candidate_attempts'
        )
        for field_name in (
            'ellipsoid_horizontal_ratio_range',
            'ellipsoid_vertical_ratio_range',
            'pyramid_base_ratio_range',
            'pyramid_height_ratio_range',
        ):
            raw_range = getattr(self, field_name)
            if not isinstance(raw_range, tuple) or len(raw_range) != 2:
                raise ValueError(f'{field_name} must be a two-value tuple.')
            lower = _positive_float(raw_range[0], name=f'{field_name}[0]')
            upper = _positive_float(raw_range[1], name=f'{field_name}[1]')
            if lower > upper:
                raise ValueError(f'{field_name} lower bound must not exceed upper bound.')
            setattr(self, field_name, (lower, upper))
        if not isinstance(self.feasibility, V2FeasibilityConfig):
            raise ValueError('feasibility must be a V2FeasibilityConfig.')


class V2ScenarioGenerationError(RuntimeError):
    """Raised when the configured scenario cannot be generated without fallback."""

    def __init__(
        self,
        curriculum_level: str,
        requested_zone_count: int,
        attempts: int,
        rejection_counts: Mapping[str, int],
        scenario_seed: int,
    ) -> None:
        self.curriculum_level = curriculum_level
        self.requested_zone_count = requested_zone_count
        self.attempts = attempts
        self.rejection_counts = dict(rejection_counts)
        self.scenario_seed = scenario_seed
        super().__init__(
            'Failed to generate V2 scenario: '
            f'curriculum_level={curriculum_level!r}, '
            f'requested_zone_count={requested_zone_count}, attempts={attempts}, '
            f'scenario_seed={scenario_seed}, '
            f'rejection_counts={dict(sorted(rejection_counts.items()))}.'
        )


class V2ScenarioGenerator:
    """Stateful seeded generator for one explicit V2 curriculum level."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        curriculum_level: str,
        *,
        seed: int | None = None,
        config: V2ScenarioGeneratorConfig | None = None,
    ) -> None:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError('scenario must be a ScenarioConfig.')
        if curriculum_level not in V2_CURRICULUM_LEVELS:
            raise ValueError('curriculum_level must be easy, medium, or hard.')
        self._validate_scenario_space(scenario)
        self.scenario = scenario
        self.curriculum_level = curriculum_level
        self.config = config or V2ScenarioGeneratorConfig()
        if not isinstance(self.config, V2ScenarioGeneratorConfig):
            raise TypeError('config must be a V2ScenarioGeneratorConfig.')
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _validate_scenario_space(scenario: ScenarioConfig) -> None:
        values = {
            'world_xy': scenario.world_xy,
            'world_z_min': scenario.world_z_min,
            'world_z_max': scenario.world_z_max,
            'target_distance': scenario.target_distance,
        }
        converted = {name: _finite_float(value, name=name) for name, value in values.items()}
        if converted['world_xy'] <= 0.0 or converted['target_distance'] <= 0.0:
            raise ValueError('world_xy and target_distance must be positive.')
        if converted['world_z_min'] < 0.0 or converted['world_z_max'] <= converted['world_z_min']:
            raise ValueError('world_z bounds must define a positive above-ground interval.')
        _positive_int(
            scenario.scenario_max_sampling_attempts,
            name='scenario_max_sampling_attempts',
        )

    def seed(self, seed: int | None = None) -> None:
        """Reset only this generator's explicit NumPy Generator."""

        self.rng = np.random.default_rng(seed)

    def generate(self) -> dict[str, Any]:
        """Generate one strict V2 scenario or raise with rejection diagnostics."""

        scenario_seed = int(
            self.rng.integers(0, np.iinfo(np.int64).max, endpoint=False)
        )
        local_rng = np.random.default_rng(scenario_seed)
        requested_zone_count = self._sample_zone_count(local_rng)
        require_direct_path_blocker = self._sample_direct_path_blocker_branch(
            local_rng,
            requested_zone_count,
        )
        overlap_allowed = self._sample_overlap_mode(local_rng)
        requested_shape_types = self._sample_requested_shape_types(
            local_rng,
            requested_zone_count,
        )
        requested_ground_contact = self._sample_requested_ground_contact(
            local_rng,
            requested_shape_types,
        )
        rejection_counts: Counter[str] = Counter()
        max_attempts = self.scenario.scenario_max_sampling_attempts

        for attempt in range(1, max_attempts + 1):
            pair = self._sample_start_goal(local_rng)
            if pair is None:
                rejection_counts['start_goal_sampling'] += 1
                continue
            state, goal = pair
            sampled = self._sample_zones(
                local_rng,
                state,
                goal,
                requested_zone_count,
                requested_shape_types=requested_shape_types,
                requested_ground_contact=requested_ground_contact,
                overlap_allowed=overlap_allowed,
                require_direct_path_blocker=require_direct_path_blocker,
                rejection_counts=rejection_counts,
            )
            if sampled is None:
                rejection_counts['zone_candidate_sampling'] += 1
                continue
            zones, reference_scales = sampled
            blocker_count = self._count_direct_path_blockers(state, goal, zones)
            if require_direct_path_blocker and blocker_count < 1:
                rejection_counts['required_direct_path_blocker_missing'] += 1
                continue
            if not require_direct_path_blocker and blocker_count != 0:
                rejection_counts['direct_path_blocker_forbidden_present'] += 1
                continue

            if not require_direct_path_blocker:
                feasibility_type = 'direct_safe_corridor'
                feasibility_passed = True
                feasibility_examined_nodes = 0
                feasibility_edge_checks = 0
            else:
                result = check_v2_geometric_feasibility(
                    state[:3],
                    goal,
                    zones,
                    self.scenario,
                    self.config.feasibility,
                )
                feasibility_type = result.algorithm
                feasibility_passed = result.reachable
                feasibility_examined_nodes = result.examined_nodes
                feasibility_edge_checks = result.edge_checks
                if not result.reachable:
                    rejection_counts[f'feasibility_{result.reason}'] += 1
                    continue

            metadata = self._scenario_metadata(
                scenario_seed=scenario_seed,
                requested_zone_count=requested_zone_count,
                requested_shape_types=requested_shape_types,
                requested_ground_contact=requested_ground_contact,
                zones=zones,
                reference_scales=reference_scales,
                overlap_allowed=overlap_allowed,
                requested_direct_path_blocker=require_direct_path_blocker,
                direct_path_blocker_count=blocker_count,
                feasibility_type=feasibility_type,
                feasibility_passed=feasibility_passed,
                feasibility_examined_nodes=feasibility_examined_nodes,
                feasibility_edge_checks=feasibility_edge_checks,
                generation_attempts=attempt,
                rejection_counts=rejection_counts,
            )
            return {
                'format': V2_ENV_SCENARIO_FORMAT,
                'format_version': V2_ENV_SCENARIO_VERSION,
                'state': state.tolist(),
                'goal': goal.tolist(),
                'zones': [zone.to_dict() for zone in zones],
                'curriculum_level': self.curriculum_level,
                'metadata': metadata,
            }

        raise V2ScenarioGenerationError(
            self.curriculum_level,
            requested_zone_count,
            max_attempts,
            rejection_counts,
            scenario_seed,
        )

    def _sample_zone_count(self, rng: np.random.Generator) -> int:
        distribution = self.config.zone_count_probabilities[self.curriculum_level]
        counts = np.asarray(tuple(distribution.keys()), dtype=np.int64)
        probabilities = np.asarray(tuple(distribution.values()), dtype=np.float64)
        return int(rng.choice(counts, p=probabilities))

    def _sample_direct_path_blocker_branch(
        self,
        rng: np.random.Generator,
        zone_count: int,
    ) -> bool:
        """Draw the corridor-blocker branch once per scenario (v2 semantics).

        The branch is drawn before any sampling attempt and every retry of
        this scenario keeps it, so a generation failure can never quietly
        switch a scenario between the blocked and unblocked populations.
        Zero-zone scenarios can never block. Hard non-empty scenarios always
        require a blocker and skip the RNG draw to preserve their sample stream.
        """

        if zone_count <= 0:
            return False
        if self.curriculum_level == 'hard':
            return True
        conditional = self.config.direct_path_blocker_probability_nonzero[
            self.curriculum_level
        ]
        return bool(rng.random() < conditional)

    def _sample_overlap_mode(self, rng: np.random.Generator) -> bool:
        if self.curriculum_level == 'easy':
            return False
        if self.curriculum_level == 'hard':
            return True
        return bool(rng.random() < self.config.medium_overlap_probability)

    def _sample_requested_shape_types(
        self,
        rng: np.random.Generator,
        zone_count: int,
    ) -> tuple[str, ...]:
        probabilities = np.asarray(
            [self.config.shape_probabilities[name] for name in V2_SHAPE_TYPES],
            dtype=np.float64,
        )
        return tuple(
            str(rng.choice(V2_SHAPE_TYPES, p=probabilities))
            for _ in range(zone_count)
        )

    def _sample_requested_ground_contact(
        self,
        rng: np.random.Generator,
        shape_types: tuple[str, ...],
    ) -> tuple[bool, ...]:
        return tuple(
            True
            if shape_type in ('triangular_pyramid', 'quadrangular_pyramid')
            else bool(rng.random() < self.config.ground_contact_probability)
            for shape_type in shape_types
        )

    def _sample_start_goal(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        scenario = self.scenario
        spec = _STAGE_SPECS[self.curriculum_level]
        distance_range = (
            scenario.target_distance * spec.distance_ratio[0],
            scenario.target_distance * spec.distance_ratio[1],
        )
        # Keep the legacy altitude scale when only the vertical ceiling grows.
        altitude_reference = min(
            float(scenario.world_z_max),
            float(scenario.world_xy) / 3.0,
        )
        state_z_range = (
            altitude_reference * spec.state_z_ratio[0],
            altitude_reference * spec.state_z_ratio[1],
        )
        goal_z_range = (
            altitude_reference * spec.goal_z_ratio[0],
            altitude_reference * spec.goal_z_ratio[1],
        )
        max_height_gap = altitude_reference * spec.max_height_gap_ratio
        for _ in range(self.config.start_goal_sampling_attempts):
            distance = float(rng.uniform(*distance_range))
            mean_y = float(
                rng.uniform(
                    -spec.mean_y_ratio * scenario.world_xy,
                    spec.mean_y_ratio * scenario.world_xy,
                )
            )
            state_z = float(rng.uniform(*state_z_range))
            goal_z = float(rng.uniform(*goal_z_range))
            delta_z = goal_z - state_z
            if abs(delta_z) > max_height_gap:
                continue
            lateral_offset = float(
                rng.uniform(
                    -spec.lateral_offset_ratio * distance,
                    spec.lateral_offset_ratio * distance,
                )
            )
            remaining_squared = distance**2 - lateral_offset**2 - delta_z**2
            if remaining_squared <= 1e-6:
                continue
            delta_x = sqrt(remaining_squared)
            state = np.array(
                [
                    -0.5 * delta_x,
                    mean_y - 0.5 * lateral_offset,
                    state_z,
                    0.0,
                    rng.uniform(*spec.psi_range),
                ],
                dtype=np.float32,
            )
            goal = np.array(
                [
                    0.5 * delta_x,
                    mean_y + 0.5 * lateral_offset,
                    goal_z,
                ],
                dtype=np.float32,
            )
            if max(abs(float(state[0])), abs(float(goal[0]))) > scenario.world_xy:
                continue
            if max(abs(float(state[1])), abs(float(goal[1]))) > scenario.world_xy:
                continue
            if not (
                scenario.world_z_min < float(state[2]) <= scenario.world_z_max
                and scenario.world_z_min < float(goal[2]) <= scenario.world_z_max
            ):
                continue
            return state, goal
        return None

    def _sample_zones(
        self,
        rng: np.random.Generator,
        state: np.ndarray,
        goal: np.ndarray,
        zone_count: int,
        *,
        requested_shape_types: tuple[str, ...],
        requested_ground_contact: tuple[bool, ...],
        overlap_allowed: bool,
        require_direct_path_blocker: bool,
        rejection_counts: Counter[str] | None = None,
    ) -> tuple[list[NoFlyZone], list[float]] | None:
        if len(requested_shape_types) != zone_count:
            raise ValueError('requested_shape_types length must match zone_count.')
        if len(requested_ground_contact) != zone_count:
            raise ValueError('requested_ground_contact length must match zone_count.')
        if any(shape_type not in V2_SHAPE_TYPES for shape_type in requested_shape_types):
            raise ValueError('requested_shape_types contains an unsupported shape type.')
        if any(type(value) is not bool for value in requested_ground_contact):
            raise ValueError('requested_ground_contact must contain only bool values.')
        if type(require_direct_path_blocker) is not bool:
            raise ValueError('require_direct_path_blocker must be bool.')
        candidate_rejections = rejection_counts if rejection_counts is not None else Counter()
        zones: list[NoFlyZone] = []
        reference_scales: list[float] = []
        must_place_blocker = require_direct_path_blocker and zone_count > 0
        reference_range = _STAGE_SPECS[self.curriculum_level].reference_scale_range

        for zone_index in range(zone_count):
            shape_type = requested_shape_types[zone_index]
            ground_contact = requested_ground_contact[zone_index]
            target_point = None
            if must_place_blocker and zone_index == 0:
                fraction = float(rng.uniform(0.30, 0.70))
                target_point = state[:3] + fraction * (goal - state[:3])
            accepted_zone: NoFlyZone | None = None
            accepted_scale = 0.0
            for _ in range(self.config.zone_candidate_attempts):
                reference_scale = float(rng.uniform(*reference_range))
                candidate = self._sample_zone_candidate(
                    rng,
                    zone_id=f'zone-{zone_index:03d}',
                    shape_type=shape_type,
                    reference_scale=reference_scale,
                    target_point=target_point,
                    ground_contact=ground_contact,
                )
                if candidate is None:
                    candidate_rejections['shape_or_position_invalid'] += 1
                    continue
                if candidate.point_clearance(state[:3]) <= self.config.start_surface_clearance:
                    candidate_rejections['start_clearance'] += 1
                    continue
                if candidate.point_clearance(goal) <= self.config.goal_surface_clearance:
                    candidate_rejections['goal_clearance'] += 1
                    continue
                if not overlap_allowed and any(
                    self._aabb_overlaps(candidate, existing) for existing in zones
                ):
                    candidate_rejections['aabb_overlap_forbidden'] += 1
                    continue
                if target_point is not None and not candidate.violates_segment(
                    state[:3],
                    goal,
                    uav_radius=self.scenario.corridor_blocking_margin,
                ):
                    candidate_rejections['required_blocker_candidate_missed'] += 1
                    continue
                accepted_zone = candidate
                accepted_scale = reference_scale
                break
            if accepted_zone is None:
                return None
            zones.append(accepted_zone)
            reference_scales.append(float(accepted_scale))
        return zones, reference_scales

    def _sample_zone_candidate(
        self,
        rng: np.random.Generator,
        *,
        zone_id: str,
        shape_type: str,
        reference_scale: float,
        target_point: np.ndarray | None,
        ground_contact: bool,
    ) -> NoFlyZone | None:
        scenario = self.scenario
        world_xy = float(scenario.world_xy)
        world_z_max = float(scenario.world_z_max)

        shape: Sphere | Ellipsoid | Box | TriangularPyramid | QuadrangularPyramid
        actual_parameters: dict[str, Any]
        ground_contact: bool

        if shape_type == 'sphere':
            radius = reference_scale
            placement = self._sample_primitive_center(
                rng, radius, radius, radius, target_point, ground_contact
            )
            if placement is None:
                return None
            shape = Sphere(placement, radius)
            actual_parameters = {'radius': radius}
        elif shape_type == 'ellipsoid':
            second_horizontal = reference_scale * float(
                rng.uniform(*self.config.ellipsoid_horizontal_ratio_range)
            )
            if bool(rng.integers(0, 2)):
                radius_x, radius_y = reference_scale, second_horizontal
            else:
                radius_x, radius_y = second_horizontal, reference_scale
            radius_z = reference_scale * float(
                rng.uniform(*self.config.ellipsoid_vertical_ratio_range)
            )
            placement = self._sample_primitive_center(
                rng, radius_x, radius_y, radius_z, target_point, ground_contact
            )
            if placement is None:
                return None
            shape = Ellipsoid(placement, radius_x, radius_y, radius_z)
            actual_parameters = {
                'radius_x': radius_x,
                'radius_y': radius_y,
                'radius_z': radius_z,
            }
        elif shape_type == 'box':
            size = 2.0 * reference_scale
            placement = self._sample_primitive_center(
                rng,
                reference_scale,
                reference_scale,
                reference_scale,
                target_point,
                ground_contact,
            )
            if placement is None:
                return None
            shape = Box(placement, size, size, size)
            actual_parameters = {'size_x': size, 'size_y': size, 'size_z': size}
        elif shape_type in ('triangular_pyramid', 'quadrangular_pyramid'):
            if ground_contact is not True:
                raise ValueError('Pyramid requested_ground_contact must be True.')
            base_size_x = reference_scale * float(
                rng.uniform(*self.config.pyramid_base_ratio_range)
            )
            base_size_y = reference_scale * float(
                rng.uniform(*self.config.pyramid_base_ratio_range)
            )
            height = reference_scale * float(
                rng.uniform(*self.config.pyramid_height_ratio_range)
            )
            if height > world_z_max:
                return None
            if shape_type == 'triangular_pyramid':
                x_bounds = (-world_xy + 0.5 * base_size_x, world_xy - 0.5 * base_size_x)
                y_bounds = (-world_xy + base_size_y / 3.0, world_xy - 2.0 * base_size_y / 3.0)
            else:
                x_bounds = (-world_xy + 0.5 * base_size_x, world_xy - 0.5 * base_size_x)
                y_bounds = (-world_xy + 0.5 * base_size_y, world_xy - 0.5 * base_size_y)
            base_xy = self._sample_xy(rng, x_bounds, y_bounds, target_point)
            if base_xy is None:
                return None
            base_center = [base_xy[0], base_xy[1], 0.0]
            if shape_type == 'triangular_pyramid':
                shape = TriangularPyramid(base_center, base_size_x, base_size_y, height)
            else:
                shape = QuadrangularPyramid(base_center, base_size_x, base_size_y, height)
            actual_parameters = {
                'base_size_x': base_size_x,
                'base_size_y': base_size_y,
                'height': height,
            }
            ground_contact = True
        else:
            raise ValueError(f'Unsupported V2 shape type: {shape_type!r}.')

        zone = NoFlyZone(
            zone_id,
            shape,
            safety_margin=0.0,
            metadata={
                'generator': V2_SCENARIO_GENERATOR_NAME,
                'requested_reference_scale': float(reference_scale),
                'actual_shape_parameters': {
                    key: float(value) for key, value in actual_parameters.items()
                },
                'ground_contact': ground_contact,
            },
        )
        return zone if self._aabb_is_inside_world(zone) else None

    def _sample_primitive_center(
        self,
        rng: np.random.Generator,
        half_x: float,
        half_y: float,
        half_z: float,
        target_point: np.ndarray | None,
        grounded: bool,
    ) -> list[float] | None:
        world_xy = float(self.scenario.world_xy)
        world_z_max = float(self.scenario.world_z_max)
        if 2.0 * half_z > world_z_max + GEOMETRY_TOLERANCE:
            return None
        xy = self._sample_xy(
            rng,
            (-world_xy + half_x, world_xy - half_x),
            (-world_xy + half_y, world_xy - half_y),
            target_point,
        )
        if xy is None:
            return None
        if grounded:
            center_z = half_z
        else:
            suspended_gap = max(1e-6, world_z_max * 1e-6)
            lower = half_z + suspended_gap
            upper = world_z_max - half_z
            if lower > upper:
                return None
            if target_point is None:
                center_z = float(rng.uniform(lower, upper))
            else:
                center_z = float(np.clip(float(target_point[2]), lower, upper))
        return [float(xy[0]), float(xy[1]), float(center_z)]

    @staticmethod
    def _sample_xy(
        rng: np.random.Generator,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        target_point: np.ndarray | None,
    ) -> tuple[float, float] | None:
        if x_bounds[0] > x_bounds[1] or y_bounds[0] > y_bounds[1]:
            return None
        if target_point is None:
            return float(rng.uniform(*x_bounds)), float(rng.uniform(*y_bounds))
        x_value = float(target_point[0])
        y_value = float(target_point[1])
        if not (x_bounds[0] <= x_value <= x_bounds[1]):
            return None
        if not (y_bounds[0] <= y_value <= y_bounds[1]):
            return None
        return x_value, y_value

    def _aabb_is_inside_world(self, zone: NoFlyZone) -> bool:
        bounds = zone.shape.bounding_box()
        return bool(
            bounds.min_corner[0] >= -self.scenario.world_xy - GEOMETRY_TOLERANCE
            and bounds.min_corner[1] >= -self.scenario.world_xy - GEOMETRY_TOLERANCE
            and bounds.min_corner[2] >= -GEOMETRY_TOLERANCE
            and bounds.max_corner[0] <= self.scenario.world_xy + GEOMETRY_TOLERANCE
            and bounds.max_corner[1] <= self.scenario.world_xy + GEOMETRY_TOLERANCE
            and bounds.max_corner[2] <= self.scenario.world_z_max + GEOMETRY_TOLERANCE
        )

    @staticmethod
    def _aabb_overlaps(first: NoFlyZone, second: NoFlyZone) -> bool:
        first_bounds = first.shape.bounding_box()
        second_bounds = second.shape.bounding_box()
        return bool(
            np.all(first_bounds.min_corner <= second_bounds.max_corner + GEOMETRY_TOLERANCE)
            and np.all(second_bounds.min_corner <= first_bounds.max_corner + GEOMETRY_TOLERANCE)
        )

    def _count_direct_path_blockers(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        zones: list[NoFlyZone],
    ) -> int:
        return sum(
            zone.violates_segment(
                state[:3],
                goal,
                uav_radius=self.scenario.corridor_blocking_margin,
            )
            for zone in zones
        )

    def _aabb_overlap_pair_count(self, zones: list[NoFlyZone]) -> int:
        return sum(
            self._aabb_overlaps(first, second)
            for index, first in enumerate(zones)
            for second in zones[index + 1 :]
        )

    def _scenario_metadata(
        self,
        *,
        scenario_seed: int,
        requested_zone_count: int,
        requested_shape_types: tuple[str, ...],
        requested_ground_contact: tuple[bool, ...],
        zones: list[NoFlyZone],
        reference_scales: list[float],
        overlap_allowed: bool,
        requested_direct_path_blocker: bool,
        direct_path_blocker_count: int,
        feasibility_type: str,
        feasibility_passed: bool,
        feasibility_examined_nodes: int,
        feasibility_edge_checks: int,
        generation_attempts: int,
        rejection_counts: Mapping[str, int],
    ) -> dict[str, Any]:
        shape_counts = {shape_type: 0 for shape_type in V2_SHAPE_TYPES}
        for zone in zones:
            shape_counts[zone.shape.to_dict()['shape_type']] += 1
        return {
            'generator': V2_SCENARIO_GENERATOR_NAME,
            'generator_version': V2_SCENARIO_GENERATOR_VERSION,
            'scenario_seed': int(scenario_seed),
            'requested_curriculum_level': self.curriculum_level,
            'effective_curriculum_level': self.curriculum_level,
            'requested_zone_count': int(requested_zone_count),
            'effective_zone_count': len(zones),
            'shape_counts': shape_counts,
            'requested_shape_types': list(requested_shape_types),
            'requested_ground_contact': list(requested_ground_contact),
            'requested_reference_scales': [float(value) for value in reference_scales],
            'overlap_allowed': bool(overlap_allowed),
            'aabb_overlap_pair_count': int(self._aabb_overlap_pair_count(zones)),
            'requested_direct_path_blocker': bool(requested_direct_path_blocker),
            'direct_path_blocker_count': int(direct_path_blocker_count),
            'feasibility_check': feasibility_type,
            'feasibility_passed': bool(feasibility_passed),
            'feasibility_examined_nodes': int(feasibility_examined_nodes),
            'feasibility_edge_checks': int(feasibility_edge_checks),
            'generation_attempts': int(generation_attempts),
            'rejection_counts': {
                key: int(value) for key, value in sorted(rejection_counts.items())
            },
        }
