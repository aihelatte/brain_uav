"""Strict curriculum, seed, BC and noise schedules for formal V2 TD3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np


V2_TD3_STAGES = ('easy', 'medium', 'hard')
DEFAULT_V2_TD3_CURRICULUM_MIXES: dict[str, dict[str, float]] = {
    'easy': {'easy': 1.0},
    'medium': {'medium': 0.8, 'easy': 0.2},
    'hard': {'hard': 0.7, 'medium': 0.2, 'easy': 0.1},
}

_STAGE_IDS = {name: index for index, name in enumerate(V2_TD3_STAGES)}
_COMPONENT_IDS = {
    'model': 1,
    'curriculum': 2,
    'easy_generator': 3,
    'medium_generator': 4,
    'hard_generator': 5,
    'replay': 6,
    'exploration': 7,
}
_STAGE_DEFAULTS = {
    'easy': (750_000, 1.5e-4, 2.5e-4),
    'medium': (750_000, 1.5e-4, 2.5e-4),
    'hard': (1_000_000, 1.125e-4, 2.125e-4),
}


def _stage(value: Any) -> str:
    if value not in V2_TD3_STAGES:
        raise ValueError('V2 TD3 stage must be easy, medium, or hard.')
    return str(value)


def _nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a non-negative integer.')
    return value


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a finite non-negative number.')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite non-negative number.') from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be a finite non-negative number.')
    return result


def v2_stage_sequence(max_stage: str) -> tuple[str, ...]:
    stage = _stage(max_stage)
    return V2_TD3_STAGES[: _STAGE_IDS[stage] + 1]


def normalize_v2_curriculum_mix(
    mix: Mapping[str, float],
    *,
    stage: str,
) -> dict[str, float]:
    """Validate and normalize a stage mix without accepting legacy levels."""

    current = _stage(stage)
    if not isinstance(mix, Mapping) or not mix:
        raise ValueError('V2 curriculum mix must be a non-empty mapping.')
    allowed = set(v2_stage_sequence(current))
    if any(type(key) is not str or key not in allowed for key in mix):
        raise ValueError(f'V2 curriculum mix for {current} contains an unsupported level.')
    values: dict[str, float] = {}
    for level, raw_weight in mix.items():
        weight = _finite_nonnegative(raw_weight, name=f'curriculum weight {level!r}')
        if weight > 0.0:
            values[level] = weight
    if not values:
        raise ValueError('V2 curriculum mix must contain a positive weight.')
    total = sum(values.values())
    return {
        level: values[level] / total
        for level in V2_TD3_STAGES
        if level in values
    }


def default_v2_curriculum_mix(stage: str) -> dict[str, float]:
    current = _stage(stage)
    return dict(DEFAULT_V2_TD3_CURRICULUM_MIXES[current])


def derive_v2_component_seed(base_seed: int, stage: str, component: str) -> int:
    """Derive a stable uint32 seed without Python's process-randomized hash."""

    seed = _nonnegative_int(base_seed, name='base_seed')
    current = _stage(stage)
    if component not in _COMPONENT_IDS:
        raise ValueError(f'Unsupported V2 random component: {component!r}.')
    sequence = np.random.SeedSequence(
        [seed, _STAGE_IDS[current], _COMPONENT_IDS[component]]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class V2CurriculumSelector:
    """Deterministic episode-level categorical selector with an owned RNG."""

    def __init__(self, mix: Mapping[str, float], *, seed: int) -> None:
        if not isinstance(mix, Mapping) or not mix:
            raise ValueError('V2 curriculum mix must be a non-empty mapping.')
        unknown = set(mix) - set(V2_TD3_STAGES)
        if unknown:
            raise ValueError(f'Unsupported V2 curriculum levels: {sorted(unknown)}.')
        inferred_stage = max(mix, key=lambda name: _STAGE_IDS.get(name, -1))
        self.mix = normalize_v2_curriculum_mix(mix, stage=inferred_stage)
        self.levels = tuple(self.mix)
        self.probabilities = np.asarray(
            [self.mix[level] for level in self.levels], dtype=np.float64
        )
        self.seed = _nonnegative_int(seed, name='seed')
        self.rng = np.random.default_rng(self.seed)

    def sample(self) -> str:
        return str(self.rng.choice(self.levels, p=self.probabilities))


def v2_bc_lambda(stage_local_step: int) -> float:
    step = _nonnegative_int(stage_local_step, name='stage_local_step')
    if step < 75_000:
        return 500.0
    if step < 150_000:
        return 150.0
    if step < 250_000:
        return 30.0
    return 5.0


@dataclass(frozen=True, slots=True)
class V2NoiseSchedule:
    exploration_initial: float = 0.020
    policy_initial: float = 0.015
    clip_initial: float = 0.030
    exploration_final: float = 0.005
    policy_final: float = 0.006
    clip_final: float = 0.012
    decay_fraction: float = 0.5

    def __post_init__(self) -> None:
        pairs = (
            ('exploration', self.exploration_initial, self.exploration_final),
            ('policy', self.policy_initial, self.policy_final),
            ('clip', self.clip_initial, self.clip_final),
        )
        for name, initial, final in pairs:
            initial_value = _finite_nonnegative(initial, name=f'{name}_initial')
            final_value = _finite_nonnegative(final, name=f'{name}_final')
            if final_value > initial_value:
                raise ValueError(f'{name}_final must not exceed {name}_initial.')
            object.__setattr__(self, f'{name}_initial', initial_value)
            object.__setattr__(self, f'{name}_final', final_value)
        decay = _finite_nonnegative(self.decay_fraction, name='decay_fraction')
        if decay <= 0.0 or decay > 1.0:
            raise ValueError('decay_fraction must be in (0, 1].')
        object.__setattr__(self, 'decay_fraction', decay)

    def values(self, stage_local_step: int, *, max_steps: int) -> tuple[float, float, float]:
        step = _nonnegative_int(stage_local_step, name='stage_local_step')
        maximum = _nonnegative_int(max_steps, name='max_steps')
        if maximum == 0:
            raise ValueError('max_steps must be greater than zero.')
        decay_steps = maximum * self.decay_fraction
        fraction = min(float(step) / decay_steps, 1.0)
        if fraction >= 1.0:
            return self.exploration_final, self.policy_final, self.clip_final
        return (
            self.exploration_initial
            + (self.exploration_final - self.exploration_initial) * fraction,
            self.policy_initial + (self.policy_final - self.policy_initial) * fraction,
            self.clip_initial + (self.clip_final - self.clip_initial) * fraction,
        )


def v2_stage_defaults(stage: str) -> tuple[int, float, float]:
    return _STAGE_DEFAULTS[_stage(stage)]


__all__ = [
    'DEFAULT_V2_TD3_CURRICULUM_MIXES',
    'V2CurriculumSelector',
    'V2NoiseSchedule',
    'V2_TD3_STAGES',
    'default_v2_curriculum_mix',
    'derive_v2_component_seed',
    'normalize_v2_curriculum_mix',
    'v2_bc_lambda',
    'v2_stage_defaults',
    'v2_stage_sequence',
]
