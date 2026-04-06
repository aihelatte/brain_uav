"""Curriculum helpers for explicit easy/easy_two_zone/medium/hard training stages."""

from __future__ import annotations

from typing import Iterable

CURRICULUM_LEVELS = ('easy', 'easy_two_zone', 'medium', 'hard')
DEFAULT_CURRICULUM_MIXES: dict[str, dict[str, float]] = {
    'easy': {'easy': 1.0},
    'easy_two_zone': {'easy_two_zone': 0.8, 'easy': 0.2},
    'medium': {'medium': 0.8, 'easy_two_zone': 0.2},
    'hard': {'hard': 0.7, 'medium': 0.2, 'easy_two_zone': 0.1},
}


def default_curriculum_mix(level: str) -> dict[str, float]:
    if level not in DEFAULT_CURRICULUM_MIXES:
        raise ValueError(f'Unsupported curriculum level: {level}')
    return dict(DEFAULT_CURRICULUM_MIXES[level])


def normalize_curriculum_mix(mix: dict[str, float] | None, *, fallback_level: str) -> dict[str, float]:
    """Normalize and validate a curriculum mix."""

    raw = mix or default_curriculum_mix(fallback_level)
    cleaned = {key: float(value) for key, value in raw.items() if float(value) > 0.0}
    if not cleaned:
        raise ValueError('Curriculum mix must contain at least one positive weight.')
    invalid = sorted(set(cleaned) - set(CURRICULUM_LEVELS))
    if invalid:
        raise ValueError(f'Unsupported curriculum levels in mix: {invalid}')
    total = sum(cleaned.values())
    return {key: value / total for key, value in cleaned.items()}


def parse_curriculum_mix(raw: str | None, *, fallback_level: str) -> dict[str, float]:
    """Parse a CLI curriculum mix string like ``hard:0.7,medium:0.2,easy_two_zone:0.1``."""

    if raw is None or not raw.strip():
        return default_curriculum_mix(fallback_level)

    parsed: dict[str, float] = {}
    for item in raw.split(','):
        token = item.strip()
        if not token:
            continue
        if ':' not in token:
            raise ValueError(f'Invalid curriculum mix token: {token!r}')
        level, weight = token.split(':', 1)
        parsed[level.strip()] = float(weight.strip())
    return normalize_curriculum_mix(parsed, fallback_level=fallback_level)


def describe_curriculum_mix(mix: dict[str, float]) -> str:
    ordered: Iterable[str] = (level for level in CURRICULUM_LEVELS if level in mix)
    return ', '.join(f'{level}:{mix[level]:.2f}' for level in ordered)
