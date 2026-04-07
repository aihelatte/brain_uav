"""Benchmark suite generation and loading helpers.

??????????????????????????? benchmark ???
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK_SUITE_NAME = 'fixed_benchmark_suite_v2'
DEFAULT_BENCHMARK_SUITE_PATH = Path('outputs/benchmarks') / f'{DEFAULT_BENCHMARK_SUITE_NAME}.json'
BENCHMARK_CATEGORIES = ('single_detour', 'double_channel', 'boundary_margin', 'wall_pressure')


@dataclass(slots=True)
class NamedScenario:
    """One fixed benchmark scenario plus its metadata."""

    scenario_id: str
    category: str
    name: str
    description: str
    scenario: dict[str, Any]
    corridor_width: float | None = None
    min_clearance_to_boundary: float | None = None
    difficulty_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scenario_payload(
    state: list[float],
    goal: list[float],
    zones: list[dict[str, Any]],
    *,
    category: str,
    scenario_id: str,
    label: str,
    description: str,
    corridor_width: float | None = None,
    min_clearance_to_boundary: float | None = None,
    difficulty_score: float | None = None,
) -> NamedScenario:
    scenario = {
        'state': state,
        'goal': goal,
        'zones': zones,
        'curriculum_level': 'benchmark',
        'scenario_id': scenario_id,
        'category': category,
        'scenario_label': label,
        'description': description,
    }
    if corridor_width is not None:
        scenario['corridor_width'] = float(corridor_width)
    if min_clearance_to_boundary is not None:
        scenario['min_clearance_to_boundary'] = float(min_clearance_to_boundary)
    if difficulty_score is not None:
        scenario['difficulty_score'] = float(difficulty_score)
    return NamedScenario(
        scenario_id=scenario_id,
        category=category,
        name=label,
        description=description,
        scenario=scenario,
        corridor_width=None if corridor_width is None else float(corridor_width),
        min_clearance_to_boundary=None if min_clearance_to_boundary is None else float(min_clearance_to_boundary),
        difficulty_score=None if difficulty_score is None else float(difficulty_score),
    )


def _distance_point_to_segment(point: tuple[float, float], start: tuple[float, float], goal: tuple[float, float]) -> float:
    px, py = point
    sx, sy = start
    gx, gy = goal
    dx = gx - sx
    dy = gy - sy
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-6:
        return math.hypot(px - sx, py - sy)
    t = ((px - sx) * dx + (py - sy) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = sx + t * dx
    proj_y = sy + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _difficulty(zone_count: int, *, corridor_width: float | None = None, min_clearance: float | None = None) -> float:
    score = 1.0 + 0.35 * zone_count
    if corridor_width is not None:
        score += max(0.0, (260.0 - corridor_width) / 180.0)
    if min_clearance is not None:
        score += max(0.0, (70.0 - min_clearance) / 60.0)
    return round(score, 3)


def _make_single_detour(rng, idx: int) -> NamedScenario:
    state_y = float(rng.uniform(-90.0, 90.0))
    goal_y = float(state_y + rng.uniform(-45.0, 45.0))
    state_z = float(rng.uniform(110.0, 160.0))
    goal_z = float(rng.uniform(110.0, 160.0))
    state = [-650.0, state_y, state_z, 0.0, 0.0]
    goal = [650.0, goal_y, goal_z]
    radius = float(rng.uniform(120.0, 190.0))
    center_x = float(rng.uniform(-80.0, 80.0))
    mean_y = 0.5 * (state_y + goal_y)
    lateral = float(rng.uniform(-120.0, 120.0))
    center_y = mean_y + lateral
    blocker_distance = _distance_point_to_segment((center_x, center_y), (state[0], state[1]), (goal[0], goal[1]))
    if blocker_distance > radius * 0.72:
        center_y = mean_y + math.copysign(radius * 0.58, lateral if abs(lateral) > 1e-3 else 1.0)
    zones = [{'center_xy': [center_x, center_y], 'radius': radius}]
    scenario_id = f'SD{idx:03d}'
    return _scenario_payload(
        state,
        goal,
        zones,
        category='single_detour',
        scenario_id=scenario_id,
        label=f'single_detour_{idx:03d}',
        description='Single hemisphere clearly blocks the direct route and forces one decisive detour.',
        difficulty_score=_difficulty(1),
    )


def _make_double_channel(rng, idx: int) -> NamedScenario:
    state_y = float(rng.uniform(-70.0, 70.0))
    goal_y = float(state_y + rng.uniform(-35.0, 35.0))
    z_ref = float(rng.uniform(115.0, 155.0))
    state = [-650.0, state_y, z_ref, 0.0, 0.0]
    goal = [650.0, goal_y, float(z_ref + rng.uniform(-12.0, 12.0))]
    radius_1 = float(rng.uniform(110.0, 180.0))
    radius_2 = float(rng.uniform(110.0, 180.0))
    corridor_width = float(rng.uniform(140.0, 260.0))
    x_base = float(rng.uniform(-40.0, 40.0))
    x_offset = float(rng.uniform(-60.0, 60.0))
    centerline_y = 0.5 * (state_y + goal_y)
    total_sep = radius_1 + radius_2 + corridor_width
    center_1 = [x_base, centerline_y - 0.5 * total_sep]
    center_2 = [x_base + x_offset, centerline_y + 0.5 * total_sep]
    zones = [
        {'center_xy': center_1, 'radius': radius_1},
        {'center_xy': center_2, 'radius': radius_2},
    ]
    scenario_id = f'DC{idx:03d}'
    return _scenario_payload(
        state,
        goal,
        zones,
        category='double_channel',
        scenario_id=scenario_id,
        label=f'double_channel_{idx:03d}',
        description='Two hemispheres form a pressured but traversable channel.',
        corridor_width=corridor_width,
        difficulty_score=_difficulty(2, corridor_width=corridor_width),
    )


def _make_boundary_margin(rng, idx: int) -> NamedScenario:
    state_y = float(rng.uniform(-250.0, 250.0))
    goal_y = float(state_y + rng.uniform(-40.0, 40.0))
    z_ref = float(rng.uniform(110.0, 150.0))
    state = [-650.0, state_y, z_ref, 0.0, 0.0]
    goal = [650.0, goal_y, float(z_ref + rng.uniform(-10.0, 10.0))]
    radius = float(rng.uniform(110.0, 170.0))
    min_clearance = float(rng.uniform(20.0, 70.0))
    path_y = 0.5 * (state_y + goal_y)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    main_center = [float(rng.uniform(-60.0, 120.0)), path_y + sign * (radius + min_clearance)]
    zones = [{'center_xy': main_center, 'radius': radius}]
    if rng.random() < 0.45:
        aux_radius = float(rng.uniform(110.0, 150.0))
        aux_center = [float(rng.uniform(180.0, 360.0)), path_y - sign * float(rng.uniform(180.0, 320.0))]
        zones.append({'center_xy': aux_center, 'radius': aux_radius})
    scenario_id = f'BM{idx:03d}'
    return _scenario_payload(
        state,
        goal,
        zones,
        category='boundary_margin',
        scenario_id=scenario_id,
        label=f'boundary_margin_{idx:03d}',
        description='The preferred route succeeds only with a small but feasible boundary margin.',
        min_clearance_to_boundary=min_clearance,
        difficulty_score=_difficulty(len(zones), min_clearance=min_clearance),
    )


def _make_wall_pressure(rng, idx: int) -> NamedScenario:
    corridor_width = float(rng.uniform(120.0, 220.0))
    corridor_center = float(rng.uniform(-120.0, 120.0))
    radius_base = float(rng.uniform(105.0, 145.0))
    state_y = float(corridor_center + rng.uniform(-25.0, 25.0))
    goal_y = float(corridor_center + rng.uniform(-25.0, 25.0))
    z_ref = float(rng.uniform(120.0, 160.0))
    state = [-670.0, state_y, z_ref, 0.0, 0.0]
    goal = [670.0, goal_y, float(z_ref + rng.uniform(-12.0, 12.0))]

    zones: list[dict[str, Any]] = []
    lower_count = int(rng.integers(2, 4))
    upper_count = int(rng.integers(2, 4))
    x_lower = float(rng.uniform(-120.0, -10.0))
    x_upper = float(rng.uniform(40.0, 180.0))
    lower_start = corridor_center - 0.5 * corridor_width - float(rng.uniform(90.0, 140.0))
    upper_start = corridor_center + 0.5 * corridor_width + float(rng.uniform(90.0, 140.0))
    vertical_step = float(rng.uniform(150.0, 210.0))

    for i in range(lower_count):
        zones.append(
            {
                'center_xy': [x_lower + float(rng.uniform(-30.0, 30.0)), lower_start - i * vertical_step],
                'radius': float(radius_base + rng.uniform(-12.0, 12.0)),
            }
        )
    for i in range(upper_count):
        zones.append(
            {
                'center_xy': [x_upper + float(rng.uniform(-30.0, 30.0)), upper_start + i * vertical_step],
                'radius': float(radius_base + rng.uniform(-12.0, 12.0)),
            }
        )
    if len(zones) < 4:
        zones.append(
            {
                'center_xy': [float(rng.uniform(220.0, 320.0)), corridor_center + float(rng.uniform(-260.0, 260.0))],
                'radius': float(radius_base + rng.uniform(-8.0, 8.0)),
            }
        )
    if len(zones) > 7:
        zones = zones[:7]

    scenario_id = f'WP{idx:03d}'
    return _scenario_payload(
        state,
        goal,
        zones,
        category='wall_pressure',
        scenario_id=scenario_id,
        label=f'wall_pressure_{idx:03d}',
        description='A dense obstacle wall leaves one pressured corridor and punishes local-minimum behavior.',
        corridor_width=corridor_width,
        difficulty_score=_difficulty(len(zones), corridor_width=corridor_width),
    )


def generate_benchmark_suite(
    *,
    seed: int = 20260407,
    count_per_category: int = 100,
    suite_name: str = DEFAULT_BENCHMARK_SUITE_NAME,
) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    categories = {
        'single_detour': [],
        'double_channel': [],
        'boundary_margin': [],
        'wall_pressure': [],
    }
    for idx in range(1, count_per_category + 1):
        categories['single_detour'].append(_make_single_detour(rng, idx))
        categories['double_channel'].append(_make_double_channel(rng, idx))
        categories['boundary_margin'].append(_make_boundary_margin(rng, idx))
        categories['wall_pressure'].append(_make_wall_pressure(rng, idx))

    scenarios: list[NamedScenario] = []
    for category in BENCHMARK_CATEGORIES:
        scenarios.extend(categories[category])

    return {
        'suite_name': suite_name,
        'seed': seed,
        'count_per_category': count_per_category,
        'total_scenarios': len(scenarios),
        'categories': list(BENCHMARK_CATEGORIES),
        'scenarios': [item.to_dict() for item in scenarios],
    }


def save_benchmark_suite(payload: dict[str, Any], path: str | Path = DEFAULT_BENCHMARK_SUITE_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return target


def load_benchmark_suite(path: str | Path = DEFAULT_BENCHMARK_SUITE_PATH) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f'Benchmark suite not found: {source}. Run generate_benchmark_suite first to freeze the evaluation set.'
        )
    return json.loads(source.read_text(encoding='utf-8'))


def build_benchmark_scenarios(path: str | Path = DEFAULT_BENCHMARK_SUITE_PATH) -> list[NamedScenario]:
    payload = load_benchmark_suite(path)
    scenarios: list[NamedScenario] = []
    for item in payload['scenarios']:
        scenarios.append(
            NamedScenario(
                scenario_id=item['scenario_id'],
                category=item['category'],
                name=item['name'],
                description=item['description'],
                scenario=item['scenario'],
                corridor_width=item.get('corridor_width'),
                min_clearance_to_boundary=item.get('min_clearance_to_boundary'),
                difficulty_score=item.get('difficulty_score'),
            )
        )
    return scenarios
