"""Benchmark suite generation and loading helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ScenarioConfig


DEFAULT_BENCHMARK_SUITE_NAME = 'fixed_benchmark_suite_v2'
DEFAULT_BENCHMARK_SUITE_PATH = Path('outputs/benchmarks') / f'{DEFAULT_BENCHMARK_SUITE_NAME}.json'
BENCHMARK_CATEGORIES = ('single_detour', 'double_channel', 'boundary_margin', 'wall_pressure')
BENCHMARK_ZONE_RADIUS_RANGE_KM = (200.0, 250.0)
BENCHMARK_START_GOAL_DISTANCE_RANGE_KM = (1700.0, 2400.0)
BENCHMARK_MIN_ZONE_SURFACE_GAP_KM = 60.0


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


def _distance_3d(a: list[float], b: list[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _distance_2d(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


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


def _point_inside_zone(point: list[float], zone: dict[str, Any]) -> bool:
    center = zone['center_xy']
    radius = float(zone['radius'])
    distance_sq = (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2 + point[2] ** 2
    return distance_sq <= radius * radius


def _zone_surface_gaps(zones: list[dict[str, Any]]) -> list[float]:
    gaps: list[float] = []
    for idx, zone_a in enumerate(zones):
        for zone_b in zones[idx + 1 :]:
            center_distance = _distance_2d(zone_a['center_xy'], zone_b['center_xy'])
            gaps.append(center_distance - float(zone_a['radius']) - float(zone_b['radius']))
    return gaps


def _difficulty(zone_count: int, *, corridor_width: float | None = None, min_clearance: float | None = None) -> float:
    score = 1.0 + 0.35 * zone_count
    if corridor_width is not None:
        score += max(0.0, (260.0 - corridor_width) / 180.0)
    if min_clearance is not None:
        score += max(0.0, (260.0 - min_clearance) / 260.0)
    return round(score, 3)


def compute_scene_metadata(scene: dict[str, Any], cfg: ScenarioConfig) -> dict[str, Any]:
    state = scene['state']
    goal = scene['goal']
    radii = [float(zone['radius']) for zone in scene['zones']]
    gaps = _zone_surface_gaps(scene['zones'])
    return {
        'distance_unit': 'km',
        'time_unit': 's',
        'speed_km_s': float(cfg.speed),
        'dt_s': float(cfg.dt),
        'max_time_s': float(cfg.max_time_s),
        'max_steps': int(cfg.max_steps),
        'goal_radius_km': float(cfg.goal_radius),
        'start_goal_distance_km': _distance_3d(state[:3], goal),
        'start_goal_horizontal_distance_km': _distance_2d(state[:2], goal[:2]),
        'zone_radius_range_km': [min(radii), max(radii)] if radii else [None, None],
        'min_zone_surface_gap_km': min(gaps) if gaps else None,
    }


def validate_benchmark_scene(scene: dict[str, Any], cfg: ScenarioConfig) -> None:
    category = scene['category']
    state = scene['state']
    goal = scene['goal']
    zones = scene['zones']
    start = state[:3]

    for label, point in (('start', start), ('goal', goal)):
        if abs(point[0]) > cfg.world_xy or abs(point[1]) > cfg.world_xy:
            raise ValueError(f'{scene["id"]}: {label} is outside X/Y boundary.')
        if not (cfg.world_z_min < point[2] < cfg.world_z_max):
            raise ValueError(f'{scene["id"]}: {label} has illegal Z height.')
    if abs(goal[2] - start[2]) > cfg.max_start_goal_height_gap:
        raise ValueError(f'{scene["id"]}: start/goal height gap is too large.')

    distance = _distance_3d(start, goal)
    min_distance, max_distance = BENCHMARK_START_GOAL_DISTANCE_RANGE_KM
    if not (min_distance <= distance <= max_distance):
        raise ValueError(f'{scene["id"]}: start-goal distance {distance:.3f} km is outside benchmark range.')

    blockers = 0
    for zone in zones:
        radius = float(zone['radius'])
        center = zone['center_xy']
        if not (BENCHMARK_ZONE_RADIUS_RANGE_KM[0] <= radius <= BENCHMARK_ZONE_RADIUS_RANGE_KM[1]):
            raise ValueError(f'{scene["id"]}: zone radius {radius:.3f} km is outside benchmark range.')
        if abs(center[0]) + radius > cfg.world_xy or abs(center[1]) + radius > cfg.world_xy:
            raise ValueError(f'{scene["id"]}: zone exceeds X/Y boundary.')
        if _point_inside_zone(start, zone) or _point_inside_zone(goal, zone):
            raise ValueError(f'{scene["id"]}: start or goal is inside a no-fly-zone.')
        if _distance_point_to_segment(tuple(center), tuple(start[:2]), tuple(goal[:2])) <= radius + cfg.corridor_blocking_margin:
            blockers += 1

    gaps = _zone_surface_gaps(zones)
    for gap in gaps:
        if gap <= BENCHMARK_MIN_ZONE_SURFACE_GAP_KM:
            raise ValueError(f'{scene["id"]}: zone surface gap {gap:.3f} km is too small.')

    if category == 'single_detour' and (len(zones) != 1 or blockers != 1):
        raise ValueError(f'{scene["id"]}: single_detour must contain one corridor blocker.')
    if category == 'double_channel':
        if len(zones) != 2:
            raise ValueError(f'{scene["id"]}: double_channel must contain two channel-forming zones.')
        corridor_width = scene.get('corridor_width')
        if corridor_width is None or not (120.0 <= float(corridor_width) <= 260.0):
            raise ValueError(f'{scene["id"]}: double_channel corridor width is outside the intended range.')
        centerline_y = 0.5 * (start[1] + goal[1])
        if (zones[0]['center_xy'][1] - centerline_y) * (zones[1]['center_xy'][1] - centerline_y) >= 0.0:
            raise ValueError(f'{scene["id"]}: double_channel zones must sit on opposite sides of the corridor.')
    if category == 'boundary_margin':
        min_clearance = scene.get('min_clearance_to_boundary')
        if min_clearance is None or float(min_clearance) < 180.0:
            raise ValueError(f'{scene["id"]}: boundary_margin clearance is too small.')
    if category == 'wall_pressure':
        if len(zones) < 3 or blockers < 1:
            raise ValueError(f'{scene["id"]}: wall_pressure must contain multiple pressured blockers.')
        if max(abs(zone['center_xy'][1]) for zone in zones) + max(float(zone['radius']) for zone in zones) >= cfg.world_xy:
            raise ValueError(f'{scene["id"]}: wall_pressure leaves no around-wall margin.')


def _scenario_payload(
    state: list[float],
    goal: list[float],
    zones: list[dict[str, Any]],
    *,
    category: str,
    scenario_id: str,
    label: str,
    description: str,
    cfg: ScenarioConfig,
    corridor_width: float | None = None,
    min_clearance_to_boundary: float | None = None,
    difficulty_score: float | None = None,
) -> NamedScenario:
    scenario = {
        'id': scenario_id,
        'state': [float(value) for value in state],
        'goal': [float(value) for value in goal],
        'zones': [
            {'center_xy': [float(zone['center_xy'][0]), float(zone['center_xy'][1])], 'radius': float(zone['radius'])}
            for zone in zones
        ],
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
    scenario.update(compute_scene_metadata(scenario, cfg))
    validate_benchmark_scene(scenario, cfg)
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


def _sample_radius(rng: Any) -> float:
    return float(rng.uniform(*BENCHMARK_ZONE_RADIUS_RANGE_KM))


def _sample_height_pair(rng: Any, cfg: ScenarioConfig) -> tuple[float, float]:
    for _ in range(100):
        start_z = float(rng.uniform(*cfg.start_z_range))
        goal_z = float(rng.uniform(*cfg.goal_z_range))
        if abs(goal_z - start_z) <= cfg.max_start_goal_height_gap:
            return start_z, goal_z
    raise ValueError('Failed to sample a legal benchmark height pair.')


def _sample_start_goal(
    rng: Any,
    cfg: ScenarioConfig,
    *,
    y_range: tuple[float, float],
    y_delta_range: tuple[float, float] = (-90.0, 90.0),
    x_abs_range: tuple[float, float] = (920.0, 1120.0),
) -> tuple[list[float], list[float]]:
    for _ in range(100):
        start_z, goal_z = _sample_height_pair(rng, cfg)
        start_y = float(rng.uniform(*y_range))
        goal_y = start_y + float(rng.uniform(*y_delta_range))
        state = [-float(rng.uniform(*x_abs_range)), start_y, start_z, 0.0, 0.0]
        goal = [float(rng.uniform(*x_abs_range)), goal_y, goal_z]
        distance = _distance_3d(state[:3], goal)
        if BENCHMARK_START_GOAL_DISTANCE_RANGE_KM[0] <= distance <= BENCHMARK_START_GOAL_DISTANCE_RANGE_KM[1]:
            return state, goal
    raise ValueError('Failed to sample a legal benchmark start/goal pair.')


def _point_on_segment(start: list[float], goal: list[float], t: float) -> tuple[float, float]:
    return (
        float(start[0] + t * (goal[0] - start[0])),
        float(start[1] + t * (goal[1] - start[1])),
    )


def _geometry_key(scene: dict[str, Any], precision: int = 1) -> tuple[Any, ...]:
    zones = tuple(
        (round(zone['center_xy'][0], precision), round(zone['center_xy'][1], precision), round(zone['radius'], precision))
        for zone in scene['zones']
    )
    return (
        scene['category'],
        tuple(round(value, precision) for value in scene['state']),
        tuple(round(value, precision) for value in scene['goal']),
        zones,
    )


def _make_single_detour(rng: Any, idx: int, cfg: ScenarioConfig) -> NamedScenario:
    state, goal = _sample_start_goal(rng, cfg, y_range=(-180.0, 180.0), y_delta_range=(-70.0, 70.0))
    radius = _sample_radius(rng)
    base_x, base_y = _point_on_segment(state, goal, float(rng.uniform(0.35, 0.65)))
    center_y = base_y + float(rng.uniform(-0.55 * radius, 0.55 * radius))
    center_x = base_x + float(rng.uniform(-120.0, 120.0))
    zones = [{'center_xy': [center_x, center_y], 'radius': radius}]
    scenario_id = f'SD{idx:03d}'
    return _scenario_payload(
        state,
        goal,
        zones,
        category='single_detour',
        scenario_id=scenario_id,
        label=f'single_detour_{idx:03d}',
        description='One 200-250 km hemisphere sits on the direct corridor and forces a decisive detour.',
        cfg=cfg,
        difficulty_score=_difficulty(1),
    )


def _make_double_channel(rng: Any, idx: int, cfg: ScenarioConfig) -> NamedScenario:
    state, goal = _sample_start_goal(rng, cfg, y_range=(-180.0, 180.0), y_delta_range=(-80.0, 80.0))
    centerline_y = 0.5 * (state[1] + goal[1])
    radius_1 = _sample_radius(rng)
    radius_2 = _sample_radius(rng)
    corridor_width = float(rng.uniform(130.0, 240.0))
    total_sep = radius_1 + radius_2 + corridor_width
    x_base, _ = _point_on_segment(state, goal, float(rng.uniform(0.38, 0.62)))
    zones = [
        {'center_xy': [x_base, centerline_y - 0.5 * total_sep], 'radius': radius_1},
        {
            'center_xy': [
                x_base + float(rng.uniform(-90.0, 90.0)),
                centerline_y + 0.5 * total_sep,
            ],
            'radius': radius_2,
        },
    ]
    scenario_id = f'DC{idx:03d}'
    return _scenario_payload(
        state,
        goal,
        zones,
        category='double_channel',
        scenario_id=scenario_id,
        label=f'double_channel_{idx:03d}',
        description='Two formal-scale hemispheres form a traversable channel without overlap.',
        cfg=cfg,
        corridor_width=corridor_width,
        difficulty_score=_difficulty(2, corridor_width=corridor_width),
    )


def _make_boundary_margin(rng: Any, idx: int, cfg: ScenarioConfig) -> NamedScenario:
    sign = 1.0 if float(rng.random()) < 0.5 else -1.0
    state, goal = _sample_start_goal(
        rng,
        cfg,
        y_range=(sign * 620.0, sign * 900.0) if sign > 0 else (sign * 900.0, sign * 620.0),
        y_delta_range=(sign * -70.0, sign * 70.0) if sign > 0 else (sign * 70.0, sign * -70.0),
    )
    radius = _sample_radius(rng)
    boundary_clearance = float(rng.uniform(200.0, 360.0))
    center_y = sign * (cfg.world_xy - boundary_clearance - radius)
    center_x, _ = _point_on_segment(state, goal, float(rng.uniform(0.35, 0.62)))
    center_x += float(rng.uniform(-120.0, 120.0))
    zones = [{'center_xy': [center_x, center_y], 'radius': radius}]
    aux_radius = _sample_radius(rng)
    aux_x = center_x + float(rng.uniform(520.0, 780.0))
    aux_y = 0.5 * (state[1] + goal[1]) - sign * float(rng.uniform(390.0, 650.0))
    zones.append({'center_xy': [aux_x, aux_y], 'radius': aux_radius})
    scenario_id = f'BM{idx:03d}'
    return _scenario_payload(
        state,
        goal,
        zones,
        category='boundary_margin',
        scenario_id=scenario_id,
        label=f'boundary_margin_{idx:03d}',
        description='A boundary-side obstacle and a secondary obstacle create pressure near the world margin.',
        cfg=cfg,
        min_clearance_to_boundary=boundary_clearance,
        difficulty_score=_difficulty(len(zones), min_clearance=boundary_clearance),
    )


def _make_wall_pressure(rng: Any, idx: int, cfg: ScenarioConfig) -> NamedScenario:
    state, goal = _sample_start_goal(rng, cfg, y_range=(-140.0, 140.0), y_delta_range=(-80.0, 80.0))
    centerline_y = 0.5 * (state[1] + goal[1])
    radii = [_sample_radius(rng), _sample_radius(rng), _sample_radius(rng)]
    gap = float(rng.uniform(80.0, 180.0))
    y_lower = centerline_y - (radii[0] + radii[1] + gap)
    y_mid = centerline_y
    y_upper = centerline_y + (radii[1] + radii[2] + gap)
    x_base, _ = _point_on_segment(state, goal, float(rng.uniform(0.40, 0.60)))
    zones = [
        {'center_xy': [x_base + float(rng.uniform(-210.0, -90.0)), y_lower], 'radius': radii[0]},
        {'center_xy': [x_base + float(rng.uniform(-70.0, 70.0)), y_mid], 'radius': radii[1]},
        {'center_xy': [x_base + float(rng.uniform(90.0, 210.0)), y_upper], 'radius': radii[2]},
    ]
    scenario_id = f'WP{idx:03d}'
    return _scenario_payload(
        state,
        goal,
        zones,
        category='wall_pressure',
        scenario_id=scenario_id,
        label=f'wall_pressure_{idx:03d}',
        description='Three formal-scale hemispheres form a pressured wall while leaving around-wall margins.',
        cfg=cfg,
        corridor_width=gap,
        difficulty_score=_difficulty(len(zones), corridor_width=gap),
    )


def _generate_category_scenarios(
    category: str,
    maker: Any,
    rng: Any,
    cfg: ScenarioConfig,
    count: int,
) -> list[NamedScenario]:
    accepted: list[NamedScenario] = []
    seen_keys: set[tuple[Any, ...]] = set()
    attempts = 0
    max_attempts = max(1, count * 200)
    last_error: Exception | None = None
    while len(accepted) < count and attempts < max_attempts:
        attempts += 1
        next_idx = len(accepted) + 1
        try:
            item = maker(rng, next_idx, cfg)
            key = _geometry_key(item.scenario)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            accepted.append(item)
        except ValueError as exc:
            last_error = exc
            continue
    if len(accepted) < count:
        raise RuntimeError(
            f'Failed to generate benchmark category={category}: accepted={len(accepted)} '
            f'requested={count} attempts={attempts} last_error={last_error}'
        )
    return accepted


def generate_benchmark_suite(
    *,
    seed: int = 20260407,
    count_per_category: int = 100,
    suite_name: str = DEFAULT_BENCHMARK_SUITE_NAME,
) -> dict[str, Any]:
    import numpy as np

    cfg = ScenarioConfig()
    rng = np.random.default_rng(seed)
    makers = {
        'single_detour': _make_single_detour,
        'double_channel': _make_double_channel,
        'boundary_margin': _make_boundary_margin,
        'wall_pressure': _make_wall_pressure,
    }
    categories = {
        category: _generate_category_scenarios(category, makers[category], rng, cfg, count_per_category)
        for category in BENCHMARK_CATEGORIES
    }

    scenarios: list[NamedScenario] = []
    for category in BENCHMARK_CATEGORIES:
        scenarios.extend(categories[category])

    return {
        'suite_name': suite_name,
        'seed': seed,
        'generation_method': 'fixed_seed_random_rejection_v4',
        'distance_unit': 'km',
        'time_unit': 's',
        'speed_km_s': float(cfg.speed),
        'dt_s': float(cfg.dt),
        'max_time_s': float(cfg.max_time_s),
        'max_steps': int(cfg.max_steps),
        'goal_radius_km': float(cfg.goal_radius),
        'zone_radius_range_km': list(BENCHMARK_ZONE_RADIUS_RANGE_KM),
        'start_goal_distance_range_km': list(BENCHMARK_START_GOAL_DISTANCE_RANGE_KM),
        'min_zone_surface_gap_km': BENCHMARK_MIN_ZONE_SURFACE_GAP_KM,
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
