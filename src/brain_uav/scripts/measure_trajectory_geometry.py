"""Offline V2 trajectory geometry measurements (docs/无法早停排查文档.md 2.4/2.5).

Not part of the production training pipeline and not imported by it. Given a
formal-stage metrics-reports directory (or a directory of saved trajectory
JSON files as written by ``V2ExperimentReporter.record_episode``), this
reproduces the two geometric analyses from the troubleshooting document:

- collision episodes (document section 2.4): the zone actually hit, the
  collision point's fraction of the start->goal route, the hit zone
  center's along-route fraction, its perpendicular distance to the
  start-goal line, its characteristic radius, and how deep the straight
  start-goal line penetrates the zone (chord length inside the zone);
- goal episodes (document section 2.5): the first blocking zone along the
  straight line, the step at which the trajectory first deviates laterally
  by more than a threshold from the straight line, the step of closest
  approach to the blocker center, the resulting decision lead (advance), and
  the maximum lateral offset.

All distances are in km (the project's world unit).
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from brain_uav.geometry import NoFlyZone, no_fly_zone_from_dict

_STRAIGHT_LINE_SAMPLES = 4096


def _load_trajectory_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    nested = source / 'trajectories' / 'training'
    candidates: list[Path] = []
    if nested.is_dir():
        candidates.extend(sorted(nested.glob('*.json')))
    else:
        for candidate in sorted(source.glob('*.json')):
            # A metrics-reports root also holds episodes/windows/stage JSON
            # files; only episode files carry a scenario payload.
            payload_probe = json.loads(candidate.read_text(encoding='utf-8'))
            if isinstance(payload_probe, dict) and 'scenario_payload' in payload_probe:
                candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f'No trajectory JSON files found under: {source}')
    return candidates


def _zone_characteristic_radius(zone: NoFlyZone) -> float:
    # The generator stores the reference scale it sampled the zone from;
    # that matches the document's "区特征半径" (~190 km) column.
    reference = zone.metadata.get('requested_reference_scale')
    if isinstance(reference, (int, float)) and float(reference) > 0.0:
        return float(reference)
    parameters = zone.metadata.get('actual_shape_parameters')
    if isinstance(parameters, dict) and parameters:
        # Box parameters are full side lengths; radius-like ones are not.
        values = [
            float(value) / 2.0 if str(key).startswith('size_') else float(value)
            for key, value in parameters.items()
        ]
        return float(max(values))
    bounds = zone.shape.bounding_box()
    return float(np.max(bounds.max_corner - bounds.min_corner) / 2.0)


def _zone_reference_center(zone: NoFlyZone) -> np.ndarray:
    bounds = zone.shape.bounding_box()
    return 0.5 * (bounds.min_corner + bounds.max_corner)


def _straight_line_frame(start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, float]:
    axis = goal - start
    route_length = float(np.linalg.norm(axis))
    if route_length <= 1e-9:
        raise ValueError('Scenario start and goal coincide.')
    return axis / route_length, route_length


def _perpendicular_distance(point: np.ndarray, start: np.ndarray, unit: np.ndarray) -> float:
    offset = point - start
    return float(np.linalg.norm(offset - np.dot(offset, unit) * unit))


def _straight_line_chord_inside(
    zone: NoFlyZone, start: np.ndarray, goal: np.ndarray, uav_radius: float,
) -> float:
    samples = np.linspace(0.0, 1.0, _STRAIGHT_LINE_SAMPLES)[:, None]
    points = start[None, :] + samples * (goal - start)[None, :]
    inside = np.asarray(
        [zone.point_clearance(point, uav_radius=uav_radius) <= 0.0 for point in points]
    )
    if not inside.any():
        return 0.0
    route_length = float(np.linalg.norm(goal - start))
    return float(inside.sum() * route_length / (_STRAIGHT_LINE_SAMPLES - 1))


def _first_hit_zone(
    trajectory: np.ndarray, zones: list[NoFlyZone], uav_radius: float,
) -> tuple[int, np.ndarray, NoFlyZone] | None:
    for index in range(len(trajectory) - 1):
        for zone in zones:
            hit = zone.shape.segment_intersection(
                trajectory[index], trajectory[index + 1]
            )
            if hit is not None and zone.point_clearance(
                trajectory[index + 1], uav_radius=uav_radius
            ) <= 1e-6:
                return index, hit.point, zone
    return None


def measure_collision(
    record: dict[str, Any], trajectory: np.ndarray, zones: list[NoFlyZone],
    *, start: np.ndarray, goal: np.ndarray, unit: np.ndarray,
    route_length: float, uav_radius: float,
) -> None:
    hit = _first_hit_zone(trajectory, zones, uav_radius)
    if hit is None:
        record['error'] = 'no geometric hit found on saved trajectory'
        return
    hit_index, _hit_point, hit_zone = hit
    zone_center = _zone_reference_center(hit_zone)
    record.update(
        {
            'collision_step': int(hit_index + 1),
            'collision_along_route_fraction': float(
                np.dot(trajectory[hit_index + 1] - start, unit) / route_length
            ),
            'hit_zone_id': hit_zone.zone_id,
            'hit_zone_center_along_route_fraction': float(
                np.dot(zone_center - start, unit) / route_length
            ),
            'hit_zone_center_perpendicular_km': _perpendicular_distance(
                zone_center, start, unit
            ),
            'hit_zone_characteristic_radius_km': _zone_characteristic_radius(hit_zone),
            'straight_line_chord_inside_hit_zone_km': _straight_line_chord_inside(
                hit_zone, start, goal, uav_radius
            ),
        }
    )


def measure_goal(
    record: dict[str, Any], trajectory: np.ndarray, zones: list[NoFlyZone],
    *, start: np.ndarray, goal: np.ndarray, unit: np.ndarray,
    route_length: float, blocking_margin: float, deviation_threshold_km: float,
) -> None:
    blockers = [
        zone for zone in zones
        if zone.violates_segment(start, goal, uav_radius=blocking_margin)
    ]
    record['straight_line_blocker_count'] = len(blockers)
    if not blockers:
        # Medium samples zone_count with a small zero-zone probability, so a
        # goal episode can legitimately have no straight-line blocker.
        record['skipped'] = 'goal episode on a block-free scenario'
        return
    blocker = min(
        blockers,
        key=lambda zone: np.dot(_zone_reference_center(zone) - start, unit),
    )
    blocker_center = _zone_reference_center(blocker)
    distances_to_center = np.linalg.norm(trajectory - blocker_center[None, :], axis=1)
    closest_step = int(np.argmin(distances_to_center))
    deviation_indices = np.flatnonzero(
        np.linalg.norm(
            (trajectory - start)
            - np.outer((trajectory - start) @ unit, unit),
            axis=1,
        ) > deviation_threshold_km
    )
    deviation_step = int(deviation_indices[0]) if len(deviation_indices) else None
    record.update(
        {
            'blocker_zone_id': blocker.zone_id,
            'blocker_center_along_route_km': float(
                np.dot(blocker_center - start, unit)
            ),
            'blocker_center_along_route_fraction': float(
                np.dot(blocker_center - start, unit) / route_length
            ),
            'blocker_center_perpendicular_km': _perpendicular_distance(
                blocker_center, start, unit
            ),
            'blocker_characteristic_radius_km': _zone_characteristic_radius(blocker),
            'deviation_step': deviation_step,
            'closest_approach_step': closest_step,
            'decision_lead_steps': (
                closest_step - deviation_step if deviation_step is not None else None
            ),
        }
    )


def measure_episode(payload: dict[str, Any], *, deviation_threshold_km: float) -> dict[str, Any]:
    scenario = payload['scenario_payload']
    start = np.asarray(scenario['state'][:3], dtype=np.float64)
    goal = np.asarray(scenario['goal'], dtype=np.float64)
    trajectory = np.asarray(payload['trajectory'], dtype=np.float64)[:, :3]
    zones = [
        no_fly_zone_from_dict(zone_payload)
        for zone_payload in scenario['zones']
    ]
    uav_radius = float(payload.get('uav_collision_radius', 0.0) or 0.0)
    blocking_margin = float(payload['scenario_config']['corridor_blocking_margin'])
    unit, route_length = _straight_line_frame(start, goal)

    lateral = np.linalg.norm(
        (trajectory - start) - np.outer((trajectory - start) @ unit, unit), axis=1,
    )
    record: dict[str, Any] = {
        'episode_length': int(payload.get('episode_length', len(trajectory) - 1)),
        'global_steps': payload.get('global_steps'),
        'outcome': payload['outcome'],
        'route_length_km': route_length,
        'zone_count': len(zones),
        'max_lateral_offset_km': float(lateral.max()),
    }

    if record['outcome'] == 'collision':
        measure_collision(
            record, trajectory, zones,
            start=start, goal=goal, unit=unit, route_length=route_length,
            uav_radius=uav_radius,
        )
    elif record['outcome'] == 'goal':
        measure_goal(
            record, trajectory, zones,
            start=start, goal=goal, unit=unit, route_length=route_length,
            blocking_margin=blocking_margin,
            deviation_threshold_km=deviation_threshold_km,
        )
    else:
        record['skipped'] = (
            f"outcome {record['outcome']!r} has no 2.4/2.5 measurement"
        )
    return record


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        'min': float(array.min()),
        'median': float(np.median(array)),
        'max': float(array.max()),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {'total': len(records)}

    collisions = [
        r for r in records if r.get('outcome') == 'collision' and 'error' not in r
    ]
    goals = [
        r for r in records if r.get('outcome') == 'goal' and 'error' not in r
        and 'skipped' not in r
    ]
    summary['collision_episodes'] = len(collisions)
    summary['goal_episodes'] = len(goals)
    summary['episodes_with_errors'] = sum(1 for r in records if 'error' in r)
    summary['episodes_skipped'] = sum(
        1 for r in records if 'skipped' in r and 'error' not in r
    )
    if collisions:
        summary['collision'] = {
            key: _quantiles([r[key] for r in collisions])
            for key in (
                'collision_along_route_fraction',
                'hit_zone_center_along_route_fraction',
                'hit_zone_center_perpendicular_km',
                'hit_zone_characteristic_radius_km',
                'straight_line_chord_inside_hit_zone_km',
            )
        }
    if goals:
        leads = [
            float(r['decision_lead_steps']) for r in goals
            if r.get('decision_lead_steps') is not None
        ]
        deviations = [
            float(r['deviation_step']) for r in goals
            if r.get('deviation_step') is not None
        ]
        summary['goal'] = {
            'blocker_center_along_route_fraction': _quantiles(
                [r['blocker_center_along_route_fraction'] for r in goals]
            ),
            'blocker_center_perpendicular_km': _quantiles(
                [r['blocker_center_perpendicular_km'] for r in goals]
            ),
            'blocker_characteristic_radius_km': _quantiles(
                [r['blocker_characteristic_radius_km'] for r in goals]
            ),
            'deviation_step': _quantiles(deviations) if deviations else {},
            'decision_lead_steps': _quantiles(leads) if leads else {},
            'max_lateral_offset_km': _quantiles(
                [r['max_lateral_offset_km'] for r in goals]
            ),
            'episodes_without_lateral_deviation': sum(
                1 for r in goals if r.get('deviation_step') is None
            ),
        }
    return summary


def _format_quantiles(entry: dict[str, float]) -> str:
    if not entry:
        return '(no data)'
    return f"min={entry['min']:.3g} median={entry['median']:.3g} max={entry['max']:.3g}"


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"episodes measured: {summary['total']} "
        f"(collision={summary['collision_episodes']}, goal={summary['goal_episodes']}, "
        f"errors={summary['episodes_with_errors']})"
    )
    collision = summary.get('collision')
    if collision:
        print('--- collision (doc 2.4)')
        for name, entry in collision.items():
            print(f'  {name}: {_format_quantiles(entry)}')
    goal = summary.get('goal')
    if goal:
        print('--- goal (doc 2.5)')
        for name, entry in goal.items():
            print(f'  {name}: {_format_quantiles(entry)}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Reproduce the geometric trajectory measurements from '
            'docs/无法早停排查文档.md sections 2.4/2.5 on saved episode JSONs.'
        )
    )
    parser.add_argument(
        '--metrics-reports', type=Path, required=True,
        help=(
            'A metrics-reports directory (with trajectories/ inside), a '
            'trajectories directory, or a single trajectory JSON file.'
        ),
    )
    parser.add_argument('--output-jsonl', type=Path, default=None)
    parser.add_argument('--output-summary-json', type=Path, default=None)
    parser.add_argument(
        '--deviation-threshold-km', type=float, default=20.0,
        help=(
            'Lateral offset from the start-goal line that counts as deviation '
            '(document section 2.5 uses 20 km).'
        ),
    )
    args = parser.parse_args(argv)

    records: list[dict[str, Any]] = []
    for path in _load_trajectory_files(args.metrics_reports):
        payload = json.loads(path.read_text(encoding='utf-8'))
        record = measure_episode(
            payload, deviation_threshold_km=args.deviation_threshold_km,
        )
        record['trajectory_file'] = path.name
        records.append(record)

    summary = summarize(records)
    print_summary(summary)
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open('w', encoding='utf-8') as handle:
            for record in records:
                handle.write(
                    json.dumps(record, allow_nan=False, ensure_ascii=False) + '\n'
                )
        print(f'wrote: {args.output_jsonl}')
    if args.output_summary_json is not None:
        args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary_json.write_text(
            json.dumps(summary, indent=2, allow_nan=False, ensure_ascii=False),
            encoding='utf-8',
        )
        print(f'wrote: {args.output_summary_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
