"""Streaming reports and shape-aware trajectory views for V2 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import csv
import json
from math import ceil, isfinite
from pathlib import Path
from statistics import mean
from time import monotonic
from typing import Any, Callable, Mapping, Sequence, TextIO

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.geometry import (
    Box, Ellipsoid, QuadrangularPyramid, Sphere, TriangularPyramid,
    no_fly_zone_from_dict,
)

V2_REPORT_OUTCOMES = ('goal', 'ground', 'boundary', 'collision', 'timeout')
V2_SAFETY_VISUALIZATION_NOTE = (
    'Conservative visual approximation; not used for collision checking.'
)


def _json_ready(value: Any, *, path: str = 'value') -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (bool, str, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f'{path} contains a non-finite number.')
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f'{path} contains a non-string mapping key.')
            result[key] = _json_ready(item, path=f'{path}.{key}')
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_ready(item, path=f'{path}[{index}]') for index, item in enumerate(value)]
    raise ValueError(f'{path} contains unsupported type {type(value).__name__}.')


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            handle.write('\n')
    except Exception as exc:
        if isinstance(exc, (ValueError, FileExistsError)):
            raise
        raise RuntimeError(f'Failed to write V2 report JSON: {path}') from exc


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle: TextIO | None = path.open('x', encoding='utf-8')
        except Exception as exc:
            raise RuntimeError(f'Failed to open V2 report JSONL: {path}') from exc

    def append(self, payload: Any) -> None:
        if self._handle is None:
            raise RuntimeError(f'V2 report JSONL is closed: {self.path}')
        try:
            self._handle.write(json.dumps(_json_ready(payload), ensure_ascii=False,
                                          sort_keys=True, allow_nan=False) + '\n')
            self._handle.flush()
        except Exception as exc:
            raise RuntimeError(f'Failed to append V2 report JSONL: {self.path}') from exc

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='') as handle:
            if fieldnames:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(_json_ready(dict(row)))
    except Exception as exc:
        if isinstance(exc, FileExistsError):
            raise
        raise RuntimeError(f'Failed to write V2 report CSV: {path}') from exc


@dataclass(frozen=True, slots=True)
class V2TrajectorySelection:
    reasons: tuple[str, ...] = ()
    progress_trigger_steps: tuple[int, ...] = ()

    @property
    def selected(self) -> bool:
        return bool(self.reasons)


class V2TrainingTrajectorySelector:
    """Deterministic per-stage progress/key selector with physical deduplication."""

    def __init__(self, *, max_steps: int, progress_limit: int = 20,
                 key_limit: int = 20, episode_group_size: int = 75) -> None:
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError('max_steps must be a positive integer.')
        if type(progress_limit) is not int or progress_limit <= 0:
            raise ValueError('progress_limit must be a positive integer.')
        if type(key_limit) is not int or key_limit < len(V2_REPORT_OUTCOMES):
            raise ValueError('key_limit must reserve every terminal outcome type.')
        if type(episode_group_size) is not int or episode_group_size <= 0:
            raise ValueError('episode_group_size must be a positive integer.')
        self._progress_limit = progress_limit
        self._key_limit = key_limit
        self._group_size = episode_group_size
        self._progress_thresholds = tuple(sorted({
            max(1, ceil(max_steps * index / progress_limit))
            for index in range(1, progress_limit + 1)
        }))
        self._next_progress = 0
        self.progress_sample_count = 0
        self.key_sample_count = 0
        self._unseen_outcomes = set(V2_REPORT_OUTCOMES)
        self._group_slots: set[tuple[int, str]] = set()

    def select(self, *, episode: int, stage_steps: int,
               outcome: str) -> V2TrajectorySelection:
        if type(episode) is not int or episode <= 0:
            raise ValueError('episode must be a positive integer.')
        if type(stage_steps) is not int or stage_steps < 0:
            raise ValueError('stage_steps must be a non-negative integer.')
        if outcome not in V2_REPORT_OUTCOMES:
            raise ValueError('outcome must be a terminal V2 outcome.')
        reasons: list[str] = []
        crossed: list[int] = []
        while (self._next_progress < len(self._progress_thresholds)
               and stage_steps >= self._progress_thresholds[self._next_progress]):
            crossed.append(self._progress_thresholds[self._next_progress])
            self._next_progress += 1
        if crossed and self.progress_sample_count < self._progress_limit:
            self.progress_sample_count += 1
            reasons.append('progress')
        group = (episode - 1) // self._group_size + 1
        category = 'goal' if outcome == 'goal' else 'failure'
        slot = (group, category)
        if outcome in self._unseen_outcomes:
            if self.key_sample_count >= self._key_limit:
                raise RuntimeError('Reserved V2 key-sample budget was exhausted.')
            self._unseen_outcomes.remove(outcome)
            self.key_sample_count += 1
            reasons.append(f'first_outcome:{outcome}')
            if slot not in self._group_slots:
                self._group_slots.add(slot)
                reasons.append(f'episode_group:{group}:{category}')
        elif slot not in self._group_slots:
            if self.key_sample_count < self._key_limit - len(self._unseen_outcomes):
                self._group_slots.add(slot)
                self.key_sample_count += 1
                reasons.append(f'episode_group:{group}:{category}')
        return V2TrajectorySelection(tuple(reasons), tuple(crossed))


class V2ValidationTrajectorySelector:
    def __init__(self) -> None:
        self._goal_seen = False
        self._failure_types: set[str] = set()
        self.sample_count = 0

    def select(self, outcome: str) -> tuple[str, ...]:
        if outcome not in V2_REPORT_OUTCOMES:
            raise ValueError('outcome must be a terminal V2 outcome.')
        if outcome == 'goal':
            if self._goal_seen:
                return ()
            self._goal_seen = True
            self.sample_count += 1
            return ('first_goal',)
        if outcome in self._failure_types or len(self._failure_types) >= 2:
            return ()
        self._failure_types.add(outcome)
        self.sample_count += 1
        return (f'first_failure:{outcome}',)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=np.float64)

    def cross(origin, first, second) -> float:
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _offset_polygon(points: np.ndarray, margin: float) -> np.ndarray:
    polygon = _convex_hull(points)
    if margin <= 0.0 or len(polygon) < 3:
        return polygon
    shifted_points: list[np.ndarray] = []
    shifted_directions: list[np.ndarray] = []
    for index, point in enumerate(polygon):
        direction = polygon[(index + 1) % len(polygon)] - point
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            continue
        outward = np.array([direction[1], -direction[0]]) / length
        shifted_points.append(point + margin * outward)
        shifted_directions.append(direction)
    result: list[np.ndarray] = []
    for index in range(len(shifted_points)):
        previous = (index - 1) % len(shifted_points)
        p = shifted_points[previous]
        r = shifted_directions[previous]
        q = shifted_points[index]
        s = shifted_directions[index]
        denominator = float(r[0] * s[1] - r[1] * s[0])
        if abs(denominator) <= 1e-12:
            result.append(q)
            continue
        delta = q - p
        parameter = float((delta[0] * s[1] - delta[1] * s[0]) / denominator)
        result.append(p + parameter * r)
    return np.asarray(result, dtype=np.float64)


def _ellipse_points(center: np.ndarray, radius_first: float,
                    radius_second: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    cosines = np.cos(angles)
    sines = np.sin(angles)
    entity = np.column_stack((center[0] + radius_first * cosines,
                              center[1] + radius_second * sines))
    normals = np.column_stack((cosines / radius_first, sines / radius_second))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return entity, entity + margin * normals


def _project_shape(shape: Any, axes: tuple[int, int],
                   margin: float) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(shape, Sphere):
        return _ellipse_points(shape.center[list(axes)], shape.radius, shape.radius, margin)
    if isinstance(shape, Ellipsoid):
        radii = shape.radii[list(axes)]
        return _ellipse_points(shape.center[list(axes)], float(radii[0]), float(radii[1]), margin)
    if isinstance(shape, Box):
        center = shape.center[list(axes)]
        half = shape.half_sizes[list(axes)]
        entity = np.asarray([
            center + [-half[0], -half[1]], center + [half[0], -half[1]],
            center + [half[0], half[1]], center + [-half[0], half[1]],
        ])
        return entity, _offset_polygon(entity, margin)
    if isinstance(shape, (TriangularPyramid, QuadrangularPyramid)):
        entity = _convex_hull(shape.vertices[:, list(axes)])
        return entity, _offset_polygon(entity, margin)
    raise ValueError(f'Unsupported V2 shape for plotting: {type(shape).__name__}.')


def export_v2_trajectory_views(target_dir: str | Path, stem: str,
                               payload: Mapping[str, Any]) -> dict[str, str]:
    """Write redrawable JSON and one headless XY/XZ/YZ visualization."""

    if not isinstance(stem, str) or not stem:
        raise ValueError('stem must be non-empty.')
    copied = _json_ready(payload)
    required = {
        'scenario_payload', 'scenario_config', 'reward_config',
        'uav_collision_radius', 'trajectory', 'actions', 'terminal_state',
        'outcome', 'episode_return', 'episode_length', 'stage', 'global_steps',
        'model_type', 'source', 'selection_reasons',
    }
    if not isinstance(copied, dict) or not required.issubset(copied):
        raise ValueError('V2 trajectory report payload is incomplete.')
    trajectory = np.asarray(copied['trajectory'], dtype=np.float64)
    actions = np.asarray(copied['actions'], dtype=np.float64)
    if (trajectory.ndim != 2 or trajectory.shape[1] != 3
            or not np.all(np.isfinite(trajectory))):
        raise ValueError('trajectory must be finite with shape [steps + 1, 3].')
    if actions.shape != (len(trajectory) - 1, 2) or not np.all(np.isfinite(actions)):
        raise ValueError('actions must be finite with shape [steps, 2].')
    if copied['outcome'] not in V2_REPORT_OUTCOMES:
        raise ValueError('trajectory outcome is invalid.')
    copied['safety_boundary_visualization_note'] = V2_SAFETY_VISUALIZATION_NOTE
    scenario_payload = copied['scenario_payload']
    if not isinstance(scenario_payload, dict):
        raise ValueError('scenario_payload must be an object.')
    zones = [no_fly_zone_from_dict(item) for item in scenario_payload.get('zones', [])]
    scenario = copied['scenario_config']
    if not isinstance(scenario, dict):
        raise ValueError('scenario_config must be an object.')
    target = Path(target_dir)
    json_path = target / f'{stem}.json'
    png_path = target / f'{stem}.png'
    _write_json(json_path, copied)
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        fig, axes_objects = plt.subplots(1, 3, figsize=(18, 6))
        projections = (
            ('XY', (0, 1), (-float(scenario['world_xy']), float(scenario['world_xy'])),
             (-float(scenario['world_xy']), float(scenario['world_xy']))),
            ('XZ', (0, 2), (-float(scenario['world_xy']), float(scenario['world_xy'])),
             (float(scenario['world_z_min']), float(scenario['world_z_max']))),
            ('YZ', (1, 2), (-float(scenario['world_xy']), float(scenario['world_xy'])),
             (float(scenario['world_z_min']), float(scenario['world_z_max']))),
        )
        start = np.asarray(scenario_payload['state'][:3], dtype=np.float64)
        goal = np.asarray(scenario_payload['goal'], dtype=np.float64)
        for axis_index, (title, projection, x_limits, y_limits) in enumerate(projections):
            axis = axes_objects[axis_index]
            first, second = projection
            axis.plot(trajectory[:, first], trajectory[:, second], color='tab:blue', label='trajectory')
            axis.scatter(start[first], start[second], color='tab:blue', marker='o', label='start')
            axis.scatter(goal[first], goal[second], color='tab:green', marker='*', s=80, label='goal')
            goal_radius = float(scenario['goal_radius'])
            goal_angles = np.linspace(0.0, 2.0 * np.pi, 121)
            axis.plot(goal[first] + goal_radius * np.cos(goal_angles),
                      goal[second] + goal_radius * np.sin(goal_angles),
                      color='tab:green', linestyle=':', label='goal radius')
            for zone_index, zone in enumerate(zones):
                effective_margin = zone.safety_margin + float(copied['uav_collision_radius'])
                entity, safe = _project_shape(zone.shape, projection, effective_margin)
                closed_entity = np.vstack((entity, entity[0]))
                axis.plot(closed_entity[:, 0], closed_entity[:, 1], color='tab:red',
                          linewidth=1.5, label='solid boundary' if zone_index == 0 else None)
                if effective_margin > 0.0:
                    closed_safe = np.vstack((safe, safe[0]))
                    axis.plot(closed_safe[:, 0], closed_safe[:, 1], color='tab:orange',
                              linestyle='--', linewidth=1.2,
                              label=('effective safety boundary (visualized)'
                                     if zone_index == 0 else None))
            axis.set_title(title)
            axis.set_xlabel(f'{title[0].lower()} (km)')
            axis.set_ylabel(f'{title[1].lower()} (km)')
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_aspect('equal', adjustable='box')
            axis.grid(alpha=0.25)
            axis.legend(loc='best', fontsize=8)
        fig.suptitle(f"{copied['source']} {copied['stage']} | {copied['outcome']} | steps={copied['episode_length']}")
        fig.text(
            0.5,
            0.01,
            V2_SAFETY_VISUALIZATION_NOTE,
            ha='center',
            fontsize=9,
            color='tab:orange',
        )
        fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
        fig.savefig(png_path, dpi=160, bbox_inches='tight')
    except Exception as exc:
        raise RuntimeError(f'Failed to render V2 trajectory view: {png_path}') from exc
    finally:
        if 'plt' in locals() and 'fig' in locals():
            plt.close(fig)
    return {'json': str(json_path), 'png': str(png_path)}


def _plot_training_windows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        x = np.arange(1, len(rows) + 1)

        def values(name: str) -> list[float]:
            return [float(row.get(name, 0.0)) for row in rows]

        axes[0].plot(x, values('goal_ratio'), marker='o')
        axes[0].set_ylabel('goal ratio')
        axes[1].plot(x, values('average_return'), marker='o')
        axes[1].set_ylabel('average return')
        axes[2].plot(x, values('average_length'), marker='o')
        axes[2].set_ylabel('average length')
        axes[3].plot(x, values('average_actor_loss'), label='actor')
        axes[3].plot(x, values('average_critic_loss'), label='critic')
        axes[3].set_ylabel('loss')
        axes[3].set_xlabel('complete/partial window')
        axes[3].legend()
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches='tight')
    except Exception as exc:
        raise RuntimeError(f'Failed to render V2 training curves: {path}') from exc
    finally:
        if 'plt' in locals() and 'fig' in locals():
            plt.close(fig)


class V2ExperimentReporter:
    """Synchronous, optional report sink for one formal V2 stage."""

    def __init__(self, output_dir: str | Path, *, stage: str, model_type: str,
                 scenario: ScenarioConfig, rewards: RewardConfig,
                 uav_collision_radius: float, max_steps: int,
                 progress_interval_seconds: float = 60.0,
                 clock: Callable[[], float] = monotonic) -> None:
        if stage not in ('easy', 'medium', 'hard'):
            raise ValueError('stage must be easy, medium, or hard.')
        if model_type not in ('ann', 'snn'):
            raise ValueError('model_type must be ann or snn.')
        if not isinstance(scenario, ScenarioConfig) or not isinstance(rewards, RewardConfig):
            raise TypeError('scenario and rewards must use project config classes.')
        interval = float(progress_interval_seconds)
        if not isfinite(interval) or interval <= 0.0:
            raise ValueError('progress_interval_seconds must be finite and positive.')
        root = Path(output_dir)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f'V2 report directory is not empty: {root}')
        root.mkdir(parents=True, exist_ok=True)
        self.output_dir = root
        self.stage = stage
        self.model_type = model_type
        self.scenario = scenario
        self.rewards = rewards
        self.uav_collision_radius = float(uav_collision_radius)
        self._clock = clock
        self._interval = interval
        self._stage_started = clock()
        self._episode_started = self._stage_started
        self._window_started = self._stage_started
        self._last_console = self._stage_started
        self._episodes = _JsonlWriter(root / 'episodes.jsonl')
        self._windows = _JsonlWriter(root / 'windows.jsonl')
        self._window_rows: list[dict[str, Any]] = []
        self._pending_episodes: list[dict[str, Any]] = []
        self._selector = V2TrainingTrajectorySelector(max_steps=max_steps)
        self._pending_candidate: int | None = None
        self._pending_validation_global_steps: int | None = None
        self._validation_writer: _JsonlWriter | None = None
        self._validation_selector: V2ValidationTrajectorySelector | None = None
        self._validation_root: Path | None = None
        self._validation_total = 0
        self._validation_completed = 0
        self._validation_started_at = 0.0
        self._closed = False

    def start_stage(self, metadata: Mapping[str, Any]) -> None:
        startup = {
            'stage': self.stage, 'model_type': self.model_type,
            'scenario_config': asdict(self.scenario),
            'reward_config': asdict(self.rewards),
            'uav_collision_radius': self.uav_collision_radius,
            'report_directory': str(self.output_dir), **dict(metadata),
        }
        _write_json(self.output_dir / 'stage_start.json', startup)
        print(f"[V2 {self.model_type.upper()} {self.stage}] start "
              f"requested_device={metadata.get('requested_device')} "
              f"resolved_device={metadata.get('resolved_device')} "
              f"max_steps={metadata.get('max_steps')} "
              f"checkpoint={metadata.get('checkpoint_output')} "
              f"episodes={self.output_dir / 'episodes.jsonl'} "
              f"images={self.output_dir / 'trajectories'}")

    def maybe_report_progress(self, *, stage_steps: int, completed_episodes: int,
                              current_episode_steps: int, actor_active: bool) -> bool:
        now = self._clock()
        if now - self._last_console < self._interval:
            return False
        print(f"[V2 {self.model_type.upper()} {self.stage}] progress "
              f"stage_steps={stage_steps} episodes={completed_episodes} "
              f"current_episode_steps={current_episode_steps} "
              f"elapsed={now - self._stage_started:.1f}s "
              f"actor={'active' if actor_active else 'frozen'}")
        self._last_console = now
        return True

    def begin_episode(self) -> None:
        """Start timing one training episode before scenario generation/reset."""

        self._episode_started = self._clock()

    def record_episode(self, record: Mapping[str, Any], *,
                       scenario_payload: Mapping[str, Any],
                       trajectory: Sequence[Sequence[float]],
                       actions: Sequence[Sequence[float]],
                       terminal_state: Sequence[float]) -> None:
        now = self._clock()
        persisted = dict(record)
        persisted['episode_elapsed_seconds'] = now - self._episode_started
        persisted['stage_elapsed_seconds'] = now - self._stage_started
        self._episodes.append(persisted)
        self._pending_episodes.append(persisted)
        selection = self._selector.select(
            episode=int(record['episode']), stage_steps=int(record['stage_steps']),
            outcome=str(record['outcome']))
        if selection.selected:
            reasons = list(selection.reasons)
            if selection.progress_trigger_steps:
                reasons.append('progress_thresholds:' + ','.join(
                    str(value) for value in selection.progress_trigger_steps))
            stem = (f"ep_{int(record['episode']):06d}_"
                    f"s{int(record['stage_steps']):09d}_{record['outcome']}")
            export_v2_trajectory_views(
                self.output_dir / 'trajectories' / 'training', stem,
                {'scenario_payload': scenario_payload,
                 'scenario_config': asdict(self.scenario),
                 'reward_config': asdict(self.rewards),
                 'uav_collision_radius': self.uav_collision_radius,
                 'trajectory': trajectory, 'actions': actions,
                 'terminal_state': terminal_state, 'outcome': record['outcome'],
                 'episode_return': record['episode_return'],
                 'episode_length': record['episode_length'], 'stage': self.stage,
                 'global_steps': record['global_steps'], 'model_type': self.model_type,
                 'source': 'training', 'selection_reasons': reasons})
    def _reported_window(self, row: Mapping[str, Any], *, partial: bool) -> dict[str, Any]:
        episodes = list(self._pending_episodes)
        result = dict(row)
        result['stage'] = self.stage
        result['partial'] = partial
        for outcome in V2_REPORT_OUTCOMES:
            result[f'{outcome}_count'] = sum(item['outcome'] == outcome for item in episodes)
        count = len(episodes)
        result['goal_ratio'] = result['goal_count'] / count if count else 0.0
        if episodes:
            last = episodes[-1]
            for name in ('replay_size', 'replay_success_fraction', 'success_replay_size',
                         'batch_success_fraction', 'actor_update_status'):
                result[name] = last[name]
        result['window_elapsed_seconds'] = sum(
            float(item['episode_elapsed_seconds']) for item in episodes
        )
        return result

    def record_window(self, row: Mapping[str, Any]) -> None:
        reported = self._reported_window(row, partial=False)
        self._windows.append(reported)
        self._window_rows.append(reported)
        print(f"[V2 {self.model_type.upper()} {self.stage}] window "
              f"episodes={reported['episode_start']}-{reported['episode_end']} "
              f"stage_steps={reported['stage_steps']} "
              f"global_steps={reported.get('global_steps', 'n/a')} "
              f"goal_ratio={reported['goal_ratio']:.3f} "
              f"return={reported['average_return']:.3f} length={reported['average_length']:.1f} "
              f"actor_loss={reported['average_actor_loss']:.6f} "
              f"critic_loss={reported['average_critic_loss']:.6f} "
              f"qualified_streak={reported['consecutive_qualified_windows']} "
              f"replay={reported.get('replay_size', 0)} "
              f"elapsed={reported['window_elapsed_seconds']:.1f}s")
        now = self._clock()
        self._pending_episodes = []
        self._window_started = now
        self._last_console = now

    def prepare_validation_candidate(self, candidate_index: int, *, global_steps: int) -> None:
        if self._validation_writer is not None or self._pending_candidate is not None:
            raise RuntimeError('A V2 validation report is already active.')
        if type(candidate_index) is not int or candidate_index <= 0:
            raise ValueError('candidate_index must be positive.')
        if type(global_steps) is not int or global_steps < 0:
            raise ValueError('global_steps must be a non-negative integer.')
        self._pending_candidate = candidate_index
        self._pending_validation_global_steps = global_steps

    def begin_validation(self, *, curriculum_level: str, scenario_count: int) -> None:
        if self._pending_candidate is None:
            raise RuntimeError('Validation candidate index was not prepared.')
        candidate = self._pending_candidate
        root = self.output_dir / 'validation' / f'candidate_{candidate:04d}'
        root.mkdir(parents=True, exist_ok=False)
        self._validation_writer = _JsonlWriter(root / 'scenarios.jsonl')
        self._validation_selector = V2ValidationTrajectorySelector()
        self._validation_root = root
        self._validation_total = scenario_count
        self._validation_completed = 0
        self._validation_started_at = self._clock()
        print(f"[V2 {self.model_type.upper()} {self.stage}] fixed validation "
              f"candidate={candidate} level={curriculum_level} scenarios={scenario_count}")

    def record_validation_scenario(self, record: Mapping[str, Any], *,
                                   scenario_payload: Mapping[str, Any],
                                   trajectory: Sequence[Sequence[float]],
                                   actions: Sequence[Sequence[float]],
                                   terminal_state: Sequence[float]) -> None:
        if (self._validation_writer is None or self._validation_selector is None
                or self._validation_root is None):
            raise RuntimeError('No V2 validation report is active.')
        self._validation_completed += 1
        persisted = dict(record)
        persisted['candidate_index'] = self._pending_candidate
        persisted['global_steps'] = self._pending_validation_global_steps
        persisted['sequence_index'] = self._validation_completed - 1
        self._validation_writer.append(persisted)
        reasons = self._validation_selector.select(str(record['outcome']))
        if reasons:
            stem = (f"scenario_{self._validation_completed:04d}_"
                    f"{record['scenario_id']}_{record['outcome']}")
            export_v2_trajectory_views(
                self._validation_root / 'trajectories', stem,
                {'scenario_payload': scenario_payload,
                 'scenario_config': asdict(self.scenario),
                 'reward_config': asdict(self.rewards),
                 'uav_collision_radius': self.uav_collision_radius,
                 'trajectory': trajectory, 'actions': actions,
                 'terminal_state': terminal_state, 'outcome': record['outcome'],
                 'episode_return': record['episode_return'],
                 'episode_length': record['episode_length'], 'stage': self.stage,
                 'global_steps': self._pending_validation_global_steps,
                 'model_type': self.model_type, 'source': 'fixed_validation',
                 'candidate_index': self._pending_candidate,
                 'selection_reasons': list(reasons)})
        if (self._validation_completed % 10 == 0
                or self._validation_completed == self._validation_total):
            print(f"[V2 {self.model_type.upper()} {self.stage}] validation progress "
                  f"candidate={self._pending_candidate} "
                  f"{self._validation_completed}/{self._validation_total}")

    def finish_validation(self, result: Mapping[str, Any]) -> None:
        if self._validation_writer is None or self._validation_root is None:
            raise RuntimeError('No V2 validation report is active.')
        self._validation_writer.close()
        self._validation_writer = None
        summary = dict(result)
        summary['candidate_index'] = self._pending_candidate
        summary['global_steps'] = self._pending_validation_global_steps
        summary['elapsed_seconds'] = self._clock() - self._validation_started_at
        _write_json(self._validation_root / 'summary.json', summary)
        print(f"[V2 {self.model_type.upper()} {self.stage}] validation complete "
              f"candidate={self._pending_candidate} outcomes={summary['outcome_counts']} "
              f"passed={summary['passed']} elapsed={summary['elapsed_seconds']:.1f}s")
        self._pending_candidate = None
        self._pending_validation_global_steps = None
        self._validation_selector = None
        self._validation_root = None

    def abort_validation(self) -> None:
        if self._validation_writer is not None:
            self._validation_writer.close()
        self._validation_writer = None
        self._validation_selector = None
        self._validation_root = None
        self._pending_candidate = None
        self._pending_validation_global_steps = None

    def finish_stage(self, result: Mapping[str, Any]) -> None:
        stage_elapsed_seconds = self._clock() - self._stage_started
        if self._pending_episodes:
            episodes = self._pending_episodes
            partial = {
                'window_index': len(self._window_rows) + 1,
                'episode_start': episodes[0]['episode'],
                'episode_end': episodes[-1]['episode'],
                'episode_count': len(episodes),
                'goal_count': sum(item['outcome'] == 'goal' for item in episodes),
                'failure_count': sum(item['outcome'] != 'goal' for item in episodes),
                'qualified': False, 'consecutive_qualified_windows': 0,
                'stage_steps': result['stage_steps'], 'candidate': False,
                'global_steps': result['global_steps_end'],
                'average_return': mean(item['episode_return'] for item in episodes),
                'average_length': mean(item['episode_length'] for item in episodes),
                'average_actor_loss': mean(item['actor_loss'] for item in episodes),
                'average_critic_loss': mean(item['critic_loss'] for item in episodes),
                'average_bc_loss': mean(item['bc_loss'] for item in episodes),
                'average_weighted_bc_contribution': mean(
                    item['weighted_bc_contribution'] for item in episodes),
                'bc_lambda': episodes[-1]['bc_lambda'],
                'exploration_noise': episodes[-1]['exploration_noise'],
                'policy_noise': episodes[-1]['policy_noise'],
                'noise_clip': episodes[-1]['noise_clip'],
            }
            reported = self._reported_window(partial, partial=True)
            self._windows.append(reported)
            self._window_rows.append(reported)
            self._pending_episodes = []
        stage_end = dict(result)
        stage_end['stage_elapsed_seconds'] = stage_elapsed_seconds
        _write_json(self.output_dir / 'stage_end.json', stage_end)
        _write_csv(self.output_dir / 'windows.csv', self._window_rows)
        _plot_training_windows(self.output_dir / 'training_curves.png', self._window_rows)
        print(f"[V2 {self.model_type.upper()} {self.stage}] finish "
              f"status={result['status']} reason={result['stop_reason']} "
              f"steps={result['stage_steps']} elapsed={stage_elapsed_seconds:.1f}s "
              f"report={self.output_dir}")

    def close(self) -> None:
        if self._closed:
            return
        self.abort_validation()
        self._episodes.close()
        self._windows.close()
        self._closed = True


class V2BCTrainingReporter:
    """Synchronous epoch logger using losses from the existing BC pass."""

    def __init__(self, output_dir: str | Path,
                 *, clock: Callable[[], float] = monotonic) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._epoch_started = clock()
        self._writer = _JsonlWriter(self.output_dir / 'epochs.jsonl')
        self._rows: list[dict[str, Any]] = []
        self._closed = False

    def record_epoch(self, record: Mapping[str, Any]) -> None:
        row = dict(record)
        now = self._clock()
        row['epoch_elapsed_seconds'] = now - self._epoch_started
        self._writer.append(row)
        self._rows.append(_json_ready(row))
        print(f"[V2 BC] epoch={row['epoch']}/{row['epochs']} "
              f"train_mse={row['train_loss']:.8f} "
              f"validation_mse={row['validation_loss']:.8f} "
              f"best_epoch={row['best_epoch']} "
              f"best_validation_mse={row['best_validation_loss']:.8f} "
              f"refreshed_best={row['refreshed_best']} "
              f"elapsed={row['epoch_elapsed_seconds']:.1f}s")
        self._epoch_started = now

    def finish(self) -> None:
        _write_csv(self.output_dir / 'epochs.csv', self._rows)
        path = self.output_dir / 'mse_curve.png'
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            fig, axis = plt.subplots(figsize=(10, 6))
            epochs = [row['epoch'] for row in self._rows]
            axis.plot(epochs, [row['train_loss'] for row in self._rows],
                      marker='o', label='train MSE')
            axis.plot(epochs, [row['validation_loss'] for row in self._rows],
                      marker='o', label='validation MSE')
            axis.set_xlabel('epoch')
            axis.set_ylabel('MSE')
            axis.grid(alpha=0.25)
            axis.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=160, bbox_inches='tight')
        except Exception as exc:
            raise RuntimeError(f'Failed to render V2 BC MSE curve: {path}') from exc
        finally:
            if 'plt' in locals() and 'fig' in locals():
                plt.close(fig)

    def close(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True


__all__ = [
    'V2_SAFETY_VISUALIZATION_NOTE',
    'V2BCTrainingReporter', 'V2ExperimentReporter',
    'V2TrainingTrajectorySelector', 'V2TrajectorySelection',
    'V2ValidationTrajectorySelector', 'export_v2_trajectory_views',
]
