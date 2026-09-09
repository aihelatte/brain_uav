"""Deterministic coarse 3-D geometric reachability for V2 scenarios.

The search is deliberately geometry-only.  It proves that a path exists in a
bounded sparse visibility graph with conservative no-fly-zone clearance; it
does not model UAV turn-rate, climb-rate, or policy behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import isfinite
from typing import Any, Sequence

import numpy as np

from brain_uav.config import ScenarioConfig
from brain_uav.geometry import GEOMETRY_TOLERANCE, NoFlyZone


def _positive_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and greater than zero.') from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f'{name} must be finite and greater than zero.')
    return result


def _nonnegative_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and non-negative.') from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and non-negative.')
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _point3(value: Any, *, name: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite vector with shape (3,).') from exc
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f'{name} must be a finite vector with shape (3,).')
    return point.copy()


@dataclass(frozen=True, slots=True)
class V2FeasibilityConfig:
    """Conservative-clearance and bounded-search settings.

    xy_step and z_step remain accepted for configuration compatibility with
    the earlier grid checker. Sparse visibility candidates are derived from
    expanded zone bounds, so the current algorithm has no grid resolution.
    """

    xy_step: float = 50.0
    z_step: float = 25.0
    max_expanded_nodes: int = 100_000
    clearance: float = 40.0

    def __post_init__(self) -> None:
        object.__setattr__(self, 'xy_step', _positive_float(self.xy_step, name='xy_step'))
        object.__setattr__(self, 'z_step', _positive_float(self.z_step, name='z_step'))
        object.__setattr__(
            self,
            'max_expanded_nodes',
            _positive_int(self.max_expanded_nodes, name='max_expanded_nodes'),
        )
        object.__setattr__(self, 'clearance', _nonnegative_float(self.clearance, name='clearance'))


@dataclass(frozen=True, slots=True)
class V2FeasibilityResult:
    """Outcome and deterministic work counters for a bounded path search.

    expanded_nodes is retained for compatibility and equals examined_nodes:
    the number of graph nodes removed from the A* frontier. edge_checks counts
    exact NoFlyZone.violates_segment calls; a conservative AABB proof may
    reject the need for such a call.
    """

    reachable: bool
    expanded_nodes: int
    reason: str
    algorithm: str
    examined_nodes: int
    edge_checks: int


_ALGORITHM = 'sparse_visibility_graph'
_WAYPOINTS_PER_ZONE = 14


def check_v2_geometric_feasibility(
    start: Any,
    goal: Any,
    zones: Sequence[NoFlyZone],
    scenario: ScenarioConfig,
    config: V2FeasibilityConfig | None = None,
) -> V2FeasibilityResult:
    """Run deterministic A* over a bounded sparse 3-D visibility graph.

    Each zone contributes at most eight expanded-AABB corners and six face
    centres. Thus a scene with Z zones has at most 2 + 14 * Z nodes. Each
    unordered node pair is evaluated once. A potentially intersecting edge is
    accepted only after exact NoFlyZone.violates_segment checks using the
    configured clearance.
    """

    if not isinstance(scenario, ScenarioConfig):
        raise TypeError('scenario must be a ScenarioConfig.')
    settings = config or V2FeasibilityConfig()
    if not isinstance(settings, V2FeasibilityConfig):
        raise TypeError('config must be a V2FeasibilityConfig.')
    if isinstance(zones, (str, bytes)) or not isinstance(zones, Sequence):
        raise TypeError('zones must be a sequence of NoFlyZone objects.')
    zone_items = tuple(zones)
    if any(not isinstance(zone, NoFlyZone) for zone in zone_items):
        raise TypeError('zones must contain only NoFlyZone objects.')

    start_point = _point3(start, name='start')
    goal_point = _point3(goal, name='goal')
    world_xy = float(scenario.world_xy)
    world_z_min = float(scenario.world_z_min)
    world_z_max = float(scenario.world_z_max)

    def in_world(point: np.ndarray) -> bool:
        return bool(
            abs(float(point[0])) <= world_xy + GEOMETRY_TOLERANCE
            and abs(float(point[1])) <= world_xy + GEOMETRY_TOLERANCE
            and float(point[2]) > world_z_min
            and float(point[2]) <= world_z_max + GEOMETRY_TOLERANCE
        )

    if not in_world(start_point) or not in_world(goal_point):
        raise ValueError('start and goal must lie inside the flyable world bounds.')

    expanded_bounds: list[tuple[NoFlyZone, np.ndarray, np.ndarray]] = []
    for zone in zone_items:
        bounds = zone.shape.bounding_box()
        expansion = zone.safety_margin + settings.clearance
        expanded_bounds.append(
            (
                zone,
                bounds.min_corner - expansion,
                bounds.max_corner + expansion,
            )
        )

    edge_checks = 0

    def result(reachable: bool, examined_nodes: int, reason: str) -> V2FeasibilityResult:
        return V2FeasibilityResult(
            reachable=reachable,
            expanded_nodes=examined_nodes,
            reason=reason,
            algorithm=_ALGORITHM,
            examined_nodes=examined_nodes,
            edge_checks=edge_checks,
        )

    def point_is_clear(point: np.ndarray) -> bool:
        return not any(
            np.all(point >= minimum)
            and np.all(point <= maximum)
            and zone.violates_point(point, uav_radius=settings.clearance)
            for zone, minimum, maximum in expanded_bounds
        )

    def edge_is_clear(first: np.ndarray, second: np.ndarray) -> bool:
        nonlocal edge_checks
        segment_minimum = np.minimum(first, second)
        segment_maximum = np.maximum(first, second)
        for zone, minimum, maximum in expanded_bounds:
            if not (
                np.all(segment_minimum <= maximum)
                and np.all(minimum <= segment_maximum)
            ):
                continue
            edge_checks += 1
            if zone.violates_segment(
                first,
                second,
                uav_radius=settings.clearance,
            ):
                return False
        return True

    def accepted_path_is_exactly_clear(path: list[np.ndarray]) -> bool:
        nonlocal edge_checks
        for first, second in zip(path, path[1:]):
            for zone in zone_items:
                edge_checks += 1
                if zone.violates_segment(
                    first,
                    second,
                    uav_radius=settings.clearance,
                ):
                    return False
        return True

    if not point_is_clear(start_point) or not point_is_clear(goal_point):
        return result(False, 0, 'endpoint_blocked')

    direct_is_clear = edge_is_clear(start_point, goal_point)
    if direct_is_clear and accepted_path_is_exactly_clear([start_point, goal_point]):
        return result(True, 0, 'reachable')

    route_epsilon = max(
        1e-6,
        min(settings.xy_step, settings.z_step) * 1e-6,
    )

    def bound_waypoints(
        minimum: np.ndarray,
        maximum: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        lower = minimum - route_epsilon
        upper = maximum + route_epsilon
        centre = 0.5 * (lower + upper)
        corners = tuple(
            np.array([x_value, y_value, z_value], dtype=np.float64)
            for x_value in (lower[0], upper[0])
            for y_value in (lower[1], upper[1])
            for z_value in (lower[2], upper[2])
        )
        horizontal_corners = tuple(
            np.array([x_value, y_value, centre[2]], dtype=np.float64)
            for x_value in (lower[0], upper[0])
            for y_value in (lower[1], upper[1])
        )
        vertical_faces = (
            np.array([centre[0], centre[1], lower[2]], dtype=np.float64),
            np.array([centre[0], centre[1], upper[2]], dtype=np.float64),
        )
        return corners + horizontal_corners + vertical_faces

    nodes: list[np.ndarray] = [start_point, goal_point]
    seen = {
        tuple(float(value) for value in start_point),
        tuple(float(value) for value in goal_point),
    }
    for _, minimum, maximum in expanded_bounds:
        for candidate in bound_waypoints(minimum, maximum):
            key = tuple(float(value) for value in candidate)
            if key in seen or not in_world(candidate) or not point_is_clear(candidate):
                continue
            seen.add(key)
            nodes.append(candidate)

    if len(nodes) > 2 + _WAYPOINTS_PER_ZONE * len(zone_items):
        raise RuntimeError('Sparse visibility graph exceeded its structural node bound.')

    def heuristic(node_index: int) -> float:
        return float(np.linalg.norm(goal_point - nodes[node_index]))

    edge_cache: dict[tuple[int, int], bool] = {(0, 1): False}
    open_heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(open_heap, (heuristic(0), 0.0, 0, 0))
    best_cost: dict[int, float] = {0: 0.0}
    parent: dict[int, int] = {}
    closed: set[int] = set()
    serial = 1
    expanded = 0

    while open_heap:
        _, cost, _, current = heapq.heappop(open_heap)
        if current in closed or cost > best_cost.get(current, float('inf')) + 1e-12:
            continue
        closed.add(current)
        expanded += 1

        if current == 1:
            path_indices = [current]
            while path_indices[-1] != 0:
                path_indices.append(parent[path_indices[-1]])
            path = [nodes[index] for index in reversed(path_indices)]
            if not accepted_path_is_exactly_clear(path):
                raise RuntimeError(
                    'Sparse visibility broad phase admitted an unsafe final path edge.'
                )
            return result(True, expanded, 'reachable')
        if expanded >= settings.max_expanded_nodes:
            return result(False, expanded, 'expansion_limit')

        current_point = nodes[current]
        for neighbor, neighbor_point in enumerate(nodes):
            if neighbor == current or neighbor in closed:
                continue
            edge_key = (
                (current, neighbor)
                if current < neighbor
                else (neighbor, current)
            )
            clear = edge_cache.get(edge_key)
            if clear is None:
                clear = edge_is_clear(current_point, neighbor_point)
                edge_cache[edge_key] = clear
            if not clear:
                continue
            step_cost = float(np.linalg.norm(neighbor_point - current_point))
            candidate_cost = cost + step_cost
            if candidate_cost >= best_cost.get(neighbor, float('inf')) - 1e-12:
                continue
            best_cost[neighbor] = candidate_cost
            parent[neighbor] = current
            heapq.heappush(
                open_heap,
                (
                    candidate_cost + heuristic(neighbor),
                    candidate_cost,
                    serial,
                    neighbor,
                ),
            )
            serial += 1

    return result(False, expanded, 'no_path')
