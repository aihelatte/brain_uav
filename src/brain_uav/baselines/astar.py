from __future__ import annotations

import heapq

import numpy as np

from ..envs import StaticNoFlyTrajectoryEnv
from .common import heading_to_action


class AStarPlanner:
    def __init__(self, env: StaticNoFlyTrajectoryEnv, grid_size: float = 40.0):
        self.env = env
        self.grid_size = grid_size

    def act(self, obs: np.ndarray) -> np.ndarray:
        del obs
        waypoint = self._next_waypoint()
        limits = np.array(
            [self.env.scenario.delta_gamma_max, self.env.scenario.delta_psi_max], dtype=np.float32
        )
        return heading_to_action(self.env.state[3], self.env.state[4], waypoint - self.env.state[:3], limits)

    def rollout(self, max_steps: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        obs, _ = self.env.reset()
        steps = max_steps or self.env.scenario.max_steps
        samples: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(steps):
            action = self.act(obs)
            samples.append((obs.copy(), action.copy()))
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        return samples

    def _next_waypoint(self) -> np.ndarray:
        start = self._to_cell(self.env.state[:3])
        goal = self._to_cell(self.env.goal)
        frontier: list[tuple[float, tuple[int, int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int, int], float] = {start: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for nxt in self._neighbors(current):
                new_cost = cost_so_far[current] + self.grid_size
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    heapq.heappush(frontier, (new_cost + self._heuristic(nxt, goal), nxt))
                    came_from[nxt] = current
        if goal not in came_from:
            return self.env.goal
        path = [goal]
        while came_from[path[-1]] is not None:
            path.append(came_from[path[-1]])
        path.reverse()
        target_cell = path[1] if len(path) > 1 else goal
        return self._to_coord(target_cell)

    def _neighbors(self, cell: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        neighbors = []
        for dx, dy, dz in [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]:
            nxt = (cell[0] + dx, cell[1] + dy, max(cell[2] + dz, 1))
            if self._valid_coord(self._to_coord(nxt)):
                neighbors.append(nxt)
        return neighbors

    def _valid_coord(self, coord: np.ndarray) -> bool:
        x, y, z = coord
        cfg = self.env.scenario
        if abs(x) > cfg.world_xy or abs(y) > cfg.world_xy or z <= cfg.world_z_min or z > cfg.world_z_max:
            return False
        for zone in self.env.zones:
            distance = (x - zone.center_xy[0]) ** 2 + (y - zone.center_xy[1]) ** 2 + z**2
            if distance <= (zone.radius + 10.0) ** 2:
                return False
        return True

    def _to_cell(self, coord: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.round(coord / self.grid_size).astype(int).tolist())

    def _to_coord(self, cell: tuple[int, int, int]) -> np.ndarray:
        return np.array(cell, dtype=np.float32) * self.grid_size

    @staticmethod
    def _heuristic(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))
