"""Lightweight process-based environment workers for TD3 rollout."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import ExperimentConfig
from ..scripts.common import make_env


def _goal_error_components(env) -> dict[str, float]:
    pos = np.asarray(env.state[:3], dtype=float)
    goal = np.asarray(env.goal, dtype=float)
    delta = pos - goal
    return {
        'goal_distance': float(np.linalg.norm(delta)),
        'xy_goal_distance': float(np.linalg.norm(delta[:2])),
        'z_goal_error': float(abs(delta[2])),
        'segment_goal_distance': float(getattr(env, 'last_segment_goal_distance', np.linalg.norm(delta))),
        'goal_reached_by_segment': bool(getattr(env, 'last_goal_reached_by_segment', False)),
        'active_goal_radius': float(getattr(env, 'active_goal_radius', env.scenario.goal_radius)),
    }


def _new_episode_goal_diagnostics(env) -> dict[str, Any]:
    components = _goal_error_components(env)
    return {
        'min_goal_distance': components['goal_distance'],
        'min_xy_goal_distance': components['xy_goal_distance'],
        'min_z_goal_error': components['z_goal_error'],
        'min_segment_goal_distance': components['segment_goal_distance'],
        'step_at_min_goal_distance': 0,
        'near_goal_step_count': 0,
        'goal_reached_by_segment_count': 0,
        'active_goal_radius': components['active_goal_radius'],
    }


def _update_episode_goal_diagnostics(env, diagnostics: dict[str, Any], step: int) -> None:
    components = _goal_error_components(env)
    if float(components['goal_distance']) <= 50.0:
        diagnostics['near_goal_step_count'] += 1
    if bool(components['goal_reached_by_segment']):
        diagnostics['goal_reached_by_segment_count'] += 1
    if float(components['goal_distance']) < float(diagnostics['min_goal_distance']):
        diagnostics['min_goal_distance'] = components['goal_distance']
        diagnostics['min_xy_goal_distance'] = components['xy_goal_distance']
        diagnostics['min_z_goal_error'] = components['z_goal_error']
        diagnostics['step_at_min_goal_distance'] = int(step)
    if float(components['segment_goal_distance']) < float(diagnostics['min_segment_goal_distance']):
        diagnostics['min_segment_goal_distance'] = components['segment_goal_distance']
    diagnostics['active_goal_radius'] = components['active_goal_radius']


def _finalize_episode_goal_diagnostics(env, diagnostics: dict[str, Any]) -> dict[str, float | int]:
    components = _goal_error_components(env)
    return {
        'min_goal_distance': float(diagnostics['min_goal_distance']),
        'min_xy_goal_distance': float(diagnostics['min_xy_goal_distance']),
        'min_z_goal_error': float(diagnostics['min_z_goal_error']),
        'final_goal_distance': float(components['goal_distance']),
        'final_xy_goal_distance': float(components['xy_goal_distance']),
        'final_z_goal_error': float(components['z_goal_error']),
        'min_segment_goal_distance': float(diagnostics['min_segment_goal_distance']),
        'step_at_min_goal_distance': int(diagnostics['step_at_min_goal_distance']),
        'near_goal_step_count': int(diagnostics['near_goal_step_count']),
        'goal_reached_by_segment_count': int(diagnostics['goal_reached_by_segment_count']),
        'active_goal_radius': float(diagnostics['active_goal_radius']),
    }


def _make_episode_record(
    env,
    env_id: int,
    info: dict[str, Any],
    episode_return: float,
    episode_length: int,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        'env_id': int(env_id),
        'return': float(episode_return),
        'length': int(episode_length),
        'outcome': info.get('outcome', 'unknown'),
        **_finalize_episode_goal_diagnostics(env, diagnostics),
        'scenario': env.export_scenario(),
        'trajectory': [point.tolist() for point in env.trajectory],
        'final_state': env.state.copy().tolist(),
        'info': {
            'goal_distance': float(info.get('goal_distance', 0.0)),
            'segment_goal_distance': float(info.get('segment_goal_distance', 0.0)),
            'goal_reached_by_segment': bool(info.get('goal_reached_by_segment', False)),
            'goal_radius': float(info.get('goal_radius', env.scenario.goal_radius)),
            'active_goal_radius': float(info.get('active_goal_radius', env.scenario.goal_radius)),
            'progress': float(info.get('progress', 0.0)),
            'steps': int(info.get('steps', episode_length)),
            'curriculum_level': info.get('curriculum_level'),
        },
    }


def _worker_loop(
    conn,
    env_id: int,
    cfg: ExperimentConfig,
    seed: int,
    curriculum_level: str,
    curriculum_mix: dict[str, float],
) -> None:
    env = None
    try:
        env = make_env(
            cfg,
            seed=seed,
            curriculum_level=curriculum_level,
            curriculum_mix=curriculum_mix,
            goal_radius_curriculum_enabled=True,
        )
        obs, _ = env.reset(seed=seed)
        episode_return = 0.0
        episode_length = 0
        diagnostics = _new_episode_goal_diagnostics(env)
        while True:
            command, payload = conn.recv()
            if command == 'reset':
                reset_seed = payload.get('seed') if isinstance(payload, dict) else None
                obs, _ = env.reset(seed=reset_seed)
                episode_return = 0.0
                episode_length = 0
                diagnostics = _new_episode_goal_diagnostics(env)
                conn.send({'ok': True, 'env_id': env_id, 'obs': obs})
            elif command == 'step':
                action = payload
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                episode_return += float(reward)
                episode_length += 1
                _update_episode_goal_diagnostics(env, diagnostics, episode_length)
                episode_record = None
                reset_obs = None
                if done:
                    episode_record = _make_episode_record(
                        env,
                        env_id,
                        info,
                        episode_return,
                        episode_length,
                        diagnostics,
                    )
                    reset_obs, _ = env.reset()
                    episode_return = 0.0
                    episode_length = 0
                    diagnostics = _new_episode_goal_diagnostics(env)
                conn.send(
                    {
                        'ok': True,
                        'env_id': env_id,
                        'next_obs': next_obs,
                        'reset_obs': reset_obs,
                        'reward': float(reward),
                        'terminated': bool(terminated),
                        'truncated': bool(truncated),
                        'done': done,
                        'info': info,
                        'episode_record': episode_record,
                    }
                )
            elif command == 'close':
                conn.send({'ok': True, 'env_id': env_id})
                break
            else:
                raise ValueError(f'Unknown worker command: {command}')
    except EOFError:
        pass
    except Exception as exc:
        try:
            conn.send(
                {
                    'ok': False,
                    'env_id': env_id,
                    'error': repr(exc),
                    'traceback': traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        if env is not None and hasattr(env, 'close'):
            try:
                env.close()
            except Exception:
                pass
        conn.close()


@dataclass(slots=True)
class _Worker:
    env_id: int
    conn: Any
    process: mp.Process


class ProcessVectorEnv:
    """Main-process handle for multiple environment worker processes."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        num_envs: int,
        base_seed: int,
        curriculum_level: str,
        curriculum_mix: dict[str, float],
        start_method: str | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError('num_envs must be positive.')
        method = start_method or ('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
        self.ctx = mp.get_context(method)
        self.workers: list[_Worker] = []
        self.closed = False
        for env_id in range(num_envs):
            parent_conn, child_conn = self.ctx.Pipe()
            seed = int(base_seed + env_id * 10000)
            process = self.ctx.Process(
                target=_worker_loop,
                args=(child_conn, env_id, cfg, seed, curriculum_level, curriculum_mix),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.workers.append(_Worker(env_id=env_id, conn=parent_conn, process=process))

    def __len__(self) -> int:
        return len(self.workers)

    def _check_response(self, response: dict[str, Any]) -> dict[str, Any]:
        if not response.get('ok', False):
            raise RuntimeError(
                f"Environment worker {response.get('env_id')} failed: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        return response

    def reset_all(self) -> list[np.ndarray]:
        for worker in self.workers:
            worker.conn.send(('reset', {'seed': None}))
        return [self._check_response(worker.conn.recv())['obs'] for worker in self.workers]

    def reset_at(self, index: int) -> np.ndarray:
        worker = self.workers[index]
        worker.conn.send(('reset', {'seed': None}))
        return self._check_response(worker.conn.recv())['obs']

    def step_at(self, indices: list[int], actions: list[np.ndarray]) -> list[dict[str, Any]]:
        if len(indices) != len(actions):
            raise ValueError('indices and actions must have the same length.')
        for index, action in zip(indices, actions):
            self.workers[index].conn.send(('step', action))
        return [self._check_response(self.workers[index].conn.recv()) for index in indices]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for worker in self.workers:
            if worker.process.is_alive():
                try:
                    worker.conn.send(('close', None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for worker in self.workers:
            try:
                if worker.process.is_alive() and worker.conn.poll(2.0):
                    worker.conn.recv()
            except (EOFError, OSError):
                pass
            worker.process.join(timeout=2.0)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=2.0)
            worker.conn.close()

    def __enter__(self) -> 'ProcessVectorEnv':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
