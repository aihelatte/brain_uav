"""Unified paper-grade benchmark entry for policies and rule-based planners."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..baselines import ArtificialPotentialFieldPlanner, HeuristicPlanner
from ..config import ExperimentConfig
from ..scenarios import (
    DEFAULT_BENCHMARK_SUITE_PATH,
    build_benchmark_scenarios,
    load_benchmark_suite,
)
from ..scripts.common import (
    DEVICE_CHOICES,
    SNN_BACKEND_CHOICES,
    configure_training_runtime,
    make_actor,
    make_env,
)
from ..scripts.evaluate import (
    AC_ENERGY_PJ,
    MAC_ENERGY_PJ,
    _count_params,
    _format_ops,
    _percentile,
    _syops_profile,
    _thop_macs,
)
from ..utils.io import ensure_dir, load_checkpoint, now_timestamp, save_json
from ..utils.seeding import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the fixed benchmark suite for policies and baselines.')
    parser.add_argument('--method', choices=['ann', 'snn', 'apf', 'heuristic'], required=True)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--benchmark-suite', type=Path, default=DEFAULT_BENCHMARK_SUITE_PATH)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-root', type=Path, default=Path('outputs/benchmark_runs'))
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--snn-backend', choices=SNN_BACKEND_CHOICES, default='torch')
    parser.add_argument('--episode-artifacts', choices=['json', 'none'], default='json')
    parser.add_argument('--reuse-obs-tensor', action='store_true')
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.method in {'ann', 'snn'} and args.checkpoint is None:
        parser.error(f'--checkpoint is required when --method {args.method}')
    if args.method in {'apf', 'heuristic'} and args.checkpoint is not None:
        parser.error(f'--checkpoint is not allowed when --method {args.method}')


def build_planner(method: str, env):
    if method == 'heuristic':
        return HeuristicPlanner(env)
    if method == 'apf':
        return ArtificialPotentialFieldPlanner(env)
    raise ValueError(f'Unsupported planner method: {method}')


def episode_min_zone_clearance(trajectory: list[list[float]] | list[np.ndarray], zones: list[dict[str, Any]]) -> float:
    if not zones:
        return float('inf')
    min_clearance = float('inf')
    for point in trajectory:
        pos = np.asarray(point, dtype=np.float32)
        for zone in zones:
            center_xy = zone['center_xy']
            radius = float(zone['radius'])
            distance = float(
                np.linalg.norm(
                    np.array([pos[0] - center_xy[0], pos[1] - center_xy[1], pos[2]], dtype=np.float32)
                )
            )
            min_clearance = min(min_clearance, distance - radius)
    return min_clearance


def _sanitize_token(value: str) -> str:
    token = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(value).strip())
    while '__' in token:
        token = token.replace('__', '_')
    return token.strip('_') or 'unknown'


def _maybe_float(value: float) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _build_output_dir(args: argparse.Namespace) -> Path:
    run_name = _sanitize_token(args.run_name) if args.run_name else f'{now_timestamp()}_{args.method}_benchmark'
    return ensure_dir(args.output_root / run_name)


def _make_episode_stem(episode_idx: int, scenario_id: str, category: str, outcome: str) -> str:
    return (
        f'ep{episode_idx:06d}_'
        f'{_sanitize_token(scenario_id)}_'
        f'{_sanitize_token(category)}_'
        f'{_sanitize_token(outcome)}'
    )


def _json_safe_config(cfg: ExperimentConfig) -> dict[str, Any]:
    return cfg.to_dict()


def _apply_checkpoint_model_config(cfg: ExperimentConfig, payload: dict[str, Any]) -> None:
    saved_config = payload.get('config')
    if not isinstance(saved_config, dict):
        return
    training_payload = saved_config.get('training')
    if not isinstance(training_payload, dict):
        return

    # Benchmark evaluation must keep the repo's current physics and reward defaults.
    # Only copy the model-construction fields needed to instantiate the actor shape/runtime.
    for key in ('hidden_dim', 'snn_time_window'):
        if key in training_payload and hasattr(cfg.training, key):
            setattr(cfg.training, key, training_payload[key])


def _build_episode_payload(
    *,
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    method: str,
    model_type: str,
    planner_type: str | None,
    checkpoint: Path | None,
    episode_idx: int,
    scenario_meta,
    total_steps_at_end: int,
    episode_return: float,
    steps: int,
    outcome: str,
    info: dict[str, Any],
    env,
    avg_decision_time_ms: float,
    max_decision_time_ms: float,
) -> dict[str, Any]:
    scenario_payload = env.export_scenario()
    scenario_payload.update(
        {
            'scenario_id': scenario_meta.scenario_id,
            'category': scenario_meta.category,
            'scenario_label': scenario_meta.name,
            'corridor_width': scenario_meta.corridor_width,
            'min_clearance_to_boundary': scenario_meta.min_clearance_to_boundary,
            'difficulty_score': scenario_meta.difficulty_score,
        }
    )
    trajectory = [point.astype(float).tolist() for point in env.trajectory]
    zone_count = len(scenario_payload['zones'])
    clearance = episode_min_zone_clearance(trajectory, scenario_payload['zones'])
    zone_radii = [float(zone['radius']) for zone in scenario_payload['zones']]
    zone_centers = [[float(coord) for coord in zone['center_xy']] for zone in scenario_payload['zones']]

    return {
        'method': method,
        'model_type': model_type,
        'planner_type': planner_type,
        'checkpoint': None if checkpoint is None else str(checkpoint),
        'episode': episode_idx,
        'scenario_id': scenario_meta.scenario_id,
        'category': scenario_meta.category,
        'scenario_label': scenario_meta.name,
        'outcome': outcome,
        'return': float(episode_return),
        'length': int(steps),
        'total_steps_at_end': int(total_steps_at_end),
        'final_state': [float(value) for value in env.state.tolist()],
        'trajectory': trajectory,
        'scenario': scenario_payload,
        'info': {
            'goal_distance': float(info.get('goal_distance', 0.0)),
            'segment_goal_distance': float(info.get('segment_goal_distance', 0.0)),
            'goal_reached_by_segment': bool(info.get('goal_reached_by_segment', False)),
            'progress': float(info.get('progress', 0.0)),
            'steps': int(info.get('steps', steps)),
            'curriculum_level': info.get('curriculum_level'),
            'active_goal_radius': float(info.get('active_goal_radius', cfg.scenario.goal_radius)),
            'scenario_id': scenario_meta.scenario_id,
            'category': scenario_meta.category,
            'scenario_label': scenario_meta.name,
            'zone_count': zone_count,
            'corridor_width': scenario_meta.corridor_width,
            'min_clearance_to_boundary': scenario_meta.min_clearance_to_boundary,
            'difficulty_score': scenario_meta.difficulty_score,
        },
        'zone_count': zone_count,
        'zone_radii': zone_radii,
        'zone_centers': zone_centers,
        'corridor_width': scenario_meta.corridor_width,
        'min_clearance_to_boundary': scenario_meta.min_clearance_to_boundary,
        'difficulty_score': scenario_meta.difficulty_score,
        'episode_min_zone_clearance': _maybe_float(clearance),
        'active_goal_radius': float(info.get('active_goal_radius', cfg.scenario.goal_radius)),
        'avg_decision_time_ms': float(avg_decision_time_ms),
        'max_decision_time_ms': float(max_decision_time_ms),
        'config': _json_safe_config(cfg),
    }


def _build_zone_count_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record['zone_count'])
        bucket = summary.setdefault(key, {'episodes': 0, 'success_count': 0, 'outcomes': {}})
        bucket['episodes'] += 1
        if record['outcome'] == 'goal':
            bucket['success_count'] += 1
        outcome = str(record['outcome'])
        bucket['outcomes'][outcome] = bucket['outcomes'].get(outcome, 0) + 1
    for bucket in summary.values():
        bucket['success_rate'] = 0.0 if bucket['episodes'] == 0 else bucket['success_count'] / bucket['episodes']
    return summary


def _build_outcome_by_zone_count(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        zone_key = str(record['zone_count'])
        bucket = summary.setdefault(zone_key, {})
        outcome = str(record['outcome'])
        bucket[outcome] = bucket.get(outcome, 0) + 1
    return summary


def _build_category_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for record in records:
        category = str(record['category'])
        bucket = summary.setdefault(
            category,
            {
                'episodes': 0,
                'success_count': 0,
                'outcomes': {},
                'steps': [],
                'returns': [],
                'goal_distances': [],
                'clearances': [],
            },
        )
        bucket['episodes'] += 1
        if record['outcome'] == 'goal':
            bucket['success_count'] += 1
        outcome = str(record['outcome'])
        bucket['outcomes'][outcome] = bucket['outcomes'].get(outcome, 0) + 1
        bucket['steps'].append(int(record['steps']))
        bucket['returns'].append(float(record['return']))
        bucket['goal_distances'].append(float(record['goal_distance']))
        if record['episode_min_zone_clearance'] is not None:
            bucket['clearances'].append(float(record['episode_min_zone_clearance']))

    result: dict[str, dict[str, Any]] = {}
    for category, bucket in summary.items():
        episodes = bucket['episodes']
        result[category] = {
            'episodes': episodes,
            'success_count': bucket['success_count'],
            'success_rate': 0.0 if episodes == 0 else bucket['success_count'] / episodes,
            'outcomes': bucket['outcomes'],
            'collision_rate': 0.0 if episodes == 0 else bucket['outcomes'].get('collision', 0) / episodes,
            'boundary_rate': 0.0 if episodes == 0 else bucket['outcomes'].get('boundary', 0) / episodes,
            'ground_rate': 0.0 if episodes == 0 else bucket['outcomes'].get('ground', 0) / episodes,
            'timeout_rate': 0.0 if episodes == 0 else bucket['outcomes'].get('timeout', 0) / episodes,
            'avg_steps': 0.0 if not bucket['steps'] else statistics.mean(bucket['steps']),
            'avg_return': 0.0 if not bucket['returns'] else statistics.mean(bucket['returns']),
            'avg_final_goal_distance': (
                0.0 if not bucket['goal_distances'] else statistics.mean(bucket['goal_distances'])
            ),
            'avg_episode_min_zone_clearance': (
                None if not bucket['clearances'] else statistics.mean(bucket['clearances'])
            ),
        }
    return result


def _build_planner_efficiency_summary(
    *,
    method: str,
    device_name: str,
    cfg: ExperimentConfig,
    records: list[dict[str, Any]],
    decision_times_ms: list[float],
    episode_decision_times_s: list[float],
) -> dict[str, Any]:
    avg_zone_count = statistics.mean(record['zone_count'] for record in records) if records else 0.0
    if method == 'heuristic':
        planner_estimated_ops_per_step = int(round(160.0 + 96.0 * avg_zone_count))
    else:
        planner_estimated_ops_per_step = int(round(128.0 + 80.0 * avg_zone_count))
    avg_decision_time_ms = 0.0 if not decision_times_ms else statistics.mean(decision_times_ms)
    return {
        'method': method,
        'model_type': 'planner',
        'planner_type': method,
        'device': device_name,
        'torch_version': torch.__version__,
        'decision_time_unit': 'ms',
        'avg_decision_time_ms': avg_decision_time_ms,
        'p50_decision_time_ms': _percentile(decision_times_ms, 50.0),
        'p95_decision_time_ms': _percentile(decision_times_ms, 95.0),
        'p99_decision_time_ms': _percentile(decision_times_ms, 99.0),
        'max_decision_time_ms': 0.0 if not decision_times_ms else max(decision_times_ms),
        'avg_episode_decision_time_s': 0.0 if not episode_decision_times_s else statistics.mean(episode_decision_times_s),
        'max_episode_decision_time_s': 0.0 if not episode_decision_times_s else max(episode_decision_times_s),
        'decision_dt_s': float(cfg.scenario.dt),
        'estimated_steps_for_1000s': int(round(1000.0 / float(cfg.scenario.dt))),
        'estimated_1000s_planning_time_s': (avg_decision_time_ms / 1000.0) * int(round(1000.0 / float(cfg.scenario.dt))),
        'planner_param_count': 0,
        'planner_estimated_ops_per_step': planner_estimated_ops_per_step,
        'planner_estimated_ops_method': 'rough constant-factor estimate based on average zone_count',
        'planner_complexity': 'O(num_zones)',
        'planner_energy_pj': None,
        'planner_energy_assumptions': 'not estimated for rule-based planners',
    }


def _timing_distribution(values: list[float]) -> dict[str, float | int]:
    return {
        'samples': len(values),
        'avg': 0.0 if not values else statistics.mean(values),
        'p50': _percentile(values, 50.0),
        'p95': _percentile(values, 95.0),
        'p99': _percentile(values, 99.0),
        'max': 0.0 if not values else max(values),
    }


def _build_policy_efficiency_summary(
    *,
    method: str,
    actor: torch.nn.Module,
    example_obs: np.ndarray,
    device: torch.device,
    decision_times_ms: list[float],
    episode_decision_times_s: list[float],
    obs_to_tensor_times_ms: list[float],
    actor_forward_times_ms: list[float],
    action_to_cpu_times_ms: list[float],
    reuse_obs_tensor: bool,
    cfg: ExperimentConfig,
    diag_stats: dict[str, float],
) -> dict[str, Any]:
    param_count, trainable_param_count = _count_params(actor)
    example_input = torch.tensor(example_obs[None, :], dtype=torch.float32, device=device)
    dense_macs, _dense_params, macs_method = _thop_macs(actor, example_input)
    dense_theoretical_flops = None if dense_macs is None else float(dense_macs) * 2.0
    syops_payload: dict[str, Any] = {}
    if method == 'snn':
        syops_payload = _syops_profile(actor, example_input)

    avg_spike_rate_l1 = None
    avg_spike_rate_l2 = None
    if diag_stats['samples'] > 0:
        avg_spike_rate_l1 = diag_stats['l1'] / diag_stats['samples']
        avg_spike_rate_l2 = diag_stats['l2'] / diag_stats['samples']

    snn_acs = syops_payload.get('snn_acs')
    snn_macs = syops_payload.get('snn_macs')
    snn_spike_aware_ops = None
    snn_energy_pj = None
    if isinstance(snn_acs, (int, float)) and isinstance(snn_macs, (int, float)):
        snn_spike_aware_ops = float(snn_acs) + float(snn_macs)
        snn_energy_pj = float(snn_acs) * AC_ENERGY_PJ + float(snn_macs) * MAC_ENERGY_PJ

    ann_macs = dense_macs if method == 'ann' else None
    ann_energy_pj = None if ann_macs is None else float(ann_macs) * MAC_ENERGY_PJ
    avg_decision_time_ms = 0.0 if not decision_times_ms else statistics.mean(decision_times_ms)
    estimated_steps_for_1000s = int(round(1000.0 / float(cfg.scenario.dt)))

    return {
        'method': method,
        'model_type': method,
        'planner_type': None,
        'device': str(device),
        'torch_version': torch.__version__,
        'decision_time_unit': 'ms',
        'avg_decision_time_ms': avg_decision_time_ms,
        'p50_decision_time_ms': _percentile(decision_times_ms, 50.0),
        'p95_decision_time_ms': _percentile(decision_times_ms, 95.0),
        'p99_decision_time_ms': _percentile(decision_times_ms, 99.0),
        'max_decision_time_ms': 0.0 if not decision_times_ms else max(decision_times_ms),
        'obs_to_tensor_time_ms': _timing_distribution(obs_to_tensor_times_ms),
        'actor_forward_time_ms': _timing_distribution(actor_forward_times_ms),
        'action_to_cpu_time_ms': _timing_distribution(action_to_cpu_times_ms),
        'decision_time_ms': _timing_distribution(decision_times_ms),
        'reuse_obs_tensor': bool(reuse_obs_tensor),
        'avg_episode_decision_time_s': 0.0 if not episode_decision_times_s else statistics.mean(episode_decision_times_s),
        'max_episode_decision_time_s': 0.0 if not episode_decision_times_s else max(episode_decision_times_s),
        'decision_dt_s': float(cfg.scenario.dt),
        'estimated_steps_for_1000s': estimated_steps_for_1000s,
        'estimated_1000s_planning_time_s': (avg_decision_time_ms / 1000.0) * estimated_steps_for_1000s,
        'param_count': int(param_count),
        'trainable_param_count': int(trainable_param_count),
        'dense_theoretical_macs': dense_macs,
        'dense_theoretical_flops': dense_theoretical_flops,
        'avg_spike_rate_l1': avg_spike_rate_l1,
        'avg_spike_rate_l2': avg_spike_rate_l2,
        'ann_macs': ann_macs,
        'ann_macs_ops': _format_ops(ann_macs),
        'ann_energy_pj': ann_energy_pj,
        'ann_macs_method': macs_method if method == 'ann' else None,
        'ann_energy_assumptions': f'ANN energy estimate uses MAC={MAC_ENERGY_PJ} pJ.' if ann_energy_pj is not None else None,
        'snn_acs': snn_acs,
        'snn_macs': snn_macs,
        'snn_ac_ops': syops_payload.get('snn_ac_ops'),
        'snn_mac_ops': syops_payload.get('snn_mac_ops'),
        'snn_spike_aware_ops': snn_spike_aware_ops,
        'snn_energy_pj': snn_energy_pj,
        'syops_method': syops_payload.get('syops_method'),
        'syops_assumptions': syops_payload.get('syops_assumptions'),
        'energy_assumptions': (
            f'SNN energy estimate uses AC={AC_ENERGY_PJ} pJ and MAC={MAC_ENERGY_PJ} pJ.'
            if snn_energy_pj is not None
            else None
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ExperimentConfig()
    set_global_seed(args.seed)
    suite_payload = load_benchmark_suite(args.benchmark_suite)
    named_scenarios = build_benchmark_scenarios(args.benchmark_suite)
    suite_size = len(named_scenarios)
    episodes = suite_size if args.episodes is None else int(args.episodes)
    if episodes > suite_size:
        raise ValueError(
            f'--episodes={episodes} exceeds benchmark suite size {suite_size} at {args.benchmark_suite}.'
        )

    checkpoint_payload: dict[str, Any] | None = None
    actor: torch.nn.Module | None = None
    planner = None
    diag_stats = {'samples': 0.0, 'l1': 0.0, 'l2': 0.0}
    resolved_device = 'cpu'
    torch_device = torch.device('cpu')

    if args.method in {'ann', 'snn'}:
        checkpoint_payload = load_checkpoint(args.checkpoint)
        _apply_checkpoint_model_config(cfg, checkpoint_payload)
        resolved_device = configure_training_runtime(
            cfg,
            model_type=args.method,
            device=args.device,
            snn_backend=args.snn_backend,
        )
        torch_device = torch.device(resolved_device)
    else:
        resolved_device = 'cpu'

    env = make_env(
        cfg,
        seed=args.seed,
        scenario_suite=None,
        goal_radius_curriculum_enabled=True,
    )
    example_obs_for_efficiency = np.zeros(env.observation_space.shape[0], dtype=np.float32)
    if args.method in {'ann', 'snn'}:
        assert checkpoint_payload is not None
        actor = make_actor(cfg, args.method, env.observation_space.shape[0], env.action_space.shape[0])
        actor.load_state_dict(checkpoint_payload['state_dict'])
        actor.to(torch_device)
        actor.eval()
    else:
        planner = build_planner(args.method, env)
    reuse_obs_tensor = bool(getattr(args, 'reuse_obs_tensor', False))
    obs_tensor_buffer: torch.Tensor | None = None
    if actor is not None and reuse_obs_tensor:
        obs_tensor_buffer = torch.empty(
            (1, env.observation_space.shape[0]),
            dtype=torch.float32,
            device=torch_device,
        )

    output_dir = _build_output_dir(args)
    episodes_dir = ensure_dir(output_dir / 'episodes')
    records: list[dict[str, Any]] = []
    total_steps = 0
    decision_times_ms: list[float] = []
    obs_to_tensor_times_ms: list[float] = []
    actor_forward_times_ms: list[float] = []
    action_to_cpu_times_ms: list[float] = []
    episode_decision_times_s: list[float] = []

    for episode_idx in range(episodes):
        scenario_meta = named_scenarios[episode_idx]
        obs, _ = env.reset(options={'scenario': scenario_meta.scenario})
        if episode_idx == 0:
            example_obs_for_efficiency = obs.copy()
        done = False
        episode_return = 0.0
        info: dict[str, Any] = {}
        episode_decisions: list[float] = []

        while not done:
            if actor is not None:
                step_start = time.perf_counter()
                with torch.inference_mode():
                    obs_tensor_start = time.perf_counter()
                    if obs_tensor_buffer is None:
                        obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32, device=torch_device)
                    else:
                        obs_tensor_buffer.copy_(
                            torch.as_tensor(obs, dtype=torch.float32, device=torch_device).view(1, -1)
                        )
                        obs_tensor = obs_tensor_buffer
                    obs_to_tensor_ms = (time.perf_counter() - obs_tensor_start) * 1000.0

                    actor_forward_start = time.perf_counter()
                    action_tensor = actor(obs_tensor)
                    actor_forward_ms = (time.perf_counter() - actor_forward_start) * 1000.0

                    action_to_cpu_start = time.perf_counter()
                    action = action_tensor.detach().cpu().numpy()[0]
                    action_to_cpu_ms = (time.perf_counter() - action_to_cpu_start) * 1000.0
                decision_ms = (time.perf_counter() - step_start) * 1000.0
                obs_to_tensor_times_ms.append(obs_to_tensor_ms)
                actor_forward_times_ms.append(actor_forward_ms)
                action_to_cpu_times_ms.append(action_to_cpu_ms)
                if hasattr(actor, 'forward_with_diagnostics') and int(total_steps + len(episode_decisions)) % 10 == 0:
                    with torch.no_grad():
                        _ = action_tensor
                        _diag_action, diag = actor.forward_with_diagnostics(obs_tensor)
                    diag_stats['samples'] += 1.0
                    diag_stats['l1'] += float(diag.get('spike_rate_l1', 0.0))
                    diag_stats['l2'] += float(diag.get('spike_rate_l2', 0.0))
            else:
                step_start = time.perf_counter()
                action = planner.act(obs)
                decision_ms = (time.perf_counter() - step_start) * 1000.0

            decision_times_ms.append(decision_ms)
            episode_decisions.append(decision_ms)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = bool(terminated or truncated)

        steps = int(env.steps)
        total_steps += steps
        episode_decision_time_s = sum(episode_decisions) / 1000.0
        episode_decision_times_s.append(episode_decision_time_s)
        outcome = str(info.get('outcome', 'other'))
        episode_payload = _build_episode_payload(
            args=args,
            cfg=cfg,
            method=args.method,
            model_type=args.method if args.method in {'ann', 'snn'} else 'planner',
            planner_type=None if args.method in {'ann', 'snn'} else args.method,
            checkpoint=args.checkpoint,
            episode_idx=episode_idx + 1,
            scenario_meta=scenario_meta,
            total_steps_at_end=total_steps,
            episode_return=episode_return,
            steps=steps,
            outcome=outcome,
            info=info,
            env=env,
            avg_decision_time_ms=statistics.mean(episode_decisions),
            max_decision_time_ms=max(episode_decisions),
        )

        episode_json_path: str | None = None
        if args.episode_artifacts == 'json':
            stem = _make_episode_stem(episode_idx + 1, scenario_meta.scenario_id, scenario_meta.category, outcome)
            episode_path = episodes_dir / f'{stem}.json'
            save_json(episode_path, episode_payload)
            episode_json_path = str(episode_path)

        records.append(
            {
                'episode': episode_idx + 1,
                'scenario_id': scenario_meta.scenario_id,
                'category': scenario_meta.category,
                'scenario_label': scenario_meta.name,
                'outcome': outcome,
                'steps': steps,
                'return': float(episode_return),
                'goal_distance': float(episode_payload['info']['goal_distance']),
                'episode_min_zone_clearance': episode_payload['episode_min_zone_clearance'],
                'zone_count': int(episode_payload['zone_count']),
                'avg_decision_time_ms': float(episode_payload['avg_decision_time_ms']),
                'max_decision_time_ms': float(episode_payload['max_decision_time_ms']),
                'episode_json': episode_json_path,
            }
        )

    success_count = sum(1 for record in records if record['outcome'] == 'goal')
    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record['outcome']] = outcomes.get(record['outcome'], 0) + 1

    goal_distances = [float(record['goal_distance']) for record in records]
    clearances = [
        float(record['episode_min_zone_clearance'])
        for record in records
        if record['episode_min_zone_clearance'] is not None
    ]

    if actor is not None:
        efficiency_summary = _build_policy_efficiency_summary(
            method=args.method,
            actor=actor,
            example_obs=example_obs_for_efficiency,
            device=torch_device,
            decision_times_ms=decision_times_ms,
            episode_decision_times_s=episode_decision_times_s,
            obs_to_tensor_times_ms=obs_to_tensor_times_ms,
            actor_forward_times_ms=actor_forward_times_ms,
            action_to_cpu_times_ms=action_to_cpu_times_ms,
            reuse_obs_tensor=reuse_obs_tensor,
            cfg=cfg,
            diag_stats=diag_stats,
        )
    else:
        efficiency_summary = _build_planner_efficiency_summary(
            method=args.method,
            device_name=resolved_device,
            cfg=cfg,
            records=records,
            decision_times_ms=decision_times_ms,
            episode_decision_times_s=episode_decision_times_s,
        )

    efficiency_path = output_dir / 'efficiency_summary.json'
    save_json(efficiency_path, efficiency_summary)
    episodes_index_path = episodes_dir / 'index.json'
    save_json(episodes_index_path, records)

    physics = {
        'dt': float(cfg.scenario.dt),
        'max_steps_per_episode': int(cfg.scenario.max_steps),
        'world_xy': float(cfg.scenario.world_xy),
        'world_z_min': float(cfg.scenario.world_z_min),
        'world_z_max': float(cfg.scenario.world_z_max),
        'goal_radius': float(cfg.scenario.goal_radius),
        'goal_radius_curriculum_enabled': True,
        'goal_radius_curriculum': cfg.scenario.goal_radius_curriculum,
        'distance_range_benchmark': list(cfg.scenario.distance_range_for_level('benchmark')),
        'radius_range_benchmark': list(cfg.scenario.radius_range_for_level('benchmark')),
        'warning_distance': float(cfg.scenario.warning_distance),
        'no_fly_zone_type': 'static_hemisphere',
    }
    summary = {
        'method': args.method,
        'model_type': args.method if args.method in {'ann', 'snn'} else 'planner',
        'planner_type': None if args.method in {'ann', 'snn'} else args.method,
        'checkpoint': None if args.checkpoint is None else str(args.checkpoint),
        'seed': args.seed,
        'evaluation_mode': 'benchmark',
        'benchmark_suite_name': suite_payload['suite_name'],
        'benchmark_suite_path': str(args.benchmark_suite),
        'benchmark_suite_seed': suite_payload['seed'],
        'benchmark_count_per_category': suite_payload['count_per_category'],
        'benchmark_total_scenarios': suite_payload['total_scenarios'],
        'categories': suite_payload['categories'],
        'episodes': episodes,
        'physics': physics,
        'total_steps': total_steps,
        'success_count': success_count,
        'success_rate': 0.0 if episodes == 0 else success_count / episodes,
        'outcomes': outcomes,
        'collision_rate': 0.0 if episodes == 0 else outcomes.get('collision', 0) / episodes,
        'boundary_rate': 0.0 if episodes == 0 else outcomes.get('boundary', 0) / episodes,
        'ground_rate': 0.0 if episodes == 0 else outcomes.get('ground', 0) / episodes,
        'timeout_rate': 0.0 if episodes == 0 else outcomes.get('timeout', 0) / episodes,
        'avg_steps': 0.0 if not records else statistics.mean(record['steps'] for record in records),
        'median_steps': 0.0 if not records else float(statistics.median(record['steps'] for record in records)),
        'p90_steps': 0.0 if not records else _percentile([float(record['steps']) for record in records], 90.0),
        'avg_return': 0.0 if not records else statistics.mean(record['return'] for record in records),
        'median_return': 0.0 if not records else float(statistics.median(record['return'] for record in records)),
        'avg_final_goal_distance': 0.0 if not goal_distances else statistics.mean(goal_distances),
        'avg_episode_min_zone_clearance': None if not clearances else statistics.mean(clearances),
        'min_episode_min_zone_clearance': None if not clearances else min(clearances),
        'category_summary': _build_category_summary(records),
        'zone_count_summary': _build_zone_count_summary(records),
        'outcome_by_zone_count': _build_outcome_by_zone_count(records),
        'output_dir': str(output_dir),
        'episodes_dir': str(episodes_dir),
        'episodes_index_path': str(episodes_index_path),
        'efficiency_summary_path': str(efficiency_path),
        'plots_dir': None,
        'records': records,
    }
    save_json(output_dir / 'summary.json', summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    summary = run_benchmark(args)
    print(f"Saved benchmark summary to {Path(summary['output_dir']) / 'summary.json'}")
    print(f"Saved benchmark efficiency summary to {summary['efficiency_summary_path']}")
    print(summary)


if __name__ == '__main__':
    main()
