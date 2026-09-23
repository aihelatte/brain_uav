"""Offline V2 BC expert-action statistics (docs/无法早停排查文档.md section 5.3).

Not part of the production training pipeline and not imported by it. Given one
V2 expert trajectory cluster (the manifest+shard layout produced by
``generate_v2_trajectory_clusters.py`` and consumed by ``train_v2_bc.py``),
this reports the statistics of the expert actions the BC stage imitates:

- per action dimension (delta_gamma, delta_psi): mean/std/percentiles and
  extremes, the fraction of steps whose action magnitude is negligible
  relative to the action limit, and the constant-predictor MSE floor of the
  BC regression task;
- the same statistics per planner (v2_apf / v2_heuristic);
- per goal-distance bin, so a near-goal homing component (if any) is visible
  instead of being drowned by the straight-cruise majority;
- per-trajectory action std, counting how many expert trajectories are
  themselves near-constant.

The motivation is docs/无法早停排查文档.md section 5.3: the V2 ANN BC
validation loss was flat from the first epoch and the medium-stage
``bc_loss`` of both actors barely moved, so the suspicion is that the easy
expert actions have near-zero variance and BC converged to a near-constant
(straight-flight) policy. This script measures exactly that on the dataset
BC actually trains on, through the same strict loader used for training.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brain_uav.observations import GOAL_FEATURE_INDEX
from brain_uav.trainers.v2_bc import load_v2_bc_trajectory_cluster

_ACTION_NAMES = ('delta_gamma', 'delta_psi')
_DEFAULT_DISTANCE_BIN_EDGES_KM = (0.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
_NEGLIGIBLE_THRESHOLDS = (1e-3, 1e-2)
_PERCENTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9)


def _action_statistics(values: np.ndarray, action_limit: np.ndarray) -> dict[str, Any]:
    if values.shape[0] == 0:
        return {'count': 0}
    stats: dict[str, Any] = {'count': int(values.shape[0])}
    for dim, name in enumerate(_ACTION_NAMES):
        column = values[:, dim].astype(np.float64)
        limit = float(action_limit[dim])
        dim_stats: dict[str, Any] = {
            'mean': float(column.mean()),
            'std': float(column.std()),
            'min': float(column.min()),
            'max': float(column.max()),
            'std_over_action_limit': float(column.std() / limit),
            'percentiles': {
                f'p{p:g}': float(np.percentile(column, p)) for p in _PERCENTILES
            },
        }
        for threshold in _NEGLIGIBLE_THRESHOLDS:
            dim_stats[f'fraction_abs_below_{threshold:g}'] = float(
                (np.abs(column) < threshold).mean()
            )
        dim_stats['fraction_abs_over_10pct_of_limit'] = float(
            (np.abs(column) > 0.1 * limit).mean()
        )
        stats[name] = dim_stats
    squared_error = ((values - values.mean(axis=0, keepdims=True)) ** 2).sum(axis=1)
    stats['constant_predictor_mse_floor'] = float(squared_error.mean())
    return stats


def _per_trajectory_rows(cluster) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shard_arrays: dict[int, dict[str, np.ndarray]] = {}
    for record in cluster.trajectories:
        arrays = shard_arrays.get(record.shard_index)
        if arrays is None:
            arrays = cluster.shard_cache.load(cluster.shards[record.shard_index].path)
            shard_arrays[record.shard_index] = arrays
        actions = arrays['actions'][record.step_start : record.step_end].astype(
            np.float64
        )
        if actions.shape[0] == 0:
            continue
        rows.append(
            {
                'trajectory_id': record.trajectory_id,
                'planner': record.planner_name,
                'steps': int(actions.shape[0]),
                'delta_gamma_std': float(actions[:, 0].std()),
                'delta_psi_std': float(actions[:, 1].std()),
            }
        )
    return rows


def _trajectory_std_summary(rows: list[dict[str, Any]], action_limit: np.ndarray) -> dict[str, Any]:
    if not rows:
        return {'trajectory_count': 0}
    gamma_stds = np.asarray([row['delta_gamma_std'] for row in rows])
    psi_stds = np.asarray([row['delta_psi_std'] for row in rows])
    near_constant = (gamma_stds < 1e-3) & (psi_stds < 1e-3)
    return {
        'trajectory_count': len(rows),
        'median_delta_gamma_std': float(np.median(gamma_stds)),
        'median_delta_psi_std': float(np.median(psi_stds)),
        'fraction_near_constant_abs_below_1e-3': float(near_constant.mean()),
    }


def analyze_cluster(
    cluster_path: str | Path,
    *,
    distance_bin_edges_km: Sequence[float],
) -> dict[str, Any]:
    cluster = load_v2_bc_trajectory_cluster(cluster_path)
    scenario = cluster.scenario_config
    horizontal = 2.0 * scenario.world_xy
    vertical = scenario.world_z_max - scenario.world_z_min
    world_diagonal = float(np.sqrt(2.0 * horizontal * horizontal + vertical * vertical))
    action_limit = np.asarray(
        [scenario.delta_gamma_max, scenario.delta_psi_max], dtype=np.float64
    )
    distance_norm_index = int(GOAL_FEATURE_INDEX['goal_distance_norm'])

    actions_parts: list[np.ndarray] = []
    planner_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    shard_arrays: dict[int, dict[str, np.ndarray]] = {}
    for record in cluster.trajectories:
        arrays = shard_arrays.get(record.shard_index)
        if arrays is None:
            arrays = cluster.shard_cache.load(cluster.shards[record.shard_index].path)
            shard_arrays[record.shard_index] = arrays
        step_count = record.step_end - record.step_start
        if step_count <= 0:
            continue
        actions_parts.append(
            arrays['actions'][record.step_start : record.step_end].astype(np.float64)
        )
        planner_parts.append(
            np.full(step_count, record.planner_name, dtype=arrays['planner_names'].dtype)
        )
        goal_features = arrays['goal_features'][
            record.step_start : record.step_end
        ].astype(np.float64)
        # The V2 world unit is already km (speed is km/step), so the
        # normalized distance times the world diagonal is a distance in km.
        distance_parts.append(goal_features[:, distance_norm_index] * world_diagonal)

    if not actions_parts:
        raise ValueError('Cluster contains no non-empty successful trajectories.')
    actions = np.concatenate(actions_parts, axis=0)
    planner_names = np.concatenate(planner_parts, axis=0)
    distances_km = np.concatenate(distance_parts, axis=0)

    report: dict[str, Any] = {
        'cluster_dir': str(cluster.root),
        'manifest_statistics': cluster.manifest.get('statistics', {}),
        'trajectory_count': cluster.trajectory_count,
        'step_count': actions.shape[0],
        'action_limit': {
            'delta_gamma': float(action_limit[0]),
            'delta_psi': float(action_limit[1]),
        },
        'overall': _action_statistics(actions, action_limit),
        'per_planner': {},
        'per_distance_bin': [],
        'per_trajectory_std_summary': _trajectory_std_summary(
            _per_trajectory_rows(cluster), action_limit
        ),
    }

    for planner in sorted(set(str(name) for name in planner_names)):
        mask = planner_names == planner
        report['per_planner'][planner] = _action_statistics(actions[mask], action_limit)

    edges = [*distance_bin_edges_km, float('inf')]
    for bin_index in range(len(edges) - 1):
        mask = (distances_km >= edges[bin_index]) & (distances_km < edges[bin_index + 1])
        entry: dict[str, Any] = {
            # None encodes the open-ended upper edge so the report stays
            # strict-JSON serializable (allow_nan=False).
            'range_km': [edges[bin_index], None if not isfinite(edges[bin_index + 1]) else edges[bin_index + 1]],
            'count': int(mask.sum()),
        }
        if mask.any():
            bin_stats = _action_statistics(actions[mask], action_limit)
            entry['delta_gamma'] = bin_stats['delta_gamma']
            entry['delta_psi'] = bin_stats['delta_psi']
        report['per_distance_bin'].append(entry)
    return report


def _format_stats(stats: dict[str, Any], name: str) -> str:
    column = stats[name]
    return (
        f"{name}: mean={column['mean']:+.2e} std={column['std']:.2e} "
        f"({column['std_over_action_limit']:.2%} of limit) "
        f"min={column['min']:+.4f} max={column['max']:+.4f} "
        f"|a|<1e-3: {column['fraction_abs_below_0.001']:.2%} "
        f"|a|>10%limit: {column['fraction_abs_over_10pct_of_limit']:.2%}"
    )


def print_report(report: dict[str, Any]) -> None:
    print(f"cluster: {report['cluster_dir']}")
    print(
        f"trajectories={report['trajectory_count']} steps={report['step_count']} "
        f"action_limit={report['action_limit']}"
    )
    print('--- overall')
    for name in _ACTION_NAMES:
        print(' ', _format_stats(report['overall'], name))
    print(
        "  constant-predictor MSE floor: "
        f"{report['overall']['constant_predictor_mse_floor']:.3e}"
    )
    for planner, stats in report['per_planner'].items():
        print(f"--- planner: {planner} (steps={stats['count']})")
        for name in _ACTION_NAMES:
            print(' ', _format_stats(stats, name))
    print('--- per goal-distance bin')
    for entry in report['per_distance_bin']:
        lo, hi = entry['range_km']
        lo_text = f'{lo:g}' if isfinite(lo) else 'inf'
        hi_text = f'{hi:g}' if hi is not None else 'inf'
        if entry['count'] == 0:
            print(f'  [{lo_text}, {hi_text}) km: (empty)')
            continue
        gamma = entry['delta_gamma']
        psi = entry['delta_psi']
        print(
            f"  [{lo_text}, {hi_text}) km n={entry['count']}: "
            f"dγ std={gamma['std']:.2e} ({gamma['std_over_action_limit']:.2%} limit) "
            f"dψ std={psi['std']:.2e} ({psi['std_over_action_limit']:.2%} limit) "
            f"dψ |mean|={abs(psi['mean']):.2e}"
        )
    traj = report['per_trajectory_std_summary']
    if traj['trajectory_count']:
        print('--- per-trajectory action std')
        print(
            f"  trajectories={traj['trajectory_count']} "
            f"median dγ std={traj['median_delta_gamma_std']:.2e} "
            f"median dψ std={traj['median_delta_psi_std']:.2e} "
            f"near-constant (both dims |std|<1e-3): "
            f"{traj['fraction_near_constant_abs_below_1e-3']:.2%}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Measure V2 BC expert-action statistics from a trajectory cluster '
            '(docs/无法早停排查文档.md section 5.3).'
        )
    )
    parser.add_argument(
        '--trajectory-cluster', type=Path, required=True,
        help='V2 expert trajectory cluster directory (manifest.json + shards).',
    )
    parser.add_argument('--output-json', type=Path, default=None)
    parser.add_argument(
        '--distance-bin-edges-km',
        type=float,
        nargs='+',
        default=list(_DEFAULT_DISTANCE_BIN_EDGES_KM),
        help='Right-exclusive goal-distance bin edges in km; the last bin is open-ended.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    edges = list(args.distance_bin_edges_km)
    if len(edges) < 2 or any(
        later <= earlier for earlier, later in zip(edges, edges[1:])
    ):
        raise SystemExit(
            '--distance-bin-edges-km must contain at least two strictly '
            'increasing values.'
        )
    report = analyze_cluster(args.trajectory_cluster, distance_bin_edges_km=edges)
    print_report(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False),
            encoding='utf-8',
        )
        print(f'wrote: {args.output_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
