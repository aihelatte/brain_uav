"""Offline SNN-vs-ANN action distribution and zone-sensitivity tool (B2/H9).

Not part of the production training pipeline. Given one ANN checkpoint and
one SNN checkpoint (both formal checkpoints or C1 periodic snapshots) that
share the same ScenarioConfig, and a fixed validation pool (for example
``v2_validation_medium.json``), this loads both actors and, for the same
batch of fixed observations:

- reports each model's action-output distribution (mean/std/variance per
  action dimension) across the whole pool, and
- for a subset of scenarios that contain at least one no-fly zone,
  perturbs the nearest zone's center by a fixed offset and measures each
  model's action-delta magnitude, as a finite-difference sensitivity
  estimate to "nearest zone distance/angle changing".

This directly supports H9 in docs/无法早停排查文档.md (whether the SNN
decision head has enough expressive capacity for the avoidance maneuvers
medium requires).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.observations import collate_v2_observations
from brain_uav.scripts.common import DEVICE_CHOICES, resolve_training_device
from brain_uav.scripts.diagnose_discount_horizon import load_actor_and_critics
from brain_uav.trainers.v2_validation import (
    V2ValidationPool,
    load_v2_validation_pool,
    scenario_config_snapshot,
)


def _actor_action(actor, observation, *, device: str) -> np.ndarray:
    batch = collate_v2_observations([observation], device=device)
    with torch.no_grad():
        return actor(batch)[0].detach().cpu().numpy().astype(np.float64)


def _zone_center_key(shape: dict[str, Any]) -> str:
    return 'base_center' if 'base_center' in shape else 'center'


def _nearest_zone_index(payload: dict[str, Any]) -> int | None:
    zones = payload['zones']
    if not zones:
        return None
    state_xy = np.asarray(payload['state'][:2], dtype=np.float64)
    distances = [
        float(
            np.linalg.norm(
                np.asarray(
                    zone['shape'][_zone_center_key(zone['shape'])][:2],
                    dtype=np.float64,
                )
                - state_xy
            )
        )
        for zone in zones
    ]
    return int(np.argmin(distances))


def _perturbed_payload(
    payload: dict[str, Any], *, zone_index: int, delta_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    perturbed = deepcopy(payload)
    shape = perturbed['zones'][zone_index]['shape']
    key = _zone_center_key(shape)
    shape[key] = [component + delta for component, delta in zip(shape[key], delta_xyz)]
    return perturbed


def _distribution_stats(actions: np.ndarray) -> dict[str, Any]:
    return {
        'mean': actions.mean(axis=0).tolist(),
        'std': actions.std(axis=0).tolist(),
        'variance': actions.var(axis=0).tolist(),
        'overall_variance': float(actions.var()),
    }


def compare_action_distributions(
    *,
    ann_checkpoint: str | Path,
    snn_checkpoint: str | Path,
    validation_pool: str | Path,
    curriculum_level: str,
    device: str = 'cpu',
    perturbation_magnitude_km: float = 5.0,
    perturbation_scenario_limit: int = 20,
) -> dict[str, Any]:
    ann_models = load_actor_and_critics(ann_checkpoint, model_type='ann', device=device)
    snn_models = load_actor_and_critics(snn_checkpoint, model_type='snn', device=device)
    if (
        scenario_config_snapshot(ann_models['scenario'])
        != scenario_config_snapshot(snn_models['scenario'])
        or ann_models['uav_collision_radius'] != snn_models['uav_collision_radius']
    ):
        raise ValueError(
            'ANN and SNN checkpoints must share one ScenarioConfig and '
            'uav_collision_radius to be compared on the same observations.'
        )
    scenario = ann_models['scenario']
    radius = ann_models['uav_collision_radius']
    pool: V2ValidationPool = load_v2_validation_pool(
        validation_pool,
        expected_level=curriculum_level,
        expected_scenario=scenario,
        expected_uav_collision_radius=radius,
    )
    env = V2StaticNoFlyTrajectoryEnv(
        scenario, ann_models['rewards'], uav_collision_radius=radius,
    )

    ann_actions: list[np.ndarray] = []
    snn_actions: list[np.ndarray] = []
    for record in pool.scenarios:
        observation, _ = env.reset(options={'scenario': record['payload']})
        ann_actions.append(_actor_action(ann_models['actor'], observation, device=device))
        snn_actions.append(_actor_action(snn_models['actor'], observation, device=device))
    ann_action_array = np.stack(ann_actions)
    snn_action_array = np.stack(snn_actions)

    perturbable = [
        record for record in pool.scenarios if record['payload']['zones']
    ][:perturbation_scenario_limit]
    per_scenario_sensitivity: list[dict[str, Any]] = []
    for record in perturbable:
        payload = record['payload']
        zone_index = _nearest_zone_index(payload)
        if zone_index is None:
            continue
        baseline_observation, _ = env.reset(options={'scenario': payload})
        baseline_ann = _actor_action(ann_models['actor'], baseline_observation, device=device)
        baseline_snn = _actor_action(snn_models['actor'], baseline_observation, device=device)
        perturbed_payload = _perturbed_payload(
            payload,
            zone_index=zone_index,
            delta_xyz=(perturbation_magnitude_km, 0.0, 0.0),
        )
        perturbed_observation, _ = env.reset(options={'scenario': perturbed_payload})
        perturbed_ann = _actor_action(ann_models['actor'], perturbed_observation, device=device)
        perturbed_snn = _actor_action(snn_models['actor'], perturbed_observation, device=device)
        per_scenario_sensitivity.append({
            'scenario_id': record['scenario_id'],
            'perturbed_zone_index': zone_index,
            'ann_action_delta_norm': float(np.linalg.norm(perturbed_ann - baseline_ann)),
            'snn_action_delta_norm': float(np.linalg.norm(perturbed_snn - baseline_snn)),
        })

    ann_sensitivities = [item['ann_action_delta_norm'] for item in per_scenario_sensitivity]
    snn_sensitivities = [item['snn_action_delta_norm'] for item in per_scenario_sensitivity]

    def _sensitivity_summary(values: list[float]) -> dict[str, float]:
        return {
            'mean': float(np.mean(values)) if values else 0.0,
            'std': float(np.std(values)) if values else 0.0,
            'max': float(np.max(values)) if values else 0.0,
        }

    return {
        'purpose': 'ann_vs_snn_action_distribution_and_zone_sensitivity',
        'validation_pool': str(validation_pool),
        'curriculum_level': curriculum_level,
        'scenario_count': pool.scenario_count,
        'perturbation_magnitude_km': perturbation_magnitude_km,
        'perturbation_scenario_count': len(per_scenario_sensitivity),
        'ann': {
            'checkpoint': str(ann_checkpoint),
            'action_distribution': _distribution_stats(ann_action_array),
            'zone_perturbation_sensitivity': _sensitivity_summary(ann_sensitivities),
        },
        'snn': {
            'checkpoint': str(snn_checkpoint),
            'action_distribution': _distribution_stats(snn_action_array),
            'zone_perturbation_sensitivity': _sensitivity_summary(snn_sensitivities),
        },
        'per_scenario_sensitivity': per_scenario_sensitivity,
    }


def _write_strict_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f'Output already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Compare ANN vs SNN action-output distributions and zone-'
            'perturbation sensitivity (B2/H9) on one fixed validation pool.'
        )
    )
    parser.add_argument('--ann-checkpoint', type=Path, required=True)
    parser.add_argument('--snn-checkpoint', type=Path, required=True)
    parser.add_argument('--validation-pool', type=Path, required=True)
    parser.add_argument('--curriculum-level', choices=('easy', 'medium', 'hard'), required=True)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--perturbation-magnitude-km', type=float, default=5.0)
    parser.add_argument('--perturbation-scenario-limit', type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_device = resolve_training_device(args.device)
    result = compare_action_distributions(
        ann_checkpoint=args.ann_checkpoint,
        snn_checkpoint=args.snn_checkpoint,
        validation_pool=args.validation_pool,
        curriculum_level=args.curriculum_level,
        device=resolved_device,
        perturbation_magnitude_km=args.perturbation_magnitude_km,
        perturbation_scenario_limit=args.perturbation_scenario_limit,
    )
    _write_strict_json(args.output, result)
    print(json.dumps({
        key: value for key, value in result.items() if key != 'per_scenario_sensitivity'
    }, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
