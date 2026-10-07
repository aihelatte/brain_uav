"""Diagnostic-only, noise-free evaluation of a terminal V2 policy checkpoint.

Failed checkpoints are readable here; these reports never change formal status.
Run with ``python -m brain_uav.scripts.diagnose_v2_policy --help``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.models import V2ANNPolicyActor, V2SNNPolicyActor
from brain_uav.observations import V2ObservationScales, collate_v2_observations
from brain_uav.trainers.v2_formal_training import (
    _architecture_from_engine_payload, load_v2_formal_checkpoint,
)
from brain_uav.trainers.v2_reporting import export_v2_trajectory_views
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from brain_uav.trainers.v2_validation import (
    V2_VALIDATION_POOL_VERSION, load_v2_validation_pool, scenario_config_from_snapshot,
)


OUTCOMES = ('goal', 'ground', 'boundary', 'collision', 'timeout')


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        stream.write(_json(value) + '\n')


def _optional_distance(value: float) -> float | None:
    if value == float('inf'):
        return None
    if not math.isfinite(value):
        raise RuntimeError('Diagnostic distance is not finite.')
    return float(value)


def load_diagnostic_inputs(*, model: str, checkpoint: str | Path,
                           validation_pool: str | Path, device: str) -> dict[str, Any]:
    """Use strict loaders and bind the original pool, without building an engine."""
    if model not in ('ann', 'snn') or device not in ('cpu', 'cuda'):
        raise ValueError('model must be ann/snn and device must be cpu/cuda.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False.')
    target_device = torch.device('cpu' if device == 'cpu' else f'cuda:{torch.cuda.current_device()}')
    checkpoint_path = Path(checkpoint).resolve()
    pool_path = Path(validation_pool).resolve()
    checkpoint_hash, pool_hash = file_sha256(checkpoint_path), file_sha256(pool_path)
    payload = load_v2_formal_checkpoint(
        checkpoint_path, expected_model_type=model, require_passed=False,
    )
    scenario = scenario_config_from_snapshot(payload['scenario_config'])
    rewards = RewardConfig(**payload['reward_config'])
    radius = float(payload['uav_collision_radius'])
    pool_metadata = payload['validation_pool']
    if pool_metadata['format_version'] != V2_VALIDATION_POOL_VERSION:
        raise ValueError('Checkpoint validation pool format version is incompatible.')
    pool = load_v2_validation_pool(
        pool_path, expected_level=payload['stage'], expected_scenario=scenario,
        expected_count=pool_metadata['scenario_count'], expected_uav_collision_radius=radius,
        expected_master_seed=pool_metadata['master_seed'], expected_stage_seed=pool_metadata['stage_seed'],
    )
    if pool.content_digest != pool_metadata['content_digest']:
        raise ValueError('Validation pool content_digest differs from checkpoint provenance.')
    engine_payload = payload['engine_checkpoint']
    if engine_payload.get('observation_contract') != V2TD3UpdateEngine._observation_contract():
        raise ValueError('Checkpoint structured observation contract is incompatible.')
    scales, encoder, checkpoint_radius, hidden, _, _, action_high, snn = (
        _architecture_from_engine_payload(engine_payload, model_type=model)
    )
    expected_scales = V2ObservationScales(
        scenario.world_xy, scenario.world_z_min, scenario.world_z_max, scenario.gamma_max,
    )
    expected_limit = torch.tensor([scenario.delta_gamma_max, scenario.delta_psi_max], dtype=torch.float32)
    action_low = torch.tensor(engine_payload['action_low'], dtype=torch.float32)
    if (scales != expected_scales or checkpoint_radius != radius
            or not torch.equal(action_high, expected_limit)
            or not torch.equal(action_low, -expected_limit)):
        raise ValueError('Checkpoint architecture/radius/action bounds disagree with ScenarioConfig.')
    gamma = float(payload['formal_config']['gamma'])
    if engine_payload['algorithm_config']['gamma'] != gamma:
        raise ValueError('Checkpoint engine gamma differs from formal_config gamma.')
    actor_type = V2ANNPolicyActor if model == 'ann' else V2SNNPolicyActor
    snn_arguments = {'time_window': snn['time_window'], 'tau': snn['tau']} if snn else {}
    actor = actor_type(scales, 2, hidden, action_high, uav_radius=radius,
                       encoder_config=encoder, **snn_arguments)
    actor_state = engine_payload['actor_state_dict']
    V2TD3UpdateEngine._validate_state_dict(actor, actor_state, name='actor_state_dict')
    # strict=True checks keys/shapes, but can overwrite fixed geometry buffers.
    for name, expected in actor.named_buffers():
        if not torch.equal(actor_state[name].detach().cpu(), expected.detach().cpu()):
            raise ValueError(f'Actor fixed buffer {name!r} differs from its declared architecture.')
    actor.load_state_dict(actor_state, strict=True)
    if not torch.equal(actor.action_limit.detach().cpu(), expected_limit):
        raise ValueError('Loaded actor action_limit disagrees with checkpoint action bounds.')
    if not all(bool(torch.isfinite(value).all()) for value in actor.state_dict().values()):
        raise ValueError('Loaded actor contains non-finite weights.')
    actor.to(target_device)
    actor.eval()
    actor.requires_grad_(False)
    if file_sha256(checkpoint_path) != checkpoint_hash or file_sha256(pool_path) != pool_hash:
        raise RuntimeError('Diagnostic inputs changed while being loaded.')
    return {
        'actor': actor, 'checkpoint_payload': payload, 'pool': pool,
        'scenario': scenario, 'rewards': rewards, 'gamma': gamma, 'device': target_device,
        'checkpoint_path': checkpoint_path, 'pool_path': pool_path,
        'checkpoint_hash': checkpoint_hash, 'pool_hash': pool_hash,
    }


def boundary_violations(state, scenario: ScenarioConfig) -> list[dict[str, Any]]:
    result = []
    for index, axis in enumerate(('x', 'y', 'z')):
        value = float(state[index])
        low = -scenario.world_xy if index < 2 else scenario.world_z_min
        high = scenario.world_xy if index < 2 else scenario.world_z_max
        if value < low or value > high:
            result.append({'axis': axis, 'direction': 'negative' if value < low else 'positive',
                           'value': value, 'limit': float(low if value < low else high)})
    return result


def _statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        'scenario_count': len(rows), 'goal_count': sum(row['outcome'] == 'goal' for row in rows),
        'outcome_counts': {name: sum(row['outcome'] == name for row in rows) for name in OUTCOMES},
    }
    result['success_rate'] = result['goal_count'] / len(rows) if rows else None
    for key in ('episode_return', 'discounted_return', 'last_reward', 'episode_length'):
        values = [row[key] for row in rows]
        result[key] = {'sum': math.fsum(values), 'mean': math.fsum(values) / len(rows) if rows else None,
                       'min': min(values) if rows else None, 'max': max(values) if rows else None}
    for key in ('last_reward_components', 'cumulative_reward_components'):
        names = rows[0][key] if rows else ()
        result[key] = {name: {'sum': math.fsum(row[key][name] for row in rows),
                             'mean': math.fsum(row[key][name] for row in rows) / len(rows)}
                       for name in names}
    return result


def _software_versions() -> dict[str, Any]:
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version()}
    for package in ('numpy', 'matplotlib', 'spikingjelly', 'gymnasium'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def run_policy_diagnostic(*, model: str, checkpoint: str | Path, validation_pool: str | Path,
                          output_dir: str | Path, device: str = 'cpu',
                          max_scenes: int | None = None) -> dict[str, Any]:
    """Evaluate full episodes; max_scenes selects only the original pool prefix."""
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Diagnostic output must be a new or empty directory: {output}')
    if max_scenes is not None and (isinstance(max_scenes, bool) or not isinstance(max_scenes, int) or max_scenes <= 0):
        raise ValueError('max_scenes must be a positive integer.')
    inputs = load_diagnostic_inputs(model=model, checkpoint=checkpoint,
                                    validation_pool=validation_pool, device=device)
    actor, payload, pool = inputs['actor'], inputs['checkpoint_payload'], inputs['pool']
    scenario, rewards, gamma = inputs['scenario'], inputs['rewards'], inputs['gamma']
    records = pool.scenarios if max_scenes is None else pool.scenarios[:max_scenes]
    output.mkdir(parents=True, exist_ok=True)
    (output / 'steps').mkdir()
    (output / 'trajectories').mkdir()
    env = V2StaticNoFlyTrajectoryEnv(
        scenario, rewards, uav_collision_radius=pool.uav_collision_radius,
        cache_ellipsoid_segment_clearance=True,
    )
    env.enable_reward_diagnostics()
    manifest = {
        'diagnostic_only': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': {
            'path': str(inputs['checkpoint_path']), 'sha256': inputs['checkpoint_hash'],
            'format': payload['format'], 'format_version': payload['format_version'],
            'status': payload['status'], 'passed_validation': payload['passed_validation'],
            'stage': payload['stage'], 'stage_steps': payload['training_result']['stage_steps'],
            'global_steps_end': payload['training_result']['global_steps_end'],
            'stop_reason': payload['training_result']['stop_reason'],
            'seed_manifest': payload['seed_manifest'],
            'initialization_source': payload['initialization_source'],
            'architecture': payload['engine_checkpoint']['architecture'],
        },
        'pool': {'path': str(inputs['pool_path']), 'sha256': inputs['pool_hash'],
                 'content_digest': pool.content_digest, 'scenario_count': pool.scenario_count,
                 'master_seed': pool.master_seed, 'stage_seed': pool.stage_seed,
                 'checkpoint_provenance': payload['validation_pool']},
        'scenario_config': payload['scenario_config'], 'reward_config': payload['reward_config'],
        'uav_collision_radius': payload['uav_collision_radius'],
        'formal_config': payload['formal_config'], 'bc_schedule': payload['bc_schedule'],
        'runtime': {
            'model': model, 'requested_device': device, 'device': str(inputs['device']),
            'device_name': torch.cuda.get_device_name(inputs['device']) if device == 'cuda' else platform.processor(),
            'actor_eval': True, 'inference_mode': True, 'exploration_noise': 0.0,
            'compile': {'enabled': False, 'backend': None, 'cuda_graphs': False},
            'cache_ellipsoid_segment_clearance': env.cache_ellipsoid_segment_clearance,
            'single_observation_fast_path': False,
            'max_scenes': max_scenes, 'evaluated_scene_count': len(records),
            'full_pool': len(records) == pool.scenario_count,
            'reward_sum_tolerance': {'rtol': 1e-10, 'atol': 1e-8},
            'discount_convention': 'sum(gamma**t * reward[t]), t starts at 0',
            'distance_250_definition': 'minimum point or flight-segment goal distance <= 250 km',
            'software': _software_versions(),
            'torch_num_threads': torch.get_num_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
        },
    }
    _write_json(output / 'manifest.json', manifest)
    rows = []
    with (output / 'episodes.jsonl').open('x', encoding='utf-8') as episodes_stream:
        for record in records:
            observation, initial_info = env.reset(options={'scenario': record['payload']})
            previous_clearance = initial_info['min_zone_clearance']
            states = [env.state.copy().tolist()]
            step_rows, actions = [], []
            episode_return = discounted_return = 0.0
            cumulative = {}
            min_point_distance = min_segment_distance = initial_info['goal_distance']
            guidance_steps = los_steps = radial_steps = 0
            max_error = 0.0
            outcome = 'running'
            while outcome == 'running':
                batch = collate_v2_observations([observation], device=inputs['device'])
                with torch.inference_mode():
                    action = actor(batch)[0].detach().cpu().numpy().astype(np.float32)
                if action.shape != (2,) or not np.all(np.isfinite(action)):
                    raise RuntimeError('Actor returned a non-finite or invalid action.')
                previous_state = env.state.copy()
                observation, reward, terminated, truncated, info = env.step(action)
                parts = info['reward_components']
                error = abs(math.fsum(parts.values()) - reward)
                if not math.isclose(math.fsum(parts.values()), reward, rel_tol=1e-10, abs_tol=1e-8):
                    raise RuntimeError('Step reward and collected components disagree.')
                max_error = max(max_error, error)
                for name, value in parts.items():
                    cumulative[name] = cumulative.get(name, 0.0) + value
                episode_return += reward
                discounted_return += gamma ** len(step_rows) * reward
                min_point_distance = min(min_point_distance, info['goal_distance'])
                min_segment_distance = min(min_segment_distance, info['segment_goal_distance'])
                los_active, radial_active = info['terminal_los_active'], info['terminal_radial_tangential_active']
                los_steps += int(los_active)
                radial_steps += int(radial_active)
                guidance_steps += int(los_active or radial_active)
                segment_clearance = min((zone.segment_clearance(
                    previous_state[:3], env.state[:3], uav_radius=pool.uav_collision_radius,
                ) for zone in env.zones), default=float('inf'))
                step_rows.append({
                    'step': env.steps, 'state_before': previous_state.tolist(),
                    'state_after': env.state.copy().tolist(), 'actor_action': action.tolist(),
                    'action': env.prev_action.copy().tolist(), 'reward': reward,
                    'reward_components': parts, 'reward_component_error': error,
                    'goal_distance_before': float(np.linalg.norm(previous_state[:3] - env.goal)),
                    'goal_distance': info['goal_distance'], 'segment_goal_distance': info['segment_goal_distance'],
                    'min_zone_clearance': _optional_distance(info['min_zone_clearance']),
                    'min_zone_clearance_before': _optional_distance(previous_clearance),
                    'min_segment_zone_clearance': _optional_distance(segment_clearance),
                    'terminal_los_active': los_active, 'terminal_radial_tangential_active': radial_active,
                    'line_to_goal_safe': info['line_to_goal_safe'], 'active_goal_radius': info['active_goal_radius'],
                    'terminated': bool(terminated), 'truncated': bool(truncated), 'outcome': info['outcome'],
                })
                previous_clearance = info['min_zone_clearance']
                states.append(env.state.copy().tolist())
                actions.append(env.prev_action.copy().tolist())
                if terminated or truncated:
                    outcome = info['outcome']
                    if outcome not in OUTCOMES:
                        raise RuntimeError(f'Unexpected terminal outcome: {outcome}')
            stem = f"scene_{record['sequence_index']:04d}"
            steps_path, tail_path = f'steps/{stem}.jsonl', f'steps/{stem}_last50.jsonl'
            for path, items in ((steps_path, step_rows), (tail_path, step_rows[-50:])):
                with (output / path).open('x', encoding='utf-8') as stream:
                    for item in items:
                        stream.write(_json(item) + '\n')
            zone_types = [zone.shape.to_dict()['shape_type'] for zone in env.zones]
            row = {
                'scenario_id': record['scenario_id'], 'sequence_index': record['sequence_index'],
                'scenario_seed': record['scenario_seed'], 'zone_count': len(env.zones), 'zone_types': zone_types,
                'outcome': outcome, 'episode_length': env.steps,
                'start_goal_distance': initial_info['goal_distance'], 'episode_return': episode_return,
                'discounted_return': discounted_return, 'last_reward': step_rows[-1]['reward'],
                'last_reward_components': step_rows[-1]['reward_components'], 'cumulative_reward_components': cumulative,
                'final_position': states[-1][:3], 'final_height': states[-1][2], 'final_goal_distance': info['goal_distance'],
                'min_point_goal_distance': min_point_distance, 'min_segment_goal_distance': min_segment_distance,
                'entered_250_range': min(min_point_distance, min_segment_distance) <= 250.0,
                'terminal_guidance_steps': guidance_steps, 'terminal_los_steps': los_steps,
                'terminal_radial_tangential_steps': radial_steps,
                'boundary_violations': boundary_violations(env.state, scenario) if outcome == 'boundary' else [],
                'max_reward_component_error': max_error, 'steps_path': steps_path, 'tail_50_path': tail_path,
                'trajectory_views': None,
            }
            if outcome != 'goal' or not env.zones:
                exported = export_v2_trajectory_views(output / 'trajectories', stem, {
                    'scenario_payload': record['payload'], 'scenario_config': payload['scenario_config'],
                    'reward_config': payload['reward_config'], 'uav_collision_radius': pool.uav_collision_radius,
                    'trajectory': [state[:3] for state in states], 'states': states, 'actions': actions,
                    'terminal_state': states[-1], 'outcome': outcome, 'episode_return': episode_return,
                    'episode_length': env.steps, 'stage': payload['stage'], 'global_steps': None,
                    'model_type': model, 'source': 'policy_diagnostic', 'diagnostic_only': True,
                    'scenario_id': record['scenario_id'], 'sequence_index': record['sequence_index'],
                    'scenario_seed': record['scenario_seed'],
                    'selection_reasons': ([outcome] if outcome != 'goal' else []) + (['zero_zone'] if not env.zones else []),
                })
                row['trajectory_views'] = {key: Path(path).relative_to(output).as_posix() for key, path in exported.items()}
            episodes_stream.write(_json(row) + '\n')
            episodes_stream.flush()
            rows.append(row)
            print(f"[{len(rows)}/{len(records)}] {record['scenario_id']} {outcome} steps={env.steps} return={episode_return:.6f}", flush=True)
    with (output / 'episodes.csv').open('x', newline='', encoding='utf-8') as stream:
        component_names = rows[0]['cumulative_reward_components']
        fields = list(rows[0]) + [f'{prefix}_{name}' for prefix in ('last', 'cumulative') for name in component_names]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: _json(value) if isinstance(value, (dict, list)) or value is None else value for key, value in row.items()}
            for prefix, key in (('last', 'last_reward_components'), ('cumulative', 'cumulative_reward_components')):
                flat.update({f'{prefix}_{name}': row[key][name] for name in component_names})
            writer.writerow(flat)
    summary = {
        'diagnostic_only': True, 'execution_status': 'completed',
        'checkpoint_status': payload['status'], 'checkpoint_passed_validation': payload['passed_validation'],
        'pool_content_digest': pool.content_digest, 'full_pool': len(records) == pool.scenario_count,
        'overall': _statistics(rows),
        'by_outcome': {name: _statistics([row for row in rows if row['outcome'] == name]) for name in OUTCOMES},
        'by_zone_count': {str(count): _statistics([row for row in rows if row['zone_count'] == count])
                          for count in sorted({row['zone_count'] for row in rows})},
        'max_reward_component_error': max(row['max_reward_component_error'] for row in rows),
    }
    _write_json(output / 'summary.json', summary)
    env.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('ann', 'snn'))
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--validation-pool', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--max-scenes', type=int, default=None, help='Original pool prefix for smoke checks; default: entire pool.')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_policy_diagnostic(**vars(args))
    print(_json({'output_dir': str(args.output_dir.resolve()), 'overall': summary['overall']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
