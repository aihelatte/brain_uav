"""Evaluate a trained checkpoint and export per-episode three-view artifacts."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..curriculum import describe_curriculum_mix, parse_curriculum_mix
from ..scenarios import (
    DEFAULT_BENCHMARK_SUITE_PATH,
    build_benchmark_scenarios,
    load_benchmark_suite,
)
from ..scripts.common import make_actor, make_env
from ..scripts.train_td3 import export_episode_result
from ..utils.io import ensure_dir, load_checkpoint, now_timestamp, save_json
from ..utils.seeding import set_global_seed

AC_ENERGY_PJ = 0.9
MAC_ENERGY_PJ = 4.6


def _sanitize_token(value: str) -> str:
    token = ''.join(ch.lower() if ch.isalnum() else '_' for ch in value.strip())
    while '__' in token:
        token = token.replace('__', '_')
    return token.strip('_') or 'unknown'


def _default_eval_root(
    evaluation_mode: str,
    curriculum_level: str | None,
    model: str,
    eval_name: str | None,
) -> Path:
    timestamp = now_timestamp()
    if eval_name:
        base_name = _sanitize_token(eval_name)
    else:
        mode_tag = curriculum_level if evaluation_mode == 'curriculum' and curriculum_level else evaluation_mode
        base_name = _sanitize_token(f'{mode_tag}_{evaluation_mode}_{model}_{timestamp}')
    return ensure_dir(Path('outputs/evals') / base_name)


def _build_zone_count_summary(records: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for record in records:
        key = str(record['zone_count'])
        bucket = summary.setdefault(key, {'episodes': 0, 'outcomes': {}})
        bucket['episodes'] += 1
        outcome = record['outcome']
        bucket['outcomes'][outcome] = bucket['outcomes'].get(outcome, 0) + 1
    return summary


def _build_outcome_by_zone_count(records: list[dict]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        outcome_bucket = summary.setdefault(record['outcome'], {})
        zone_key = str(record['zone_count'])
        outcome_bucket[zone_key] = outcome_bucket.get(zone_key, 0) + 1
    return summary


def _build_category_summary(records: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for record in records:
        category = record.get('category', 'uncategorized')
        bucket = summary.setdefault(
            category,
            {
                'episodes': 0,
                'goal_count': 0,
                'step_values': [],
                'return_values': [],
                'outcomes': {},
            },
        )
        bucket['episodes'] += 1
        bucket['step_values'].append(record['steps'])
        bucket['return_values'].append(record['return'])
        outcome = record['outcome']
        bucket['outcomes'][outcome] = bucket['outcomes'].get(outcome, 0) + 1
        if outcome == 'goal':
            bucket['goal_count'] += 1
    result: dict[str, dict] = {}
    for category, bucket in summary.items():
        result[category] = {
            'episodes': bucket['episodes'],
            'success_rate': bucket['goal_count'] / bucket['episodes'],
            'collision_rate': bucket['outcomes'].get('collision', 0) / bucket['episodes'],
            'boundary_rate': bucket['outcomes'].get('boundary', 0) / bucket['episodes'],
            'ground_rate': bucket['outcomes'].get('ground', 0) / bucket['episodes'],
            'timeout_rate': bucket['outcomes'].get('timeout', 0) / bucket['episodes'],
            'avg_steps': statistics.mean(bucket['step_values']),
            'avg_return': statistics.mean(bucket['return_values']),
            'outcomes': bucket['outcomes'],
        }
    return result


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return float(values_sorted[int(k)])
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return float(d0 + d1)


def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _thop_macs(model: torch.nn.Module, example_input: torch.Tensor) -> tuple[float | None, float | None, str]:
    try:
        from thop import profile  # type: ignore
    except Exception:
        return None, None, 'thop_unavailable'
    try:
        macs, params = profile(model, inputs=(example_input,), verbose=False)
        return float(macs), float(params), 'thop_profile'
    except Exception:
        return None, None, 'thop_failed'


def _coerce_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(',', '')
        if not text:
            return None
        unit_scale = {
            'k': 1e3,
            'm': 1e6,
            'g': 1e9,
            't': 1e12,
        }
        parts = text.split()
        number_text = parts[0]
        scale = 1.0
        if number_text[-1:].lower() in unit_scale:
            scale = unit_scale[number_text[-1].lower()]
            number_text = number_text[:-1]
        elif len(parts) > 1 and parts[1][:1].lower() in unit_scale:
            scale = unit_scale[parts[1][:1].lower()]
        try:
            return float(number_text) * scale
        except ValueError:
            return None
    return None


def _json_safe(value: object) -> object:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def _format_ops(value: float | None) -> str | None:
    if value is None:
        return None
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f'{value / 1e9:.2f} G Ops'
    if abs_value >= 1e6:
        return f'{value / 1e6:.2f} M Ops'
    if abs_value >= 1e3:
        return f'{value / 1e3:.2f} K Ops'
    return f'{value:.1f} Ops'


def _parse_syops_payload(payload: object) -> tuple[dict[str, float | None], dict[str, object]]:
    base: dict[str, float | None] = {
        'syops_total': None,
        'snn_acs': None,
        'snn_macs': None,
        'syops_energy': None,
    }
    extras: dict[str, object] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            extras[key] = _json_safe(value)
        for key in ('syops_total', 'syops', 'total_syops', 'ops', 'total_ops'):
            if base['syops_total'] is None:
                base['syops_total'] = _coerce_number(payload.get(key))
        for key in ('acs', 'ac', 'snn_acs', 'total_acs'):
            if base['snn_acs'] is None:
                base['snn_acs'] = _coerce_number(payload.get(key))
        for key in ('macs', 'mac', 'snn_macs', 'total_macs'):
            if base['snn_macs'] is None:
                base['snn_macs'] = _coerce_number(payload.get(key))
        for key in ('energy', 'syops_energy', 'total_energy'):
            if base['syops_energy'] is None:
                base['syops_energy'] = _coerce_number(payload.get(key))
        return base, _json_safe(extras)

    try:
        import numpy as np  # type: ignore
        is_array = isinstance(payload, np.ndarray)
    except Exception:
        is_array = False

    if isinstance(payload, (list, tuple)) or is_array:
        extras['raw'] = _json_safe(payload)
        sequence = payload
        if len(sequence) == 2:
            syops_count = sequence[0]
            extras['params'] = sequence[1]
        else:
            syops_count = sequence
        syops_values = _json_safe(syops_count)
        if isinstance(syops_values, list) and len(syops_values) >= 3:
            base['syops_total'] = _coerce_number(syops_values[0])
            base['snn_acs'] = _coerce_number(syops_values[1])
            base['snn_macs'] = _coerce_number(syops_values[2])
            if len(syops_values) > 3:
                extras['spike_rate'] = _json_safe(syops_values[3])
        else:
            base['syops_total'] = _coerce_number(syops_values)
        return base, _json_safe(extras)

    return base, extras


def _syops_profile(model: torch.nn.Module, example_input: torch.Tensor) -> dict[str, object]:
    try:
        from syops import get_model_complexity_info  # type: ignore
    except Exception:
        return {
            'syops_method': 'syops_unavailable',
            'syops_assumptions': 'syops package is not available in the environment.',
        }

    input_shape = tuple(example_input.shape[1:])
    try:
        ops, params = get_model_complexity_info(
            model,
            input_shape,
            [(example_input.detach(), None)],
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print('  SyOps OPs:', ops)
        print('  SyOps Params:', params)
        base, extras = _parse_syops_payload((ops, params))
        formatted_ops: dict[str, object] = {}
        safe_ops = _json_safe(ops)
        if isinstance(safe_ops, list) and len(safe_ops) >= 3:
            formatted_ops = {
                'syops_total_ops': safe_ops[0],
                'snn_ac_ops': safe_ops[1],
                'snn_mac_ops': safe_ops[2],
            }
        return {
            **base,
            **formatted_ops,
            'syops_method': 'syops.get_model_complexity_info',
            'syops_assumptions': 'Values reported directly by syops.get_model_complexity_info.',
            'syops_raw': extras,
        }
    except Exception as exc:
        return {
            'syops_method': 'syops_failed',
            'syops_assumptions': f'syops profiling failed: {type(exc).__name__}.',
        }


def _make_episode_stem(record: dict) -> str:
    parts = [
        f"ep_{record['episode']:03d}",
        _sanitize_token(record['outcome']),
    ]
    if record.get('category'):
        parts.append(_sanitize_token(str(record['category'])))
    if record.get('scenario'):
        parts.append(_sanitize_token(str(record['scenario'])))
    if record.get('scenario_id'):
        parts.append(_sanitize_token(str(record['scenario_id'])))
    parts.append(f"zones{record['zone_count']}")
    return '_'.join(parts)


def _measure_1000s_planning_time(actor: torch.nn.Module, obs_samples: list, device: torch.device) -> float:
    if not obs_samples:
        return 0.0
    steps = 1000
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(steps):
            obs = obs_samples[i % len(obs_samples)]
            obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
            _ = actor(obs_tensor)
    end = time.perf_counter()
    return float(end - start)


def _apply_checkpoint_config(cfg: ExperimentConfig, payload: dict) -> None:
    saved_config = payload.get('config')
    if not isinstance(saved_config, dict):
        return
    for section_name in ('scenario', 'rewards', 'training'):
        section_payload = saved_config.get(section_name)
        section = getattr(cfg, section_name)
        if not isinstance(section_payload, dict):
            continue
        for key, value in section_payload.items():
            if hasattr(section, key):
                setattr(section, key, value)


def evaluate_policy(
    checkpoint: Path,
    model: str,
    episodes: int,
    seed: int,
    evaluation_mode: str,
    curriculum_level: str | None = None,
    curriculum_mix: dict[str, float] | None = None,
    output_dir: Path | None = None,
    eval_name: str | None = None,
    benchmark_suite_path: Path = DEFAULT_BENCHMARK_SUITE_PATH,
) -> dict:
    """Run evaluation and export a three-view result for every episode."""

    cfg = ExperimentConfig()
    checkpoint_payload = load_checkpoint(checkpoint)
    _apply_checkpoint_config(cfg, checkpoint_payload)
    set_global_seed(seed)
    env = make_env(
        cfg,
        seed=seed,
        scenario_suite=None,
        curriculum_level=curriculum_level if evaluation_mode == 'curriculum' else None,
        curriculum_mix=curriculum_mix if evaluation_mode == 'curriculum' else None,
    )
    obs, _ = env.reset(seed=seed)
    actor = make_actor(cfg, model, obs.shape[0], env.action_space.shape[0])
    actor.load_state_dict(checkpoint_payload['state_dict'])
    actor.eval()

    device = next(actor.parameters()).device
    warmup_forward_steps = 10
    with torch.no_grad():
        for _ in range(warmup_forward_steps):
            _ = actor(torch.tensor(obs[None, :], dtype=torch.float32, device=device))

    output_root = ensure_dir(output_dir) if output_dir is not None else _default_eval_root(
        evaluation_mode,
        curriculum_level,
        model,
        eval_name,
    )
    episodes_dir = ensure_dir(output_root / 'episodes')

    config_payload = cfg.to_dict()
    config_payload['evaluation_mode'] = evaluation_mode
    config_payload['curriculum_level'] = curriculum_level
    config_payload['curriculum_mix'] = curriculum_mix

    benchmark_suite = None
    named_scenarios = []
    if evaluation_mode == 'benchmark':
        benchmark_suite = load_benchmark_suite(benchmark_suite_path)
        named_scenarios = build_benchmark_scenarios(benchmark_suite_path)
        if episodes > len(named_scenarios):
            raise ValueError(
                f'--episodes={episodes} exceeds benchmark suite size {len(named_scenarios)} at {benchmark_suite_path}. '
                'Benchmark evaluation does not loop or repeat scenarios.'
            )

    successes = 0
    collisions = 0
    total_steps = 0
    total_return = 0.0
    step_counts: list[int] = []
    return_values: list[float] = []
    episode_times: list[float] = []
    per_inference: list[float] = []
    outcomes: dict[str, int] = {}
    records: list[dict] = []
    obs_samples: list = []

    diag_sample_stride = 10
    diag_sample_limit = 5000
    diag_samples = 0
    sum_spike_rate_l1 = 0.0
    sum_spike_rate_l2 = 0.0
    sum_dense_macs = 0.0

    for ep in range(episodes):
        scenario_id = None
        category = None
        corridor_width = None
        min_clearance_to_boundary = None
        difficulty_score = None
        scenario_label = None
        if evaluation_mode == 'benchmark':
            scenario = named_scenarios[ep]
            obs, _ = env.reset(options={'scenario': scenario.scenario})
            scenario_id = scenario.scenario_id
            category = scenario.category
            scenario_label = scenario.name
            corridor_width = scenario.corridor_width
            min_clearance_to_boundary = scenario.min_clearance_to_boundary
            difficulty_score = scenario.difficulty_score
            scenario_name = scenario.name
        else:
            obs, _ = env.reset(seed=seed + ep)
            if evaluation_mode == 'curriculum':
                scenario_name = env.last_curriculum_level or (curriculum_level or 'curriculum')
                scenario_label = scenario_name
            else:
                scenario_name = f'random_{ep + 1:03d}'
                scenario_label = scenario_name

        done = False
        steps = 0
        episode_return = 0.0
        inference_times: list[float] = []
        info: dict = {}

        while not done:
            if len(obs_samples) < 1000:
                obs_samples.append(obs.copy())
            step_start = time.perf_counter()
            with torch.no_grad():
                obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
                action_tensor = actor(obs_tensor)
            infer_t = time.perf_counter() - step_start
            inference_times.append(infer_t)
            per_inference.append(infer_t)
            action = action_tensor.detach().cpu().numpy()[0]

            if hasattr(actor, 'forward_with_diagnostics') and (total_steps % diag_sample_stride == 0):
                if diag_samples < diag_sample_limit:
                    with torch.no_grad():
                        _, diag = actor.forward_with_diagnostics(obs_tensor)
                    diag_samples += 1
                    sum_spike_rate_l1 += float(diag.get('spike_rate_l1', 0.0))
                    sum_spike_rate_l2 += float(diag.get('spike_rate_l2', 0.0))
                    sum_dense_macs += float(diag.get('dense_macs_estimate', 0.0))

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            steps += 1
            total_steps += 1
            done = terminated or truncated

        episode_time = sum(inference_times)
        total_return += float(episode_return)
        return_values.append(float(episode_return))
        episode_times.append(episode_time)
        step_counts.append(steps)
        outcome = info['outcome']
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == 'goal':
            successes += 1
        if outcome == 'collision':
            collisions += 1

        scenario_payload = env.export_scenario()
        scenario_payload.update(
            {
                'scenario_id': scenario_id,
                'category': category,
                'scenario_label': scenario_label,
                'corridor_width': corridor_width,
                'min_clearance_to_boundary': min_clearance_to_boundary,
                'difficulty_score': difficulty_score,
            }
        )
        zone_count = len(scenario_payload['zones'])
        zone_radii = [float(zone['radius']) for zone in scenario_payload['zones']]
        zone_centers = [list(zone['center_xy']) for zone in scenario_payload['zones']]
        curriculum_value = info.get('curriculum_level')
        episode_record = {
            'episode': ep + 1,
            'total_steps': total_steps,
            'return': float(episode_return),
            'length': steps,
            'outcome': outcome,
            'actor_loss': None,
            'critic_loss': None,
            'scenario': scenario_payload,
            'trajectory': [point.tolist() for point in env.trajectory],
            'final_state': env.state.copy().tolist(),
            'info': {
                'goal_distance': float(info['goal_distance']),
                'progress': float(info.get('progress', 0.0)),
                'steps': steps,
                'curriculum_level': curriculum_value,
                'scenario_name': scenario_name,
                'scenario_id': scenario_id,
                'category': category,
                'scenario_label': scenario_label,
                'evaluation_mode': evaluation_mode,
                'zone_count': zone_count,
                'corridor_width': corridor_width,
                'min_clearance_to_boundary': min_clearance_to_boundary,
                'difficulty_score': difficulty_score,
            },
        }
        artifact_paths = export_episode_result(
            episodes_dir,
            _make_episode_stem(
                {
                    'episode': ep + 1,
                    'outcome': outcome,
                    'scenario': scenario_label,
                    'scenario_id': scenario_id,
                    'category': category,
                    'zone_count': zone_count,
                }
            ),
            episode_record,
            config_payload,
        )

        records.append(
            {
                'episode': ep + 1,
                'scenario_id': scenario_id,
                'category': category,
                'scenario': scenario_name,
                'scenario_label': scenario_label,
                'outcome': outcome,
                'steps': steps,
                'return': float(episode_return),
                'goal_distance': float(info['goal_distance']),
                'avg_inference_time_ms': 1000.0 * statistics.mean(inference_times),
                'max_inference_time_ms': 1000.0 * max(inference_times),
                'curriculum_level': curriculum_value,
                'zone_count': zone_count,
                'zone_radii': zone_radii,
                'zone_centers': zone_centers,
                'corridor_width': corridor_width,
                'min_clearance_to_boundary': min_clearance_to_boundary,
                'difficulty_score': difficulty_score,
                'episode_json': artifact_paths['json'],
                'episode_png': artifact_paths['png'],
            }
        )

    zone_count_summary = _build_zone_count_summary(records)
    outcome_by_zone_count = _build_outcome_by_zone_count(records)
    category_summary = _build_category_summary(records) if evaluation_mode == 'benchmark' else {}

    device_name = str(device)
    param_count, trainable_param_count = _count_params(actor)
    example_input = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
    dense_macs, dense_params, macs_method = _thop_macs(actor, example_input)
    syops_payload: dict[str, object] = {}
    if model == 'snn':
        syops_payload = _syops_profile(actor, example_input)

    avg_spike_rate_l1 = None
    avg_spike_rate_l2 = None
    avg_dense_macs = None
    avg_effective_macs = None
    if diag_samples > 0:
        avg_spike_rate_l1 = sum_spike_rate_l1 / diag_samples
        avg_spike_rate_l2 = sum_spike_rate_l2 / diag_samples
        avg_dense_macs = sum_dense_macs / diag_samples

    dense_theoretical_macs = dense_macs if dense_macs is not None else avg_dense_macs
    dense_theoretical_flops = None
    if dense_theoretical_macs is not None:
        dense_theoretical_flops = float(dense_theoretical_macs) * 2.0

    if model == 'ann':
        effective_macs = dense_theoretical_macs
    else:
        effective_macs = None

    snn_acs = syops_payload.get('snn_acs')
    snn_macs = syops_payload.get('snn_macs')
    syops_total_ops = syops_payload.get('syops_total_ops')
    snn_ac_ops = syops_payload.get('snn_ac_ops')
    snn_mac_ops = syops_payload.get('snn_mac_ops')
    snn_spike_aware_ops = None
    snn_energy_pj = None
    if isinstance(snn_acs, (int, float)) and isinstance(snn_macs, (int, float)):
        snn_spike_aware_ops = float(snn_acs) + float(snn_macs)
        snn_energy_pj = float(snn_acs) * AC_ENERGY_PJ + float(snn_macs) * MAC_ENERGY_PJ

    ann_macs = dense_theoretical_macs if model == 'ann' else None
    ann_energy_pj = None
    if ann_macs is not None:
        ann_energy_pj = float(ann_macs) * MAC_ENERGY_PJ

    macs_assumptions = None
    if model == 'ann' and macs_method != 'thop_profile':
        macs_assumptions = 'dense_theoretical_macs unavailable from thop.'

    decision_dt_s = float(cfg.scenario.dt)
    estimated_steps_for_1000s = int(round(1000.0 / decision_dt_s))
    avg_inference_time_ms = 1000.0 * statistics.mean(per_inference) if per_inference else 0.0
    estimated_1000s_planning_time_s = (avg_inference_time_ms / 1000.0) * estimated_steps_for_1000s

    measured_1000s_planning_time_s = _measure_1000s_planning_time(actor, obs_samples, device)

    p50_inference_time_ms = _percentile([t * 1000.0 for t in per_inference], 50.0)
    p95_inference_time_ms = _percentile([t * 1000.0 for t in per_inference], 95.0)
    p99_inference_time_ms = _percentile([t * 1000.0 for t in per_inference], 99.0)

    efficiency_summary = {
        'device': device_name,
        'torch_version': torch.__version__,
        'model_type': model,
        'param_count': param_count,
        'trainable_param_count': trainable_param_count,
        'dense_theoretical_macs': dense_theoretical_macs,
        'dense_theoretical_flops': dense_theoretical_flops,
        'effective_macs': effective_macs,
        'ann_macs': ann_macs,
        'ann_macs_ops': _format_ops(ann_macs),
        'ann_energy_pj': ann_energy_pj,
        'ann_energy_assumptions': f'ANN energy estimate uses MAC={MAC_ENERGY_PJ} pJ.' if ann_energy_pj is not None else None,
        'ann_macs_method': macs_method if model == 'ann' else None,
        'avg_spike_rate_l1': avg_spike_rate_l1,
        'avg_spike_rate_l2': avg_spike_rate_l2,
        'macs_counting_method': macs_method if model == 'ann' else syops_payload.get('syops_method'),
        'assumptions': macs_assumptions,
        'syops_total': syops_payload.get('syops_total'),
        'syops_total_ops': syops_total_ops,
        'snn_syops': snn_spike_aware_ops,
        'snn_spike_aware_ops': snn_spike_aware_ops,
        'snn_acs': snn_acs,
        'snn_macs': snn_macs,
        'snn_ac_ops': snn_ac_ops,
        'snn_mac_ops': snn_mac_ops,
        'syops_energy': snn_energy_pj,
        'snn_energy_pj': snn_energy_pj,
        'energy_assumptions': f'SNN energy estimate uses AC={AC_ENERGY_PJ} pJ and MAC={MAC_ENERGY_PJ} pJ.' if snn_energy_pj is not None else None,
        'syops_method': syops_payload.get('syops_method'),
        'syops_assumptions': syops_payload.get('syops_assumptions'),
        'syops_raw': syops_payload.get('syops_raw'),
        'dense_macs_method': macs_method,
        'avg_inference_time_ms': avg_inference_time_ms,
        'p50_inference_time_ms': p50_inference_time_ms,
        'p95_inference_time_ms': p95_inference_time_ms,
        'p99_inference_time_ms': p99_inference_time_ms,
        'max_inference_time_ms': 1000.0 * max(per_inference) if per_inference else 0.0,
        'avg_episode_time_s': statistics.mean(episode_times) if episode_times else 0.0,
        'max_episode_time_s': max(episode_times) if episode_times else 0.0,
        'decision_dt_s': decision_dt_s,
        'estimated_steps_for_1000s': estimated_steps_for_1000s,
        'estimated_1000s_planning_time_s': estimated_1000s_planning_time_s,
        'measured_1000s_planning_time_s': measured_1000s_planning_time_s,
    }

    summary = {
        'model': model,
        'episodes': episodes,
        'total_steps': total_steps,
        'success_rate': successes / episodes,
        'collision_rate': outcomes.get('collision', 0) / episodes,
        'boundary_rate': outcomes.get('boundary', 0) / episodes,
        'ground_rate': outcomes.get('ground', 0) / episodes,
        'timeout_rate': outcomes.get('timeout', 0) / episodes,
        'avg_steps': statistics.mean(step_counts),
        'avg_return': statistics.mean(return_values),
        'avg_episode_time_s': statistics.mean(episode_times),
        'avg_inference_time_ms': 1000.0 * statistics.mean(per_inference),
        'p50_inference_time_ms': p50_inference_time_ms,
        'p95_inference_time_ms': p95_inference_time_ms,
        'p99_inference_time_ms': p99_inference_time_ms,
        'max_inference_time_ms': 1000.0 * max(per_inference),
        'outcomes': outcomes,
        'records': records,
        'zone_count_summary': zone_count_summary,
        'outcome_by_zone_count': outcome_by_zone_count,
        'category_summary': category_summary,
        'evaluation_mode': evaluation_mode,
        'curriculum_level': curriculum_level,
        'curriculum_mix': curriculum_mix,
        'benchmark_suite_name': None if benchmark_suite is None else benchmark_suite['suite_name'],
        'benchmark_suite_path': None if benchmark_suite is None else str(Path(benchmark_suite_path)),
        'output_dir': str(output_root),
        'episodes_dir': str(episodes_dir),
        'param_count': param_count,
        'trainable_param_count': trainable_param_count,
        'dense_theoretical_macs': dense_theoretical_macs,
        'dense_theoretical_flops': dense_theoretical_flops,
        'effective_macs': effective_macs,
        'ann_macs': ann_macs,
        'ann_macs_ops': _format_ops(ann_macs),
        'ann_energy_pj': ann_energy_pj,
        'ann_energy_assumptions': f'ANN energy estimate uses MAC={MAC_ENERGY_PJ} pJ.' if ann_energy_pj is not None else None,
        'ann_macs_method': macs_method if model == 'ann' else None,
        'avg_spike_rate_l1': avg_spike_rate_l1,
        'avg_spike_rate_l2': avg_spike_rate_l2,
        'macs_counting_method': macs_method if model == 'ann' else syops_payload.get('syops_method'),
        'assumptions': macs_assumptions,
        'syops_total': syops_payload.get('syops_total'),
        'syops_total_ops': syops_total_ops,
        'snn_syops': snn_spike_aware_ops,
        'snn_spike_aware_ops': snn_spike_aware_ops,
        'snn_acs': snn_acs,
        'snn_macs': snn_macs,
        'snn_ac_ops': snn_ac_ops,
        'snn_mac_ops': snn_mac_ops,
        'syops_energy': snn_energy_pj,
        'snn_energy_pj': snn_energy_pj,
        'energy_assumptions': f'SNN energy estimate uses AC={AC_ENERGY_PJ} pJ and MAC={MAC_ENERGY_PJ} pJ.' if snn_energy_pj is not None else None,
        'syops_method': syops_payload.get('syops_method'),
        'syops_assumptions': syops_payload.get('syops_assumptions'),
        'syops_raw': syops_payload.get('syops_raw'),
        'dense_macs_method': macs_method,
        'max_episode_time_s': max(episode_times) if episode_times else 0.0,
        'decision_dt_s': decision_dt_s,
        'estimated_steps_for_1000s': estimated_steps_for_1000s,
        'estimated_1000s_planning_time_s': estimated_1000s_planning_time_s,
        'measured_1000s_planning_time_s': measured_1000s_planning_time_s,
    }

    efficiency_path = output_root / 'efficiency_summary.json'
    save_json(efficiency_path, efficiency_summary)
    summary['efficiency_summary_path'] = str(efficiency_path)
    save_json(output_root / 'summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a trained policy.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', choices=['snn', 'ann'], required=True)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--evaluation-mode', choices=['benchmark', 'curriculum', 'random'], default='benchmark')
    parser.add_argument('--curriculum-level', choices=['easy', 'easy_two_zone', 'medium', 'hard'], default=None)
    parser.add_argument('--curriculum-mix', type=str, default=None)
    parser.add_argument('--benchmark-suite', type=Path, default=DEFAULT_BENCHMARK_SUITE_PATH)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--eval-name', type=str, default=None)
    args = parser.parse_args()

    curriculum_mix = None
    if args.evaluation_mode == 'benchmark' and args.episodes is None:
        benchmark_suite_payload = load_benchmark_suite(args.benchmark_suite)
        args.episodes = int(benchmark_suite_payload['total_scenarios'])
        print(
            f"[evaluate] benchmark mode with no --episodes provided; defaulting to full suite size {args.episodes}"
        )
    elif args.episodes is None:
        args.episodes = 16
    if args.evaluation_mode == 'curriculum':
        if args.curriculum_level is None:
            raise ValueError('--curriculum-level is required when --evaluation-mode curriculum')
        curriculum_mix = parse_curriculum_mix(args.curriculum_mix, fallback_level=args.curriculum_level)
    results = evaluate_policy(
        args.checkpoint,
        args.model,
        args.episodes,
        args.seed,
        args.evaluation_mode,
        curriculum_level=args.curriculum_level,
        curriculum_mix=curriculum_mix,
        output_dir=args.output,
        eval_name=args.eval_name,
        benchmark_suite_path=args.benchmark_suite,
    )
    if args.evaluation_mode == 'curriculum':
        print(f"Curriculum evaluation: level={args.curriculum_level}, mix={describe_curriculum_mix(curriculum_mix)}")
    if args.evaluation_mode == 'benchmark':
        print(f"Benchmark suite: {results['benchmark_suite_name']} @ {results['benchmark_suite_path']}")
    print(f"Saved evaluation summary to {Path(results['output_dir']) / 'summary.json'}")
    if 'efficiency_summary_path' in results:
        print(f"Saved efficiency summary to {results['efficiency_summary_path']}")
    print(f"Saved per-episode artifacts to {results['episodes_dir']}")
    print(results)


if __name__ == '__main__':
    main()
