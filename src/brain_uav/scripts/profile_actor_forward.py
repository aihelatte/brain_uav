"""Microbenchmark actor forward latency for ANN/SNN policies.

This script intentionally does not run environment steps or benchmark scenarios.
It profiles fixed-shape actor inputs only, so it cannot change paper benchmark
success/failure statistics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import ExperimentConfig
from ..scripts.common import (
    DEVICE_CHOICES,
    SNN_BACKEND_CHOICES,
    configure_training_runtime,
    make_actor,
    make_env,
)
from ..utils.io import load_checkpoint, save_json
from ..utils.seeding import set_global_seed


MODEL_CHOICES = ('ann', 'snn')
MODE_CHOICES = ('baseline', 'cuda-graph', 'both')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Profile ANN/SNN actor forward latency.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', choices=MODEL_CHOICES, default='snn')
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto')
    parser.add_argument('--snn-backend', choices=SNN_BACKEND_CHOICES, default='torch')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--mode', choices=MODE_CHOICES, default='both')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=None)
    return parser


def _apply_checkpoint_model_config(cfg: ExperimentConfig, payload: dict[str, Any]) -> None:
    saved_config = payload.get('config')
    if not isinstance(saved_config, dict):
        return
    training_payload = saved_config.get('training')
    if not isinstance(training_payload, dict):
        return
    for key in ('hidden_dim', 'snn_time_window'):
        if key in training_payload and hasattr(cfg.training, key):
            setattr(cfg.training, key, training_payload[key])


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (q / 100.0)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_times(values: list[float], *, prefix: str) -> dict[str, float]:
    return {
        f'avg_{prefix}_time_ms': 0.0 if not values else float(statistics.mean(values)),
        f'p50_{prefix}_time_ms': percentile(values, 50.0),
        f'p95_{prefix}_time_ms': percentile(values, 95.0),
        f'p99_{prefix}_time_ms': percentile(values, 99.0),
        f'max_{prefix}_time_ms': 0.0 if not values else float(max(values)),
    }


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _empty_mode_result(*, samples: int, warmup: int, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        'samples': int(samples),
        'warmup': int(warmup),
        'device': str(device),
    }
    result.update(summarize_times([], prefix='actor_forward'))
    return result


def _action_diff(actions: torch.Tensor | None, baseline_actions: torch.Tensor | None) -> dict[str, float | None]:
    if actions is None or baseline_actions is None:
        return {
            'max_abs_action_diff_vs_baseline': None,
            'mean_abs_action_diff_vs_baseline': None,
        }
    diff = (actions - baseline_actions).abs()
    return {
        'max_abs_action_diff_vs_baseline': float(diff.max().detach().cpu()),
        'mean_abs_action_diff_vs_baseline': float(diff.mean().detach().cpu()),
    }


def profile_baseline_forward(
    actor: torch.nn.Module,
    obs_tensors: list[torch.Tensor],
    *,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    times_ms: list[float] = []
    actions: list[torch.Tensor] = []

    with torch.inference_mode():
        for idx in range(warmup):
            _ = actor(obs_tensors[idx % len(obs_tensors)])
        _sync_if_cuda(device)

        for obs_tensor in obs_tensors:
            _sync_if_cuda(device)
            start = time.perf_counter()
            action = actor(obs_tensor)
            _sync_if_cuda(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)
            actions.append(action.detach().clone())

    result = _empty_mode_result(samples=len(obs_tensors), warmup=warmup, device=device)
    result.update(summarize_times(times_ms, prefix='actor_forward'))
    result['graph_available'] = None
    result['graph_error'] = None
    result['_actions'] = torch.cat(actions, dim=0) if actions else None
    return result


def profile_cuda_graph_forward(
    actor: torch.nn.Module,
    obs_tensors: list[torch.Tensor],
    *,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    result = _empty_mode_result(samples=len(obs_tensors), warmup=warmup, device=device)
    result.update(
        {
            'avg_input_copy_time_ms': 0.0,
            'graph_available': False,
            'graph_error': None,
            '_actions': None,
        }
    )
    if device.type != 'cuda':
        result['graph_error'] = 'CUDA Graph mode requires device=cuda.'
        return result

    replay_times_ms: list[float] = []
    input_copy_times_ms: list[float] = []
    actions: list[torch.Tensor] = []

    try:
        static_input = torch.empty_like(obs_tensors[0], device=device)
        with torch.inference_mode():
            for idx in range(warmup):
                static_input.copy_(obs_tensors[idx % len(obs_tensors)])
                _ = actor(static_input)
            torch.cuda.synchronize(device)

            graph = torch.cuda.CUDAGraph()
            static_input.copy_(obs_tensors[0])
            with torch.cuda.graph(graph):
                static_output = actor(static_input)

            for obs_tensor in obs_tensors:
                torch.cuda.synchronize(device)
                copy_start = time.perf_counter()
                static_input.copy_(obs_tensor)
                torch.cuda.synchronize(device)
                input_copy_times_ms.append((time.perf_counter() - copy_start) * 1000.0)

                torch.cuda.synchronize(device)
                replay_start = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize(device)
                replay_times_ms.append((time.perf_counter() - replay_start) * 1000.0)
                actions.append(static_output.detach().clone())
    except Exception as exc:  # pragma: no cover - CUDA-specific failure path.
        result['graph_error'] = f'{type(exc).__name__}: {exc}'
        return result

    result.update(summarize_times(replay_times_ms, prefix='actor_forward'))
    result['avg_input_copy_time_ms'] = 0.0 if not input_copy_times_ms else float(statistics.mean(input_copy_times_ms))
    result['graph_available'] = True
    result['graph_error'] = None
    result['_actions'] = torch.cat(actions, dim=0) if actions else None
    return result


def _load_actor_and_obs(args: argparse.Namespace) -> tuple[torch.nn.Module, list[torch.Tensor], torch.device]:
    if args.samples <= 0:
        raise ValueError('--samples must be positive.')
    if args.warmup < 0:
        raise ValueError('--warmup must be non-negative.')

    set_global_seed(args.seed)
    cfg = ExperimentConfig()
    checkpoint_payload = load_checkpoint(args.checkpoint)
    if not isinstance(checkpoint_payload, dict):
        raise TypeError('Checkpoint payload must be a dict.')
    _apply_checkpoint_model_config(cfg, checkpoint_payload)
    resolved_device = configure_training_runtime(
        cfg,
        model_type=args.model,
        device=args.device,
        snn_backend=args.snn_backend,
    )
    device = torch.device(resolved_device)

    env = make_env(cfg, seed=args.seed, scenario_suite=None, goal_radius_curriculum_enabled=True)
    base_obs, _ = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = make_actor(cfg, args.model, obs_dim, action_dim)
    state_dict = checkpoint_payload.get('state_dict')
    if state_dict is None:
        raise KeyError('Checkpoint payload must contain a "state_dict" key.')
    actor.load_state_dict(state_dict)
    actor.to(device)
    actor.eval()

    rng = np.random.default_rng(args.seed)
    obs_tensors: list[torch.Tensor] = []
    base_obs = np.asarray(base_obs, dtype=np.float32)
    for _ in range(args.samples):
        perturb = rng.normal(loc=0.0, scale=1e-3, size=base_obs.shape).astype(np.float32)
        obs = (base_obs + perturb).astype(np.float32)
        obs_tensors.append(torch.tensor(obs[None, :], dtype=torch.float32, device=device))
    return actor, obs_tensors, device


def _public_mode_result(result: dict[str, Any], *, model: str, snn_backend: str) -> dict[str, Any]:
    public = {key: value for key, value in result.items() if key != '_actions'}
    public['model'] = model
    public['snn_backend'] = snn_backend
    return public


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    actor, obs_tensors, device = _load_actor_and_obs(args)
    payload: dict[str, Any] = {
        'checkpoint': str(args.checkpoint),
        'mode': args.mode,
        'model': args.model,
        'snn_backend': args.snn_backend,
        'device': str(device),
        'samples': int(args.samples),
        'warmup': int(args.warmup),
        'results': {},
    }

    baseline_result: dict[str, Any] | None = None
    baseline_actions: torch.Tensor | None = None
    if args.mode in {'baseline', 'both'}:
        baseline_result = profile_baseline_forward(actor, obs_tensors, warmup=args.warmup, device=device)
        baseline_actions = baseline_result.get('_actions')
        baseline_result.update(_action_diff(baseline_actions, baseline_actions))
        payload['results']['baseline'] = _public_mode_result(
            baseline_result,
            model=args.model,
            snn_backend=args.snn_backend,
        )

    if args.mode in {'cuda-graph', 'both'}:
        if baseline_actions is None and device.type == 'cuda':
            baseline_reference = profile_baseline_forward(actor, obs_tensors, warmup=args.warmup, device=device)
            baseline_actions = baseline_reference.get('_actions')
        cuda_graph_result = profile_cuda_graph_forward(actor, obs_tensors, warmup=args.warmup, device=device)
        cuda_graph_actions = cuda_graph_result.get('_actions')
        cuda_graph_result.update(_action_diff(cuda_graph_actions, baseline_actions))
        payload['results']['cuda_graph'] = _public_mode_result(
            cuda_graph_result,
            model=args.model,
            snn_backend=args.snn_backend,
        )

    return payload


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_profile(args)
    if args.output is not None:
        save_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
