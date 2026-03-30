from __future__ import annotations

import argparse
from pathlib import Path

import torch
from thop import profile

from ..config import ExperimentConfig
from ..scripts.common import make_actor, make_env
from ..utils.io import load_checkpoint, save_json


def describe_ann(checkpoint: Path) -> tuple[float, float]:
    cfg = ExperimentConfig()
    env = make_env(cfg, seed=cfg.training.seed)
    obs, _ = env.reset(seed=cfg.training.seed)
    actor = make_actor(cfg, 'ann', obs.shape[0], env.action_space.shape[0])
    actor.load_state_dict(load_checkpoint(checkpoint)['state_dict'])
    dummy = torch.randn(1, obs.shape[0])
    macs, params = profile(actor, inputs=(dummy,), verbose=False)
    return float(macs), float(params)


def describe_snn(checkpoint: Path, sample_count: int = 32) -> tuple[float, float, float, float, float, str]:
    cfg = ExperimentConfig()
    env = make_env(cfg, seed=cfg.training.seed)
    obs, _ = env.reset(seed=cfg.training.seed)
    actor = make_actor(cfg, 'snn', obs.shape[0], env.action_space.shape[0])
    actor.load_state_dict(load_checkpoint(checkpoint)['state_dict'])
    dummy = torch.randn(1, obs.shape[0])
    dense_macs, params = profile(actor, inputs=(dummy,), verbose=False)
    effective_macs = []
    spike_rate_l1 = []
    spike_rate_l2 = []
    backend = 'unknown'
    for idx in range(sample_count):
        obs, _ = env.reset(seed=cfg.training.seed + idx)
        with torch.no_grad():
            _, diag = actor.forward_with_diagnostics(torch.tensor(obs[None, :], dtype=torch.float32))
        backend = diag['backend']
        effective_macs.append(diag['effective_macs_estimate'])
        spike_rate_l1.append(diag['spike_rate_l1'])
        spike_rate_l2.append(diag['spike_rate_l2'])
    return float(dense_macs), float(params), float(sum(effective_macs) / len(effective_macs)), float(sum(spike_rate_l1) / len(spike_rate_l1)), float(sum(spike_rate_l2) / len(spike_rate_l2)), backend


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare FLOPs/MACs of ANN and SNN actors.')
    parser.add_argument('--snn', type=Path, required=True)
    parser.add_argument('--ann', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    snn_dense_macs, snn_params, snn_effective_macs, spike_rate_l1, spike_rate_l2, backend = describe_snn(args.snn)
    ann_macs, ann_params = describe_ann(args.ann)
    results = {
        'snn_backend': backend,
        'snn_dense_macs': snn_dense_macs,
        'snn_effective_macs': snn_effective_macs,
        'ann_macs': ann_macs,
        'snn_params': snn_params,
        'ann_params': ann_params,
        'spike_rate_l1': spike_rate_l1,
        'spike_rate_l2': spike_rate_l2,
        'effective_mac_reduction_ratio': 1.0 - (snn_effective_macs / ann_macs if ann_macs else 0.0),
    }
    if args.output:
        save_json(args.output, results)
    print(results)


if __name__ == '__main__':
    main()
