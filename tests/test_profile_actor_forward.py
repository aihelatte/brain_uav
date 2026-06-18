"""Tests for the actor-forward microbenchmark script."""

from __future__ import annotations

import unittest

import torch

from brain_uav.scripts.profile_actor_forward import (
    build_parser,
    profile_cuda_graph_forward,
    summarize_times,
)


class TestProfileActorForward(unittest.TestCase):
    def test_parser_accepts_all_modes(self):
        parser = build_parser()
        for mode in ('baseline', 'cuda-graph', 'both'):
            with self.subTest(mode=mode):
                args = parser.parse_args(['--checkpoint', 'model.pt', '--mode', mode])
                self.assertEqual(args.mode, mode)
                self.assertEqual(args.model, 'snn')
                self.assertEqual(args.samples, 1000)
                self.assertEqual(args.warmup, 100)

    def test_summarize_times_percentiles(self):
        summary = summarize_times([1.0, 2.0, 3.0, 4.0], prefix='actor_forward')
        self.assertEqual(summary['avg_actor_forward_time_ms'], 2.5)
        self.assertEqual(summary['p50_actor_forward_time_ms'], 2.5)
        self.assertAlmostEqual(summary['p95_actor_forward_time_ms'], 3.85)
        self.assertEqual(summary['max_actor_forward_time_ms'], 4.0)

    def test_cuda_graph_on_cpu_reports_unavailable(self):
        actor = torch.nn.Linear(3, 2).eval()
        obs_tensors = [torch.zeros((1, 3), dtype=torch.float32)]
        result = profile_cuda_graph_forward(actor, obs_tensors, warmup=1, device=torch.device('cpu'))
        self.assertFalse(result['graph_available'])
        self.assertIn('graph_error', result)
        self.assertIn('CUDA Graph mode requires device=cuda', result['graph_error'])


if __name__ == '__main__':
    unittest.main()
