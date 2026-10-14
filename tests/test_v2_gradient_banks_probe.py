"""CPU/mocked contracts only; CUDA graph addresses are NOT validated here."""
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock
import json

import torch
from brain_uav.scripts import probe_v2_gradient_banks as probe


class TestGradientBanksProbe(TestCase):
    def test_nonfinite_clip_preserves_values_and_loss_priority(self):
        for loss, expected in ((1., 2), (float('nan'), 1)):
            p = torch.nn.Parameter(torch.ones(3))
            p.grad = torch.tensor([float('inf'), -float('inf'), 4.])
            before = p.grad.clone()
            error = probe.preserving_clip([p], torch.tensor(loss), torch.tensor(0), 1, 2)
            self.assertEqual(int(error), expected)
            self.assertTrue(torch.equal(p.grad, before))
        p.grad = torch.tensor([3., 4., 0.])
        error = probe.preserving_clip([p], torch.tensor(1.), torch.tensor(0), 1, 2)
        self.assertEqual(int(error), 0)
        torch.testing.assert_close(p.grad, torch.tensor([3., 4., 0.])/(5.+1e-6))

    def test_publication_keeps_old_on_loss_failure_and_publishes_inf(self):
        state = probe.ProbeState('ann', 'cpu')
        executor = probe.BankProbe.__new__(probe.BankProbe)
        executor.s = state
        executor.published = [None, None]
        executor.banks = [[[torch.zeros_like(p) for p in params] for _ in range(2)] for params in state.params]
        executor.graphs = {}
        errors = [0, 1, 2, 3, 4, 0]
        calls = []
        def capture(key, x):
            output = torch.zeros(3, dtype=torch.long)
            def replay():
                error = errors.pop(0)
                output.copy_(torch.tensor([error, int(error != 1), int(error in (0, 4))]))
                for group, bank in enumerate(key[-2:]):
                    for g in executor.banks[group][bank]:
                        g.fill_(float('inf') if error in (2, 4) else len(calls)+1.)
                calls.append(key)
            executor.graphs[key] = SimpleNamespace(replay=replay), x, x.new_zeros(4), output
        executor.capture = capture
        x = torch.zeros(4, 4)
        executor.step(x, False, 'actor', [0]*4)
        old = [p.grad for params in state.params for p in params]
        old_values = [None if g is None else g.clone() for g in old]
        executor.step(x, False, 'actor', [1, 0, 0, 0])
        for p, ref, value in zip([p for params in state.params for p in params], old, old_values):
            self.assertIs(p.grad, ref)
            if ref is not None:
                torch.testing.assert_close(ref, value, rtol=0, atol=0)
        executor.step(x, False, 'actor', [0, 1, 0, 0])
        self.assertTrue(torch.isinf(state.params[0][1].grad).all())
        self.assertIs(state.params[1][1].grad, old[len(state.params[0])+1])
        actor_old = state.params[1][1].grad
        executor.step(x, False, 'actor', [0, 0, 1, 0])
        self.assertIs(state.params[1][1].grad, actor_old)
        executor.step(x, False, 'actor', [0, 0, 0, 1])
        self.assertTrue(torch.isinf(state.params[1][1].grad).all())
        executor.step(x, True, 'actor', [0]*4)
        self.assertIsNotNone(state.params[0][0].grad)
        self.assertIsNotNone(state.params[1][0].grad)
        self.assertEqual(state.counts, dict(updates=2, critic_targets=4, actors=2))

    def test_capture_restores_gradients_none_persistent_and_rng(self):
        from contextlib import ExitStack
        state = probe.ProbeState('ann', 'cpu')
        executor = probe.BankProbe.__new__(probe.BankProbe)
        executor.s, executor.max_graphs = state, 16
        executor.banks = [[[torch.ones_like(p)*7 for p in params] for _ in range(2)] for params in state.params]
        executor.graphs, executor.capture_evidence, executor.capture_seconds = {}, [], 0.
        executor.stream = SimpleNamespace(wait_stream=lambda s: None)
        state.params[0][1].grad = executor.banks[0][0][1]
        state.params[1][1].grad = torch.ones_like(state.params[1][1])*9
        refs = [p.grad for params in state.params for p in params]
        before = [probe.numeric_snapshot(t) for t in state.persistent()]
        rng = torch.get_rng_state()
        cuda_rng = torch.tensor([1], dtype=torch.uint8)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(torch.cuda, 'current_stream', return_value=executor.stream))
            stack.enter_context(mock.patch.object(torch.cuda, 'stream', return_value=nullcontext()))
            stack.enter_context(mock.patch.object(torch.cuda, 'graph', return_value=nullcontext()))
            stack.enter_context(mock.patch.object(torch.cuda, 'CUDAGraph', return_value=object()))
            stack.enter_context(mock.patch.object(torch.cuda, 'synchronize'))
            stack.enter_context(mock.patch.object(torch.cuda, 'get_rng_state', return_value=cuda_rng))
            stack.enter_context(mock.patch.object(torch.cuda, 'set_rng_state'))
            stack.enter_context(mock.patch.object(probe.GatedAdam, 'step'))
            executor.capture((4, False, 'actor', 1, 1), torch.ones(4, 4))
            executor.progress = dict(model='ann', index=1)
            with mock.patch.object(executor, 'operation', side_effect=AssertionError()), \
                 self.assertRaises(AssertionError):
                executor.capture((2, True, 'actor', 0, 0), torch.ones(2, 4))
            self.assertEqual(executor.progress['stage'], 'capture_warmup')
            self.assertFalse(executor.capture_evidence[-1]['completed'])
            self.assertNotIn((2, True, 'actor', 0, 0), executor.graphs)
        probe.assert_state_values(state.persistent(), before)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for p, ref in zip([p for params in state.params for p in params], refs):
            self.assertIs(p.grad, ref)
        torch.testing.assert_close(state.params[0][1].grad, torch.full_like(state.params[0][1], 7.))
        torch.testing.assert_close(state.params[1][1].grad, torch.full_like(state.params[1][1], 9.))
        self.assertTrue(executor.capture_evidence[0]['restored'])

    def test_cache_bound_rejects_before_capture_work(self):
        executor = probe.BankProbe.__new__(probe.BankProbe)
        executor.max_graphs = 1
        executor.graphs = {'existing': object()}
        with self.assertRaisesRegex(RuntimeError, 'cache exhausted'):
            executor.capture((4, False, 'actor', 0, 0), None)

    def test_capability_failure_is_nonzero_with_traceback_and_no_passes(self):
        with TemporaryDirectory() as directory, mock.patch.object(probe, 'require_capability', side_effect=AssertionError()):
            self.assertEqual(probe.main(['--output-dir', directory]), 1)
            report = json.loads((Path(directory)/'summary.json').read_text())
            self.assertFalse(report['passed'])
            self.assertFalse(report['completed'])
            self.assertEqual(report['models'], {})
            self.assertEqual(report['location']['stage'], 'capability')
            self.assertIn('AssertionError', report['failure']['traceback'])
