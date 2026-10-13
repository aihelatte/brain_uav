"""CPU contracts only: these tests do NOT validate CUDA capture or fused Adam."""
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch
from brain_uav.scripts import probe_v2_gpu_td3 as probe
from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer


class TestGPUTD3CapabilityProbe(unittest.TestCase):
    def test_update_warmup_wait_follows_reset_current_stream_work(self):
        events = []
        stream = SimpleNamespace(wait_stream=lambda current: events.append('wait'))
        def reset(*args, **kwargs):
            events.extend(['reset_sample_clones', 'reset_input_restore'])
        def operation(mode):
            events.append('warmup_operation')
            raise AssertionError()  # stop before any actual CUDA work
        fake = SimpleNamespace(persistent=lambda: [], operation=operation)
        def stream_context(value):
            events.append('enter_update_stream')
            return nullcontext()
        progress, evidence = {}, {}
        with mock.patch.object(probe, 'UpdateProbe', return_value=fake), \
             mock.patch.object(probe, 'probe_snn_reset', side_effect=reset), \
             mock.patch.object(torch.cuda, 'Stream', return_value=stream), \
             mock.patch.object(torch.cuda, 'current_stream', return_value=object()), \
             mock.patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([0], dtype=torch.uint8)), \
             mock.patch.object(torch.cuda, 'stream', side_effect=stream_context):
            with self.assertRaises(AssertionError):
                probe.probe_update('snn', progress=progress, evidence=evidence)
        self.assertEqual(events, ['wait', 'reset_sample_clones', 'reset_input_restore',
                                  'wait', 'enter_update_stream', 'warmup_operation'])
        self.assertEqual(progress['stage'], 'update_warmup')
        self.assertEqual(progress['model'], 'snn')
        self.assertEqual(progress['mode'], 'critic_only')
        self.assertFalse(evidence['passed'])
        self.assertEqual(evidence['completed_captures'], [])

    def test_empty_assertion_reports_location_traceback_and_partial_evidence(self):
        def adam(*, progress, evidence):
            evidence.update(passed=True, skipped_nonfinite=['Inf', 'NaN'])
        def update(model, *, progress, evidence):
            if model == 'ann':
                evidence.update(passed=True, continuous_replays=[dict(index=0)])
                return
            evidence.update(passed=False, continuous_replays=[dict(index=0)], error_barriers=[])
            probe._location(progress, 'update_error_barrier', model=model,
                            mode='actor', injection=2)
            raise AssertionError()
        with tempfile.TemporaryDirectory() as root, \
             mock.patch.object(probe, 'require_capability'), \
             mock.patch.object(torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(torch.cuda, 'get_device_name', return_value='test-only'), \
             mock.patch.object(probe, 'probe_adam_gate', side_effect=adam), \
             mock.patch.object(probe, 'probe_update', side_effect=update):
            self.assertEqual(probe.main(['--output-dir', root]), 1)
            result = json.loads((Path(root) / 'summary.json').read_text())
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'], 'capability_probe_failed')
        self.assertEqual(result['failure_context'], dict(stage='update_error_barrier',
                         model='snn', mode='actor', replay_index=None, injection=2))
        self.assertIn('AssertionError', result['traceback'])
        self.assertIn('raise AssertionError()', result['traceback'])
        self.assertTrue(result['adam_gate']['passed'])
        self.assertTrue(result['updates']['ann']['passed'])
        self.assertFalse(result['updates']['snn']['passed'])
        self.assertEqual(result['updates']['snn']['continuous_replays'], [dict(index=0)])
        self.assertEqual(result['updates']['snn']['error_barriers'], [])

    def test_unavailable_cuda_is_a_failed_capability_gate(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(torch.cuda, 'is_available', return_value=False):
            self.assertEqual(probe.main(['--output-dir', root]), 1)
            result = json.loads((Path(root) / 'summary.json').read_text())
            self.assertFalse(result['passed'])
            self.assertFalse(result['gpu_td3_update_implemented'])
            self.assertEqual(result['status'], 'waiting_cuda_capability_validation')
            with self.assertRaises(ValueError):
                probe.main(['--output-dir', root])

    def test_device_error_priority_and_target_commit_selection(self):
        error = torch.tensor(0)
        error = probe.first_error(error, torch.tensor(True), 2)
        error = probe.first_error(error, torch.tensor(True), 3)
        self.assertEqual(int(error), 2)
        target = torch.tensor([1., 2.])
        initial = target.clone()
        probe.gated_targets([torch.tensor([float('nan'), float('inf')])], [target], error)
        torch.testing.assert_close(target, initial, rtol=0, atol=0)
        online = torch.tensor([3., 4.])
        probe.gated_targets([online], [target], torch.tensor(0))
        expected = initial.mul(.995).add_(online, alpha=.005)
        torch.testing.assert_close(target, expected, rtol=0, atol=0)

    def test_replay_payload_matches_actual_current_arrays(self):
        replay = V2ReplayBuffer(20, 2, 6, success_replay_fraction=.25)
        budget = probe.replay_budget(20, 6, .25)
        # Verify arithmetic against existing layout rather than a diagnostic capacity claim.
        primary = sum(getattr(replay, name).nbytes for name in (
            'ego_features', 'goal_features', 'zone_features', 'zone_count',
            'next_ego_features', 'next_goal_features', 'next_zone_features', 'next_zone_count',
            'action', 'reward', 'done', 'success', 'failure', 'near_goal', 'line_to_goal_safe',
            'sample_weight', 'write_id')) + replay._sampling_weight_tree._tree.nbytes
        success = sum(getattr(replay, name).nbytes for name in (
            'success_ego_features', 'success_goal_features', 'success_zone_features', 'success_zone_count',
            'success_next_ego_features', 'success_next_goal_features', 'success_next_zone_features',
            'success_next_zone_count', 'success_action', 'success_reward', 'success_done',
            'success_near_goal', 'success_line_to_goal_safe'))
        self.assertEqual(budget['primary_bytes'], primary)
        self.assertEqual(budget['success_bytes'], success)
        self.assertFalse(budget['fit_validated'])

    def test_gate_api_does_not_insert_missing_gradients(self):
        # Only inspect submitted lists on CPU, never pretend the mocked kernel executes.
        p = torch.nn.Parameter(torch.tensor([1.]))
        missing = torch.nn.Parameter(torch.tensor([2.]))
        p.grad = torch.ones_like(p)
        optimizer = probe.GatedAdam([p, missing], .001)
        with mock.patch.object(torch, '_fused_adam_') as kernel:
            optimizer.step(torch.tensor(0.))
        self.assertEqual(len(kernel.call_args.args[0]), 1)
        self.assertIs(kernel.call_args.args[0][0], p)
        self.assertIsNone(missing.grad)
        self.assertEqual(float(optimizer.steps[1]), 0.)

    def test_clip_arithmetic_matches_existing_clip_and_retains_error(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        reference = torch.nn.Parameter(p.detach().clone())
        p.grad = torch.tensor([3., 4.])
        reference.grad = p.grad.clone()
        error = probe.checked_clip([p], torch.tensor(2.), torch.tensor(0), 1, 2)
        torch.nn.utils.clip_grad_norm_([reference], 1., error_if_nonfinite=True)
        self.assertEqual(int(error), 0)
        torch.testing.assert_close(p.grad, reference.grad, rtol=0, atol=0)
        p.grad = torch.tensor([float('inf'), 4.])
        error = probe.checked_clip([p], torch.tensor(float('nan')), torch.tensor(0), 1, 2)
        self.assertEqual(int(error), 1)  # first loss failure wins over gradient failure


if __name__ == '__main__':
    unittest.main()
