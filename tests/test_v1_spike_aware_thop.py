"""Small, hand-countable checks for V1 THOP synaptic accounting."""

import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.config import ScenarioConfig
from brain_uav.models.ann import ANNPolicyActor
from brain_uav.models.snn import SNNPolicyActor
from brain_uav.utils.v1_spike_aware_thop import profile_spike_aware_thop


class TestSpikeAwareThop(unittest.TestCase):
    def make_actor(self, snn=False):
        args = dict(state_dim=24, action_dim=2, hidden_dim=2,
                    action_limit=torch.ones(2), scenario=ScenarioConfig())
        if snn:
            return SNNPolicyActor(**args, time_window=4, backend='torch').eval()
        return ANNPolicyActor(**args).eval()

    def test_ann_dense_even_for_zero_input_and_isolated(self):
        actor = self.make_actor()
        obs = np.zeros(24, dtype=np.float32)
        before = {k: v.clone() for k, v in actor.state_dict().items()}
        expected = actor(torch.from_numpy(obs[None])).detach().clone()
        rng = torch.get_rng_state().clone()
        report = profile_spike_aware_thop(actor, [obs, obs])
        self.assertEqual(report['sample_count'], 2)
        self.assertEqual(report['mean_macs'], 24*2 + 2*2 + 2*2)
        self.assertEqual(report['mean_acs'], 0)
        self.assertEqual(report['mean_bias_additions'], 6)
        self.assertEqual(report['parameter_count'], 62)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before.keys(), actor.state_dict().keys())
        for k, v in before.items():
            self.assertTrue(torch.equal(v, actor.state_dict()[k]))
        self.assertTrue(torch.equal(expected, actor(torch.from_numpy(obs[None]))))
        self.assertFalse(actor.training)
        self.assertFalse(any(m._forward_hooks for m in actor.modules()))

    def test_snn_counts_time_steps_and_never_reclassifies_readout(self):
        actor = self.make_actor(snn=True)
        with torch.no_grad():
            actor.fc1.weight.zero_()
            actor.fc1.bias.fill_(4)  # LIF1 spikes on every time step.
        obs = np.zeros(24, dtype=np.float32)
        expected = actor(torch.from_numpy(obs[None])).detach().clone()
        report = profile_spike_aware_thop(actor, [obs])
        self.assertEqual(report['mean_macs'], 48 + 4)
        self.assertEqual(report['mean_acs'], 4*2*2)
        self.assertEqual(report['mean_bias_additions'], 2 + 4*2 + 2)
        self.assertEqual(report['layers']['fc3']['mean_macs'], 4)
        self.assertTrue(torch.equal(expected, actor(torch.from_numpy(obs[None]))))
        with torch.no_grad():
            actor.fc1.bias.zero_()
        quiet = profile_spike_aware_thop(actor, [obs])
        self.assertEqual(quiet['mean_acs'], 0)
        self.assertEqual(quiet['mean_macs'], 52)
        self.assertFalse(any(m._forward_hooks for m in actor.modules()))

    def test_rejects_invalid_inputs_fallback_and_missing_thop(self):
        actor = self.make_actor()
        for samples in ([], [np.zeros(23)], [np.full(24, np.nan)]):
            with self.assertRaises(ValueError):
                profile_spike_aware_thop(actor, samples)
        with mock.patch.dict('sys.modules', {'thop': None}):
            with self.assertRaises(ImportError):
                profile_spike_aware_thop(actor, [np.zeros(24)])
        with mock.patch('brain_uav.models.snn.HAS_SPIKINGJELLY', False):
            fallback = self.make_actor(snn=True)
        with self.assertRaises(ValueError):
            profile_spike_aware_thop(fallback, [np.zeros(24)])

    def test_nonbinary_spikes_fail_without_polluting_original_actor(self):
        actor = self.make_actor(snn=True)
        before = {k: v.clone() for k, v in actor.state_dict().items()}
        handle = actor.lif1.register_forward_hook(lambda m, x, y: y + 0.25)
        try:
            with self.assertRaisesRegex(ValueError, 'not binary'):
                profile_spike_aware_thop(actor, [np.zeros(24)])
        finally:
            handle.remove()
        self.assertEqual(before.keys(), actor.state_dict().keys())
        for k, v in before.items():
            self.assertTrue(torch.equal(v, actor.state_dict()[k]))
        self.assertFalse(any(m._forward_hooks for m in actor.modules()))

    def test_original_v1_dimensions_have_hand_computed_synaptic_counts(self):
        args = dict(state_dim=24, action_dim=2, hidden_dim=128,
                    action_limit=torch.ones(2), scenario=ScenarioConfig())
        for actor, expected_macs in (
            (ANNPolicyActor(**args), 19712),
            (SNNPolicyActor(**args, time_window=4, backend='torch'), 3328),
        ):
            report = profile_spike_aware_thop(actor, [np.zeros(24)])
            self.assertEqual(report['mean_macs'], expected_macs)
            self.assertEqual(report['parameter_count'], 19970)


if __name__ == '__main__':
    unittest.main()
