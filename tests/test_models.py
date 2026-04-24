"""Tests for ANN scaling and SNN forward paths."""

import unittest

import torch

from brain_uav.config import ScenarioConfig
from brain_uav.models import ANNPolicyActor, FixedObsScaler, SNNPolicyActor
from brain_uav.models.snn import HAS_SPIKINGJELLY, validate_snn_backend
from brain_uav.scripts.common import build_log_prefix


class TestModelUtilities(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = ScenarioConfig()
        self.state_dim = 5 + 3 + 4 + 4 * self.scenario.nearest_zone_count
        self.obs = torch.randn(4, self.state_dim)
        self.action_limit = torch.tensor(
            [self.scenario.delta_gamma_max, self.scenario.delta_psi_max], dtype=torch.float32
        )

    def test_fixed_obs_scaler_preserves_shape(self):
        scaler = FixedObsScaler(self.scenario, self.state_dim)
        scaled = scaler(self.obs)
        self.assertEqual(scaled.shape, self.obs.shape)
        self.assertTrue(torch.isfinite(scaled).all())

    def test_snn_forward_does_not_depend_on_diagnostics_path(self):
        actor = SNNPolicyActor(self.state_dim, 2, 32, 4, self.action_limit)
        actor.forward_with_diagnostics = lambda _: (_ for _ in ()).throw(AssertionError('unexpected diagnostics call'))
        action = actor(self.obs)
        self.assertEqual(action.shape, (4, 2))

    def test_snn_forward_with_diagnostics_smoke(self):
        actor = SNNPolicyActor(self.state_dim, 2, 32, 4, self.action_limit)
        action = actor(self.obs)
        action_diag, diagnostics = actor.forward_with_diagnostics(self.obs)
        self.assertEqual(action.shape, (4, 2))
        self.assertEqual(action_diag.shape, (4, 2))
        self.assertIn('spike_rate_l1', diagnostics)
        self.assertIn('spike_rate_l2', diagnostics)
        if HAS_SPIKINGJELLY:
            self.assertEqual(actor.lif1.step_mode, 'm')
            self.assertEqual(actor.lif2.step_mode, 'm')

    def test_ann_actor_uses_fixed_obs_scaler(self):
        actor = ANNPolicyActor(self.state_dim, 2, 32, self.action_limit, self.scenario)
        action = actor(self.obs)
        self.assertEqual(action.shape, (4, 2))

    def test_snn_backend_validation_for_torch(self):
        self.assertEqual(validate_snn_backend('torch'), 'torch')

    def test_log_prefix_format(self):
        self.assertEqual(build_log_prefix('ann', 'easy'), '[ANN easy]')
        self.assertEqual(build_log_prefix('snn', 'bc'), '[SNN bc]')


if __name__ == '__main__':
    unittest.main()
