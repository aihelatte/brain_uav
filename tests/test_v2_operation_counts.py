"""Hand-calculated MAC/AC accounting on real V2 actor paths."""

import unittest

import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.geometry import NoFlyZone, Sphere
from brain_uav.models import V2ANNPolicyActor, V2SNNPolicyActor, ZoneSetEncoderConfig
from brain_uav.observations import V2ObservationScales, collate_v2_observations
from test_v2_static_no_fly_env import _scenario


class TestV2OperationCounts(unittest.TestCase):
    def inputs(self, zone_count):
        scenario = ScenarioConfig()
        scales = V2ObservationScales(scenario.world_xy, scenario.world_z_min,
                                     scenario.world_z_max, scenario.gamma_max)
        env = V2StaticNoFlyTrajectoryEnv(scenario, RewardConfig())
        zones = [NoFlyZone(f'zone_{i}', Sphere([40 + 10 * i, 100, 100], 2))
                 for i in range(zone_count)]
        obs, _ = env.reset(options={'scenario': _scenario(
            [0, 0, 100, 0, 0], [100, 0, 100], zones)})
        env.close()
        return scales, collate_v2_observations([obs], device='cpu')

    def actor(self, model, scales):
        config = ZoneSetEncoderConfig(hidden_dim=4, num_heads=2, num_layers=1,
                                      ffn_dim=8)
        cls = V2ANNPolicyActor if model == 'ann' else V2SNNPolicyActor
        actor = cls(scales, 2, 3, torch.ones(2), encoder_config=config)
        actor.eval().requires_grad_(False)
        return actor

    def test_fused_linears_attention_einsum_pooling_and_masked_empty_token(self):
        from brain_uav.utils.v2_operation_counts import count_v2_actor_operations
        for zones, encoder_macs in ((0, 324), (2, 1516)):
            with self.subTest(zones=zones):
                scales, batch = self.inputs(zones)
                actor = self.actor('ann', scales)
                with torch.inference_mode():
                    expected = actor(batch)
                    action, counts = count_v2_actor_operations(actor, batch)
                    after = actor(batch)
                # Encoder: 92*N + 72 + 168*(N+1) + 84*(N+1)**2.
                # Head: 8*3 + 3*3 + 3*2 = 39. Masking does not skip work.
                self.assertTrue(torch.equal(action, expected))
                self.assertTrue(torch.equal(after, expected))
                self.assertEqual(counts['encoder_macs'], encoder_macs)
                self.assertEqual(counts['head_dense_macs'], 39)
                self.assertEqual(counts['macs'], encoder_macs + 39)
                self.assertEqual(counts['dense_macs'], counts['macs'])
                self.assertEqual(counts['acs'], 0)
                self.assertEqual(counts['decision_count'], 1)
                self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks
                                     for m in actor.modules()))

    def test_only_binary_fc2_becomes_acs_across_all_four_time_steps(self):
        from brain_uav.utils.v2_operation_counts import count_v2_actor_operations
        scales, batch = self.inputs(2)
        actor = self.actor('snn', scales)
        actor.snn_head.fc1.weight.data.zero_()
        for bias, spike_count in ((0.0, 0), (10.0, 12)):
            with self.subTest(bias=bias):
                actor.snn_head.fc1.bias.data.fill_(bias)
                with torch.inference_mode():
                    expected = actor(batch)
                    action, counts = count_v2_actor_operations(actor, batch)
                    repeated, repeated_counts = count_v2_actor_operations(actor, batch)
                self.assertTrue(torch.equal(action, expected))
                self.assertTrue(torch.equal(repeated, expected))
                self.assertEqual(counts, repeated_counts)
                self.assertEqual(counts['encoder_macs'], 1516)
                self.assertEqual(counts['head_dense_macs'], 24 + 4 * 9 + 6)
                self.assertEqual(counts['head_macs'], 24 + 6)
                self.assertEqual(counts['spike_count_l1'], spike_count)
                self.assertEqual(counts['spike_slots_l1'], 4 * 3)
                self.assertEqual(counts['acs'], spike_count * 3)
                self.assertEqual(counts['macs'], 1516 + 30)
                self.assertEqual(counts['dense_macs'], 1516 + 66)
                self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks
                                     for m in actor.modules()))

    def test_summaries_weight_decisions_and_spike_slots_correctly(self):
        from brain_uav.utils.v2_operation_counts import summarize_operation_counts
        first = dict(decision_count=1, dense_macs=100, macs=80, acs=10,
                     encoder_macs=60, head_dense_macs=40, head_macs=20,
                     spike_count_l1=2, spike_slots_l1=4)
        long_episode = {key: value * 3 for key, value in first.items()}
        long_episode['acs'], long_episode['spike_count_l1'] = 60, 12
        summary = summarize_operation_counts([first, long_episode])
        self.assertEqual(summary['decision_count'], 4)
        self.assertEqual(summary['totals']['acs'], 70)
        self.assertEqual(summary['mean_per_decision']['acs'], 17.5)
        self.assertEqual(summary['mean_per_decision']['macs'], 80)
        self.assertEqual(summary['spike_rate_l1'], 14 / 16)


if __name__ == '__main__':
    unittest.main()
