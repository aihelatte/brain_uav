"""Projected policy and differentiable online spike activity contracts."""
import unittest
from unittest import mock
import torch
from brain_uav.models import V2ANNPolicyActor, V2SNNPolicyActor
from test_v2_ann import _observation
from brain_uav.observations import V2ObservationScales, collate_v2_observations
from brain_uav.models.zone_set_encoder import ZoneSetEncoderConfig


class TestProjectedPolicy(unittest.TestCase):
    def actor(self, cls):
        return cls(V2ObservationScales(100., 0., 50., .6), 2, 192,
                   torch.tensor([.2, .3]), projection_dim=32)

    def test_projection_dimensions_actions_and_single_forward(self):
        for cls in (V2ANNPolicyActor, V2SNNPolicyActor):
            actor = self.actor(cls)
            self.assertEqual(tuple(actor.projection.weight.shape), (32, 128))
            for count in (0, 1, 6):
                batch = collate_v2_observations([_observation(count)])
                action = actor(batch)
                self.assertTrue(torch.all(action.abs() <= actor.action_limit))
                if cls is V2SNNPolicyActor:
                    with mock.patch.object(actor.snn_head, '_forward_impl', wraps=actor.snn_head._forward_impl) as forward:
                        other, rate = actor.forward_with_activity(batch)
                    self.assertEqual(forward.call_count, 1)
                    torch.testing.assert_close(other, action)
                    self.assertEqual(rate.shape, ())
                    self.assertTrue(rate.requires_grad)
                    torch.testing.assert_close(actor(batch), action)

    def test_encoder_width_is_read_and_masked_padding_cannot_change_action(self):
        scales = V2ObservationScales(100., 0., 50., .6)
        for cls in (V2ANNPolicyActor, V2SNNPolicyActor):
            actor = cls(scales, 2, 192, torch.tensor([.2, .3]), projection_dim=32,
                        encoder_config=ZoneSetEncoderConfig(hidden_dim=16))
            self.assertEqual(actor.projection.in_features, 32)
            batch = collate_v2_observations([_observation(0), _observation(6)])
            expected = actor(batch)
            batch.zone_features[0].fill_(float('nan'))
            torch.testing.assert_close(actor(batch), expected)
            for bad in (True, 0, -1, 32., float('inf')):
                with self.assertRaises(ValueError):
                    cls(scales, 2, 192, torch.ones(2), projection_dim=bad)

    def test_first_layer_regularization_gradient_scope(self):
        actor = self.actor(V2SNNPolicyActor)
        with torch.no_grad():
            actor.snn_head.fc1.bias.fill_(3.)
        _, rate = actor.forward_with_activity(collate_v2_observations([_observation(1)]))
        loss = torch.relu(rate - .17).square()
        self.assertGreater(float(loss.detach()), 0.)
        loss.backward()
        self.assertIsNotNone(actor.snn_head.fc1.weight.grad)
        self.assertIsNotNone(actor.projection.weight.grad)
        self.assertGreater(actor.snn_head.fc1.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(actor.projection.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0
                            for p in actor.zone_set_encoder.parameters()))
        self.assertIsNone(actor.snn_head.fc2.weight.grad)
        self.assertIsNone(actor.snn_head.action_layer.weight.grad)
